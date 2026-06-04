#!/usr/bin/env python
"""Redraw the selected Fig. 7 failure case with zoomed ROI and EPE comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from dav2_wrapper import DAv2FeatureExtractor
from eval_depth_raft_region_decomp import _build_model as _build_g_z_model
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_CANDIDATES = PACKAGE_ROOT / "figures" / "sintel_fig6_dab_strict_w80" / "candidates.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "figures" / "revision_v2" / "paper_fig7_failure_case_zoom_v2"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_DAV2 = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
MODEL_ORDER = ("baseline", "g_only", "z_only", "dge", "full")
MODEL_LABELS = {
    "baseline": "B",
    "g_only": "+G",
    "z_only": "+Z",
    "dge": "+DGE",
    "full": "+DGE+DAB",
}
RED = "#ff3b30"
PANEL_ASPECT = 1024 / 436


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", type=str, default="cave_4")
    parser.add_argument("--frame-idx", type=int, default=4)
    parser.add_argument("--crop", type=str, default="288,160,368,240")
    parser.add_argument("--zoom-scale", type=float, default=1.25)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _scene_paths(sintel_root: Path, scene: str, frame_idx: int) -> dict[str, Path]:
    return {
        "image1": sintel_root / "training" / "clean" / scene / f"frame_{frame_idx:04d}.png",
        "image2": sintel_root / "training" / "clean" / scene / f"frame_{frame_idx + 1:04d}.png",
        "flow": sintel_root / "training" / "flow" / scene / f"frame_{frame_idx:04d}.flo",
    }


def _parse_box(text: str) -> tuple[int, int, int, int]:
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError(f"Expected crop as x0,y0,x1,y1, got {text!r}")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box: {values}")
    return x0, y0, x1, y1


def _robust_norm(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(x[valid], [lo, hi])
    if p_hi <= p_lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def _gradient_mag(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _flow_gradient_mag(flow: np.ndarray) -> np.ndarray:
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(ux * ux + uy * uy + vx * vx + vy * vy)


@torch.no_grad()
def _extract_depth(extractor: DAv2FeatureExtractor, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_t = torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device) / 255.0
    depth = extractor.extract_depth(image_t)[0, 0].detach().cpu().numpy()
    return depth.astype(np.float32)


def _load_candidate(payload: dict[str, Any], scene: str, frame_idx: int, box: tuple[int, int, int, int]) -> dict[str, Any]:
    for item in payload.get("top_candidates", []):
        item_box = tuple(int(v) for v in item["crop_xyxy"])
        if str(item["scene"]) == scene and int(item["frame_idx"]) == frame_idx and item_box == box:
            return item
    raise ValueError(f"No candidate found for scene={scene}, frame={frame_idx}, crop={box}")


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    aspect: float,
    scale: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    roi_w = x1 - x0
    roi_h = y1 - y0
    zoom_h = max(int(round(roi_h * scale)), roi_h)
    zoom_w = max(int(round(zoom_h * aspect)), roi_w)
    if zoom_w > width:
        zoom_w = width
        zoom_h = max(roi_h, min(height, int(round(zoom_w / aspect))))
    if zoom_h > height:
        zoom_h = height
        zoom_w = max(roi_w, min(width, int(round(zoom_h * aspect))))

    zx0 = int(round(cx - zoom_w / 2))
    zy0 = int(round(cy - zoom_h / 2))
    zx0 = max(0, min(width - zoom_w, zx0))
    zy0 = max(0, min(height - zoom_h, zy0))
    return zx0, zy0, zx0 + zoom_w, zy0 + zoom_h


def _crop_panel(panel: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return panel[y0:y1, x0:x1]


def _imshow_panel(
    ax: plt.Axes,
    panel: np.ndarray,
    *,
    cmap: str | None,
    extent: tuple[int, int, int, int],
) -> None:
    if panel.ndim == 2:
        ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper", extent=extent)
    else:
        ax.imshow(panel, origin="upper", extent=extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1 / PANEL_ASPECT)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_roi(ax: plt.Axes, box: tuple[int, int, int, int], *, lw: float) -> None:
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RED, lw=lw))


def _draw_badge(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.92,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
    )


@torch.no_grad()
def _predict_scene_flow(
    *,
    model,
    sintel_root: Path,
    scene: str,
    frame_idx: int,
    device: torch.device,
) -> np.ndarray:
    flow_prev = None
    model.eval()
    for current_idx in range(1, frame_idx + 1):
        paths = _scene_paths(sintel_root, scene, current_idx)
        image1 = torch.from_numpy(_read_rgb(paths["image1"])).permute(2, 0, 1).float()[None].to(device)
        image2 = torch.from_numpy(_read_rgb(paths["image2"])).permute(2, 0, 1).float()[None].to(device)
        padder = InputPadder(image1.shape, coarsest_scale=8)
        image1_pad, image2_pad = padder.pad(image1, image2)
        flow_low, flow_pr = model(
            image1_pad,
            image2_pad,
            iters=12,
            flow_init=flow_prev,
            test_mode=True,
            bw=False,
        )
        flow_up = padder.unpad(flow_pr[0]).detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        flow_prev = forward_interpolate(flow_low[0])[None].to(device)
    return flow_up


def _predict_error_maps(
    *,
    payload: dict[str, Any],
    sintel_root: Path,
    scene: str,
    frame_idx: int,
    gt_flow: np.ndarray,
    dav2_weights: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    checkpoints = payload["checkpoints"]
    baseline_checkpoint = Path(checkpoints["baseline"]).expanduser().resolve()
    error_maps: dict[str, np.ndarray] = {}
    for model_name in ("dge", "full"):
        model = _build_g_z_model(
            device=device,
            baseline_checkpoint=baseline_checkpoint,
            fusion_checkpoint=Path(checkpoints[model_name]).expanduser().resolve(),
            dav2_weights=dav2_weights,
        )
        pred = _predict_scene_flow(
            model=model,
            sintel_root=sintel_root,
            scene=scene,
            frame_idx=frame_idx,
            device=device,
        )
        error_maps[model_name] = np.sqrt(np.sum((pred - gt_flow) ** 2, axis=-1)).astype(np.float32)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return error_maps


def _normalise_error_maps(error_maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    merged = np.concatenate([value[np.isfinite(value)].ravel() for value in error_maps.values()])
    if merged.size == 0:
        return {name: np.zeros_like(value, dtype=np.float32) for name, value in error_maps.items()}
    vmax = float(np.percentile(merged, 98.0))
    if vmax <= 1e-6:
        return {name: np.zeros_like(value, dtype=np.float32) for name, value in error_maps.items()}
    return {name: np.clip(value / vmax, 0.0, 1.0).astype(np.float32) for name, value in error_maps.items()}


def _draw_epe_panel(ax: plt.Axes, crop_epe: dict[str, float]) -> None:
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    values = [float(crop_epe[name]) for name in MODEL_ORDER]
    colors = ["#b8b8b8", "#8dbad7", "#9ccf8a", "#4c78a8", "#e45756"]
    xpos = np.arange(len(values))
    ax.bar(xpos, values, color=colors, width=0.72)
    ax.set_title("EPE in red ROI", fontsize=8.6, fontweight="bold", pad=4)
    low = min(values) - 0.08
    high = max(values) + 0.12
    ax.set_ylim(low, high)
    ax.set_ylabel("EPE", fontsize=7.2, labelpad=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.7)
    ax.tick_params(axis="y", labelsize=6.7, width=0.5, length=2)
    ax.grid(axis="y", color="#d9d9d9", lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    for x, y in zip(xpos, values):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")

    dge_idx = MODEL_ORDER.index("dge")
    full_idx = MODEL_ORDER.index("full")
    y = max(values[dge_idx], values[full_idx]) + 0.065
    ax.plot([dge_idx, dge_idx, full_idx, full_idx], [y - 0.015, y, y, y - 0.015], color="#333333", lw=0.65)
    ax.text(
        0.5 * (dge_idx + full_idx),
        y + 0.012,
        f"Delta={values[full_idx] - values[dge_idx]:+.2f}",
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
    )


def _save_all(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates_json.expanduser().resolve()
    sintel_root = args.sintel_root.expanduser().resolve()
    dav2_weights = args.dav2_weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    box = _parse_box(args.crop)

    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidate = _load_candidate(payload, args.scene, args.frame_idx, box)
    crop_epe = {name: float(candidate["crop_epe"][name]) for name in MODEL_ORDER}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    paths = _scene_paths(sintel_root, args.scene, args.frame_idx)
    image = _read_rgb(paths["image1"])
    gt_flow = frame_utils.readFlow(str(paths["flow"])).astype(np.float32)
    height, width = image.shape[:2]

    extractor = DAv2FeatureExtractor(str(dav2_weights), device=device, precision="auto")
    extractor.eval()
    depth = _extract_depth(extractor, image, device)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_error_maps = _predict_error_maps(
        payload=payload,
        sintel_root=sintel_root,
        scene=args.scene,
        frame_idx=args.frame_idx,
        gt_flow=gt_flow,
        dav2_weights=dav2_weights,
        device=device,
    )
    error_maps = _normalise_error_maps(raw_error_maps)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    depth_norm = _robust_norm(depth)
    panels = [
        ("Input", image, None, ""),
        ("RGB Gradient", _robust_norm(_gradient_mag(gray)), "magma", ""),
        ("DAv2 Depth", depth_norm, "viridis", ""),
        ("Norm. Depth Gradient", _robust_norm(_gradient_mag(depth_norm)), "magma", ""),
        ("GT Flow Gradient", _robust_norm(_flow_gradient_mag(gt_flow)), "magma", ""),
        ("+DGE Error", error_maps["dge"], "viridis", f"EPE {crop_epe['dge']:.2f}"),
        ("+DGE+DAB Error", error_maps["full"], "viridis", f"EPE {crop_epe['full']:.2f}"),
    ]

    zoom_box = _expand_box(box, width=width, height=height, aspect=PANEL_ASPECT, scale=args.zoom_scale)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.6,
        }
    )
    fig = plt.figure(figsize=(17.4, 4.28), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        len(panels),
        width_ratios=[1] * len(panels),
        height_ratios=[1, 1],
        left=0.025,
        right=0.995,
        bottom=0.075,
        top=0.865,
        wspace=0.035,
        hspace=0.145,
    )
    axes = np.empty((2, len(panels)), dtype=object)
    full_extent = (0, width, height, 0)
    zoom_extent = (zoom_box[0], zoom_box[2], zoom_box[3], zoom_box[1])

    for col, (title, panel, cmap, badge) in enumerate(panels):
        ax_full = fig.add_subplot(grid[0, col])
        ax_zoom = fig.add_subplot(grid[1, col])
        axes[0, col] = ax_full
        axes[1, col] = ax_zoom
        _imshow_panel(ax_full, panel, cmap=cmap, extent=full_extent)
        _draw_roi(ax_full, box, lw=1.35)
        zoom_panel = _crop_panel(panel, zoom_box)
        _imshow_panel(ax_zoom, zoom_panel, cmap=cmap, extent=zoom_extent)
        _draw_roi(ax_zoom, box, lw=1.05)
        if badge:
            _draw_badge(ax_full, badge)
            _draw_badge(ax_zoom, badge)
        ax_full.set_title(title, fontsize=8.9, fontweight="bold", pad=3)

    axes[0, 0].text(-0.055, 0.5, "Full", transform=axes[0, 0].transAxes, ha="right", va="center", fontsize=7.8, fontweight="bold")
    axes[1, 0].text(-0.055, 0.5, "Zoom", transform=axes[1, 0].transAxes, ha="right", va="center", fontsize=7.8, fontweight="bold")

    x0, y0, x1, y1 = box
    zx0, zy0, zx1, zy1 = zoom_box
    for xy_a, xy_b in (((x0, y1), (zx0, zy0)), ((x1, y1), (zx1, zy0))):
        fig.add_artist(
            ConnectionPatch(
                xyA=xy_a,
                coordsA=axes[0, 0].transData,
                xyB=xy_b,
                coordsB=axes[1, 0].transData,
                color=RED,
                lw=0.7,
                clip_on=False,
            )
        )

    fig.suptitle(
        f"Failure case: texture-rich but depth-continuous region | {args.scene} frame_{args.frame_idx:04d} | "
        f"ROI EPE: +DGE={crop_epe['dge']:.2f}, +DGE+DAB={crop_epe['full']:.2f} "
        f"(Delta={crop_epe['full'] - crop_epe['dge']:+.2f})",
        fontsize=10.2,
        fontweight="bold",
        y=0.955,
    )
    _save_all(fig, output)
    plt.close(fig)

    metadata_path = output.with_suffix(".json")
    metadata = {
        "source_candidates": str(candidates_path),
        "scene": args.scene,
        "frame_idx": args.frame_idx,
        "crop_xyxy": list(box),
        "zoom_xyxy": list(zoom_box),
        "crop_epe": crop_epe,
        "dab_delta": crop_epe["full"] - crop_epe["dge"],
        "error_map_percentile_scale": "98th percentile over +DGE and +DGE+DAB error maps",
        "outputs": {
            "png": str(output.with_suffix(".png")),
            "pdf": str(output.with_suffix(".pdf")),
            "tiff": str(output.with_suffix(".tiff")),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved_png={output.with_suffix('.png')}")
    print(f"saved_pdf={output.with_suffix('.pdf')}")
    print(f"saved_tiff={output.with_suffix('.tiff')}")
    print(f"saved_metadata={metadata_path}")
    print(f"epe_dge={crop_epe['dge']:.4f} epe_dge_dab={crop_epe['full']:.4f} delta={crop_epe['full'] - crop_epe['dge']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
