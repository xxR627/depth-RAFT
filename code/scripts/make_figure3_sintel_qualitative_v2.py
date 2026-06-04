#!/usr/bin/env python
"""Data-driven Figure 3 qualitative selection for Sintel clean train."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
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

import datasets_un
from eval_depth_raft_region_decomp import _build_model as _build_g_z_model
from eval_sun_raft_region_decomp import (
    COARSEST_SCALE,
    ITERS,
    _load_model as _load_baseline_model,
    _patched_get_extention_factory,
)
from sintel_occ_utils import _official_region_masks
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures"
CSV_FIELDS = [
    "scene_name",
    "scene_category",
    "frame_idx",
    "pair",
    "baseline_epe",
    "ours_epe",
    "baseline_unmatched_epe",
    "ours_unmatched_epe",
    "delta_unmatched",
    "relative_unmatched_improvement",
    "unmatched_pixels",
]
COLUMNS = ["Image", "GT Flow", "Sun-RAFT", "Ours", "Sun-RAFT Error", "Ours Error", "Unmatched"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_OUTPUT_DIR / "sintel_pair_deltas.csv")
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT_DIR / "figure3_selection_metadata.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _scene_category(scene_name: str) -> str:
    return re.sub(r"_\d+$", "", scene_name)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask > 0


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)


def _make_colorwheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    colorwheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    colorwheel[col:col + yg, 1] = 255
    col += yg
    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    colorwheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col:col + cb, 2] = 255
    col += cb
    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    colorwheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col:col + mr, 0] = 255
    return colorwheel


def _flow_to_color_fallback(flow_uv: np.ndarray) -> np.ndarray:
    u = flow_uv[:, :, 0].copy()
    v = flow_uv[:, :, 1].copy()
    unknown = np.isnan(u) | np.isnan(v) | (np.abs(u) > 1e7) | (np.abs(v) > 1e7)
    u[unknown] = 0
    v[unknown] = 0
    rad = np.sqrt(u ** 2 + v ** 2)
    rad_max = max(float(np.max(rad)), 1e-5)
    u /= rad_max
    v /= rad_max
    colorwheel = _make_colorwheel()
    ncols = colorwheel.shape[0]
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    image = np.zeros((*u.shape, 3), dtype=np.uint8)
    for channel in range(3):
        col0 = colorwheel[k0, channel] / 255.0
        col1 = colorwheel[k1, channel] / 255.0
        col = (1 - f) * col0 + f * col1
        col[rad <= 1] = 1 - rad[rad <= 1] * (1 - col[rad <= 1])
        col[rad > 1] *= 0.75
        image[:, :, channel] = np.floor(255 * col)
    image[unknown] = 0
    return image


def _flow_to_color(flow: np.ndarray) -> np.ndarray:
    try:
        import flow_vis  # type: ignore

        return flow_vis.flow_to_color(flow, convert_to_bgr=False)
    except Exception:
        return _flow_to_color_fallback(flow)


def _overlay_unmatched(image: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    output = image.astype(np.float32).copy()
    red = np.zeros_like(output)
    red[:, :, 0] = 255.0
    output[mask] = (1 - alpha) * output[mask] + alpha * red[mask]
    return np.clip(output, 0, 255).astype(np.uint8)


def _scene_paths(sintel_root: Path, scene: str, frame_idx: int) -> dict[str, Path]:
    return {
        "image1": sintel_root / "training" / "clean" / scene / f"frame_{frame_idx:04d}.png",
        "image2": sintel_root / "training" / "clean" / scene / f"frame_{frame_idx + 1:04d}.png",
        "flow": sintel_root / "training" / "flow" / scene / f"frame_{frame_idx:04d}.flo",
        "occ": sintel_root / "training" / "occlusions" / scene / f"frame_{frame_idx:04d}.png",
    }


def _run_model_pair_metrics(
    *,
    model,
    sintel_root: Path,
    device: torch.device,
    log_every: int,
    label: str,
) -> dict[tuple[str, int], dict[str, float | int | str]]:
    val_dataset = datasets_un.MpiSintel(
        split="training",
        dstype="clean",
        show_extra_info=True,
        read_flow_gt=True,
    )
    output: dict[tuple[str, int], dict[str, float | int | str]] = {}
    flow_prev = None
    sequence_prev = None
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt_cpu, _, (sequence, frame_zero_based) = val_dataset[val_id]
            if sequence != sequence_prev:
                flow_prev = None
            image1 = image1[None].to(device)
            image2 = image2[None].to(device)
            height, width = int(flow_gt_cpu.shape[-2]), int(flow_gt_cpu.shape[-1])
            masks = _official_region_masks(
                sintel_root=sintel_root,
                sequence=sequence,
                frame_zero_based=frame_zero_based,
                height=height,
                width=width,
            )
            unmatched = masks["unmatched_occ"].squeeze(0).squeeze(0)
            padder = InputPadder(image1.shape, coarsest_scale=COARSEST_SCALE)
            image1_pad, image2_pad = padder.pad(image1, image2)
            flow_low, flow_pr = model(
                image1_pad,
                image2_pad,
                iters=ITERS,
                flow_init=flow_prev,
                test_mode=True,
                bw=False,
            )
            flow_pred = padder.unpad(flow_pr[0]).detach().cpu()
            epe_map = torch.sum((flow_pred - flow_gt_cpu) ** 2, dim=0).sqrt()
            unmatched_pixels = int(unmatched.sum().item())
            unmatched_epe = float(epe_map[unmatched].mean().item()) if unmatched_pixels else float("nan")
            frame_idx = int(frame_zero_based + 1)
            output[(sequence, frame_idx)] = {
                "scene_name": sequence,
                "scene_category": _scene_category(sequence),
                "frame_idx": frame_idx,
                "pair": f"frame_{frame_idx:04d}-frame_{frame_idx + 1:04d}",
                f"{label}_epe": float(epe_map.mean().item()),
                f"{label}_unmatched_epe": unmatched_epe,
                "unmatched_pixels": unmatched_pixels,
            }
            flow_prev = forward_interpolate(flow_low[0])[None].to(device)
            sequence_prev = sequence
            if ((val_id + 1) % log_every == 0) or (val_id + 1 == len(val_dataset)):
                print(
                    f"[{label}] pair={val_id + 1}/{len(val_dataset)} scene={sequence} "
                    f"frame={frame_idx:04d} elapsed_sec={time.time() - start_time:.1f}",
                    flush=True,
                )
    return output


def _write_delta_csv(
    csv_path: Path,
    baseline_metrics: dict[tuple[str, int], dict[str, float | int | str]],
    ours_metrics: dict[tuple[str, int], dict[str, float | int | str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(baseline_metrics.keys()):
        base = baseline_metrics[key]
        ours = ours_metrics[key]
        baseline_unmatched = float(base["baseline_unmatched_epe"])
        ours_unmatched = float(ours["ours_unmatched_epe"])
        delta = baseline_unmatched - ours_unmatched
        relative = delta / baseline_unmatched if math.isfinite(delta) and baseline_unmatched > 0 else float("nan")
        row = {
            "scene_name": base["scene_name"],
            "scene_category": base["scene_category"],
            "frame_idx": int(base["frame_idx"]),
            "pair": base["pair"],
            "baseline_epe": float(base["baseline_epe"]),
            "ours_epe": float(ours["ours_epe"]),
            "baseline_unmatched_epe": baseline_unmatched,
            "ours_unmatched_epe": ours_unmatched,
            "delta_unmatched": delta,
            "relative_unmatched_improvement": relative,
            "unmatched_pixels": int(base["unmatched_pixels"]),
        }
        rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _load_delta_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in (
                "frame_idx",
                "baseline_epe",
                "ours_epe",
                "baseline_unmatched_epe",
                "ours_unmatched_epe",
                "delta_unmatched",
                "relative_unmatched_improvement",
                "unmatched_pixels",
            ):
                if key in ("frame_idx", "unmatched_pixels"):
                    parsed[key] = int(float(parsed[key]))
                else:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _select_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [
        row for row in rows
        if math.isfinite(float(row["delta_unmatched"]))
        and int(row["unmatched_pixels"]) > 0
        and row["scene_name"] != "ambush_5"
    ]
    sorted_rows = sorted(valid, key=lambda row: float(row["delta_unmatched"]), reverse=True)
    top20 = sorted_rows[:20]
    selected: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    rejected: list[dict[str, Any]] = []
    for row in top20:
        category = str(row["scene_category"])
        reasons = []
        if category in seen_categories:
            reasons.append("duplicate_scene_category")
        if float(row["baseline_unmatched_epe"]) <= 5.0:
            reasons.append("baseline_unmatched_epe<=5")
        if float(row["ours_unmatched_epe"]) >= 0.7 * float(row["baseline_unmatched_epe"]):
            reasons.append("relative_improvement<30pct")
        if reasons:
            rejected.append({"scene_name": row["scene_name"], "frame_idx": row["frame_idx"], "reasons": reasons})
            continue
        selected.append(row)
        seen_categories.add(category)
        if len(selected) == 4:
            break
    if len(selected) < 4:
        raise RuntimeError(
            f"Only selected {len(selected)} rows from top20 under strict criteria; "
            "inspect top20 in metadata or relax criteria."
        )
    metadata = {
        "selection_rule": (
            "Sort by delta_unmatched descending; from top20 choose 4 different scene categories, "
            "baseline_unmatched_epe>5, ours_unmatched_epe < 0.7*baseline, excluding ambush_5."
        ),
        "top20": top20,
        "selected": selected,
        "rejected_from_top20_before_selection_completed": rejected,
    }
    return selected, metadata


def _predict_scene_target(model, sintel_root: Path, scene: str, target_frame: int, device: torch.device) -> np.ndarray:
    flow_prev = None
    target_flow: np.ndarray | None = None
    model.eval()
    with torch.no_grad():
        for frame_idx in range(1, target_frame + 1):
            paths = _scene_paths(sintel_root, scene, frame_idx)
            image1 = _image_to_tensor(_read_rgb(paths["image1"]), device)
            image2 = _image_to_tensor(_read_rgb(paths["image2"]), device)
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
            flow_up = padder.unpad(flow_pr[0]).detach().cpu().permute(1, 2, 0).numpy()
            flow_prev = forward_interpolate(flow_low[0])[None].to(device)
            if frame_idx == target_frame:
                target_flow = flow_up
    if target_flow is None:
        raise RuntimeError(f"Failed to predict {scene} frame {target_frame}")
    return target_flow


def _collect_selected_predictions(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
    baseline: dict[str, np.ndarray] = {}
    for row in selected:
        key = f"{row['scene_name']}/frame_{int(row['frame_idx']):04d}"
        print(f"[figure-baseline] {key}", flush=True)
        baseline[key] = _predict_scene_target(
            baseline_model, args.sintel_root, str(row["scene_name"]), int(row["frame_idx"]), device
        )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ours_model = _build_g_z_model(
        device=device,
        baseline_checkpoint=args.baseline_checkpoint,
        fusion_checkpoint=args.ours_checkpoint,
        dav2_weights=args.dav2_weights,
    )
    ours: dict[str, np.ndarray] = {}
    for row in selected:
        key = f"{row['scene_name']}/frame_{int(row['frame_idx']):04d}"
        print(f"[figure-ours] {key}", flush=True)
        ours[key] = _predict_scene_target(
            ours_model, args.sintel_root, str(row["scene_name"]), int(row["frame_idx"]), device
        )
    del ours_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return baseline, ours


def _build_figure(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    baseline: dict[str, np.ndarray],
    ours: dict[str, np.ndarray],
) -> dict[str, Any]:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(4, 7, figsize=(16, 8))
    metrics: dict[str, Any] = {}
    for row_idx, row in enumerate(selected):
        scene = str(row["scene_name"])
        frame_idx = int(row["frame_idx"])
        key = f"{scene}/frame_{frame_idx:04d}"
        paths = _scene_paths(args.sintel_root, scene, frame_idx)
        image = _read_rgb(paths["image1"])
        gt_flow = frame_utils.readFlow(str(paths["flow"])).astype(np.float32)
        occ = _read_mask(paths["occ"])
        base_flow = baseline[key]
        ours_flow = ours[key]
        base_epe = np.sqrt(np.sum((base_flow - gt_flow) ** 2, axis=-1))
        ours_epe = np.sqrt(np.sum((ours_flow - gt_flow) ** 2, axis=-1))
        metrics[key] = {
            "frame_pair": f"frame_{frame_idx:04d}-frame_{frame_idx + 1:04d}",
            "baseline_epe": float(base_epe.mean()),
            "ours_epe": float(ours_epe.mean()),
            "baseline_unmatched_epe": float(base_epe[occ].mean()) if occ.any() else None,
            "ours_unmatched_epe": float(ours_epe[occ].mean()) if occ.any() else None,
            "delta_unmatched": row["delta_unmatched"],
            "relative_unmatched_improvement": row["relative_unmatched_improvement"],
        }
        panels = [
            image,
            _flow_to_color(gt_flow),
            _flow_to_color(base_flow),
            _flow_to_color(ours_flow),
            base_epe,
            ours_epe,
            _overlay_unmatched(image, occ, alpha=0.25),
        ]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx in (4, 5):
                ax.imshow(panel, cmap="viridis", vmin=0, vmax=3, aspect="auto")
                epe_value = metrics[key]["baseline_epe"] if col_idx == 4 else metrics[key]["ours_epe"]
                ax.text(
                    0.03,
                    0.94,
                    f"{epe_value:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                    bbox={"facecolor": "white", "alpha": 0.88, "pad": 1.5, "edgecolor": "none"},
                )
            else:
                ax.imshow(panel, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(COLUMNS[col_idx], fontweight="bold", fontsize=11, pad=5)
            if col_idx == 0:
                ax.set_ylabel(scene, rotation=0, labelpad=34, va="center", ha="right", fontsize=11)

    caption = (
        "Figure 3: Qualitative comparison on Sintel clean train.\n"
        "Our method achieves clear improvements in unmatched regions\n"
        "(highlighted in red overlay), particularly at occlusion boundaries."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.12, wspace=0.02, hspace=0.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "figure3_sintel_qualitative_v2.pdf"
    png_path = args.output_dir / "figure3_sintel_qualitative_v2.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"saved_pdf={pdf_path}", flush=True)
    print(f"saved_png={png_path}", flush=True)
    return metrics


def _compute_or_load_rows(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    if args.csv_path.is_file() and not args.force_recompute:
        print(f"[csv] using existing {args.csv_path}", flush=True)
        return _load_delta_csv(args.csv_path)

    baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
    baseline_metrics = _run_model_pair_metrics(
        model=baseline_model,
        sintel_root=args.sintel_root,
        device=device,
        log_every=args.log_every,
        label="baseline",
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ours_model = _build_g_z_model(
        device=device,
        baseline_checkpoint=args.baseline_checkpoint,
        fusion_checkpoint=args.ours_checkpoint,
        dav2_weights=args.dav2_weights,
    )
    ours_metrics = _run_model_pair_metrics(
        model=ours_model,
        sintel_root=args.sintel_root,
        device=device,
        log_every=args.log_every,
        label="ours",
    )
    del ours_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = _write_delta_csv(args.csv_path, baseline_metrics, ours_metrics)
    print(f"saved_csv={args.csv_path}", flush=True)
    return rows


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.ours_checkpoint = args.ours_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.csv_path = args.csv_path.expanduser().resolve()
    args.metadata_json = args.metadata_json.expanduser().resolve()
    for path in (args.baseline_checkpoint, args.ours_checkpoint, args.dav2_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    patched_get_extention = _patched_get_extention_factory(args.sintel_root)
    original_get_extention = datasets_un.get_extention
    datasets_un.get_extention = patched_get_extention
    try:
        rows = _compute_or_load_rows(args, device)
        selected, metadata = _select_rows(rows)
        baseline, ours = _collect_selected_predictions(args, selected, device)
        figure_metrics = _build_figure(args, selected, baseline, ours)
    finally:
        datasets_un.get_extention = original_get_extention

    metadata.update({
        "csv_path": args.csv_path,
        "baseline_checkpoint": args.baseline_checkpoint,
        "ours_checkpoint": args.ours_checkpoint,
        "figure_metrics": figure_metrics,
    })
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(_to_builtin(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"saved_metadata={args.metadata_json}", flush=True)
    print(json.dumps(_to_builtin({"selected": selected}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
