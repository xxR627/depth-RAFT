#!/usr/bin/env python
"""Data-driven Figure 4 qualitative selection for KITTI 2015 train."""

from __future__ import annotations

import argparse
import csv
import json
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

from eval_depth_raft_region_decomp import _build_model as _build_g_z_model
from eval_sun_raft_region_decomp import _load_model as _load_baseline_model
from utils import frame_utils
from utils.utils import InputPadder


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_KITTI_ROOT = Path(r"G:\flow_data\KITTI\data_scene_flow\training")
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures"
COLUMNS = ["Image", "GT Flow", "Sun-RAFT", "Ours", "Sun-RAFT Error", "Ours Error", "Valid GT"]
CSV_FIELDS = [
    "frame_id",
    "baseline_epe",
    "ours_epe",
    "delta_epe",
    "relative_improvement",
    "valid_pixels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_OUTPUT_DIR / "kitti_pair_deltas.csv")
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT_DIR / "figure4_selection_metadata.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


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


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)


def _read_kitti_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    flow, valid = frame_utils.readFlowKITTI(str(path))
    return flow.astype(np.float32), (valid > 0)


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


def _flow_to_color(flow: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    flow_for_color = flow.copy()
    if valid is not None:
        flow_for_color[~valid] = np.nan
    try:
        import flow_vis  # type: ignore

        image = flow_vis.flow_to_color(np.nan_to_num(flow_for_color, nan=0.0), convert_to_bgr=False)
    except Exception:
        image = _flow_to_color_fallback(flow_for_color)
    if valid is not None:
        image[~valid] = 0
    return image


def _overlay_valid_gt(image: np.ndarray, valid: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    output = image.astype(np.float32).copy()
    cyan = np.zeros_like(output)
    cyan[:, :, 1] = 220.0
    cyan[:, :, 2] = 255.0
    output[valid] = (1 - alpha) * output[valid] + alpha * cyan[valid]
    return np.clip(output, 0, 255).astype(np.uint8)


def _frame_paths(kitti_root: Path, frame_id: str) -> dict[str, Path]:
    return {
        "image1": kitti_root / "image_2" / f"{frame_id}_10.png",
        "image2": kitti_root / "image_2" / f"{frame_id}_11.png",
        "flow_occ": kitti_root / "flow_occ" / f"{frame_id}_10.png",
    }


def _predict_pair(model, kitti_root: Path, frame_id: str, device: torch.device) -> np.ndarray:
    paths = _frame_paths(kitti_root, frame_id)
    image1 = _image_to_tensor(_read_rgb(paths["image1"]), device)
    image2 = _image_to_tensor(_read_rgb(paths["image2"]), device)
    padder = InputPadder(image1.shape, mode="kitti", coarsest_scale=8)
    image1_pad, image2_pad = padder.pad(image1, image2)
    with torch.no_grad():
        _, flow_pr = model(image1_pad, image2_pad, iters=12, test_mode=True, bw=False)
        pred = padder.unpad(flow_pr[0]).detach().cpu().permute(1, 2, 0).numpy()
    return pred.astype(np.float32)


def _run_model_metrics(
    model,
    kitti_root: Path,
    frame_ids: list[str],
    device: torch.device,
    label: str,
    log_every: int,
) -> dict[str, dict[str, float | int | str]]:
    model.eval()
    output: dict[str, dict[str, float | int | str]] = {}
    start_time = time.time()
    for idx, frame_id in enumerate(frame_ids):
        paths = _frame_paths(kitti_root, frame_id)
        flow_gt, valid = _read_kitti_flow(paths["flow_occ"])
        pred = _predict_pair(model, kitti_root, frame_id, device)
        epe = np.sqrt(np.sum((pred - flow_gt) ** 2, axis=-1))
        valid_count = int(valid.sum())
        epe_mean = float(epe[valid].mean()) if valid_count else float("nan")
        output[frame_id] = {
            "frame_id": frame_id,
            f"{label}_epe": epe_mean,
            "valid_pixels": valid_count,
        }
        if ((idx + 1) % log_every == 0) or (idx + 1 == len(frame_ids)):
            print(
                f"[kitti-figure4:{label}] pair={idx + 1}/{len(frame_ids)} "
                f"frame={frame_id} elapsed_sec={time.time() - start_time:.1f}",
                flush=True,
            )
    return output


def _write_delta_csv(
    csv_path: Path,
    baseline: dict[str, dict[str, float | int | str]],
    ours: dict[str, dict[str, float | int | str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in sorted(baseline):
        baseline_epe = float(baseline[frame_id]["baseline_epe"])
        ours_epe = float(ours[frame_id]["ours_epe"])
        rows.append(
            {
                "frame_id": frame_id,
                "baseline_epe": baseline_epe,
                "ours_epe": ours_epe,
                "delta_epe": baseline_epe - ours_epe,
                "relative_improvement": (baseline_epe - ours_epe) / baseline_epe if baseline_epe > 0 else float("nan"),
                "valid_pixels": int(baseline[frame_id]["valid_pixels"]),
            }
        )
    rows.sort(key=lambda row: row["delta_epe"], reverse=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _load_delta_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "frame_id": row["frame_id"],
                    "baseline_epe": float(row["baseline_epe"]),
                    "ours_epe": float(row["ours_epe"]),
                    "delta_epe": float(row["delta_epe"]),
                    "relative_improvement": float(row["relative_improvement"]),
                    "valid_pixels": int(float(row["valid_pixels"])),
                }
            )
    rows.sort(key=lambda row: row["delta_epe"], reverse=True)
    return rows


def _select_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if 3.0 <= float(row["baseline_epe"]) <= 15.0
        and float(row["ours_epe"]) < 0.7 * float(row["baseline_epe"])
    ]
    eligible.sort(key=lambda row: row["delta_epe"], reverse=True)
    selected = eligible[:3]
    if len(selected) < 3:
        raise RuntimeError(
            f"Only selected {len(selected)} KITTI rows under criteria; "
            "inspect metadata top candidates before relaxing."
        )
    metadata = {
        "selection_rule": (
            "Sort by delta_epe descending after filtering baseline_epe in [3,15] "
            "and ours_epe < 0.7*baseline_epe."
        ),
        "top20_all": rows[:20],
        "top20_eligible": eligible[:20],
        "selected": selected,
    }
    return selected, metadata


def _compute_or_load_rows(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    if args.csv_path.is_file() and not args.force_recompute:
        print(f"[csv] using existing {args.csv_path}", flush=True)
        return _load_delta_csv(args.csv_path)

    frame_ids = [path.name.replace("_10.png", "") for path in sorted((args.kitti_root / "image_2").glob("*_10.png"))]
    if not frame_ids:
        raise RuntimeError(f"No KITTI image pairs under {args.kitti_root}")

    baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
    baseline_metrics = _run_model_metrics(
        baseline_model, args.kitti_root, frame_ids, device, "baseline", args.log_every
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
    ours_metrics = _run_model_metrics(ours_model, args.kitti_root, frame_ids, device, "ours", args.log_every)
    del ours_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = _write_delta_csv(args.csv_path, baseline_metrics, ours_metrics)
    print(f"saved_csv={args.csv_path}", flush=True)
    return rows


def _collect_selected_predictions(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
    baseline: dict[str, np.ndarray] = {}
    for idx, row in enumerate(selected):
        frame_id = str(row["frame_id"])
        print(f"[figure4-baseline] {frame_id}", flush=True)
        baseline[frame_id] = _predict_pair(baseline_model, args.kitti_root, frame_id, device)
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
    for idx, row in enumerate(selected):
        frame_id = str(row["frame_id"])
        print(f"[figure4-ours] {frame_id}", flush=True)
        ours[frame_id] = _predict_pair(ours_model, args.kitti_root, frame_id, device)
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
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(3, 7, figsize=(16, 6))
    metrics: dict[str, Any] = {}
    for row_idx, row in enumerate(selected):
        frame_id = str(row["frame_id"])
        paths = _frame_paths(args.kitti_root, frame_id)
        image = _read_rgb(paths["image1"])
        flow_gt, valid = _read_kitti_flow(paths["flow_occ"])
        base_flow = baseline[frame_id]
        ours_flow = ours[frame_id]
        base_epe = np.sqrt(np.sum((base_flow - flow_gt) ** 2, axis=-1))
        ours_epe = np.sqrt(np.sum((ours_flow - flow_gt) ** 2, axis=-1))
        base_sparse = np.where(valid, base_epe, np.nan)
        ours_sparse = np.where(valid, ours_epe, np.nan)
        metrics[frame_id] = {
            "baseline_epe": float(np.nanmean(base_sparse)),
            "ours_epe": float(np.nanmean(ours_sparse)),
            "delta_epe": float(np.nanmean(base_sparse) - np.nanmean(ours_sparse)),
            "relative_improvement": float((np.nanmean(base_sparse) - np.nanmean(ours_sparse)) / np.nanmean(base_sparse)),
            "valid_pixels": int(valid.sum()),
        }
        panels = [
            image,
            _flow_to_color(flow_gt, valid=valid),
            _flow_to_color(base_flow),
            _flow_to_color(ours_flow),
            base_sparse,
            ours_sparse,
            _overlay_valid_gt(image, valid, alpha=0.35),
        ]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx in (4, 5):
                ax.imshow(image, aspect="auto", alpha=0.35)
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
                epe_value = metrics[frame_id]["baseline_epe"] if col_idx == 4 else metrics[frame_id]["ours_epe"]
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
                ax.set_title(COLUMNS[col_idx], fontweight="bold", fontsize=10, pad=5)
            if col_idx == 0:
                ax.set_ylabel(frame_id, rotation=0, labelpad=30, va="center", ha="right", fontsize=10)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.04, wspace=0.02, hspace=0.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "figure4_kitti_qualitative.pdf"
    png_path = args.output_dir / "figure4_kitti_qualitative.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved_pdf={pdf_path}", flush=True)
    print(f"saved_png={png_path}", flush=True)
    return metrics


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.ours_checkpoint = args.ours_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.kitti_root = args.kitti_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.csv_path = args.csv_path.expanduser().resolve()
    args.metadata_json = args.metadata_json.expanduser().resolve()
    for path in (args.baseline_checkpoint, args.ours_checkpoint, args.dav2_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    for dirname in ("image_2", "flow_occ"):
        if not (args.kitti_root / dirname).is_dir():
            raise FileNotFoundError(args.kitti_root / dirname)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    rows = _compute_or_load_rows(args, device)
    selected, metadata = _select_rows(rows)
    baseline, ours = _collect_selected_predictions(args, selected, device)
    figure_metrics = _build_figure(args, selected, baseline, ours)

    metadata.update(
        {
            "csv_path": args.csv_path,
            "baseline_checkpoint": args.baseline_checkpoint,
            "ours_checkpoint": args.ours_checkpoint,
            "figure_metrics": figure_metrics,
        }
    )
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(_to_builtin(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"saved_metadata={args.metadata_json}", flush=True)
    print(json.dumps(_to_builtin({"selected": selected}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
