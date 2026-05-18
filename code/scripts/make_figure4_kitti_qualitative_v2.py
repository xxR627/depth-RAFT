#!/usr/bin/env python
"""Figure 4 v2: keep Figure 4 KITTI row selection and add SMURF qualitative/error columns.

Protocol note: use the Sintel-finetuned SMURF checkpoint zero-shot on KITTI so the comparison
matches the existing Sun-RAFT / Ours evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import make_figure4_kitti_qualitative as v1


PACKAGE_ROOT = v1.PACKAGE_ROOT
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_KITTI_ROOT = Path(r"G:\flow_data\KITTI\data_scene_flow\training")
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures"
DEFAULT_SELECTION_METADATA = DEFAULT_OUTPUT_DIR / "figure4_selection_metadata.json"
DEFAULT_SMURF_REPO = Path(r"G:\smurf_pytorch_port")
DEFAULT_SMURF_CKPT = DEFAULT_SMURF_REPO / "checkpoints" / "smurf_sintel.pt"
COLUMNS = [
    "Image",
    "GT Flow",
    "Sun-RAFT",
    "SMURF",
    "Ours",
    "Sun-RAFT Error",
    "SMURF Error",
    "Ours Error",
    "Valid GT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--selection-metadata", type=Path, default=DEFAULT_SELECTION_METADATA)
    parser.add_argument("--smurf-repo", type=Path, default=DEFAULT_SMURF_REPO)
    parser.add_argument("--smurf-checkpoint", type=Path, default=DEFAULT_SMURF_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT_DIR / "figure4_selection_metadata_v2.json")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _to_builtin(obj: Any) -> Any:
    return v1._to_builtin(obj)


def _import_smurf(smurf_repo: Path):
    smurf_repo_str = str(smurf_repo)
    if smurf_repo_str not in sys.path:
        sys.path.insert(0, smurf_repo_str)
    from smurf import raft_smurf

    return raft_smurf


def _load_selected_rows(selection_metadata: Path) -> list[dict[str, Any]]:
    payload = json.loads(selection_metadata.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, list) or not selected:
        raise RuntimeError(f"No selected rows found in {selection_metadata}")
    return selected


def _read_smurf_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = torchvision.io.read_image(str(path), mode=torchvision.io.ImageReadMode.RGB).float()
    image = 2.0 * (image / 255.0) - 1.0
    return image[None].to(device)


def _pad_to_multiple_of_8(image1: torch.Tensor, image2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    h, w = int(image1.shape[-2]), int(image1.shape[-1])
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return image1, image2, h, w
    pad = (0, pad_w, 0, pad_h)
    return (
        F.pad(image1, pad, mode="replicate"),
        F.pad(image2, pad, mode="replicate"),
        h,
        w,
    )


def _predict_smurf_pair(model, kitti_root: Path, frame_id: str, device: torch.device) -> np.ndarray:
    paths = v1._frame_paths(kitti_root, frame_id)
    image1 = _read_smurf_tensor(paths["image1"], device)
    image2 = _read_smurf_tensor(paths["image2"], device)
    image1, image2, h, w = _pad_to_multiple_of_8(image1, image2)
    with torch.no_grad():
        flow_predictions = model(image1, image2)
    flow = flow_predictions[-1][0, :, :h, :w].detach().cpu().permute(1, 2, 0).numpy()
    return flow.astype(np.float32)


def _collect_smurf_predictions(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    raft_smurf = _import_smurf(args.smurf_repo)
    model = raft_smurf(checkpoint=str(args.smurf_checkpoint)).to(device).eval()
    predictions: dict[str, np.ndarray] = {}
    for row in selected:
        frame_id = str(row["frame_id"])
        print(f"[figure4-smurf] {frame_id}", flush=True)
        predictions[frame_id] = _predict_smurf_pair(model, args.kitti_root, frame_id, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def _build_figure(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    baseline: dict[str, np.ndarray],
    smurf: dict[str, np.ndarray],
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

    fig, axes = plt.subplots(len(selected), len(COLUMNS), figsize=(20, 6))
    metrics: dict[str, Any] = {}
    for row_idx, row in enumerate(selected):
        frame_id = str(row["frame_id"])
        paths = v1._frame_paths(args.kitti_root, frame_id)
        image = v1._read_rgb(paths["image1"])
        flow_gt, valid = v1._read_kitti_flow(paths["flow_occ"])
        base_flow = baseline[frame_id]
        smurf_flow = smurf[frame_id]
        ours_flow = ours[frame_id]
        base_epe = np.sqrt(np.sum((base_flow - flow_gt) ** 2, axis=-1))
        smurf_epe = np.sqrt(np.sum((smurf_flow - flow_gt) ** 2, axis=-1))
        ours_epe = np.sqrt(np.sum((ours_flow - flow_gt) ** 2, axis=-1))
        base_sparse = np.where(valid, base_epe, np.nan)
        smurf_sparse = np.where(valid, smurf_epe, np.nan)
        ours_sparse = np.where(valid, ours_epe, np.nan)
        metrics[frame_id] = {
            "baseline_epe": float(np.nanmean(base_sparse)),
            "smurf_epe": float(np.nanmean(smurf_sparse)),
            "ours_epe": float(np.nanmean(ours_sparse)),
            "delta_epe_baseline_to_ours": float(np.nanmean(base_sparse) - np.nanmean(ours_sparse)),
            "delta_epe_smurf_to_ours": float(np.nanmean(smurf_sparse) - np.nanmean(ours_sparse)),
            "valid_pixels": int(valid.sum()),
        }
        panels = [
            image,
            v1._flow_to_color(flow_gt, valid=valid),
            v1._flow_to_color(base_flow),
            v1._flow_to_color(smurf_flow),
            v1._flow_to_color(ours_flow),
            base_sparse,
            smurf_sparse,
            ours_sparse,
            v1._overlay_valid_gt(image, valid, alpha=0.35),
        ]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx in (5, 6, 7):
                ax.imshow(image, aspect="auto", alpha=0.35)
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
                if col_idx == 5:
                    epe_value = metrics[frame_id]["baseline_epe"]
                elif col_idx == 6:
                    epe_value = metrics[frame_id]["smurf_epe"]
                else:
                    epe_value = metrics[frame_id]["ours_epe"]
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

    caption = (
        "Figure 4 v2: Qualitative comparison on KITTI 2015 train with SMURF added as an external baseline.\n"
        "Rows are inherited from the original Figure 4 selection based on Sun-RAFT vs Ours improvement, with SMURF shown as an external reference."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.12, wspace=0.02, hspace=0.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "figure4_kitti_qualitative_v2.pdf"
    png_path = args.output_dir / "figure4_kitti_qualitative_v2.png"
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
    args.selection_metadata = args.selection_metadata.expanduser().resolve()
    args.smurf_repo = args.smurf_repo.expanduser().resolve()
    args.smurf_checkpoint = args.smurf_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.metadata_json = args.metadata_json.expanduser().resolve()
    for path in (
        args.baseline_checkpoint,
        args.ours_checkpoint,
        args.dav2_weights,
        args.selection_metadata,
        args.smurf_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.smurf_repo.is_dir():
        raise FileNotFoundError(args.smurf_repo)
    for dirname in ("image_2", "flow_occ"):
        if not (args.kitti_root / dirname).is_dir():
            raise FileNotFoundError(args.kitti_root / dirname)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    selected = _load_selected_rows(args.selection_metadata)
    baseline, ours = v1._collect_selected_predictions(args, selected, device)
    smurf = _collect_smurf_predictions(args, selected, device)
    figure_metrics = _build_figure(args, selected, baseline, smurf, ours)

    metadata = {
        "selection_source": args.selection_metadata,
        "protocol_note": (
            "SMURF uses the Sintel-finetuned checkpoint and is evaluated zero-shot on KITTI "
            "to match the Sintel-trained Sun-RAFT / Ours protocol."
        ),
        "selected": selected,
        "figure_metrics": figure_metrics,
        "baseline_checkpoint": args.baseline_checkpoint,
        "ours_checkpoint": args.ours_checkpoint,
        "smurf_checkpoint": args.smurf_checkpoint,
        "outputs": {
            "pdf": args.output_dir / "figure4_kitti_qualitative_v2.pdf",
            "png": args.output_dir / "figure4_kitti_qualitative_v2.png",
        },
    }
    args.metadata_json.write_text(json.dumps(_to_builtin(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"saved_metadata={args.metadata_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

