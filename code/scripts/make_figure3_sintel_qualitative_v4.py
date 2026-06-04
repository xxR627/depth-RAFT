#!/usr/bin/env python
"""Figure 3 v4: keep v3 row selection and add a SMURF qualitative column."""

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

import make_figure3_sintel_qualitative_v2 as v2


PACKAGE_ROOT = v2.PACKAGE_ROOT
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures"
DEFAULT_SELECTION_METADATA = DEFAULT_OUTPUT_DIR / "figure3_selection_metadata_v3.json"
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
    "Unmatched",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--selection-metadata", type=Path, default=DEFAULT_SELECTION_METADATA)
    parser.add_argument("--smurf-repo", type=Path, default=DEFAULT_SMURF_REPO)
    parser.add_argument("--smurf-checkpoint", type=Path, default=DEFAULT_SMURF_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT_DIR / "figure3_selection_metadata_v4.json")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _to_builtin(obj: Any) -> Any:
    return v2._to_builtin(obj)


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


def _predict_smurf_pair(model, sintel_root: Path, scene: str, frame_idx: int, device: torch.device) -> np.ndarray:
    paths = v2._scene_paths(sintel_root, scene, frame_idx)
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
        key = f"{row['scene_name']}/frame_{int(row['frame_idx']):04d}"
        print(f"[figure-smurf] {key}", flush=True)
        predictions[key] = _predict_smurf_pair(model, args.sintel_root, str(row["scene_name"]), int(row["frame_idx"]), device)
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
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(len(selected), len(COLUMNS), figsize=(20, 8))
    metrics: dict[str, Any] = {}
    for row_idx, row in enumerate(selected):
        scene = str(row["scene_name"])
        frame_idx = int(row["frame_idx"])
        key = f"{scene}/frame_{frame_idx:04d}"
        paths = v2._scene_paths(args.sintel_root, scene, frame_idx)
        image = v2._read_rgb(paths["image1"])
        gt_flow = v2.frame_utils.readFlow(str(paths["flow"])).astype(np.float32)
        occ = v2._read_mask(paths["occ"])
        base_flow = baseline[key]
        smurf_flow = smurf[key]
        ours_flow = ours[key]
        base_epe = np.sqrt(np.sum((base_flow - gt_flow) ** 2, axis=-1))
        smurf_epe = np.sqrt(np.sum((smurf_flow - gt_flow) ** 2, axis=-1))
        ours_epe = np.sqrt(np.sum((ours_flow - gt_flow) ** 2, axis=-1))
        metrics[key] = {
            "frame_pair": f"frame_{frame_idx:04d}-frame_{frame_idx + 1:04d}",
            "baseline_epe": float(base_epe.mean()),
            "smurf_epe": float(smurf_epe.mean()),
            "ours_epe": float(ours_epe.mean()),
            "baseline_unmatched_epe": float(base_epe[occ].mean()) if occ.any() else None,
            "smurf_unmatched_epe": float(smurf_epe[occ].mean()) if occ.any() else None,
            "ours_unmatched_epe": float(ours_epe[occ].mean()) if occ.any() else None,
            "delta_unmatched": row.get("delta_unmatched"),
            "relative_unmatched_improvement": row.get("relative_unmatched_improvement"),
        }
        panels = [
            image,
            v2._flow_to_color(gt_flow),
            v2._flow_to_color(base_flow),
            v2._flow_to_color(smurf_flow),
            v2._flow_to_color(ours_flow),
            base_epe,
            smurf_epe,
            ours_epe,
            v2._overlay_unmatched(image, occ, alpha=0.25),
        ]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx in (5, 6, 7):
                ax.imshow(panel, cmap="viridis", vmin=0, vmax=3, aspect="auto")
                if col_idx == 5:
                    epe_value = metrics[key]["baseline_epe"]
                elif col_idx == 6:
                    epe_value = metrics[key]["smurf_epe"]
                else:
                    epe_value = metrics[key]["ours_epe"]
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
        "Figure 3 v4: Qualitative comparison on Sintel clean train with SMURF added as an external baseline.\n"
        "Our method still shows clearer recovery in unmatched regions (red overlay), especially near occlusion boundaries."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.12, wspace=0.02, hspace=0.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "figure3_sintel_qualitative_v4.pdf"
    png_path = args.output_dir / "figure3_sintel_qualitative_v4.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"saved_pdf={pdf_path}", flush=True)
    print(f"saved_png={png_path}", flush=True)
    return metrics


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.ours_checkpoint = args.ours_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    selected = _load_selected_rows(args.selection_metadata)
    baseline, ours = v2._collect_selected_predictions(args, selected, device)
    smurf = _collect_smurf_predictions(args, selected, device)
    figure_metrics = _build_figure(args, selected, baseline, smurf, ours)

    metadata = {
        "selection_source": args.selection_metadata,
        "selected": selected,
        "figure_metrics": figure_metrics,
        "baseline_checkpoint": args.baseline_checkpoint,
        "ours_checkpoint": args.ours_checkpoint,
        "smurf_checkpoint": args.smurf_checkpoint,
        "outputs": {
            "pdf": args.output_dir / "figure3_sintel_qualitative_v4.pdf",
            "png": args.output_dir / "figure3_sintel_qualitative_v4.png",
        },
    }
    args.metadata_json.write_text(json.dumps(_to_builtin(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"saved_metadata={args.metadata_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
