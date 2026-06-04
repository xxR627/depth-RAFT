#!/usr/bin/env python
"""Find interior Sintel crops where DAB is neutral/slightly worse under Fig. 6 ablation checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
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
from find_sintel_texture_depth_failure_cases import (
    _crop,
    _extract_depth,
    _flow_gradient_mag,
    _gradient_mag,
    _iter_sintel_pairs,
    _predict_scene_sequence,
    _robust_norm,
    _suppress_overlaps,
    _to_builtin,
    _touches_border,
    _window_slices,
)
from make_figure3_sintel_qualitative_v2 import _read_rgb, _scene_paths
from utils import frame_utils


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_G_ONLY_CKPT = Path(r"G:\SCI NO.1\henrytask\paper_ablation_phi_d_short_10k5k\phi_only\sintel_5k\step_5000.pth")
DEFAULT_Z_ONLY_CKPT = Path(r"G:\SCI NO.1\henrytask\paper_ablation_phi_d_short_10k5k\d_only\sintel_5k\step_5000.pth")
DEFAULT_DGE_CKPT = Path(r"G:\SCI NO.1\henrytask\ablation_dab_off_chairs105k_sintel35k\step_17500.pth")
DEFAULT_FULL_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2 = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures" / "sintel_fig6_dab_neutral_cases"

MODEL_ORDER = ("baseline", "g_only", "z_only", "dge", "full")
MODEL_LABELS = {
    "baseline": "Baseline",
    "g_only": "+G",
    "z_only": "+Z",
    "dge": "+DGE",
    "full": "+DGE+DAB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--g-only-checkpoint", type=Path, default=DEFAULT_G_ONLY_CKPT)
    parser.add_argument("--z-only-checkpoint", type=Path, default=DEFAULT_Z_ONLY_CKPT)
    parser.add_argument("--dge-checkpoint", type=Path, default=DEFAULT_DGE_CKPT)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT_DIR / "candidates.json")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--window-stride", type=int, default=48)
    parser.add_argument("--border-margin", type=int, default=32)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means scan all available pairs.")
    parser.add_argument(
        "--max-pairs-per-scene",
        type=int,
        default=0,
        help="Optional cap per scene after scene filtering. 0 means keep all frames from each scene.",
    )
    parser.add_argument("--min-rgb-grad", type=float, default=0.35)
    parser.add_argument("--min-flow-grad", type=float, default=0.30)
    parser.add_argument("--max-depth-grad", type=float, default=0.22)
    parser.add_argument("--min-dab-delta", type=float, default=0.0)
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Optional comma-separated scene names, e.g. 'market_5,cave_4,bandage_2'.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.g_only_checkpoint = args.g_only_checkpoint.expanduser().resolve()
    args.z_only_checkpoint = args.z_only_checkpoint.expanduser().resolve()
    args.dge_checkpoint = args.dge_checkpoint.expanduser().resolve()
    args.full_checkpoint = args.full_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.metadata_json = args.metadata_json.expanduser().resolve()
    return args


def _build_models(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    models: dict[str, Any] = {
        "baseline": _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint),
        "g_only": _build_g_z_model(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.g_only_checkpoint,
            dav2_weights=args.dav2_weights,
        ),
        "z_only": _build_g_z_model(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.z_only_checkpoint,
            dav2_weights=args.dav2_weights,
        ),
        "dge": _build_g_z_model(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.dge_checkpoint,
            dav2_weights=args.dav2_weights,
        ),
        "full": _build_g_z_model(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.full_checkpoint,
            dav2_weights=args.dav2_weights,
        ),
    }
    return models


def _epe_means_for_box(epe_maps: dict[str, np.ndarray], box: tuple[int, int, int, int]) -> dict[str, float]:
    return {name: float(np.mean(_crop(epe_map, box))) for name, epe_map in epe_maps.items()}


def _score_candidate(
    *,
    crop_epe: dict[str, float],
    rgb_grad_mean: float,
    depth_grad_mean: float,
    flow_grad_mean: float,
    min_rgb_grad: float,
    min_flow_grad: float,
    max_depth_grad: float,
    min_dab_delta: float,
) -> dict[str, float] | None:
    if rgb_grad_mean < min_rgb_grad or flow_grad_mean < min_flow_grad or depth_grad_mean > max_depth_grad:
        return None

    baseline = float(crop_epe["baseline"])
    g_only = float(crop_epe["g_only"])
    z_only = float(crop_epe["z_only"])
    dge = float(crop_epe["dge"])
    full = float(crop_epe["full"])
    dab_delta = full - dge
    if dab_delta < min_dab_delta:
        return None

    deviations = [abs(g_only - baseline), abs(z_only - baseline), abs(dge - baseline), abs(full - baseline)]
    mean_abs_delta = float(np.mean(deviations))
    max_abs_delta = float(np.max(deviations))
    step_range = float(max(crop_epe.values()) - min(crop_epe.values()))
    cue_score = float(0.35 * rgb_grad_mean + 0.35 * flow_grad_mean - 0.35 * depth_grad_mean)
    rank_score = float(cue_score - 2.0 * mean_abs_delta - 1.0 * max_abs_delta - 0.5 * step_range)

    return {
        "dab_delta": float(dab_delta),
        "mean_abs_delta_vs_baseline": mean_abs_delta,
        "max_abs_delta_vs_baseline": max_abs_delta,
        "step_range": step_range,
        "cue_score": cue_score,
        "score": rank_score,
    }


def _save_candidate_figure(
    *,
    output_path: Path,
    scene: str,
    frame_idx: int,
    image: np.ndarray,
    depth_norm: np.ndarray,
    rgb_grad_norm: np.ndarray,
    depth_grad_norm: np.ndarray,
    flow_grad_norm: np.ndarray,
    box: tuple[int, int, int, int],
    crop_epe: dict[str, float],
    dab_delta: float,
) -> None:
    x0, y0, x1, y1 = box
    panels = [
        ("Input", image),
        ("RGB Gradient", rgb_grad_norm),
        ("DAv2 Depth", depth_norm),
        ("Norm. Depth Gradient", depth_grad_norm),
        ("GT Flow Gradient", flow_grad_norm),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.8), constrained_layout=False)
    for ax, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            cmap = "viridis" if title == "DAv2 Depth" else "magma"
            ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(panel)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ff3b30", lw=2.0)
        ax.add_patch(rect)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    title_line1 = (
        f"{scene} frame_{frame_idx:04d} | crop=({x0},{y0})-({x1},{y1}) | "
        f"B={crop_epe['baseline']:.2f} | +G={crop_epe['g_only']:.2f} | +Z={crop_epe['z_only']:.2f}"
    )
    title_line2 = (
        f"+DGE={crop_epe['dge']:.2f} | +DGE+DAB={crop_epe['full']:.2f} | "
        f"delta(DAB-DGE)={dab_delta:+.2f}"
    )
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=10.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.82, bottom=0.02, wspace=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    args = _resolve_args(parse_args())
    allowed_scenes = {scene.strip() for scene in args.scenes.split(",") if scene.strip()} or None

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    for path in (
        args.baseline_checkpoint,
        args.g_only_checkpoint,
        args.z_only_checkpoint,
        args.dge_checkpoint,
        args.full_checkpoint,
        args.dav2_weights,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    models = _build_models(args, device)
    extractor = models["full"].dav2_extractor
    extractor.eval()

    pairs = _iter_sintel_pairs(args.sintel_root, allowed_scenes=allowed_scenes)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    scene_to_frames: dict[str, list[int]] = {}
    for scene, frame_idx in pairs:
        scene_to_frames.setdefault(scene, []).append(frame_idx)
    if args.max_pairs_per_scene > 0:
        for scene, frame_list in list(scene_to_frames.items()):
            scene_to_frames[scene] = sorted(frame_list)[: args.max_pairs_per_scene]
        pairs = [(scene, frame_idx) for scene, frame_list in sorted(scene_to_frames.items()) for frame_idx in frame_list]

    pred_cache: dict[str, dict[tuple[str, int], np.ndarray]] = {name: {} for name in MODEL_ORDER}
    candidates: list[dict[str, Any]] = []

    scene_items = sorted(scene_to_frames.items())
    processed_pairs = 0
    for scene_idx, (scene, frame_list) in enumerate(scene_items, start=1):
        frame_list = sorted(frame_list)
        print(
            f"[scene] {scene_idx}/{len(scene_items)} {scene} frames={len(frame_list)} "
            f"range={frame_list[0]:04d}-{frame_list[-1]:04d}",
            flush=True,
        )
        for model_name in MODEL_ORDER:
            print(f"[predict] {scene} model={model_name}", flush=True)
            scene_predictions = _predict_scene_sequence(models[model_name], args.sintel_root, scene, frame_list, device)
            for frame_idx, pred in scene_predictions.items():
                pred_cache[model_name][(scene, frame_idx)] = pred

        for frame_idx in frame_list:
            processed_pairs += 1
            print(f"[scan] {processed_pairs}/{len(pairs)} {scene}/frame_{frame_idx:04d}", flush=True)
            paths = _scene_paths(args.sintel_root, scene, frame_idx)
            image = _read_rgb(paths["image1"])
            gt_flow = frame_utils.readFlow(str(paths["flow"])).astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            epe_maps: dict[str, np.ndarray] = {}
            for model_name in MODEL_ORDER:
                pred = pred_cache[model_name][(scene, frame_idx)]
                epe_maps[model_name] = np.sqrt(np.sum((pred - gt_flow) ** 2, axis=-1))

            depth = _extract_depth(extractor, image, device)
            depth_norm = _robust_norm(depth)
            rgb_grad_norm = _robust_norm(_gradient_mag(gray))
            depth_grad_norm = _robust_norm(_gradient_mag(depth_norm))
            flow_grad_norm = _robust_norm(_flow_gradient_mag(gt_flow))

            height, width = gt_flow.shape[:2]
            for box in _window_slices(height, width, args.window_size, args.window_stride):
                if _touches_border(box, width=width, height=height, margin=args.border_margin):
                    continue
                crop_epe = _epe_means_for_box(epe_maps, box)
                rgb_grad_mean = float(np.mean(_crop(rgb_grad_norm, box)))
                depth_grad_mean = float(np.mean(_crop(depth_grad_norm, box)))
                flow_grad_mean = float(np.mean(_crop(flow_grad_norm, box)))
                score_payload = _score_candidate(
                    crop_epe=crop_epe,
                    rgb_grad_mean=rgb_grad_mean,
                    depth_grad_mean=depth_grad_mean,
                    flow_grad_mean=flow_grad_mean,
                    min_rgb_grad=args.min_rgb_grad,
                    min_flow_grad=args.min_flow_grad,
                    max_depth_grad=args.max_depth_grad,
                    min_dab_delta=args.min_dab_delta,
                )
                if score_payload is None:
                    continue
                candidates.append(
                    {
                        "scene": scene,
                        "frame_idx": frame_idx,
                        "crop_xyxy": list(box),
                        "crop_epe": crop_epe,
                        "rgb_grad_mean": rgb_grad_mean,
                        "depth_grad_mean": depth_grad_mean,
                        "flow_grad_mean": flow_grad_mean,
                        **score_payload,
                    }
                )

    candidates.sort(
        key=lambda item: (
            float(item["mean_abs_delta_vs_baseline"]),
            float(item["max_abs_delta_vs_baseline"]),
            -float(item["dab_delta"]),
            -float(item["cue_score"]),
        )
    )
    candidates = _suppress_overlaps(candidates, args.nms_iou)
    top_candidates = candidates[: args.top_k]

    for rank, item in enumerate(top_candidates, start=1):
        scene = str(item["scene"])
        frame_idx = int(item["frame_idx"])
        box = tuple(int(v) for v in item["crop_xyxy"])
        paths = _scene_paths(args.sintel_root, scene, frame_idx)
        image = _read_rgb(paths["image1"])
        gt_flow = frame_utils.readFlow(str(paths["flow"])).astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth = _extract_depth(extractor, image, device)
        depth_norm = _robust_norm(depth)
        rgb_grad_norm = _robust_norm(_gradient_mag(gray))
        depth_grad_norm = _robust_norm(_gradient_mag(depth_norm))
        flow_grad_norm = _robust_norm(_flow_gradient_mag(gt_flow))
        figure_path = args.output_dir / f"{rank:02d}_{scene}_frame_{frame_idx:04d}.png"
        _save_candidate_figure(
            output_path=figure_path,
            scene=scene,
            frame_idx=frame_idx,
            image=image,
            depth_norm=depth_norm,
            rgb_grad_norm=rgb_grad_norm,
            depth_grad_norm=depth_grad_norm,
            flow_grad_norm=flow_grad_norm,
            box=box,
            crop_epe={name: float(item["crop_epe"][name]) for name in MODEL_ORDER},
            dab_delta=float(item["dab_delta"]),
        )
        item["figure_path"] = figure_path

    payload = {
        "checkpoints": {
            "baseline": args.baseline_checkpoint,
            "g_only": args.g_only_checkpoint,
            "z_only": args.z_only_checkpoint,
            "dge": args.dge_checkpoint,
            "full": args.full_checkpoint,
        },
        "labels": {name: MODEL_LABELS[name] for name in MODEL_ORDER},
        "dav2_weights": args.dav2_weights,
        "sintel_root": args.sintel_root,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "border_margin": args.border_margin,
        "nms_iou": args.nms_iou,
        "min_rgb_grad": args.min_rgb_grad,
        "min_flow_grad": args.min_flow_grad,
        "max_depth_grad": args.max_depth_grad,
        "min_dab_delta": args.min_dab_delta,
        "scenes": sorted(allowed_scenes) if allowed_scenes is not None else None,
        "num_pairs_scanned": len(pairs),
        "num_candidates_total": len(candidates),
        "top_candidates": top_candidates,
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    print(f"saved_metadata={args.metadata_json}", flush=True)
    for item in top_candidates:
        crop_epe = item["crop_epe"]
        print(
            f"candidate scene={item['scene']} frame={int(item['frame_idx']):04d} "
            f"crop={tuple(item['crop_xyxy'])} B={float(crop_epe['baseline']):.4f} "
            f"G={float(crop_epe['g_only']):.4f} Z={float(crop_epe['z_only']):.4f} "
            f"DGE={float(crop_epe['dge']):.4f} FULL={float(crop_epe['full']):.4f} "
            f"delta_dab={float(item['dab_delta']):+.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
