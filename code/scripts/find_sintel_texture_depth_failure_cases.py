#!/usr/bin/env python
"""Find Sintel failure/neutral cases in texture-rich but depth-smooth regions."""

from __future__ import annotations

import argparse
import json
import math
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

from dav2_wrapper import DAv2FeatureExtractor
from eval_depth_raft_region_decomp import _build_model as _build_g_z_model
from eval_sun_raft_region_decomp import _load_model as _load_baseline_model
from make_figure3_sintel_qualitative_v2 import _read_rgb, _scene_paths
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_FULL_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DGE_ONLY_CKPT = Path(r"G:\SCI NO.1\henrytask\ablation_dab_off_chairs105k_sintel35k\step_35000.pth")
DEFAULT_DAV2 = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures" / "sintel_texture_depth_failure_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL_CKPT)
    parser.add_argument("--dge-only-checkpoint", type=Path, default=DEFAULT_DGE_ONLY_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2)
    parser.add_argument(
        "--reference-model",
        choices=("baseline", "dge_only"),
        default="baseline",
        help="Reference model used in delta_epe = full_epe - ref_epe.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT_DIR / "candidates.json")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument(
        "--border-margin",
        type=int,
        default=0,
        help="Reject windows that touch the image border within this many pixels.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="Optional IoU threshold for candidate suppression on the same frame. 0 disables NMS.",
    )
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means scan all available pairs.")
    parser.add_argument(
        "--max-pairs-per-scene",
        type=int,
        default=0,
        help="Optional cap per scene after scene filtering. 0 means keep all frames from each scene.",
    )
    parser.add_argument("--min-rgb-grad", type=float, default=0.45)
    parser.add_argument("--min-flow-grad", type=float, default=0.40)
    parser.add_argument("--max-depth-grad", type=float, default=0.25)
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Optional comma-separated scene names, e.g. 'market_5,cave_4,bandage_2'.",
    )
    parser.add_argument("--device", type=str, default="cuda")
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


def _iter_sintel_pairs(sintel_root: Path, allowed_scenes: set[str] | None = None) -> list[tuple[str, int]]:
    clean_root = sintel_root / "training" / "clean"
    pairs: list[tuple[str, int]] = []
    for scene_dir in sorted(clean_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if allowed_scenes is not None and scene_dir.name not in allowed_scenes:
            continue
        frame_ids = sorted(int(path.stem.split("_")[1]) for path in scene_dir.glob("frame_*.png"))
        for frame_idx in frame_ids[:-1]:
            flow_path = sintel_root / "training" / "flow" / scene_dir.name / f"frame_{frame_idx:04d}.flo"
            if flow_path.is_file():
                pairs.append((scene_dir.name, frame_idx))
    return pairs


def _window_slices(height: int, width: int, window: int, stride: int) -> list[tuple[int, int, int, int]]:
    ys = list(range(0, max(height - window, 0) + 1, stride))
    xs = list(range(0, max(width - window, 0) + 1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != max(height - window, 0):
        ys.append(max(height - window, 0))
    if xs[-1] != max(width - window, 0):
        xs.append(max(width - window, 0))
    return [(x0, y0, min(x0 + window, width), min(y0 + window, height)) for y0 in ys for x0 in xs]


def _crop(a: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1]


def _touches_border(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    margin: int,
) -> bool:
    if margin <= 0:
        return False
    x0, y0, x1, y1 = box
    return x0 < margin or y0 < margin or x1 > width - margin or y1 > height - margin


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _suppress_overlaps(candidates: list[dict[str, Any]], iou_thresh: float) -> list[dict[str, Any]]:
    if iou_thresh <= 0.0:
        return candidates
    kept: list[dict[str, Any]] = []
    for item in candidates:
        box = tuple(int(v) for v in item["crop_xyxy"])
        scene = str(item["scene"])
        frame_idx = int(item["frame_idx"])
        should_keep = True
        for prev in kept:
            if str(prev["scene"]) != scene or int(prev["frame_idx"]) != frame_idx:
                continue
            prev_box = tuple(int(v) for v in prev["crop_xyxy"])
            if _box_iou(box, prev_box) >= iou_thresh:
                should_keep = False
                break
        if should_keep:
            kept.append(item)
    return kept


def _predict_scene_sequence(
    model,
    sintel_root: Path,
    scene: str,
    target_frames: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    wanted = set(int(frame_idx) for frame_idx in target_frames)
    if not wanted:
        return {}
    max_frame = max(wanted)
    predictions: dict[int, np.ndarray] = {}
    flow_prev = None
    model.eval()
    with torch.no_grad():
        for frame_idx in range(1, max_frame + 1):
            paths = _scene_paths(sintel_root, scene, frame_idx)
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
            if frame_idx in wanted:
                predictions[frame_idx] = flow_up
    missing = sorted(wanted - set(predictions))
    if missing:
        raise RuntimeError(f"Missing predictions for {scene}: {missing}")
    return predictions


def _score_candidate(
    *,
    delta_epe_map: np.ndarray,
    rgb_grad_norm: np.ndarray,
    depth_grad_norm: np.ndarray,
    flow_grad_norm: np.ndarray,
    box: tuple[int, int, int, int],
    min_rgb_grad: float,
    min_flow_grad: float,
    max_depth_grad: float,
) -> dict[str, float] | None:
    delta_crop = _crop(delta_epe_map, box)
    rgb_crop = _crop(rgb_grad_norm, box)
    depth_crop = _crop(depth_grad_norm, box)
    flow_crop = _crop(flow_grad_norm, box)

    delta_mean = float(np.mean(delta_crop))
    if not math.isfinite(delta_mean) or delta_mean < 0.0:
        return None

    rgb_mean = float(np.mean(rgb_crop))
    depth_mean = float(np.mean(depth_crop))
    flow_mean = float(np.mean(flow_crop))
    if rgb_mean < min_rgb_grad or flow_mean < min_flow_grad or depth_mean > max_depth_grad:
        return None

    score = delta_mean + 0.35 * rgb_mean + 0.35 * flow_mean - 0.35 * depth_mean
    return {
        "score": float(score),
        "delta_epe_mean": delta_mean,
        "rgb_grad_mean": rgb_mean,
        "depth_grad_mean": depth_mean,
        "flow_grad_mean": flow_mean,
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
    ref_epe: np.ndarray,
    full_epe: np.ndarray,
    box: tuple[int, int, int, int],
    reference_label: str,
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

    ref_mean = float(np.mean(_crop(ref_epe, box)))
    full_mean = float(np.mean(_crop(full_epe, box)))
    delta_mean = full_mean - ref_mean
    fig.suptitle(
        f"{scene} frame_{frame_idx:04d} | crop=({x0},{y0})-({x1},{y1}) | "
        f"{reference_label} EPE={ref_mean:.2f} | full EPE={full_mean:.2f} | delta={delta_mean:+.2f}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.86, bottom=0.02, wspace=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.full_checkpoint = args.full_checkpoint.expanduser().resolve()
    args.dge_only_checkpoint = args.dge_only_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.metadata_json = args.metadata_json.expanduser().resolve()
    allowed_scenes = {scene.strip() for scene in args.scenes.split(",") if scene.strip()} or None

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    reference_label = "Sun-RAFT" if args.reference_model == "baseline" else "DGE-only"
    baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
    full_model = _build_g_z_model(
        device=device,
        baseline_checkpoint=args.baseline_checkpoint,
        fusion_checkpoint=args.full_checkpoint,
        dav2_weights=args.dav2_weights,
    )
    dge_only_model = None
    if args.reference_model == "dge_only":
        if not args.dge_only_checkpoint.is_file():
            raise FileNotFoundError(args.dge_only_checkpoint)
        dge_only_model = _build_g_z_model(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.dge_only_checkpoint,
            dav2_weights=args.dav2_weights,
        )

    extractor = DAv2FeatureExtractor(str(args.dav2_weights), device=device, precision="auto")
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

    ref_cache: dict[tuple[str, int], np.ndarray] = {}
    full_cache: dict[tuple[str, int], np.ndarray] = {}
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
        ref_predictions = _predict_scene_sequence(
            baseline_model if args.reference_model == "baseline" else dge_only_model,
            args.sintel_root,
            scene,
            frame_list,
            device,
        )
        full_predictions = _predict_scene_sequence(full_model, args.sintel_root, scene, frame_list, device)
        for frame_idx, pred in ref_predictions.items():
            ref_cache[(scene, frame_idx)] = pred
        for frame_idx, pred in full_predictions.items():
            full_cache[(scene, frame_idx)] = pred

        for frame_idx in frame_list:
            processed_pairs += 1
            print(f"[scan] {processed_pairs}/{len(pairs)} {scene}/frame_{frame_idx:04d}", flush=True)
            paths = _scene_paths(args.sintel_root, scene, frame_idx)
            image = _read_rgb(paths["image1"])
            gt_flow = frame_utils.readFlow(str(paths["flow"])).astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            ref_flow = ref_cache[(scene, frame_idx)]
            full_flow = full_cache[(scene, frame_idx)]
            ref_epe = np.sqrt(np.sum((ref_flow - gt_flow) ** 2, axis=-1))
            full_epe = np.sqrt(np.sum((full_flow - gt_flow) ** 2, axis=-1))
            delta_epe = full_epe - ref_epe

            depth = _extract_depth(extractor, image, device)
            depth_norm = _robust_norm(depth)
            rgb_grad_norm = _robust_norm(_gradient_mag(gray))
            depth_grad_norm = _robust_norm(_gradient_mag(depth_norm))
            flow_grad_norm = _robust_norm(_flow_gradient_mag(gt_flow))

            height, width = delta_epe.shape
            for box in _window_slices(height, width, args.window_size, args.window_stride):
                if _touches_border(box, width=width, height=height, margin=args.border_margin):
                    continue
                score_payload = _score_candidate(
                    delta_epe_map=delta_epe,
                    rgb_grad_norm=rgb_grad_norm,
                    depth_grad_norm=depth_grad_norm,
                    flow_grad_norm=flow_grad_norm,
                    box=box,
                    min_rgb_grad=args.min_rgb_grad,
                    min_flow_grad=args.min_flow_grad,
                    max_depth_grad=args.max_depth_grad,
                )
                if score_payload is None:
                    continue
                candidates.append(
                    {
                        "scene": scene,
                        "frame_idx": frame_idx,
                        "crop_xyxy": list(box),
                        "reference_model": args.reference_model,
                        "reference_label": reference_label,
                        "ref_crop_epe": float(np.mean(_crop(ref_epe, box))),
                        "full_crop_epe": float(np.mean(_crop(full_epe, box))),
                        **score_payload,
                    }
                )

    candidates.sort(key=lambda item: (float(item["delta_epe_mean"]), float(item["score"])), reverse=True)
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
        ref_epe = np.sqrt(np.sum((ref_cache[(scene, frame_idx)] - gt_flow) ** 2, axis=-1))
        full_epe = np.sqrt(np.sum((full_cache[(scene, frame_idx)] - gt_flow) ** 2, axis=-1))

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
            ref_epe=ref_epe,
            full_epe=full_epe,
            box=box,
            reference_label=reference_label,
        )
        item["figure_path"] = figure_path

    payload = {
        "reference_model": args.reference_model,
        "reference_label": reference_label,
        "baseline_checkpoint": args.baseline_checkpoint,
        "full_checkpoint": args.full_checkpoint,
        "dge_only_checkpoint": args.dge_only_checkpoint if args.dge_only_checkpoint.is_file() else None,
        "dav2_weights": args.dav2_weights,
        "sintel_root": args.sintel_root,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "border_margin": args.border_margin,
        "nms_iou": args.nms_iou,
        "scenes": sorted(allowed_scenes) if allowed_scenes is not None else None,
        "num_pairs_scanned": len(pairs),
        "num_candidates_total": len(candidates),
        "top_candidates": top_candidates,
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    print(f"saved_metadata={args.metadata_json}", flush=True)
    for item in top_candidates:
        print(
            f"candidate scene={item['scene']} frame={int(item['frame_idx']):04d} "
            f"crop={tuple(item['crop_xyxy'])} delta_epe={float(item['delta_epe_mean']):+.4f} "
            f"rgb={float(item['rgb_grad_mean']):.3f} depth={float(item['depth_grad_mean']):.3f} "
            f"flow={float(item['flow_grad_mean']):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
