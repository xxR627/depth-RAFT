#!/usr/bin/env python
"""Cross-domain KITTI 2015 train eval for Sun-RAFT baseline and G+Z checkpoint.

Sun-RAFT Table 2 reports KITTI EPE as the mean of per-pair EPE values, while
FL is the official pixel-weighted outlier percentage. Keep the pixel-weighted
EPE in details for traceability, but expose Table-2-aligned EPE at top level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

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
from eval_sun_raft_region_decomp import _load_model as _load_baseline_model, _resolve_device
from utils import frame_utils
from utils.utils import InputPadder


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_KITTI_ROOT = Path(r"G:\flow_data\KITTI\data_scene_flow\training")
DEFAULT_OUTPUT_JSON = PACKAGE_ROOT / "results" / "main" / "kitti_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = np.array(frame_utils.read_gen(str(path))).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)


def _read_flow(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    flow_np, valid_np = frame_utils.readFlowKITTI(str(path))
    flow = torch.from_numpy(flow_np).float().to(device)
    valid = torch.from_numpy(valid_np > 0).bool().to(device)
    return flow, valid


def _accumulate_metrics(
    *,
    pred_flow: torch.Tensor,
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    sum_epe: float,
    total_valid: int,
    total_outliers: int,
    pair_epes: list[float],
    pair_f1s: list[float],
) -> tuple[float, int, int]:
    epe = torch.sum((pred_flow - flow_gt) ** 2, dim=-1).sqrt()
    mag = torch.sum(flow_gt ** 2, dim=-1).sqrt()
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return sum_epe, total_valid, total_outliers
    valid_epe = epe[valid]
    outlier = ((epe > 3.0) & ((epe / torch.clamp(mag, min=1e-9)) > 0.05) & valid)
    outlier_count = int(outlier.sum().item())
    sum_epe += float(valid_epe.sum().item())
    total_valid += valid_count
    total_outliers += outlier_count
    pair_epes.append(float(valid_epe.mean().item()))
    pair_f1s.append(100.0 * outlier_count / valid_count)
    return sum_epe, total_valid, total_outliers


def _finalize(sum_epe: float, total_valid: int, total_outliers: int, pair_epes: list[float], pair_f1s: list[float]):
    return {
        "epe": sum_epe / max(total_valid, 1),
        "f1": 100.0 * total_outliers / max(total_valid, 1),
        "total_valid_pixels": total_valid,
        "pair_mean_epe": float(np.mean(pair_epes)) if pair_epes else float("nan"),
        "pair_mean_f1": float(np.mean(pair_f1s)) if pair_f1s else float("nan"),
    }


def _run_model(
    *,
    model,
    kitti_root: Path,
    max_pairs: int,
    log_every: int,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    image1_paths = sorted((kitti_root / "image_2").glob("*_10.png"))
    if max_pairs > 0:
        image1_paths = image1_paths[:max_pairs]
    if not image1_paths:
        raise RuntimeError(f"No KITTI image pairs found under {kitti_root}")

    all_sum = noc_sum = 0.0
    all_valid = noc_valid = 0
    all_out = noc_out = 0
    all_pair_epes: list[float] = []
    all_pair_f1s: list[float] = []
    noc_pair_epes: list[float] = []
    noc_pair_f1s: list[float] = []
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for idx, image1_path in enumerate(image1_paths):
            stem = image1_path.name.replace("_10.png", "")
            image2_path = kitti_root / "image_2" / f"{stem}_11.png"
            flow_occ_path = kitti_root / "flow_occ" / f"{stem}_10.png"
            flow_noc_path = kitti_root / "flow_noc" / f"{stem}_10.png"
            if not image2_path.is_file() or not flow_occ_path.is_file() or not flow_noc_path.is_file():
                raise FileNotFoundError(f"Missing KITTI pair/flow files for {stem}")

            image1 = _load_image(image1_path, device)
            image2 = _load_image(image2_path, device)
            padder = InputPadder(image1.shape, mode="kitti", coarsest_scale=8)
            image1_pad, image2_pad = padder.pad(image1, image2)
            _, flow_pr = model(image1_pad, image2_pad, iters=12, test_mode=True, bw=False)
            pred = padder.unpad(flow_pr[0]).permute(1, 2, 0)

            flow_occ, valid_occ = _read_flow(flow_occ_path, device)
            flow_noc, valid_noc = _read_flow(flow_noc_path, device)
            all_sum, all_valid, all_out = _accumulate_metrics(
                pred_flow=pred,
                flow_gt=flow_occ,
                valid=valid_occ,
                sum_epe=all_sum,
                total_valid=all_valid,
                total_outliers=all_out,
                pair_epes=all_pair_epes,
                pair_f1s=all_pair_f1s,
            )
            noc_sum, noc_valid, noc_out = _accumulate_metrics(
                pred_flow=pred,
                flow_gt=flow_noc,
                valid=valid_noc,
                sum_epe=noc_sum,
                total_valid=noc_valid,
                total_outliers=noc_out,
                pair_epes=noc_pair_epes,
                pair_f1s=noc_pair_f1s,
            )

            if ((idx + 1) % log_every == 0) or (idx + 1 == len(image1_paths)):
                print(
                    f"[kitti:{label}] pair={idx + 1}/{len(image1_paths)} frame={stem} "
                    f"elapsed_sec={time.time() - start_time:.1f}",
                    flush=True,
                )

    all_metrics = _finalize(all_sum, all_valid, all_out, all_pair_epes, all_pair_f1s)
    noc_metrics = _finalize(noc_sum, noc_valid, noc_out, noc_pair_epes, noc_pair_f1s)
    return {
        "epe_all": all_metrics["pair_mean_epe"],
        "epe_noc": noc_metrics["pair_mean_epe"],
        "f1_all": all_metrics["f1"],
        "f1_noc": noc_metrics["f1"],
        "details": {
            "all": all_metrics,
            "noc": noc_metrics,
            "num_pairs": len(image1_paths),
        },
    }


def _delta_pct(baseline: dict[str, float], ours: dict[str, float]) -> dict[str, str]:
    return {
        "epe_all": f"{100.0 * (ours['epe_all'] - baseline['epe_all']) / baseline['epe_all']:+.2f}%",
        "epe_noc": f"{100.0 * (ours['epe_noc'] - baseline['epe_noc']) / baseline['epe_noc']:+.2f}%",
        "f1_all": f"{ours['f1_all'] - baseline['f1_all']:+.2f} abs%",
        "f1_noc": f"{ours['f1_noc'] - baseline['f1_noc']:+.2f} abs%",
    }


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.ours_checkpoint = args.ours_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.kitti_root = args.kitti_root.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()

    payload: dict[str, Any] = {
        "status": "FAILED",
        "settings": {
            "baseline_checkpoint": args.baseline_checkpoint,
            "ours_checkpoint": args.ours_checkpoint,
            "dav2_weights": args.dav2_weights,
            "kitti_root": args.kitti_root,
            "protocol": (
                "KITTI 2015 train 200 pairs, iters=12, padding=kitti, no finetune. "
                "Table-2-aligned EPE uses pair_mean_epe; FL uses pixel-weighted outlier rate."
            ),
        },
    }
    try:
        for path, label in (
            (args.baseline_checkpoint, "baseline checkpoint"),
            (args.ours_checkpoint, "ours checkpoint"),
            (args.dav2_weights, "DAv2 weights"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        for dirname in ("image_2", "flow_occ", "flow_noc"):
            if not (args.kitti_root / dirname).is_dir():
                raise FileNotFoundError(f"Missing KITTI directory: {args.kitti_root / dirname}")

        device = _resolve_device()
        baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
        baseline = _run_model(
            model=baseline_model,
            kitti_root=args.kitti_root,
            max_pairs=args.max_pairs,
            log_every=args.log_every,
            device=device,
            label="baseline",
        )
        del baseline_model
        torch.cuda.empty_cache()

        ours_model = _build_g_z_model(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.ours_checkpoint,
            dav2_weights=args.dav2_weights,
        )
        ours = _run_model(
            model=ours_model,
            kitti_root=args.kitti_root,
            max_pairs=args.max_pairs,
            log_every=args.log_every,
            device=device,
            label="ours_step35000",
        )

        baseline_compact = {key: float(baseline[key]) for key in ("epe_all", "epe_noc", "f1_all", "f1_noc")}
        ours_compact = {key: float(ours[key]) for key in ("epe_all", "epe_noc", "f1_all", "f1_noc")}
        payload.update(
            {
                "baseline": baseline_compact,
                "ours_step35000": ours_compact,
                "delta_pct": _delta_pct(baseline_compact, ours_compact),
                "details": {"baseline": baseline["details"], "ours_step35000": ours["details"]},
                "status": "PASSED",
            }
        )
        _write_json(args.output_json, payload)
        print(f"saved_json={args.output_json}")
        print(f"baseline={baseline_compact}")
        print(f"ours_step35000={ours_compact}")
        print(f"delta_pct={payload['delta_pct']}")
        return 0
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_json, payload)
        print(f"FAILED: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



