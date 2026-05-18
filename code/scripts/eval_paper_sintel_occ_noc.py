#!/usr/bin/env python
"""Evaluate Sun-RAFT baseline and G+Z checkpoint on official Sintel occ/noc splits."""

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

import datasets_un
from eval_depth_raft_region_decomp import _build_model as _build_g_z_model
from eval_sun_raft_region_decomp import (
    COARSEST_SCALE,
    ITERS,
    _load_model as _load_baseline_model,
    _patched_get_extention_factory,
    _resolve_device,
)
from sintel_occ_utils import (
    PASSES,
    REGIONS,
    _finalize_accumulator,
    _init_accumulator,
    _official_region_masks,
    _update_metrics,
)
from utils.utils import InputPadder, forward_interpolate


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_OUTPUT_JSON = PACKAGE_ROOT / "results" / "main" / "sintel_occ_noc_eval.json"
EXPECTED_ALL = {
    "baseline": {"clean": 1.6942, "final": 2.5935},
    "ours_step35000": {"clean": 1.4970, "final": 2.4693},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--log-every", type=int, default=50)
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


def _run_pass(
    *,
    model,
    sintel_root: Path,
    dstype: str,
    max_pairs: int,
    log_every: int,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    val_dataset = datasets_un.MpiSintel(
        split="training",
        dstype=dstype,
        show_extra_info=True,
        read_flow_gt=True,
    )
    total_pairs = len(val_dataset) if max_pairs <= 0 else min(len(val_dataset), max_pairs)
    accumulator = _init_accumulator()
    flow_prev = None
    sequence_prev = None
    start_time = time.time()

    with torch.no_grad():
        for val_id in range(total_pairs):
            image1, image2, flow_gt_cpu, _, (sequence, frame) = val_dataset[val_id]
            if sequence != sequence_prev:
                flow_prev = None

            image1 = image1[None].to(device)
            image2 = image2[None].to(device)
            height, width = int(flow_gt_cpu.shape[-2]), int(flow_gt_cpu.shape[-1])
            masks_cpu = _official_region_masks(
                sintel_root=sintel_root,
                sequence=sequence,
                frame_zero_based=frame,
                height=height,
                width=width,
            )

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
            flow_pred = padder.unpad(flow_pr[0]).cpu()
            _update_metrics(accumulator, flow_pred, flow_gt_cpu, masks_cpu)
            flow_prev = forward_interpolate(flow_low[0])[None].to(device)
            sequence_prev = sequence

            if ((val_id + 1) % log_every == 0) or (val_id + 1 == total_pairs):
                elapsed = time.time() - start_time
                print(
                    f"[sintel-occ-noc:{label}:{dstype}] pair={val_id + 1}/{total_pairs} "
                    f"scene={sequence} frame={frame + 1:04d} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    return _finalize_accumulator(accumulator)


def _compact(pass_result: dict[str, Any]) -> dict[str, float]:
    return {
        "all": float(pass_result["all"]["epe"]),
        "matched": float(pass_result["matched_noc"]["epe"]),
        "unmatched": float(pass_result["unmatched_occ"]["epe"]),
        "official_valid": float(pass_result["official_valid"]["epe"]),
    }


def _run_model(model, *, sintel_root: Path, max_pairs: int, log_every: int, device: torch.device, label: str):
    return {
        dstype: _run_pass(
            model=model,
            sintel_root=sintel_root,
            dstype=dstype,
            max_pairs=max_pairs,
            log_every=log_every,
            device=device,
            label=label,
        )
        for dstype in PASSES
    }


def _make_delta(baseline: dict[str, dict[str, float]], ours: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "clean_matched": ours["clean"]["matched"] - baseline["clean"]["matched"],
        "clean_unmatched": ours["clean"]["unmatched"] - baseline["clean"]["unmatched"],
        "final_matched": ours["final"]["matched"] - baseline["final"]["matched"],
        "final_unmatched": ours["final"]["unmatched"] - baseline["final"]["unmatched"],
        "clean_all": ours["clean"]["all"] - baseline["clean"]["all"],
        "final_all": ours["final"]["all"] - baseline["final"]["all"],
    }


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.ours_checkpoint = args.ours_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()

    payload: dict[str, Any] = {
        "status": "FAILED",
        "settings": {
            "baseline_checkpoint": args.baseline_checkpoint,
            "ours_checkpoint": args.ours_checkpoint,
            "dav2_weights": args.dav2_weights,
            "sintel_root": args.sintel_root,
            "official_mask_source": args.sintel_root / "training" / "occlusions",
            "protocol": "Sun-RAFT eval warm-start, iters=12, mixed_precision=True, bw=False",
            "regions": {region: region for region in REGIONS},
        },
    }
    try:
        if not args.baseline_checkpoint.is_file():
            raise FileNotFoundError(f"Missing baseline checkpoint: {args.baseline_checkpoint}")
        if not args.ours_checkpoint.is_file():
            raise FileNotFoundError(f"Missing ours checkpoint: {args.ours_checkpoint}")
        if not args.dav2_weights.is_file():
            raise FileNotFoundError(f"Missing DAv2 weights: {args.dav2_weights}")
        if not (args.sintel_root / "training" / "occlusions").is_dir():
            raise FileNotFoundError(f"Missing Sintel occlusion masks: {args.sintel_root / 'training' / 'occlusions'}")

        device = _resolve_device()
        patched_get_extention = _patched_get_extention_factory(args.sintel_root)
        original_get_extention = datasets_un.get_extention
        datasets_un.get_extention = patched_get_extention
        try:
            baseline_model = _load_baseline_model(device=device, checkpoint_path=args.baseline_checkpoint)
            baseline_full = _run_model(
                baseline_model,
                sintel_root=args.sintel_root,
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
            ours_full = _run_model(
                ours_model,
                sintel_root=args.sintel_root,
                max_pairs=args.max_pairs,
                log_every=args.log_every,
                device=device,
                label="ours_step35000",
            )
        finally:
            datasets_un.get_extention = original_get_extention

        baseline = {dstype: _compact(baseline_full[dstype]) for dstype in PASSES}
        ours = {dstype: _compact(ours_full[dstype]) for dstype in PASSES}
        payload.update(
            {
                "baseline": baseline,
                "ours_step35000": ours,
                "delta": _make_delta(baseline, ours),
                "details": {
                    "baseline": baseline_full,
                    "ours_step35000": ours_full,
                },
                "expected_all": EXPECTED_ALL,
                "status": "PASSED",
            }
        )
        _write_json(args.output_json, payload)
        print(f"saved_json={args.output_json}")
        print(
            f"baseline clean all/matched/unmatched = {baseline['clean']['all']:.6f} / "
            f"{baseline['clean']['matched']:.6f} / {baseline['clean']['unmatched']:.6f}"
        )
        print(
            f"ours clean all/matched/unmatched = {ours['clean']['all']:.6f} / "
            f"{ours['clean']['matched']:.6f} / {ours['clean']['unmatched']:.6f}"
        )
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



