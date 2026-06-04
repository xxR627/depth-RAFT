#!/usr/bin/env python
"""Evaluate a Depth-RAFT G+Z+DAB trainable checkpoint on Sintel clean/final."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
import evaluate

import eval_depth_raft_region_decomp as fusion_region_eval
import eval_sun_raft_region_decomp as baseline_eval


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_BASELINE_FINAL = {"clean_epe": 1.694220781326294, "final_epe": 2.5935211181640625}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--trainable-checkpoint", type=Path, required=True)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dav2-input-scale", type=float, default=1.0)
    parser.add_argument("--sintel-passes", nargs="+", choices=("clean", "final"), default=["clean", "final"])
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _evaluate_clean_final(model, passes: list[str]) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    clean_final: dict[str, float] = {}

    with torch.no_grad():
        for dstype in passes:
            val_dataset = datasets_un.MpiSintel(
                split="training",
                dstype=dstype,
                show_extra_info=True,
                read_flow_gt=True,
            )
            total_pixels = 0
            sum_epe = 0.0
            flow_prev, sequence_prev = None, None

            for val_id in evaluate.tqdm(range(len(val_dataset))):
                image1, image2, flow_gt, _, (sequence, _frame) = val_dataset[val_id]
                padder = evaluate.InputPadder(image1.shape, coarsest_scale=8)
                image1, image2 = padder.pad(image1[None].to(device), image2[None].to(device))

                if sequence != sequence_prev:
                    flow_prev = None

                flow_low, flow_pr = model(image1, image2, iters=12, flow_init=flow_prev, test_mode=True)
                flow = padder.unpad(flow_pr[0]).cpu()
                flow_prev = evaluate.forward_interpolate(flow_low[0])[None].to(device)

                epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt().view(-1)
                total_pixels += int(epe.numel())
                sum_epe += float(epe.sum().item())
                sequence_prev = sequence

            clean_final[f"{dstype}_epe"] = sum_epe / max(total_pixels, 1)

    return clean_final


def _resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.trainable_checkpoint = args.trainable_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()

    patched_get_extention = baseline_eval._patched_get_extention_factory(args.sintel_root)
    original_datasets_get_extention = datasets_un.get_extention
    original_evaluate_get_extention = evaluate.datasets_un.get_extention
    datasets_un.get_extention = patched_get_extention
    evaluate.datasets_un.get_extention = patched_get_extention

    try:
        model = fusion_region_eval._build_model(
            device=_resolve_device(args.device),
            baseline_checkpoint=args.baseline_checkpoint,
            fusion_checkpoint=args.trainable_checkpoint,
            dav2_weights=args.dav2_weights,
            dav2_input_scale=args.dav2_input_scale,
        )
        clean_final = _evaluate_clean_final(model, args.sintel_passes)
    finally:
        datasets_un.get_extention = original_datasets_get_extention
        evaluate.datasets_un.get_extention = original_evaluate_get_extention

    payload = {
        "checkpoint": str(args.trainable_checkpoint),
        "dav2_input_scale": args.dav2_input_scale,
        "sintel_passes": args.sintel_passes,
        "clean_final": clean_final,
        "delta_vs_baseline": {
            key: value - DEFAULT_BASELINE_FINAL[key]
            for key, value in clean_final.items()
            if key in DEFAULT_BASELINE_FINAL
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



