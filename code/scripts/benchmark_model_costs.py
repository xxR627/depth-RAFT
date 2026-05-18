#!/usr/bin/env python
"""Benchmark parameter count, inference latency, and memory for paper reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
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

from utils import frame_utils
from utils.utils import InputPadder

import eval_sun_raft_region_decomp as baseline_eval
import eval_depth_raft_region_decomp as fusion_eval


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2 = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL = Path(r"G:\flow_data\sintel")
DEFAULT_OUTPUT = PACKAGE_ROOT / "results" / "main" / "model_costs_benchmark.json"
DEFAULT_REPORT = PACKAGE_ROOT / "results" / "main" / "model_costs_benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=30)
    parser.add_argument("--iters", type=int, default=12)
    return parser.parse_args()


def _load_pair(sintel_root: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image1_path = sintel_root / "training" / "clean" / "alley_1" / "frame_0001.png"
    image2_path = sintel_root / "training" / "clean" / "alley_1" / "frame_0002.png"
    image1 = np.array(frame_utils.read_gen(str(image1_path))).astype(np.uint8)
    image2 = np.array(frame_utils.read_gen(str(image2_path))).astype(np.uint8)
    image1_t = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to(device)
    image2_t = torch.from_numpy(image2).permute(2, 0, 1).float()[None].to(device)
    return image1_t, image2_t


def _hidden_dav2_numel(model: torch.nn.Module) -> int:
    extractor = getattr(model, "dav2_extractor", None)
    if extractor is None or not hasattr(extractor, "_dav2"):
        return 0
    return sum(param.numel() for param in extractor._dav2.parameters())


def _count_params(model: torch.nn.Module) -> dict[str, int]:
    registered_total = sum(param.numel() for param in model.parameters())
    hidden_dav2 = _hidden_dav2_numel(model)
    fusion = sum(param.numel() for name, param in model.named_parameters() if name.startswith("dav2_fusion."))
    cnet = sum(param.numel() for name, param in model.named_parameters() if name.startswith("cnet."))
    fnet = sum(param.numel() for name, param in model.named_parameters() if name.startswith("fnet."))
    update = sum(param.numel() for name, param in model.named_parameters() if name.startswith("update_block."))
    flow_head = sum(param.numel() for name, param in model.named_parameters() if "flow_head" in name)
    return {
        "registered_total": registered_total,
        "hidden_dav2_frozen": hidden_dav2,
        "total_including_hidden_dav2": registered_total + hidden_dav2,
        "dav2_fusion": fusion,
        "cnet": cnet,
        "fnet": fnet,
        "update_block": update,
        "flow_head": flow_head,
        "trainable_during_ours_finetune": fusion + cnet,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _benchmark_forward(
    model: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    *,
    iters: int,
    warmup: int,
    measure: int,
) -> dict[str, Any]:
    device = image1.device
    padder = InputPadder(image1.shape, coarsest_scale=baseline_eval.COARSEST_SCALE)
    image1_pad, image2_pad = padder.pad(image1, image2)

    def one_forward() -> None:
        flow_low, flow_pr = model(image1_pad, image2_pad, iters=iters, test_mode=True, bw=False)
        _ = padder.unpad(flow_pr)
        del flow_low, flow_pr

    for _ in range(warmup):
        one_forward()
    _sync(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms: list[float] = []
    for _ in range(measure):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            one_forward()
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(float(start.elapsed_time(end)))
        else:
            import time

            start_time = time.perf_counter()
            one_forward()
            timings_ms.append((time.perf_counter() - start_time) * 1000.0)

    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "mean_ms": statistics.fmean(timings_ms),
        "std_ms": statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "peak_memory_gb": peak_mem_gb,
        "warmup": warmup,
        "measure": measure,
        "iters": iters,
    }


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["baseline"]
    ours = payload["ours"]
    lines = [
        "# Model Cost Benchmark",
        "",
        "| Method | Registered Params (M) | Frozen DAv2 Params (M) | Total Params (M) | Trainable Params in Reported Training (M) | Time / Pair (ms) | Peak Mem (GB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, item in (("Sun-RAFT baseline", baseline), ("Ours (+G+Z)", ours)):
        params = item["params"]
        bench = item["benchmark"]
        lines.append(
            f"| {label} | {params['registered_total'] / 1e6:.2f} | "
            f"{params['hidden_dav2_frozen'] / 1e6:.2f} | "
            f"{params['total_including_hidden_dav2'] / 1e6:.2f} | "
            f"{params['trainable_reported'] / 1e6:.2f} | "
            f"{bench['mean_ms']:.1f} +/- {bench['std_ms']:.1f} | "
            f"{bench['peak_memory_gb']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Inference is measured on one Sintel pair (`alley_1/frame_0001-0002`) with the same 12-iteration eval setting used elsewhere.",
            "- DAv2 is frozen and hidden from `state_dict`; it is counted separately as frozen runtime capacity.",
            "- Baseline trainable count assumes standard full-model Sun-RAFT training.",
            "- Our trainable count is the actual fine-tuning scope: `cnet + dav2_fusion`; fnet, update block, flow head, and DAv2 are frozen.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stable benchmark numbers.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    image1, image2 = _load_pair(args.sintel_root, device)
    payload: dict[str, Any] = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "settings": {
            "baseline_checkpoint": args.baseline_checkpoint,
            "ours_checkpoint": args.ours_checkpoint,
            "dav2_weights": args.dav2_weights,
            "iters": args.iters,
            "image_shape": list(image1.shape),
        },
    }

    baseline_model = baseline_eval._load_model(device, args.baseline_checkpoint)
    baseline_params = _count_params(baseline_model)
    baseline_params["trainable_reported"] = baseline_params["registered_total"]
    payload["baseline"] = {
        "params": baseline_params,
        "benchmark": _benchmark_forward(
            baseline_model,
            image1,
            image2,
            iters=args.iters,
            warmup=args.warmup,
            measure=args.measure,
        ),
    }
    del baseline_model
    torch.cuda.empty_cache()

    ours_model = fusion_eval._build_model(device, args.baseline_checkpoint, args.ours_checkpoint, args.dav2_weights)
    ours_params = _count_params(ours_model)
    ours_params["trainable_reported"] = ours_params["trainable_during_ours_finetune"]
    payload["ours"] = {
        "params": ours_params,
        "benchmark": _benchmark_forward(
            ours_model,
            image1,
            image2,
            iters=args.iters,
            warmup=args.warmup,
            measure=args.measure,
        ),
    }

    payload["delta"] = {
        "time_ms": payload["ours"]["benchmark"]["mean_ms"] - payload["baseline"]["benchmark"]["mean_ms"],
        "time_pct": 100.0
        * (payload["ours"]["benchmark"]["mean_ms"] - payload["baseline"]["benchmark"]["mean_ms"])
        / payload["baseline"]["benchmark"]["mean_ms"],
        "peak_memory_gb": payload["ours"]["benchmark"]["peak_memory_gb"]
        - payload["baseline"]["benchmark"]["peak_memory_gb"],
        "total_params_m": (
            payload["ours"]["params"]["total_including_hidden_dav2"]
            - payload["baseline"]["params"]["total_including_hidden_dav2"]
        )
        / 1e6,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    _write_report(args.report_md, payload)
    print(json.dumps(_to_builtin(payload["delta"]), indent=2), flush=True)
    print(f"saved_json={args.output_json}", flush=True)
    print(f"saved_report={args.report_md}", flush=True)


if __name__ == "__main__":
    main()



