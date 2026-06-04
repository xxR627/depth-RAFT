#!/usr/bin/env python
"""Evaluate a Depth-RAFT G+Z+DAB checkpoint on Sintel clean train with region metrics."""

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
from raft import RAFT, adapt_state_dict_for_cnet_depth_input, allowed_missing_checkpoint_keys
from utils.utils import InputPadder, forward_interpolate

import eval_sun_raft_region_decomp as baseline_eval


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_FUSION_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_BASELINE_REGION_JSON = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--fusion-checkpoint", type=Path, default=DEFAULT_FUSION_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--baseline-region-json", type=Path, default=DEFAULT_BASELINE_REGION_JSON)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def _build_model(
    device: torch.device,
    baseline_checkpoint: Path,
    fusion_checkpoint: Path,
    dav2_weights: Path,
    *,
    dav2_input_scale: float = 1.0,
) -> RAFT:
    config = {
        "mixed_precision": True,
        "fnet_norm": "instance",
        "cnet_norm": "batch",
        "bw": False,
    }
    fusion_payload = torch.load(str(fusion_checkpoint), map_location="cpu", weights_only=False)
    model = RAFT(
        config,
        use_depth_raft=True,
        dav2_weights_path=str(dav2_weights),
        use_gru_checkpointing=True,
        dav2_input_scale=dav2_input_scale,
    ).to(device)

    state_dict = torch.load(str(baseline_checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported baseline checkpoint format: {baseline_checkpoint}")
    state_dict = baseline_eval._strip_module_prefix(state_dict)
    state_dict = adapt_state_dict_for_cnet_depth_input(state_dict, model.state_dict())
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = allowed_missing_checkpoint_keys(model)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("dav2_fusion.") and key not in allowed_missing
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise RuntimeError(
            f"Baseline checkpoint mismatch: unexpected={incompatible.unexpected_keys}, invalid_missing={invalid_missing}"
        )

    if "trainable_state" in fusion_payload:
        trainable_state = fusion_payload["trainable_state"]
    elif "dav2_fusion_state" in fusion_payload:
        trainable_state = {f"dav2_fusion.{key}": value for key, value in fusion_payload["dav2_fusion_state"].items()}
    else:
        raise RuntimeError(f"Unsupported fusion checkpoint format: {fusion_checkpoint}")

    model_state = model.state_dict()
    missing = []
    for key, value in trainable_state.items():
        if key not in model_state:
            missing.append(key)
            continue
        model_state[key] = value
    if missing:
        raise RuntimeError(f"Trainable state keys not present in model: {missing}")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


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


def _write_report(path: Path, payload: dict[str, Any], baseline_payload: dict[str, Any] | None) -> None:
    lines: list[str] = []
    lines.append("# Depth-RAFT Region Decomposition")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Fusion checkpoint: `{payload['settings']['fusion_checkpoint']}`")
    lines.append(f"- Baseline checkpoint: `{payload['settings']['baseline_checkpoint']}`")
    lines.append(f"- Eval protocol: `{payload['protocol']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Region | Weighted mean EPE | Pixel share |")
    lines.append("| --- | ---: | ---: |")
    for region in baseline_eval.REGIONS:
        region_payload = payload[f"region_{region}"]
        lines.append(
            f"| {region} | {region_payload['epe_weighted_mean']:.4f} | {region_payload['pct_of_total_pixels']:.2f}% |"
        )
    lines.append(f"| Overall | {payload['overall_epe']:.4f} | 100.00% |")
    lines.append("")
    if baseline_payload is not None:
        lines.append("## Delta Vs Sun-RAFT Baseline")
        lines.append("")
        lines.append("| Region | Baseline | Fusion | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for region in baseline_eval.REGIONS:
            baseline_value = baseline_payload[f"region_{region}"]["epe_weighted_mean"]
            fused_value = payload[f"region_{region}"]["epe_weighted_mean"]
            lines.append(f"| {region} | {baseline_value:.4f} | {fused_value:.4f} | {fused_value - baseline_value:+.4f} |")
        lines.append(
            f"| Overall | {baseline_payload['overall_epe']:.4f} | {payload['overall_epe']:.4f} | {payload['overall_epe'] - baseline_payload['overall_epe']:+.4f} |"
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.fusion_checkpoint = args.fusion_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    if args.baseline_region_json is not None:
        args.baseline_region_json = args.baseline_region_json.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()
    if args.report_md is None:
        args.report_md = args.output_json.with_suffix(".md")
    else:
        args.report_md = args.report_md.expanduser().resolve()

    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")
    if not args.fusion_checkpoint.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {args.fusion_checkpoint}")
    if not args.dav2_weights.is_file():
        raise FileNotFoundError(f"DAv2 weights not found: {args.dav2_weights}")
    if not (args.sintel_root / "training" / "clean").is_dir():
        raise FileNotFoundError(f"Invalid Sintel root: {args.sintel_root}")

    device = baseline_eval._resolve_device()
    model = _build_model(
        device=device,
        baseline_checkpoint=args.baseline_checkpoint,
        fusion_checkpoint=args.fusion_checkpoint,
        dav2_weights=args.dav2_weights,
    )

    patched_get_extention = baseline_eval._patched_get_extention_factory(args.sintel_root)
    original_get_extention = datasets_un.get_extention
    datasets_un.get_extention = patched_get_extention
    try:
        val_dataset = datasets_un.MpiSintel(
            split="training",
            dstype="clean",
            show_extra_info=True,
            read_flow_gt=True,
        )
        total_pairs = len(val_dataset) if args.max_pairs <= 0 else min(len(val_dataset), args.max_pairs)
        region_sum_epe = {region: 0.0 for region in baseline_eval.REGIONS}
        region_total_pixels = {region: 0 for region in baseline_eval.REGIONS}
        region_pair_means = {region: [] for region in baseline_eval.REGIONS}
        overall_sum_epe = 0.0
        overall_total_pixels = 0
        flow_prev = None
        sequence_prev = None

        with torch.no_grad():
            for val_id in range(total_pairs):
                image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
                image1 = image1[None].to(device)
                image2 = image2[None].to(device)

                if sequence != sequence_prev:
                    flow_prev = None

                padder = InputPadder(image1.shape, coarsest_scale=baseline_eval.COARSEST_SCALE)
                image1_pad, image2_pad = padder.pad(image1, image2)
                flow_low, flow_pr = model(
                    image1_pad,
                    image2_pad,
                    iters=baseline_eval.ITERS,
                    flow_init=flow_prev,
                    test_mode=True,
                    bw=False,
                )
                flow = padder.unpad(flow_pr[0]).cpu()
                flow_prev = forward_interpolate(flow_low[0])[None].to(device)

                epe_map = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
                overall_sum_epe += float(epe_map.sum().item())
                overall_total_pixels += int(epe_map.numel())

                occlusion_mask = baseline_eval._read_occlusion_mask(
                    baseline_eval._occlusion_path(args.sintel_root, sequence, frame)
                ).bool()
                region_masks = baseline_eval._make_region_masks(
                    flow_gt=flow_gt.unsqueeze(0),
                    occlusion_mask=occlusion_mask,
                )
                for region_name, region_mask in region_masks.items():
                    region_mask_2d = region_mask.squeeze(0).squeeze(0)
                    num_pixels = int(region_mask_2d.sum().item())
                    if num_pixels == 0:
                        continue
                    region_epe = epe_map[region_mask_2d]
                    region_sum_epe[region_name] += float(region_epe.sum().item())
                    region_total_pixels[region_name] += num_pixels
                    region_pair_means[region_name].append(float(region_epe.mean().item()))

                sequence_prev = sequence
                if ((val_id + 1) % args.log_every == 0) or (val_id + 1 == total_pairs):
                    print(f"[fusion-region-eval] pair={val_id + 1}/{total_pairs} scene={sequence} frame={frame + 1:04d}")
    finally:
        datasets_un.get_extention = original_get_extention

    overall_epe = overall_sum_epe / max(overall_total_pixels, 1)
    results: dict[str, Any] = {
        "status": "PASSED",
        "protocol": baseline_eval.PROTOCOL_NAME,
        "dataset": f"sintel_clean_train_{total_pairs}_pairs",
        "A_B_C_definition_source": "legacy diagnostic split retained for figure selection",
        "distribution_basis": "per_pair_region_mean_epe",
        "settings": {
            "baseline_checkpoint": args.baseline_checkpoint,
            "fusion_checkpoint": args.fusion_checkpoint,
            "dav2_weights": args.dav2_weights,
            "sintel_root": args.sintel_root,
            "iters": baseline_eval.ITERS,
            "warm_start": True,
            "mixed_precision": True,
            "bw": False,
            "coarsest_scale": baseline_eval.COARSEST_SCALE,
        },
        "overall_epe": overall_epe,
        "overall_total_pixels": overall_total_pixels,
    }

    for region in baseline_eval.REGIONS:
        total_pixels = region_total_pixels[region]
        region_payload = {
            "epe_weighted_mean": region_sum_epe[region] / max(total_pixels, 1),
            **baseline_eval._percentiles(region_pair_means[region]),
            "total_pixels": total_pixels,
            "pct_of_total_pixels": 100.0 * total_pixels / max(overall_total_pixels, 1),
            "num_pairs_with_region_pixels": len(region_pair_means[region]),
        }
        results[f"region_{region}"] = region_payload

    if results["region_B"]["total_pixels"] > 0:
        results["region_B"]["epe_contribution_pct_of_overall"] = (
            100.0
            * results["region_B"]["epe_weighted_mean"]
            * results["region_B"]["total_pixels"]
            / max(overall_epe * overall_total_pixels, 1e-8)
        )

    results["sanity_checks"] = {
        "a_plus_b_equals_total_pixels": (
            results["region_A"]["total_pixels"] + results["region_B"]["total_pixels"] == overall_total_pixels
        ),
        "a_plus_b_total_pixels": results["region_A"]["total_pixels"] + results["region_B"]["total_pixels"],
        "a_plus_b_plus_c_total_pixels": (
            results["region_A"]["total_pixels"]
            + results["region_B"]["total_pixels"]
            + results["region_C"]["total_pixels"]
        ),
        "note": "A and B partition the full image; C overlaps with A/B by definition.",
    }

    baseline_payload = None
    if args.baseline_region_json is not None and args.baseline_region_json.is_file():
        baseline_payload = json.loads(args.baseline_region_json.read_text(encoding="utf-8"))
        results["comparison_with_sun_baseline"] = {
            "overall_delta": results["overall_epe"] - baseline_payload["overall_epe"],
            "region_A_delta": results["region_A"]["epe_weighted_mean"] - baseline_payload["region_A"]["epe_weighted_mean"],
            "region_B_delta": results["region_B"]["epe_weighted_mean"] - baseline_payload["region_B"]["epe_weighted_mean"],
            "region_C_delta": results["region_C"]["epe_weighted_mean"] - baseline_payload["region_C"]["epe_weighted_mean"],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_to_builtin(results), indent=2) + "\n", encoding="utf-8")
    _write_report(args.report_md, results, baseline_payload)
    print(f"saved_json={args.output_json}")
    print(f"saved_report={args.report_md}")
    print(f"overall_epe={results['overall_epe']:.6f}")
    print(f"region_A_epe={results['region_A']['epe_weighted_mean']:.6f}")
    print(f"region_B_epe={results['region_B']['epe_weighted_mean']:.6f}")
    print(f"region_C_epe={results['region_C']['epe_weighted_mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

