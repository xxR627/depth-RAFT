#!/usr/bin/env python
"""Evaluate Sun-RAFT on Sintel clean train with the diagnostic A/B/C region split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import datasets_un
from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_ARTIFACT_ROOT = PACKAGE_ROOT / "results" / "main"
DEFAULT_OUTPUT_JSON = DEFAULT_ARTIFACT_ROOT / "sun_raft_baseline_region_decomp_v1.json"
DEFAULT_REPORT_MD = DEFAULT_ARTIFACT_ROOT / "sun_raft_baseline_region_decomp_v1.md"
DEFAULT_REFERENCE_JSON = None
PROTOCOL_NAME = "sun_raft_eval_py_default_warmstart_iters12_mixed_precision_bw_false"
REGIONS = ("A", "B", "C")
COARSEST_SCALE = 8
ITERS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debugging. 0 means full dataset.")
    return parser.parse_args()


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")
    return torch.device("cuda")


def _patched_get_extention_factory(sintel_root: Path):
    base_root = sintel_root.parent.resolve()

    def _get_extention() -> str:
        return str(base_root) + os.sep

    return _get_extention


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _load_model(device: torch.device, checkpoint_path: Path) -> RAFT:
    config = {
        "mixed_precision": True,
        "fnet_norm": "instance",
        "cnet_norm": "batch",
        "bw": False,
    }
    model = RAFT(config).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = _strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def _mesh_grid(batch_size: int, height: int, width: int) -> torch.Tensor:
    x_base = torch.arange(0, width).repeat(batch_size, height, 1)
    y_base = torch.arange(0, height).repeat(batch_size, width, 1).transpose(1, 2)
    return torch.stack([x_base, y_base], 1)


def _norm_grid(v_grid: torch.Tensor, full_padded_img_h: int | None = None, full_padded_img_w: int | None = None):
    if full_padded_img_h is None or full_padded_img_w is None:
        _, _, height, width = v_grid.size()
    else:
        height, width = full_padded_img_h, full_padded_img_w

    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (width - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (height - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)


def _flow_warp(x: torch.Tensor, flow12: torch.Tensor, pad: str = "border", mode: str = "bilinear") -> torch.Tensor:
    batch_size, _, height, width = flow12.size()
    base_grid = _mesh_grid(batch_size, height, width).type_as(x)
    v_grid = _norm_grid(base_grid + flow12, full_padded_img_h=x.shape[2], full_padded_img_w=x.shape[3])
    return torch.nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)


def _mask_out_of_image(coords: torch.Tensor, pad_h: int = 0, pad_w: int = 0, max_height=None, max_width=None):
    pad_h = float(pad_h)
    pad_w = float(pad_w)
    if len(coords.shape) != 4:
        raise NotImplementedError()
    if max_height is None:
        max_height = float(coords.shape[2] - 1)
    if max_width is None:
        max_width = float(coords.shape[3] - 1)

    mask = torch.logical_and(
        torch.logical_and(coords[:, 0, :, :] >= pad_w, coords[:, 0, :, :] <= max_width),
        torch.logical_and(coords[:, 1, :, :] >= pad_h, coords[:, 1, :, :] <= max_height),
    )
    return mask.type(torch.float).unsqueeze(dim=1)


def _position_plus_flow(flow: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = flow.shape
    coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(batch_size, 1, 1, 1)
    return coords.to(flow.device) + flow


def _get_occu_mask_bidirection(flow12_list, flow21_list, scale: float = 0.01, bias: float = 0.5):
    flows12 = torch.cat(flow12_list, dim=0).detach()
    flows21 = torch.cat(flow21_list, dim=0).detach()
    flow21_warped = _flow_warp(flows21, flows12, pad="zeros")
    flow12_diff = flows12 + flow21_warped
    mag = (flows12 * flows12).sum(1, keepdim=True) + (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occs = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    occs = 1 - occs.float()
    corresponding_pixels = _position_plus_flow(flows12)
    out_of_img_mask = _mask_out_of_image(corresponding_pixels)
    occs = occs * out_of_img_mask
    return torch.chunk(occs, len(flow12_list))


def _inverse_splat_flow(flow_fw: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = flow_fw.shape
    base_grid = _mesh_grid(batch_size, height, width).to(flow_fw.device, flow_fw.dtype)
    target = base_grid + flow_fw
    x = target[:, 0].reshape(batch_size, -1)
    y = target[:, 1].reshape(batch_size, -1)
    neg_flow = (-flow_fw).reshape(batch_size, 2, -1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    backward = torch.zeros(batch_size, 2, height * width, device=flow_fw.device, dtype=flow_fw.dtype)
    weights_sum = torch.zeros(batch_size, height * width, device=flow_fw.device, dtype=flow_fw.dtype)

    def scatter(x_idx: torch.Tensor, y_idx: torch.Tensor, weight: torch.Tensor) -> None:
        valid = (x_idx >= 0) & (x_idx <= width - 1) & (y_idx >= 0) & (y_idx <= height - 1)
        flat_index = (y_idx.clamp(0, height - 1) * width + x_idx.clamp(0, width - 1)).long()
        weight = weight * valid.float()
        weights_sum.scatter_add_(1, flat_index, weight)
        for channel in range(2):
            backward[:, channel].scatter_add_(1, flat_index, neg_flow[:, channel] * weight)

    scatter(x0, y0, (x1 - x) * (y1 - y))
    scatter(x1, y0, (x - x0) * (y1 - y))
    scatter(x0, y1, (x1 - x) * (y - y0))
    scatter(x1, y1, (x - x0) * (y - y0))
    backward = backward / weights_sum.clamp_min(1e-6).unsqueeze(1)
    return backward.view(batch_size, 2, height, width)


def _flow_edge_band_mask(flow: torch.Tensor, threshold: float, radius: int) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=flow.device,
        dtype=flow.dtype,
    ).unsqueeze(1) / 8.0
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=flow.device,
        dtype=flow.dtype,
    ).unsqueeze(1) / 8.0
    kernel_x = kernel_x.repeat(2, 1, 1, 1)
    kernel_y = kernel_y.repeat(2, 1, 1, 1)
    grad_x = F.conv2d(flow, kernel_x, padding=1, groups=2)
    grad_y = F.conv2d(flow, kernel_y, padding=1, groups=2)
    grad_mag = torch.sqrt((grad_x ** 2 + grad_y ** 2).sum(dim=1, keepdim=True))
    edge_mask = grad_mag >= threshold
    if radius > 0:
        kernel_size = 2 * radius + 1
        edge_mask = F.max_pool2d(edge_mask.float(), kernel_size=kernel_size, stride=1, padding=radius) > 0
    return edge_mask


def _read_occlusion_mask(occlusion_path: Path) -> torch.Tensor:
    mask = cv2.imread(str(occlusion_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read Sintel occlusion mask: {occlusion_path}")
    return torch.from_numpy(mask > 0).unsqueeze(0).unsqueeze(0)


def _make_region_masks(*, flow_gt: torch.Tensor, occlusion_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    flow_bw = _inverse_splat_flow(flow_gt)
    fb_consistency_mask = _get_occu_mask_bidirection([flow_gt], [flow_bw])[0] > 0.5
    edge_mask = _flow_edge_band_mask(flow_gt, threshold=1.0, radius=2)
    return {
        "A": fb_consistency_mask & (~occlusion_mask),
        "B": occlusion_mask | (~fb_consistency_mask),
        "C": edge_mask,
    }


def _occlusion_path(sintel_root: Path, sequence: str, frame_zero_based: int) -> Path:
    return sintel_root / "training" / "occlusions" / sequence / f"frame_{frame_zero_based + 1:04d}.png"


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "epe_min": float("nan"),
            "epe_max": float("nan"),
            "epe_median": float("nan"),
            "epe_p25": float("nan"),
            "epe_p75": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "epe_min": float(np.min(arr)),
        "epe_max": float(np.max(arr)),
        "epe_median": float(np.median(arr)),
        "epe_p25": float(np.percentile(arr, 25)),
        "epe_p75": float(np.percentile(arr, 75)),
    }


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")


def _load_reference_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any], reference_payload: dict[str, Any] | None) -> None:
    lines: list[str] = []
    lines.append("# Sun-RAFT A/B/C Region Decomposition")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Protocol: `{payload['protocol']}`")
    lines.append(f"- Dataset: `{payload['dataset']}`")
    lines.append(f"- Checkpoint: `{payload['settings']['checkpoint']}`")
    lines.append("- A/B/C is a retained diagnostic split used for qualitative sample selection.")
    lines.append("")
    lines.append("## Sun-RAFT Metrics")
    lines.append("")
    lines.append("| Region | Weighted mean EPE | Pixel share |")
    lines.append("| --- | ---: | ---: |")
    for region in REGIONS:
        region_payload = payload[f"region_{region}"]
        lines.append(
            f"| {region} | {region_payload['epe_weighted_mean']:.4f} | {region_payload['pct_of_total_pixels']:.2f}% |"
        )
    lines.append(f"| Overall | {payload['overall_epe']:.4f} | 100.00% |")
    lines.append("")

    if reference_payload is not None:
        lines.append("## Comparison With External Reference")
        lines.append("")
        lines.append("| Region | Reference | Sun-RAFT | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for region in REGIONS:
            reference_epe = reference_payload[f"region_{region}"]["epe_weighted_mean"]
            sun_epe = payload[f"region_{region}"]["epe_weighted_mean"]
            lines.append(f"| {region} | {reference_epe:.4f} | {sun_epe:.4f} | {sun_epe - reference_epe:+.4f} |")
        lines.append(f"| Overall | {reference_payload['overall_epe']:.4f} | {payload['overall_epe']:.4f} | {payload['overall_epe'] - reference_payload['overall_epe']:+.4f} |")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"Sun-RAFT baseline: A EPE = {payload['region_A']['epe_weighted_mean']:.4f}, "
        f"B EPE = {payload['region_B']['epe_weighted_mean']:.4f}, "
        f"C EPE = {payload['region_C']['epe_weighted_mean']:.4f}, "
        f"B-region pixel share = {payload['region_B']['pct_of_total_pixels']:.2f}%."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()
    args.report_md = args.report_md.expanduser().resolve()
    if args.reference_json is not None:
        args.reference_json = args.reference_json.expanduser().resolve()

    payload: dict[str, Any] = {
        "status": "FAILED",
        "protocol": PROTOCOL_NAME,
        "dataset": None,
        "A_B_C_definition_source": "legacy diagnostic split retained for figure selection",
        "distribution_basis": "per_pair_region_mean_epe",
        "settings": {
            "checkpoint": args.checkpoint,
            "sintel_root": args.sintel_root,
            "iters": ITERS,
            "warm_start": True,
            "mixed_precision": True,
            "bw": False,
            "coarsest_scale": COARSEST_SCALE,
        },
    }

    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        if not (args.sintel_root / "training" / "clean").is_dir():
            raise FileNotFoundError(f"Invalid Sintel root: {args.sintel_root}")

        device = _resolve_device()
        model = _load_model(device=device, checkpoint_path=args.checkpoint)

        patched_get_extention = _patched_get_extention_factory(args.sintel_root)
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
            payload["dataset"] = f"sintel_clean_train_{total_pairs}_pairs"

            region_sum_epe = {region: 0.0 for region in REGIONS}
            region_total_pixels = {region: 0 for region in REGIONS}
            region_pair_means = {region: [] for region in REGIONS}
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
                    flow = padder.unpad(flow_pr[0]).cpu()
                    flow_prev = forward_interpolate(flow_low[0])[None].to(device)

                    epe_map = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
                    overall_sum_epe += float(epe_map.sum().item())
                    overall_total_pixels += int(epe_map.numel())

                    occlusion_mask = _read_occlusion_mask(_occlusion_path(args.sintel_root, sequence, frame)).bool()
                    region_masks = _make_region_masks(
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
                        print(f"[eval] pair={val_id + 1}/{total_pairs} scene={sequence} frame={frame + 1:04d}")
        finally:
            datasets_un.get_extention = original_get_extention

        overall_epe = overall_sum_epe / max(overall_total_pixels, 1)
        results: dict[str, Any] = {
            "status": "PASSED",
            "protocol": PROTOCOL_NAME,
            "dataset": payload["dataset"],
            "A_B_C_definition_source": payload["A_B_C_definition_source"],
            "distribution_basis": payload["distribution_basis"],
            "settings": payload["settings"],
            "overall_epe": overall_epe,
            "overall_total_pixels": overall_total_pixels,
        }
        for region in REGIONS:
            total_pixels = region_total_pixels[region]
            region_payload = {
                "epe_weighted_mean": region_sum_epe[region] / max(total_pixels, 1),
                **_percentiles(region_pair_means[region]),
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

        reference_payload = _load_reference_json(args.reference_json)
        if reference_payload is not None:
            results["comparison_with_external_reference"] = {
                "overall_delta": results["overall_epe"] - reference_payload["overall_epe"],
                "region_A_delta": results["region_A"]["epe_weighted_mean"] - reference_payload["region_A"]["epe_weighted_mean"],
                "region_B_delta": results["region_B"]["epe_weighted_mean"] - reference_payload["region_B"]["epe_weighted_mean"],
                "region_C_delta": results["region_C"]["epe_weighted_mean"] - reference_payload["region_C"]["epe_weighted_mean"],
            }

        _write_json(args.output_json, results)
        _write_report(args.report_md, results, reference_payload)
        print(f"saved_json={args.output_json}")
        print(f"saved_report={args.report_md}")
        print(f"overall_epe={results['overall_epe']:.6f}")
        print(f"region_A_epe={results['region_A']['epe_weighted_mean']:.6f}")
        print(f"region_B_epe={results['region_B']['epe_weighted_mean']:.6f}")
        print(f"region_C_epe={results['region_C']['epe_weighted_mean']:.6f}")
        return 0
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_json, payload)
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

