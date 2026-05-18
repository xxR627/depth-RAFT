"""Utilities for official Sintel matched/unmatched evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PASSES = ("clean", "final")
REGIONS = ("all", "official_valid", "matched_noc", "unmatched_occ")


def _official_region_masks(
    *,
    sintel_root: Path,
    sequence: str,
    frame_zero_based: int,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    occ_path = sintel_root / "training" / "occlusions" / sequence / f"frame_{frame_zero_based + 1:04d}.png"
    invalid_path = sintel_root / "training" / "invalid" / sequence / f"frame_{frame_zero_based + 1:04d}.png"
    if not occ_path.is_file():
        raise FileNotFoundError(f"Sintel occlusion mask not found: {occ_path}")
    if not invalid_path.is_file():
        raise FileNotFoundError(f"Sintel invalid mask not found: {invalid_path}")

    occ = cv2.imread(str(occ_path), cv2.IMREAD_UNCHANGED)
    invalid = cv2.imread(str(invalid_path), cv2.IMREAD_UNCHANGED)
    if occ is None or invalid is None:
        raise RuntimeError(f"Failed to read Sintel masks: {occ_path}, {invalid_path}")
    if occ.shape[:2] != (height, width) or invalid.shape[:2] != (height, width):
        raise RuntimeError(
            f"Sintel mask shape mismatch for {sequence} frame {frame_zero_based}: "
            f"occ={occ.shape}, invalid={invalid.shape}, expected={(height, width)}"
        )

    occ_mask = torch.from_numpy(occ > 0)
    valid = torch.from_numpy(invalid == 0)
    masks = {
        "all": torch.ones_like(valid, dtype=torch.bool),
        "official_valid": valid,
        "matched_noc": valid & (~occ_mask),
        "unmatched_occ": valid & occ_mask,
    }
    return {name: mask[None, None] for name, mask in masks.items()}


def _init_accumulator() -> dict[str, Any]:
    return {
        "sum_epe": {region: 0.0 for region in REGIONS},
        "total_pixels": {region: 0 for region in REGIONS},
        "pair_means": {region: [] for region in REGIONS},
        "total_pairs": 0,
    }


def _update_metrics(
    accumulator: dict[str, Any],
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> None:
    epe_map = torch.sum((flow_pred - flow_gt) ** 2, dim=0).sqrt()
    for region in REGIONS:
        mask = masks[region].squeeze(0).squeeze(0)
        num_pixels = int(mask.sum().item())
        if num_pixels == 0:
            continue
        values = epe_map[mask]
        accumulator["sum_epe"][region] += float(values.sum().item())
        accumulator["total_pixels"][region] += num_pixels
        accumulator["pair_means"][region].append(float(values.mean().item()))
    accumulator["total_pairs"] += 1


def _finalize_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"num_pairs": accumulator["total_pairs"]}
    for region in REGIONS:
        total_pixels = accumulator["total_pixels"][region]
        pair_means = np.asarray(accumulator["pair_means"][region], dtype=np.float64)
        output[region] = {
            "epe": accumulator["sum_epe"][region] / max(total_pixels, 1),
            "total_pixels": total_pixels,
            "num_pairs_with_pixels": int(pair_means.size),
            "pair_mean_epe": float(pair_means.mean()) if pair_means.size else float("nan"),
            "pair_median_epe": float(np.median(pair_means)) if pair_means.size else float("nan"),
        }
    if output["all"]["total_pixels"] > 0:
        for region in REGIONS:
            output[region]["pct_of_all_pixels"] = (
                100.0 * output[region]["total_pixels"] / output["all"]["total_pixels"]
            )
    if output["official_valid"]["total_pixels"] > 0:
        for region in REGIONS:
            output[region]["pct_of_official_valid_pixels"] = (
                100.0 * output[region]["total_pixels"] / output["official_valid"]["total_pixels"]
            )
    return output

