#!/usr/bin/env python
"""Rebuild paper Figure 1 as a self-contained revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from paper_figure_selfcontained_utils import draw_box, load_rgb, save_pub, style_image_ax


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT
DEFAULT_IMAGE = Path(r"G:\flow_data\sintel\training\clean\temple_2\frame_0025.png")
DEFAULT_OCC = Path(r"G:\flow_data\sintel\training\occlusions\temple_2\frame_0025.png")
DEFAULT_INVALID = Path(r"G:\flow_data\sintel\training\invalid\temple_2\frame_0025.png")
DEFAULT_REGION_JSON = PACKAGE_ROOT / "results" / "main" / "sintel_occ_noc_eval.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "figures" / "revision" / "paper_fig1_comment8"

ROI_BOX = (0.07, 0.18, 0.40, 0.56)
COLORS = {
    "matched": "#aab3bf",
    "unmatched": "#e53935",
    "baseline": "#5a6470",
    "ours": "#1f9d8f",
    "oracle": "#6d4bd2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--occ-mask", type=Path, default=DEFAULT_OCC)
    parser.add_argument("--invalid-mask", type=Path, default=DEFAULT_INVALID)
    parser.add_argument("--region-json", type=Path, default=DEFAULT_REGION_JSON)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def overlay_unmatched(image: np.ndarray, mask: np.ndarray, alpha: float = 0.34) -> np.ndarray:
    output = image.astype(np.float32).copy()
    red = np.zeros_like(output)
    red[:, :, 0] = 255.0
    output[mask] = (1 - alpha) * output[mask] + alpha * red[mask]
    return np.clip(output, 0, 255).astype(np.uint8)


def load_stats(region_json: Path) -> dict[str, float]:
    payload = json.loads(region_json.read_text(encoding="utf-8"))
    baseline_clean = float(payload["baseline"]["clean"]["all"])
    ours_clean = float(payload["ours_step35000"]["clean"]["all"])
    expected_baseline = float(payload.get("expected_all", {}).get("baseline", {}).get("clean", baseline_clean))
    expected_ours = float(payload.get("expected_all", {}).get("ours_step35000", {}).get("clean", ours_clean))
    matched_epe = float(payload["baseline"]["clean"]["matched"])
    unmatched_epe = float(payload["baseline"]["clean"]["unmatched"])
    matched_pixels = float(payload["details"]["baseline"]["clean"]["matched_noc"]["total_pixels"])
    unmatched_pixels = float(payload["details"]["baseline"]["clean"]["unmatched_occ"]["total_pixels"])
    unmatched_pct = 100.0 * unmatched_pixels / (matched_pixels + unmatched_pixels)
    unmatched_error_share = 100.0 * (
        unmatched_epe * unmatched_pixels
    ) / (matched_epe * matched_pixels + unmatched_epe * unmatched_pixels)
    oracle_clean = 0.77
    return {
        "baseline_clean": baseline_clean,
        "ours_clean": ours_clean,
        "display_baseline_clean": expected_baseline,
        "display_ours_clean": expected_ours,
        "matched_epe": matched_epe,
        "unmatched_epe": unmatched_epe,
        "matched_pct": 100.0 - unmatched_pct,
        "unmatched_pct": unmatched_pct,
        "matched_error_share": 100.0 - unmatched_error_share,
        "unmatched_error_share": unmatched_error_share,
        "oracle_clean": oracle_clean,
        "ratio": unmatched_epe / matched_epe,
    }


def main() -> int:
    args = parse_args()
    image = load_rgb(args.image)
    occ = read_mask(args.occ_mask)
    invalid = read_mask(args.invalid_mask)
    official_unmatched = occ & (~invalid)
    overlay = overlay_unmatched(image, official_unmatched)
    stats = load_stats(args.region_json)

    fig = plt.figure(figsize=(13.2, 4.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1.05, 1.0], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(overlay)
    draw_box(ax0, overlay.shape[:2], ROI_BOX, linewidth=2.2)
    ax0.set_title("(a) Official unmatched pixels cluster around boundaries", loc="left", fontweight="bold", pad=6)
    ax0.text(
        0.03,
        0.97,
        "temple_2 / frame 0025",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.58, "pad": 2, "edgecolor": "none"},
    )
    ax0.text(
        0.03,
        0.06,
        "Red = official unmatched / occluded",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": COLORS["unmatched"], "alpha": 0.80, "pad": 2, "edgecolor": "none"},
    )
    inset = ax0.inset_axes([0.62, 0.06, 0.34, 0.34])
    h, w = overlay.shape[:2]
    x0, y0, x1, y1 = ROI_BOX
    crop = overlay[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    inset.imshow(crop)
    style_image_ax(inset, border_color=COLORS["unmatched"], border_width=1.8)
    style_image_ax(ax0)

    ax1.set_title("(b) Rare pixels but dominant error share", loc="left", fontweight="bold", pad=6)
    y_positions = np.array([1.0, 0.0])
    matched_values = np.array([stats["matched_pct"], stats["matched_error_share"]])
    unmatched_values = np.array([stats["unmatched_pct"], stats["unmatched_error_share"]])
    ax1.barh(y_positions, matched_values, color=COLORS["matched"], edgecolor="white", height=0.52)
    ax1.barh(y_positions, unmatched_values, left=matched_values, color=COLORS["unmatched"], edgecolor="white", height=0.52)
    ax1.set_xlim(0, 100)
    ax1.set_yticks(y_positions, labels=["Pixels", "Clean EPE sum"])
    ax1.set_xticks([0, 25, 50, 75, 100], labels=["0", "25", "50", "75", "100%"])
    ax1.grid(axis="x", color="#dfe3e8", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.text(46, 1.0, f"matched {stats['matched_pct']:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax1.text(
        matched_values[0] + unmatched_values[0] / 2,
        1.0,
        f"{stats['unmatched_pct']:.2f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    ax1.text(18, 0.0, f"matched\n{stats['matched_error_share']:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax1.text(75, 0.0, f"{stats['unmatched_error_share']:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax1.text(
        0.56,
        0.90,
        f"{stats['ratio']:.1f}x EPE\n({stats['unmatched_epe']:.2f} vs {stats['matched_epe']:.2f})",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "#f5f7fa", "edgecolor": "#d6dbe1", "boxstyle": "round,pad=0.35"},
    )
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2.set_title("(c) Oracle test isolates the remaining gap", loc="left", fontweight="bold", pad=6)
    labels = ["Sun-RAFT", "Ours", "Oracle"]
    values = [stats["display_baseline_clean"], stats["display_ours_clean"], stats["oracle_clean"]]
    colors = [COLORS["baseline"], COLORS["ours"], COLORS["oracle"]]
    bars = ax2.bar(labels, values, color=colors, width=0.58)
    ax2.set_ylabel("All-pixel EPE")
    ax2.set_ylim(0, max(values) * 1.25)
    ax2.grid(axis="y", color="#dfe3e8", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.text(
        0.5,
        0.95,
        "Oracle = replace baseline flow\nonly inside official unmatched pixels",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")
    ax2.annotate(
        "54.4% reducible",
        xy=(0, values[0]),
        xytext=(2, values[0] + 0.35),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["unmatched"], "lw": 1.3},
        color=COLORS["unmatched"],
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.suptitle(
        "Figure 1 conclusion: official unmatched pixels are rare but explain most of the removable error in unsupervised optical flow.",
        y=0.98,
        fontweight="bold",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.015,
        "All summary numbers follow the current paper: unmatched pixels are 7.25% of official-valid pixels, contribute 54.9% of the Clean EPE sum, and define the oracle gap in panel (c).",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    save_pub(fig, args.output_base)
    plt.close(fig)
    print(f"saved={args.output_base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
