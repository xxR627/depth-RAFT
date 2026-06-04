#!/usr/bin/env python
"""Recompose paper Figure 6 as a self-contained ablation figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_selfcontained_utils import (
    draw_error_legend,
    draw_flow_legend,
    draw_swatch,
    load_rgb,
    save_pub,
    slice_spans,
    style_image_ax,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "figures" / "figure6_ablation_qualitative_crop.png"
OUTPUT = REPO_ROOT / "figures" / "revision" / "paper_fig6_ablation_comment8"
METHODS = ["Baseline", "+ G-only", "+ Z-only", "+ G+Z", "+ G+Z+DAB"]
TOP_VALUES = [7.80, 7.30, 7.62, 7.25, 7.11]
BOTTOM_VALUES = [23.47, 21.79, 12.05, 10.77, 10.05]
X_SPANS = slice_spans(20, 2458, [(442, 454), (845, 857), (1248, 1261), (1652, 1665), (2056, 2068)])
Y_SPANS = [(136, 394), (483, 763), (848, 1113), (1196, 1400)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-base", type=Path, default=OUTPUT)
    return parser.parse_args()


def extract_grid(image: np.ndarray) -> list[list[np.ndarray]]:
    grid: list[list[np.ndarray]] = []
    for y0, y1 in Y_SPANS:
        row: list[np.ndarray] = []
        for x0, x1 in X_SPANS:
            row.append(image[y0 : y1 + 1, x0 : x1 + 1])
        grid.append(row)
    return grid


def main() -> int:
    args = parse_args()
    source = load_rgb(args.source)
    grid = extract_grid(source)

    fig, axes = plt.subplots(
        5,
        6,
        figsize=(17.6, 9.2),
        gridspec_kw={"height_ratios": [0.54, 1.0, 1.05, 1.0, 1.05], "wspace": 0.03, "hspace": 0.12},
    )

    draw_flow_legend(axes[0, 0])
    draw_error_legend(axes[0, 1])
    draw_swatch(axes[0, 2], "#d62728", "Red overlay", "official unmatched region")
    draw_swatch(axes[0, 3], "#d62728", "Red box", "enlarged crop region")
    axes[0, 4].axis("off")
    axes[0, 5].axis("off")
    fig.suptitle(
        "Figure 6 conclusion: the Z branch yields the largest gains in the unmatched-dominated crop, and DAB provides the final boundary cleanup.",
        y=0.992,
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.70,
        0.955,
        "Top pair = ambush_2; bottom pair = temple_3. The leftmost column shows the full frame and the enlarged unmatched crop used for the comparison.",
        ha="center",
        va="center",
        fontsize=7.8,
    )

    for row_idx in range(4):
        for col_idx in range(6):
            axes[row_idx + 1, col_idx].imshow(grid[row_idx][col_idx])
            style_image_ax(axes[row_idx + 1, col_idx], border_color="#d62728" if row_idx in (1, 3) else None)

    for col_idx, method in enumerate(METHODS, start=1):
        axes[1, col_idx].set_title(f"{method}\n{TOP_VALUES[col_idx - 1]:.2f}", fontweight="bold", pad=4)
        axes[3, col_idx].text(
            0.5,
            1.02,
            f"{BOTTOM_VALUES[col_idx - 1]:.2f}",
            transform=axes[3, col_idx].transAxes,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )

    fig.text(
        0.5,
        0.014,
        "The existing crop figure already isolates the two failure modes; this revision adds a shared flow legend, a shared clipped-EPE legend, and explicit red-mask / red-box explanations.",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    save_pub(fig, args.output_base)
    plt.close(fig)
    print(f"saved={args.output_base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
