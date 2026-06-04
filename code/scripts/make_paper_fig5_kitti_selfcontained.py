#!/usr/bin/env python
"""Recompose paper Figure 5 from the exported KITTI qualitative plate."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_selfcontained_utils import (
    crop_norm,
    draw_box,
    draw_error_legend,
    draw_flow_legend,
    draw_swatch,
    load_rgb,
    save_pub,
    slice_spans,
    style_image_ax,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "figures" / "figure4_kitti_qualitative_v3.png"
OUTPUT = REPO_ROOT / "figures" / "revision" / "paper_fig5_kitti_comment8"
COLUMN_TITLES = [
    "Image",
    "GT Flow",
    "Sun-RAFT",
    "SMURF",
    "Ours",
    "Sun-RAFT Error",
    "SMURF Error",
    "Ours Error",
    "Sparse valid GT",
]
X_SPANS = slice_spans(70, 5948, [(70, 308), (925, 936), (1553, 1564), (2181, 2192), (2809, 2820), (3437, 3448), (4065, 4076), (4693, 4704), (5321, 5332)])
Y_SPANS = slice_spans(90, 2938, [(90, 94), (550, 572), (1027, 1049), (1505, 1527), (1983, 2004), (2460, 2482)])
SELECTED = [
    {"row_idx": 1, "label": "000148", "roi": (0.47, 0.24, 0.70, 0.46)},
    {"row_idx": 2, "label": "000068", "roi": (0.08, 0.18, 0.34, 0.56)},
    {"row_idx": 4, "label": "000095", "roi": (0.19, 0.16, 0.58, 0.51)},
]


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
        7,
        len(COLUMN_TITLES),
        figsize=(18.4, 11.7),
        gridspec_kw={"height_ratios": [0.50, 1.0, 0.92, 1.0, 0.92, 1.0, 0.92], "wspace": 0.03, "hspace": 0.10},
    )

    draw_flow_legend(axes[0, 0])
    draw_error_legend(axes[0, 1])
    draw_swatch(axes[0, 2], "#d62728", "Red box", "enlarged weak-texture region")
    draw_swatch(axes[0, 3], "#1bb6d9", "Cyan overlay", "KITTI sparse valid GT support")
    axes[0, 4].axis("off")
    axes[0, 5].axis("off")
    axes[0, 6].axis("off")
    axes[0, 7].axis("off")
    axes[0, 8].axis("off")
    fig.suptitle(
        "Figure 5 conclusion: the depth prior sharpens weak-texture vehicles and shadowed boundaries under Sintel→KITTI zero-shot transfer.",
        y=0.992,
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.63,
        0.955,
        "Full row = global context. Red-box zoom row = enlarged transfer-failure region. Error maps are shown only on sparse valid GT pixels.",
        ha="center",
        va="center",
        fontsize=7.8,
    )

    for col_idx, title in enumerate(COLUMN_TITLES):
        axes[1, col_idx].set_title(title, fontweight="bold", pad=4)

    for block_idx, item in enumerate(SELECTED):
        full_row = 1 + block_idx * 2
        crop_row = full_row + 1
        panels = grid[item["row_idx"]]
        roi = item["roi"]
        for col_idx, panel in enumerate(panels):
            ax = axes[full_row, col_idx]
            ax.imshow(panel)
            if col_idx == 0:
                draw_box(ax, panel.shape[:2], roi, linewidth=2.0)
                ax.set_ylabel(item["label"], rotation=0, labelpad=30, ha="right", va="center", fontweight="bold")
            style_image_ax(ax)
            crop_ax = axes[crop_row, col_idx]
            crop = crop_norm(panel, roi, pad=0.04 if col_idx not in (5, 6, 7) else 0.02)
            crop_ax.imshow(crop)
            style_image_ax(crop_ax, border_color="#d62728")
        axes[crop_row, 0].set_ylabel("zoom", rotation=0, labelpad=30, ha="right", va="center", color="#d62728", fontweight="bold")

    for row_idx in range(1, axes.shape[0]):
        for col_idx in range(axes.shape[1]):
            if row_idx > 1 and row_idx % 2 == 1:
                axes[row_idx, col_idx].set_title("")

    fig.text(
        0.5,
        0.014,
        "Representative rows from the existing exported figure. The enlarged rows expose the far-vehicle and shadow-boundary improvements that were too small in the original page layout.",
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
