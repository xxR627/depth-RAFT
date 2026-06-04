#!/usr/bin/env python
"""Revise Figure 6 from the old Comment #8 ablation layout.

This keeps the original four-row evidence structure:
full/crop pair for ambush_2, then full/crop pair for temple_3. The left column
is regenerated from the original Sintel images so the crop is not obscured by a
red overlay; method flow/error crops are reused from the prior figure source.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from PIL import Image

import make_comment8_figures_v2 as base


REPO_ROOT = base.REPO_ROOT
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUT_DIR = REPO_ROOT / "figures" / "revision_v2"
SOURCE = REPO_ROOT / "figures" / "figure6_ablation_qualitative_crop.png"
META = WORKSPACE_ROOT / "henrytask" / "paper_figures" / "figure6_ablation_qualitative_crop.json"
OUTPUT_BASE = OUTPUT_DIR / "paper_fig6_ablation_comment8_oldbase_v8"
RED = base.RED
PANEL_SIZE = (460, 260)
METHODS = [("baseline", "Baseline"), ("g_only", "+G"), ("z_only", "+Z"), ("g_z", "+G+Z"), ("g_z_dab", "+G+Z+DAB")]
X_SPANS = base.ABLATION_X_SPANS
Y_SPANS = base.ABLATION_Y_SPANS


def configure_style() -> None:
    base.configure_style()
    mpl.rcParams.update(
        {
            "font.size": 7.3,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.3,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def resize_rgb(image: np.ndarray, size: tuple[int, int] = PANEL_SIZE) -> np.ndarray:
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize(size, Image.Resampling.LANCZOS))


def load_metadata() -> dict:
    return json.loads(META.read_text(encoding="utf-8"))


def extract_grid() -> list[list[np.ndarray]]:
    source = base.load_rgb(SOURCE)
    return base.extract_plate_grid(source, X_SPANS, Y_SPANS)


def draw_epe_badge(ax: plt.Axes, value: float) -> None:
    ax.text(
        0.035,
        0.92,
        f"EPE {value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "alpha": 0.92, "pad": 1.9, "edgecolor": "none"},
        zorder=30,
    )


def style_ax(ax: plt.Axes, border: str | None = None) -> None:
    base.style_image_ax(ax, border, 0.9 if border else 1.0)


def draw_full_panel(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    panel = resize_rgb(image)
    h, w = image.shape[:2]
    sx = PANEL_SIZE[0] / w
    sy = PANEL_SIZE[1] / h
    x1, y1, x2, y2 = box
    return panel, (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def connect_crop(fig: plt.Figure, full_ax: plt.Axes, crop_ax: plt.Axes, rect: tuple[float, float, float, float]) -> None:
    x1, _y1, x2, y2 = rect
    pairs = (((x1, y2), (0, 0)), ((x2, y2), (PANEL_SIZE[0], 0)))
    for source, target in pairs:
        fig.add_artist(
            ConnectionPatch(
                xyA=source,
                xyB=target,
                coordsA=full_ax.transData,
                coordsB=crop_ax.transData,
                color=RED,
                linewidth=0.8,
                clip_on=False,
                zorder=40,
            )
        )


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUTPUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.015)


def main() -> int:
    configure_style()
    meta = load_metadata()
    grid = extract_grid()
    samples = meta["samples"]

    fig, axes = plt.subplots(
        4,
        6,
        figsize=(15.6, 7.0),
        gridspec_kw={"height_ratios": [1, 1, 1, 1], "wspace": 0.026, "hspace": 0.11},
    )

    for col_idx, (_key, label) in enumerate(METHODS, start=1):
        axes[0, col_idx].set_title(label, fontweight="bold", pad=2.2)

    for sample_idx, sample in enumerate(samples):
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        box = tuple(int(v) for v in sample["crop_bounds_xyxy"])
        row0 = sample_idx * 2

        image = base.load_rgb(base.SINTEL_ROOT / "training" / pass_name / scene / f"frame_{frame:04d}.png")
        full_panel, rect = draw_full_panel(image, box)
        crop_panel = resize_rgb(base.crop_xyxy(image, box))

        full_ax = axes[row0, 0]
        full_ax.imshow(full_panel)
        base.draw_box(full_ax, full_panel.shape[:2], (rect[0] / PANEL_SIZE[0], rect[1] / PANEL_SIZE[1], rect[2] / PANEL_SIZE[0], rect[3] / PANEL_SIZE[1]), lw=1.15)
        full_ax.set_ylabel(f"{pass_name}\n{scene}\n{frame:04d}-{frame + 1:04d}", rotation=0, labelpad=31, ha="right", va="center", fontweight="bold")
        style_ax(full_ax)

        crop_ax = axes[row0 + 1, 0]
        crop_ax.imshow(crop_panel)
        style_ax(crop_ax, RED)
        connect_crop(fig, full_ax, crop_ax, rect)

        for method_idx, (method_key, _label) in enumerate(METHODS, start=1):
            flow_panel = resize_rgb(grid[row0][method_idx])
            error_panel = resize_rgb(grid[row0 + 1][method_idx])
            axes[row0, method_idx].imshow(flow_panel)
            style_ax(axes[row0, method_idx])
            axes[row0 + 1, method_idx].imshow(error_panel)
            style_ax(axes[row0 + 1, method_idx], RED)
            draw_epe_badge(axes[row0 + 1, method_idx], float(sample["metrics"][method_key]["crop_focus_epe"]))

    fig.subplots_adjust(left=0.062, right=0.996, top=0.93, bottom=0.035)
    save_figure(fig)
    plt.close(fig)
    print(f"saved={OUTPUT_BASE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
