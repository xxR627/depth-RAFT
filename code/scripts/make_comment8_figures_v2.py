#!/usr/bin/env python
"""Create the second-pass self-contained figures for Editor Comment #8.

This script intentionally does not reuse the rejected ``paper_fig*_comment8``
outputs. It rebuilds the visual argument from the current manuscript logic and
the existing experiment exports:

* Fig. 1 removes the obsolete old panel b and keeps the oracle-error evidence.
* Figs. 4 and 5 recompose the original qualitative plates into readable
  full-row plus enlarged-crop layouts.
* Fig. 6 keeps the ablation crop evidence but adds consistent legends and
  clearer scene blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "figures"
OUTPUT_DIR = FIGURE_DIR / "revision_v2"

SINTEL_ROOT = Path(r"G:\flow_data\sintel")
FIG1_IMAGE = SINTEL_ROOT / "training" / "clean" / "temple_2" / "frame_0025.png"
FIG1_OCC = SINTEL_ROOT / "training" / "occlusions" / "temple_2" / "frame_0025.png"
FIG1_INVALID = SINTEL_ROOT / "training" / "invalid" / "temple_2" / "frame_0025.png"
REGION_JSON = REPO_ROOT / "results" / "main" / "sintel_occ_noc_eval.json"
SUMMARY_JSON = REPO_ROOT / "results" / "main" / "step_35000_paper_summary.json"

SINTEL_PLATE = FIGURE_DIR / "figure3_sintel_qualitative_v5.png"
KITTI_PLATE = FIGURE_DIR / "figure4_kitti_qualitative_v3.png"
ABLATION_PLATE = FIGURE_DIR / "figure6_ablation_qualitative_crop.png"
ABLATION_CACHE = Path(r"G:\SCI NO.1\henrytask\paper_figures\ablation_qualitative_cache")

RED = "#d62728"
CYAN = "#1bb6d9"
INK = "#1f2933"
MUTED = "#667085"
BASELINE = "#4f5b68"
OURS = "#149b8f"
ORACLE = "#6b4fd3"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L")) > 0


def save_figure(fig: plt.Figure, output_base: Path, dpi: int = 450) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)


def style_image_ax(ax: plt.Axes, border_color: str | None = None, border_width: float = 1.25) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if border_color is None:
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)


def slice_spans(start: int, end: int, gap_segments: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for gap_start, gap_end in gap_segments:
        if cursor <= gap_start - 1:
            spans.append((cursor, gap_start - 1))
        cursor = gap_end + 1
    if cursor <= end:
        spans.append((cursor, end))
    return spans


def crop_norm(image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    x_pad = pad * (x1 - x0)
    y_pad = pad * (y1 - y0)
    x0 = max(0.0, x0 - x_pad)
    y0 = max(0.0, y0 - y_pad)
    x1 = min(1.0, x1 + x_pad)
    y1 = min(1.0, y1 + y_pad)
    left = int(round(x0 * w))
    top = int(round(y0 * h))
    right = max(left + 1, int(round(x1 * w)))
    bottom = max(top + 1, int(round(y1 * h)))
    return image[top:bottom, left:right]


def draw_box(ax: plt.Axes, image_shape: tuple[int, int], box: tuple[float, float, float, float], *, lw: float = 1.4) -> None:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    rect = mpl.patches.Rectangle(
        (x0 * w, y0 * h),
        (x1 - x0) * w,
        (y1 - y0) * h,
        fill=False,
        edgecolor=RED,
        linewidth=lw,
    )
    ax.add_patch(rect)


def overlay_mask(image: np.ndarray, mask: np.ndarray, *, alpha: float = 0.34, color: str = RED) -> np.ndarray:
    out = image.astype(np.float32).copy()
    rgb = np.array(mpl.colors.to_rgb(color), dtype=np.float32) * 255.0
    out[mask] = (1.0 - alpha) * out[mask] + alpha * rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def add_mask_contour(ax: plt.Axes, mask: np.ndarray, *, color: str = RED, lw: float = 0.85) -> None:
    ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=lw)


def _make_colorwheel() -> np.ndarray:
    segments = (15, 6, 4, 11, 13, 6)
    ry, yg, gc, cb, bm, mr = segments
    ncols = sum(segments)
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255
    return colorwheel


def flow_to_color(flow_uv: np.ndarray) -> np.ndarray:
    u = flow_uv[:, :, 0].copy()
    v = flow_uv[:, :, 1].copy()
    unknown = np.isnan(u) | np.isnan(v) | (np.abs(u) > 1e7) | (np.abs(v) > 1e7)
    u[unknown] = 0
    v[unknown] = 0
    rad = np.sqrt(u * u + v * v)
    rad_max = max(float(np.max(rad)), 1e-5)
    u /= rad_max
    v /= rad_max
    colorwheel = _make_colorwheel()
    ncols = colorwheel.shape[0]
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    image = np.zeros((*u.shape, 3), dtype=np.uint8)
    for channel in range(3):
        col0 = colorwheel[k0, channel] / 255.0
        col1 = colorwheel[k1, channel] / 255.0
        col = (1 - f) * col0 + f * col1
        inside = rad <= 1
        col[inside] = 1 - rad[inside] * (1 - col[inside])
        col[~inside] *= 0.75
        image[:, :, channel] = np.floor(255 * col)
    image[unknown] = 0
    return image


def flow_wheel(size: int = 220) -> np.ndarray:
    y, x = np.mgrid[-1:1 : complex(size), -1:1 : complex(size)]
    flow = np.stack([x, -y], axis=-1)
    image = flow_to_color(flow)
    radius = np.sqrt(x * x + y * y)
    image[radius > 1.0] = 255
    return image


def draw_flow_legend(ax: plt.Axes) -> None:
    ax.imshow(flow_wheel(220))
    ax.set_title("Flow colour wheel", loc="center", fontweight="bold", pad=1.5)
    ax.text(0.50, 0.91, "up", transform=ax.transAxes, ha="center", va="center", fontsize=5.8)
    ax.text(0.50, 0.09, "down", transform=ax.transAxes, ha="center", va="center", fontsize=5.8)
    ax.text(0.09, 0.50, "left", transform=ax.transAxes, ha="center", va="center", fontsize=5.8)
    ax.text(0.91, 0.50, "right", transform=ax.transAxes, ha="center", va="center", fontsize=5.8)
    style_image_ax(ax)


def draw_error_legend(ax: plt.Axes, *, vmax: float = 3.0) -> None:
    gradient = np.linspace(0, vmax, 512, dtype=np.float32)[None, :]
    ax.imshow(gradient, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title("Error map EPE", loc="left", fontweight="bold", pad=1.5)
    ax.set_yticks([])
    ax.set_xticks([0, 255, 511], labels=["0", f"{vmax/2:.1f}", f"{vmax:.0f}+"], fontsize=6.2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_error_strip(ax: plt.Axes, *, vmax: float = 3.0) -> None:
    gradient = np.linspace(0, vmax, 512, dtype=np.float32)[None, :]
    ax.imshow(gradient, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks([0, 511], labels=["0", f"{vmax:.0f}+"], fontsize=5.8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_swatch(ax: plt.Axes, color: str, label: str, note: str) -> None:
    ax.axis("off")
    ax.add_patch(mpl.patches.Rectangle((0.04, 0.34), 0.15, 0.34, color=color, transform=ax.transAxes))
    ax.text(0.25, 0.58, label, transform=ax.transAxes, ha="left", va="center", fontweight="bold", color=INK, fontsize=6.6)
    if note:
        ax.text(0.25, 0.34, note, transform=ax.transAxes, ha="left", va="center", color=MUTED, fontsize=5.8)


def hide_top_left_badge(image: np.ndarray) -> np.ndarray:
    """Remove embedded numeric badges from legacy qualitative error panels."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[: max(1, int(0.17 * h)), : max(1, int(0.20 * w))] = 255
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def extract_plate_grid(
    image: np.ndarray,
    x_spans: list[tuple[int, int]],
    y_spans: list[tuple[int, int]],
) -> list[list[np.ndarray]]:
    return [[image[y0 : y1 + 1, x0 : x1 + 1] for x0, x1 in x_spans] for y0, y1 in y_spans]


SINTEL_X_SPANS = slice_spans(
    80,
    6025,
    [(80, 415), (1029, 1040), (1654, 1664), (2278, 2289), (2903, 2914), (3528, 3538), (4152, 4163), (4777, 4788), (5402, 5412)],
)
SINTEL_Y_SPANS = slice_spans(
    110,
    3345,
    [(110, 113), (557, 578), (1022, 1043), (1487, 1508), (1951, 1973), (2416, 2437), (2881, 2902)],
)

KITTI_X_SPANS = slice_spans(
    70,
    5948,
    [(70, 308), (925, 936), (1553, 1564), (2181, 2192), (2809, 2820), (3437, 3448), (4065, 4076), (4693, 4704), (5321, 5332)],
)
KITTI_Y_SPANS = slice_spans(90, 2938, [(90, 94), (550, 572), (1027, 1049), (1505, 1527), (1983, 2004), (2460, 2482)])

ABLATION_X_SPANS = slice_spans(20, 2458, [(442, 454), (845, 857), (1248, 1261), (1652, 1665), (2056, 2068)])
ABLATION_Y_SPANS = [(136, 394), (483, 763), (848, 1113), (1196, 1400)]


def load_fig1_stats() -> dict[str, float]:
    region = json.loads(REGION_JSON.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    main = summary["sintel_train_main_eval"]
    return {
        "sun_clean": float(main["baseline"]["clean_epe"]),
        "ours_clean": float(main["ours_step35000"]["clean_epe"]),
        "oracle_clean": 0.77,
        "sun_unmatched": float(region["baseline"]["clean"]["unmatched"]),
        "ours_unmatched": float(region["ours_step35000"]["clean"]["unmatched"]),
    }


def make_fig1() -> None:
    image = load_rgb(FIG1_IMAGE)
    occ = read_mask(FIG1_OCC)
    invalid = read_mask(FIG1_INVALID)
    unmatched = occ & (~invalid)
    overlay = overlay_mask(image, unmatched, alpha=0.34, color=RED)
    stats = load_fig1_stats()

    sintel_grid = extract_plate_grid(load_rgb(SINTEL_PLATE), SINTEL_X_SPANS, SINTEL_Y_SPANS)
    temple_2_error = hide_top_left_badge(sintel_grid[1][5])
    temple_mask_small = cv2.resize(unmatched.astype(np.uint8), (temple_2_error.shape[1], temple_2_error.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    fig = plt.figure(figsize=(12.4, 4.15), constrained_layout=False)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.16, 0.58, 1.12, 0.96], wspace=0.22)
    ax_mask = fig.add_subplot(gs[0, 0])
    ax_mask_zoom = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[0, 3])

    roi = (0.08, 0.18, 0.42, 0.58)
    ax_mask.imshow(overlay)
    draw_box(ax_mask, overlay.shape[:2], roi, lw=1.8)
    style_image_ax(ax_mask)
    ax_mask.set_title("(a) Official unmatched mask", loc="left", fontweight="bold")
    ax_mask_zoom.imshow(crop_norm(overlay, roi, pad=0.035))
    ax_mask_zoom.set_title("zoom", loc="left", fontweight="bold", color=RED)
    style_image_ax(ax_mask_zoom, RED, 1.4)

    ax_err.imshow(temple_2_error)
    add_mask_contour(ax_err, temple_mask_small, lw=0.9)
    style_image_ax(ax_err)
    ax_err.set_title("(b) Baseline error aligns with unmatched boundaries", loc="left", fontweight="bold")

    labels = ["Sun-RAFT", "Depth-RAFT", "Oracle"]
    values = [stats["sun_clean"], stats["ours_clean"], stats["oracle_clean"]]
    colors = [BASELINE, OURS, ORACLE]
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, values, color=colors, width=0.62)
    ax_bar.set_title("(c) Oracle replacement bounds the removable error", loc="left", fontweight="bold")
    ax_bar.set_ylabel("Sintel Clean EPE")
    ax_bar.set_xticks(x, labels, rotation=0)
    ax_bar.set_ylim(0, 1.92)
    ax_bar.grid(axis="y", color="#d9dee7", linewidth=0.7)
    ax_bar.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")
    ax_bar.annotate(
        "replace only official\nunmatched pixels",
        xy=(2, values[2]),
        xytext=(1.72, 1.72),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.0},
        ha="right",
        va="center",
        color=RED,
        fontsize=6.8,
        fontweight="bold",
    )
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)

    fig.suptitle(
        "Figure 1. Unmatched regions localize the main removable error source.",
        y=0.98,
        fontsize=12.0,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.10)
    save_figure(fig, OUTPUT_DIR / "paper_fig1_comment8_v2", dpi=450)
    plt.close(fig)


def draw_panel_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    grid: list[list[np.ndarray]],
    selected: list[dict[str, object]],
    columns: list[str],
    *,
    title_offset_row: int = 1,
    valid_box_color: str = RED,
    metric_label_cols: set[int] | None = None,
) -> None:
    metric_label_cols = metric_label_cols or set()
    for col_idx, title in enumerate(columns):
        axes[title_offset_row, col_idx].set_title(title, fontweight="bold", pad=2.5)

    for block_idx, item in enumerate(selected):
        full_row = title_offset_row + block_idx * 2
        zoom_row = full_row + 1
        source_row = int(item["row_idx"])
        label = str(item["label"])
        roi = item["roi"]  # type: ignore[assignment]
        panels = grid[source_row]

        for col_idx, panel in enumerate(panels):
            panel_view = hide_top_left_badge(panel) if col_idx in metric_label_cols else panel
            ax = axes[full_row, col_idx]
            ax.imshow(panel_view)
            if col_idx == 0:
                draw_box(ax, panel_view.shape[:2], roi, lw=1.35)  # type: ignore[arg-type]
                ax.set_ylabel(label, rotation=0, labelpad=24, ha="right", va="center", fontweight="bold")
            style_image_ax(ax)

            zoom_ax = axes[zoom_row, col_idx]
            pad = 0.03 if col_idx in (5, 6, 7) else 0.05
            zoom_ax.imshow(crop_norm(panel_view, roi, pad=pad))  # type: ignore[arg-type]
            style_image_ax(zoom_ax, valid_box_color, 1.0)
        axes[zoom_row, 0].set_ylabel("zoom", rotation=0, labelpad=24, ha="right", va="center", color=valid_box_color, fontweight="bold")


def make_fig4() -> None:
    grid = extract_plate_grid(load_rgb(SINTEL_PLATE), SINTEL_X_SPANS, SINTEL_Y_SPANS)
    columns = [
        "Image",
        "GT flow",
        "Sun-RAFT",
        "SMURF",
        "Depth-RAFT",
        "Sun error",
        "SMURF error",
        "Depth-RAFT error",
        "Unmatched mask",
    ]
    selected = [
        {"row_idx": 1, "label": "temple_2", "roi": (0.20, 0.16, 0.72, 0.70)},
        {"row_idx": 2, "label": "ambush_2", "roi": (0.35, 0.55, 0.74, 0.98)},
    ]

    fig, axes = plt.subplots(
        5,
        9,
        figsize=(15.8, 8.9),
        gridspec_kw={"height_ratios": [0.42, 1.0, 0.95, 1.0, 0.95], "wspace": 0.018, "hspace": 0.080},
    )
    axes[0, 0].axis("off")
    axes[0, 0].text(0.02, 0.58, "Flow colours:\nstandard RAFT wheel", transform=axes[0, 0].transAxes, ha="left", va="center", fontweight="bold", fontsize=6.0)
    draw_error_strip(axes[0, 1])
    draw_swatch(axes[0, 2], RED, "Mask", "red overlay")
    draw_swatch(axes[0, 3], RED, "Crop", "red box")
    for col in range(4, 9):
        axes[0, col].axis("off")
    draw_panel_grid(fig, axes, grid, selected, columns, title_offset_row=1, metric_label_cols={5, 6, 7})

    fig.suptitle(
        "Figure 4. Depth-RAFT suppresses cross-boundary leakage in high-occlusion Sintel scenes.",
        y=0.995,
        fontsize=11.6,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.025)
    save_figure(fig, OUTPUT_DIR / "paper_fig4_sintel_comment8_v2", dpi=430)
    plt.close(fig)


def make_fig5() -> None:
    grid = extract_plate_grid(load_rgb(KITTI_PLATE), KITTI_X_SPANS, KITTI_Y_SPANS)
    columns = [
        "Image",
        "GT flow",
        "Sun-RAFT",
        "SMURF",
        "Depth-RAFT",
        "Sun error",
        "SMURF error",
        "Depth-RAFT error",
        "Valid GT",
    ]
    selected = [
        {"row_idx": 3, "label": "000090", "roi": (0.02, 0.31, 0.38, 0.76)},
        {"row_idx": 4, "label": "000095", "roi": (0.00, 0.50, 0.42, 0.98)},
    ]

    fig, axes = plt.subplots(
        5,
        9,
        figsize=(16.2, 8.55),
        gridspec_kw={"height_ratios": [0.42, 1.0, 0.95, 1.0, 0.95], "wspace": 0.018, "hspace": 0.080},
    )
    axes[0, 0].axis("off")
    axes[0, 0].text(0.02, 0.58, "Flow colours:\nstandard RAFT wheel", transform=axes[0, 0].transAxes, ha="left", va="center", fontweight="bold", fontsize=6.0)
    draw_error_strip(axes[0, 1])
    draw_swatch(axes[0, 2], RED, "Crop", "red box")
    draw_swatch(axes[0, 3], CYAN, "Valid GT", "cyan overlay")
    for col in range(4, 9):
        axes[0, col].axis("off")
    draw_panel_grid(fig, axes, grid, selected, columns, title_offset_row=1, metric_label_cols={5, 6, 7})

    fig.suptitle(
        "Figure 5. Monocular depth priors improve Sintel-to-KITTI transfer on weak-texture vehicles and shadowed boundaries.",
        y=0.995,
        fontsize=11.4,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.025)
    save_figure(fig, OUTPUT_DIR / "paper_fig5_kitti_comment8_v2", dpi=430)
    plt.close(fig)


FIG6_METHODS = [
    ("baseline", "Baseline"),
    ("g_only", "+G"),
    ("z_only", "+Z"),
    ("g_z", "+G+Z"),
    ("g_z_dab", "+G+Z+DAB"),
]

FIG6_SAMPLES = [
    {
        "pass": "clean",
        "scene": "ambush_2",
        "frame": 12,
        "title": "clean / ambush_2 / 0012",
        "box": (360, 240, 760, 430),
    },
    {
        "pass": "final",
        "scene": "temple_3",
        "frame": 42,
        "title": "final / temple_3 / 0042",
        "box": (800, 70, 1020, 220),
    },
]


def read_flow_file(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = np.fromfile(handle, np.float32, count=1)[0]
        if magic != 202021.25:
            raise RuntimeError(f"Invalid .flo file: {path}")
        width = int(np.fromfile(handle, np.int32, count=1)[0])
        height = int(np.fromfile(handle, np.int32, count=1)[0])
        data = np.fromfile(handle, np.float32, count=height * width * 2)
    return data.reshape(height, width, 2)


def crop_xyxy(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return array[y1:y2, x1:x2]


def draw_box_xyxy(ax: plt.Axes, box: tuple[int, int, int, int], *, lw: float = 1.8) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(
        mpl.patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=RED,
            linewidth=lw,
        )
    )


def fig6_cache_path(sample: dict[str, object], method_key: str) -> Path:
    return ABLATION_CACHE / f"{sample['pass']}_{sample['scene']}_frame_{int(sample['frame']):04d}_{method_key}.npy"


def fig6_unmatched_mask(sample: dict[str, object]) -> np.ndarray:
    pass_name = str(sample["pass"])
    scene = str(sample["scene"])
    frame = int(sample["frame"])
    occ = read_mask(SINTEL_ROOT / "training" / "occlusions" / scene / f"frame_{frame:04d}.png")
    invalid_path = SINTEL_ROOT / "training" / "invalid" / scene / f"frame_{frame:04d}.png"
    invalid = read_mask(invalid_path) if invalid_path.is_file() else np.zeros_like(occ, dtype=bool)
    return occ & (~invalid)


def make_fig6_from_cache() -> None:
    missing: list[Path] = []
    for sample in FIG6_SAMPLES:
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        for method_key, _ in FIG6_METHODS:
            path = fig6_cache_path(sample, method_key)
            if not path.is_file():
                missing.append(path)
        for path in (
            SINTEL_ROOT / "training" / pass_name / scene / f"frame_{frame:04d}.png",
            SINTEL_ROOT / "training" / "flow" / scene / f"frame_{frame:04d}.flo",
        ):
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing Fig. 6 source files:\n" + "\n".join(str(path) for path in missing[:8]))

    error_vmax = 12.0
    fig, axes = plt.subplots(
        5,
        6,
        figsize=(14.8, 8.15),
        gridspec_kw={"height_ratios": [0.38, 0.92, 1.0, 0.92, 1.0], "wspace": 0.028, "hspace": 0.12},
    )
    axes[0, 0].axis("off")
    axes[0, 0].text(0.02, 0.58, "Flow colours:\nstandard RAFT wheel", transform=axes[0, 0].transAxes, ha="left", va="center", fontweight="bold", fontsize=6.0)
    draw_error_strip(axes[0, 1], vmax=error_vmax)
    draw_swatch(axes[0, 2], RED, "Mask", "red overlay")
    draw_swatch(axes[0, 3], RED, "Crop", "red box")
    axes[0, 4].axis("off")
    axes[0, 5].axis("off")

    for example_idx, sample in enumerate(FIG6_SAMPLES):
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        box = sample["box"]  # type: ignore[assignment]
        title = str(sample["title"])
        row = 1 + example_idx * 2

        image = load_rgb(SINTEL_ROOT / "training" / pass_name / scene / f"frame_{frame:04d}.png")
        gt = read_flow_file(SINTEL_ROOT / "training" / "flow" / scene / f"frame_{frame:04d}.flo")
        unmatched = fig6_unmatched_mask(sample)

        ax_context = axes[row, 0]
        ax_context.imshow(image)
        draw_box_xyxy(ax_context, box, lw=1.7)  # type: ignore[arg-type]
        ax_context.set_title(title, loc="left", fontweight="bold", pad=2.5)
        ax_context.set_ylabel("context", rotation=0, labelpad=28, ha="right", va="center", fontweight="bold")
        style_image_ax(ax_context)

        crop_image = crop_xyxy(image, box)  # type: ignore[arg-type]
        crop_mask = crop_xyxy(unmatched, box)  # type: ignore[arg-type]
        ax_crop = axes[row + 1, 0]
        ax_crop.imshow(overlay_mask(crop_image, crop_mask, alpha=0.46, color=RED))
        ax_crop.set_ylabel("unmatched\ncrop", rotation=0, labelpad=28, ha="right", va="center", fontweight="bold")
        style_image_ax(ax_crop, RED, 1.0)

        for col_idx, (method_key, label) in enumerate(FIG6_METHODS, start=1):
            pred = np.load(fig6_cache_path(sample, method_key))
            epe = np.sqrt(np.sum((pred - gt) ** 2, axis=2))

            flow_ax = axes[row, col_idx]
            flow_ax.imshow(flow_to_color(crop_xyxy(pred, box)))  # type: ignore[arg-type]
            if example_idx == 0:
                flow_ax.set_title(label, fontweight="bold", pad=2.5)
            style_image_ax(flow_ax)

            error_ax = axes[row + 1, col_idx]
            error_ax.imshow(crop_xyxy(epe, box), cmap="viridis", vmin=0.0, vmax=error_vmax)  # type: ignore[arg-type]
            style_image_ax(error_ax)

    fig.suptitle(
        "Figure 6. Z-context gives the largest unmatched-region gain; DAB-Smooth provides the final boundary cleanup.",
        y=0.985,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.060, right=0.995, top=0.90, bottom=0.025)
    save_figure(fig, OUTPUT_DIR / "paper_fig6_ablation_comment8_v2", dpi=430)
    plt.close(fig)


def make_fig6() -> None:
    if ABLATION_CACHE.is_dir():
        make_fig6_from_cache()
        return

    grid = extract_plate_grid(load_rgb(ABLATION_PLATE), ABLATION_X_SPANS, ABLATION_Y_SPANS)
    methods = ["Context", "Baseline", "+G", "+Z", "+G+Z", "+G+Z+DAB"]
    ambush_values = ["", "7.80", "7.30", "7.62", "7.25", "7.11"]
    temple_values = ["", "23.47", "21.79", "12.05", "10.77", "10.05"]

    fig, axes = plt.subplots(
        5,
        6,
        figsize=(14.8, 8.45),
        gridspec_kw={"height_ratios": [0.30, 1.0, 1.0, 1.0, 1.0], "wspace": 0.028, "hspace": 0.085},
    )
    axes[0, 0].axis("off")
    axes[0, 0].text(0.02, 0.72, "Flow colours:\nstandard RAFT wheel", transform=axes[0, 0].transAxes, ha="left", va="center", fontweight="bold", fontsize=6.2)
    draw_error_strip(axes[0, 1])
    draw_swatch(axes[0, 2], RED, "Red overlay", "official unmatched mask")
    draw_swatch(axes[0, 3], RED, "Red box", "enlarged crop")
    axes[0, 4].axis("off")
    axes[0, 5].axis("off")

    for col_idx, method in enumerate(methods):
        title = method
        if col_idx > 0:
            title = f"{method}\nambush EPE {ambush_values[col_idx]}"
        axes[1, col_idx].set_title(title, fontweight="bold", pad=2.5)

    row_labels = [
        "ambush_2\nflow crop",
        "ambush_2\nerror crop",
        "temple_3\nflow crop",
        "temple_3\nerror crop",
    ]
    for row_idx in range(4):
        for col_idx in range(6):
            ax = axes[row_idx + 1, col_idx]
            ax.imshow(grid[row_idx][col_idx])
            border = RED if col_idx == 0 and row_idx in (1, 3) else None
            style_image_ax(ax, border, 1.0)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], rotation=0, labelpad=32, ha="right", va="center", fontweight="bold")
            if row_idx == 2 and col_idx > 0:
                ax.set_title(f"{methods[col_idx]}\ntemple EPE {temple_values[col_idx]}", fontweight="bold", pad=2.5)

    fig.suptitle(
        "Figure 6. Z-context gives the largest unmatched-region gain; DAB-Smooth provides the final boundary cleanup.",
        y=0.995,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.014,
        "The two scene blocks use the same method order and the same flow/error legends. Numbers are average EPE inside the highlighted crop.",
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=6.4,
    )
    save_figure(fig, OUTPUT_DIR / "paper_fig6_ablation_comment8_v2", dpi=430)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "fig1", "fig4", "fig5", "fig6"),
        default="all",
        help="Generate one figure or the full Comment #8 figure set.",
    )
    return parser.parse_args()


def main() -> int:
    configure_style()
    args = parse_args()
    targets = {
        "fig1": make_fig1,
        "fig4": make_fig4,
        "fig5": make_fig5,
        "fig6": make_fig6,
    }
    if args.only == "all":
        for name, fn in targets.items():
            print(f"[comment8-v2] generating {name}", flush=True)
            fn()
    else:
        print(f"[comment8-v2] generating {args.only}", flush=True)
        targets[args.only]()
    print(f"[comment8-v2] output_dir={OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
