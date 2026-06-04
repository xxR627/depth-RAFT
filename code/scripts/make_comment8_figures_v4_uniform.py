#!/usr/bin/env python
"""Uniform-size Python redraws for Comment #8 figures.

All image panels are resized to fixed pixel dimensions before plotting. Figures
4-6 use ROI-to-crop connector lines; Figure 1 does not use a crop or ROI box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from PIL import Image, ImageDraw

import make_comment8_figures_v2 as base


OUTPUT_DIR = base.OUTPUT_DIR
WORKSPACE_ROOT = base.REPO_ROOT.parent
HENRY_FIGURE_DIR = WORKSPACE_ROOT / "henrytask" / "paper_figures"
RED = base.RED
BASELINE = base.BASELINE
OURS = base.OURS
ORACLE = base.ORACLE
LATEST_SUFFIX = "v8"

GRID_SIZE = (430, 285)  # width, height for every Fig. 4/5 image and crop cell
FIG6_CELL_SIZE = (500, 225)  # width, height for every Fig. 6 cell
FIG1_SIZE = (1180, 500)


def _resample() -> int:
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize(size, _resample()))


def resize_scalar(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return base.cv2.resize(image.astype(np.float32), size, interpolation=base.cv2.INTER_LINEAR)


def letterbox_rgb(image: np.ndarray, size: tuple[int, int], fill: int = 255) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    width, height = size
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = resize_rgb(image, (new_w, new_h))
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, (scale, left, top, new_w, new_h)


def letterbox_scalar(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = resize_scalar(image, (new_w, new_h))
    canvas = np.full((height, width), np.nan, dtype=np.float32)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _cover_offsets(
    src_shape: tuple[int, int],
    size: tuple[int, int],
    focus_xy: tuple[float, float] | None = None,
) -> tuple[float, int, int, int, int]:
    width, height = size
    src_h, src_w = src_shape[:2]
    scale = max(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    if focus_xy is None:
        focus_x = src_w / 2.0
        focus_y = src_h / 2.0
    else:
        focus_x, focus_y = focus_xy
    left = int(round(focus_x * scale - width / 2.0))
    top = int(round(focus_y * scale - height / 2.0))
    left = min(max(0, left), max(0, new_w - width))
    top = min(max(0, top), max(0, new_h - height))
    return scale, left, top, new_w, new_h


def cover_rgb(
    image: np.ndarray,
    size: tuple[int, int],
    focus_xy: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    width, height = size
    scale, left, top, new_w, new_h = _cover_offsets(image.shape[:2], size, focus_xy)
    resized = resize_rgb(image, (new_w, new_h))
    return resized[top : top + height, left : left + width], (scale, left, top, new_w, new_h)


def cover_scalar(image: np.ndarray, size: tuple[int, int], focus_xy: tuple[float, float] | None = None) -> np.ndarray:
    width, height = size
    scale, left, top, new_w, new_h = _cover_offsets(image.shape[:2], size, focus_xy)
    resized = resize_scalar(image, (new_w, new_h))
    return resized[top : top + height, left : left + width]


def configure_style() -> None:
    base.configure_style()
    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_base: Path, dpi: int = 600) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.012)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.012)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.012)


def show_image(ax: plt.Axes, image: np.ndarray, border: str | None = None, lw: float = 0.9) -> None:
    ax.imshow(image)
    base.style_image_ax(ax, border, lw)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def draw_epe_badge(ax: plt.Axes, value: float, *, fontsize: float = 9.4) -> None:
    ax.text(
        0.035,
        0.93,
        f"EPE {value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=fontsize,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.90, "pad": 1.9, "edgecolor": "none"},
        zorder=40,
    )


def make_context_zoom_composite(image: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    top_h = int(height * 0.47)
    gap = 14
    crop_h = height - top_h - gap
    red_rgb = tuple(int(round(255 * channel)) for channel in mpl.colors.to_rgb(RED))

    context = Image.fromarray(image.astype(np.uint8)).resize((width, top_h), _resample())
    crop = Image.fromarray(base.crop_xyxy(image, box).astype(np.uint8)).resize((width, crop_h), _resample())
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(context, (0, 0))
    crop_y = top_h + gap
    canvas.paste(crop, (0, crop_y))

    draw = ImageDraw.Draw(canvas)
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = box
    sx = width / image_w
    sy = top_h / image_h
    rect = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    draw.rectangle(rect, outline=red_rgb, width=3)
    draw.rectangle((0, crop_y, width - 1, height - 1), outline=red_rgb, width=3)
    draw.line((rect[0], rect[3], 0, crop_y), fill=red_rgb, width=2)
    draw.line((rect[2], rect[3], width - 1, crop_y), fill=red_rgb, width=2)
    return np.asarray(canvas)


def add_panel_letter(ax: plt.Axes, text: str, x: float = -0.045, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def connect_crop(
    fig: plt.Figure,
    context_ax: plt.Axes,
    crop_ax: plt.Axes,
    context_size: tuple[int, int],
    crop_size: tuple[int, int],
    roi: tuple[float, float, float, float],
    *,
    lw: float = 0.85,
) -> None:
    width, height = context_size
    crop_width, _crop_height = crop_size
    x0, y0, x1, y1 = roi
    pairs = (
        ((x0 * width, y1 * height), (0, 0)),
        ((x1 * width, y1 * height), (crop_width, 0)),
    )
    for source, target in pairs:
        fig.add_artist(
            ConnectionPatch(
                xyA=source,
                xyB=target,
                coordsA=context_ax.transData,
                coordsB=crop_ax.transData,
                color=RED,
                linewidth=lw,
                clip_on=False,
                zorder=30,
            )
        )


def connect_crop_rect(
    fig: plt.Figure,
    context_ax: plt.Axes,
    crop_ax: plt.Axes,
    rect: tuple[float, float, float, float],
    crop_size: tuple[int, int],
    *,
    lw: float = 0.85,
) -> None:
    x0, _y0, x1, y1 = rect
    crop_width, _crop_height = crop_size
    pairs = (((x0, y1), (0, 0)), ((x1, y1), (crop_width, 0)))
    for source, target in pairs:
        fig.add_artist(
            ConnectionPatch(
                xyA=source,
                xyB=target,
                coordsA=context_ax.transData,
                coordsB=crop_ax.transData,
                color=RED,
                linewidth=lw,
                clip_on=False,
                zorder=30,
            )
        )


def crop_resize(panel: np.ndarray, roi: tuple[float, float, float, float], size: tuple[int, int], col_idx: int) -> np.ndarray:
    pad = 0.03 if col_idx in (5, 6, 7) else 0.05
    return resize_rgb(base.crop_norm(panel, roi, pad=pad), size)


def make_fig1() -> None:
    image = base.load_rgb(base.FIG1_IMAGE)
    occ = base.read_mask(base.FIG1_OCC)
    invalid = base.read_mask(base.FIG1_INVALID)
    unmatched = occ & (~invalid)
    overlay = base.overlay_mask(image, unmatched, alpha=0.34, color=RED)
    overlay = resize_rgb(overlay, FIG1_SIZE)

    sintel_grid = base.extract_plate_grid(base.load_rgb(base.SINTEL_PLATE), base.SINTEL_X_SPANS, base.SINTEL_Y_SPANS)
    error_map = resize_rgb(base.hide_top_left_badge(sintel_grid[1][5]), FIG1_SIZE)
    stats = base.load_fig1_stats()

    fig = plt.figure(figsize=(12.4, 2.85), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.88], wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    show_image(ax_a, overlay)
    add_panel_letter(ax_a, "(a)")
    ax_a.text(
        0.025,
        0.92,
        "temple_2 / frame 0025",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8.2,
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2.0, "edgecolor": "none"},
    )
    ax_a.text(
        0.025,
        0.055,
        "red = official unmatched/occluded",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=8.2,
        fontweight="bold",
        bbox={"facecolor": RED, "alpha": 0.78, "pad": 2.0, "edgecolor": "none"},
    )

    show_image(ax_b, error_map)
    add_panel_letter(ax_b, "(b)")

    labels = ["Sun-RAFT", "Depth-RAFT", "Oracle"]
    values = [stats["sun_clean"], stats["ours_clean"], stats["oracle_clean"]]
    colors = [BASELINE, OURS, ORACLE]
    x = np.arange(len(labels))
    bars = ax_c.bar(x, values, color=colors, width=0.62)
    add_panel_letter(ax_c, "(c)", x=-0.08)
    ax_c.set_ylabel("Sintel Clean EPE")
    ax_c.set_xticks(x, labels)
    ax_c.set_ylim(0, 1.92)
    ax_c.grid(axis="y", color="#d9dee7", linewidth=0.7)
    ax_c.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax_c.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=6.8)
    ax_c.annotate(
        "replace only official\nunmatched pixels",
        xy=(2, values[2]),
        xytext=(1.70, 1.72),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.0},
        ha="right",
        va="center",
        color=RED,
        fontsize=6.5,
        fontweight="bold",
    )
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    fig.subplots_adjust(left=0.024, right=0.996, top=0.88, bottom=0.15)
    save_figure(fig, OUTPUT_DIR / f"paper_fig1_comment8_{LATEST_SUFFIX}")
    plt.close(fig)


def draw_uniform_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    grid: list[list[np.ndarray]],
    selected: Sequence[dict[str, object]],
    columns: Sequence[str],
    *,
    size: tuple[int, int] = GRID_SIZE,
    metric_label_cols: set[int] | None = None,
) -> None:
    metric_label_cols = metric_label_cols or set()
    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=2.0)

    for block_idx, item in enumerate(selected):
        full_row = block_idx * 2
        crop_row = full_row + 1
        source_row = int(item["row_idx"])
        label = str(item["label"])
        roi = item["roi"]  # type: ignore[assignment]
        epe_payload = item.get("epe", {})  # type: ignore[assignment]
        full_epe = epe_payload.get("full", {}) if isinstance(epe_payload, dict) else {}
        crop_epe = epe_payload.get("crop", full_epe) if isinstance(epe_payload, dict) else {}
        panels = grid[source_row]

        first_context_shape: tuple[int, int] | None = None
        for col_idx, panel in enumerate(panels):
            panel_view = base.hide_top_left_badge(panel) if col_idx in metric_label_cols else panel
            context = resize_rgb(panel_view, size)
            crop = crop_resize(panel_view, roi, size, col_idx)  # type: ignore[arg-type]

            context_ax = axes[full_row, col_idx]
            show_image(context_ax, context)
            if col_idx in full_epe:
                draw_epe_badge(context_ax, float(full_epe[col_idx]))
            if col_idx == 0:
                base.draw_box(context_ax, context.shape[:2], roi, lw=1.15)  # type: ignore[arg-type]
                context_ax.set_ylabel(label, rotation=0, labelpad=24, ha="right", va="center", fontweight="bold")
                first_context_shape = context.shape[:2]

            crop_ax = axes[crop_row, col_idx]
            show_image(crop_ax, crop, RED, 0.8)
            if col_idx in crop_epe:
                draw_epe_badge(crop_ax, float(crop_epe[col_idx]))

        if first_context_shape is not None:
            connect_crop(fig, axes[full_row, 0], axes[crop_row, 0], size, size, roi)  # type: ignore[arg-type]


def make_fig4() -> None:
    grid = base.extract_plate_grid(base.load_rgb(base.SINTEL_PLATE), base.SINTEL_X_SPANS, base.SINTEL_Y_SPANS)
    columns = ["Image", "GT flow", "Sun-RAFT", "SMURF", "Depth-RAFT", "Sun error", "SMURF error", "Depth-RAFT error", "Unmatched mask"]
    metrics = load_json(HENRY_FIGURE_DIR / "figure3_selection_metadata_v5.json").get("figure_metrics", {})
    temple = metrics.get("temple_2/frame_0025", {})
    ambush = metrics.get("ambush_2/frame_0012", {})
    selected = [
        {
            "row_idx": 1,
            "label": "temple_2",
            "roi": (0.20, 0.16, 0.72, 0.70),
            "epe": {
                "full": {5: temple.get("baseline_epe", 15.28), 6: temple.get("smurf_epe", 16.25), 7: temple.get("ours_epe", 9.15)},
                "crop": {
                    5: temple.get("baseline_unmatched_epe", 42.62),
                    6: temple.get("smurf_unmatched_epe", 42.48),
                    7: temple.get("ours_unmatched_epe", 23.92),
                },
            },
        },
        {
            "row_idx": 2,
            "label": "ambush_2",
            "roi": (0.35, 0.55, 0.74, 0.98),
            "epe": {
                "full": {5: ambush.get("baseline_epe", 2.57), 6: ambush.get("smurf_epe", 2.85), 7: ambush.get("ours_epe", 1.05)},
                "crop": {
                    5: ambush.get("baseline_unmatched_epe", 15.63),
                    6: ambush.get("smurf_unmatched_epe", 16.29),
                    7: ambush.get("ours_unmatched_epe", 4.66),
                },
            },
        },
    ]

    fig, axes = plt.subplots(
        4,
        9,
        figsize=(15.8, 5.25),
        gridspec_kw={"height_ratios": [1, 1, 1, 1], "wspace": 0.018, "hspace": 0.105},
    )
    draw_uniform_grid(fig, axes, grid, selected, columns, metric_label_cols={5, 6, 7})
    fig.subplots_adjust(left=0.045, right=0.996, top=0.90, bottom=0.035)
    save_figure(fig, OUTPUT_DIR / f"paper_fig4_sintel_comment8_{LATEST_SUFFIX}")
    plt.close(fig)


def make_fig5() -> None:
    grid = base.extract_plate_grid(base.load_rgb(base.KITTI_PLATE), base.KITTI_X_SPANS, base.KITTI_Y_SPANS)
    columns = ["Image", "GT flow", "Sun-RAFT", "SMURF", "Depth-RAFT", "Sun error", "SMURF error", "Depth-RAFT error", "Valid GT"]
    metrics = load_json(HENRY_FIGURE_DIR / "figure4_selection_metadata_v3.json").get("figure_metrics", {})
    frame_090 = metrics.get("000090", {})
    frame_103 = metrics.get("000103", {})
    selected = [
        {
            "row_idx": 3,
            "label": "000090",
            "roi": (0.02, 0.31, 0.38, 0.76),
            "epe": {
                "full": {
                    5: frame_090.get("baseline_epe", 39.30),
                    6: frame_090.get("smurf_epe", 71.11),
                    7: frame_090.get("ours_epe", 4.45),
                },
                "crop": {
                    5: 60.374813079833984,
                    6: 90.88699340820312,
                    7: 5.249618053436279,
                },
            },
        },
        {
            "row_idx": 5,
            "label": "000103",
            "roi": (0.16, 0.28, 0.54, 0.76),
            "epe": {
                "full": {
                    5: frame_103.get("baseline_epe", 25.68),
                    6: frame_103.get("smurf_epe", 20.06),
                    7: frame_103.get("ours_epe", 11.00),
                },
                "crop": {
                    5: 8.343094825744629,
                    6: 3.554377317428589,
                    7: 3.0560429096221924,
                },
            },
        },
    ]

    fig, axes = plt.subplots(
        4,
        9,
        figsize=(15.8, 5.25),
        gridspec_kw={"height_ratios": [1, 1, 1, 1], "wspace": 0.018, "hspace": 0.105},
    )
    draw_uniform_grid(fig, axes, grid, selected, columns, metric_label_cols={5, 6, 7})
    fig.subplots_adjust(left=0.043, right=0.996, top=0.90, bottom=0.035)
    save_figure(fig, OUTPUT_DIR / f"paper_fig5_kitti_comment8_{LATEST_SUFFIX}")
    plt.close(fig)


def ensure_fig6_sources() -> None:
    missing: list[Path] = []
    for sample in base.FIG6_SAMPLES:
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        for method_key, _label in base.FIG6_METHODS:
            path = base.fig6_cache_path(sample, method_key)
            if not path.is_file():
                missing.append(path)
        for path in (
            base.SINTEL_ROOT / "training" / pass_name / scene / f"frame_{frame:04d}.png",
            base.SINTEL_ROOT / "training" / "flow" / scene / f"frame_{frame:04d}.flo",
        ):
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing Fig. 6 source files:\n" + "\n".join(str(path) for path in missing[:8]))


def make_fig6() -> None:
    if not base.ABLATION_CACHE.is_dir():
        raise FileNotFoundError(f"Missing ablation cache directory: {base.ABLATION_CACHE}")
    ensure_fig6_sources()
    fig6_metrics_payload = load_json(HENRY_FIGURE_DIR / "figure6_ablation_qualitative_crop.json")
    fig6_metrics: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}
    fig6_boxes: dict[tuple[str, str, int], tuple[int, int, int, int]] = {}
    for sample in fig6_metrics_payload.get("samples", []):
        key = (str(sample.get("pass")), str(sample.get("scene")), int(sample.get("frame")))
        if sample.get("crop_bounds_xyxy") is not None:
            fig6_boxes[key] = tuple(int(value) for value in sample["crop_bounds_xyxy"])  # type: ignore[assignment]
        fig6_metrics[key] = {
            method_key: {
                "full": float(values.get("full_epe")),
                "crop": float(values.get("crop_focus_epe")),
            }
            for method_key, values in sample.get("metrics", {}).items()
            if isinstance(values, dict) and values.get("full_epe") is not None and values.get("crop_focus_epe") is not None
        }

    columns = ["Image", "Baseline", "+G", "+Z", "+G+Z", "+G+Z+DAB"]
    fig, axes = plt.subplots(
        4,
        6,
        figsize=(14.8, 7.80),
        gridspec_kw={
            "height_ratios": [1, 1, 1, 1],
            "wspace": 0.026,
            "hspace": 0.105,
        },
    )
    error_cmap = plt.get_cmap("viridis").copy()
    error_cmap.set_bad(color="white", alpha=1.0)
    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=2.0)

    for example_idx, sample in enumerate(base.FIG6_SAMPLES):
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        sample_key = (pass_name, scene, frame)
        box = fig6_boxes.get(sample_key, sample["box"])  # type: ignore[assignment]
        full_row = example_idx * 2
        crop_row = full_row + 1
        sample_metrics = fig6_metrics.get((pass_name, scene, frame), {})

        image = base.load_rgb(base.SINTEL_ROOT / "training" / pass_name / scene / f"frame_{frame:04d}.png")
        gt = base.read_flow_file(base.SINTEL_ROOT / "training" / "flow" / scene / f"frame_{frame:04d}.flo")
        x0, y0, x1, y1 = box
        roi_focus = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        full_image, full_meta = cover_rgb(image, FIG6_CELL_SIZE, roi_focus)
        crop_image, _crop_meta = cover_rgb(base.crop_xyxy(image, box), FIG6_CELL_SIZE)  # type: ignore[arg-type]
        full_scale, full_left, full_top, _full_w, _full_h = full_meta
        rect = (
            x0 * full_scale - full_left,
            y0 * full_scale - full_top,
            x1 * full_scale - full_left,
            y1 * full_scale - full_top,
        )

        full_image_ax = axes[full_row, 0]
        show_image(full_image_ax, full_image)
        full_image_ax.add_patch(
            mpl.patches.Rectangle(
                (rect[0], rect[1]),
                rect[2] - rect[0],
                rect[3] - rect[1],
                fill=False,
                edgecolor=RED,
                linewidth=1.15,
            )
        )
        full_image_ax.set_ylabel(f"{scene}\n{frame:04d}-{frame + 1:04d}", rotation=0, labelpad=28, ha="right", va="center", fontweight="bold")

        crop_image_ax = axes[crop_row, 0]
        show_image(crop_image_ax, crop_image, RED, 0.8)
        connect_crop_rect(fig, full_image_ax, crop_image_ax, rect, FIG6_CELL_SIZE)

        for col_idx, (method_key, _label) in enumerate(base.FIG6_METHODS, start=1):
            pred = np.load(base.fig6_cache_path(sample, method_key))
            epe = np.sqrt(np.sum((pred - gt) ** 2, axis=2))

            full_error = cover_scalar(epe, FIG6_CELL_SIZE, roi_focus)
            crop_error = cover_scalar(base.crop_xyxy(epe, box), FIG6_CELL_SIZE)  # type: ignore[arg-type]
            axes[full_row, col_idx].imshow(full_error, cmap=error_cmap, vmin=0.0, vmax=12.0)
            base.style_image_ax(axes[full_row, col_idx])
            axes[crop_row, col_idx].imshow(crop_error, cmap=error_cmap, vmin=0.0, vmax=12.0)
            base.style_image_ax(axes[crop_row, col_idx], RED, 0.8)
            if method_key in sample_metrics:
                draw_epe_badge(axes[full_row, col_idx], sample_metrics[method_key]["full"], fontsize=10.0)
                draw_epe_badge(axes[crop_row, col_idx], sample_metrics[method_key]["crop"], fontsize=10.0)

    fig.subplots_adjust(left=0.060, right=0.996, top=0.91, bottom=0.035)
    save_figure(fig, OUTPUT_DIR / "paper_fig6_ablation_comment8_v13")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "fig1", "fig4", "fig5", "fig6"), default="all")
    return parser.parse_args()


def main() -> int:
    configure_style()
    targets = {
        "fig1": make_fig1,
        "fig4": make_fig4,
        "fig5": make_fig5,
        "fig6": make_fig6,
    }
    args = parse_args()
    if args.only == "all":
        for name, fn in targets.items():
            print(f"[comment8-v4] generating {name}", flush=True)
            fn()
    else:
        print(f"[comment8-v4] generating {args.only}", flush=True)
        targets[args.only]()
    print(f"[comment8-v4] output_dir={OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
