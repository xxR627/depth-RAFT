#!/usr/bin/env python
"""Create connected-crop Python redraws for Editor Comment #8 figures.

The v3 redraw keeps the v2 data sources and panel choices, but removes figure
titles and zoom labels. Each context ROI is connected to the enlarged crop with
red guide lines so the crop relationship remains readable after export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

import make_comment8_figures_v2 as base


OUTPUT_DIR = base.OUTPUT_DIR
RED = base.RED
CYAN = base.CYAN
INK = base.INK
BASELINE = base.BASELINE
OURS = base.OURS
ORACLE = base.ORACLE


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
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.018)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.018)


def show_image(ax: plt.Axes, image: np.ndarray, border: str | None = None, lw: float = 1.0) -> None:
    ax.imshow(image)
    base.style_image_ax(ax, border, lw)


def add_panel_letter(ax: plt.Axes, label: str, x: float = -0.03, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def _box_points_norm(image_shape: tuple[int, int], box: tuple[float, float, float, float]) -> dict[str, tuple[float, float]]:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    return {
        "tl": (x0 * w, y0 * h),
        "tr": (x1 * w, y0 * h),
        "bl": (x0 * w, y1 * h),
        "br": (x1 * w, y1 * h),
    }


def connect_vertical(
    fig: plt.Figure,
    context_ax: plt.Axes,
    crop_ax: plt.Axes,
    context_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    roi: tuple[float, float, float, float],
    *,
    color: str = RED,
    lw: float = 0.9,
) -> None:
    points = _box_points_norm(context_shape, roi)
    crop_h, crop_w = crop_shape[:2]
    pairs = ((points["bl"], (0, 0)), (points["br"], (crop_w, 0)))
    for src, dst in pairs:
        fig.add_artist(
            ConnectionPatch(
                xyA=src,
                xyB=dst,
                coordsA=context_ax.transData,
                coordsB=crop_ax.transData,
                color=color,
                linewidth=lw,
                clip_on=False,
                zorder=20,
            )
        )


def connect_horizontal(
    fig: plt.Figure,
    context_ax: plt.Axes,
    crop_ax: plt.Axes,
    context_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    roi: tuple[float, float, float, float],
    *,
    color: str = RED,
    lw: float = 0.9,
) -> None:
    points = _box_points_norm(context_shape, roi)
    crop_h, _crop_w = crop_shape[:2]
    pairs = ((points["tr"], (0, 0)), (points["br"], (0, crop_h)))
    for src, dst in pairs:
        fig.add_artist(
            ConnectionPatch(
                xyA=src,
                xyB=dst,
                coordsA=context_ax.transData,
                coordsB=crop_ax.transData,
                color=color,
                linewidth=lw,
                clip_on=False,
                zorder=20,
            )
        )


def crop_panel(panel: np.ndarray, roi: tuple[float, float, float, float], col_idx: int) -> np.ndarray:
    pad = 0.03 if col_idx in (5, 6, 7) else 0.05
    return base.crop_norm(panel, roi, pad=pad)


def make_fig1() -> None:
    image = base.load_rgb(base.FIG1_IMAGE)
    occ = base.read_mask(base.FIG1_OCC)
    invalid = base.read_mask(base.FIG1_INVALID)
    unmatched = occ & (~invalid)
    overlay = base.overlay_mask(image, unmatched, alpha=0.34, color=RED)
    stats = base.load_fig1_stats()

    sintel_grid = base.extract_plate_grid(base.load_rgb(base.SINTEL_PLATE), base.SINTEL_X_SPANS, base.SINTEL_Y_SPANS)
    temple_error = base.hide_top_left_badge(sintel_grid[1][5])

    roi = (0.08, 0.18, 0.42, 0.58)
    crop = base.crop_norm(overlay, roi, pad=0.035)

    fig = plt.figure(figsize=(12.2, 3.05), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.18, 0.58, 1.16, 0.86], wspace=0.20)
    ax_image = fig.add_subplot(gs[0, 0])
    ax_crop = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[0, 3])

    show_image(ax_image, overlay)
    base.draw_box(ax_image, overlay.shape[:2], roi, lw=1.55)
    add_panel_letter(ax_image, "(a)")
    ax_image.text(
        0.02,
        0.055,
        "red = official unmatched/occluded",
        transform=ax_image.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=7.0,
        fontweight="bold",
        bbox={"facecolor": RED, "alpha": 0.78, "pad": 2.2, "edgecolor": "none"},
    )

    show_image(ax_crop, crop, RED, 1.0)
    connect_horizontal(fig, ax_image, ax_crop, overlay.shape[:2], crop.shape[:2], roi, lw=0.9)

    show_image(ax_error, temple_error)
    add_panel_letter(ax_error, "(b)")

    labels = ["Sun-RAFT", "Depth-RAFT", "Oracle"]
    values = [stats["sun_clean"], stats["ours_clean"], stats["oracle_clean"]]
    colors = [BASELINE, OURS, ORACLE]
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, values, color=colors, width=0.62)
    add_panel_letter(ax_bar, "(c)", x=-0.08)
    ax_bar.set_ylabel("Sintel Clean EPE")
    ax_bar.set_xticks(x, labels)
    ax_bar.set_ylim(0, 1.92)
    ax_bar.grid(axis="y", color="#d9dee7", linewidth=0.7)
    ax_bar.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=6.7,
        )
    ax_bar.annotate(
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
        ax_bar.spines[side].set_visible(False)

    fig.subplots_adjust(left=0.025, right=0.995, top=0.92, bottom=0.15)
    save_figure(fig, OUTPUT_DIR / "paper_fig1_comment8_v3")
    plt.close(fig)


def draw_connected_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    grid: list[list[np.ndarray]],
    selected: Sequence[dict[str, object]],
    columns: Sequence[str],
    *,
    metric_label_cols: set[int] | None = None,
    row_label_pad: int = 22,
) -> None:
    metric_label_cols = metric_label_cols or set()
    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=2.5)

    for block_idx, item in enumerate(selected):
        full_row = block_idx * 2
        crop_row = full_row + 1
        source_row = int(item["row_idx"])
        label = str(item["label"])
        roi = item["roi"]  # type: ignore[assignment]
        panels = grid[source_row]

        first_crop_shape: tuple[int, int] | None = None
        for col_idx, panel in enumerate(panels):
            panel_view = base.hide_top_left_badge(panel) if col_idx in metric_label_cols else panel
            ax = axes[full_row, col_idx]
            show_image(ax, panel_view)
            if col_idx == 0:
                base.draw_box(ax, panel_view.shape[:2], roi, lw=1.2)  # type: ignore[arg-type]
                ax.set_ylabel(label, rotation=0, labelpad=row_label_pad, ha="right", va="center", fontweight="bold")

            crop = crop_panel(panel_view, roi, col_idx)  # type: ignore[arg-type]
            zoom_ax = axes[crop_row, col_idx]
            show_image(zoom_ax, crop, RED, 0.85)
            if col_idx == 0:
                first_crop_shape = crop.shape[:2]

        if first_crop_shape is not None:
            connect_vertical(
                fig,
                axes[full_row, 0],
                axes[crop_row, 0],
                panels[0].shape[:2],
                first_crop_shape,
                roi,  # type: ignore[arg-type]
                lw=0.85,
            )


def make_fig4() -> None:
    grid = base.extract_plate_grid(base.load_rgb(base.SINTEL_PLATE), base.SINTEL_X_SPANS, base.SINTEL_Y_SPANS)
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
        4,
        9,
        figsize=(15.8, 5.70),
        gridspec_kw={"height_ratios": [1.0, 0.96, 1.0, 0.96], "wspace": 0.018, "hspace": 0.105},
    )
    draw_connected_grid(fig, axes, grid, selected, columns, metric_label_cols={5, 6, 7})
    fig.subplots_adjust(left=0.045, right=0.996, top=0.92, bottom=0.035)
    save_figure(fig, OUTPUT_DIR / "paper_fig4_sintel_comment8_v3")
    plt.close(fig)


def make_fig5() -> None:
    grid = base.extract_plate_grid(base.load_rgb(base.KITTI_PLATE), base.KITTI_X_SPANS, base.KITTI_Y_SPANS)
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
        4,
        9,
        figsize=(16.2, 5.55),
        gridspec_kw={"height_ratios": [1.0, 0.96, 1.0, 0.96], "wspace": 0.018, "hspace": 0.105},
    )
    draw_connected_grid(fig, axes, grid, selected, columns, metric_label_cols={5, 6, 7})
    fig.subplots_adjust(left=0.043, right=0.996, top=0.92, bottom=0.035)
    save_figure(fig, OUTPUT_DIR / "paper_fig5_kitti_comment8_v3")
    plt.close(fig)


def _fig6_cache_path(sample: dict[str, object], method_key: str) -> Path:
    return base.fig6_cache_path(sample, method_key)


def _ensure_fig6_sources() -> None:
    missing: list[Path] = []
    for sample in base.FIG6_SAMPLES:
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        for method_key, _ in base.FIG6_METHODS:
            path = _fig6_cache_path(sample, method_key)
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
    _ensure_fig6_sources()

    error_vmax = 12.0
    columns = ["Image", "Baseline", "+G", "+Z", "+G+Z", "+G+Z+DAB"]
    fig, axes = plt.subplots(
        4,
        6,
        figsize=(14.8, 5.55),
        gridspec_kw={"height_ratios": [1.0, 0.98, 1.0, 0.98], "wspace": 0.028, "hspace": 0.125},
    )
    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=2.5)

    for example_idx, sample in enumerate(base.FIG6_SAMPLES):
        pass_name = str(sample["pass"])
        scene = str(sample["scene"])
        frame = int(sample["frame"])
        box = sample["box"]  # type: ignore[assignment]
        row = example_idx * 2

        image = base.load_rgb(base.SINTEL_ROOT / "training" / pass_name / scene / f"frame_{frame:04d}.png")
        gt = base.read_flow_file(base.SINTEL_ROOT / "training" / "flow" / scene / f"frame_{frame:04d}.flo")
        unmatched = base.fig6_unmatched_mask(sample)
        x1, y1, x2, y2 = box  # type: ignore[misc]
        norm_box = (x1 / image.shape[1], y1 / image.shape[0], x2 / image.shape[1], y2 / image.shape[0])

        context_ax = axes[row, 0]
        show_image(context_ax, image)
        base.draw_box_xyxy(context_ax, box, lw=1.3)  # type: ignore[arg-type]
        context_ax.set_ylabel(f"{scene}\n{frame:04d}", rotation=0, labelpad=28, ha="right", va="center", fontweight="bold")

        crop_image = base.crop_xyxy(image, box)  # type: ignore[arg-type]
        crop_mask = base.crop_xyxy(unmatched, box)  # type: ignore[arg-type]
        crop_overlay = base.overlay_mask(crop_image, crop_mask, alpha=0.46, color=RED)
        crop_ax = axes[row + 1, 0]
        show_image(crop_ax, crop_overlay, RED, 0.85)
        connect_vertical(fig, context_ax, crop_ax, image.shape[:2], crop_overlay.shape[:2], norm_box, lw=0.85)

        for col_idx, (method_key, _label) in enumerate(base.FIG6_METHODS, start=1):
            pred = np.load(_fig6_cache_path(sample, method_key))
            epe = np.sqrt(np.sum((pred - gt) ** 2, axis=2))

            flow_ax = axes[row, col_idx]
            show_image(flow_ax, base.flow_to_color(base.crop_xyxy(pred, box)))  # type: ignore[arg-type]

            error_ax = axes[row + 1, col_idx]
            error_ax.imshow(base.crop_xyxy(epe, box), cmap="viridis", vmin=0.0, vmax=error_vmax)  # type: ignore[arg-type]
            base.style_image_ax(error_ax, RED, 0.85)

    fig.subplots_adjust(left=0.060, right=0.996, top=0.91, bottom=0.035)
    save_figure(fig, OUTPUT_DIR / "paper_fig6_ablation_comment8_v3")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "fig1", "fig4", "fig5", "fig6"),
        default="all",
        help="Generate one figure or the full connected-crop redraw set.",
    )
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
            print(f"[comment8-v3] generating {name}", flush=True)
            fn()
    else:
        print(f"[comment8-v3] generating {args.only}", flush=True)
        targets[args.only]()
    print(f"[comment8-v3] output_dir={OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
