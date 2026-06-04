#!/usr/bin/env python
"""Helpers for self-contained paper figure revisions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "figure.titlesize": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_pub(fig: plt.Figure, output_base: Path, dpi: int = 400) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)


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
    right = int(round(x1 * w))
    bottom = int(round(y1 * h))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return image[top:bottom, left:right]


def draw_box(
    ax: plt.Axes,
    image_shape: tuple[int, int],
    box: tuple[float, float, float, float],
    *,
    color: str = "#d62728",
    linewidth: float = 1.8,
) -> None:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    rect = mpl.patches.Rectangle(
        (x0 * w, y0 * h),
        (x1 - x0) * w,
        (y1 - y0) * h,
        fill=False,
        linewidth=linewidth,
        edgecolor=color,
    )
    ax.add_patch(rect)


def style_image_ax(ax: plt.Axes, border_color: str | None = None, border_width: float = 1.4) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if border_color is None:
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(border_width)
        spine.set_edgecolor(border_color)


def _make_colorwheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
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


def flow_to_color_fallback(flow_uv: np.ndarray) -> np.ndarray:
    u = flow_uv[:, :, 0].copy()
    v = flow_uv[:, :, 1].copy()
    unknown = np.isnan(u) | np.isnan(v) | (np.abs(u) > 1e7) | (np.abs(v) > 1e7)
    u[unknown] = 0
    v[unknown] = 0
    rad = np.sqrt(u**2 + v**2)
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


def make_flow_wheel_image(size: int = 220) -> np.ndarray:
    y, x = np.mgrid[-1:1 : complex(size), -1:1 : complex(size)]
    flow = np.stack([x, -y], axis=-1)
    image = flow_to_color_fallback(flow)
    radius = np.sqrt(x**2 + y**2)
    image[radius > 1.0] = 255
    return image


def draw_flow_legend(ax: plt.Axes) -> None:
    wheel = make_flow_wheel_image(220)
    ax.imshow(wheel)
    ax.set_title("Flow wheel", loc="left", fontweight="bold", pad=2)
    style_image_ax(ax)
    ax.text(0.5, 1.02, "up", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    ax.text(0.5, -0.06, "down", transform=ax.transAxes, ha="center", va="top", fontsize=7)
    ax.text(-0.02, 0.5, "left", transform=ax.transAxes, ha="right", va="center", fontsize=7)
    ax.text(1.02, 0.5, "right", transform=ax.transAxes, ha="left", va="center", fontsize=7)


def draw_error_legend(ax: plt.Axes, *, vmax: float = 3.0, cmap: str = "viridis") -> None:
    gradient = np.linspace(0.0, vmax, 512, dtype=np.float32)[None, :]
    ax.imshow(gradient, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_title("Error EPE (clipped)", loc="left", fontweight="bold", pad=2)
    ax.set_yticks([])
    ax.set_xticks([0, 256, 511], labels=["0", f"{vmax/2:.1f}", f"{vmax:.0f}+"])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_swatch(ax: plt.Axes, color: str, title: str, subtitle: str | None = None) -> None:
    swatch = np.ones((24, 24, 3), dtype=np.uint8) * 255
    rgb = np.asarray(mpl.colors.to_rgb(color))
    swatch[2:22, 2:22] = np.round(255 * rgb).astype(np.uint8)
    ax.imshow(swatch)
    style_image_ax(ax)
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    if subtitle:
        ax.text(1.08, 0.5, subtitle, transform=ax.transAxes, ha="left", va="center", fontsize=7)
