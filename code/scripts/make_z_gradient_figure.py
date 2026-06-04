#!/usr/bin/env python
"""Create a Z-gradient vs image-gradient motivation figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from dav2_wrapper import DAv2FeatureExtractor
from utils import frame_utils


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_DAV2 = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL = Path(r"G:\flow_data\sintel")
DEFAULT_FIG = PACKAGE_ROOT / "figures" / "figure_z_gradient_vs_image_gradient.png"
DEFAULT_PDF = PACKAGE_ROOT / "figures" / "figure_z_gradient_vs_image_gradient.pdf"
DEFAULT_META = PACKAGE_ROOT / "figures" / "figure_z_gradient_vs_image_gradient_metadata.json"


SCENES = [
    ("market_5", 30),
    ("temple_2", 25),
    ("ambush_2", 12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def _robust_norm(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(x[valid], [lo, hi])
    if p_hi <= p_lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def _gradient_mag(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _flow_gradient_mag(flow: np.ndarray) -> np.ndarray:
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(ux * ux + uy * uy + vx * vx + vy * vy)


def _load_sample(sintel_root: Path, scene: str, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    image_path = sintel_root / "training" / "clean" / scene / f"frame_{frame_idx:04d}.png"
    flow_path = sintel_root / "training" / "flow" / scene / f"frame_{frame_idx:04d}.flo"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not flow_path.is_file():
        raise FileNotFoundError(flow_path)
    image = np.array(frame_utils.read_gen(str(image_path))).astype(np.uint8)
    flow = np.array(frame_utils.read_gen(str(flow_path))).astype(np.float32)
    return image, flow


@torch.no_grad()
def _extract_depth(extractor: DAv2FeatureExtractor, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_t = torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device) / 255.0
    depth = extractor.extract_depth(image_t)[0, 0].detach().cpu().numpy()
    return depth.astype(np.float32)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DAv2FeatureExtractor(str(args.dav2_weights), device=device, precision="auto")
    extractor.eval()

    _configure_style()
    fig, axes = plt.subplots(len(SCENES), 5, figsize=(15.5, 8.4), constrained_layout=False)
    titles = ["Image", "Image Gradient", "DAv2 $Z$", "$\\nabla Z$", "GT Flow Gradient"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    metadata: dict[str, Any] = {
        "scenes": [],
        "definition": {
            "image_gradient": "Sobel magnitude on grayscale Sintel clean frame.",
            "z_gradient": "Sobel magnitude on frozen DAv2 depth map Z.",
            "gt_flow_gradient": "Sobel magnitude over GT flow channels, used as a motion-boundary proxy.",
        },
    }

    for row, (scene, frame_idx) in enumerate(SCENES):
        image, flow = _load_sample(args.sintel_root, scene, frame_idx)
        depth = _extract_depth(extractor, image, device)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        image_grad = _gradient_mag(gray)
        depth_grad = _gradient_mag(_robust_norm(depth))
        flow_grad = _flow_gradient_mag(flow)

        panels = [
            (image, None),
            (_robust_norm(image_grad), "magma"),
            (_robust_norm(depth), "viridis"),
            (_robust_norm(depth_grad), "magma"),
            (_robust_norm(flow_grad), "magma"),
        ]
        for col, (panel, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].text(
            -0.06,
            0.5,
            f"{scene}\nframe_{frame_idx:04d}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
        )
        metadata["scenes"].append(
            {
                "scene": scene,
                "frame_idx": frame_idx,
                "image": str(args.sintel_root / "training" / "clean" / scene / f"frame_{frame_idx:04d}.png"),
                "flow": str(args.sintel_root / "training" / "flow" / scene / f"frame_{frame_idx:04d}.flo"),
            }
        )

    fig.subplots_adjust(left=0.105, right=0.995, top=0.93, bottom=0.035, wspace=0.025, hspace=0.08)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300)
    fig.savefig(args.output_pdf)
    plt.close(fig)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(_to_builtin(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"saved_png={args.output_png}")
    print(f"saved_pdf={args.output_pdf}")
    print(f"saved_metadata={args.metadata_json}")


if __name__ == "__main__":
    main()
