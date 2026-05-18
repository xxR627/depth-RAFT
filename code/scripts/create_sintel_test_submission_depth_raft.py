#!/usr/bin/env python
"""Create Sintel clean/final test-set submission files for Sun-RAFT + G+Z checkpoints.

The Sintel test split has no public ground truth, so this script only writes `.flo`
files in the expected clean/final/sequence/frame_XXXX.flo structure and optionally
packages them as a zip for benchmark-server upload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for path in (REPO_ROOT, CORE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import datasets_un
from eval_depth_raft_region_decomp import _build_model
from eval_sun_raft_region_decomp import COARSEST_SCALE, ITERS, _patched_get_extention_factory, _resolve_device
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_OURS_CKPT = PACKAGE_ROOT / "checkpoints" / "depth_raft_g_z_dab_step35000_best.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "sintel_test_submission_step35000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug cap per pass. 0 means full test split.")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a zip archive.")
    return parser.parse_args()


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


@torch.no_grad()
def _write_pass(
    *,
    model,
    sintel_root: Path,
    dstype: str,
    output_dir: Path,
    device: torch.device,
    iters: int,
    log_every: int,
    max_pairs: int,
) -> dict[str, Any]:
    dataset = datasets_un.MpiSintel(
        split="test",
        aug_params=None,
        dstype=dstype,
        show_extra_info=True,
        is_test=True,
    )
    total = len(dataset) if max_pairs <= 0 else min(max_pairs, len(dataset))
    flow_prev = None
    sequence_prev = None
    start_time = time.time()
    written = 0
    model.eval()

    for test_id in range(total):
        image1, image2, (sequence, frame_zero_based) = dataset[test_id]
        if sequence != sequence_prev:
            flow_prev = None

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape, coarsest_scale=COARSEST_SCALE)
        image1_pad, image2_pad = padder.pad(image1, image2)
        flow_low, flow_pr = model(
            image1_pad,
            image2_pad,
            iters=iters,
            flow_init=flow_prev,
            test_mode=True,
            bw=False,
        )
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        flow_prev = forward_interpolate(flow_low[0])[None].to(device)

        seq_dir = output_dir / dstype / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)
        frame_one_based = int(frame_zero_based) + 1
        frame_utils.writeFlow(str(seq_dir / f"frame_{frame_one_based:04d}.flo"), flow)
        sequence_prev = sequence
        written += 1

        if ((test_id + 1) % log_every == 0) or (test_id + 1 == total):
            print(
                f"[{dstype}] written={test_id + 1}/{total} "
                f"sequence={sequence} frame={frame_one_based:04d} "
                f"elapsed_sec={time.time() - start_time:.1f}",
                flush=True,
            )

    return {"dstype": dstype, "num_pairs": written, "elapsed_sec": time.time() - start_time}


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    for path in (args.baseline_checkpoint, args.checkpoint, args.dav2_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.sintel_root / "test" / "clean").is_dir():
        raise FileNotFoundError(f"Missing Sintel test split under {args.sintel_root}")

    device = _resolve_device()
    model = _build_model(
        device=device,
        baseline_checkpoint=args.baseline_checkpoint,
        fusion_checkpoint=args.checkpoint,
        dav2_weights=args.dav2_weights,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patched_get_extention = _patched_get_extention_factory(args.sintel_root)
    original_get_extention = datasets_un.get_extention
    datasets_un.get_extention = patched_get_extention
    payload: dict[str, Any] = {
        "status": "RUNNING",
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint,
        "dav2_weights": args.dav2_weights,
        "sintel_root": args.sintel_root,
        "output_dir": args.output_dir,
        "protocol": {
            "split": "test",
            "passes": ["clean", "final"],
            "iters": args.iters,
            "warm_start": True,
            "coarsest_scale": COARSEST_SCALE,
            "note": "No local EPE is available for Sintel test; upload the zip to the Sintel benchmark server.",
        },
        "passes": {},
    }
    meta_path = args.output_dir / "submission_metadata.json"
    try:
        for dstype in ("clean", "final"):
            payload["passes"][dstype] = _write_pass(
                model=model,
                sintel_root=args.sintel_root,
                dstype=dstype,
                output_dir=args.output_dir,
                device=device,
                iters=args.iters,
                log_every=args.log_every,
                max_pairs=args.max_pairs,
            )
            meta_path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    finally:
        datasets_un.get_extention = original_get_extention

    zip_path = None
    if not args.no_zip:
        zip_base = str(args.output_dir)
        zip_path = Path(shutil.make_archive(zip_base, "zip", root_dir=args.output_dir))
    payload["status"] = "PASSED"
    payload["zip_path"] = zip_path
    meta_path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_to_builtin(payload), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



