#!/usr/bin/env python
"""Evaluate a series of Depth-RAFT trainable checkpoints on Sintel clean/final."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from eval_depth_raft_clean_final import parse_args as _unused_parse  # noqa: F401
import eval_depth_raft_clean_final as clean_final_eval


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--dav2-weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel-root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="step_*.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=str, default="eval_log.csv")
    parser.add_argument("--output-json", type=str, default="eval_summary.json")
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


def _extract_step(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pth$", path.name)
    if not match:
        raise ValueError(f"Could not parse step from checkpoint name: {path.name}")
    return int(match.group(1))


def _evaluate_checkpoint(
    *,
    baseline_checkpoint: Path,
    trainable_checkpoint: Path,
    dav2_weights: Path,
    sintel_root: Path,
) -> dict[str, float]:
    patched_get_extention = clean_final_eval.baseline_eval._patched_get_extention_factory(sintel_root)
    original_datasets_get_extention = clean_final_eval.datasets_un.get_extention
    original_evaluate_get_extention = clean_final_eval.evaluate.datasets_un.get_extention
    clean_final_eval.datasets_un.get_extention = patched_get_extention
    clean_final_eval.evaluate.datasets_un.get_extention = patched_get_extention

    try:
        model = clean_final_eval.fusion_region_eval._build_model(
            device=clean_final_eval.baseline_eval._resolve_device(),
            baseline_checkpoint=baseline_checkpoint,
            fusion_checkpoint=trainable_checkpoint,
            dav2_weights=dav2_weights,
        )
        return clean_final_eval._evaluate_clean_final(model)
    finally:
        clean_final_eval.datasets_un.get_extention = original_datasets_get_extention
        clean_final_eval.evaluate.datasets_un.get_extention = original_evaluate_get_extention


def main() -> int:
    args = parse_args()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    checkpoints = sorted(args.checkpoint_dir.glob(args.glob), key=_extract_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.glob} under {args.checkpoint_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / args.output_csv
    json_path = args.output_dir / args.output_json

    rows: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "clean_epe", "final_epe", "checkpoint"])

        for checkpoint_path in checkpoints:
            step = _extract_step(checkpoint_path)
            metrics = _evaluate_checkpoint(
                baseline_checkpoint=args.baseline_checkpoint,
                trainable_checkpoint=checkpoint_path,
                dav2_weights=args.dav2_weights,
                sintel_root=args.sintel_root,
            )
            row = {
                "step": step,
                "clean_epe": float(metrics["clean_epe"]),
                "final_epe": float(metrics["final_epe"]),
                "checkpoint": checkpoint_path,
            }
            rows.append(row)
            writer.writerow([step, row["clean_epe"], row["final_epe"], str(checkpoint_path)])
            print(
                f"eval_done step={step} clean_epe={row['clean_epe']:.6f} "
                f"final_epe={row['final_epe']:.6f} checkpoint={checkpoint_path}"
            )

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "rows": rows,
        "best_clean": min(rows, key=lambda item: item["clean_epe"]),
        "best_final": min(rows, key=lambda item: item["final_epe"]),
    }
    json_path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
