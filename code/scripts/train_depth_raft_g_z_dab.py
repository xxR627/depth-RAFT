#!/usr/bin/env python
"""Train the final Depth-RAFT model: Sun-RAFT + G + Z + DAB-Smooth."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent
CORE_ROOT = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import datasets_un
import evaluate
from config.config_loader import load_json_config
from losses.flow_loss import FlowLoss
from raft import RAFT, adapt_state_dict_for_cnet_depth_input, allowed_missing_checkpoint_keys
from utils.ar_augmentor import FlowAugmentor
from utils.warp_utils import get_guassian_consistency_mask, get_occu_mask_backward, get_occu_mask_bidirection

try:
    from torch.cuda.amp import GradScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("GradScaler is required for this training script.") from exc


DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_CONFIG = REPO_ROOT / "config" / "Sun-RAFT_chairs-sintel.json"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_ROOT = Path(r"G:\flow_data\sintel")
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "runs" / "depth_raft_g_z_dab"
DEFAULT_PHASE_INDEX = 0

TRAINABLE_MODULES = ("fusion", "cnet")
DEBUG_MODULE_NAME_ORDER = TRAINABLE_MODULES + ("fnet", "flow_head", "update_block")


def _match_fusion(name: str) -> bool:
    return name.startswith("dav2_fusion.")


def _match_cnet(name: str) -> bool:
    return name.startswith("cnet.")


def _match_fnet(name: str) -> bool:
    return name.startswith("fnet.")


def _match_flow_head(name: str) -> bool:
    return name.startswith("update_block.flow_head.")


def _match_update_block(name: str) -> bool:
    return name.startswith("update_block.") and not _match_flow_head(name)


MODULE_MATCHERS: dict[str, Callable[[str], bool]] = {
    "fusion": _match_fusion,
    "cnet": _match_cnet,
    "fnet": _match_fnet,
    "flow_head": _match_flow_head,
    "update_block": _match_update_block,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dav2_weights", type=Path, default=DEFAULT_DAV2_WEIGHTS)
    parser.add_argument("--sintel_root", type=Path, default=DEFAULT_SINTEL_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_steps", type=int, default=35000)
    parser.add_argument(
        "--schedule_total_steps",
        type=int,
        default=None,
        help="Optional original schedule length when continuing a segmented run.",
    )
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_config",
        type=str,
        default="",
        help='Optional JSON such as \'{"fusion": 1e-4, "cnet": 1e-4}\'.',
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pct_start", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument(
        "--save_steps",
        type=str,
        default="",
        help="Comma-separated global steps to save in addition to --save_every.",
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_dataset", type=str, default="sintel_un")
    parser.add_argument("--phase_index", type=int, default=DEFAULT_PHASE_INDEX)
    parser.add_argument("--beta_smoothness", type=float, default=0.5)
    parser.add_argument(
        "--global_step_offset",
        type=int,
        default=0,
        help="Step offset used for checkpoint naming/logging when continuing a run.",
    )
    parser.add_argument(
        "--init_trainable_checkpoint",
        type=Path,
        default=None,
        help="Optional Depth-RAFT checkpoint used to initialize trainable G/cnet weights.",
    )
    parser.add_argument(
        "--resume_training_checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume trainable weights plus optimizer/scheduler state.",
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="train_metrics.csv",
        help="Metrics CSV filename inside output_dir.",
    )
    parser.add_argument("--skip_final_eval", action="store_true")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    return device


def _parse_step_set(raw_steps: str, *, total_steps: int) -> set[int]:
    if not raw_steps.strip():
        return set()

    parsed: set[int] = set()
    for chunk in raw_steps.split(","):
        value = chunk.strip()
        if not value:
            continue
        step = int(value)
        if step <= 0:
            raise ValueError(f"Checkpoint save step must be positive, got: {step}")
        if step > total_steps:
            raise ValueError(f"Checkpoint save step {step} exceeds total_steps={total_steps}")
        parsed.add(step)
    return parsed


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _patch_sintel_root(sintel_root: Path) -> None:
    root_prefix = str(sintel_root.resolve().parent) + os.sep
    datasets_un.get_extention = lambda prefix=root_prefix: prefix
    evaluate.datasets_un.get_extention = lambda prefix=root_prefix: prefix


def _load_config(config_path: Path, args: argparse.Namespace) -> dict:
    config = load_json_config(str(config_path))
    config["gpus"] = [0]
    config["mixed_precision"] = True
    phase = args.phase_index
    config["train"]["dataset"][phase] = args.train_dataset
    config["train"]["batch_size"][phase] = args.batch_size
    config["train"]["num_steps"][phase] = args.total_steps
    config["clip"] = args.grad_clip
    return config


def _build_model(config: dict, args: argparse.Namespace, device: torch.device) -> RAFT:
    model = RAFT(
        config,
        use_depth_raft=True,
        dav2_weights_path=str(args.dav2_weights),
        use_gru_checkpointing=True,
    ).to(device)
    model.train()
    model.freeze_bn()
    return model


def _load_baseline_checkpoint(model: RAFT, checkpoint_path: Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = _strip_module_prefix(state_dict)
    state_dict = adapt_state_dict_for_cnet_depth_input(state_dict, model.state_dict())
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = allowed_missing_checkpoint_keys(model)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("dav2_fusion.") and key not in allowed_missing
    ]
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading baseline checkpoint: {incompatible.unexpected_keys}")
    if invalid_missing:
        raise RuntimeError(f"Non-fusion missing keys while loading baseline checkpoint: {invalid_missing}")
    print(f"baseline_checkpoint_loaded={checkpoint_path}")
    print(f"baseline_missing_keys={incompatible.missing_keys}")


def _parse_lr_config(raw_lr_config: str, *, default_lr: float) -> dict[str, float]:
    if not raw_lr_config.strip():
        return {module_name: float(default_lr) for module_name in TRAINABLE_MODULES}

    try:
        parsed = json.loads(raw_lr_config)
    except json.JSONDecodeError as exc:
        relaxed_json = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', raw_lr_config.strip())
        try:
            parsed = json.loads(relaxed_json)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw_lr_config)
            except (SyntaxError, ValueError) as literal_exc:
                raise ValueError(f"Invalid --lr_config JSON: {exc}") from literal_exc

    if not isinstance(parsed, dict):
        raise ValueError("--lr_config must decode to a JSON object.")

    lr_config: dict[str, float] = {}
    for module_name, lr in parsed.items():
        if module_name not in TRAINABLE_MODULES:
            raise ValueError(f"Unsupported trainable module '{module_name}'. Use one of {TRAINABLE_MODULES}.")
        lr_value = float(lr)
        if lr_value <= 0:
            raise ValueError(f"Learning rate for module '{module_name}' must be > 0, got {lr_value}")
        lr_config[module_name] = lr_value

    return lr_config


def _module_param_names(model: RAFT, module_name: str) -> list[str]:
    matcher = MODULE_MATCHERS[module_name]
    return [name for name, _parameter in model.named_parameters() if matcher(name)]


def _freeze_params(model: RAFT, lr_config: dict[str, float]) -> tuple[int, tuple[str, ...]]:
    trainable_modules = tuple(lr_config.keys())
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(MODULE_MATCHERS[module_name](name) for module_name in trainable_modules)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable_modules={trainable_modules}")
    print(f"effective_lr_config={json.dumps(lr_config, sort_keys=True)}")
    print("use_depth_raft=True")
    print(f"trainable_params={trainable}")
    for module_name in DEBUG_MODULE_NAME_ORDER:
        module_params = _module_param_names(model, module_name)
        trainable_count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if MODULE_MATCHERS[module_name](name) and parameter.requires_grad
        )
        total_count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if MODULE_MATCHERS[module_name](name)
        )
        if not module_params:
            print(f"module_status {module_name}: missing")
            continue
        status = "trainable" if module_name in trainable_modules else "frozen"
        print(f"module_status {module_name}: {status} trainable_params={trainable_count} total_params={total_count}")
    return trainable, trainable_modules


def build_param_groups(model: RAFT, lr_config: dict[str, float]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    seen_param_ids: set[int] = set()
    for module_name, lr in lr_config.items():
        params = [
            parameter
            for name, parameter in model.named_parameters()
            if MODULE_MATCHERS[module_name](name) and parameter.requires_grad
        ]
        deduped_params = []
        for parameter in params:
            param_id = id(parameter)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            deduped_params.append(parameter)
        if deduped_params:
            groups.append({"params": deduped_params, "lr": float(lr), "name": module_name})
    return groups


def _load_trainable_state(model: RAFT, checkpoint_path: Path, label: str) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "trainable_state" not in checkpoint:
        raise RuntimeError(f"Unsupported {label} checkpoint format: {checkpoint_path}")

    model_state = model.state_dict()
    missing = []
    for key, value in checkpoint["trainable_state"].items():
        if key not in model_state:
            missing.append(key)
            continue
        model_state[key] = value
    if missing:
        raise RuntimeError(f"{label} trainable state keys not present in model: {missing}")

    incompatible = model.load_state_dict(model_state, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("dav2_fusion.") and key not in allowed_missing_checkpoint_keys(model)
    ]
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading {label} checkpoint: {incompatible.unexpected_keys}")
    if invalid_missing:
        raise RuntimeError(f"Invalid missing keys while loading {label} checkpoint: {invalid_missing}")
    print(f"{label}_checkpoint_loaded={checkpoint_path}")


def _load_resume_optimizer_scheduler(
    checkpoint: dict,
    checkpoint_path: Path,
    optimizer: AdamW,
    scheduler: OneCycleLR,
) -> None:
    if "optimizer_state" not in checkpoint or "scheduler_state" not in checkpoint:
        raise RuntimeError(f"Resume checkpoint missing optimizer/scheduler state: {checkpoint_path}")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    print(f"resume_optimizer_scheduler_loaded={checkpoint_path}")


def _build_loader(config: dict, phase: int):
    loader, dataset_len = datasets_un.fetch_dataloader(config, phase)
    print(f"train_dataset_pairs={dataset_len} dataset={config['train']['dataset'][phase]} phase_index={phase}")
    return loader


def _cycle_dataloader(loader):
    while True:
        for batch in loader:
            yield batch


def _current_flow_loss_weight(config: dict, step: int, total_steps: int, phase: int) -> float:
    if not config["train"]["loss"]["ar"]:
        return 0.0
    activation_start = (config["train"]["loss"]["ar_start"][phase] / 100.0) * total_steps
    increase_end = activation_start + (config["train"]["loss"]["ar_increasing"] / 100.0) * total_steps
    target = config["train"]["loss"]["ar_weight"][phase]
    if step < activation_start:
        return 0.0
    if step < increase_end:
        return (step - activation_start) * target / max(increase_end - activation_start, 1.0)
    return target


def _extract_dab_depths(depth_extractor, image1: torch.Tensor, image2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        depth_fw_bw = depth_extractor.extract_depth(
            torch.cat((image1, image2), dim=0).float().div(255.0).clamp(0.0, 1.0)
        )
    g_holder = getattr(depth_extractor, "_g_holder", None)
    if isinstance(g_holder, dict):
        g_holder.clear()
    batch_size = image1.shape[0]
    return torch.split(depth_fw_bw, [batch_size, batch_size], dim=0)


def _training_step(
    config: dict,
    model: RAFT,
    batch,
    optimizer: AdamW,
    phase: int,
    scaler: GradScaler,
    loss_module: FlowLoss,
    cur_step: int,
    ar_augmentor,
    flow_loss_current_weight: float,
    *,
    depth_smooth_extractor,
    beta_smoothness: float,
    device: torch.device,
):
    iterations = config["train"]["iters"]
    image1, image2, img1_ph, img2_ph, full_img_dict = batch
    del full_img_dict
    image1, image2 = image1.to(device), image2.to(device)
    img1_ph, img2_ph = img1_ph.to(device), img2_ph.to(device)

    depth1_smooth, depth2_smooth = _extract_dab_depths(depth_smooth_extractor, image1, image2)
    flow_predictions, flow_bw_predictions = model(image1=img1_ph, image2=img2_ph, iters=iterations, bw=config["bw"])

    if config["occ_method"][phase] == "brox":
        occlusions = get_occu_mask_bidirection(flow_predictions, flow_bw_predictions, mask_out=config["mask_out"])
        occlusions_bw = get_occu_mask_bidirection(flow_bw_predictions, flow_predictions, mask_out=config["mask_out"])
    elif config["occ_method"][phase] == "wan":
        occlusions = get_occu_mask_backward(
            flow_predictions,
            flow_bw_predictions,
            mask_out=config["mask_out"],
            detach=config["detach"],
        )
        occlusions_bw = get_occu_mask_backward(
            flow_bw_predictions,
            flow_predictions,
            mask_out=config["mask_out"],
            detach=config["detach"],
        )
    else:
        raise ValueError(f"Unsupported occ_method: {config['occ_method'][phase]}")

    if config["train"]["loss"]["ar"]:
        with torch.no_grad():
            flow_predictions_teacher, flow_bw_predictions_teacher = model(
                image1=image1,
                image2=image2,
                iters=iterations,
                bw=config["bw"],
            )
        if config["teacher_student_masking"]:
            teacher_mask_fw = get_guassian_consistency_mask(
                flow=flow_predictions_teacher[-1],
                flow_bw=flow_bw_predictions_teacher[-1],
                sigma=0.003,
            )
            teacher_mask_bw = get_guassian_consistency_mask(
                flow=flow_bw_predictions_teacher[-1],
                flow_bw=flow_predictions_teacher[-1],
                sigma=0.003,
            )
            img1_img2_aug, img2_img1_aug, flow_fw_flow_bw_truth, aug_teacher_mask = ar_augmentor(
                torch.cat((img1_ph, img2_ph), dim=0),
                torch.cat((img2_ph, img1_ph), dim=0),
                torch.cat((flow_predictions_teacher[-1], flow_bw_predictions_teacher[-1]), dim=0),
                step=cur_step,
                teacher_mask=torch.cat((teacher_mask_fw, teacher_mask_bw), dim=0),
            )
        else:
            img1_img2_aug, img2_img1_aug, flow_fw_flow_bw_truth = ar_augmentor(
                torch.cat((img1_ph, img2_ph), dim=0),
                torch.cat((img2_ph, img1_ph), dim=0),
                torch.cat((flow_predictions_teacher[-1], flow_bw_predictions_teacher[-1]), dim=0),
                step=cur_step,
                teacher_mask=None,
            )
            aug_teacher_mask = None
        flow_predictions_of_aug_imgs = model(
            image1=img1_img2_aug,
            image2=img2_img1_aug,
            iters=iterations,
            bw=False,
        )
    else:
        flow_predictions_of_aug_imgs, flow_fw_flow_bw_truth, aug_teacher_mask = None, None, None

    batch_size = image1.shape[0]
    metrics = loss_module(
        flow12_list=flow_predictions,
        flow21_list=flow_bw_predictions,
        occ12_list=occlusions,
        occ21_list=occlusions_bw,
        img1=image1,
        img2=image2,
        flow_predictions_of_aug_imgs=flow_predictions_of_aug_imgs,
        flow_gt=flow_fw_flow_bw_truth,
        teacher_student_masking=config["teacher_student_masking"],
        teacher_mask=aug_teacher_mask,
        flow_loss_current_weight=flow_loss_current_weight,
        depth1_smooth=depth1_smooth,
        depth2_smooth=depth2_smooth,
        beta_smoothness=beta_smoothness,
    )
    metric_names = ["total_loss", "L_ph", "L_sm", "flow_loss", "epe_to_semi_gt"]
    metrics = {k: v.sum() / batch_size for k, v in zip(metric_names, metrics)}
    metrics["flow_loss_weight"] = flow_loss_current_weight

    scaler.scale(metrics["total_loss"]).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
    scaler.step(optimizer)
    scaler.update()
    return metrics


def _make_checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"step_{step}.pth"


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: RAFT,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    args: argparse.Namespace,
    trainable_modules: tuple[str, ...],
    global_step: int,
) -> Path:
    checkpoint_path = _make_checkpoint_path(output_dir, global_step)
    trainable_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if any(MODULE_MATCHERS[module_name](key) for module_name in trainable_modules)
    }
    saved_args = dict(vars(args))
    saved_args.update({"use_depth_raft": True, "use_dab_smoothness": True})
    payload = {
        "step": global_step,
        "local_step": step,
        "method": "Depth-RAFT-G-Z-DAB",
        "trainable_modules": trainable_modules,
        "trainable_state": trainable_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": saved_args,
    }
    torch.save(payload, checkpoint_path)
    print(f"checkpoint_saved={checkpoint_path}")
    return checkpoint_path


def _write_metrics_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "step",
            "total_loss",
            "L_ph",
            "L_sm",
            "flow_loss",
            "epe_to_semi_gt",
            "flow_loss_weight",
            "lr",
            "lr_groups_json",
        ])


def _should_save_checkpoint(step: int, *, total_steps: int, save_every: int, save_steps: Iterable[int]) -> bool:
    if step == total_steps:
        return True
    if save_every > 0 and step % save_every == 0:
        return True
    return step in save_steps


def _format_lr_groups(optimizer: AdamW) -> dict[str, float]:
    formatted: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        formatted[name] = float(group["lr"])
    return formatted


def _append_metrics(path: Path, step: int, metrics: dict[str, float], lr: float, lr_groups: dict[str, float]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            step,
            metrics["total_loss"],
            metrics["L_ph"],
            metrics["L_sm"],
            metrics["flow_loss"],
            metrics["epe_to_semi_gt"],
            metrics["flow_loss_weight"],
            lr,
            json.dumps(lr_groups, sort_keys=True),
        ])


def _capture_module_snapshots(model: RAFT) -> dict[str, tuple[str, torch.Tensor]]:
    snapshots: dict[str, tuple[str, torch.Tensor]] = {}
    for module_name in DEBUG_MODULE_NAME_ORDER:
        for name, parameter in model.named_parameters():
            if MODULE_MATCHERS[module_name](name):
                snapshots[module_name] = (name, parameter.detach().cpu().clone())
                break
    return snapshots


def _print_module_update_summary(model: RAFT, before: dict[str, tuple[str, torch.Tensor]]) -> None:
    for module_name in DEBUG_MODULE_NAME_ORDER:
        if module_name not in before:
            print(f"param_update_check module={module_name} status=missing")
            continue
        name_before, tensor_before = before[module_name]
        current_tensor = dict(model.named_parameters())[name_before].detach().cpu()
        max_abs_delta = float((current_tensor - tensor_before).abs().max().item())
        changed = max_abs_delta > 0.0
        requires_grad = bool(dict(model.named_parameters())[name_before].requires_grad)
        print(
            f"param_update_check module={module_name} name={name_before} "
            f"requires_grad={requires_grad} changed={changed} max_abs_delta={max_abs_delta:.8f}"
        )


def _to_float_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            output[key] = float(value.detach().item())
        else:
            output[key] = float(value)
    return output


def _run_final_eval(model: RAFT, output_dir: Path, step: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        results = evaluate.validate_sintel(model, f"depth_raft_g_z_dab_step_{step}", iters=12)
    payload = {"clean_epe": float(results["clean"]), "final_epe": float(results["final"])}
    final_eval_path = output_dir / "final_eval.json"
    final_eval_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"final_eval={payload}")
    model.train()
    model.freeze_bn()
    return payload


def main() -> int:
    args = parse_args()
    args.baseline_ckpt = args.baseline_ckpt.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.dav2_weights = args.dav2_weights.expanduser().resolve()
    args.sintel_root = args.sintel_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.init_trainable_checkpoint is not None:
        args.init_trainable_checkpoint = args.init_trainable_checkpoint.expanduser().resolve()
    if args.resume_training_checkpoint is not None:
        args.resume_training_checkpoint = args.resume_training_checkpoint.expanduser().resolve()

    if args.resume_training_checkpoint is not None and args.init_trainable_checkpoint is not None:
        raise ValueError("--resume_training_checkpoint and --init_trainable_checkpoint are mutually exclusive.")
    if not args.baseline_ckpt.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_ckpt}")
    if not args.dav2_weights.is_file():
        raise FileNotFoundError(f"DAv2 weights not found: {args.dav2_weights}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not (args.sintel_root / "test" / "clean").is_dir():
        raise FileNotFoundError(f"Sintel test split not found under: {args.sintel_root}")
    if args.init_trainable_checkpoint is not None and not args.init_trainable_checkpoint.is_file():
        raise FileNotFoundError(f"Init trainable checkpoint not found: {args.init_trainable_checkpoint}")
    if args.resume_training_checkpoint is not None and not args.resume_training_checkpoint.is_file():
        raise FileNotFoundError(f"Resume training checkpoint not found: {args.resume_training_checkpoint}")
    if args.global_step_offset < 0:
        raise ValueError(f"global_step_offset must be >= 0, got {args.global_step_offset}")
    if args.schedule_total_steps is not None and args.schedule_total_steps <= 0:
        raise ValueError(f"schedule_total_steps must be positive, got {args.schedule_total_steps}")
    if args.phase_index < 0:
        raise ValueError(f"phase_index must be >= 0, got {args.phase_index}")
    if args.beta_smoothness < 0:
        raise ValueError(f"beta_smoothness must be >= 0, got {args.beta_smoothness}")
    if Path(args.metrics_filename).name != args.metrics_filename:
        raise ValueError(f"metrics_filename must be a plain filename, got: {args.metrics_filename}")

    resume_checkpoint = None
    if args.resume_training_checkpoint is not None:
        resume_checkpoint = torch.load(str(args.resume_training_checkpoint), map_location="cpu")
        if not isinstance(resume_checkpoint, dict) or "step" not in resume_checkpoint:
            raise RuntimeError(f"Unsupported resume checkpoint format: {args.resume_training_checkpoint}")
        resume_step = int(resume_checkpoint["step"])
        if args.global_step_offset not in (0, resume_step):
            raise ValueError(
                f"--global_step_offset={args.global_step_offset} does not match resume checkpoint step={resume_step}."
            )
        args.global_step_offset = resume_step

    os.chdir(REPO_ROOT)
    device = _resolve_device(args.device)
    _patch_sintel_root(args.sintel_root)
    config = _load_config(args.config, args)
    save_steps = _parse_step_set(args.save_steps, total_steps=args.total_steps + args.global_step_offset)
    model = _build_model(config, args, device)
    _load_baseline_checkpoint(model, args.baseline_ckpt)
    if args.init_trainable_checkpoint is not None:
        _load_trainable_state(model, args.init_trainable_checkpoint, "init_trainable")
    if resume_checkpoint is not None:
        _load_trainable_state(model, args.resume_training_checkpoint, "resume_model_state")

    effective_lr_config = _parse_lr_config(args.lr_config, default_lr=args.lr)
    trainable_params, trainable_modules = _freeze_params(model, effective_lr_config)
    param_groups = build_param_groups(model, effective_lr_config)
    if not param_groups:
        raise RuntimeError("No trainable parameter groups were constructed.")
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, eps=config["epsilon"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[float(group["lr"]) for group in param_groups],
        total_steps=args.total_steps + 100,
        pct_start=args.pct_start,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    if resume_checkpoint is not None:
        _load_resume_optimizer_scheduler(resume_checkpoint, args.resume_training_checkpoint, optimizer, scheduler)

    scaler = GradScaler(enabled=config["mixed_precision"])
    phase = args.phase_index
    loss_module = FlowLoss(config["train"]["loss"], phase=phase).to(device)
    ar_augmentor = FlowAugmentor(**config["train"]["loss"]["aug_settings"][phase]) if config["train"]["loss"]["ar"] else None
    depth_smooth_extractor = model.dav2_extractor
    depth_smooth_extractor.eval()
    print(f"dab_smoothness_enabled=True beta_smoothness={args.beta_smoothness:.4f}")

    loader = _build_loader(config, phase)
    iterator = _cycle_dataloader(loader)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / args.metrics_filename
    _write_metrics_header(metrics_csv)

    print(
        f"training_start total_steps={args.total_steps} global_step_offset={args.global_step_offset} "
        f"batch_size={args.batch_size} trainable_params={trainable_params}"
    )
    schedule_total_steps = args.schedule_total_steps or (args.global_step_offset + args.total_steps)
    print(f"schedule_total_steps={schedule_total_steps}")
    print(f"optimizer_group_names={[group['name'] for group in param_groups]}")
    print(f"save_every={args.save_every}")
    print(f"save_steps={sorted(save_steps)}")
    start_time = time.time()
    param_snapshots = _capture_module_snapshots(model)

    last_metrics: dict[str, float] | None = None
    for step in range(args.total_steps):
        global_step = args.global_step_offset + step + 1
        optimizer.zero_grad(set_to_none=True)
        batch = next(iterator)
        flow_loss_weight = _current_flow_loss_weight(config, global_step, schedule_total_steps, phase)
        metrics = _training_step(
            config,
            model,
            batch,
            optimizer,
            phase,
            scaler,
            loss_module,
            step,
            ar_augmentor,
            flow_loss_weight,
            depth_smooth_extractor=depth_smooth_extractor,
            beta_smoothness=args.beta_smoothness,
            device=device,
        )
        scheduler.step()

        last_metrics = _to_float_metrics(metrics)
        lr = float(scheduler.get_last_lr()[0])
        lr_groups = _format_lr_groups(optimizer)
        _append_metrics(metrics_csv, global_step, last_metrics, lr, lr_groups)

        if (step + 1) % args.log_every == 0 or step == 0:
            elapsed = time.time() - start_time
            print(
                f"step={global_step}/{args.global_step_offset + args.total_steps} "
                f"total={last_metrics['total_loss']:.4f} flow_loss={last_metrics['flow_loss']:.4f} "
                f"epe={last_metrics['epe_to_semi_gt']:.4f} lr={lr:.7f} elapsed_sec={elapsed:.1f}"
            )
        if global_step == args.global_step_offset + 1 or global_step % 500 == 0:
            print(f"lr_groups step={global_step} values={json.dumps(lr_groups, sort_keys=True)}")

        if _should_save_checkpoint(
            global_step,
            total_steps=args.global_step_offset + args.total_steps,
            save_every=args.save_every,
            save_steps=save_steps,
        ):
            _save_checkpoint(
                args.output_dir,
                step + 1,
                model,
                optimizer,
                scheduler,
                args,
                trainable_modules,
                global_step=global_step,
            )

    _print_module_update_summary(model, param_snapshots)
    if not args.skip_final_eval:
        _run_final_eval(model, args.output_dir, args.global_step_offset + args.total_steps)

    print(f"training_done output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

