from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from raft import RAFT, adapt_state_dict_for_cnet_depth_input
from utils.utils import InputPadder


PACKAGE_ROOT = REPO_ROOT.parent
DEFAULT_BASELINE_CKPT = PACKAGE_ROOT / "checkpoints" / "sun_raft_sintel_baseline.pth"
DEFAULT_DAV2_WEIGHTS = PACKAGE_ROOT / "checkpoints" / "depth_anything_v2_vits_frozen.pth"
DEFAULT_SINTEL_PAIR = (
    Path(r"G:\flow_data\sintel\training\clean\alley_1\frame_0001.png"),
    Path(r"G:\flow_data\sintel\training\clean\alley_1\frame_0002.png"),
)


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this equivalence test.")
    return torch.device("cuda")


def _strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _read_image_0_255(path: Path, device: torch.device) -> torch.Tensor:
    image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


def _load_baseline_state_dict():
    checkpoint = torch.load(str(DEFAULT_BASELINE_CKPT), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {DEFAULT_BASELINE_CKPT}")
    return _strip_module_prefix(state_dict)


def _build_baseline_model(device: torch.device) -> RAFT:
    model = RAFT(
        {"mixed_precision": True, "fnet_norm": "instance", "cnet_norm": "batch", "bw": False},
    ).to(device)
    model.load_state_dict(_load_baseline_state_dict(), strict=True)
    model.eval()
    return model


def _build_depth_model(device: torch.device) -> RAFT:
    model = RAFT(
        {"mixed_precision": True, "fnet_norm": "instance", "cnet_norm": "batch", "bw": False},
        use_depth_raft=True,
        dav2_weights_path=str(DEFAULT_DAV2_WEIGHTS),
    ).to(device)
    state_dict = adapt_state_dict_for_cnet_depth_input(_load_baseline_state_dict(), model.state_dict())
    incompatible = model.load_state_dict(state_dict, strict=False)
    assert incompatible.unexpected_keys == []
    assert "cnet.depth_conv1.weight" in incompatible.missing_keys
    assert all(key == "cnet.depth_conv1.weight" or key.startswith("dav2_fusion.") for key in incompatible.missing_keys)
    model.eval()
    return model


def test_depth_raft_zero_init_equivalence() -> None:
    device = _resolve_device()
    image1_path, image2_path = DEFAULT_SINTEL_PAIR
    if not image1_path.is_file() or not image2_path.is_file():
        raise FileNotFoundError(f"Sintel pair not found: {image1_path}, {image2_path}")
    if not DEFAULT_DAV2_WEIGHTS.is_file():
        raise FileNotFoundError(f"DAv2 weights not found: {DEFAULT_DAV2_WEIGHTS}")

    baseline = _build_baseline_model(device)
    depth_model = _build_depth_model(device)

    assert depth_model.cnet.depth_conv1 is not None
    assert float(depth_model.cnet.depth_conv1.weight.abs().max()) == 0.0

    image1 = _read_image_0_255(image1_path, device)
    image2 = _read_image_0_255(image2_path, device)
    padder = InputPadder(image1.shape, coarsest_scale=8)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        baseline_low, baseline_up = baseline(image1, image2, iters=12, test_mode=True, bw=False)
        depth_low, depth_up = depth_model(image1, image2, iters=12, test_mode=True, bw=False)

    assert torch.allclose(baseline_low.float(), depth_low.float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(baseline_up.float(), depth_up.float(), atol=1e-5, rtol=1e-5)
