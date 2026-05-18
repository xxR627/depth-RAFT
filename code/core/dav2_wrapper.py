"""Frozen Depth-Anything-V2 wrapper used by Depth-RAFT.

The wrapper exposes DAv2 high-level features G and the DAv2 depth map Z while
keeping the external depth network frozen and out of the trainable state.

Implementation notes
--------------------
- The underlying DAv2 model is intentionally stored as an *unregistered* attribute via
  ``object.__setattr__``. This keeps its ~24.8M frozen parameters out of
  ``state_dict()`` and ``named_parameters()``.
- The wrapper itself has no trainable parameters and always keeps the DAv2 runtime in eval mode.
- ``precision="auto"`` follows the current local verification policy:
  CUDA -> fp16, CPU -> fp32. If bf16 is desired explicitly, pass ``precision="bf16"``.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import os
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PrecisionMode = Literal["auto", "fp16", "bf16", "fp32"]
PathLike = Union[str, os.PathLike]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _default_depth_anything_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "Depth-Anything-V2"


def _resolve_depth_anything_repo_root(explicit_root: Optional[PathLike]) -> Path:
    candidates = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root))

    env_root = os.environ.get("DEPTH_ANYTHING_V2_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(_default_depth_anything_repo_root())

    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        checked.append(str(candidate))
        if (candidate / "depth_anything_v2" / "dpt.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve Depth-Anything-V2 repo root. Checked:\n- " + "\n- ".join(checked)
    )


def _resolve_weights_path(explicit_path: PathLike) -> Path:
    candidate = Path(explicit_path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"DAv2 weights not found: {candidate}")
    return candidate


def _import_depth_anything(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from depth_anything_v2.dpt import DepthAnythingV2  # pylint: disable=import-outside-toplevel

    return DepthAnythingV2


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _patch_size_from_model(model: nn.Module) -> int:
    patch_size = getattr(model.pretrained, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(model.pretrained.patch_embed, "patch_size", None)
    if isinstance(patch_size, tuple):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError(f"Unexpected patch size tuple: {patch_size}")
        patch_size = patch_size[0]
    if not isinstance(patch_size, int):
        raise ValueError(f"Could not infer DAv2 patch size, got {patch_size!r}")
    return patch_size


def _next_multiple(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _resolve_runtime_device(explicit_device: Optional[Union[str, torch.device]]) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DAv2FeatureExtractor(nn.Module):
    """Frozen DAv2-S feature extractor that returns pooled G features at 1/16, 1/8 and 1/4.

    Parameters
    ----------
    weights_path:
        Path to the frozen DAv2-S checkpoint.
    encoder:
        DAv2 encoder name. The requested workflow currently targets ``"vits"``.
    depth_anything_repo_root:
        Optional override for the vendored ``Depth-Anything-V2`` runtime.
    precision:
        One of ``"auto"``, ``"fp16"``, ``"bf16"``, ``"fp32"``.
        ``"auto"`` resolves to fp16 on CUDA and fp32 on CPU.
    device:
        Optional initial runtime device. The hidden DAv2 runtime is moved lazily if forward inputs
        arrive on a different device later.

    Notes
    -----
    - The wrapper always runs DAv2 in eval mode and under ``torch.no_grad()``.
    - ``image`` must be ``[B, 3, H, W]`` in ``[0, 1]``.
    - G is captured with a ``forward_pre_hook`` on ``depth_head.scratch.output_conv2`` and then
      cropped back to the original valid ``H x W`` region before pooling.
    """

    def __init__(
        self,
        weights_path: PathLike,
        encoder: str = "vits",
        *,
        depth_anything_repo_root: Optional[PathLike] = None,
        precision: PrecisionMode = "auto",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder {encoder!r}. Expected one of {tuple(MODEL_CONFIGS)}.")

        self.weights_path = str(_resolve_weights_path(weights_path))
        self.encoder = encoder
        self.precision = precision
        self.depth_anything_repo_root = str(
            _resolve_depth_anything_repo_root(depth_anything_repo_root)
        )
        self.initial_device = _resolve_runtime_device(device)

        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        dav2_model = self._build_hidden_dav2()
        object.__setattr__(self, "_dav2", dav2_model)
        object.__setattr__(self, "_g_holder", {})
        object.__setattr__(
            self,
            "_g_hook_handle",
            dav2_model.depth_head.scratch.output_conv2.register_forward_pre_hook(self._capture_g_pre_hook),
        )

        self.patch_size = _patch_size_from_model(self._dav2)
        self.runtime_device = next(self._dav2.parameters()).device
        self.runtime_dtype = next(self._dav2.parameters()).dtype
        self.actual_precision = self._dtype_to_precision_name(self.runtime_dtype)

    def _build_hidden_dav2(self) -> nn.Module:
        DepthAnythingV2 = _import_depth_anything(Path(self.depth_anything_repo_root))
        dav2_model = DepthAnythingV2(**MODEL_CONFIGS[self.encoder])

        state_dict = torch.load(self.weights_path, map_location="cpu")
        dav2_model.load_state_dict(state_dict, strict=True)
        _freeze_module(dav2_model)
        dav2_model.eval()

        runtime_dtype = self._target_dtype_for_device(self.initial_device)
        dav2_model = dav2_model.to(device=self.initial_device, dtype=runtime_dtype)
        dav2_model.eval()
        return dav2_model

    def _target_dtype_for_device(self, device: torch.device) -> torch.dtype:
        if self.precision == "auto":
            return torch.float16 if device.type == "cuda" else torch.float32
        if self.precision == "fp16":
            if device.type != "cuda":
                raise RuntimeError("precision='fp16' requires a CUDA device.")
            return torch.float16
        if self.precision == "bf16":
            if device.type != "cuda":
                raise RuntimeError("precision='bf16' requires a CUDA device.")
            return torch.bfloat16
        if self.precision == "fp32":
            return torch.float32
        raise ValueError(f"Unsupported precision mode: {self.precision}")

    @staticmethod
    def _dtype_to_precision_name(dtype: torch.dtype) -> str:
        mapping = {
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float32: "fp32",
        }
        return mapping.get(dtype, str(dtype))

    def _capture_g_pre_hook(self, _module: nn.Module, inputs: Tuple[Tensor, ...]) -> None:
        self._g_holder["g"] = inputs[0].detach()

    def _refresh_runtime_state(self) -> None:
        parameter = next(self._dav2.parameters())
        self.runtime_device = parameter.device
        self.runtime_dtype = parameter.dtype
        self.actual_precision = self._dtype_to_precision_name(self.runtime_dtype)

    def _ensure_runtime(self, device: torch.device) -> None:
        target_dtype = self._target_dtype_for_device(device)
        if self.runtime_device != device or self.runtime_dtype != target_dtype:
            self._dav2.to(device=device, dtype=target_dtype)
            self._dav2.eval()
            self._refresh_runtime_state()

    def _autocast_context(self, device: torch.device):
        if device.type == "cuda" and self.runtime_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.runtime_dtype)
        return nullcontext()

    def _normalize_and_pad(self, image: Tensor) -> Tuple[Tensor, int, int]:
        if image.ndim != 4:
            raise ValueError(f"Expected image with shape [B, 3, H, W], got {tuple(image.shape)}.")
        if image.shape[1] != 3:
            raise ValueError(f"Expected image with 3 channels, got {image.shape[1]}.")

        image = image.to(device=self.runtime_device, dtype=torch.float32)
        mean = self._mean.to(device=self.runtime_device)
        std = self._std.to(device=self.runtime_device)
        image = (image - mean) / std

        _, _, height, width = image.shape
        padded_height = _next_multiple(height, self.patch_size)
        padded_width = _next_multiple(width, self.patch_size)
        pad_bottom = padded_height - height
        pad_right = padded_width - width

        if pad_bottom or pad_right:
            image = F.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")

        return image, height, width

    def _pool_scales(self, g: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        g_1_16 = F.avg_pool2d(g, kernel_size=16, stride=16)
        g_1_8 = F.avg_pool2d(g, kernel_size=8, stride=8)
        g_1_4 = F.avg_pool2d(g, kernel_size=4, stride=4)
        return g_1_16.float(), g_1_8.float(), g_1_4.float()

    def _extract_g_valid(self, image: Tensor) -> Tensor:
        self._ensure_runtime(image.device)
        normalized_padded, height, width = self._normalize_and_pad(image)
        self._g_holder.clear()

        with torch.no_grad(), self._autocast_context(self.runtime_device):
            _ = self._dav2(normalized_padded)

        g = self._g_holder.get("g")
        if g is None:
            raise RuntimeError(
                "Failed to capture DAv2 G from depth_head.scratch.output_conv2. Hook did not fire."
            )

        return g[..., :height, :width].contiguous().float()

    def _extract_depth_valid(self, image: Tensor) -> Tensor:
        self._ensure_runtime(image.device)
        normalized_padded, height, width = self._normalize_and_pad(image)

        with torch.no_grad(), self._autocast_context(self.runtime_device):
            depth = self._dav2(normalized_padded)

        if depth.ndim != 3:
            raise RuntimeError(f"Expected DAv2 depth output with shape [B, H, W], got {tuple(depth.shape)}.")

        return depth[:, None, :height, :width].contiguous().float()

    def train(self, mode: bool = True) -> "DAv2FeatureExtractor":
        del mode
        super().train(False)
        self._dav2.eval()
        return self

    def _apply(self, fn):
        super()._apply(fn)
        self._dav2._apply(fn)
        self._dav2.eval()
        self._refresh_runtime_state()
        return self

    def extra_repr(self) -> str:
        return (
            f"encoder={self.encoder!r}, precision={self.precision!r}, "
            f"actual_precision={self.actual_precision!r}, patch_size={self.patch_size}"
        )

    def extract_raw_g(self, image: Tensor) -> Tensor:
        """Extract valid-crop DAv2 G at the original image resolution.

        Parameters
        ----------
        image:
            Tensor of shape ``[B, 3, H, W]`` with values in ``[0, 1]``.

        Returns
        -------
        Tensor
            Float32 tensor of shape ``[B, 32, H, W]`` containing the valid-crop penultimate
            DAv2 high-level decoder feature map captured before ``scratch.output_conv2``.
        """

        return self._extract_g_valid(image)

    def extract_depth(self, image: Tensor) -> Tensor:
        """Extract valid-crop DAv2 depth at the original image resolution.

        Parameters
        ----------
        image:
            Tensor of shape ``[B, 3, H, W]`` with values in ``[0, 1]``.

        Returns
        -------
        Tensor
            Float32 tensor of shape ``[B, 1, H, W]`` containing the valid-crop DAv2 depth map Z.
        """

        return self._extract_depth_valid(image)

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract DAv2 G and return avg-pooled features at 1/16, 1/8, and 1/4 scales.

        Parameters
        ----------
        image:
            Tensor of shape ``[B, 3, H, W]`` with values in ``[0, 1]``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(g_1_16, g_1_8, g_1_4)`` in float32, where each tensor keeps 32 channels and uses
            floor-style spatial downsampling from ``avg_pool2d``.
        """
        return self._pool_scales(self.extract_raw_g(image))
