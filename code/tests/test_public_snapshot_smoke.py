from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))


def test_dav2_fnet_fusion_identity_initialization_cpu() -> None:
    from dav2_bottneck import DAv2FNetFusion

    fusion = DAv2FNetFusion([4, 8, 16])
    fnet_feats = (
        torch.randn(2, 4, 8, 8),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 16, 2, 2),
    )
    g_feats = (
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 2, 2),
    )

    outputs = fusion(fnet_feats, g_feats, freeze_fnet=True)

    assert len(outputs) == len(fnet_feats)
    for output, expected in zip(outputs, fnet_feats):
        assert output.shape == expected.shape
        assert torch.allclose(output, expected.float(), atol=1e-6, rtol=1e-6)


def test_dab_smooth_loss_cpu_smoke() -> None:
    from losses.loss_blocks import smooth_grad_1st_or_fusion, smooth_grad_2nd_or_fusion

    flow = torch.randn(2, 2, 16, 16)
    image = torch.rand(2, 3, 16, 16)
    depth = torch.rand(2, 1, 16, 16)

    loss_1st = smooth_grad_1st_or_fusion(flow, image, depth, alpha=10, beta=0.5)
    loss_2nd = smooth_grad_2nd_or_fusion(flow, image, depth, alpha=10, beta=0.5)

    assert loss_1st.ndim == 0
    assert loss_2nd.ndim == 0
    assert torch.isfinite(loss_1st)
    assert torch.isfinite(loss_2nd)


def test_vendored_dav2_runtime_present() -> None:
    from dav2_wrapper import _default_depth_anything_repo_root

    root = _default_depth_anything_repo_root()
    assert (root / "depth_anything_v2" / "dpt.py").is_file()
