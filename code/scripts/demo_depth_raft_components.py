from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = CODE_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from dav2_bottneck import DAv2FNetFusion
from dav2_wrapper import _default_depth_anything_repo_root
from extractor import BasicContextEncoder
from losses.loss_blocks import smooth_grad_1st, smooth_grad_1st_or_fusion


def demo_dge_g_identity() -> float:
    """Show that the G-branch starts as an identity-preserving fnet fusion."""

    torch.manual_seed(7)
    fusion = DAv2FNetFusion([8])
    fnet_feat = (torch.randn(1, 8, 8, 8),)
    g_feat = (torch.randn(1, 32, 8, 8),)

    fused = fusion(fnet_feat, g_feat, freeze_fnet=True)[0]
    return float((fused - fnet_feat[0]).abs().max().detach())


def demo_dge_z_zero_init() -> float:
    """Show that the Z input path is present and zero-initialized."""

    cnet = BasicContextEncoder(output_dim=16, norm_fn="group", in_channels=4)
    assert cnet.depth_conv1 is not None
    return float(cnet.depth_conv1.weight.abs().max().detach())


def demo_dab_smooth() -> tuple[float, float]:
    """Compare RGB-only smoothness and DAB-Smooth on synthetic tensors."""

    torch.manual_seed(11)
    flow = torch.randn(1, 2, 16, 16)
    image = torch.rand(1, 3, 16, 16)
    depth = torch.rand(1, 1, 16, 16)

    rgb_only = smooth_grad_1st(flow, image, alpha=10)
    dab = smooth_grad_1st_or_fusion(flow, image, depth, alpha=10, beta=0.5)
    return float(rgb_only), float(dab)


def main() -> None:
    dav2_root = _default_depth_anything_repo_root()
    dge_g_error = demo_dge_g_identity()
    dge_z_weight = demo_dge_z_zero_init()
    rgb_loss, dab_loss = demo_dab_smooth()

    print("Depth-RAFT public component demo")
    print(f"- Vendored DAv2 runtime present: {(dav2_root / 'depth_anything_v2' / 'dpt.py').is_file()}")
    print(f"- DGE-G identity initialization max error: {dge_g_error:.3e}")
    print(f"- DGE-Z zero-initialized depth path max weight: {dge_z_weight:.3e}")
    print(f"- RGB-only smoothness loss: {rgb_loss:.6f}")
    print(f"- DAB-Smooth loss with RGB+depth boundaries: {dab_loss:.6f}")

    if dge_g_error > 1e-6:
        raise SystemExit("DGE-G identity initialization check failed.")
    if dge_z_weight != 0.0:
        raise SystemExit("DGE-Z zero initialization check failed.")
    if not torch.isfinite(torch.tensor([rgb_loss, dab_loss])).all():
        raise SystemExit("DAB-Smooth finite-loss check failed.")

    print("Demo passed.")


if __name__ == "__main__":
    main()
