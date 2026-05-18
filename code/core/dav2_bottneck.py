"""DAv2 G-feature bottleneck and zero-init fusion layers for Sun-RAFT fnet injection."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DAv2BottNeck(nn.Module):
    """Per-scale BottNeck that projects DAv2 G features from 32 channels into a learned residual branch."""

    def __init__(self, in_channels: int = 32, hidden_channels: int = 64, out_channels: int = 32) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

    def forward(self, g: Tensor) -> Tensor:
        return self.block(g)


class DAv2FNetFusion(nn.Module):
    """Fuse three-scale DAv2 G features into Sun-RAFT fnet outputs with identity-initialized 1x1 projectors.

    The projectors are initialized as:
    - identity on the original fnet channels
    - zero on the G branch
    - zero bias

    This guarantees that the fusion path is functionally equivalent to the baseline fnet at initialization.
    """

    def __init__(self, fnet_channels_per_scale: Sequence[int], g_out_channels: int = 32) -> None:
        super().__init__()
        self.fnet_channels_per_scale = list(fnet_channels_per_scale)
        self.g_out_channels = g_out_channels

        self.bottnecks = nn.ModuleList(
            [
                DAv2BottNeck(in_channels=32, hidden_channels=64, out_channels=g_out_channels)
                for _ in self.fnet_channels_per_scale
            ]
        )
        self.projectors = nn.ModuleList(
            [
                nn.Conv2d(channels + g_out_channels, channels, kernel_size=1, bias=True)
                for channels in self.fnet_channels_per_scale
            ]
        )
        self._init_projectors_as_identity()

    def _init_projectors_as_identity(self) -> None:
        for projector, fnet_channels in zip(self.projectors, self.fnet_channels_per_scale):
            with torch.no_grad():
                projector.weight.zero_()
                for channel_idx in range(fnet_channels):
                    projector.weight[channel_idx, channel_idx, 0, 0] = 1.0
                projector.bias.zero_()

    def _fuse_single_scale(
        self,
        scale_index: int,
        fnet_feat: Tensor,
        g_feat: Tensor,
        freeze_fnet: bool = True,
    ) -> Tensor:
        if freeze_fnet:
            fnet_feat = fnet_feat.detach()
        if g_feat.shape[-2:] != fnet_feat.shape[-2:]:
            g_feat = F.interpolate(g_feat, size=fnet_feat.shape[-2:], mode="bilinear", align_corners=False)

        g_projected = self.bottnecks[scale_index](g_feat.float())
        fused = torch.cat([fnet_feat.float(), g_projected], dim=1)
        return self.projectors[scale_index](fused)

    def forward(
        self,
        fnet_feats: Sequence[Tensor],
        g_feats: Sequence[Tensor],
        freeze_fnet: bool = True,
    ) -> tuple[Tensor, ...]:
        outputs = []
        for scale_index, (fnet_feat, g_feat) in enumerate(zip(fnet_feats, g_feats)):
            outputs.append(self._fuse_single_scale(scale_index, fnet_feat, g_feat, freeze_fnet=freeze_fnet))
        return tuple(outputs)

    def forward_paired(
        self,
        fnet_feats: Sequence[tuple[Tensor, Tensor]],
        g_feats: Sequence[tuple[Tensor, Tensor]],
        freeze_fnet: bool = True,
    ):
        outputs = []
        for scale_index, ((fmap1, fmap2), (g1, g2)) in enumerate(zip(fnet_feats, g_feats)):
            outputs.append(
                (
                    self._fuse_single_scale(scale_index, fmap1, g1, freeze_fnet=freeze_fnet),
                    self._fuse_single_scale(scale_index, fmap2, g2, freeze_fnet=freeze_fnet),
                )
            )
        return type(fnet_feats)(outputs)
