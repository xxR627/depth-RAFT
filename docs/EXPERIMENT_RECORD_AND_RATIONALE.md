# Method Record

## Final Method

Depth-RAFT adds frozen DAv2 geometry priors to Sun-RAFT without changing the recurrent update block, correlation layer, or optical-flow head.

- `G` enhances the matching feature stream.
- `Z` enhances the context encoder input.
- `DAB-Smooth` uses normalized RGB and depth gradients to reduce smoothing across likely boundaries.

The final checkpoint is `checkpoints/depth_raft_g_z_dab_step35000_best.pth`.

## Implementation

`G` is extracted from frozen DAv2 intermediate features. It is passed through a bottleneck and fused with fnet features by `DAv2FNetFusion`. The projector is initialized so the RGB fnet feature channels are preserved and the G branch starts at zero contribution.

`Z` is extracted from the frozen DAv2 depth head. The cnet first convolution is expanded from RGB to RGB+Z. The new Z-channel weights are zero-initialized, so loading a Sun-RAFT checkpoint remains identity-safe at initialization.

DAB-Smooth is implemented in `code/core/losses/loss_blocks.py` through OR-style fusion of RGB-gradient and Z-gradient boundary weights. The training script passes DAv2 depth maps into `FlowLoss`, which falls back to normal smoothness if no depth is provided.

## Training Recipe

The final Sintel stage uses:

- Baseline: `checkpoints/sun_raft_sintel_baseline.pth`
- Initialization: `checkpoints/depth_raft_g_z_chairs105k_start.pth`
- DAv2 weights: `checkpoints/depth_anything_v2_vits_frozen.pth`
- Dataset: Sintel unsupervised training split
- Steps: 35K
- Batch size: 3
- Trainable modules: G fusion and cnet
- Frozen modules: DAv2, fnet, update block, and flow head
- DAB beta: 0.5

## Reported Metrics

| Method | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
| --- | ---: | ---: | ---: | ---: |
| Sun-RAFT baseline | 1.6942 | 2.5935 | 4.7653 | 12.6358 |
| Depth-RAFT G+Z+DAB | 1.4970 | 2.4693 | 3.9854 | 11.6720 |

Official Sintel Clean split:

| Method | Matched EPE | Unmatched EPE |
| --- | ---: | ---: |
| Sun-RAFT baseline | 0.8182 | 12.7358 |
| Depth-RAFT G+Z+DAB | 0.7100 | 11.5416 |

## Ablation Summary

| Setting | Sintel Clean | Sintel Final | Matched Clean | Unmatched Clean |
| --- | ---: | ---: | ---: | ---: |
| Sun-RAFT baseline | 1.6942 | 2.5935 | 0.8182 | 12.7358 |
| + G-only | 1.6268 | 2.5597 | 0.7694 | 12.4172 |
| + Z-only | 1.5833 | 2.5849 | 0.7514 | 12.0456 |
| + G+Z | 1.5266 | 2.5041 | 0.7202 | 11.6700 |
| + G+Z+DAB | 1.4970 | 2.4693 | 0.7100 | 11.5416 |

The final method improves both matched and unmatched regions. It should be described as a geometry-prior improvement to unsupervised optical flow, not as a complete solution to unmatched or occluded pixels.
