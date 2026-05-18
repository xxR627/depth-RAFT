# Depth-RAFT

Depth-RAFT is a cleaned research package for the final Sun-RAFT + frozen DAv2 geometry-prior model.

Public method naming:

- `G`: frozen DAv2 high-level features injected into the matching feature stream.
- `Z`: frozen DAv2 depth map injected into the context encoder input.
- `DAB-Smooth`: depth-aware boundary smoothness regularizer.

The released code keeps only the final G+Z+DAB path and the evaluation/figure scripts used for the reported results. Exploratory branches, ablation runners, logs, caches, and temporary checkpoints have been moved out of this package.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | Model, losses, datasets, training, evaluation, and figure scripts. |
| `checkpoints/` | Sun-RAFT baselines, frozen DAv2 weights, Chairs-start, and final Depth-RAFT checkpoint. |
| `third_party/Depth-Anything-V2/` | Vendored DAv2 runtime code used by the wrapper. |
| `results/main/` | Final result tables and evaluation summaries. |
| `results/ablations/` | Clean module-ablation table used by the paper. |
| `figures/` | Paper figures and qualitative examples. |
| `docs/` | Manifest, reproduction commands, naming, and package audit. |

## Main Results

| Method | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
| --- | ---: | ---: | ---: | ---: |
| Sun-RAFT baseline | 1.6942 | 2.5935 | 4.7653 | 12.6358 |
| Depth-RAFT G+Z+DAB | 1.4970 | 2.4693 | 3.9854 | 11.6720 |

Official Sintel Clean matched/unmatched:

| Method | Matched | Unmatched |
| --- | ---: | ---: |
| Sun-RAFT baseline | 0.8182 | 12.7358 |
| Depth-RAFT G+Z+DAB | 0.7100 | 11.5416 |

## Setup

Install Python dependencies from `requirements.txt`. The DAv2 runtime code needed by this project is vendored under `third_party/Depth-Anything-V2/`; set `DEPTH_ANYTHING_V2_ROOT` only if you want to override it with another checkout. The frozen DAv2 weights are stored in `checkpoints/depth_anything_v2_vits_frozen.pth`.

Sintel and KITTI datasets are not included. Update the dataset paths in the commands under `docs/REPRO_COMMANDS.md`.

## Checkpoints

Large `.pth` files should be stored with Git LFS before pushing to GitHub.

| File | Purpose |
| --- | --- |
| `checkpoints/sun_raft_sintel_baseline.pth` | Sun-RAFT Sintel baseline. |
| `checkpoints/sun_raft_kitti_baseline.pth` | Sun-RAFT KITTI reference checkpoint. |
| `checkpoints/depth_anything_v2_vits_frozen.pth` | Frozen DAv2-S weights. |
| `checkpoints/depth_raft_g_z_chairs105k_start.pth` | Chairs-stage Depth-RAFT start for Sintel fine-tuning. |
| `checkpoints/depth_raft_g_z_dab_step35000_best.pth` | Final reported checkpoint. |

Start with `docs/MANIFEST.md` and `docs/REPRO_COMMANDS.md` for the runnable entry points.
