# Depth-RAFT

Depth-RAFT is a geometry-aware extension of Sun-RAFT for unsupervised optical flow. The central idea is to use frozen monocular depth from Depth Anything V2 (DAv2) as an external geometric prior when photometric supervision is unreliable, especially around unmatched or occluded pixels.

This public review snapshot releases the method-specific components needed to inspect and run the proposed contribution:

- the `DGE` module
- the `DAB-Smooth` loss
- evaluation, submission, figure-generation, and analysis scripts
- configuration files

## What This Repository Demonstrates

The manuscript's main technical claim is implemented in three places:

| Paper component | What it does | Code evidence |
| --- | --- | --- |
| DGE-G | Injects frozen DAv2 high-level geometric features into the RAFT matching feature stream before correlation construction. The fusion projector is identity-initialized so the public code can verify that the added branch does not disturb the baseline at initialization. | `code/core/dav2_bottneck.py`, `code/core/raft.py` |
| DGE-Z | Injects the frozen DAv2 depth map into the context encoder through a separate zero-initialized depth convolution, preserving the RGB checkpoint path while adding geometric context for recurrent refinement. | `code/core/extractor.py`, `code/core/raft.py` |
| DAB-Smooth | Replaces RGB-only edge-aware smoothness with a depth-aware boundary smoothness loss that combines RGB gradients and normalized depth gradients. | `code/core/losses/loss_blocks.py`, `code/core/losses/flow_loss.py` |

Public method naming used in the paper and code:

- `G`: frozen DAv2 high-level features injected into the matching feature stream
- `Z`: frozen DAv2 depth map injected into the context encoder input
- `DAB-Smooth`: depth-aware boundary smoothness regularizer

## Current Public Scope

| Path | Contents |
| --- | --- |
| `code/core/` | Depth-RAFT implementation, including DGE integration and DAB-Smooth loss. |
| `code/scripts/` | Evaluation, submission, figure-generation, and analysis utilities. |
| `code/config/` | Configuration files and config loader. |
| `code/tests/` | Lightweight implementation checks. |
| `third_party/Depth-Anything-V2/` | Vendored DAv2 runtime code used by the wrapper. |

This public snapshot does not include the full training pipeline, training assets, experimental logs, generated paper figures, result tables, or pretrained checkpoints.

## Release Note

The remaining codebase components, pretrained checkpoints, and the complete training and reproduction pipeline will be uploaded in full immediately after the manuscript is accepted.

## Setup

Recommended environment: Python 3.9 or 3.10 with PyTorch installed from the matching CUDA or CPU wheel for your machine.

Install Python dependencies from `requirements.txt`. The DAv2 runtime code needed by this project is vendored under `third_party/Depth-Anything-V2/`; set `DEPTH_ANYTHING_V2_ROOT` only if you want to override it with another checkout.

Datasets are not included. Update dataset paths directly in the evaluation commands you run.

## Reviewer Quick Check

After cloning the repository, reviewers can verify that the public method components import and execute on synthetic CPU tensors:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m pytest code/tests -q
```

Expected result:

```text
4 passed, 1 skipped
```

The lightweight smoke tests check the DGE-G identity initialization, DGE-Z zero initialization, DAB-Smooth loss, the vendored DAv2 runtime path, and the public component demo without requiring datasets or pretrained checkpoints.

To see the components directly, run:

```bash
python code/scripts/demo_depth_raft_components.py
```

Expected output contains:

```text
Depth-RAFT public component demo
- Vendored DAv2 runtime present: True
- DGE-G identity initialization max error: ...
- DGE-Z zero-initialized depth path max weight: 0.000e+00
- RGB-only smoothness loss: ...
- DAB-Smooth loss with RGB+depth boundaries: ...
Demo passed.
```

One optional equivalence test additionally verifies that the zero-initialized Depth-RAFT path matches the Sun-RAFT baseline on a real Sintel image pair. This optional test is skipped unless CUDA, checkpoints, and image paths are available. To run it, set:

```bash
export DEPTH_RAFT_BASELINE_CKPT=/path/to/sun_raft_sintel_baseline.pth
export DEPTH_RAFT_DAV2_WEIGHTS=/path/to/depth_anything_v2_vits_frozen.pth
export DEPTH_RAFT_SINTEL_IMAGE1=/path/to/frame_0001.png
export DEPTH_RAFT_SINTEL_IMAGE2=/path/to/frame_0002.png
python -m pytest code/tests/test_depth_raft_identity_equivalence.py -q
```

## Available Code

Representative public entry points:

- `code/scripts/demo_depth_raft_components.py`
- `code/scripts/eval_depth_raft_clean_final.py`
- `code/scripts/eval_depth_raft_region_decomp.py`
- `code/scripts/eval_paper_sintel_occ_noc.py`
- `code/scripts/eval_paper_kitti_occ_noc.py`
- `code/scripts/eval_sun_raft_region_decomp.py`
- `code/scripts/create_sintel_test_submission_depth_raft.py`
- `code/scripts/benchmark_model_costs.py`
- `code/scripts/make_figure3_sintel_qualitative_v2.py`
- `code/scripts/make_figure4_kitti_qualitative.py`
- `code/scripts/make_z_gradient_figure.py`

Core implementation files:

- `code/core/raft.py`
- `code/core/dav2_wrapper.py`
- `code/core/dav2_bottneck.py`
- `code/core/extractor.py`
- `code/core/losses/loss_blocks.py`
- `code/core/losses/flow_loss.py`

## Reported Results in the Manuscript

This public snapshot is intended to make the method inspectable during review. The manuscript reports the full experimental results, including:

| Setting | Sun-RAFT | Depth-RAFT |
| --- | ---: | ---: |
| Sintel Clean EPE | 1.69 | 1.50 |
| Sintel Clean unmatched-region EPE | 12.74 | 11.54 |
| Sintel Final EPE | 2.60 | 2.47 |
| Zero-shot KITTI 2015 EPE | 4.76 | 3.99 |

The full checkpoints, training assets, and complete reproduction pipeline will be released after acceptance as described above.
