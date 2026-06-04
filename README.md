# Depth-RAFT

This repository currently releases only the method-specific components needed to inspect the proposed approach:

- the `DGE` module
- the `DAB-Smooth` loss
- evaluation, submission, figure-generation, and analysis scripts
- configuration files

Public method naming:

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

Install Python dependencies from `requirements.txt`. The DAv2 runtime code needed by this project is vendored under `third_party/Depth-Anything-V2/`; set `DEPTH_ANYTHING_V2_ROOT` only if you want to override it with another checkout.

Datasets are not included. Update dataset paths directly in the evaluation commands you run.

## Available Code

Representative public entry points:

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
