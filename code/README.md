# Depth-RAFT Code

This public code folder is intentionally limited to the method components needed for inspection and evaluation.

Included:

- `DGE` implementation
- `DAB-Smooth` implementation
- evaluation scripts
- submission export script
- figure-generation and analysis utilities
- configuration files

Not included in this public snapshot:

- the full training pipeline
- pretrained checkpoints
- training logs and intermediate artifacts
- generated figures, result tables, and manuscript files

Public naming:

- `G`: frozen DAv2 high-level feature branch injected into fnet features
- `Z`: frozen DAv2 depth map injected into the context encoder input
- `DAB`: depth-aware boundary smoothness

## Paper-to-Code Map

| Paper component | Files | Minimal check |
| --- | --- | --- |
| DGE-G feature-side depth prior | `core/dav2_bottneck.py`, `core/raft.py` | `python code/scripts/demo_depth_raft_components.py` prints the identity-initialization error. |
| DGE-Z context-side depth prior | `core/extractor.py`, `core/raft.py` | The same demo prints the zero-initialized depth-path weight. |
| DAB-Smooth loss-side depth prior | `core/losses/loss_blocks.py`, `core/losses/flow_loss.py` | The same demo prints RGB-only and DAB-Smooth losses on synthetic tensors. |

Available scripts:

- `scripts/demo_depth_raft_components.py`
- `scripts/eval_depth_raft_clean_final.py`
- `scripts/eval_depth_raft_region_decomp.py`
- `scripts/eval_paper_sintel_occ_noc.py`
- `scripts/eval_paper_kitti_occ_noc.py`
- `scripts/eval_sun_raft_region_decomp.py`
- `scripts/create_sintel_test_submission_depth_raft.py`
- `scripts/benchmark_model_costs.py`
- `scripts/make_figure3_sintel_qualitative_v2.py`
- `scripts/make_figure4_kitti_qualitative.py`
- `scripts/make_z_gradient_figure.py`

Key implementation files:

- `core/raft.py`: Sun-RAFT baseline and Depth-RAFT model integration
- `core/dav2_wrapper.py`: frozen DAv2 wrapper for G and Z extraction
- `core/dav2_bottneck.py`: G bottleneck and identity-initialized fusion projector
- `core/extractor.py`: fnet/cnet encoders with zero-initialized Z input path
- `core/losses/`: original Sun-RAFT losses plus DAB-Smooth

DAv2 runtime code is vendored at `../third_party/Depth-Anything-V2/`; `DEPTH_ANYTHING_V2_ROOT` can override that path.

Quick implementation check:

Recommended environment: Python 3.9 or 3.10 with PyTorch installed from the matching CUDA or CPU wheel for your machine.

```bash
python -m pytest code/tests -q
```

Expected result in the public review snapshot:

```text
4 passed, 1 skipped
```

The public smoke tests and demo do not require datasets or checkpoints. The optional checkpoint equivalence test is skipped unless CUDA and the `DEPTH_RAFT_BASELINE_CKPT`, `DEPTH_RAFT_DAV2_WEIGHTS`, `DEPTH_RAFT_SINTEL_IMAGE1`, and `DEPTH_RAFT_SINTEL_IMAGE2` paths are provided.
