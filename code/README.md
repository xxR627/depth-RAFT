# depth-RAFT Code

This public code folder is intentionally limited to the method components needed for inspection and evaluation.

Included:

- `DGE` implementation
- `DAB-Smooth` implementation
- evaluation scripts
- submission export script
- configuration files

Not included in this public snapshot:

- the full training pipeline
- pretrained checkpoints
- training logs and intermediate artifacts
- figure-generation and manuscript-support utilities

Public naming:

- `G`: frozen DAv2 high-level feature branch injected into fnet features
- `Z`: frozen DAv2 depth map injected into the context encoder input
- `DAB`: depth-aware boundary smoothness

Available scripts:

- `scripts/eval_depth_raft_clean_final.py`
- `scripts/eval_depth_raft_region_decomp.py`
- `scripts/eval_paper_sintel_occ_noc.py`
- `scripts/eval_paper_kitti_occ_noc.py`
- `scripts/eval_sun_raft_region_decomp.py`
- `scripts/create_sintel_test_submission_depth_raft.py`

Key implementation files:

- `core/raft.py`: Sun-RAFT baseline and Depth-RAFT model integration
- `core/dav2_wrapper.py`: frozen DAv2 wrapper for G and Z extraction
- `core/dav2_bottneck.py`: G bottleneck and identity-initialized fusion projector
- `core/extractor.py`: fnet/cnet encoders with zero-initialized Z input path
- `core/losses/`: original Sun-RAFT losses plus DAB-Smooth

DAv2 runtime code is vendored at `../third_party/Depth-Anything-V2/`; `DEPTH_ANYTHING_V2_ROOT` can override that path.
