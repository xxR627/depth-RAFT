# depth-RAFT Code

Runnable code for the final Sun-RAFT + G + Z + DAB method.

Public naming:

- `G`: frozen DAv2 high-level feature branch injected into fnet features.
- `Z`: frozen DAv2 depth map injected into the context encoder input.
- `DAB`: depth-aware boundary smoothness.

Key scripts:

- `scripts/train_depth_raft_g_z_dab.py`: final training entry.
- `scripts/eval_depth_raft_clean_final.py`: Sintel Clean/Final evaluation.
- `scripts/eval_paper_sintel_occ_noc.py`: official Sintel matched/unmatched evaluation.
- `scripts/eval_paper_kitti_occ_noc.py`: KITTI 2015 evaluation.
- `scripts/create_sintel_test_submission_depth_raft.py`: Sintel test-flow export.
- `scripts/benchmark_model_costs.py`: parameter/runtime/memory benchmark.
- `scripts/make_z_gradient_figure.py`: RGB-gradient vs Z-gradient visualization.

Key implementation files:

- `core/raft.py`: Sun-RAFT baseline and Depth-RAFT model integration.
- `core/dav2_wrapper.py`: frozen DAv2 wrapper for G and Z extraction.
- `core/dav2_bottneck.py`: G bottleneck and identity-initialized fusion projector.
- `core/extractor.py`: fnet/cnet encoders with zero-initialized Z input path.
- `core/losses/`: original Sun-RAFT losses plus DAB-Smooth.

DAv2 runtime code is vendored at `../third_party/Depth-Anything-V2/`; `DEPTH_ANYTHING_V2_ROOT` can override that path.

The package intentionally excludes abandoned exploratory branches and ablation runners.
