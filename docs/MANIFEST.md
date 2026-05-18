# Manifest

## Code

- `code/core/raft.py`: Sun-RAFT baseline plus the final Depth-RAFT G+Z integration.
- `code/core/dav2_wrapper.py`: frozen DAv2 wrapper for extracting G features and Z depth.
- `code/core/dav2_bottneck.py`: G bottleneck and identity-initialized fusion projector.
- `code/core/extractor.py`: fnet/cnet encoders, including the zero-initialized Z input path.
- `code/core/losses/flow_loss.py`: Sun-RAFT unsupervised loss plus DAB-Smooth.
- `code/scripts/train_depth_raft_g_z_dab.py`: final training entry.
- `code/scripts/eval_depth_raft_clean_final.py`: Sintel Clean/Final train evaluation.
- `code/scripts/eval_paper_sintel_occ_noc.py`: official Sintel matched/unmatched evaluation.
- `code/scripts/eval_paper_kitti_occ_noc.py`: KITTI 2015 evaluation.
- `code/scripts/create_sintel_test_submission_depth_raft.py`: Sintel test-flow export.
- `code/scripts/benchmark_model_costs.py`: parameter/runtime/memory benchmark.
- `code/scripts/make_z_gradient_figure.py`: RGB-gradient vs Z-gradient visualization.

## Third-Party Runtime

- `third_party/Depth-Anything-V2/depth_anything_v2/`: vendored DAv2 runtime files used by `code/core/dav2_wrapper.py`.
- `third_party/Depth-Anything-V2/LICENSE`: upstream DAv2 license.

## Checkpoints

- `checkpoints/sun_raft_sintel_baseline.pth`: Sun-RAFT Sintel baseline checkpoint.
- `checkpoints/sun_raft_kitti_baseline.pth`: Sun-RAFT KITTI reference checkpoint.
- `checkpoints/depth_anything_v2_vits_frozen.pth`: frozen DAv2-S weights.
- `checkpoints/depth_raft_g_z_chairs105k_start.pth`: Chairs-105K Depth-RAFT start checkpoint.
- `checkpoints/depth_raft_g_z_dab_step35000_best.pth`: final reported checkpoint.

## Results

- `results/main/step_35000_paper_summary.md`: headline results.
- `results/main/sintel_occ_noc_eval.json`: official matched/unmatched split.
- `results/main/kitti_eval.json`: KITTI 2015 cross-domain evaluation.
- `results/main/model_costs_benchmark.md`: parameter/runtime/memory table.
- `results/main/table2_style_results.md`: paper table draft.
- `results/ablations/module_ablation_table.md`: clean G/Z/DAB module ablation table.

## Figures

- `figures/总框图.png`: method overview figure.
- `figures/figure1_motivation_v2.*`: motivation figure.
- `figures/figure3_sintel_qualitative_v4.*`: Sintel qualitative comparisons.
- `figures/figure4_kitti_qualitative_v2.*`: KITTI qualitative comparisons.
- `figures/figure5_beta_sweep.*`: DAB beta sensitivity.
- `figures/figure6_training_curve.*`: Sintel fine-tuning curve.
- `figures/figure_z_gradient_vs_image_gradient.*`: image-gradient vs Z-gradient motivation.
- `figures/temple_2_flow_outputs/`: selected qualitative flow/error example.
- `figures/temple_2_dav2_z/`: selected DAv2 Z visualization example.

## Excluded From Public Package

The package excludes abandoned runnable branches, intermediate logs, cache files, obsolete raw ablation directories, diagnostic-injection experiments, and duplicate checkpoints. The backup copy is outside this repository.
