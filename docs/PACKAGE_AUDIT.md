# Package Audit

Root: `G:\depth-RAFT`

## Included

| Folder | Purpose |
| --- | --- |
| `code/` | Final model, loss, training, evaluation, and figure scripts. |
| `checkpoints/` | Baselines, frozen DAv2 weights, Chairs-start checkpoint, and final checkpoint. |
| `third_party/Depth-Anything-V2/` | Vendored DAv2 runtime code required by the wrapper. |
| `docs/` | Manifest, reproduction commands, naming, and this audit. |
| `figures/` | Paper figures and selected qualitative outputs. |
| `results/main/` | Final metric summaries and paper tables. |
| `results/ablations/` | Clean module-ablation table only. |

## Removed Or Archived

Moved out of the repository into `G:\depth-RAFT_pruned_backup_20260516_184019`:

- Python caches, logs, PID files, and generated artifacts.
- Abandoned extra-loss, teacher, and diagnostic-injection code paths.
- Old ablation runner scripts and raw short-run experiment folders.
- Duplicate or obsolete checkpoints.
- Smoke/protocol-sweep outputs and machine-specific raw JSON with stale absolute paths.

The DAv2 source copied into `third_party/Depth-Anything-V2/` contains only the Python runtime package, upstream README, requirements, and license. Cached bytecode, example media, and the upstream checkpoint copy were not copied.

## Public Entry Points

| Task | Script |
| --- | --- |
| Final training | `code/scripts/train_depth_raft_g_z_dab.py` |
| Sintel Clean/Final eval | `code/scripts/eval_depth_raft_clean_final.py` |
| Sintel matched/unmatched eval | `code/scripts/eval_paper_sintel_occ_noc.py` |
| KITTI eval | `code/scripts/eval_paper_kitti_occ_noc.py` |
| Sintel test submission | `code/scripts/create_sintel_test_submission_depth_raft.py` |
| Model cost | `code/scripts/benchmark_model_costs.py` |
| Z-gradient figure | `code/scripts/make_z_gradient_figure.py` |

## Validation

Before handoff, run:

```powershell
python -m compileall -q code\core code\scripts code\tests code\evaluate.py code\custom_logger.py
```

Then scan for stale public names and machine paths:

```powershell
rg -n "<old-name-or-local-path>" code README.md docs results figures
```

Expected remaining implementation-specific exception: `dav2_fusion` checkpoint/module key.
