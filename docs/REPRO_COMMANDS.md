# Reproduction Commands

Commands are written for PowerShell from the repository root. Replace dataset roots with local paths.

Default local dataset roots used during development:

- Sintel: `G:\flow_data\sintel`
- KITTI 2015 train: `G:\flow_data\KITTI\data_scene_flow\training`

## Sintel Clean / Final

```powershell
python code\scripts\eval_depth_raft_clean_final.py `
  --baseline-checkpoint checkpoints\sun_raft_sintel_baseline.pth `
  --trainable-checkpoint checkpoints\depth_raft_g_z_dab_step35000_best.pth `
  --dav2-weights checkpoints\depth_anything_v2_vits_frozen.pth `
  --sintel-root G:\flow_data\sintel `
  --output-json results\main\step_35000_clean_final_eval_rerun.json
```

## Official Sintel Matched / Unmatched

```powershell
python code\scripts\eval_paper_sintel_occ_noc.py `
  --baseline-checkpoint checkpoints\sun_raft_sintel_baseline.pth `
  --ours-checkpoint checkpoints\depth_raft_g_z_dab_step35000_best.pth `
  --dav2-weights checkpoints\depth_anything_v2_vits_frozen.pth `
  --sintel-root G:\flow_data\sintel `
  --output-json results\main\sintel_occ_noc_eval_rerun.json
```

## KITTI 2015 Cross-Domain

```powershell
python code\scripts\eval_paper_kitti_occ_noc.py `
  --baseline-checkpoint checkpoints\sun_raft_sintel_baseline.pth `
  --ours-checkpoint checkpoints\depth_raft_g_z_dab_step35000_best.pth `
  --dav2-weights checkpoints\depth_anything_v2_vits_frozen.pth `
  --kitti-root G:\flow_data\KITTI\data_scene_flow\training `
  --output-json results\main\kitti_eval_rerun.json
```

## Model Cost

```powershell
python code\scripts\benchmark_model_costs.py `
  --baseline-checkpoint checkpoints\sun_raft_sintel_baseline.pth `
  --ours-checkpoint checkpoints\depth_raft_g_z_dab_step35000_best.pth `
  --dav2-weights checkpoints\depth_anything_v2_vits_frozen.pth
```

## Final Training

The reported final stage starts from the Chairs-105K Depth-RAFT checkpoint and fine-tunes on Sintel for 35K steps.

```powershell
python code\scripts\train_depth_raft_g_z_dab.py `
  --baseline_ckpt checkpoints\sun_raft_sintel_baseline.pth `
  --init_trainable_checkpoint checkpoints\depth_raft_g_z_chairs105k_start.pth `
  --dav2_weights checkpoints\depth_anything_v2_vits_frozen.pth `
  --sintel_root G:\flow_data\sintel `
  --output_dir runs\depth_raft_g_z_dab `
  --batch_size 3 `
  --total_steps 35000 `
  --schedule_total_steps 35000 `
  --save_every 2500 `
  --beta_smoothness 0.5
```

The training script always enables the final Depth-RAFT path: G injection, Z context input, and DAB-Smooth. It trains only the G-fusion projector and cnet.
