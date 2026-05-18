# Step 35000 Paper Summary

Checkpoint: `checkpoints/depth_raft_g_z_dab_step35000_best.pth`

Method: Sun-RAFT + G + Z + DAB.

## Sintel Train Main Evaluation

Use these numbers as the headline Sintel clean/final result.

| Method | Clean EPE | Final EPE |
|---|---:|---:|
| Sun-RAFT baseline | 1.6942 | 2.5935 |
| Ours step35000 | 1.4970 | 2.4693 |
| Delta | -0.1972 | -0.1242 |
| Relative improvement | -11.64% | -4.79% |

## Sintel Official Matched / Unmatched Regions

Use these numbers when discussing the official matched/noc and unmatched/occ split.

| Split | Method | Matched / Noc EPE | Unmatched / Occ EPE |
|---|---|---:|---:|
| Clean | Sun-RAFT baseline | 0.8182 | 12.7358 |
| Clean | Ours step35000 | 0.7100 | 11.5416 |
| Clean | Delta | -0.1082 | -1.1942 |
| Final | Sun-RAFT baseline | 1.5647 | 15.6279 |
| Final | Ours step35000 | 1.4845 | 14.8409 |
| Final | Delta | -0.0802 | -0.7871 |

Note: the official-region script has its own recomputed `all` field. For paper headline clean/final, use the main evaluation table above.

## KITTI 2015 Cross-Domain Evaluation

No KITTI finetuning; direct forward on KITTI 2015 train 200 pairs. EPE is aligned to Sun-RAFT Table 2: mean of per-pair EPE values. FL is the official pixel-weighted KITTI outlier rate.

| Method | EPE-all | EPE-noc | FL-all | FL-noc |
|---|---:|---:|---:|---:|
| Sun-RAFT baseline | 4.7653 | 2.8171 | 12.6358 | 8.6209 |
| Ours step35000 | 3.9854 | 2.2007 | 11.6720 | 7.8067 |
| Improvement | -16.37% | -21.88% | -0.96 abs% | -0.81 abs% |

Traceability note: the previous pixel-weighted EPE-all values were 5.1097 -> 4.2655. Those are retained in `kitti_eval.json` but should not be compared to Sun-RAFT Table 2.

## One-Line Summary

On Sintel train, our Sun-RAFT + G + Z + DAB checkpoint at step35000 improves Clean EPE from 1.6942 to 1.4970 and Final EPE from 2.5935 to 2.4693. It also improves official unmatched/occluded EPE on Clean from 12.7358 to 11.5416 and transfers positively to KITTI 2015, reducing Table-2-aligned KITTI EPE-all from 4.7653 to 3.9854.


