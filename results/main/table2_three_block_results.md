# Three-Block Table 2-Style Results

Lower is better. KITTI EPE follows the Sun-RAFT Table 2 convention: mean of per-image EPE values. KITTI FL is the official pixel-weighted outlier rate.

## Trained on Chairs

| Method | Chairs Val | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
|---|---:|---:|---:|---:|---:|
| Sun-RAFT baseline checkpoint | 2.324 | -- | -- | -- | -- |
| + G only, Chairs 10K | 2.241 | -- | -- | -- | -- |
| + Z only, Chairs 10K | 2.170 | -- | -- | -- | -- |
| + G + Z, Chairs 105K (ours) | 2.068 | 1.721 | 2.754 | 4.690 | 13.481 |

## Trained on Sintel Test

| Method | Chairs Val | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
|---|---:|---:|---:|---:|---:|
| Sun-RAFT baseline | 2.324 | 1.694 | 2.594 | 4.765 | 12.636 |
| + G only | 2.241 | 1.627 | 2.560 | -- | -- |
| + Z only | 2.170 | 1.583 | 2.585 | -- | -- |
| + G + Z, DAB off | 2.068* | 1.527 | 2.504 | -- | -- |
| + G + Z + DAB (Ours) | 2.071 | **1.497** | **2.469** | **3.985** | **11.672** |

## Trained on KITTI Test

| Method | Chairs Val | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
|---|---:|---:|---:|---:|---:|
| Sun-RAFT KITTI checkpoint | -- | -- | -- | 2.176 | 7.661 |
| + G + Z + DAB (Ours, KITTI direction) | -- | -- | -- | -- | -- |

Notes:

- `*` means Chairs Val is measured at the selected Chairs pretraining checkpoint before Sintel fine-tuning.
- Ours/Sintel test was re-evaluated on Chairs Val after Sintel fine-tuning: `2.0708`, essentially unchanged from the Chairs105K start.
- `--` means not evaluated under that protocol.
- The current paper claim should use the `Trained on Sintel Test` block. The KITTI-trained block is only a reference unless we run KITTI-direction fine-tuning.
- To avoid overclaiming, state that current KITTI results are cross-domain from the Sintel-direction checkpoint.

