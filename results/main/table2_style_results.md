# Table 2-Style Results (Grouped by Training Domain)

This version follows the Sun-RAFT Table 2 spirit more closely by grouping rows into checkpoints trained on Chairs, Sintel, and KITTI.

| Trained on | Method | Checkpoint / protocol | Chairs Val | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
|---|---|---|---:|---:|---:|---:|---:|
| **Chairs** | + G only | Chairs 10K pretrain checkpoint | 2.241 | -- | -- | -- | -- |
|  | + Z only | Chairs 10K pretrain checkpoint | 2.170 | -- | -- | -- | -- |
|  | + G + Z selected init (ours) | Chairs 105K pretrain checkpoint | **2.068** | 1.721 | 2.754 | 4.690 | 13.481 |
| **Sintel** | Sun-RAFT baseline (reproduced) | Official Sintel test checkpoint | 2.324 | 1.694 | 2.594 | 4.765 | 12.636 |
|  | + G only | Chairs 10K -> Sintel 5K | 2.241* | 1.627 | 2.560 | -- | -- |
|  | + Z only | Chairs 10K -> Sintel 5K | 2.170* | 1.583 | 2.585 | -- | -- |
|  | + G + Z, DAB off | Chairs 105K -> Sintel 17.5K | **2.068*** | 1.527 | 2.504 | -- | -- |
|  | + G + Z + DAB (Ours) | Chairs 105K -> Sintel 35K | **2.071** | **1.497** | **2.469** | **3.985** | **11.672** |
| **KITTI** | Sun-RAFT baseline (official KITTI ckpt) | Official KITTI checkpoint, A0, 12 iters | -- | -- | -- | **2.176** | **7.661** |

Notes:

- Lower is better. Best numbers within each training block are bolded.
- KITTI EPE is aligned to Sun-RAFT Table 2: mean of per-pair EPE values. KITTI FL is the official pixel-weighted outlier rate.
- `*` means the Chairs Val number comes from the corresponding pre-Sintel Chairs checkpoint.
- The final `G + Z + DAB` Ours checkpoint was re-evaluated on Chairs Val after Sintel fine-tuning: `2.0708`, essentially unchanged from the Chairs105K start.
- The `G only` and `Z only` Sintel rows are short 10K+5K directional ablations, not the final full training recipe.
- The KITTI block is included for Table-2-style grouped presentation. Our main result remains the Sintel-trained `G + Z + DAB` row.

Recommended paper wording:

> When reorganized into Sun-RAFT Table 2 style blocks, our main comparison remains the Sintel-trained section. There, `G + Z + DAB` improves Sintel Clean/Final from 1.694/2.594 to 1.497/2.469, and also improves cross-domain KITTI from 4.765/12.636 to 3.985/11.672 without KITTI fine-tuning. The separate KITTI-trained Sun-RAFT checkpoint is reported only to preserve the original paper's grouped presentation style.

