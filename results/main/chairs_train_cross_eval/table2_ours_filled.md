# Table 2 Ours Rows Filled

Lower is better. KITTI EPE follows the Sun-RAFT Table 2 convention: mean of per-pair EPE values. KITTI FL is the official pixel-weighted outlier rate.

## Rows to Fill in the Paper Table

| Method | Trained on | Chairs Val | Sintel Clean | Sintel Final | KITTI EPE | KITTI FL |
|---|---|---:|---:|---:|---:|---:|
| Ours | Chairs train | 2.07 | 1.7206 | 2.7542 | 4.6895 | 13.4810 |
| Ours | Sintel test | 2.0708 | 1.4970 | 2.4693 | 3.9854 | 11.6720 |

The final Sintel-finetuned Ours checkpoint was re-evaluated on Chairs Val: `2.0708`. This is essentially unchanged from the Chairs105K start (`2.068`, +0.0028 EPE), so there is no meaningful Chairs forgetting signal.

## Interpretation

- The Chairs-trained checkpoint improves Chairs Val strongly but does not transfer well to Sintel by itself: `1.7206 / 2.7542`, worse than the Sun-RAFT Sintel checkpoint `1.6942 / 2.5935`.
- The Chairs-trained checkpoint slightly improves KITTI EPE compared with the Sun-RAFT Sintel baseline (`4.7653 -> 4.6895`) but worsens KITTI FL (`12.6358 -> 13.4810`).
- The final Sintel-finetuned Ours row remains the main result: `1.4970 / 2.4693 / 3.9854 / 11.6720`.

## Source Summary

The raw run JSON files used to assemble this table were archived during cleanup. The final paper summary is `results/main/step_35000_paper_summary.json`.
