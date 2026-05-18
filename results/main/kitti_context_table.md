# KITTI 2015 Context Table

This table separates cross-domain evaluation from KITTI in-domain fine-tuning. The main paper should avoid comparing our Sintel-trained checkpoint directly against KITTI-finetuned SOTA numbers as if they were the same setting.

| Method | Training setting | KITTI train EPE | KITTI train FL | Comparable to ours? |
| --- | --- | ---: | ---: | --- |
| UFlow | Sintel test | 7.67 | 17.41 | Yes, setting-level context |
| SMURF-EAS (MF) | Sintel test | 4.47 | 12.55 | Yes, but EAS uses arbitrary scaling |
| Sun-RAFT | Sintel test | 4.76 | 12.63 | Yes, direct baseline |
| Muun-RAFT | Sintel test | 4.39 | 12.35 | Yes, related baseline |
| Ours (+G+Z+DAB) | Sintel test | **3.99** | **11.67** | Yes, direct comparison |
| Sun-RAFT | KITTI test / in-domain | 2.17 | 7.66 | No, KITTI-finetuned/in-domain reference |

Suggested paper note:

> For KITTI, we report cross-domain results using the same Sintel-trained checkpoint setting as Sun-RAFT. We additionally list KITTI-finetuned/in-domain numbers as context only, since they are not directly comparable to our Sintel-trained model.

Important protocol note:

- The Sun-RAFT Table 2 KITTI EPE uses pair-mean EPE. Using pixel-weighted EPE gives larger values and makes the baseline appear mismatched. With pair-mean EPE, our local Sun-RAFT baseline is 4.765, matching the reported 4.76.

