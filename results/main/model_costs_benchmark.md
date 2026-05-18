# Model Cost Benchmark

| Method | Registered Params (M) | Frozen DAv2 Params (M) | Total Params (M) | Trainable Params in Reported Training (M) | Time / Pair (ms) | Peak Mem (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sun-RAFT baseline | 5.26 | 0.00 | 5.26 | 5.26 | 77.9 +/- 5.9 | 0.48 |
| Ours (+G+Z) | 5.37 | 24.79 | 30.16 | 1.18 | 152.4 +/- 6.9 | 0.60 |

Notes:
- Inference is measured on one Sintel pair (`alley_1/frame_0001-0002`) with the same 12-iteration eval setting used elsewhere.
- DAv2 is frozen and hidden from `state_dict`; it is counted separately as frozen runtime capacity.
- Baseline trainable count assumes standard full-model Sun-RAFT training.
- Our trainable count is the actual fine-tuning scope: `cnet + dav2_fusion`; fnet, update block, flow head, and DAv2 are frozen.

