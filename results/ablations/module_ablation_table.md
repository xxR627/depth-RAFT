# Module Ablation Table

All rows use the same paper-facing naming: `G` for DAv2 high-level features, `Z` for the DAv2 depth map, and `DAB` for depth-aware boundary smoothness.

| Setting | Sintel Clean | Sintel Final | Matched Clean | Unmatched Clean |
| --- | ---: | ---: | ---: | ---: |
| Sun-RAFT baseline | 1.6942 | 2.5935 | 0.8182 | 12.7358 |
| + G-only | 1.6268 | 2.5597 | 0.7694 | 12.4172 |
| + Z-only | 1.5833 | 2.5849 | 0.7514 | 12.0456 |
| + G+Z | 1.5266 | 2.5041 | 0.7202 | 11.6700 |
| + G+Z+DAB | 1.4970 | 2.4693 | 0.7100 | 11.5416 |

Use this table for the module ablation section. Lower is better.
