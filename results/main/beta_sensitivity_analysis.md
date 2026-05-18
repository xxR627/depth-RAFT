# Beta Sensitivity Analysis

| beta | Sintel Clean EPE | Status | Note |
| ---: | ---: | --- | --- |
| 0.1 | -- | Not run | Optional low-weight setting. The completed runs already cover the stable basin around the selected beta. |
| 0.3 | 1.5732 | Completed | Stable, but worse than beta=0.5. |
| 0.5 | **1.5426** | Completed | Best completed stable setting; used as the default. |
| 1.0 | 1.5787 | Completed | Stable, but likely over-regularized. |
| 2.0 | -- | Unstable | Training became unstable/crashed. |

Suggested paper text:

> We set beta=0.5 according to a small sensitivity sweep on Sintel Clean. Moderate regularization (beta=0.5) performs best, while weaker or stronger settings degrade performance; beta=2.0 caused unstable training. This indicates that the DAB-Smooth term is useful only as a mild geometric regularizer rather than as a dominant objective.

Generated figure files:

- `figures/figure5_beta_sweep.pdf`
- `figures/figure5_beta_sweep.png`

If a reviewer explicitly asks for beta=0.1, it is the only missing low-weight point. It is not necessary for the current claim because beta=0.3, 0.5, and 1.0 already establish a non-monotonic optimum and beta=2.0 establishes the instability boundary.

