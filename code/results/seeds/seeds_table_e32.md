# Cross-seed robustness table (pooled per-image pairing)

## Per-seed mean delta (CGA - B0), mean +/- std of per-image deltas

| seed | n | EN | SF | AG | MI | QABF | VIF |
|---|---|---|---|---|---|---|---|
| 42 | 300 | +0.0099±0.0916 | -0.0879±0.0428 | -0.1366±0.0548 | +0.6549±0.3270 | +0.0110±0.0072 | +0.0058±0.0086 |
| 123 | 300 | +0.0173±0.0862 | -0.0255±0.0523 | -0.0911±0.0470 | +0.6503±0.3275 | +0.0038±0.0062 | +0.0037±0.0076 |
| 3407 | 300 | +0.0154±0.0819 | +0.0038±0.0511 | -0.0686±0.0484 | +0.5928±0.3101 | +0.0035±0.0057 | +0.0039±0.0073 |

## Pooled paired Wilcoxon (image = unit) + Holm

| metric | mean_delta | median_delta | n | p | p_holm | sign-consistent seeds |
|---|---|---|---|---|---|---|
| EN | +0.0142 | +0.0189 | 900 | 5.29e-12 | 5.29e-12 | 3+/0- |
| SF | -0.0366 | -0.0379 | 900 | 2.82e-56 | 7.95e-56 | 1+/2- |
| AG | -0.0988 | -0.0969 | 900 | 2.79e-143 | 1.40e-142 | 0+/3- |
| MI | +0.6327 | +0.6044 | 900 | 2.34e-147 | 1.40e-146 | 3+/0- |
| QABF | +0.0061 | +0.0054 | 900 | 1.54e-91 | 6.17e-91 | 3+/0- |
| VIF | +0.0045 | +0.0042 | 900 | 2.65e-56 | 7.95e-56 | 3+/0- |

*Seed-level tests intentionally omitted (n=3, min Wilcoxon p = 0.25);*
*direction consistency is reported descriptively.*
