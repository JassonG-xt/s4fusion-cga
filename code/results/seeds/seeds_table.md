# Cross-seed robustness table (pooled per-image pairing)

## Per-seed mean delta (CGA - B0), mean +/- std of per-image deltas

| seed | n | EN | SF | AG | MI | QABF | VIF |
|---|---|---|---|---|---|---|---|
| 42 | 300 | +0.0130±0.1039 | -0.1261±0.0659 | -0.1628±0.0704 | +0.7266±0.3745 | +0.0155±0.0081 | +0.0087±0.0102 |
| 123 | 300 | +0.0153±0.0850 | -0.0496±0.0515 | -0.1246±0.0520 | +0.6964±0.3398 | +0.0097±0.0070 | +0.0068±0.0077 |
| 3407 | 300 | +0.0239±0.0847 | -0.0381±0.0458 | -0.1012±0.0474 | +0.6967±0.3444 | +0.0086±0.0064 | +0.0065±0.0079 |

## Pooled paired Wilcoxon (image = unit) + Holm

| metric | mean_delta | median_delta | n | p | p_holm | sign-consistent seeds |
|---|---|---|---|---|---|---|
| EN | +0.0174 | +0.0227 | 900 | 1.31e-16 | 1.31e-16 | 3+/0- |
| SF | -0.0713 | -0.0650 | 900 | 3.62e-123 | 1.09e-122 | 0+/3- |
| AG | -0.1295 | -0.1187 | 900 | 7.25e-148 | 4.35e-147 | 0+/3- |
| MI | +0.7066 | +0.7066 | 900 | 3.35e-147 | 1.68e-146 | 3+/0- |
| QABF | +0.0113 | +0.0110 | 900 | 1.18e-137 | 4.71e-137 | 3+/0- |
| VIF | +0.0073 | +0.0073 | 900 | 6.30e-98 | 1.26e-97 | 3+/0- |

*Seed-level tests intentionally omitted (n=3, min Wilcoxon p = 0.25);*
*direction consistency is reported descriptively.*
