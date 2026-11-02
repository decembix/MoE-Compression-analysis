# DO-ACP recovery rank ablation

The ridge solve and rank-32 exact compact SVD are shared; lower-rank folds are truncations of that same solution.

| Rank | Reconstruction weighted MSE | PPL | PIQA | OBQA | ARC-C | Shared recovery preprocessing (s) | Peak GPU bytes | Parameters after folding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00214294 | 52.9661 | 0.647987 | 0.334 | 0.294369 | 0 | 10081482240 | 5006559232 |
| 4 | 0.00144737 | 47.3586 | 0.655604 | 0.334 | 0.309727 | 4.71882 | 10081482240 | 5006559232 |
| 8 | 0.00125207 | 46.9519 | 0.654516 | 0.34 | 0.307167 | 4.71882 | 10081482240 | 5006559232 |
| 16 | 0.000998002 | 46.367 | 0.65506 | 0.334 | 0.313993 | 4.71882 | 10081482240 | 5006559232 |
| 32 | 0.00068354 | 45.541 | 0.653428 | 0.328 | 0.302048 | 4.71882 | 10081482240 | 5006559232 |
