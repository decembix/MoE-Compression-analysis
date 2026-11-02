# Main comparison

| ID | Method | Params (B) | Recovery | Rank | PPL | PIQA | OBQA | ARC-C | Recovery Time (s) | Peak VRAM (bytes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | Teacher | 6.9192 | No | 0 | 23.1497 | 0.8079 | 0.4580 | 0.4855 | N/A | 13907727872 |
| A0 | DO-ACP | 5.0066 | No | 0 | 52.9661 | 0.6480 | 0.3340 | 0.2944 | N/A | 10081482240 |
| A1 | DO-ACP | 5.0066 | Yes | 8 | 46.9519 | 0.6545 | 0.3400 | 0.3072 | 4.7188 | 10081482240 |
| B0 | MC-SMoE merging-only | 5.0066 | No | 0 | 28.4274 | 0.6632 | 0.3320 | 0.2935 | N/A | 10081482240 |
| B1 | MC-SMoE merging-only | 5.0066 | Yes | 8 | 29.3831 | 0.6540 | 0.3380 | 0.3046 | 4.6750 | 10081482240 |
| C0 | HC-SMoE adaptation | 5.0066 | No | 0 | 40.8625 | 0.7318 | 0.3640 | 0.4113 | N/A | 10081482240 |
| C1 | HC-SMoE adaptation | 5.0066 | Yes | 8 | 41.2602 | 0.7291 | 0.3540 | 0.4061 | 4.6951 | 10081482240 |

## Recovery gains

- do_acp: post - pre PPL = -6.0143; PPL improvement (pre - post) = 6.0143.
- mc_smoe: post - pre PPL = 0.9557; PPL improvement (pre - post) = -0.9557.
- hc_smoe: post - pre PPL = 0.3977; PPL improvement (pre - post) = -0.3977.
