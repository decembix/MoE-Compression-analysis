# DO-ACP recovery component ablation

The compressed-router variants replace only the 64-way router with a 45-way usage-weighted router in memory.

| Variant | Teacher Routing | Low-rank Correction | Router | Rank | Params | PPL | PIQA | OBQA | ARC-C | Peak GPU bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | No | No | compressed | 0 | 5005936640 | 44.3476 | 0.65778 | 0.328 | 0.332765 | 10080237056 |
| Router Recovery | Yes | No | teacher_mapped | 0 | 5006559232 | 52.9661 | 0.647987 | 0.334 | 0.294369 | 10081482240 |
| Expert Recovery | No | Yes | compressed | 8 | 5005936640 | 44.1306 | 0.658868 | 0.334 | 0.333618 | 10080237056 |
| Combined Recovery | Yes | Yes | teacher_mapped | 8 | 5006559232 | 46.9519 | 0.654516 | 0.34 | 0.307167 | 10081482240 |
