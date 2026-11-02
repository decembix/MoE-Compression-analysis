# Main 5B Seed 42 Results

This directory contains the final evaluation artifacts for the `main_5b_seed42` run.

## Result summaries

- [`main_comparison.md`](main_comparison.md): overall comparison across the teacher and compression methods.
- [`component_ablation.md`](component_ablation.md): router/expert recovery ablation results.
- [`rank_ablation.md`](rank_ablation.md): reconstruction-rank ablation results.
- [`reconstruction.md`](reconstruction.md): aggregate reconstruction-error comparison.
- [`layerwise_reconstruction.md`](layerwise_reconstruction.md): layer-wise reconstruction analysis.

Each summary also has a CSV version for downstream analysis. The `*.json` files contain the raw evaluation records for individual method/rank configurations.

## Naming

- `teacher_rank0.json`: uncompressed teacher baseline.
- `do_acp_rank*.json`: DO-ACP evaluations at the indicated reconstruction rank.
- `mc_smoe_rank*.json`: MC-SMoE evaluations.
- `hc_smoe_rank*.json`: HC-SMoE evaluations.
- `*_router_compressed.json`: variants that also compress the router.

All files in this directory are result artifacts only; checkpoints and runtime logs are intentionally excluded.
