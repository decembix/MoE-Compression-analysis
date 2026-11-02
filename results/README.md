# 측정 결과

모든 값은 기존 exp0–exp2의 실제 측정값입니다. 새 training/compression/benchmark를 실행하지 않았습니다.

- [compression/](compression/): Teacher-input local error/decomposition, single-layer NLL, full-model benchmarks, Oracle, correlation.
- [recovery/](recovery/): Expert-only LoRA rank8/16, layer-selection, Oracle reference, residual.
- [diagnosis/](diagnosis/): Original/independent gradient alignment, output interpolation, magnitude transfer.
- [visualization/](visualization/): 전체32 sequences의 block metrics와 고정 sequence0의 raw-coordinate/scatter/profile CSV.

[manifest.json](manifest.json)은 source와 published SHA256를 기록합니다. 원래 CSV의 절대 checkpoint 경로만 `local-experiment/...` 상대 표기로 바꿨으며 모든 수치 field는 보존했습니다. Source 위치는 원 연구의 exp0/exp1/exp2 archive를 뜻하고 이 repository에 해당 archive 전체가 포함된 것은 아닙니다.

[입력 조건·단위·CI 정의](../reports/methodology.md)를 먼저 확인하세요. 서로 다른 입력/문맥의 metric을 섞지 않으며, single-layer gain을 더해 전체 recovery를 추정하지 않습니다.
