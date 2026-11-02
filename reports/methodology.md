# 입력·평가·시각화 방법

## 모델과 입력

Teacher: allenai/OLMoE-1B-7B-0924, revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`. C4 revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.

- **x_T local comparison:** Teacher trajectory의 normalized pre-MoE 입력을 원래/압축 branch에 동일하게 적용. Raw output viewer와 exp0 output error의 조건.
- **x_C Local Oracle A / LoRA target:** Frozen full compressed model에서 얻은 실제 normalized pre-MoE 입력. Teacher MoE target f_T(x_C)를 계산하며 teacher-trajectory output과 구분.
- **Single-layer compression sensitivity:** Teacher의 MoE 하나만 compressed branch로 바꾸고 전체 NLL 측정.
- **Local Oracle recovery:** Full compressed model에서 MoE 하나만 Teacher 함수로 교체하고 전체 NLL 측정. 다른 layer 문맥이 다르므로 위 sensitivity와 동일한 분해가 아님.

MoE branch output은 residual addition 전이며 attention까지 포함한 decoder output과 다릅니다. Layer는0–15. HC는 원래64 router slots→32/48 physical merged experts alias를 유지합니다. REAP은 retained experts만 사용합니다.

## 데이터와 불확실성

Original C4 validation32×1024,32736 shifted targets. Calibration/training은 C4 train split입니다. Independent는 같은 validation shard에서 seed20261031로 shuffle해 original의 모든73 source document를 제외한 새52 source documents의32×1024입니다. Layer/adapter는 새 표본을 선택하기 전에 고정됐습니다.

Packed sequences가 문서를 공유할 수 있어10000회 paired-sequence bootstrap(seed42)95% CI는 탐색적입니다. Original은 layer selection에 사용됐고 independent alpha 최대값을 골라 새로운 tuned-model test 성능으로 주장하지 않습니다. Magnitude transfer는 original에서만 선택한 alpha를 independent에 그대로 적용했습니다. 새 domain/outer test 검증은 아닙니다.

## 지표

- Relative L2 = sqrt(sum||y_C−y_T||² / sum||y_T||²). Global ratio와 mean token ratio는 다름.
- Cosine = token별 cosine 평균. Norm ratio = aggregate output norm / Teacher norm.
- Gain = NLL(compressed)−NLL(intervention). 양수는 개선. PPL=exp(NLL).
- Gradient g=d mean NLL / d y_C. First-order gain=−sum(g·delta). Layer row는 global mean, sequence row는 자기 sequence mean 기준.
- Token gradient/correction norm≤1e−12이면 cosine undefined로 처리. 마지막 token slot은 next target이 없고 causal gradient0.
- Token의 actual-next-target gain은 해당 branch slot만의 인과 효과가 아님. Gradient는 이후 loss의 영향도 포함할 수 있음.
- Native BF16 forward/backward, FP32 loss. Frozen suffix를 layer별 VJP로 재계산하고 native forward와 bitwise 동일성을 검사.

## Interpolation

`y(alpha)=y_C+alpha*(y_variant−y_C)`를 branch output에서 수행했습니다. FP32 연산 뒤 BF16 cast, alpha0/1은 native endpoint. Parameter scaling이 아닙니다. Alpha grid0,.25,.5,.75,1,1.25; L15 local probe−.01,.01,.05,.1도 제공됩니다. Oracle gain은 수학적 upper bound가 아닙니다.

## Viewer와 export 범위

상단 block metric은 exp0의 전체32×1024 집계입니다. Raw vector/heatmap/scatter는 sequence0의 tokens0–127/features0–255이며, token profiles는1024 tokens와 전체2048 features를 사용합니다. 임의의 feature 인접 관계에 공간적 의미를 부여하지 않습니다.

정적 대표 vector는 token32/first128 features. Heatmap은 tokens16–79/first128 features. 정적 scatter는 tokens16–127의 전체 features에서 seed42의8192좌표이며 두 method에 같은 좌표를 적용했습니다. L3/L15는 예시이고 대표 성능으로 일반화하지 않습니다.

Vector/scatter의 값은 정규화하지 않았습니다. Heatmap은 같은 layer/window의 세 output에 공통 scale, 두 difference에도 공통 scale을 사용합니다. Static symmetric-log color는 pooled|value|90th percentile을 linear core threshold로 사용하며 clipping은 없습니다. Viewer는 linear/asinh 색상을 선택하며 원본 output은 바꾸지 않습니다. Layer/window마다 color range가 달라질 수 있습니다.

Sub-MoE/SVD, block별 benchmark ACC intervention은 미실험입니다. 공개된 집계 CSV/뷰어 subset으로 시각화만 재생성할 수 있으며 전체 모델 실험 재현은 이 저장소의 범위가 아닙니다.
