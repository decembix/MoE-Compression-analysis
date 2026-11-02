# Expert output recovery와 NLL의 불일치 진단

## 결론

HC-SMoE L15의 기존 실패를 “correction 방향이 전부 잘못되었다”로 해석할 근거는 없다. Baseline에서 LoRA correction은 NLL gradient의 반대 방향 성분을 갖고, 같은 correction을 절반만 적용하면 original과 independent 모두 크게 개선된다. 이 case는 **과도한 finite correction magnitude와 비선형 downstream response에 더 부합**한다. H1의 일반적인 민감 방향 근사 부족을 배제하거나 H2만으로 원인을 확정하지는 않는다.

독립 샘플에서는 HC L15 α=1도 positive gain 점추정치를 보이지만 95% CI는 0을 포함한다. 기존의 음수 점추정치를 재현 가능한 보편적 실패로 취급할 수 없다. HC L3가 L10보다 좋았던 순위 역시 독립 표본에서 유지되지 않았다. REAP L15의 큰 positive gain은 두 표본에서 유지된다.

새 training/optimizer/compression은 실행하지 않았다. 기존 50% compressed checkpoint와 exp1의 고정 Rank16 adapter 5개만 사용했다. 모든 모델 parameter는 frozen이고 HC alias / REAP retained-expert 구조는 보존했다.

## 설계와 데이터

Teacher는 pinned `allenai/OLMoE-1B-7B-0924` revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`. Local Oracle A target은 실제 compressed normalized pre-MoE 입력에서 계산한 `f_T(x_C)`다. Teacher 전체 trajectory output을 섞지 않았다.

- Original: exp0/exp1의 기존 C4 validation 32×1024. Layer-selection에 사용된 표본이다.
- Independent: 같은 pinned validation shard를 seed 20261031로 shuffle하고 기존 모든 source document를 제외한 새 32×1024. Checkpoint/layer/rank를 고정한 뒤 선정했다. Train/calibration은 train split이다. Document IDs, 제외 목록과 SHA256는 `data/manifest.json`에 있다.
- 각 평가: 32,768 input tokens / 32,736 shifted targets. Sequence packing으로 문서가 sequence에 걸칠 수 있다. CI는 10,000회 paired-sequence bootstrap(seed42)의 탐색적 95% percentile 구간이다. 여러 alpha 비교에 대한 다중검정 보정은 하지 않았다. 새 분포/새 seed/독립 outer test는 아니다.

같은 method 내 baseline에서만 layer gain을 직접 비교한다. HC와 REAP의 raw gain 크기는 baseline과 compression 구조가 달라 strategy superiority의 직접 근거로 쓰지 않는다.

## Independent NLL 및 output error

Independent baseline HC NLL **4.165079**, PPL **64.3977**; REAP NLL **4.061642**, PPL **58.0696**.

| Case | Original LoRA gain [95% CI] | Independent LoRA gain [95% CI] | Independent Oracle A gain |
|---|---|---|---|
| HC L10 r16 | +0.010096 [+0.003826, +0.018530] | +0.023480 [+0.013170, +0.034579] | +0.179931 |
| HC L3 r16 | +0.028470 [+0.018047, +0.046945] | +0.022097 [+0.019341, +0.025071] | +0.138402 |
| HC L15 r16 | -0.000870 [-0.009362, +0.008239] | +0.010112 [-0.004894, +0.026061] | +0.073157 |
| REAP L15 r16 | +0.062133 [+0.055764, +0.068806] | +0.054390 [+0.037691, +0.070005] | +0.105224 |
| REAP L4 r16 | +0.012687 [+0.007705, +0.017775] | +0.023640 [+0.017300, +0.029885] | +0.096456 |

| Case | Independent L2 before → after | Cosine before → after | LoRA NLL / PPL | Oracle NLL / PPL |
|---|---|---|---|---|
| HC L10 | 0.7921 → 0.5582 | 0.7397 → 0.7771 | 4.141598 / 62.9033 | 3.985148 / 53.7932 |
| HC L3 | 0.8614 → 0.6698 | 0.8171 → 0.8160 | 4.142982 / 62.9904 | 4.026677 / 56.0743 |
| HC L15 | 0.4458 → 0.1844 | 0.9528 → 0.9801 | 4.154967 / 63.7498 | 4.091922 / 59.8548 |
| REAP L15 | 0.7255 → 0.2796 | 0.9509 → 0.9510 | 4.007253 / 54.9956 | 3.956418 / 52.2698 |
| REAP L4 | 1.2863 → 0.3586 | 0.7170 → 0.7040 | 4.038002 / 56.7129 | 3.965186 / 52.7301 |

Independent에서도 모든 LoRA case의 output L2가 감소했다. 하지만 error 감소량, cosine 변화, NLL gain의 순위는 서로 같지 않다. Local Oracle A gain은 reference이고 수학적인 recovery upper bound가 아니다. Ratio는 `independent_evaluation.csv`에 기록했다.

## A. NLL gradient alignment

`g = d(mean held-out NLL)/d y_C`, `δL=y_LoRA−y_C`, `δT=f_T(x_C)−y_C`. `First-order predicted gain = −Σ g·δ`. 음수 cosine/dot은 baseline에서의 descent alignment다. Dot과 cosine은 correction magnitude 및 gradient norm의 영향을 다르게 받으므로 함께 봐야 한다.

정확한 compressed native prefix를 사용하고 frozen suffix의 reverse layer-VJP를 계산했다. Branch 다음 residual addition의 미분은 identity이므로 decoder output gradient를 해당 MoE branch gradient로 대응시킨다. 재계산 batch8 forward 출력과 기존 prefix가 bitwise 일치함을 검사했다. BF16 forward/backward, FP32 loss/reduction을 사용한다. Gradient는 downstream top-k의 현재 routing selection에 대한 piecewise derivative이며 큰 intervention에는 그대로 적용되지 않는다.

| Split / case | cos(g,δL) | cos(g,δT) | −g·δL | Actual LoRA gain | −g·δT | Actual Oracle gain |
|---|---|---|---|---|---|---|
| original HC L10 | -0.003387 | -0.017133 | +0.075537 | +0.010096 | +0.439860 | +0.178965 |
| original HC L3 | -0.004376 | -0.008959 | +0.058903 | +0.028470 | +0.448662 | +0.163921 |
| original HC L15 | -0.006089 | -0.012188 | +0.148895 | -0.000870 | +0.340779 | +0.063975 |
| original REAP L15 | -0.008449 | -0.012953 | +0.157859 | +0.062133 | +0.268328 | +0.142678 |
| original REAP L4 | -0.000010 | -0.000171 | +0.007097 | +0.012687 | +0.124604 | +0.063023 |
| independent HC L10 | -0.004404 | -0.017035 | +0.109916 | +0.023480 | +0.466239 | +0.179931 |
| independent HC L3 | -0.004129 | -0.006204 | +0.052074 | +0.022097 | +0.255236 | +0.138402 |
| independent HC L15 | -0.006118 | -0.010943 | +0.158940 | +0.010112 | +0.314230 | +0.073157 |
| independent REAP L15 | -0.007729 | -0.010792 | +0.147240 | +0.054390 | +0.221947 | +0.105224 |
| independent REAP L4 | -0.000297 | -0.001859 | +0.032068 | +0.023640 | +0.207519 | +0.096456 |

모든 case의 global LoRA cosine 점추정치는 음수이다. Original REAP L4는 거의 0이므로 미약한 정렬을 과장해서는 안 되며, cosine/predicted-gain CI는 layer CSV에 제공한다. 따라서 α≈0에서의 전체 correction을 일괄적인 uphill direction으로 볼 수 없다. 그러나 flattened cosine 절댓값은 작고 token 단위로 유리한 성분과 불리한 성분이 공존한다. Negative cosine 자체는 α=1의 개선을 보장하지 않는다. 1차 예측과 실제 gain의 차이는 curvature, 정규화, downstream routing 변화(해당하는 earlier layer), BF16 수치효과를 포함하는 비선형 remainder이며 단일 원인의 측정값이 아니다.

| Split / case | LoRA token descent fraction | Mean token cosine | Near-zero g / slots | Positive / negative predicted contributions |
|---|---|---|---|---|
| original HC L10 | 0.5297 | -0.002600 | 32 / 32768 | +0.134718 / -0.059180 |
| original HC L3 | 0.5040 | -0.000152 | 32 / 32768 | +0.065288 / -0.006386 |
| original HC L15 | 0.4267 | 0.005290 | 32 / 32768 | +0.271006 / -0.122111 |
| original REAP L15 | 0.7159 | -0.014647 | 32 / 32768 | +0.239464 / -0.081605 |
| original REAP L4 | 0.5649 | -0.004391 | 32 / 32768 | +0.094466 / -0.087369 |
| independent HC L10 | 0.5595 | -0.005347 | 32 / 32768 | +0.163293 / -0.053377 |
| independent HC L3 | 0.5189 | -0.001359 | 32 / 32768 | +0.057858 / -0.005784 |
| independent HC L15 | 0.4537 | 0.004260 | 32 / 32768 | +0.285171 / -0.126231 |
| independent REAP L15 | 0.6840 | -0.012845 | 32 / 32768 | +0.233504 / -0.086264 |
| independent REAP L4 | 0.5663 | -0.004602 | 32 / 32768 | +0.087358 / -0.055290 |

Token cosine은 gradient 또는 correction norm ≤1e−12이면 계산하지 않고 빈 값으로 둔다. 마지막 slot은 다음 target이 없고 causal gradient가 0이다. Token cosine 평균과 layer flattened cosine은 다르다. Sequence별 결과 640행과 branch-slot 결과 655,360행을 별도 CSV로 제공한다.

Token의 `actual_next_target_nll_gain`은 같은 위치 다음 target loss의 차이다. Branch slot gradient는 이후 token loss도 포함할 수 있으므로 이것을 그 slot correction만의 인과적 actual gain으로 해석하면 안 된다. Layer predicted gain은 모든 slot의 global-normalized dot을 합한 값이고 sequence predicted gain은 해당 sequence mean loss 기준이다.

## B. Branch output interpolation

`y(α)=y_C+α(y_variant−y_C)`를 **MoE branch output**에서 수행했다. FP32 interpolation 뒤 BF16 output으로 cast하고 α0/1은 native endpoint 그대로 사용했다. Parameter scaling은 하지 않았다. LoRA와 Teacher correction에 동일 grid를 적용했다.

| Split / case / correction | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---|---|---|---|---|---|
| original HC L10 lora | +0.000000 | +0.010164 | +0.015180 | +0.015487 | +0.010096 | +0.001909 |
| original HC L10 teacher | +0.000000 | +0.083634 | +0.132506 | +0.163297 | +0.178965 | +0.185834 |
| original HC L3 lora | +0.000000 | +0.013577 | +0.022602 | +0.027844 | +0.028470 | +0.028377 |
| original HC L3 teacher | +0.000000 | +0.075926 | +0.118689 | +0.147250 | +0.163921 | +0.169982 |
| original HC L15 lora | +0.000000 | +0.027160 | +0.034697 | +0.024667 | -0.000870 | -0.039372 |
| original HC L15 teacher | +0.000000 | +0.065882 | +0.095022 | +0.092383 | +0.063975 | +0.015040 |
| original REAP L15 lora | +0.000000 | +0.033848 | +0.055891 | +0.064987 | +0.062133 | +0.047800 |
| original REAP L15 teacher | +0.000000 | +0.060455 | +0.105374 | +0.133308 | +0.142678 | +0.132928 |
| original REAP L4 lora | +0.000000 | +0.007615 | +0.011371 | +0.013245 | +0.012687 | +0.009365 |
| original REAP L4 teacher | +0.000000 | +0.037196 | +0.057849 | +0.066767 | +0.063023 | +0.049509 |
| independent HC L10 lora | +0.000000 | +0.015936 | +0.024885 | +0.026094 | +0.023480 | +0.017876 |
| independent HC L10 teacher | +0.000000 | +0.086932 | +0.135608 | +0.164198 | +0.179931 | +0.187575 |
| independent HC L3 lora | +0.000000 | +0.008808 | +0.016193 | +0.020231 | +0.022097 | +0.021786 |
| independent HC L3 teacher | +0.000000 | +0.057168 | +0.096017 | +0.122862 | +0.138402 | +0.146018 |
| independent HC L15 lora | +0.000000 | +0.029648 | +0.039843 | +0.032344 | +0.010112 | -0.024667 |
| independent HC L15 teacher | +0.000000 | +0.061656 | +0.091756 | +0.093918 | +0.073157 | +0.034156 |
| independent REAP L15 lora | +0.000000 | +0.031603 | +0.051643 | +0.059232 | +0.054390 | +0.037472 |
| independent REAP L15 teacher | +0.000000 | +0.049230 | +0.084155 | +0.103538 | +0.105224 | +0.089175 |
| independent REAP L4 lora | +0.000000 | +0.008137 | +0.014203 | +0.020614 | +0.023640 | +0.023730 |
| independent REAP L4 teacher | +0.000000 | +0.050617 | +0.080175 | +0.095129 | +0.096456 | +0.084981 |

각 case/split에서 α0 replay는 native compressed sequence loss와 일치했다. Original α1은 exp1 full-LoRA / exp0 Local Oracle A sequence loss와 일치했다. Independent α1은 동일 고정 adapter / teacher target을 실제 x_C에서 계산해 평가했다.

Grid 내 최대치는 탐색적이다. 특히 independent에서 가장 큰 alpha gain을 골라 검증된 tuned 성능으로 보고하지 않는다. 아래는 **original에서만 선택한 alpha**를 independent에 그대로 적용한 별도 비교다.

| Case | α selected on original | Independent gain at α | Independent gain at 1 | Improvement vs 1 [95% CI] |
|---|---|---|---|---|
| HC L10 | 0.75 | +0.026094 | +0.023480 | +0.002614 [+0.000729, +0.004403] |
| HC L3 | 1.0 | +0.022097 | +0.022097 | +0.000000 [+0.000000, +0.000000] |
| HC L15 | 0.5 | +0.039843 | +0.010112 | +0.029731 [+0.022473, +0.036680] |
| REAP L15 | 0.75 | +0.059232 | +0.054390 | +0.004842 [+0.000885, +0.009159] |
| REAP L4 | 0.75 | +0.020614 | +0.023640 | -0.003027 [-0.006034, +0.000001] |

L15의 작은 alpha 보충 진단:

| Split / case | Autograd gain slope | Central finite-difference slope (h=.01) | Gain at .01 / .05 / .1 |
|---|---|---|---|
| original HC L15 | +0.148895 | +0.145947 | +0.001499 / +0.007148 / +0.013299 |
| original REAP L15 | +0.157859 | +0.184800 | +0.001907 / +0.007683 / +0.015105 |
| independent HC L15 | +0.158940 | +0.163620 | +0.001701 / +0.007646 / +0.014339 |
| independent REAP L15 | +0.147240 | +0.176244 | +0.001707 / +0.007087 / +0.013887 |

중앙 차분은 `(NLL(−.01)−NLL(+.01))/.02`다. BF16 quantization 때문에 autograd와 수치적으로 같다고 강제하지 않았다. 작은 alpha의 실제 gain 부호와 gradient 예측을 함께 확인하는 보조 자료이다.

## Q1. HC-SMoE L15: MSE는 감소했는데 왜 NLL이 개선되지 않았는가?

Original α1 gain은 -0.000870, 95% CI [-0.009362, +0.008239]로 0을 포함한다. 따라서 먼저 통계적으로 확정된 악화가 아니었다. Independent α1은 +0.010112, CI [-0.004894, +0.026061]로 역시 0을 포함한다.

Baseline gradient에서 LoRA cosine은 original -0.006089, independent -0.006118; predicted gain은 각각 +0.148895, +0.158940다. 반면 α=.5의 실제 gain은 original +0.034697, independent +0.039843로 α1보다 크다. Original에서 선택한 α=0.5를 independent에 이전한 개선량은 α1 대비 +0.029731, CI [+0.022473, +0.036680]이다.

Teacher correction 자체도 α1이 NLL 최적점은 아니었다. Original Teacher α=.5 gain +0.095022는 α1 +0.063975보다 컸고, independent Teacher α=.75 gain +0.093918 역시 α1 +0.073157보다 컸다. Teacher output을 정확히 복원하면 local target MSE는 0이지만 그보다 NLL이 더 좋은 partial restoration이 존재한다. 따라서 이 현상을 LoRA 근사 오차만의 문제로 환원할 수 없다. Teacher branch와 compressed residual context의 조합이 task loss의 최적 output이라는 보장도 없다.

Token 단위 descent-aligned 비율은 original 약 42.7%, independent 약 45.4%로 절반 미만이고 mean token cosine은 양수다. Global cosine이 음수인 것은 gradient/correction norm이 큰 유익한 성분의 가중 효과를 포함한다. 따라서 민감 방향에 대한 국소적 오정렬이 사라졌다는 뜻은 아니다.

이는 같은 direction이 작은 magnitude에서는 유효하다는 증거로 H2에 더 부합한다. 다만 direction의 모든 component가 바람직하다는 의미는 아니다. MSE는 모든 output 좌표를 동일하게 벌점화하지만 NLL은 residual stream, final normalization, LM head/softmax를 통해 다르게 반응한다. **L15는 마지막 decoder layer**이므로 이 case의 suffix에는 추가 attention/MoE/router가 없다. 따라서 이 sweep에서 later-layer rerouting을 원인으로 설명할 수 없다. Final norm/head/CE의 비선형 response와 correction의 성분 및 크기를 조사하는 것이 타당하다. Alias를 바꾸지 않은 단순 output scaling만으로 개선되므로 alias 구조의 변경이 필수라는 증거도 아니다.

## Q2. HC L3 Rank16이 L10보다 좋았던 원인은 direction/magnitude인가?

Original L3−L10 gain 차이는 +0.018375, CI [+0.009453, +0.031195]. Independent에서는 -0.001384, CI [-0.013000, +0.009231]로 순위가 유지되지 않는다.

Original에서 L3 cosine -0.004376은 L10 -0.003387보다 더 음수지만, predicted gain은 L3 +0.058903, L10 +0.075537로 반대 순서다. Cosine만으로 실제 gain 순위를 설명할 수 없다. L10은 원래 alpha보다 작은 alpha에서 개선되고 L3는 1 부근에서 plateau하는 차이가 있어 magnitude/비선형 response도 관련된다. 그러나 이 자료로 특정 layer가 본질적으로 더 좋은 direction을 학습했다고 확정할 수 없다.

| Split / model / layers A−B | Gain A−B [95% CI] |
|---|---|
| original HC L3−L10 | +0.018375 [+0.009453, +0.031195] |
| original HC L10−L15 | +0.010965 [+0.000563, +0.022457] |
| original REAP L15−L4 | +0.049446 [+0.041355, +0.057454] |
| independent HC L3−L10 | -0.001384 [-0.013000, +0.009231] |
| independent HC L10−L15 | +0.013368 [+0.001860, +0.024363] |
| independent REAP L15−L4 | +0.030749 [+0.012964, +0.047040] |

## Q3. REAP L15의 큰 gain과 gradient alignment

LoRA cosine은 original -0.008449, independent -0.007729; predicted gain은 +0.157859, +0.147240; actual α1 gain은 +0.062133, +0.054390다. 두 표본에서 descent alignment와 positive actual gain이 함께 유지된다. Token descent-aligned 비율도 original 약 71.6%, independent 약 68.4%로 HC L15보다 높고 mean token cosine은 음수다. 이는 token 전반에 걸친 정렬이 더 넓게 나타난다는 관찰이다. 하지만 gain 크기는 cosine뿐 아니라 gradient/correction norm과 finite-alpha response에도 좌우된다. Gradient layer CSV의 두 norm과 dot을 함께 제공했다. Gradient 정렬이 높은 gain의 유일한 인과 원인이라고 주장하지 않는다.

REAP은 삭제 expert를 복구하지 않고 retained expert adapter만 사용했다. 따라서 유효한 correction이 현 구조에서 표현 가능함은 보여 주지만, Teacher correction과의 residual 및 도달 가능한 더 큰 gain의 한계를 식별하지는 않는다.

## Q4. Uniform MSE를 대체할 NLL-aware objective의 근거

근거는 있다. Uniform output error가 감소해도 finite NLL response가 일치하지 않았고, baseline descent alignment조차 α1 성능을 보장하지 않았다. 단순 gradient-norm-weighted MSE도 방향 및 curvature를 모두 반영하지 못한다. 실제 추가 진단은 다음과 같다.

| Split / case | Uniform MSE before → after | g-norm-weighted MSE before → after |
|---|---|---|
| original HC L10 | 0.00035816 → 0.00018368 | 0.0013466 → 0.00043865 |
| original HC L3 | 3.4603e-05 → 2.356e-05 | 0.00034031 → 0.00029625 |
| original HC L15 | 0.0057537 → 0.0011654 | 0.0060547 → 0.0012986 |
| original REAP L15 | 0.0076926 → 0.0012457 | 0.0075991 → 0.0012783 |
| original REAP L4 | 0.00016649 → 1.3211e-05 | 2.7717e-05 → 1.2339e-05 |
| independent HC L10 | 0.00037201 → 0.00018475 | 0.0013088 → 0.00035389 |
| independent HC L3 | 2.7182e-05 → 1.6436e-05 | 5.1144e-05 → 2.9265e-05 |
| independent HC L15 | 0.0059646 → 0.0010201 | 0.0059841 → 0.0011054 |
| independent REAP L15 | 0.0071364 → 0.0010602 | 0.0069148 → 0.0010555 |
| independent REAP L4 | 0.00018293 → 1.4216e-05 | 3.5543e-05 → 1.3675e-05 |

다만 이 결과는 uniform MSE가 항상 실패하거나 NLL-aware training이 반드시 더 낫다는 증명이 아니다. HC L15의 상당 부분은 training 변경 없이 alpha calibration으로 개선됐고, 독립 표본에서는 α1도 positive였다. 다음 objective 비교에서 출력 MSE의 weight만 바꾸고 magnitude를 방치하면 원인 분리가 되지 않을 수 있다.

## Q5. Alias를 변경하지 않고 추가 개선할 가능성

있다. Original-selected alpha를 independent에 이전한 위 표가 **기존 checkpoint/alias/expert/router를 유지한** 실제 증거다. HC L15의 개선은 가장 분명한 예다. 이것은 branch-level calibration의 가능성이며, 그 값을 LoRA parameter scaling으로 대체해도 동일하다는 주장은 아니다. 실제 output mixing은 base와 adapted branch 출력 모두를 요구하므로 추가 FFN 계산이 필요할 수 있다. 이번 진단은 배포 latency/throughput 개선을 측정하지 않았다. Gate/up/down의 여러 nonlinear path 때문에 두 scaling은 다르다.

현재 결과만으로 alias-specific LoRA의 추가 이득을 추정할 수 없다. Exp1의 physical merged expert adapter는 alias 간 서로 다른 함수를 직접 표현하지 못하는 구조적 제약이 있지만, 관찰된 L15 문제에 그 구조 변경이 필수였다고 볼 수 없다. Alias-specific adapter는 별도 trainable parameter 증가 및 routing-conditioned capacity confound도 통제해야 한다.

## 다음 연구 우선순위

1. **NLL-aware Output Distillation**을 다음 training 비교의 우선 후보로 둔다. 먼저 기존 uniform-MSE checkpoint에 original/별도 calibration set으로 고정한 branch alpha를 적용한 강한 baseline을 둔다. 동일 rank, token budget, trainable count로 비교하고 새 outer test에서 검증한다. NLL-aware 후보는 frozen compressed suffix를 통한 task NLL 항 또는 trust-region 안의 gradient/Hessian-aware correction loss를 MSE regularizer와 결합한다. Baseline의 선형 `g·δ`만 최소화하면 관측한 overshoot를 재현할 수 있으므로 magnitude constraint와 gradient 갱신/finite-NLL 검증이 필요하다. 이번에는 어떤 후보도 학습하지 않았다.
2. **Uniform Output MSE**는 baseline으로 유지한다. 가장 먼저 할 수 있는 무학습 개선은 독립적으로 검증된 alpha calibration이다. 이번 결과는 fixed α1의 한계를 보였으며 uniform MSE 전체를 폐기할 근거는 아니다.
3. **Alias-specific LoRA**는 calibrated physical-expert LoRA + NLL-aware 비교 뒤에 검증한다. 현재 실험은 alias 구조가 실패의 필수 원인임을 보이지 않았다. 추후에는 alias collision/subgroup residual과 NLL sensitivity를 측정하고 parameter-budget-matched control을 둔다.

## 검증·자원·한계

GPU 0만 사용했고 allocator cap 8 GiB, 관측 peak allocated **1.922 GiB**다. GPU 1 사용자 프로세스는 변경하지 않았다. CPU layer offload/target cache를 사용했다. `configs/completed.json`, `supplement_completed.json`에 실행 검증이 있다. 원본 compressed checkpoint 19개 파일과 adapter/config 10개 파일의 SHA256를 시작/종료에 비교했다. Trainable parameter update, checkpoint merge/save, 새 compression, downstream benchmark rerun은 없다.

두 split 각 32 packed sequence, adapter seed 하나, Rank16 다섯 case에 한정된다. 동일 validation shard 내 새 sample이며 distribution shift 검증이 아니다. Original은 layer selection에 사용됐고 independent alpha sweep도 exploratory다. Source-document disjointness는 exact sample overlap을 막지만 의미적 유사성/전체 데이터의 전역 dedup을 보장하지 않는다. Native BF16 gradient/rounding과 piecewise top-k derivative 때문에 1차 예측을 실제 finite gain 또는 causal direction decomposition으로 취급하지 않는다. H1과 H2를 완전히 식별하는 단일 실험은 아니며, 관찰은 magnitude 문제에 대한 직접 증거와 direction/curvature에 대한 남은 가설을 구분한다.

## 파일

- `../results/diagnosis/gradient_alignment.csv`, `gradient_alignment_sequences.csv`, `gradient_alignment_tokens.csv`
- `../results/diagnosis/output_interpolation.csv`, `independent_evaluation.csv`
- `../results/diagnosis/magnitude_transfer.csv`, `layer_comparison.csv`, `local_finite_difference.csv`, `residual_diagnostics.csv`
- `plots/01_output_interpolation.*` 등 figure 5종(PNG/PDF/CSV)
- `README.md`: normalization, 재현 명령, zero-gradient 처리와 token 해석
- `configs/final_validation.json`, `../results/diagnosis/artifact_manifest.json`: 최종 검증/provenance


공개 범위: [집계 CSV](../results/diagnosis/)와 그림을 제공하며 원래 로컬 실행의 cache/config/log는 포함하지 않습니다. 본문의 configs 경로는 원 실험 기록의 위치입니다. [분석 개요](overview.md) · [입력 및 방법](methodology.md)
