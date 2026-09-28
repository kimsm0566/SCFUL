# Dirichlet Support 반응 분석 설계

## 목적

균등 데이터량 Dirichlet `alpha=0.1` 연합학습에서 client 0을 삭제하면, 각
클래스의 남은 학습 support가 서로 다른 비율로 감소한다. 이 분석은 그 감소율과
`original`·client 0 제외 재학습 model 사이의 reconstruction behavior 변화가 어떤
관계를 갖는지 측정한다.

주 질문은 다음이다.

```text
client 0이 한 클래스에서 많이 차지할수록,
client 0 제거 뒤 그 클래스의 retrain behavior는 original과 더 달라지는가?
```

이 실험은 target output을 의도적으로 망가뜨리는 성공 기준을 사용하지 않는다.
모든 unlearning 또는 repair model은 client 0 제외 재학습 model을 기준으로 해석한다.

## Support 정의

클래스 `k`에 대해 다음을 정의한다.

```math
n_{t,k} = \text{client 0의 class } k \text{ train sample 수},

n_{r,k} = \text{나머지 client의 class } k \text{ train sample 수},

\rho_k = \frac{n_{t,k}}{n_{t,k}+n_{r,k}},

s_k = 1-\rho_k.
```

- `rho_k`: client 0 삭제로 사라지는 class `k` support 비율
- `s_k`: 삭제 뒤 남아 있는 class `k` support 비율

`shared`와 `target-specific`은 이분법이 아니라 `s_k`의 연속적인 수준으로
해석한다. 다만 보고 편의를 위해 client 0에 해당 class sample이 최소 500개인
label만 다음 구간에 넣는다.

| 구간 | 조건 | 의미 |
| --- | --- | --- |
| `shared_high` | `rho_k <= 0.20` | 최소 80%가 retained clients에 남음 |
| `medium_support` | `0.20 < rho_k < 0.60` | support가 일부 감소 |
| `low_support` | `rho_k >= 0.60` | 해당 semantic의 대부분이 client 0과 함께 사라짐 |
| `target_specific` | `n_r,k = 0` | retained support가 전혀 없음 |
| `not_targeted` | `n_t,k < 500` | client 0 contribution이 작아 구간 평균에서 제외 |

`target_specific`은 `low_support`보다 우선한다. Dirichlet `alpha=0.1`에서 이
구간이 항상 생기지는 않으며, 생기지 않는 것은 정상 결과로 기록한다.

## 분석 순서

### 1. 기존 run의 support audit

`split_manifest.json`의 `train_allocation`에서 client 0 및 retained class count를
계산한다. `metrics/per_class.csv`의 model별 class metric과 합쳐
`metrics/support_by_class.csv`를 만든다. 이 artifact에는 raw image나 train index를
복사하지 않는다.

### 2. 사전 정의된 support 구간과 seed screening

global seed 0--4에서 split만 생성해 구간별 label 존재 여부를 점검한다. model 결과를
보지 않고 split count만 사용해 full SCU 비교 seed를 선택한다.

현재 splitter와 CIFAR-10 class count 5,000으로 사전 점검한 결과, global seed 3의
client 0은 다음 두 조건을 동시에 만족한다.

```text
label 1: 3,656 / 5,000 삭제, rho=0.731  -> low_support
label 7:   896 / 5,000 삭제, rho=0.179  -> shared_high
```

따라서 seed 3은 서로 다른 support regime를 한 삭제 대상 안에서 비교하는
centralized full SCU 분석 후보로 사용한다. 이 선택은 model metric 이전에 고정한다.

### 3. Original과 재학습 기준의 support-response 측정

global seed 0--4 각각에서 10-client, 200-round original 및 client 0 제외
retrain model만 학습한다. `joint_forgetting`, target-only `forget-only`, naive repair,
conflict-aware repair는 이 단계에서 실행하지 않는다.

각 run의 public CIFAR-10 test class별로 다음을 저장한다.

```text
seed, label, support_group, rho, s,
model, mse_input, psnr, fg_output, fg_mu, fg_latent,
mse_delta_to_retrain
```

주 그림은 `x=rho_k`, `y=original의 fg_output` scatter다. 같은 data에 대해
`original MSE - retrain MSE`도 보조 지표로 기록한다. 다섯 seed의 50 label point와
구간별 평균·표준편차를 함께 보고하며, 단조 상관은 사전 가정이 아니라 관측 결과로만
해석한다.

### 4. Full SCU 비교와 향후 distributed repair

seed 3에 대해 원본·재학습·중앙집중 full SCU를 200 rounds / 200 SCU epochs로
학습한다. full SCU는 server가 `D_t`와 pooled `D_r`을 일시적으로 받는 분석용
centralized comparator이며, 제안 federated method가 아니다.

같은 `support_by_class.csv` schema로 `SCU_central_full`의 class별 retrain gap을
추가한다. Support 구간별로 다음을 비교한다.

```text
original ↔ retrain
SCU_central_full ↔ retrain
```

현재 target-only forgetting은 user 요구에 따라 실행하지 않는다. 분산 repair 비교는
새 joint forgetting과 distributed repair를 설계한 뒤, 같은 artifact schema와 seed 3
조건을 재사용해 추가한다. Server는 그 단계에서도 retained raw data를 받지 않는다.

## 산출물

각 original/retrain 또는 full SCU run에는 다음이 추가된다.

```text
metrics/support_by_class.csv
```

여러 run을 합치는 분석 명령은 다음을 만든다.

```text
outputs/dirichlet_support_response/<UTC timestamp>_seed-0/
  config.json
  source_runs.json
  metrics/support_response.csv
  metrics/support_group_summary.csv
  figures/original_support_response.png
  figures/model_support_response.png
  run.log
```

결과 문서는 `docs/experiment/08-18-01-dirichlet-support-반응-분석.md`로 기록한다.
그 문서에는 각 실행 명령, seed, config, output directory, class별 support table,
group별 결과, full SCU 결과를 분리해 적는다.

## 성공 기준과 해석 제한

- 원본과 재학습의 global average가 가까워도 class별 변화가 없다고 결론 내리지 않는다.
- `rho`와 class별 gap 사이에 뚜렷한 관계가 없을 수 있다. Reconstruction codec의
  class 간 일반화와 모델 stochasticity가 원인이 될 수 있으며, 그 경우도 중요한
  음성 결과다.
- full SCU가 shared-high에서 utility를 유지하는 것은 충분 조건이 아니다. 각
  support regime에서 retrain gap이 original 및 재학습 변동성보다 작아야
  retrain-equivalent 방향이라고 해석한다.
- 이 단계는 membership privacy 또는 latent client identity leakage를 평가하지 않는다.
- 다섯 seed는 초기 근거이며, 최종 주장은 추가 seed와 채널 조건에서 재검증해야 한다.
