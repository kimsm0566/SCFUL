# Dirichlet Support 반응 분석

## 결론

균등 client data 수를 유지한 Dirichlet `alpha=0.1` split에서 client 0을 삭제하면
class별 남은 training support는 실제로 크게 달라진다. 그러나 이번 5개 seed의
original·재학습 비교에서는 **삭제 비율이 큰 class일수록 output gap이 커진다는
뚜렷한 관계를 관찰하지 못했다.**

이는 "support 감소가 중요하지 않다"는 결론이 아니다. 이번 reconstruction codec,
200-round training, 5 seed, 10개 유효 class-label point 조건에서는 class support
감소만으로 public-test reconstruction behavior의 차이를 분명히 보지 못했다는
음성 결과다. 이후 방법의 성공은 여전히 class별 client 0 제외 재학습 model을
기준으로 판단해야 한다.

## 질문과 정의

각 class `k`에서 다음을 계산한다.

```math
rho_k = \frac{n_{0,k}}{n_{0,k}+n_{r,k}}
```

- `n_0,k`: 삭제 client 0이 가진 class `k` train sample 수
- `n_r,k`: 남은 9 clients가 가진 class `k` train sample 수
- `rho_k`: client 0 삭제와 함께 사라지는 class support 비율

따라서 `rho=0.731`은 해당 class training support의 73.1%가 client 0과 함께
제거된다는 뜻이다. `shared`와 `specific`을 class가 존재하는지 여부만으로 나누지
않고, 이 연속적인 support 감소율로 해석했다.

분석용 구간은 client 0에 해당 label sample이 500개 이상인 경우에만 부여했다.

| 구간 | 조건 |
| --- | --- |
| `shared_high` | `rho <= 0.20` |
| `medium_support` | `0.20 < rho < 0.60` |
| `low_support` | `rho >= 0.60` |
| `target_specific` | retained count = 0 |
| `not_targeted` | client 0 count < 500 |

## 실행 설정

```text
dataset                   CIFAR-10
clients                   10
client당 train/test       5,000 / 1,000
split                     equal-size Dirichlet alpha=0.1
삭제 대상                 client_0 전체
seed                      0, 1, 2, 3, 4
original / 재학습         FedAvg 200 rounds, local epoch 1
batch size / learning rate 200 / 1e-4
channel                   power-normalized AWGN 10 dB
```

이 단계는 original과 client 0 제외 재학습 model만 만들었다. SCU, target-only
forgetting, naive repair, conflict-aware repair는 실행하지 않았다.

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_support.json \
  --seed <0부터 4> \
  --output-root outputs/dirichlet_support_original_retrain

python src/main.py \
  --experiment support_response \
  --runs-root outputs/dirichlet_support_original_retrain \
  --output-root outputs/dirichlet_support_response_original_retrain \
  --seed 0
```

## Run별 전체 결과

전체 public test 평균의 output gap은 class별 변화를 희석할 수 있으므로 보조 지표로만
제시한다. 기준은 client 0 제외 재학습 model이다.

| Global seed | Original MSE | Retrain MSE | Original output gap |
| ---: | ---: | ---: | ---: |
| 0 | 0.015191 | 0.015132 | 0.000283 |
| 1 | 0.015620 | 0.015862 | 0.000343 |
| 2 | 0.015411 | 0.015155 | 0.000208 |
| 3 | 0.015154 | 0.015480 | 0.000193 |
| 4 | 0.015267 | 0.015301 | 0.000233 |

## Support 구간별 class 결과

50개 seed-label 중 client 0 contribution이 500개 이상인 유효 point는 10개였다.
나머지 40개는 client 0 contribution이 작아 `not_targeted`로 분류했다. Strict
`target_specific` point는 없었다.

| Support 구간 | 유효 label 수 | 평균 rho | Original output gap 평균 | Original MSE − Retrain MSE 평균 |
| --- | ---: | ---: | ---: | ---: |
| `shared_high` | 2 | 0.147 | 0.000234 | -0.000101 |
| `medium_support` | 6 | 0.411 | 0.000246 | 0.000037 |
| `low_support` | 2 | 0.772 | 0.000275 | -0.000481 |

유효 10 point에서 `rho`와 original output gap의 Pearson correlation은 `0.164`,
rank correlation은 `0.067`이었다. 이 표본 수와 범위에서는 단조 관계의 근거가
되지 않는다. 50개 전체 point를 모두 포함해도 Pearson correlation은 `0.007`로
거의 0이었다.

scatter figure는 다음 artifact에 있다.

```text
outputs/dirichlet_support_response_original_retrain/
  20260817T161156Z_seed-0/figures/original_support_response.png
```

## 사전 선택된 seed 3

full SCU 비교 seed는 model 결과를 보기 전에 split count만으로 선택했다. seed 3의
client 0은 한 deletion target 안에서 서로 다른 두 support regime를 가졌다.

| Label | client 0 train | retained train | rho | 구간 | Original output gap |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 3,656 | 1,344 | 0.731 | `low_support` | 0.000224 |
| 7 | 896 | 4,104 | 0.179 | `shared_high` | 0.000209 |

두 label의 original output gap은 모두 작은 범위였고, low-support label 1이
shared-high label 7보다 명확히 더 큰 변화라고 볼 근거는 없었다.

## 해석과 다음 단계

이 결과는 Dirichlet `alpha=0.1`에서 support 감소율을 무시해도 된다는 뜻이 아니다.
오히려 다음 두 점을 명확히 한다.

1. support는 `있음/없음`이 아니라 삭제 비율 `rho`로 기록해야 한다.
2. support 감소와 reconstruction behavior의 관계는 가정하지 말고 class별
   retrain 기준으로 직접 측정해야 한다.

가능한 이유는 codec이 다른 class에서 얻은 저수준 image structure를 일반화하거나,
남은 1,344개도 reconstruction에 충분하거나, 현재 200-round/5-seed 조건의
stochasticity가 class effect보다 클 수 있기 때문이다. 이를 분리하려면 더 많은
seed, 더 긴 학습 일정, strict target-specific controlled split, LPIPS/MS-SSIM 및
target training member 평가가 필요하다.

이 artifact는 이후 중앙집중 full SCU와 distributed repair가 support regime별로
client 0 제외 재학습 model을 얼마나 근사하는지 평가하는 공통 schema로 사용한다.
