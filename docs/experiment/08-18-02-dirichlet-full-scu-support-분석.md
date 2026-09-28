# Dirichlet Full SCU Support 분석

## 결론

Dirichlet `alpha=0.1`, seed 3에서 중앙집중 full SCU는 reconstruction MSE를
유지했지만, client 0 제외 재학습 model과의 class별 output gap을 original보다
줄이지 못했다. 사전에 선택한 low-support label 1과 shared-high label 7 모두에서
full SCU의 gap은 original보다 컸다.

다만 차이의 절대값은 모두 작다. 이 결과는 full SCU가 utility를 붕괴시켰다는
결론이 아니라, 현재 목적식과 hyperparameter가 support regime에 따라
재학습 behavior를 선택적으로 따라간다는 증거를 제공하지 못했다는 뜻이다.

## 실행 설정

```text
dataset                   CIFAR-10
clients                   10, client당 train/test 5,000 / 1,000
split                     equal-size Dirichlet alpha=0.1
global seed / split seed  3 / 1221617604
삭제 대상                 client_0 전체
original / 재학습         FedAvg 200 rounds, local epoch 1
full SCU                  original에서 200 epochs, 5,000 optimizer steps
batch size / learning rate 200 / 1e-4
channel                   power-normalized AWGN 10 dB
lambda_KL/rec/ss          1.0 / 0.1 / 1.0
beta / epsilon            1e-4 / 1e-8
```

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_centralized_full_scu.json \
  --seed 3 \
  --output-root outputs/dirichlet_support_centralized_full_scu
```

결과 directory는 다음이다.

```text
outputs/dirichlet_support_centralized_full_scu/20260817T161304Z_seed-3/
```

이 model은 server가 client 0의 `D_t`와 client 1--9의 pooled raw `D_r`를
일시적으로 사용한 **중앙집중 full SCU 비교선**이다. Federated distributed repair가
아니며 raw retained data 비중앙화 제약을 만족하지 않는다. Raw image는 artifact에
저장하지 않았다.

## 전체 public test 결과

| Model | Input MSE | PSNR | Retrain과의 output gap |
| --- | ---: | ---: | ---: |
| original | 0.015154 | 18.195 dB | 0.000193 |
| retrain_without_client_0 | 0.015480 | 18.102 dB | 0.000000 |
| SCU_central_full | 0.014845 | 18.284 dB | 0.000551 |

Full SCU는 input MSE가 나빠지지 않았다. 그러나 retrain과의 output gap은 original의
`0.000193`보다 `0.000551`로 더 컸다. 이 수치만으로 시각적 차이를 단정하지는 않지만,
현재 full SCU가 이 seed에서 retrain-equivalent 방향으로 더 가까워졌다고 말할 수는
없다.

## Support regime별 직접 비교

seed 3은 model 학습 전에 다음 support 조건으로 선택됐다.

| Label | Support 구간 | rho | Original MSE | Retrain MSE | Full SCU MSE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `low_support` | 0.731 | 0.019825 | 0.020403 | 0.019540 |
| 7 | `shared_high` | 0.179 | 0.017070 | 0.017376 | 0.016651 |

같은 label에서 retrain 기준 output gap은 다음과 같다.

| Label | Support 구간 | Original gap | Full SCU gap |
| ---: | --- | ---: | ---: |
| 1 | `low_support` | 0.000224 | 0.000611 |
| 7 | `shared_high` | 0.000209 | 0.000548 |

두 regime 모두 absolute gap은 작은 범위이고, full SCU가 두 label의 reconstruction
MSE를 유지했다. 그러나 low-support label에서만 retrain 쪽으로 더 이동하거나,
shared-high label에서만 original utility를 선택적으로 보존했다는 pattern은 보이지
않는다. 이번 full SCU는 두 구간에서 모두 original보다 retrain output과 더 달라졌다.

## Full SCU loss 진단

| 지표 | 5,000 step 평균 | 마지막 step |
| --- | ---: | ---: |
| total loss | 0.125109 | - |
| joint-unlearning loss | 0.091800 | 0.108493 |
| compensation loss | 0.115896 | 0.119129 |
| contrastive loss | 0.094009 | 0.097734 |
| joint dynamic weight | 0.354971 | 0.476636 |
| compensation dynamic weight | 0.645029 | 0.523364 |

compensation은 평균적으로 더 큰 dynamic weight를 받았고, full SCU가 target-only
loss처럼 reconstruction을 붕괴시키지는 않았다. 하지만 centralized retained data를
사용한 이 비교선조차 class별 retrain functional gap을 줄이지 못했으므로, 이후
distributed repair는 단순히 full SCU loss를 나누어 계산하는 것만으로 충분하다고
가정해서는 안 된다.

## 한계와 의미

- 이 결과는 seed 3 하나의 full SCU 관측이다.
- Dirichlet 5-seed 분석에서 strict target-specific label은 나타나지 않았다.
- target training member, LPIPS/MS-SSIM, membership attack, latent client identity
  leakage, SNR variation은 아직 평가하지 않았다.
- 현재 target-only forgetting과 legacy repair는 user 요구에 따라 실행하지 않았다.

그럼에도 이 비교는 향후 방법의 기준을 명확히 한다. 제안 distributed repair는
server가 raw `D_r`를 보지 않으면서도, support regime별로 client 0 제외 재학습
codec의 behavior를 근사해야 한다. Shared semantic을 무조건 보존하거나,
low-support semantic을 무조건 약화하는 것이 목표가 아니다.
