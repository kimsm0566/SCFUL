# FedOSD 호환 중앙집중 Full SCU 실행 안내

## 목적

이 실험은 client 0 전체 삭제 상황에서 중앙집중 full SCU compensation이
삭제 대상 제외 재학습 model을 얼마나 근사하는지 측정한다. 서버가 `D_t`와 pooled
`D_r`를 볼 수 있는 것은 **이 비교선에서만** 허용한다.

```text
D_t = client_0 train data 5,000개
D_r = client_1--9 train data를 합친 45,000개
```

이는 향후 distributed repair가 대체해야 할 centralized remaining semantic knowledge의
상한을 확인하는 실험이다. 제안 방법이나 privacy-preserving federated method로
표현하지 않는다.

## 세 model

```text
original
  10개 client가 200 communication rounds FedAvg로 학습한 codec

retrain_without_client_0
  client_0을 처음부터 제외한 9개 client가 같은 initial state에서
  200 communication rounds FedAvg로 학습한 codec

SCU_central_full
  original에서 시작해 server가 D_t와 D_r을 함께 사용하여
  200 local epochs full SCU compensation을 학습한 codec
```

`forget-only` model은 만들지 않는다.

## Full SCU loss

한 step에서 server는 target batch `x_t`와 retained batch `x_r`를 함께 사용한다.

```math
L_{JU}^{raw}
= \lambda_{KL}KL(q_U(z|x_t)||q_G(z|x_t))
 - \lambda_{rec}MSE(\hat{x}_t,x_t)
```

```math
L_{CC}^{raw}
= \lambda_{ss}\left[
  \beta KL(q_U(z|x_r)||p(z))
  + MSE(\hat{x}_r,x_r)
  + \frac{1-\cos(z_r,z_r^+)}{1-\cos(z_t,z_r)+\epsilon}
\right]
```

`z_t`, `z_r`는 각각 target/retained posterior mean이고, `z_r^+`는
`x_hat_r.detach()`를 다시 encoder에 넣은 posterior mean이다. SCU clone의
`VAE_unl_ss`처럼 raw loss의 absolute magnitude로 detached dynamic weight를 계산한다.

```math
L_{TOTAL} = L_{JU} + L_{CC}
```

여기서 `L_JU`, `L_CC`는 각각 dynamic weight가 곱해진 joint-unlearning과
centralized compensation 항이다.

## Hyperparameter

| 항목 | 값 |
| --- | --- |
| Dataset | CIFAR-10 train 50,000 / test 10,000 |
| Client 수 | 10 |
| Data split | FedOSD balanced pathological, client당 2 label |
| Client당 train/test | 5,000 / 1,000 |
| 삭제 대상 | client_0 전체 |
| original/retrain | 200 rounds, local epoch 1 |
| Full SCU | 200 epochs, 5,000 optimizer steps |
| Batch size | 200 |
| Optimizer | Adam, learning rate 1e-4 |
| latent dimension | 64 |
| AWGN SNR | 10 dB |
| `lambda_KL`, `lambda_rec`, `lambda_ss` | 1.0, 0.1, 1.0 |
| `beta`, `epsilon` | 1e-4, 1e-8 |

Client 0과 client 1은 label 0·1을 각각 2,500개씩 공유한다. 따라서 client 0을
제외해 재학습해도 client 1에 shared semantic support가 남는다.

## 실행

```bash
cd /workspace
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_pat_scu.json \
  --seed 0 \
  --output-root outputs/centralized_full_scu
```

작은 MPI protocol 확인에는 smoke config를 사용한다.

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_pat_scu_smoke.json \
  --seed 0 \
  --output-root outputs/centralized_full_scu_smoke
```

## Artifact와 해석

```text
checkpoints/initial.pt
checkpoints/original.pt
checkpoints/retrain_without_client_0.pt
checkpoints/SCU_central_full.pt
metrics/summary.json
metrics/per_class.csv
metrics/per_client.csv
```

Server는 SCU 단계 종료 직후 pooled raw arrays를 해제하며, artifact에 raw data를 쓰지
않는다. 주 지표는 `retrain_without_client_0`과의 output/posterior functional gap이다.
Target MSE 증가만으로 성공을 주장하지 않는다.
