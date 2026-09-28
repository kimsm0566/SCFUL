> **상태: 무효 실험 기록**
>
> 이 문서는 target-only `SCU forget-only`를 사용한 초기 기록이다. Full SCU의
> centralized compensation을 포함하지 않으므로 이후 baseline, 결과 비교, 연구
> 결론에 사용하지 않는다. 현재 유효한 설정은
> `08-17-03-중앙집중-full-scu-실행-안내.md`를 따른다.

# FedOSD 호환 SCU 세 모델 CIFAR-10 실험

## 목적

FedOSD clone이 사용하는 balanced pathological non-IID data setting에서
reconstruction semantic codec의 client-level deletion behavior를 확인한다.
비교 대상은 세 model이다.

```text
original
  10개 client가 모두 참여한 연합학습 결과

retrain_without_client_0
  client_0의 전체 학습 data를 처음부터 제외하고 다시 연합학습한 기준 model

SCU
  original codec을 client_0이 자신의 data로 SCU식 forgetting한 결과
```

성공은 target reconstruction MSE를 단순히 키우는 것이 아니라, `SCU`가
`retrain_without_client_0`의 reconstruction과 posterior behavior에 얼마나
가까운지로 판단한다.

## FedOSD와 맞춘 data setting

FedOSD의 실행 예시는 CIFAR-10에서 `N=10`, `NC=2`, `balance=True`를 쓴다.
이 실험도 같은 label-slot 규칙을 사용한다.

| 항목 | 값 |
| --- | --- |
| client 수 | 10 |
| client당 train data | 5,000개 |
| client당 held-out test data | 1,000개 |
| client당 label 수 | 정확히 2개 |
| label당 보유 client 수 | 정확히 2개 |
| 삭제 대상 | client_0 전체 5,000개 |
| original/retrain | FedAvg, 200 communication rounds, local epoch 1 |
| batch size | 200 |
| channel | power-normalized AWGN, 10 dB |

FedOSD clone의 `separate_data()`를 따라 label을 오름차순으로 순차 배정한다.
따라서 client 0과 1은 label 0·1을 공유하고, 이어지는 client pair도 같은
방식으로 두 label씩 공유한다. 이는 target client의 semantic이 다른 retained
client에도 남아 있는, client deletion에 적합한 shared-support 조건이다.

FedOSD clone은 CIFAR-10 train/test를 한 pool에 합친 뒤 client 내부에서 다시
train/test로 나눈다. 여기서는 held-out test가 학습에 섞이지 않도록 원래
CIFAR-10 train 50,000개와 test 10,000개를 분리한다. label incidence와
client별 sample 수는 FedOSD setting과 같다.

## 이 실험에서 SCU가 의미하는 것

SCU 원본 구현은 한 codec 안에서 삭제 data와 남은 data를 paired batch로 받는다.
삭제 batch에는 reconstruction을 나쁘게 하는 forgetting loss를 적용하되, 목적식의
양의 KL 항은 unlearned posterior가 원본 posterior에서 과도하게 멀어지지 않도록
제약한다.

```math
L_{forget} = \lambda_{KL} KL(q_U(z|x_t) || q_G(z|x_t))
             - \lambda_{rec} MSE(\hat{x}_t, x_t)
```

학습은 이 loss를 최소화한다. 따라서 첫 항은 posterior 차이를 줄이고, 둘째 항은
reconstruction MSE를 키운다. 이 baseline은 posterior를 지우는 방법이라기보다
target reconstruction을 손상시키는 방법으로 해석해야 한다.

원본의 self-sharing variant는 같은 주체가 가진 남은 data에도 reconstruction,
VAE regularization, latent similarity 항을 적용한다. 하지만 이번 요청은
**client_0 전체를 삭제**하는 조건이므로 client_0 내부의 남은 data는 없다.
따라서 self-sharing term을 인위적으로 만들지 않고 `L_forget`만 쓰는
**SCU forget-only**를 `SCU`로 표기한다. Encoder와 decoder는 함께 갱신하며,
원본 codec은 gradient 없이 고정된 teacher다.

SCU는 원래 federated aggregation 방법이 아니다. 따라서 이 세 모델 baseline에서
SCU 단계는 남은 9개 client의 raw data나 update를 사용하지 않는다. FedOSD처럼
target gradient와 retained gradient를 직교화하거나, 남은 client가 repair하는
절차도 이번 비교에는 넣지 않는다. 그것들은 이후 distributed repair 방법의
별도 ablation이다.

`200 rounds`는 original과 retrain의 200 communication rounds를 뜻한다. SCU는
상대 client를 집계하지 않으므로 동일한 계산 단위를 200 local unlearning epochs로
정한다. batch size 200에서 client_0은 epoch당 25 optimizer steps, 총 5,000 steps를
수행한다.

## Hyperparameter

```text
codec latent dimension       64
codec base channels          32
optimizer                    Adam
learning rate                1e-4
VAE beta                     1e-4
SCU lambda_KL                1.0
SCU lambda_rec               0.1
SCU lambda_distill           0.0 (사용하지 않음)
seed                         0
split seed                   1
```

원본과 재학습은 byte-identical `initial.pt`에서 시작한다. 각 client의 data order,
posterior sampling, channel noise에는 phase/client/round 기반 seed를 사용한다.

## 실행

컨테이너에서 server 1개와 client 10개, 총 11 MPI process로 실행한다.

```bash
cd /workspace
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_pat_scu.json \
  --seed 0 \
  --output-root outputs/federated_cifar10_fedosd_pat_scu
```

작은 MPI protocol 검증에는 다음 config를 사용한다.

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_pat_scu_smoke.json \
  --seed 0 \
  --output-root outputs/federated_fedosd_pat_scu_smoke
```

## 산출물

각 실행은 timestamp와 seed를 포함한 새 directory를 만든다.

```text
outputs/federated_cifar10_fedosd_pat_scu/<timestamp>_seed-0/
  config.json
  environment.json
  split_manifest.json
  checkpoints/
    initial.pt
    original.pt
    retrain_without_client_0.pt
    SCU.pt
  metrics/
    summary.json
    per_class.csv
    per_client.csv
  run.log
```

`summary.json`의 `fg_output`, `fg_mu`, `fg_latent`는 각각 재학습 기준 model과의
reconstruction output MSE, posterior mean MSE, symmetric posterior KL 차이다.
작을수록 target client를 제외해 재학습한 model의 behavior에 가깝다.
