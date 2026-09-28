# 중앙집중 Full SCU CIFAR-10 seed 0 결과

## 결론

중앙집중 full SCU는 이전 target-only `forget-only` 실행에서 나타난 reconstruction
붕괴를 제거했다. 전체 test MSE는 `0.290229`에서 `0.014737`로, target client 0의
MSE는 `0.336139`에서 `0.016215`로 회복됐다. 따라서 retained data 기반의
compensation 자체는 효과가 있다.

다만 이 pathological split에서는 original model이 이미 client 0 제외 재학습 model에
매우 가깝다. Full SCU의 output functional gap은 `0.000738`로 original의
`0.000260`보다 약 2.84배 크다. 즉 이번 seed 0 결과는 **중앙의 retained data가
utility 붕괴를 막는다는 근거**이지, 현재 hyperparameter의 full SCU가 재학습 결과를
가장 잘 근사한다는 근거는 아니다.

## 실행 조건

```text
실행 시각                 2026-08-17 KST
결과 directory            outputs/centralized_full_scu/20260817T062235Z_seed-0/
dataset                   CIFAR-10 train 50,000 / test 10,000
client 수                 10
data split                FedOSD balanced pathological, client당 2 label
client당 train/test       5,000 / 1,000
삭제 대상                 client_0 전체 train data 5,000개
retained data             client_1--9 train data pooled 45,000개
seed / split seed         0 / 1
channel                   power-normalized AWGN, 10 dB
```

실행 명령은 다음과 같다.

```bash
cd /workspace
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_pat_scu.json \
  --seed 0 \
  --output-root outputs/centralized_full_scu
```

세 model은 다음과 같이 학습했다.

| Model | 학습 방법 |
| --- | --- |
| `original` | 10 clients, FedAvg 200 rounds, local epoch 1 |
| `retrain_without_client_0` | client 0을 처음부터 제외한 9 clients, 같은 initial state에서 FedAvg 200 rounds |
| `SCU_central_full` | `original`에서 시작. server가 `D_t`와 pooled `D_r`을 일시적으로 보유하고 200 SCU epochs 학습 |

모든 학습은 Adam, batch size 200, learning rate `1e-4`, `beta=1e-4`를 사용했다.
Codec은 latent dimension 64, base channels 32다. Full SCU의
`lambda_KL=1.0`, `lambda_rec=0.1`, `lambda_ss=1.0`, `epsilon=1e-8`이다.

Full SCU는 매 epoch에서 shuffled target loader와 shuffled retained loader를
`zip`으로 pair한다. 따라서 target 5,000개에 대응하는 25 batch, 총 5,000 steps를
수행했다. 각 epoch에 pooled retained 45,000개 전체를 한 번씩 쓰지는 않지만,
retained loader를 epoch마다 다시 섞으므로 전체 pool에서 새 5,000개 subset을 본다.

## 분할 확인

`split_manifest.json`을 확인했다.

- 모든 client는 train 5,000개와 test 1,000개를 가진다.
- client 0과 client 1은 label 0·1을 각각 2,500개씩 보유한다.
- 따라서 client 0을 삭제해도 label 0·1의 shared semantic support가 client 1에 남아 있다.
- `SCU_central_full` 단계의 raw `D_t`, `D_r`는 server memory에서만 일시적으로 사용하고, artifact에는 저장하지 않는다.

## 전체 test 결과

기준 model은 `retrain_without_client_0`이다. `output gap`, `mu gap`, `latent gap`은
각각 이 기준 model과의 reconstruction MSE, posterior mean MSE, symmetric posterior
KL이다. 모두 작을수록 client 0을 제외하고 처음부터 학습한 결과에 가깝다.

| Model | Input MSE ↓ | PSNR ↑ | Output gap ↓ | Mu gap ↓ | Latent gap ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.015428 | 18.117 dB | 0.000260 | 0.073300 | 15.920227 |
| `retrain_without_client_0` | 0.015418 | 18.120 dB | 0.000000 | 0.000000 | 0.000000 |
| `SCU_central_full` | 0.014737 | 18.316 dB | 0.000738 | 0.073529 | 15.903289 |

`SCU_central_full`의 input MSE와 PSNR는 잘 유지됐다. 하지만 재학습 기준과의
output gap은 original보다 크다. 이 split에서 실제 client deletion이 만드는
functional change가 작다는 점을 함께 고려해야 한다.

## 삭제 대상과 shared client 확인

client 0은 삭제 대상이고, client 1은 동일 label 0·1을 가진 retained client다.

| Model | client 0 MSE | client 0 output gap | client 1 MSE | client 1 output gap |
| --- | ---: | ---: | ---: | ---: |
| `original` | 0.017139 | 0.000302 | 0.017318 | 0.000287 |
| `retrain_without_client_0` | 0.017209 | 0.000000 | 0.017363 | 0.000000 |
| `SCU_central_full` | 0.016215 | 0.001055 | 0.016541 | 0.000946 |

Full SCU는 삭제 대상과 shared retained client 모두에서 reconstruction 품질을
보존했다. 반면 양쪽 output gap은 original보다 커졌다. 즉 compensation은
shared semantic을 망가뜨리는 문제는 막았지만, 현재 목적식·가중치에서는
재학습 기준 방향으로 필요한 만큼만 이동시키지 못했다.

## Full SCU loss 진단

원본 SCU clone의 dynamic weighting에 맞춰 다음을 사용했다.

```math
L_{TOTAL}=L_{JU}+L_{CC}
```

여기서 `L_JU`는 target posterior-to-original KL과 negative target reconstruction
항이고, `L_CC`는 retained prior KL, retained reconstruction, cosine contrastive
compensation 항이다. 두 raw loss의 절댓값 비율로 detached dynamic weight를 계산한
뒤 합산한다.

| 지표 | 전체 5,000 step 평균 | 마지막 step |
| --- | ---: | ---: |
| 총 loss | 0.109589 | - |
| Joint-unlearning loss | 0.054315 | 0.024170 |
| Compensation loss | 0.114832 | 0.111171 |
| Contrastive loss | 0.092674 | 0.089668 |
| target reconstruction MSE | 0.020122 | 0.018060 |
| retained reconstruction MSE | 0.017868 | 0.017165 |
| joint weight | 0.257706 | 0.178584 |
| compensation weight | 0.742294 | 0.821416 |

보상항은 학습 평균에서 약 74%, 마지막 step에서는 약 82%의 dynamic weight를 받았다.
따라서 target-only loss가 decoder를 무너뜨리는 방향을 중앙 retained data가 강하게
상쇄한 것으로 해석할 수 있다.

## target-only 실행과의 관계

이전 `forget-only` artifact는 정식 비교에 사용하지 않는다. 그러나 같은 data split과
200 epoch 조건에서 full compensation을 넣었을 때의 안정성 차이는 분명하다.

| 방식 | 전체 MSE | client 0 MSE | output gap |
| --- | ---: | ---: | ---: |
| 이전 target-only `forget-only` (무효 기록) | 0.290229 | 0.336139 | 0.277976 |
| 이번 `SCU_central_full` | 0.014737 | 0.016215 | 0.000738 |

이는 후속 federated method가 풀어야 할 문제가 명확하다는 뜻이다. Full SCU는
중앙의 pooled `D_r`에서 remaining semantic knowledge를 받아 utility를 보호한다.
제안할 distributed repair는 raw `D_r`를 모으지 않고도 이 역할을 근사해야 한다.

## 학습량과 한계

FedAvg reconstruction loss는 여전히 완전히 평탄하지 않다.

| Phase | round 1 | round 100 | round 200 |
| --- | ---: | ---: | ---: |
| original | 0.063966 | 0.022677 | 0.018638 |
| retrain_without_client_0 | 0.063209 | 0.022220 | 0.018613 |

따라서 이 결과는 200-round, seed 0의 통제된 관측이다. 다음 단계에서는 적어도
여러 global seed와 SCU `lambda_rec`, `lambda_ss`, epoch sensitivity를 추가해
full SCU가 original보다 재학습 기준에 가까워지는 영역이 있는지 확인해야 한다.
Membership inference, latent client-identity leakage, SNR 변화는 아직 평가하지 않았다.
