# Public Data 없는 Distributed Decoder CC Seed-0 결과

## 질문

Target client의 joint unlearning 뒤 server public data를 사용하지 않고, remaining
clients가 각자의 local raw data로 SCU의 decoder compensation을 계산한 뒤 update만
server에 보내면 reconstruction utility와 deletion signal을 복구할 수 있는가?

## 구현한 Baseline

```text
Target client u
  D_u로 joint encoder-decoder unlearning
  -> G_JU만 server로 전송

Server
  G_JU를 remaining clients에 broadcast

Remaining client k != u
  local D_k로 decoder-side reconstruction CC
  -> Delta_k만 server로 전송

Server
  sample-weighted FedAvg(Delta_k)
  -> FedSCU_DistributedDecoderCC
```

Public calibration data와 erased representation buffer는 생성하거나 전송하지 않는다.
Server의 CIFAR-10 test set은 최종 평가에만 사용한다.

Local CC에서는 JU encoder를 고정하고 decoder와 receiver-side semantic head만
업데이트한다.

\[
L_{DecCC}^{k}
=20\,\operatorname{MSE}(\hat{x}_k,x_k)
+\frac{1}{30}\operatorname{CE}(h(\hat{x}_k),y_k).
\]

Semantic CE는 SCU-compatible backdoor 진단을 위해 joint training한 receiver-side
head를 복구하는 항이다. Core reconstruction CC는 MSE 항이다.

이번 pilot은 distributed CC 1 round, client당 local epoch 1이다. 각 remaining
client는 5,000장을 batch 200으로 25 step 처리하고, 전체 9 clients는 총 225 local
steps와 45,000 sample exposures를 사용한다. 이는 기존 Public-CC의 40,000 sample
exposures와 비슷하지만, 각 local update는 25 step 뒤 한 번만 평균되므로 server-side
200 sequential optimizer steps와 같은 optimization depth를 뜻하지 않는다.

## 설정과 실행

- Dataset: CIFAR-10 train 50,000장 전체를 client에 할당
- Clients: 10, client당 5,000장
- Original/Retrain: 200 rounds, local epoch 1
- Batch size: 200
- Channel: 20 dB AWGN
- Joint unlearning: 5 epochs
- Backdoor: SCU class 1 to 2, white 2x2 patch, target-local 6%=300장
- Global seed: 0

### 2-class/client

Config:
[`federated_cifar10_fedosd_2class_scu_distributed_decoder_cc.json`](../../src/configs/federated_cifar10_fedosd_2class_scu_distributed_decoder_cc.json)

```bash
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_2class_scu_distributed_decoder_cc.json \
  --seed 0 \
  --output-root outputs/2026-09-05-02-federated-scu-distributed-decoder-cc-2class
```

Artifact:

```text
outputs/2026-09-05-02-federated-scu-distributed-decoder-cc-2class/
  20260905T043512Z_seed-0/
```

### Equal-size Dirichlet alpha=0.1

Config:
[`federated_cifar10_dirichlet_01_scu_distributed_decoder_cc.json`](../../src/configs/federated_cifar10_dirichlet_01_scu_distributed_decoder_cc.json)

```bash
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_scu_distributed_decoder_cc.json \
  --seed 0 \
  --output-root outputs/2026-09-05-03-federated-scu-distributed-decoder-cc-dirichlet01
```

Artifact:

```text
outputs/2026-09-05-03-federated-scu-distributed-decoder-cc-dirichlet01/
  20260905T043939Z_seed-0/
```

Dirichlet target은 client 8이다. Public holdout 없이 class-1 sample 1,332장을
보유하며 그중 300장을 poison했다.

## 결과

현재 reconstruction MSE와 output gap은 posterior-mean/no-channel 평가다. Backdoor
ASR은 model 간 posterior와 AWGN randomness를 공유하는 paired 20 dB 평가다.

### 2-class/client

| Model | 전체 MSE | PSNR | Output gap to Retrain | Target MSE | Retained MSE | ASR | Clean semantic acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.013098 | 18.828 | 0.000459 | 0.014926 | 0.012895 | 68.8% | 15.53% |
| Retrain | 0.013449 | 18.713 | 0 | 0.015649 | 0.013205 | 2.9% | 22.52% |
| Client JU | 0.298914 | 5.245 | 0.285257 | 0.271811 | 0.301925 | 100.0% | 10.00% |
| Distributed Decoder CC | 0.029571 | 15.291 | 0.016707 | 0.028133 | 0.029730 | 94.5% | 11.61% |

Decoder CC는 JU와 Retrain 사이의 전체 MSE 손상 중 94.4%를 회복했다. 그러나 최종
MSE는 Retrain의 2.20배이고 output gap은 Original보다 36.4배 크다.
`ASR_original >> ASR_retrain` 조건은 성립하지만 JU와 Decoder CC 모두 ASR을
제거하지 못했다. 따라서 이 조건에서는 unlearning 실패다.

### Equal-size Dirichlet alpha=0.1

| Model | 전체 MSE | PSNR | Output gap to Retrain | Target MSE | Retained MSE | ASR | Clean semantic acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.012630 | 18.986 | 0.000256 | 0.011903 | 0.012711 | 23.2% | 25.67% |
| Retrain | 0.012714 | 18.957 | 0 | 0.012001 | 0.012793 | 14.0% | 26.73% |
| Client JU | 0.293032 | 5.331 | 0.278077 | 0.300175 | 0.292238 | 0.0% | 10.00% |
| Distributed Decoder CC | 0.041966 | 13.771 | 0.029799 | 0.041792 | 0.041986 | 1.0% | 14.81% |

Decoder CC는 JU와 Retrain 사이의 전체 MSE 손상 중 89.6%를 회복했다. 하지만 최종
MSE는 Retrain의 3.30배이고 output gap은 Original보다 116.3배 크다. ASR은 낮지만
Original과 Retrain의 차이가 9.2 percentage points뿐이므로 강한 backdoor 삭제
근거로 사용하지 않는다.

## Efficiency 및 통신

| Split | JU + Distributed CC time | Retrain 대비 speedup | Method 통신량 | Retrain 통신량 |
| --- | ---: | ---: | ---: | ---: |
| 2-class | 1.559 s | 72.35x | 81,795,600 bytes | 14,723,208,000 bytes |
| Dirichlet 0.1 | 1.537 s | 73.09x | 81,795,600 bytes | 14,723,208,000 bytes |

Algorithmic 통신량은 target JU의 model download/upload와 remaining 9 clients의 CC
model download/update upload를 포함하며 MPI framing은 제외한다. Retraining보다
180배 작고 erased representation buffer 통신은 0 byte다.

## 결론

이 실험은 다음을 확인했다.

1. Raw remaining-client data와 public data를 server에 보내지 않고 decoder CC
   update만 모으는 MPI 경로가 정상 동작한다.
2. 정식 checkpoint 비교에서도 최종 encoder는 JU encoder와 byte-identical하며
   decoder는 변경됐다.
3. Decoder-only distributed CC는 JU의 심각한 reconstruction 붕괴를 대부분
   복구하지만 Retrain 수준까지 복구하지 못한다.
4. Encoder를 JU 상태로 고정했기 때문에 latent functional gap은 전혀 복구되지
   않는다. Decoder만으로 이 손상을 흡수하기 어렵다.
5. 단순 1-round FedAvg decoder compensation은 현재 조건에서 client-level
   unlearning 성공 방법이 아니라 **distributed repair baseline**이다.

후속 full method가 필요하다면 remaining clients의 encoder-side compensation과
repair update가 forgetting을 되돌리지 않도록 하는 conflict-aware aggregation을
별도로 검증해야 한다.
