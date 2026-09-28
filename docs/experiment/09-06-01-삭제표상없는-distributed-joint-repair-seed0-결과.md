# 삭제 표상 없는 Distributed Joint Repair seed 0 결과

## 질문

SCU-style target-client Joint Unlearning 뒤에 erased representation이나 public
data를 전송하지 않고, remaining clients가 local retained data로 encoder와 decoder를
함께 repair하면 retraining 기준에 가까워지는가? 또한 full parameter-space JU
direction을 보호하는 conflict-aware projection이 forgetting을 보존하면서 utility를
회복하는가?

## 설정

- Dataset: CIFAR-10
- Clients: 10
- Target: 2-class는 client 0, Dirichlet 0.1은 client 8
- Split: balanced 2-class/client 및 equal-size Dirichlet alpha 0.1
- Original/retrain: 200 federated rounds, local epoch 1
- Batch size: 200
- Channel: 20 dB AWGN
- Target JU: 5 local epochs
- Distributed repair: 1 round, retained client local epoch 1
- Local repair: encoder, decoder, receiver semantic head 공동 update
- Local loss: reconstruction weight 20, VIB prior KL weight 0.0001,
  semantic weight 1/30, original-model distillation weight 0
- Backdoor: SCU CIFAR-10 protocol, class 1 to class 2, target-local poison 6%
- Seed: 0
- Public calibration data: 사용하지 않음
- Erased representation buffer: 전송하지 않음

## 실행 명령

```bash
docker compose exec -T scful-dev mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_2class_scu_distributed_joint_repair.json \
  --seed 0 \
  --output-root outputs/2026-09-06-02-distributed-joint-repair-2class

docker compose exec -T scful-dev mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_scu_distributed_joint_repair.json \
  --seed 0 \
  --output-root outputs/2026-09-06-03-distributed-joint-repair-dirichlet01
```

## 2-class 결과

| Model | All MSE | PSNR | Output gap to retrain | Target MSE | Retained MSE | ASR | Clean semantic acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.013098 | 18.828 | 0.000459 | 0.014926 | 0.012895 | 68.8% | 15.53% |
| Retrain without client 0 | 0.013449 | 18.713 | 0 | 0.015649 | 0.013205 | 2.9% | 22.52% |
| Joint Unlearning only | 0.298914 | 5.245 | 0.285257 | 0.271811 | 0.301925 | 100.0% | 10.00% |
| Joint repair + FedAvg | 0.022050 | 16.566 | 0.007813 | 0.025267 | 0.021693 | 83.1% | 12.02% |
| Joint repair + conflict-aware | 0.045959 | 13.376 | 0.032076 | 0.051972 | 0.045290 | 100.0% | 10.00% |

FedAvg joint repair는 JU가 만든 MSE 손상의 96.99%를 회복했지만 최종 MSE는
retrain의 1.64배이고 ASR은 83.1%로 높다. Strong marker 조건에서 client-level
unlearning 성공으로 볼 수 없다.

Conflict-aware projection은 9개 client update 전부를 충돌로 판정했다. 평균 cosine은
filter 전 -0.39585에서 filter 후 거의 0이 되었지만, 필요한 utility repair까지
제거하여 MSE와 functional gap을 악화시켰다.

## Dirichlet 0.1 결과

| Model | All MSE | PSNR | Output gap to retrain | Target MSE | Retained MSE | ASR | Clean semantic acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.012630 | 18.986 | 0.000256 | 0.011903 | 0.012711 | 23.2% | 25.67% |
| Retrain without client 8 | 0.012714 | 18.957 | 0 | 0.012001 | 0.012793 | 14.0% | 26.73% |
| Joint Unlearning only | 0.293032 | 5.331 | 0.278077 | 0.300175 | 0.292238 | 0.0% | 10.00% |
| Joint repair + FedAvg | 0.023255 | 16.335 | 0.010310 | 0.022424 | 0.023348 | 0.2% | 13.20% |
| Joint repair + conflict-aware | 0.040789 | 13.895 | 0.027169 | 0.039504 | 0.040932 | 0.0% | 10.01% |

FedAvg joint repair는 JU MSE 손상의 96.24%를 회복했지만 최종 MSE는 retrain의
1.83배다. ASR은 retrain보다 낮지만 Original과 Retrain 차이가 9.2 percentage
points뿐이므로 이 split의 backdoor marker는 약하다. 낮은 ASR만으로 성공을 주장할
수 없다.

Conflict-aware projection은 이 분포에서도 9개 update 전부를 충돌로 판정했다.
평균 cosine은 filter 전 -0.43113에서 filter 후 거의 0이 되었고, MSE는 FedAvg보다
나빠졌다.

## Decoder-only pilot과 비교

| Split | Decoder-only MSE | Joint FedAvg MSE | Decoder-only output gap | Joint FedAvg output gap | Decoder-only ASR | Joint FedAvg ASR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-class | 0.029571 | 0.022050 | 0.016707 | 0.007813 | 94.5% | 83.1% |
| Dirichlet 0.1 | 0.041966 | 0.023255 | 0.029799 | 0.010310 | 1.0% | 0.2% |

Encoder까지 repair하는 것은 두 분포 모두 decoder-only보다 utility와 retrain output
gap을 개선했다. 따라서 joint repair 방향 자체는 유효하다. 그러나 retain-only
reconstruction repair만으로 strong 2-class backdoor contribution을 제거하지는 못했다.

## 효율

| Split | Retrain time | Proposed online time | Speedup | Proposed communication | Retrain communication |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-class | 113.954 s | 1.796 s | 63.47x | 81,795,600 B | 14,723,208,000 B |
| Dirichlet 0.1 | 112.952 s | 1.709 s | 66.08x | 81,795,600 B | 14,723,208,000 B |

Online time은 target JU와 conflict-aware repair만 합산한다. FedAvg ablation을 함께
실행한 실험 전체 wall time과는 구분한다. 통신량은 raw tensor byte 기준의 logical
model download/upload이며 MPI framing은 제외한다.

## 결론

1. Erased representation과 public data 없이 remaining clients가 full codec을 repair하는
   통신 경로는 정상 작동한다.
2. Encoder joint repair는 decoder-only repair보다 reconstruction과 output gap을 크게
   개선한다.
3. 현재의 full parameter-space 단일-direction projection은 모든 client update를
   conflict로 판정하여 utility repair를 과도하게 제거한다.
4. 2-class strong-marker 기준으로 FedAvg와 conflict-aware 방식 모두 retraining과
   충분히 동등하지 않으며 backdoor deletion에도 실패했다.
5. 다음 conflict-aware 설계는 full update가 아니라 target-sensitive layer, parameter
   mask, 또는 latent subspace에만 적용해야 한다. 이 결과는 selective protection이
   필요한 근거다.

## Artifacts

- Smoke:
  `outputs/2026-09-06-01-distributed-joint-repair-smoke/20260906T085906Z_seed-0`
- 2-class:
  `outputs/2026-09-06-02-distributed-joint-repair-2class/20260906T085957Z_seed-0`
- Dirichlet 0.1:
  `outputs/2026-09-06-03-distributed-joint-repair-dirichlet01/20260906T090423Z_seed-0`

각 full run은 `completed` 로그, 6개 checkpoint, summary/per-client/per-class/backdoor/
efficiency metric을 포함한다. Erased representation buffer artifact는 없다.
