# Variational DeepJSCC Reference Bank: Seeds 0--2

## Purpose

Unlearning method와 hyperparameter를 바꿀 때마다 Original과 Retrain을 다시 200 rounds
학습하지 않도록, 현재 main setting의 immutable reference checkpoint bank를 생성했다.

이 문서의 결과는 unlearning 결과가 아니다. 각 scenario에서 다음 세 reference state를
만들고 검증한 결과다.

> 이 bank의 semantic head는 96-unit legacy head다. 현재 기본 256-unit
> head와 architecture fingerprint가 다르므로 새 실험에 재사용하지 않는다.

> 주의: 이 bank는 각 global round에서 local Adam을 새로 만든 legacy 설정으로
> 생성되었다. 2026-09-07에 기본 설정을 client별 Adam state 유지 방식으로 바꿨으므로,
> 이후 본 실험에서 이 checkpoint들을 새 설정의 reference로 재사용하면 안 된다.
> Artifact 자체는 과거 결과의 재현성을 위해 그대로 보존한다.

- Initial state
- Original: 10 clients 전체로 학습한 global model
- Retrain: target client를 제외하고 같은 initial state에서 처음부터 다시 학습한 model

## Fixed Training Setting

- Model: `variational_deepjscc_cifar10`
- Dataset: CIFAR-10
- Clients: 10
- Samples per client: 3,000
- Public reservation: class당 100장, 총 1,000장
- FL rounds: 200
- Local epochs: 1
- Batch size: 50
- Optimizer learning rate: 0.0001
- Training channel: AWGN 20 dB
- Backdoor: SCU CIFAR-10 protocol, target-local poison ratio 6%
- Splits: balanced 2-class/client, equal-size Dirichlet 0.1, and IID
- Seeds: 0, 1, 2
- 2-class target client: 0
- Dirichlet 0.1 target client: 8
- IID target client: 0

Reference configs:

- `src/configs/federated_cifar10_fedosd_2class_vdeepjscc_reference.json`
- `src/configs/federated_cifar10_dirichlet_01_vdeepjscc_reference.json`
- `src/configs/federated_cifar10_iid_vdeepjscc_reference.json`

## Commands

각 seed는 기존 persistent container `scful-dev`에서 다음 형식으로 실행했다.

```text
docker exec -w /workspace scful-dev mpirun --allow-run-as-root -np 11 \
  python src/main.py \
  --experiment federated_deletion \
  --config <reference-config> \
  --seed <0-or-1-or-2> \
  --output-root <unique-reference-output-root>
```

각 완료 run에는 다음 in-place standard audit를 적용했다.

```text
docker exec -w /workspace scful-dev python \
  src/main.py --experiment reconstruction_channel_audit \
  --source-run <reference-run-directory> \
  --seed <seed> --audit-device cuda:0 --snr_db 20 --in-place
```

## Reference Artifacts

| Split | Seed | Output root | Run directory | Fingerprint prefix |
| --- | ---: | --- | --- | --- |
| 2-class | 0 | `outputs/2026-09-07-10-reference-vdeepjscc-2class-seed0` | `20260907T045605Z_seed-0` | `3134d21c2ffd` |
| 2-class | 1 | `outputs/2026-09-07-11-reference-vdeepjscc-2class-seed1` | `20260907T050321Z_seed-1` | `033889e60b8f` |
| 2-class | 2 | `outputs/2026-09-07-12-reference-vdeepjscc-2class-seed2` | `20260907T051030Z_seed-2` | `6653b53fcd2a` |
| Dirichlet 0.1 | 0 | `outputs/2026-09-07-13-reference-vdeepjscc-dirichlet01-seed0` | `20260907T051741Z_seed-0` | `516c6f24bbbb` |
| Dirichlet 0.1 | 1 | `outputs/2026-09-07-14-reference-vdeepjscc-dirichlet01-seed1` | `20260907T052452Z_seed-1` | `21c04f09ed43` |
| Dirichlet 0.1 | 2 | `outputs/2026-09-07-15-reference-vdeepjscc-dirichlet01-seed2` | `20260907T053201Z_seed-2` | `7288703b0cc5` |
| IID | 0 | `outputs/2026-09-07-16-reference-vdeepjscc-iid-seed0` | `20260907T073607Z_seed-0` | `89d862ff5a06` |
| IID | 1 | `outputs/2026-09-07-17-reference-vdeepjscc-iid-seed1` | `20260907T074326Z_seed-1` | `e0d7b8fbdf38` |
| IID | 2 | `outputs/2026-09-07-18-reference-vdeepjscc-iid-seed2` | `20260907T075044Z_seed-2` | `251ec5405260` |

각 run의 `reference_manifest.json`에는 complete scenario, full fingerprint, checkpoint
relative path, byte size, SHA-256, Original/Retrain wall time이 기록되어 있다. 후속
unlearning run은 fingerprint와 SHA-256을 다시 확인한 뒤 reference를 load해야 한다.

## Headline Metrics

모든 reconstruction 결과는 official CIFAR-10 test set 또는 target-client erased set에
대해 paired AWGN 20 dB에서 측정했다. Accuracy, ASR, MIA, BA는 0--1이 아니라 읽기
쉽도록 Accuracy/ASR/BA만 percentage로 표시했다. MIA는 ROC-AUC다.

| Split | Seed | Model | Clean RMSE-255 | Erased RMSE-255 | Clean PSNR | Erased PSNR | Accuracy (%) | ASR (%) | MIA AUC | BA (%) | Runtime (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-class | 0 | Original | 21.739 | 22.810 | 21.386 | 20.968 | 14.40 | 82.10 | 0.478 | 82.78 | 217.89 |
| 2-class | 0 | Retrain | 22.035 | 23.199 | 21.269 | 20.822 | 23.52 | 1.70 | 0.480 | 0.00 | 197.19 |
| 2-class | 1 | Original | 22.591 | 23.782 | 21.052 | 20.606 | 13.78 | 84.50 | 0.458 | 85.56 | 218.99 |
| 2-class | 1 | Retrain | 23.155 | 24.548 | 20.838 | 20.330 | 20.12 | 16.80 | 0.455 | 11.11 | 199.47 |
| 2-class | 2 | Original | 23.145 | 24.417 | 20.842 | 20.377 | 12.93 | 87.70 | 0.458 | 86.11 | 220.52 |
| 2-class | 2 | Retrain | 23.188 | 24.530 | 20.826 | 20.337 | 21.90 | 14.60 | 0.456 | 16.67 | 200.44 |
| Dirichlet 0.1 | 0 | Original | 22.336 | 22.087 | 21.151 | 21.248 | 30.89 | 12.20 | 0.483 | 11.11 | 219.89 |
| Dirichlet 0.1 | 0 | Retrain | 22.319 | 22.056 | 21.157 | 21.260 | 32.51 | 12.10 | 0.482 | 9.44 | 199.01 |
| Dirichlet 0.1 | 1 | Original | 23.043 | 21.865 | 20.880 | 21.336 | 31.17 | 3.10 | 0.462 | 3.33 | 219.19 |
| Dirichlet 0.1 | 1 | Retrain | 22.529 | 21.279 | 21.076 | 21.572 | 27.04 | 0.00 | 0.462 | 0.00 | 199.50 |
| Dirichlet 0.1 | 2 | Original | 23.222 | 22.764 | 20.813 | 20.986 | 29.89 | 14.80 | 0.445 | 18.89 | 222.77 |
| Dirichlet 0.1 | 2 | Retrain | 22.500 | 22.012 | 21.087 | 21.278 | 28.45 | 5.50 | 0.445 | 6.67 | 202.01 |
| IID | 0 | Original | 22.519 | 22.584 | 21.080 | 21.055 | 40.97 | 2.50 | 0.504 | 0.56 | 223.67 |
| IID | 0 | Retrain | 21.371 | 21.455 | 21.534 | 21.500 | 42.05 | 1.40 | 0.502 | 0.56 | 202.86 |
| IID | 1 | Original | 22.063 | 22.314 | 21.258 | 21.159 | 42.50 | 1.50 | 0.472 | 1.67 | 222.64 |
| IID | 1 | Retrain | 22.412 | 22.711 | 21.121 | 21.006 | 41.50 | 0.60 | 0.472 | 1.11 | 202.02 |
| IID | 2 | Original | 22.552 | 22.422 | 21.067 | 21.117 | 40.12 | 2.10 | 0.487 | 1.11 | 224.53 |
| IID | 2 | Retrain | 22.976 | 22.875 | 20.905 | 20.944 | 39.83 | 1.30 | 0.488 | 1.67 | 203.16 |

`metrics/headline_metrics.csv`에 위 headline 값과 exact field name이 저장되어 있다.
Normalized MSE와 추가 MIA 진단은 `metrics/headline_extended_metrics.csv`와
`metrics/reconstruction_mia_summary.csv`에 남긴다.

## Validation

- 9개 run 모두 200 FL rounds, 10 clients, client당 3,000장, batch 50, AWGN 20 dB 확인
- 9개 scenario fingerprint가 모두 서로 다름
- 총 27개 Initial/Original/Retrain checkpoint의 byte size와 SHA-256 재계산 일치
- 9개 `metrics/headline_metrics.csv` 존재
- Original/Retrain 18개 row에서 RMSE-255, PSNR, Accuracy, ASR, MIA, BA, Runtime 누락 없음

2-class에서는 세 seed 모두 Original의 ASR/BA가 Retrain보다 뚜렷하게 높다. Dirichlet
0.1에서는 seed 0의 Original과 Retrain ASR/BA 분리가 거의 없고 seed 1도 backdoor가
약하다. 따라서 이 reference들은 reconstruction과 retrain-equivalence 비교에는 사용할
수 있지만, Dirichlet seed 0의 backdoor 결과만으로 unlearning 성공 여부를 판정해서는
안 된다.

IID에서는 세 seed 모두 Original ASR이 1.5--2.5%, BA가 0.56--1.67%로
backdoor가 사전에 거의 학습되지 않았다. 따라서 IID reference는 reconstruction,
Accuracy, MIA, retrain-equivalence 비교에는 사용하되, 현재 poison 6% 조건의
ASR/BA로는 unlearning을 판정하지 않는다. IID backdoor 진단이 필요하면
poison ratio를 단순히 키우기 전에 target-client marker가 global model에 학습되는
조건을 별도 ablation으로 재설계해야 한다.
