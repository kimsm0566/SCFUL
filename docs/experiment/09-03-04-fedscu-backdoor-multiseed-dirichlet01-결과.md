# FedSCU Backdoor Multi-seed 및 Dirichlet α=0.1 결과

## 범위

SCU-compatible backdoor를 포함한 `FedSCU-PubCC`를 다음 두 non-IID 조건에서
global seed 0, 1, 2로 평가했다.

- CIFAR-10, 10 clients, client당 4,900장
- Original/Retrain: FedAvg 200 rounds, local epoch 1
- Batch size 200, 20 dB AWGN
- Target-local SCU poison ratio 6%인 294장
- Joint unlearning 5 epochs
- Public-data contrastive compensation 40 epochs
- `lambda_contrastive=0.1`, `lambda_public_reconstruction=20`
- Public calibration data: CIFAR-10 train에서 class당 100장, 총 1,000장

비교한 model은 `Original`, target client를 제외한 `Retrain`,
`client_joint_unlearning`(`JU`), `FedSCU_PubCC`(`FedSCU`)이다.

현재 `fg_output`과 reconstruction MSE는 posterior-mean/no-channel 경로의 결과다.
Backdoor ASR만 model 간 같은 posterior noise와 AWGN noise를 쓰는 paired 20 dB
평가다. 따라서 이 문서의 `fg_output`을 아직 paired-channel reconstruction gap으로
부르지 않는다.

## 실행 설정과 Artifact

### 2-class/client

Config:
[`federated_cifar10_fedosd_2class_scu_backdoor.json`](../../src/configs/federated_cifar10_fedosd_2class_scu_backdoor.json)

```bash
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_2class_scu_backdoor.json \
  --seed <0|1|2> \
  --output-root outputs/<output-root>
```

- seed 0:
  `outputs/2026-09-03-09-federated-scu-backdoor-2class/20260903T080158Z_seed-0/`
- seed 1:
  `outputs/2026-09-03-10-federated-scu-backdoor-2class-multiseed/20260903T083650Z_seed-1/`
- seed 2:
  `outputs/2026-09-03-10-federated-scu-backdoor-2class-multiseed/20260903T084105Z_seed-2/`
- Target client: client 0

Seed 0은 이미 checkpoint hash 재현성을 확인한 canonical run을 재사용했다. Seed 1과
2만 새로 실행했다.

### Equal-size Dirichlet α=0.1

Config:
[`federated_cifar10_dirichlet_01_scu_backdoor.json`](../../src/configs/federated_cifar10_dirichlet_01_scu_backdoor.json)

```bash
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_scu_backdoor.json \
  --seed <0|1|2> \
  --output-root outputs/2026-09-03-12-federated-scu-backdoor-dirichlet01-multiseed
```

- seed 0: `20260903T085003Z_seed-0/`
- seed 1: `20260903T085427Z_seed-1/`
- seed 2: `20260903T085847Z_seed-2/`
- Target client: client 8

Target client는 결과를 본 뒤 선택하지 않았다. 실행 전에 세 split의 class-1 보유량을
검사했으며 client 8만 seed 0, 1, 2에서 각각 1,305장, 440장, 1,233장으로 SCU
poison 294장을 모두 제공할 수 있어 고정했다. Client 0은 seed 0에서 263장뿐이어서
사용할 수 없었다.

첫 seed 0 시도인
`outputs/2026-09-03-11-federated-scu-backdoor-dirichlet01-multiseed/`는 metric 병합
단계의 `retrain_without_client_0` 하드코딩 때문에 `completed` 전에 종료됐다. 이
불완전 artifact는 결과에서 제외했다. Runner가 실제 target ID를 checkpoint와 metric
reference에 반영하도록 고친 뒤 관련 테스트 29개를 통과시키고 index 12에서 다시
실행했다.

## Seed별 핵심 결과

| Split | Seed | Target | Original ASR | Retrain ASR | FedSCU ASR | FedSCU 전체 output gap | Retrain 대비 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-class | 0 | 0 | 74.1% | 2.1% | 0.3% | 0.001872 | 52.30x |
| 2-class | 1 | 0 | 89.7% | 10.4% | 2.5% | 0.001566 | 51.67x |
| 2-class | 2 | 0 | 44.4% | 1.4% | 3.2% | 0.001617 | 52.90x |
| Dirichlet 0.1 | 0 | 8 | 19.3% | 14.6% | 1.7% | 0.002165 | 51.75x |
| Dirichlet 0.1 | 1 | 8 | 3.4% | 0.0% | 1.3% | 0.001482 | 53.10x |
| Dirichlet 0.1 | 2 | 8 | 16.2% | 4.5% | 2.1% | 0.001720 | 51.75x |

## 3-seed 평균

표의 `±`는 global seed 세 개에 대한 sample standard deviation이다. `Target MSE`는
삭제 client test partition, `Retained MSE`는 나머지 client test partition의 동일
크기 가중 평균이다.

### 2-class/client

| Model | 전체 MSE | PSNR (dB) | Output gap to Retrain | Target MSE | Retained MSE | ASR | Clean semantic acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.013437 ± 0.001154 | 18.728 ± 0.367 | 0.000486 ± 0.000115 | 0.015497 ± 0.001691 | 0.013208 ± 0.001095 | 69.40 ± 23.01% | 16.53 ± 2.43% |
| Retrain | 0.013826 ± 0.001155 | 18.603 ± 0.361 | 0 | 0.016250 ± 0.001725 | 0.013557 ± 0.001092 | 4.63 ± 5.01% | 19.20 ± 3.23% |
| JU | 0.293713 ± 0.012359 | 5.323 ± 0.183 | 0.282480 ± 0.009054 | 0.265338 ± 0.012396 | 0.296865 ± 0.012373 | 33.33 ± 57.74% | 10.00 ± 0.00% |
| FedSCU | 0.015048 ± 0.001233 | 18.235 ± 0.352 | 0.001685 ± 0.000164 | 0.017320 ± 0.001776 | 0.014795 ± 0.001175 | 2.00 ± 1.51% | 29.87 ± 2.18% |

### Equal-size Dirichlet α=0.1

| Model | 전체 MSE | PSNR (dB) | Output gap to Retrain | Target MSE | Retained MSE | ASR | Clean semantic acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.013017 ± 0.000762 | 18.860 ± 0.251 | 0.000472 ± 0.000253 | 0.012956 ± 0.001472 | 0.013024 ± 0.000688 | 12.97 ± 8.43% | 25.88 ± 1.98% |
| Retrain | 0.012693 ± 0.000395 | 18.966 ± 0.134 | 0 | 0.012413 ± 0.000619 | 0.012724 ± 0.000383 | 6.37 ± 7.48% | 25.87 ± 3.74% |
| JU | 0.205731 ± 0.145400 | 8.244 ± 4.902 | 0.191344 ± 0.144564 | 0.200706 ± 0.146300 | 0.206289 ± 0.145386 | 0.00 ± 0.00% | 11.62 ± 2.80% |
| FedSCU | 0.014485 ± 0.000402 | 18.392 ± 0.120 | 0.001789 ± 0.000347 | 0.014152 ± 0.000292 | 0.014522 ± 0.000444 | 1.70 ± 0.40% | 31.83 ± 1.47% |

## Efficiency

| Split | FedSCU online time | Retrain 대비 speedup | FedSCU algorithmic communication | Retrain algorithmic communication |
| --- | ---: | ---: | ---: | ---: |
| 2-class | 2.142 ± 0.020 s | 52.29 ± 0.61x | 8,441,704 bytes | 14,723,208,000 bytes |
| Dirichlet 0.1 | 2.169 ± 0.021 s | 52.20 ± 0.78x | 8,441,704 bytes | 14,723,208,000 bytes |

현재 accounting에서 FedSCU의 논리적 payload는 model 1회 download, update 1회 upload,
262,144-byte erased latent buffer다. MPI framing은 제외한다. Retraining 대비 payload는
약 1,744배 작다.

## 해석

### 확인된 점

1. 2-class/client에서는 세 seed 모두 `ASR_original >> ASR_retrain`이 성립했다.
   FedSCU ASR은 평균 2.00%로 Retrain의 4.63%와 같은 낮은 범위에 들어갔다.
2. JU-only는 reconstruction과 clean semantic accuracy를 크게 손상했으며,
   Public-CC가 이 utility를 크게 복구했다.
3. Online unlearning은 두 split 모두 약 2.1초이며 full retraining보다 약 52배
   빨랐다.

### 아직 성공이라고 할 수 없는 점

1. 2-class에서 FedSCU의 전체 output gap 0.001685는 Original의 0.000486보다 약
   3.47배 크다. 전체 MSE도 Retrain보다 8.84% 높다.
2. Dirichlet 0.1에서도 FedSCU output gap 0.001789는 Original의 0.000472보다 약
   3.79배 크며, 전체 MSE는 Retrain보다 14.12% 높다.
3. 따라서 현재 방법은 backdoor marker 제거와 JU utility 복구는 보였지만,
   reconstruction 기준으로 Original보다 Retrain에 가까워지지는 않았다.
4. Dirichlet의 Original/Retrain ASR 차이는 seed별 4.7, 3.4, 11.7 percentage
   points에 그쳐 Original backdoor가 충분히 강하지 않다. 낮아진 FedSCU ASR을
   Dirichlet client-level 삭제의 강한 증거로 해석하지 않는다.

현재 결론은 다음과 같다.

```text
2-class: backdoor contribution 제거는 multi-seed로 재현됐지만 retrain functional
         equivalence는 달성하지 못했다.

Dirichlet α=0.1: 실행과 utility repair는 재현됐지만 backdoor 신호가 약하고
                 retrain functional gap도 개선되지 않았다.
```
