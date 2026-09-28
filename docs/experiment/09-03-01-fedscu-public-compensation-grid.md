# FedSCU-PubCC Grid Search Plan

## 목적

`FedSCU-PubCC`가 retrain-without-target codec에 가까워지도록 client-side joint
unlearning(JU)과 server-side public contrastive compensation(CC)의 hyperparameter를
탐색한다.

원본 FL model과 retrain-without-client model은 각 data split에서 한 번만 200 rounds
학습한다. 그 뒤 두 checkpoint를 재사용해 JU와 CC 조합을 탐색하므로, 각 candidate마다
FL을 다시 학습하지 않는다.

## 고정 조건

- Dataset: CIFAR-10
- Clients: 10
- FL rounds: 200
- Local epochs: 1
- Batch size: 200
- Channel: AWGN, 10 dB train SNR
- Target client: client 0
- Public calibration data: training split 전 class별 100장 holdout, 총 1,000장
- Erased representation buffer: posterior mean 1,024개
- Screening: public test class별 100장, 총 1,000장

## Data Splits

1. Equal-size Dirichlet non-IID: `alpha = 0.1`
2. FedOSD balanced pathological split: client당 class 2개

공개 calibration data 1,000장을 학습 client 분할 전에 제외하므로 각 client는
4,900개 sample을 받는다. Pathological split에서는 각 label이 두 client에 나뉘어
client당 두 label, label당 2,450개가 배정된다.

## Grid

| Hyperparameter | Values |
| --- | --- |
| JU epochs | 5, 10, 20, 40 |
| Server CC epochs | 5, 10, 20, 40 |
| `lambda_contrastive` | 0.1, 0.5, 1.0 |
| `lambda_public_reconstruction` | 1, 5, 10, 20 |

각 split은 `4 x 4 x 3 x 4 = 192` candidates이며, 두 split을 합쳐 384 candidates다.

## Selection Rule

각 candidate는 public test screening subset에서 다음 lexicographic rule로 비교한다.

```text
minimize(
  functional gap output,
  functional gap posterior mean,
  functional gap posterior KL,
  reconstruction MSE
)
```

Grid에서는 screening metric으로 빠르게 순위를 정하고, 각 split의 최상위 candidate에
대해서만 전체 public test 및 client별 평가를 수행한다. 최종 해석은 target MSE가 아니라
retrain-without-target와의 functional gap 및 retained-client reconstruction utility에
기반한다.

## Configs and Outputs

- Dirichlet: `src/configs/federated_cifar10_dirichlet_01_scu_public_compensation_grid.json`
- 2-class pathological: `src/configs/federated_cifar10_fedosd_2class_scu_public_compensation_grid.json`

각 grid invocation은 다음 output root 규칙을 따른다.

```text
outputs/2026-09-03-01-fedscu-dirichlet01-grid/
outputs/2026-09-03-02-fedscu-grid-gpu-smoke/
outputs/2026-09-03-03-fedscu-dirichlet01-grid/
outputs/2026-09-03-04-fedscu-2class-grid/
```

각 output에는 최상위 JU/FedSCU-PubCC checkpoint와 모든 candidate의
`metrics/public_compensation_grid.csv`를 저장한다.

## 실행

모든 실행은 Docker container에서만 수행한다.

```bash
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_scu_public_compensation_grid.json \
  --seed 0 --output-root outputs/2026-09-03-03-fedscu-dirichlet01-grid
```

```bash
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_2class_scu_public_compensation_grid.json \
  --seed 0 --output-root outputs/2026-09-03-04-fedscu-2class-grid
```

이 첫 grid는 seed 0 screening이다. 최상위 setting은 이후 seed 1, 2에서 독립적으로
재실행해 variance를 보고한다.
