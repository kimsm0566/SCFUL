# Reconstruction MIA Audit

## 목적

완료된 federated deletion run에서 삭제 client의 training sample이 model에 남아 있는지
reconstruction-loss 기반 black-box membership inference attack(MIA)으로 점검한다.

이는 target-client reconstruction MSE를 올리는 진단이 아니다. Original, retrain-without-
target, joint unlearning, FedSCU-PubCC가 target client의 과거 training sample을 얼마나
구별 가능하게 만드는지를 같은 공격으로 비교한다.

## 공격 설정

- Member: 삭제 대상 client 0의 원래 local training sample
- Non-member: 어떤 FL training이나 public compensation에도 사용하지 않은 CIFAR-10 test sample
- Class matching: 각 target label 안에서 member/non-member 수를 1:1로 맞춘다. 따라서 attack이
  label-distribution 차이만 이용할 수 없다.
- Feature: deterministic posterior-mean reconstruction MSE 하나. 공격자는 input image와 codec의
  reconstruction output만 query하는 black-box setting이다.
- Calibration/evaluation 분리: 각 class-matched pair의 절반으로 threshold와 방향을 선택하고,
  나머지 절반에서 AUC와 balanced accuracy를 측정한다.

Class label은 paired evaluation pool을 만들기 위해서만 사용한다. Attack feature나 threshold
feature에는 전달하지 않는다.

## 해석

- `mia_auc_low_loss_member`: 낮은 reconstruction loss를 member 증거로 사용할 때 ROC-AUC.
- `mia_auc_best_direction`: attack 방향을 바꿀 수 있는 공격자의 능력을 포함한 AUC. `0.5`가
  random guessing이고, 높을수록 membership leakage가 크다.
- `attack_balanced_accuracy`: 독립 evaluation half에서 calibration된 threshold의 balanced accuracy.
  `0.5`가 random guessing이다.

실제 삭제 효과는 FedSCU-PubCC의 leakage가 Original보다 줄고, retrain-without-target 기준에
가까운지로 판단한다. 한 seed의 AUC만으로 결론 내리지 않으며, 이후 seed 1, 2에서도 같은
attack을 반복한다.

## 실행

모든 실행은 Docker container에서만 한다. Source run은 leaf run directory여야 한다.

```bash
docker compose exec -T scful-dev python src/main.py \
  --experiment membership_inference_audit \
  --source-run outputs/2026-09-03-03-fedscu-dirichlet01-grid/20260903T060634Z_seed-0 \
  --source-run outputs/2026-09-03-04-fedscu-2class-grid/20260903T061248Z_seed-0 \
  --seed 0 --audit-device cuda:0 \
  --output-root outputs/2026-09-03-05-reconstruction-mia-audit
```

새 output에는 source run을 기록한 `metrics/reconstruction_mia_metadata.json`, model별
`metrics/reconstruction_mia_summary.csv`, class별 `metrics/reconstruction_mia_per_class.csv`를
저장한다. 이 audit은 offline evaluation이며, proposed unlearning/repair pipeline으로 raw
client data를 전송하거나 centralize하지 않는다.

## Seed 0 결과

Audit artifact:

```text
outputs/2026-09-03-05-reconstruction-mia-audit/20260903T065534Z_seed-0/
```

| Split | Model | Best-direction AUC | Held-out balanced accuracy |
| --- | --- | ---: | ---: |
| Dirichlet `alpha=0.1` | Original | 0.5170 | 0.4891 |
|  | Retrain without client 0 | 0.5159 | 0.4969 |
|  | Client JU only | 0.5028 | 0.5014 |
|  | FedSCU-PubCC | 0.5153 | 0.4915 |
| 2 classes/client | Original | 0.5145 | 0.4835 |
|  | Retrain without client 0 | 0.5132 | 0.4855 |
|  | Client JU only | 0.5073 | 0.4990 |
|  | FedSCU-PubCC | 0.5128 | 0.4990 |

두 split 모두 reconstruction-loss MIA의 AUC가 0.5에 가까워, 이 공격은 target
membership을 유효하게 구별하지 못했다. FedSCU-PubCC는 Original보다 AUC가 아주 조금
낮지만, Original 자체도 거의 random-guess 수준이다. 따라서 이 결과는 FedSCU-PubCC의
privacy 개선을 강하게 입증하지는 못한다. 다만 적어도 이 holdout-calibrated,
class-matched black-box attack에서 leakage가 악화되었다는 증거도 없다.

다음 평가에서는 seed 1, 2 반복 및 보다 강한 attack feature 또는 posterior-access
threat model을 별도로 보고해야 한다.
