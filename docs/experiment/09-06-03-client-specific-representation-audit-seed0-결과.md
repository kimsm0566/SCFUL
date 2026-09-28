# Client-Specific Representation Audit seed 0 결과

## 결론

현재 정의한 `delta_local = z_C - z_U`는 target-client-specific contribution의
신뢰할 만한 proxy로 확인되지 않았다. 2-class와 Dirichlet 0.1 모두 true retraining
deletion direction과 약하게 정렬되거나 사실상 정렬되지 않았다. Local JU는 encoder
representation보다 decoder를 크게 손상시켰고, delta-local client-ID probe도 target
identity 신호가 강하지 않았다.

계획에 정의한 gate가 실패했으므로 `z_proxy + gamma * delta_local`을 synthetic
negative로 사용하는 full contrastive refinement는 구현하지 않았다.

## 설정

- CIFAR-10, 10 clients
- Split: balanced 2-class/client, equal-size Dirichlet alpha 0.1
- Public calibration data: train split에서 class별 100개, 총 1,000개
- Client data: client별 4,900개
- Original 및 target 제외 retrain: 200 rounds
- Batch size: 200
- Channel: 20 dB AWGN
- Target normal local training: 1 epoch
- Target SCU-style Joint Unlearning: 5 epochs
- Remaining-client proxy: 1 federated round, local epoch 1
- Client-ID probe: client별 test partition의 70% train, 30% test
- Seed: 0
- Backdoor: SCU CIFAR-10 class 1 to 2, target-local poison ratio 6%

## 생성한 모델

- `G_t`: original global codec
- `C_u`: `G_t`에서 target normal local training한 codec
- `U_u`: 같은 `G_t`에서 target local JU한 codec
- `G_proxy_-u`: `G_t`에서 remaining clients가 1 round training한 proxy
- `G_retrain_-u`: target 없이 처음부터 retraining한 gold standard

## Experiment A: local direction과 true deletion direction

| Split | Candidate direction | Flattened cosine | Sample cosine | Linear CKA | Norm ratio | Procrustes residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2-class | C minus U | 0.1858 | 0.1773 | 0.3569 | 0.2450 | 0.8359 |
| 2-class | C minus G control | 0.1762 | 0.1671 | 0.3698 | 0.2214 | 0.7965 |
| 2-class | G minus U control | 0.1710 | 0.1768 | 0.3135 | 0.0380 | 1.0250 |
| 2-class | G minus proxy control | -0.1301 | -0.1167 | 0.5769 | 0.0416 | 0.5214 |
| Dirichlet 0.1 | C minus U | 0.0056 | 0.0098 | 0.4967 | 0.1230 | 0.6689 |
| Dirichlet 0.1 | C minus G control | -0.0124 | -0.0097 | 0.5036 | 0.1273 | 0.6591 |
| Dirichlet 0.1 | G minus U control | 0.0822 | 0.1045 | 0.3679 | 0.0276 | 1.0842 |
| Dirichlet 0.1 | G minus proxy control | 0.0871 | 0.0886 | 0.4888 | 0.0758 | 0.6682 |

2-class에서 local direction cosine은 normal-local-training control보다 0.0096 높을
뿐이고 CKA는 오히려 낮다. Dirichlet에서는 cosine이 거의 0이며 다른 control보다
낮다. 따라서 `C_u - U_u`가 true client deletion direction을 선택적으로 포착한다고
보기 어렵다.

CKA는 rotation과 방향 부호에 불변이므로 단독 성공 근거로 사용하지 않았다.

추가 distribution distance:

| Split | Direction | RBF MMD squared | Sliced Wasserstein | Raw MSE |
| --- | --- | ---: | ---: | ---: |
| 2-class | C minus U vs true | 0.3877 | 0.2843 | 0.07834 |
| Dirichlet 0.1 | C minus U vs true | 0.4281 | 0.1974 | 0.04371 |

## Experiment B: client-ID leakage

| Split | Representation | Client-ID accuracy | Macro AUC | Target-client AUC |
| --- | --- | ---: | ---: | ---: |
| 2-class | Original | 20.30% | 0.7197 | 0.7387 |
| 2-class | C_u | 20.03% | 0.7199 | 0.7394 |
| 2-class | U_u | 20.13% | 0.7197 | 0.7386 |
| 2-class | Retrain | 19.63% | 0.7149 | 0.7420 |
| 2-class | C minus U | 20.73% | 0.7107 | 0.7237 |
| Dirichlet 0.1 | Original | 18.47% | 0.6498 | 0.5594 |
| Dirichlet 0.1 | C_u | 18.70% | 0.6500 | 0.5589 |
| Dirichlet 0.1 | U_u | 18.57% | 0.6498 | 0.5594 |
| Dirichlet 0.1 | Retrain | 18.77% | 0.6498 | 0.5668 |
| Dirichlet 0.1 | C minus U | 18.40% | 0.6477 | 0.5441 |

Chance accuracy는 10%다. `C_u - U_u`의 target-client AUC는 두 split 모두 retrain보다
낮다. 이 차이가 target-client identity를 더 집중적으로 담는다는 증거가 없다.

Non-IID class distribution이 client-ID 예측에 포함되므로 절대 leakage가 아니라
retrain과의 차이로 해석했다.

## Experiment C: shared-information preservation

| Split | Pair to retrain | CKA | SVCCA | Latent raw MSE | Paired-channel output MSE | Input reconstruction MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2-class | C_u | 0.9775 | 0.9910 | 0.09112 | 0.000884 | 0.01340 |
| 2-class | U_u | 0.9758 | 0.9913 | 0.07992 | 0.295347 | 0.30909 |
| Dirichlet 0.1 | C_u | 0.9967 | 0.9948 | 0.04368 | 0.000388 | 0.01294 |
| Dirichlet 0.1 | U_u | 0.9969 | 0.9948 | 0.04296 | 0.303949 | 0.32004 |

Encoder-only 구조 지표는 C_u와 U_u 모두 retrain에 매우 가깝고 지표별 순위도
혼재한다. 반면 U_u의 decoder output과 input reconstruction은 두 split 모두 완전히
붕괴한다. 따라서 local JU가 shared codec behavior를 보존하면서 target-specific
representation만 제거했다는 가정은 성립하지 않는다.

현재 JU loss의 KL term은 encoder posterior를 original에 가깝게 유지하는 반면
reconstruction ascent는 decoder를 크게 손상시킨다. 그 결과 encoder difference인
`C_u - U_u`는 JU로 제거한 semantic contribution보다 target normal local-training
drift의 영향을 크게 받는다.

## Remaining proxy 진단

| Split | Pair | Paired-channel output MSE | Latent raw MSE |
| --- | --- | ---: | ---: |
| 2-class | Original vs retrain | 0.000659 | 0.08085 |
| 2-class | Proxy vs retrain | 0.000674 | 0.08187 |
| Dirichlet 0.1 | Original vs retrain | 0.000281 | 0.04312 |
| Dirichlet 0.1 | Proxy vs retrain | 0.000326 | 0.04280 |

한 round remaining proxy도 original보다 retrain에 일관되게 가까워지지 않았다.
따라서 현재 `z_proxy`를 clean positive representation으로 간주하기도 어렵다.

## Backdoor sanity check

| Split | Original ASR | Retrain ASR | C_u ASR | U_u ASR | Proxy ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-class | 74.1% | 2.1% | 0.0% | 0.0% | 26.1% |
| Dirichlet 0.1 | 19.3% | 14.6% | 0.1% | 0.0% | 16.5% |

2-class marker는 강하지만 C_u가 정상 local training 한 번만으로 ASR 0%가 되었다.
이는 C_u가 target contribution을 안정적으로 보존한 model이라는 해석에 반한다.
Dirichlet marker는 Original과 Retrain 차이가 4.7 percentage points로 약하다.

## Gate 판정

| Gate | 판정 | 근거 |
| --- | --- | --- |
| Local direction과 true direction 정렬 | 실패 | cosine 0.186 및 0.006, control 대비 이점 없음 |
| U_u의 shared-information 보존 | 실패 | encoder 구조는 유사하지만 decoder output MSE 약 0.30 |
| Delta-local의 target-specific leakage | 실패 | target AUC가 retrain보다 높지 않음 |
| Stable remaining positive proxy | 실패 | 1-round proxy가 output 기준 retrain에 더 가까워지지 않음 |
| Backdoor marker separation | 2-class만 통과 | Dirichlet separation이 약함 |

## 다음 설계에 대한 의미

현재 방식으로 full contrastive refinement를 진행하면 synthetic negative가
client-specific contribution이 아니라 local-training drift와 JU-induced damage를
반영할 가능성이 크다. 우선 다음 중 하나로 target-specific signal 정의를 바꿔야 한다.

1. Parameter 또는 latent difference를 retained-client control directions로 residualize
2. JU 전체가 아니라 target-sensitive encoder mask에 제한된 unlearning direction 사용
3. 여러 local steps와 seeds에서 안정적인 공통 subspace만 추출
4. Backdoor 또는 target-exclusive signal로 specificity를 먼저 검증한 뒤 일반 데이터로 확장

## 실행 명령

```bash
docker compose exec -T scful-dev mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_2class_client_specific_representation_audit.json \
  --seed 0 \
  --output-root outputs/2026-09-06-05-client-specific-representation-audit-2class

docker compose exec -T scful-dev mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_client_specific_representation_audit.json \
  --seed 0 \
  --output-root outputs/2026-09-06-06-client-specific-representation-audit-dirichlet01
```

## Artifacts

- Smoke:
  `outputs/2026-09-06-04-client-specific-representation-audit-smoke/20260906T092129Z_seed-0`
- 2-class:
  `outputs/2026-09-06-05-client-specific-representation-audit-2class/20260906T092219Z_seed-0`
- Dirichlet 0.1:
  `outputs/2026-09-06-06-client-specific-representation-audit-dirichlet01/20260906T092640Z_seed-0`

각 run은 model checkpoint, direction/shared/leakage CSV, functional/backdoor summary,
public representation tensor, split manifest와 phase runtime을 포함한다.
