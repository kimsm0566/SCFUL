# Remaining-Client Consensus Experiment Plan

## 목적

이 실험은 target client 내부의 두 모델 차이에서 client-specific negative를
추정하는 대신, remaining clients의 public-data response consensus가
`retrain_without_target`의 유효한 proxy인지 먼저 검증한다.

핵심 목표는 다음과 같다.

```text
target client joint forgetting
-> remaining clients의 public response 수집
-> response consensus가 retrain을 근사하는지 검증
-> 검증 gate를 통과한 경우에만 consensus distillation
```

이 실험에서는 client-specific negative나 contrastive loss를 사용하지 않는다.

## 모델과 데이터 흐름

1. 전체 client로 학습한 current global codec을 `G_t`로 둔다.
2. target client를 제외하고 처음부터 다시 학습한 codec을
   `G_retrain_minus_u`로 둔다.
3. target client는 `G_t`에서 SCU joint unlearning을 수행해 `U_u`를 만든다.
4. 각 remaining client는 동일한 `G_t`에서 시작해 자신의 retained data로
   한 번의 local adaptation을 수행한다.
5. 서버의 동일한 public samples와 동일한 AWGN noise를 각 remaining client에
   제공한다.
6. 각 remaining client는 다음 response만 서버로 보낸다.

```text
posterior mean
channel을 통과한 reconstruction output
semantic logits (semantic head가 있는 경우)
```

Raw retained-client data는 서버로 보내지 않는다.

## Consensus 구성

각 response는 local sample 수를 가중치로 사용해 sample-wise 평균한다.

```text
z_R = weighted mean of remaining-client posterior means
x_hat_R = weighted mean of remaining-client reconstruction outputs
```

`x_hat_R`은 `z_R`을 하나의 decoder에 통과시킨 출력이 아니다. 서로 다른
remaining-client codec의 reconstruction을 직접 평균한 ensemble teacher
target이다.

비교를 위해 동일한 local states를 parameter FedAvg한
`remaining_parameter_fedavg_proxy`도 함께 만든다. 이것은 consensus 방법의
필수 구성요소가 아니라 prediction consensus와 parameter aggregation을
비교하기 위한 audit baseline이다.

## Retrain-proxy 검증

Public samples와 channel noise를 고정한 상태에서 다음 세 candidate를
`G_retrain_minus_u`와 비교한다.

- Remaining-client prediction consensus
- Original global model
- Remaining-client parameter FedAvg proxy

Representation 비교 지표:

- Posterior-mean raw MSE
- Linear CKA
- SVCCA
- Procrustes-aligned distance
- Flattened cosine similarity

Functional 비교 지표:

- Paired-channel reconstruction output MSE
- Public input reconstruction MSE
- Semantic-logit MSE

## Distillation 실행 gate

Consensus distillation은 자동으로 실행하지 않는다. 다음 두 조건을 모두
충족해야 한다.

```text
consensus posterior-mean MSE가 original 대비 5% 이상 개선
consensus paired-output MSE가 original 대비 5% 이상 개선
```

이 gate는 raw coordinate representation 지표만으로 결론을 내리지 않기 위해
functional output 조건을 동시에 요구한다. 둘 중 하나라도 실패하면
distillation을 중단하고 consensus 가설이 현재 설정에서 지지되지 않은 것으로
기록한다.

## Conditional consensus distillation

Gate를 통과한 경우에만 `U_u`를 초기값으로 다음 matching loss를 사용한다.

```text
representation matching loss
+ weighted output matching loss
+ optional semantic-logit matching loss
```

Teacher는 original model이나 target-client model이 아니라 remaining-client
response consensus다. Target-client data에 대한 original-model distillation은
수행하지 않는다.

## 비교 checkpoint

- Original global model
- Retrain without target client
- Target local joint-unlearned model
- Remaining-client parameter FedAvg proxy
- Remaining-consensus distilled model (gate 통과 시에만 생성)

최종 판단은 distilled model이 target data를 단순히 나쁘게 복원하는지가 아니라,
retained data와 target data 모두에서 retrain-without-target의 behavior에
가까워지는지를 기준으로 한다.

## 실험 설정

먼저 synthetic smoke test를 수행한 뒤 다음 두 CIFAR-10 seed-0 실험을 수행한다.

- 10 clients, 2 classes per client, 200 federated rounds, batch size 200
- 10 clients, Dirichlet alpha 0.1, 200 federated rounds, batch size 200

두 CIFAR-10 실험 모두 public calibration data는 class당 100개를 사용하고,
remaining-client local adaptation은 1 epoch으로 시작한다.

## 통신 및 privacy 기록

다음 항목을 별도로 기록한다.

- Public payload broadcast bytes
- Shared channel-noise broadcast bytes
- Remaining-client response upload bytes
- Parameter-proxy audit를 위한 추가 model-state upload bytes
- Target JU, remaining local adaptation, server distillation update steps
- Retraining 대비 wall-clock time

Public responses도 local model behavior를 노출할 수 있으므로 완전한 privacy
보장으로 간주하지 않는다. 이번 1차 구현은 individual client responses를
artifact에 저장하지 않고 aggregate consensus만 저장한다.
