# Client-Specific Representation Unlearning with Contrastive Refinement

## 0. 목적

현재 global codec에서 target client의 contribution을 shared contribution과
client-specific contribution으로 분해할 수 있는지 먼저 검증한다. 최종 목표는
federated-unlearned codec이 target client 없이 처음부터 학습한 retrain codec과
기능적으로 가까워지는 것이다.

이 문서의 첫 단계는 full contrastive method 구현이 아니다. 다음 가설이 성립하는지
확인하는 representation audit이다.

```text
target local trained representation
- target local unlearned representation
~= target client를 제거했을 때의 true representation shift
```

가설이 지지되지 않으면 이 차이를 synthetic negative로 사용하는 refinement를
진행하지 않는다.

## 1. 모델 구성

Deletion 직전 current global model을 `G_t`로 둔다. 모든 파생 model은 동일한 `G_t`
state에서 시작한다.

### 1.1 Target local trained model `C_u`

Target client가 `G_t`에서 시작하여 local data `D_u`로 정상 reconstruction training을
짧게 수행한다.

```text
G_t + target normal local training -> C_u
```

`C_u`는 target client contribution이 한 번 더 반영된 operational model이다. 이것이
target contribution만 포함하는 clean decomposition이라고 가정하지 않는다.

### 1.2 Target local unlearned model `U_u`

동일한 `G_t`에서 시작하여 target client가 `D_u`로 SCU-style joint encoder-decoder
unlearning을 수행한다.

```text
G_t + target local Joint Unlearning -> U_u
```

### 1.3 Remaining-client proxy `G_proxy_-u`

동일한 `G_t`에서 remaining clients가 각자의 local retained data로 짧게 정상
training하고 update만 server로 보낸다. Server는 이를 sample-weighted FedAvg한다.

```text
G_t + short retained-client federated training -> G_proxy_-u
```

이 model은 historical target influence를 이미 포함한 `G_t`에서 출발하므로 true
retrain model이 아니다. Retained direction을 제공하는 proxy로만 해석한다.

### 1.4 Gold-standard retrain `G_retrain_-u`

Target client를 제외하고 동일한 initialization에서 처음부터 federated retraining한
codec이다. 모든 가설 검증의 기준이다.

## 2. Public representation 구성

Server가 client split 전에 별도로 분리한 public/calibration data `D_pub`를 사용한다.
동일한 public input 순서에 대해 각 model의 posterior mean을 추출한다.

```text
z_t       = encoder(G_t, D_pub)
z_C       = encoder(C_u, D_pub)
z_U       = encoder(U_u, D_pub)
z_proxy   = encoder(G_proxy_-u, D_pub)
z_retrain = encoder(G_retrain_-u, D_pub)

delta_local = z_C - z_U
delta_true  = z_t - z_retrain
```

Posterior sample noise를 제거하기 위해 기본 representation은 posterior mean으로
고정한다.

## 3. Experiment A: local direction과 true deletion direction

`delta_local`과 `delta_true`를 같은 public sample 단위로 비교한다.

측정값:

- global flattened cosine similarity
- per-sample cosine similarity의 평균과 표준편차
- linear CKA
- orthogonal Procrustes-aligned normalized MSE
- median-bandwidth RBF MMD squared
- sliced Wasserstein distance
- raw MSE와 norm ratio는 supplementary

Cosine과 CKA는 높을수록 alignment가 강하다. 나머지 distance는 낮을수록 가깝다.
절대 임계값을 미리 성공 기준으로 만들지 않고, 다음 control direction들과 함께 본다.

- `z_C - z_t`: target normal local-training direction
- `z_t - z_U`: target local-unlearning direction
- `z_t - z_proxy`: remaining-client proxy shift

## 4. Experiment B: client-ID leakage

Public data는 client ID가 없으므로 leakage probe에는 client별 test partition을 사용한다.
각 candidate representation extractor로 모든 client test sample을 encoding하고,
client ID를 예측하는 multinomial linear probe를 학습한다.

평가 representation:

- `G_t` posterior mean
- `C_u` posterior mean
- `U_u` posterior mean
- `G_proxy_-u` posterior mean
- `G_retrain_-u` posterior mean
- sample별 `C_u posterior mean - U_u posterior mean`

각 client 안에서 deterministic train/test split을 만든다. Train split으로 feature
standardization과 logistic-regression probe를 fit하고 held-out split에서 다음을
측정한다.

- client-ID accuracy
- balanced accuracy
- macro one-vs-rest ROC-AUC
- target-client one-vs-rest ROC-AUC

Non-IID split에서는 client ID와 class distribution이 강하게 연결될 수 있으므로
절대 leakage가 낮아야 한다고 해석하지 않는다. 핵심은 candidate가 retrain과 얼마나
가까운지다. `delta_local` probe가 높더라도 그것만으로 target-specific contribution을
증명하지 않는다.

## 5. Experiment C: shared-information preservation

Local JU가 target-specific information만 제거하고 shared information을 보존한다면
`U_u`가 `C_u`보다 retrain 및 remaining proxy에 가까워져야 한다.

Public data에서 다음 pair를 비교한다.

- `C_u` vs `G_retrain_-u`
- `U_u` vs `G_retrain_-u`
- `C_u` vs `G_proxy_-u`
- `U_u` vs `G_proxy_-u`

측정값:

- linear CKA
- regularized SVCCA mean canonical correlation
- Procrustes-aligned normalized MSE
- posterior mean raw MSE
- deterministic decoder output MSE
- input reconstruction MSE

Primary gate는 `U_u`가 `C_u`보다 retrain에 더 가까운지다. Representation metric과
decoder output이 서로 다른 결론을 내면 full refinement를 보류한다.

## 6. Full contrastive refinement 진입 조건

다음 조건이 여러 seed와 두 non-IID split에서 반복될 때만 full method를 구현한다.

1. `delta_local`과 `delta_true`의 cosine 및 CKA가 control보다 의미 있게 높다.
2. `U_u`가 `C_u`보다 retrain에 representation과 decoder output 모두에서 가깝다.
3. `delta_local`이 단순히 전체 reconstruction 손상이나 class-distribution 차이만
   반영한다는 반증이 약하다.
4. Backdoor marker를 쓰는 경우 Original과 Retrain의 ASR separation이 충분하다.

## 7. 조건부 full refinement

가설이 통과한 경우에만 public data에서 다음 representation을 만든다.

```text
positive           = z_proxy
client_direction   = z_C - z_U
synthetic_negative = z_proxy + gamma * client_direction
candidate          = encoder(G_FU, D_pub)
```

Candidate representation은 positive와 가까워지고 synthetic negative와 멀어지도록
InfoNCE-style loss로 학습한다. Decoder utility는 public reconstruction loss로
보호한다. Optional distillation은 public data 또는 retained-client data에서만 사용하고
target raw data에는 적용하지 않는다.

초기 ablation:

- gamma
- temperature
- contrastive weight
- reconstruction weight
- proxy training rounds
- JU epochs
- posterior mean vs normalized posterior mean

Mutual-information 해석은 의도에 대한 설명으로만 사용한다. 명시적인 MI estimator나
variational bound가 없으면 exact MI optimization이라고 주장하지 않는다.

## 8. Pilot 조건

- CIFAR-10, 10 clients
- 2-class/client 및 equal-size Dirichlet alpha 0.1
- Original/retrain 200 rounds
- Batch size 200
- Public data 100 samples/class
- Client data 4,900 samples/client
- AWGN 20 dB
- Target local normal training 1 epoch
- Target SCU-style JU 5 epochs
- Remaining proxy 1 federated round, local epoch 1
- Seed 0부터 시작하고 가설이 유망할 때 multi-seed로 확장
- SCU-compatible backdoor는 보조 sanity check로 유지

## 9. 필수 산출물

- 모든 model checkpoint
- `delta_local` vs `delta_true` direction metrics
- shared-preservation pairwise metrics
- client-ID leakage probe metrics
- functional reconstruction/backdoor summary
- public split과 client split manifest
- target/proxy local-training diagnostics
- 사용한 seed, channel, public data 크기, runtime

## 10. Seed 0 pilot 상태

2-class와 Dirichlet alpha 0.1에서 1단계 audit를 완료했다. 두 조건 모두 full
contrastive refinement 진입 조건을 통과하지 못했으므로 refinement는 아직 구현하지
않았다. 결과와 해석은
[`09-06-03-client-specific-representation-audit-seed0-결과.md`](09-06-03-client-specific-representation-audit-seed0-결과.md)에 기록한다.
