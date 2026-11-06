# Proposed Unlearning Pipeline

## Overview

The method should focus on the stages after a client-level deletion request:

```text
1. Initial federated codec training
2. Client-level unlearning request
3. Joint codec forgetting
4. Distributed retain repair
5. Conflict-aware aggregation
6. Evaluation against retrain baseline
```

The main novelty should be stages 4 and 5.

## Step 6: Current evaluation contract (2026-11-02)

Original, retrain-without-target, unlearned codec 순서로 clean/erased PSNR·SSIM,
retain FID, marker MPR/PUL, Runtime을 비교한다. 어느 모델의 utility도 생략하지 않는다.
MPR은 검출 건수와 N을 함께 표시하며 PUL이 undefined여도 보고한다. S1은 연속
marker projection 강도/NCC를 함께 비교한다. S1/S2는 추가 전역 template-bank에서 outside
반응과 pixel-union coverage를 검사한다. Local/global 검출기를 구분하고 raw-control
FPR/TPR 및 한계를 보고한다. dFIL은 보조 진단이고 IDI/CKA/MIA는 핵심 판정에서 제외한다.
Local MPR v2는 raw clean 최대 반응 기준으로 고정한다. Amplitude/NCC는 같은 held-out
500장의 실제 측정값을 보고한다. 0/100점 환산이나 그 기준에 따른 NA 처리는 하지 않는다.
FSCU/FSCUv4 종료 후 channel audit가 같은 실행 안에서 core CSV/JSON과 figure를 저장한다.
Reference-only 실행에도 같은 core suite를 저장하며, 기존 reference checkpoint는
재학습하거나 덮어쓰지 않는다. 단순 계산 구현 완료와 실제 실험 검증 완료를 구분한다.

dFIL은 encoder의 pre-channel Gaussian posterior를 평가하는 **offline metric**이다.
Target forgetting loss나 remaining-client parameter Fisher 계산으로 사용하지 않는다.
출력 지표 역시 decoder만의 삭제 여부를 분리해서 증명하지 않는다.
상세 계산과 미검증 항목은
[확정 지표·구현 상태](../experiment/10-22-01-dfil-core-metrics.md)를 따른다.
아래 independent observer와 output agreement는 보조 평가이며 핵심 지표 목록을 대체하지 않는다.

## Independent remaining-client evaluator

평가 classifier는 위 codec optimization과 별도의 준비 단계다. Target을 제외한
remaining clients만 clean raw image와 원래 label로 CE/FedAvg 학습하며 classifier는
새로 초기화한다. Local train/validation index를 분리하고 sample-weighted validation
CE로 checkpoint를 선택한다. 공식 test는 선택이 끝난 뒤에만 평가한다.
이후 하나의 classifier를 freeze하고 decoded image의 stop-gradient/no-grad 경로에서만
호출한다. Classifier CE나 retrain agreement 지표는 forgetting/repair loss가 아니다.
세 codec의 class agreement, probability JS, logit MSE와 이미지 간 PSNR/SSIM을
clean/erased split별로 보고한다. Target/split이 바뀌면 observer provenance도 검증한다.

## Step 3: Joint Codec Forgetting

Target client `u` performs an active unlearning round before its local data is permanently deleted.

The purpose is to weaken target-client-specific contribution in both:

- encoder `f_theta`
- decoder `g_phi`

A basic forgetting loss can be structured as:

```text
L_forget =
    L_target_forgetting
  + beta L_proximal
  + gamma L_update_mask
```

Possible terms:

- `L_target_forgetting`: weakens target-client-specific reconstruction mapping
- `L_proximal`: prevents excessive drift from the original codec
- `L_update_mask`: focuses updates on target-sensitive parameters or latent subspaces

This stage alone is not the final contribution. Its job is to produce a forgotten-but-damaged codec that needs repair.

### Variational DeepJSCC에서 SCU loss 적용

기본 DeepJSCC는 deterministic channel code만 출력하므로 SCU의 posterior KL을 그대로
계산할 수 없다. 등록된 main codec v3는 마지막 encoder layer를 posterior mean과
softplus standard-deviation head로 확장하고 prior KL을 `batchmean_sum`으로 계산한다.
따라서 target client에서 다음 두 신호를 동시에 사용할 수 있다.

- current posterior와 frozen original posterior 사이의 encoder KL
- current decoder reconstruction과 frozen original decoder behavior 사이의 joint
  forgetting signal

Joint forgetting은 variational encoder와 DeepJSCC decoder 모두를 update한다. 이
변경은 local target-client forgetting을 가능하게 하기 위한 것이며, distributed retain
repair와 conflict-aware aggregation이라는 전체 연구 목표를 대체하지 않는다.

## Step 4: Distributed Retain Repair

This is the core contribution.

### SCU 이식 기준선: client JU + public-data server CC

본격적인 distributed retain repair 이전에, target client가 자신의 `D_u`로
SCU-style joint unlearning을 수행하고 server가 공개 데이터 `D_pub`로만
contrastive compensation을 수행하는 `FSCU` 기준선을 둔다.

```text
target client: G_current, G_fix, D_u -> short JU step, optional refreshed B_e
server:        G_JU, B_e, D_pub -> short public CC step -> next G_current
target/server: configured cycle만큼 교대 -> G_FSCU
```

Server는 raw `D_u`나 `D_k (k != u)`를 받지 않는다. 다만 `B_e`도 민감한
representation일 수 있으므로 raw-data-free라는 사실을 representation privacy
보장으로 해석하지 않으며, compensation 뒤 server buffer를 삭제하고 leakage를
진단한다. Public representations는 retained-client data의 대체물이라는 가설을
검증해야 한다. 상세 loss와 통신 가정은
`federated_scu_public_compensation.md`에 둔다.

Canonical latent는 power-normalized posterior mean으로 통일한다. 기본 FSCU는 JU 5
epoch와 CC 40 epoch의 총 budget을 5 cycle로 나누고, 매 cycle 뒤 erased buffer를
refresh한다. Original static, first post-JU static, refresh buffer와 stochastic self
positive를 ablation으로 비교한다. Posterior log-variance, entropy와 prior KL을 함께
기록해 stochastic positive의 variance-collapse 가능성을 점검한다.

이 assisted refresh 구조는 target client가 삭제 완료 전 여러 step에 참여할 수 있어야
한다. 또한 SCU의 IID 기반 retraining 분석을 non-IID 2-class나 Dirichlet split의
이론적 보장으로 사용하지 않는다.

코드에서 `FSCU`는 위 public-data 경로만 가리키고, conflict-aware distributed repair는
`FSCUv4`로 구분한다.


### Public data 없는 distributed decoder CC 기준선

Server public data를 제거한 기준선에서는 target client가 erased representation
buffer 없이 `G_JU`만 반환한다. Remaining client `k`는 자기 raw data `D_k`로
decoder-side reconstruction loss를 계산하고 model update만 server에 보낸다.

```text
target client:    G, D_u -> G_JU
remaining client: G_JU, D_k -> decoder-side Delta_k
server:           sample-weighted FedAvg({Delta_k})
```

Encoder는 `G_JU` state로 고정하고 decoder만 업데이트한다. 별도 semantic classifier는
고정 observer라 loss, 업데이트, 전송 대상이 아니다. 이 기준선은 raw-data centralization과 target latent 전송을 모두
피하지만, encoder 손상을 복구하지 않고 conflict-aware filtering도 적용하지 않는다.
따라서 full method가 아니라 distributed decoder repair의 하한선으로 해석한다.

### Remaining-client public-response consensus 탐색

Target 내부 model difference가 client-specific negative를 제공하지 못한 뒤,
remaining clients가 retraining behavior를 직접 제공할 수 있는지 별도로 검증한다.

```text
remaining client k:
    current global codec에서 retained data로 short local adaptation
    -> 동일 public inputs와 동일 AWGN noise에 대한 posterior/output 계산
    -> public responses만 server로 전송

server:
    sample-weighted response consensus 생성
    -> original보다 retrain에 가까운지 사전 gate로 검증
    -> gate 통과 시에만 joint-unlearned codec에 distillation
```

이 경로는 target negative를 사용하지 않는다. Response consensus는 raw remaining
data를 중앙화하지 않지만 local model behavior를 노출할 수 있으므로 privacy가
자동으로 보장된다고 주장하지 않는다. 또한 current global codec에서 짧게 적응한
remaining model들은 target client의 historical contribution을 여전히 포함한다.

Seed 0의 2-class와 Dirichlet 0.1에서는 prediction consensus가 original보다
retrain paired-output gap을 각각 1.39%, 7.18% 증가시켰다. 따라서 현재의 1-epoch
local adaptation과 직접 평균 방식은 gate를 통과하지 못했고 conditional
distillation은 실행하지 않았다. 이 결과는 public response averaging만으로 target의
historical influence가 제거되지 않음을 보여주는 negative result다.

### 삭제 representation 없는 distributed joint repair

SCU encoder contrastive CC를 remaining clients에서 그대로 계산하려면 target의
erased representation을 negative로 배포해야 한다. 이 전송을 피하기 위해 full
method 후보에서는 SCU-style JU와 retain repair의 역할을 분리한다.

```text
target client:
    original global codec + local target data
    -> joint encoder-decoder unlearning
    -> JU model update only
    -> target data deletion

remaining client k:
    JU codec + local retained data D_k
    -> channel-aware reconstruction repair
    -> optional retained-only original-model distillation
    -> encoder + decoder update only
    -> frozen semantic classifier may be used only for evaluation

server:
    retained-client updates
    -> FedAvg or conflict-aware filtering
    -> repaired unlearned codec
```

Local repair는 target data나 target representation을 사용하지 않는다. 기본 pilot
loss는 reconstruction과 VIB prior KL이며 original-model distillation은 0으로 꺼 둔다.
Classifier CE는 codec으로 역전파하지 않는다. Distillation을 켜는 ablation에서도
teacher는 retained client의 local input에서만 평가한다.

초기 conflict-aware rule은 full parameter-space JU update를 forgetting direction으로
두고, 그 방향과 음의 내적을 갖는 repair component를 client별로 제거한다. 이 방식은
구현이 단순하지만 JU가 shared reconstruction 손상까지 포함하면 지나치게 많은 repair
성분을 차단할 수 있다. 따라서 충돌 client 비율과 filtering 전후 cosine을 반드시
기록하고 retrain functional gap으로 판단한다.

### FSCUv4: retained-only feature distillation

FSCUv4는 위 distributed joint repair를 variational DeepJSCC에 다시 등록하고 frozen
Original teacher를 remaining-client local data에서만 사용한다. Target JU가 끝난 동일
checkpoint에서 세 repair branch를 시작한다.

```text
normal repair  = reconstruction + prior KL
output KD      = normal repair + paired-channel reconstruction distillation
feature KD     = normal repair + shallow encoder/decoder feature distillation
```

Teacher와 student는 posterior mean channel path와 동일한 AWGN realization을 사용한다.
Feature KD는 flattened feature map을 sample별 L2 normalize한 뒤 cosine direction을
맞춘다. Server는 JU update를 고정 forgetting direction으로 두고 각 remaining-client
repair update 중 그 방향과 음의 내적을 갖는 평행 성분을 제거한 뒤 sample-weighted
aggregation한다. Public data나 target representation은 사용하지 않는다. 상세 정의는
`fscuv4_retained_only_feature_kd.md`에 둔다.

### Step 3.5: 공개 데이터 기반 선택적 server repair

target local forgetting과 distributed retain repair 사이에는 탐색적 server 단계를
넣을 수 있다. 이 단계에서 server는 target 또는 remaining client의 raw data를 받지
않고, 별도로 준비된 공개 데이터 `D_pub`만 사용한다.

server가 사용하는 세 model state는 다음과 같다.

```text
G-u   = target client u를 제외한 마지막 FL round local model의 aggregation
U     = target client u의 정상 마지막-round local model
G-un  = target client u가 local unlearning한 model
```

각 state의 역할은 구분해야 한다.

- `G-u`는 remaining-client reference이며, target client를 처음부터 제외하고 다시
  학습한 기준 model은 아니다.
- `U`와 `G-u`는 `D_pub`에서 비교해 target client가 remaining clients와 다르게
  학습한 latent 영역을 찾는 데 쓴다.
- `G-un`은 server candidate의 시작점이다. target 삭제 신호를 담지만 retained
  semantic까지 손상했을 수 있다.

server는 `D_pub`에서 `U`와 `G-u`의 normalized posterior mean 차이가 큰 latent
coordinate를 target-specific 보호 영역으로 선택한다.

```text
선택된 target-specific coordinate
    -> candidate가 G-un에 가깝게 유지

선택되지 않은 coordinate
    -> candidate를 G-u 방향으로 복구
```

reconstruction도 같은 원리를 sample별로 적용한다. 선택 영역에서 `U`와 `G-u`의
차이가 큰 공개 sample일수록 `G-un` 보호 비중을 높이고, 그 외 sample은 `G-u` 쪽으로
더 복구한다.

이 방식은 `G-u` 전체를 `G-un`에 맞추는 direct matching과 의도적으로 다르다.
direct matching은 target local forgetting의 손상을 shared semantic 방향까지 복사할 수
있다. 이 선택적 단계는 target-specific information의 proxy일 뿐이므로, 반드시 full
retraining 기준과 privacy 진단으로 검증해야 한다.

After forgetting, the server broadcasts the forgotten codec `M^F` to remaining clients. Each remaining client `k != u` repairs the codec using only local retained data `D_k`.

A candidate local repair objective is:

```text
L_repair^k =
    L_rec^k
  + lambda_dist L_distill^k
  + lambda_cons L_codec_consistency^k
  + lambda_ch L_channel_robust^k
```

Term meanings:

- `L_rec^k`: preserve reconstruction quality on retained data
- `L_distill^k`: preserve useful original-model behavior on retained data only
- `L_codec_consistency^k`: maintain encoder-decoder latent/output consistency
- `L_channel_robust^k`: preserve reconstruction under channel variation

Important rule:

```text
Distillation must be applied only on remaining-client data, not target-client data.
```

Otherwise, the original model may reintroduce forgotten target-client knowledge.

## Step 5: Conflict-Aware Aggregation

Naive FedAvg repair can cancel forgetting. The server should filter repair updates that move the model back toward target-client contribution.

Let:

```text
Delta_F     = forgetting update direction
Delta_R^k   = repair update from remaining client k
```

If `Delta_R^k` conflicts with the forgetting direction, the conflicting component is removed or downweighted:

```text
Delta_R_filtered^k =
    Delta_R^k - Proj_conflict(Delta_R^k, Delta_F)
```

The server aggregates only filtered repair updates:

```text
Delta_R = Aggregate_k(Delta_R_filtered^k)
M^U = M^F + Delta_R
```

This separates the method from simple "unlearning followed by FL fine-tuning."

## Working Hypothesis

Distributed repair should recover retained reconstruction utility while conflict-aware aggregation prevents repair from reintroducing the forgotten client's contribution.
