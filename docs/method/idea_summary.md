# Research Idea Summary

## Goal

Apply client-level federated unlearning to reconstruction-oriented semantic communication.

The target system is an end-to-end semantic codec:

```text
source x -> encoder f_theta -> channel H -> decoder g_phi -> reconstruction x_hat
```

Multiple clients train this codec through federated learning. When client `u` requests deletion, the system should remove the contribution of `D_u` from the global encoder and decoder.

## Correct Unlearning Objective

### Current evaluation agreement (2026-11-02)

핵심 평가 지표는 PSNR, SSIM, retain FID, marker MPR/PUL, Runtime이다.
모든 표와 figure는 Original, Retrain-without-target, Unlearned(FSCUv4) 순서이며,
clean/erased PSNR·SSIM과 retain FID를 세 모델 모두 보고한다. MPR은 frozen detector의
절대 검출률로 건수와 N을 함께 표시한다. S1은 연속 marker projection 강도와 NCC도
함께 보고한다. S1/S2에는 고정 위치뿐 아니라 raw-calibrated template-bank의 전역/outside
반응과 검출 픽셀 합집합 coverage를 추가한다. 임의 색 번짐이나 삭제 증명으로 해석하지 않는다.
Local MPR v2는 raw clean 최대 반응을 기준으로 하며 이전 midpoint 결과와 구분한다.
Amplitude/NCC는 같은 held-out 500장의 실제 측정값으로 보고하며 0/100점 환산은 하지 않는다.
dFIL은 보조적인 encoder 입력 민감도 진단이며 별도의 인코더 삭제 합격 조건이 아니다.
정의·구현 위치·검증 상태는
[확정 지표 구현](../experiment/10-22-01-dfil-core-metrics.md)을 단일 기준으로 사용한다.
이하 과거 탐색 결과의 client-ID/latent gap 등은 역사적 진단이며 현재 핵심 지표가 아니다.
dFIL은 입력에 대한 Fisher로, 학습 parameter의 diagonal Fisher와 다르다.
낮은 dFIL이나 높은 PUL 하나로 client 삭제를 증명하지 않고 retraining 기준과
retained utility를 함께 해석한다. 최근 논의한 Fisher 기반 joint forgetting은
설계 가설이며, 이번 평가 구현이 그 방법론을 구현하거나 검증한 것은 아니다.

The goal is not to make target-client reconstruction fail arbitrarily.

The goal is:

```text
unlearned codec ~= codec retrained from scratch without client u
```

If the retrained-without-client codec can still reconstruct target-client-like samples because similar information exists in remaining clients, the unlearned codec may also reconstruct them. This is not a failure by itself.

## Main Contribution

The main contribution should be distributed repair.

SCU performs contrastive compensation with erased data and remaining data in a centralized setting. In federated learning, the server cannot access remaining clients' raw data. Therefore, compensation must be redesigned as a privacy-preserving distributed repair process.

The proposed direction:

```text
joint codec forgetting
-> distributed retain repair
-> conflict-aware aggregation
-> retrain-equivalent reconstruction behavior
```

## Registered Main Codec

앞으로 CIFAR-10 본 실험의 기본 codec은 `VariationalDeepJSCC-CIFAR10-v3`이다.
원본 DeepJSCC의 CIFAR-10용 5-layer convolutional encoder와 대칭적인 5-layer
transposed-convolutional decoder를 유지하고, encoder의 마지막 channel-symbol layer만
같은 크기의 posterior mean과 scale을 함께 출력하도록 확장한다. Scale은
`softplus(raw_scale) + 1e-6`으로 계산하며 초기 log-variance는 `-7`이다.

```text
original DeepJSCC-style encoder
-> posterior mean and softplus-standard-deviation heads
-> reparameterized latent sample
-> per-sample power normalization
-> physical channel
-> original DeepJSCC-style decoder
```

이 변경으로 SCU-style VIB loss와 encoder/decoder joint unlearning을 사용할 수 있다.
Prior KL은 latent dimension을 합한 뒤 batch에서 평균하는 `batchmean_sum` convention을
사용한다. 기본 posterior sample은 `16 x 8 x 8 = 1,024` real channel symbols이며,
두 real symbol을 한 complex channel use로 보면 512 complex channel uses와
channel bandwidth ratio `1/6`에 해당한다.

Public 1,000장을 제외한 CIFAR-10 train 49,000장을 codec 학습에 모두 사용하고,
IID 10-client에 client당 4,900장씩 배분한다. Codec model과 reconstruction config에는
encoder와 decoder만 포함한다. Semantic 평가자는 target client를 제외한 remaining
clients의 clean raw image로 새로 초기화해 독립적으로 연합학습한다. Client별 held-out
validation CE로 checkpoint를 선택하고 고정한 뒤 Original, Retrain, Unlearned에 같은
classifier를 적용한다. Classifier CE를 codec으로 역전파하지 않으며, classifier의
별도 FedAvg/optimizer/checkpoint는 codec과 공유하지 않는다. 과거 전체 49,000장으로
학습한 observer는 legacy 결과 재현용이며 target-free evaluator로 간주하지 않는다.

출력 비교는 erased/clean 입력에서 PSNR-to-Retrain·SSIM-to-Retrain과 공통 observer의
class agreement, probability JS, logit MSE를 함께 기록한다. 원본 이미지 대비 복원
품질과 두 모델 출력 사이의 유사성을 구분하며 latent 일치를 필수 성공 조건으로
두지 않는다. 이 지표들은 실증적인 행동 유사성이며 삭제의 완전한 증명은 아니다.

이 모델은 원본 DeepJSCC 자체가 아니라 **원본 구조의 variational adaptation**으로
표기한다. FedSFR의 federated optimization 기여와도 구분한다.

Code-level reconstruction model은 `VariationalDeepJSCC`, semantic observer는
`Cifar10Classifier`로 분리한다. Clean reconstruction baseline은 semantic loss 없이
codec을 reconstruction과 VIB prior KL로만 학습한다. 등록 v3 objective의 reconstruction
term은 각 sample의 `3 x 32 x 32` RGB value에 대한 squared error를 모두 합한 뒤 batch에서
평균하는 per-example pixel SSE다. 이는 PSNR과 RMSE-255 계산에 사용하는 normalized
per-pixel MSE와 구분한다. Frozen observer는 decoded image를 평가할 뿐이며 observer CE를
codec update에 사용하지 않는다.

## Immediate SCU Porting Baseline

Before the full distributed-repair method, evaluate a narrower baseline named
`FSCU`:

```text
target client short SCU-style joint-unlearning step
-> server short public-data compensation step
-> alternate for configured cycles
```

The target client sends a joint-unlearned codec update and a detached erased
representation buffer; the server uses only disjoint public data for
compensation. This is not a substitute for distributed retain repair and does
not by itself guarantee representation privacy. Its research question is
whether public data can approximate SCU's centralized retained-data
compensation. See `federated_scu_public_compensation.md`.

FSCU의 canonical contrastive representation은 power-normalized posterior mean이다.
기본 positive는 동일 public image의 두 augmentation에서 얻은 deterministic canonical
representation이며, erased buffer는 매 interleaving cycle 뒤 refresh한다. Original,
first post-JU static, refresh buffer와 stochastic-posterior positive는 ablation으로 둔다.
SCU의 IID 기반 분석은 non-IID client-level deletion에 그대로 적용하지 않고 모든
retrain equivalence를 실험적으로 검증한다.

`FSCU`는 이 repository에서 public-data compensation 기준선만 뜻한다. 현재 제안
방법은 별도 이름 `FSCUv4`로 등록한다.


## FSCUv4: Conflict-Aware Retained Repair

`FSCUv4`는 target-client SCU-style joint unlearning 뒤 remaining clients가 각자의
retained raw data에서만 frozen Original teacher를 사용하는 distributed repair
방법이다. Public data와 erased representation을 repair에 사용하지 않는다. 같은 JU
checkpoint에서 normal repair, paired-channel output KD, shallow encoder/decoder feature
KD를 비교한다. Server는 각 client repair update가 target JU 방향을 되돌리는지 검사하고,
음의 내적을 갖는 평행 성분을 client별로 제거한 뒤 sample-weighted aggregation한다.

Teacher와 student는 같은 retained input과 같은 AWGN draw를 사용한다. Feature KD는
첫 encoder PReLU와 RGB output 직전 decoder PReLU의 normalized feature direction을
보존한다. Target client는 repair에 참여하지 않으며 Original teacher를 target data에
적용하지 않는다. Codec objective에는 reconstruction과 VIB KL만 들어가고 frozen
classifier는 평가 observer로만 사용한다. 상세 방법은
`fscuv4_retained_only_feature_kd.md`에 둔다.

Variational DeepJSCC로 다시 실행한 seed 0 결과에서 public CC는 JU-only의 심한
reconstruction collapse를 두 non-IID split 모두 복구했다. 그러나 retrain paired-output
gap은 20 dB AWGN에서 original global model보다 약 2.99배와 3.26배 커졌다. 따라서 현재 고정
hyperparameter의 `FSCU`는 utility-recovery baseline으로는 작동하지만
retrain-equivalence를 개선한 최종 방법으로 보지 않는다. 상세 결과는
`docs/experiment/09-07-01-vdeepjscc-fedscu-public-cc-seed0-결과.md`에 둔다.

Public data가 없는 두 번째 baseline은 target-client JU 뒤 remaining clients가
각자의 raw retained data로 decoder-side reconstruction CC를 수행하고 update만
server에 보내 sample-weighted FedAvg하는 `FedSCU-DistributedDecoderCC`다.
Target raw data, remaining raw data, erased representation buffer는 전송하지 않는다.
Encoder는 JU state로 고정하므로 이 baseline은 distributed repair 통신 경로와
decoder utility 회복만 진단하며, full joint repair나 conflict-aware aggregation을
대체하지 않는다.

삭제 representation을 remaining clients에게 배포하지 않는 full-codec repair
경로는 `FedSCU-DistributedJointRepair`로 둔다. Target client는 SCU-style JU
state만 server에 반환하고 삭제 데이터를 폐기한다. Remaining clients는 local
retained data에서 encoder와 decoder를 함께 repair하고 model update만 보낸다.
고정 semantic classifier는 외부 평가 observer로만 사용하고 repair하거나 전송하지
않으며 classifier CE를 codec으로 역전파하지 않는다. Server는 다음 두 집계를 같은 JU
출발점에서 비교한다.

```text
DistributedJointRepair-FedAvg
    = sample-weighted aggregation of raw retained-client repair updates

DistributedJointRepair-ConflictAware
    = remove each repair update's component that reverses the JU direction
      -> sample-weighted aggregation
```

이 경로는 public data와 erased representation buffer를 모두 요구하지 않는다.
단, 전체 parameter 공간에서 하나의 JU 방향만 보호하는 projection은 shared utility
repair까지 제거할 수 있으므로 selective mask, layer-wise projection, 또는
target-sensitive subspace와 비교해야 한다.

또 다른 탐색 가설은 public input에서 target normal local model `C_u`와 target
local-unlearned model `U_u`의 latent 차이를 client-specific direction으로 사용하는
것이다. 이 가설은 full contrastive refinement 전에 true retraining direction,
shared-information preservation, client-ID leakage로 검증한다. Seed 0의 2-class와
Dirichlet 0.1 pilot에서는 방향 cosine이 각각 0.186과 0.006이었고 decoder utility가
붕괴했으므로 초기 가설은 통과하지 못했다. 현재 결과로는 이 차이를 synthetic
negative로 직접 사용하지 않는다. 상세 결과는
`docs/experiment/09-06-03-client-specific-representation-audit-seed0-결과.md`에 둔다.

이 실패 뒤에는 remaining clients가 동일한 public inputs에 반환한 posterior mean과
channel-conditioned reconstruction을 직접 sample-weighted averaging하여 retraining
proxy로 쓰는 `RemainingClientConsensus` 가설을 검증했다. 모든 local model은 현재
global model에서 시작해 retained data로 1 epoch 적응하며, 동일한 AWGN realization을
사용한다. Seed 0의 2-class에서는 consensus의 retrain output gap이 original보다
1.39% 커졌고, Dirichlet 0.1에서는 7.18% 커졌다. 따라서 두 split 모두 사전에 정한
5% representation/output 동시 개선 gate를 통과하지 못했다. 현재 구현에서는
consensus distillation을 실행하지 않으며, 단순 response averaging을 distributed
repair의 새 기본 방법으로 채택하지 않는다. 상세 결과는
`docs/experiment/09-06-05-remaining-client-consensus-seed0-결과.md`에 둔다.
