# FSCU Interleaved Canonical-Latent Experiment Plan

## 목적

Public-data FSCU의 기존 sequential 구조와 latent 불일치를 정리한다.

```text
기존: target JU 전체 완료 -> 고정 post-JU buffer -> server CC 전체 완료
변경: 짧은 target JU -> optional buffer refresh -> 짧은 server CC -> 반복
```

이 실험은 `FSCU` public-data baseline을 안정화한다. Remaining-client distributed
repair와 conflict-aware aggregation은 `FSCUv4` 범위로 둔다.

## 고정 조건

- Codec: `variational_deepjscc_cifar10`
- Dataset: CIFAR-10
- Clients: 10
- Samples per client: 3,000
- Splits: balanced 2-class/client, equal-size Dirichlet 0.1
- Batch size: 50
- Original and retrain training: 200 FL rounds
- Training/evaluation channel: AWGN 20 dB
- Seeds: 0, 1, 2
- Target deletion: one whole client
- Reference checkpoints: compatible completed Original/Retrain run reuse

## Canonical representation

FSCU의 buffer, deterministic contrastive branch와 representation diagnostics는 모두
다음을 사용한다.

```text
Z_canonical = sqrt(latent_dim) * posterior_mean / posterior_mean_norm
```

이는 main Variational DeepJSCC codec에서 AWGN 직전의 deterministic channel input이다.
Raw posterior mean, sampled posterior와 channel output을 같은 이름 `Z`로 섞어 쓰지 않는다.

## 기본 interleaving 설정

- Total target JU epochs: 5
- Total server CC epochs: 40
- Interleaving cycles: 5
- Per cycle: JU 1 epoch, CC 8 epochs
- Target/server Adam state: 각 역할 안에서 cycle 사이에 유지
- Negative buffer: 매 JU 뒤 refresh
- Buffer sampling: label-stratified, 최대 1,024개
- Public positive: 동일 image의 random crop/flip 두 view에서 deterministic canonical latent
- Public prior KL weight: 0
- Decoder preservation weight: 0
- Erased semantic CE ascent: enabled

총 JU/CC update budget은 기존 sequential FSCU와 동일하게 유지한다.

## 순차 실험

### A. Interleaving 효과

다른 조건과 총 epoch budget을 고정하고 다음을 비교한다.

- Sequential: 1 cycle
- Interleaved-5: 5 cycles

JU-only collapse, 최종 clean/erased reconstruction, retrain functional gap과 runtime을
비교한다.

### B. Negative buffer

- `original`: original codec에서 한 번 추출
- `post_ju_static`: 첫 JU step 직후 한 번 추출
- `refresh`: 매 JU step 직후 재추출

각 방식에서 buffer 전송 횟수와 bytes를 함께 기록한다. Refresh가 functional gap을
줄이지 못하면 추가 target 참여 비용을 정당화할 수 없다.

### C. Public positive

- `stochastic_self`: 같은 posterior에서 두 sample을 생성
- `deterministic_augmentation`: 두 image augmentation의 normalized posterior mean

다음 진단값을 cycle별로 기록한다.

- posterior mean log-variance
- posterior standard deviation
- Gaussian posterior entropy
- prior KL
- positive cosine
- erased-negative mean/max cosine

Stochastic 방식의 contrastive loss 감소가 log-variance와 entropy의 급격한 감소로
설명되면 posterior variance collapse로 해석한다.

### D. JU 안정화

- Erased semantic CE ascent: on/off
- Decoder preservation weight: 0과 양의 후보
- JU only와 interleaved CC 조합

Decoder preservation은 동일 target canonical latent와 AWGN realization에서 current
decoder와 frozen original decoder output의 MSE를 사용한다. 이는 SCU Eq. 4의 정확한
decoder-distribution KL이 아니라 안정화 surrogate로만 표기한다.

## 평가

모든 실행은 다음 headline 지표를 기록한다.

- RMSE-255
- PSNR
- Accuracy
- ASR
- MIA
- BA
- Runtime

주 판단 기준은 Original 대비 개선이 아니라 retrain-without-target과의 차이다.
Clean test, erased target data, 가능하면 retained-client partition을 따로 계산한다.
Interleaving/buffer ablation에서는 추가로 paired-channel output gap, model 전송 bytes,
buffer 전송 bytes와 cycle별 JU/CC loss를 보고한다.

## 해석 제한

- Refresh는 target client가 삭제 완료 전 여러 assisted step에 참여할 수 있다는 가정이다.
- Erased latent 전송은 raw-data-free이지만 representation privacy를 보장하지 않는다.
- SCU의 retraining-equivalence 분석은 IID 가정을 사용한다. 두 non-IID split에서의
  equivalence는 이론적 보장이 아니라 실험적으로 검증할 목표다.
- Public CC는 remaining-client utility를 직접 학습하지 않는다. 이 baseline의 안정화가
  distributed retain repair를 대체하지 않는다.

## 구현 smoke 검증

- Config: `src/configs/federated_cifar10_vdeepjscc_scu_public_compensation_smoke.json`
- Seed: 0
- MPI: server 1 rank + clients 10 ranks
- Interleaving: JU 1 epoch와 CC 1 epoch를 2 cycle
- Output:
  `outputs/2026-09-08-04-fscu-posterior-diagnostics-smoke/20260908T042002Z_seed-0`

이 smoke run에서 독립 JU-only checkpoint, 두 interleaved JU/CC cycle, 두 번의 buffer
refresh, CC 전후 posterior diagnostics와 최종 headline metric 생성까지 완료됐다.
Synthetic data와 1-round reference를 사용하는 실행이므로 성능 결과로 해석하지 않는다.
