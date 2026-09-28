# Remaining-client 독립 classifier와 출력 비교

2026-09-22: 구현 및 CPU 소규모 검증만 수행. GPU 학습/본 평가 결과가 아니다.

## 학습 경계

- `--source-run`의 exact client split/target을 재사용한다. Client를 다시 섞지 않는다.
- Target rank는 참여하지 않는다. MPI rank 수는 `remaining clients + server`다.
- 각 remaining rank는 자기 clean CIFAR-10 raw image와 원래 label만 사용한다.
  Trigger 주입, target 데이터, 원본 classifier warm-start, codec feature를 사용하지 않는다.
- 새 `Cifar10Classifier`를 CE로 학습하고 sample-weighted FedAvg한다. 매 round local
  Adam state를 새로 만든다. Server에는 raw train/validation image를 보내지 않는다.
- Classifier의 입력 정규화는 기존 CIFAR-10 고정 mean/std, augmentation은 crop/flip이다.
- 각 client 내부에서 기본 10%를 class별 validation으로 분리한다. 작은 class는 최소
  한 개의 train example을 남기므로 작은 smoke의 정확한 비율은 10%와 다를 수 있다.
- Validation은 각 client에서 평가한 CE 합/정확도와 개수만 집계한다. 공식 CIFAR-10 test는
  checkpoint 선택이 끝난 뒤 한 번 평가한다. 별도 public reservation은 변경하지 않는다.

## 학습량과 비용

`src/configs/remaining_classifier.json`은 안전한 기본 `device=cpu`다. GPU 학습을
실행하려면 사용 가능 여부를 확인한 후 config의 device를 명시적으로 바꾼다.
Batch 200, local epoch 1, Adam LR .001, round 350 이후 .0001이다.
800 rounds는 안전 상한이지 필요한 학습량이라는 결론이 아니다.

- Round 1, 매 5 rounds, 마지막 round에 local validation.
- Validation CE가 .0001 이상 개선되지 않는 평가가 20회 연속이면 종료.
- 종료 시점과 별개로 평가된 round 중 최소 validation CE checkpoint를 선택.
- Train CE/accuracy는 local augmented training 중 측정한 값이다. Validation은
  집계된 global classifier로 clean held-out image에서 평가한다.
- Observer 학습+validation wall time을 따로 저장하고 codec retrain/unlearning Runtime에
  포함하지 않는다. 공식 test 평가와 파일 저장 시간도 학습 runtime과 구분한다.

## 실행 예시 — 아직 실행하지 않은 본 학습

아래 source는 IID codec 완료 run이다. 10 clients 중 target 0을 제외하므로 MPI는
server 1 + remaining 9 = 10 ranks다. 실행 날짜와 당일 순번에 맞는 새 output root를
매번 지정한다. 아래 날짜/순번은 예시이며 같은 root로 재실행하지 않는다.

```bash
docker compose exec -T scful-dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun --allow-run-as-root --oversubscribe -np 10 \
  python src/main.py --experiment remaining_classifier \
  --source-run outputs/2026-09-10-21-ibju-fedavg-iid/20260910T140049Z_seed-0 \
  --config src/configs/remaining_classifier.json --seed 0 \
  --output-root outputs/2026-09-22-01-remaining-classifier-iid
```

`--smoke`는 각 remaining client에서 최대 20개 실제 CIFAR-10 example, 2 rounds,
작은 classifier로 plumbing만 검증한다. Smoke observer는 full audit에서 로드할 수 없다.

## 기존 codec checkpoint 평가

새 observer의 완료 run directory를 `--observer-run`에 넣는다. 기존 codec을 재학습하거나
reference를 수정하지 않는다. Observer를 바꿀 때 `--in-place`는 금지하며 새 audit root를
사용한다. 아래 `OBSERVER_RUN`은 실제 완료 경로로 교체한다.

```bash
docker compose exec -T scful-dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/main.py --experiment reconstruction_channel_audit \
  --source-run outputs/2026-09-10-21-ibju-fedavg-iid/20260910T140049Z_seed-0 \
  --observer-run OBSERVER_RUN --audit-device cpu --snr_db 20 --batch_size 200 \
  --seed 0 --output-root outputs/2026-09-22-02-remaining-observer-iid-audit
```

새 FSCU/FSCUv4 실행에도 동일한 `--observer-run`을 전달할 수 있다. 실제 source split과
target이 일치해야 한다. CLI override는 저장 config의 `evaluation.observer_override`와
실제 classifier metadata에 기록한다. Classifier metadata/state가 reference fingerprint에 포함되므로
평가자 변경 뒤 기존 reference가 재사용되지 않을 수 있다. 이미 학습한 codec을 새 평가자로
평가만 하려면 위 audit 경로를 사용한다. 기존 config의 49k observer 경로는 과거 결과
재현용으로 보존하며, 새 observer를 생략했을 때 자동으로 target-free라고 표시하지 않는다.

## 평가 정의

한 개의 frozen classifier에 최종 RGB 복원 이미지를 넣는다. 모든 평가를 `no_grad()`에서
실행하고 classifier 입력을 detach한다. 새로운 CE/KD/semantic loss는 codec에 추가하지
않는다. `--observer-run`을 사용하는 codec 경로는 semantic update 설정이 0인지 검증한다.

각 split에서 같은 입력과 같은 AWGN draw, posterior mean, 지정 SNR로 다음을 계산한다.

| 지표 | 비교 대상/정의 |
| --- | --- |
| 기존 PSNR, 새 SSIM-input | 복원 이미지와 원본 입력의 품질 |
| PSNR-to-Retrain | 같은 입력의 두 복원 출력 사이 pooled MSE에서 계산; peak=1 |
| SSIM-to-Retrain | 같은 입력의 두 복원 출력 사이 구조적 유사성 |
| Class agreement | 두 출력에 공통 classifier를 적용한 argmax 일치 비율 |
| Probability JS | softmax 확률의 Jensen-Shannon divergence, natural-log nats, T=1 |
| Logit MSE | 10개 logit의 평균제곱차; shift-sensitive 보조 지표 |
| Semantic accuracy | 동일 posterior-mean 복원의 정답 대비 accuracy; agreement와 구분 |

SSIM은 RGB channel 평균, 11x11 Gaussian sigma 1.5, valid window, population covariance,
K1=.01, K2=.03, data range 1, downsampling 없음으로 고정한다. MS-SSIM이 아니다.
PSNR-to-Retrain은 MSE floor 1e-12로 120 dB에서 표시를 제한하며 `psnr_to_retrain_capped`를
기록한다. 수학적으로 완전히 같은 출력의 PSNR은 무한대다. Retrain 자기 비교의 SSIM=1,
agreement=1, JS=0은 정의상 성립하며 삭제 성공의 근거가 아니다.

기존 headline Accuracy/ASR/BA는 기존 paired sampled-posterior 경로를 유지한다.
추가 semantic accuracy와 agreement는 이미지 PSNR/SSIM과 같은 paired posterior-mean
출력에 대한 지표이므로 두 프로토콜을 혼동하지 않는다. Erased train에는 source manifest의
실제 poison 주입이 재현될 수 있으며 그 label은 trigger label이다. Erased semantic
accuracy를 clean-label accuracy라고 해석하지 않는다.

이미지/semantic agreement는 output-level 근접성의 실증 지표이며 단독 삭제 증명은 아니다.
Original-to-Retrain도 같은 조건으로 보고하고 기본 8개 headline 지표를 유지한다.

## Artifact와 검증

Observer 학습:

- `checkpoints/initial.pt`, `checkpoints/observer.pt`
- `split_manifest.json`, `observer_manifest.json`: source exact indices, target,
  client별 train/validation indices/class counts, config/seed, protocol/state/checkpoint hash
- `metrics/observer_training.csv`, `metrics/observer_summary.json`
- `figures/remaining_classifier_training.png`, `figures/remaining_classifier_partition.png`

Audit:

- `metrics/retrain_output_semantic_comparison.csv`
- `metrics/clean_test_output_comparison_per_sample.csv`, `metrics/erased_train_output_comparison_per_sample.csv`
- `predictions/clean_test_semantic_logits.pt`, `predictions/erased_train_semantic_logits.pt`
- `metrics/output_comparison_protocol.json`, `figures/retrain_output_semantic_comparison.png`
- 기존 headline/FID/MIA/BA artifact도 함께 생성

Target/split/observer hash가 맞지 않으면 거부한다. Raw logits는 동일 입력 위치와 dataset
index를 함께 보존한다. 그림의 모든 점은 CSV/JSON에서 재구성 가능하다.

CPU 검증 명령:

```bash
docker compose exec -T scful-dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q src/tests/test_remaining_observer.py
```

실제 CIFAR-10 소량 데이터로 MPI 3 ranks(서버+remaining 2), 2-round 학습/체크포인트 선택,
그래프, target 제외, provenance/hash 거부, 평가 gradient 차단, retrain 자기 일치를 검증한다.
Audit integration test의 Inception만 저비용 test double로 대체하며 이 FID는 연구 결과가
아니다. 실제 실험에서는 기존 frozen pytorch-fid Inception 2048-D를 그대로 사용한다.
