# IID reconstruction과 classification 모델 완전 분리

## 결정

IID model-quality 실험은 다음 두 FL task를 완전히 독립적으로 실행한다.

```text
Reconstruction:
raw image -> variational DeepJSCC encoder -> AWGN -> decoder -> reconstruction

Classification:
raw image -> CIFAR-10 classifier -> class logits
```

- Reconstruction model은 MSE와 codec 고유의 KL regularization으로만 학습한다.
- Reconstruction 지표는 normalized MSE, RMSE-255, PSNR이다.
- Classification model은 원본 client image와 label의 CE로만 학습한다.
- Classification 지표는 CE와 accuracy다.
- Reconstruction output을 classifier에 입력하지 않는다.
- 두 model 사이로 gradient를 전달하지 않는다.
- Model state, optimizer, FedAvg, checkpoint, history, plot을 공유하지 않는다.

두 task가 공유하는 것은 비교를 위한 CIFAR-10 client index뿐이다. Seed 0 기본 설정은
public 예약 index를 제외한 class-balanced 30,000장을 10개 client에 IID로 나누며,
client마다 각 class 300장을 갖는다. 두 task의 client-pool hash와 client별 index는 같아야
한다.

## 독립 학습 그래프

두 task는 plot도 공유하지 않는다. 각 실행은 같은 run 안에
`metrics/training_history.csv`, `metrics/performance_history.csv`와 다음 figure를 자동으로
남긴다.

- Reconstruction: `figures/<split>_training_loss.png`에 global round별 total objective,
  reconstruction term, beta-weighted KL term을 표시하고,
  `figures/<split>_training_performance.png`에 periodic held-out RMSE-255와 PSNR을 표시한다.
- Classification: `figures/<split>_classification_training.png`에 train/held-out CE와
  accuracy를 표시한다. MSE, RMSE-255, PSNR은 계산하거나 그리지 않는다.

Train과 held-out/test series는 명시적으로 구분하고, final round와 실제 선택 checkpoint를
표시한다. Smoothing이 필요하면 raw series를 유지한 채 window가 기록된 overlay로만
추가한다. Figure의 모든 point는 CSV row에서 재생성 가능해야 하며, 필수 figure가 없으면
해당 task의 run을 완료로 기록하지 않는다. Reconstruction과 classification round budget은
각자의 held-out curve로 따로 판단한다.

## 전처리

Reconstruction은 image를 `[0,1]` 범위로 바꾸고 padding 4 random crop과 확률 0.5
horizontal flip을 적용한다. 같은 augmented image를 codec 입력과 MSE target으로 쓴다.
Decoder가 sigmoid output을 내므로 reconstruction target에 mean/std normalization을
적용하지 않는다.

Classification은 image를 `[0,1]`로 바꾼 뒤 같은 crop/flip을 적용하고,
`Cifar10Classifier` 입력에서 CIFAR-10 channel mean/std normalization을 정확히 한 번
적용한다.

## 모델 감사와 수정

기존 variational codec은 raw log-variance를 hard clamp했다. 200-round checkpoint에서
거의 모든 raw log-variance가 하한 아래로 내려가 variance head gradient가 사실상 0이
되는 문제가 확인됐다. 초기 posterior noise도 mean보다 커 encoder reconstruction
gradient가 decoder보다 지나치게 작았다.

새 `variational_deepjscc_cifar10_v2`는 다음을 적용한다.

- sigmoid 기반 smooth bounded log-variance
- initial log-variance `-7`
- log-variance output channel의 명시적 zero-weight/bias initialization
- 모든 training/unlearning helper의 elementwise-mean KL convention
- legacy codec과 혼용되지 않는 format version 2

실제 CIFAR-10 64장 sanity에서 encoder 초기 gradient는 약 16.7배 커졌고 log-variance
head gradient는 전부 finite/nonzero였다. Power normalization 후 평균 symbol power는 1,
20 dB AWGN의 실측 SNR은 19.964 dB였으며 reconstruction shape와 sigmoid 범위도 정상이다.

독립 classifier는 288,746-parameter GroupNorm-SiLU CNN이다. Raw CIFAR-10을 이용한
별도 architecture sanity run에서 test accuracy 81.63%를 기록해 model 자체가 50% ceiling을
갖는 구조는 아님을 확인했다. 이 sanity run은 federated 800-round 결과가 아니다.

## MPI smoke 검증

다음 11-rank smoke를 각각 독립 실행했다.

```text
outputs/2026-09-09-12-independent-reconstruction-smoke/20260909T045541Z_seed-0/
outputs/2026-09-09-13-independent-classification-smoke/20260909T045550Z_seed-0/
```

- 두 split manifest의 client-pool SHA-256가 일치한다.
- Reconstruction checkpoint에는 `encoder.*`, `decoder.*` key만 있다.
- Classification checkpoint에는 `features.*`, `output.*`, input-normalization buffer만 있다.
- Reconstruction CSV에는 MSE/RMSE-255/PSNR만 기록된다.
- Classification CSV에는 CE/accuracy만 기록된다.
- 두 config 모두 backdoor가 없고 client Adam state를 round 사이에 유지하지 않는다.

## 폐기한 해석

이 문서의 이전 버전에서 reconstruction에 frozen classifier를 연결해 decoded-image
accuracy를 재고 CE gradient를 codec으로 전달한 것은 요구사항을 잘못 해석한 실험이다.
해당 과거 artifact는 provenance 때문에 남겨 두되, 독립 reconstruction/classification
성능이나 본 800-round 결과로 사용하지 않는다.

## 800-round 실행

- Reconstruction config: `src/configs/federated_cifar10_iid_vdeepjscc_800rounds.json`
- Classification config: `src/configs/federated_cifar10_iid_classifier_800rounds.json`

최종 수치와 artifact는 두 실행 완료 후 이 문서에 추가한다.
