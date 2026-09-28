# Retrain 기준 FID 구현

## 범위와 해석

Reconstruction codec의 학습 loss나 구조를 바꾸지 않고 평가 지표만 추가한다.
`PSNR`과 `BA`는 기존 구현을 유지한다. `SSIM`은 아직 구현하지 않았다.

`FID-to-Retrain`은 candidate와 target client 없이 재학습한 codec이 **동일한 입력**을
복원한 이미지 집합의 특징 분포를 비교한다. 원본 이미지와 복원 이미지의 FID가 아니다.
이 reference 선택은 [Realistic I2I, Table 1](https://arxiv.org/html/2502.04260v2)의
CIFAR-10 AutoEncoder 실험을 참고한다. 본 프로젝트의 channel/전처리 protocol까지
동일하게 재현했다는 뜻은 아니다.

- `clean_fid_to_retrain`: 공식 CIFAR-10 test 입력 10,000개의 복원 출력 비교
- `erased_fid_to_retrain`: 삭제 client의 전체 local train 입력의 복원 출력 비교
- 모든 checkpoint에서 같은 입력 순서, posterior mean, 같은 AWGN noise, 기본 20 dB
- 삭제 집합에 등록 backdoor가 있으면 원래 poison 위치를 재현해 입력한다.
- 낮을수록 retrain의 출력 분포와 가깝다. Retrain 자기 비교는 정의상 0이다.

FID는 이미지별 일치가 아니라 특징 분포의 평균과 공분산 차이다. 기존 paired-output
MSE gap, utility 및 BA를 대체하지 않으며 단독으로 학습 기여 삭제를 증명하지 않는다.
Original FID도 반드시 함께 보며, Original부터 Retrain과 구분되지 않는 시나리오에서
Unlearned FID만 낮다는 이유로 성공을 주장하지 않는다. 표본 크기가 다른 clean/erased
FID의 절대 크기를 직접 비교하지 않는다.

## 구현 protocol

- `pytorch-fid==0.3.0`의 FID용 InceptionV3, final average pool 2,048-D.
  일반 torchvision ImageNet 분류 가중치로 대체하지 않는다.
- Float32 RGB reconstruction [0,1]을 이미지 파일 저장/8-bit 양자화 없이 전달한다.
- Resize 299x299 bilinear `align_corners=False`, 이어서 `2*x-1` 변환은
  [pytorch-fid 구현](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py)을 따른다.
- Feature extractor는 eval/frozen/no-grad이며 codec과 optimizer를 공유하지 않는다.
- Float64 batch-merge 평균/공분산을 split 전체에 누적한다. 공분산 분모는 N-1이며
  minibatch별 FID를 계산한 뒤 평균하지 않는다.
- 최종 거리는 `pytorch_fid.fid_score.calculate_frechet_distance`를 사용한다.
  동일 통계의 자기 비교는 수치 오차 없이 0을 반환하고, 작은 음수 roundoff만 0으로
  보정한다. 유효하지 않은 음수/비유한 값은 오류로 처리한다.
- Inception inference batch는 `evaluation.fid_batch_size`, 기본 32. Codec evaluation
  batch와 별도이므로 codec batch 200을 Inception에 한꺼번에 넣지 않는다.
- 전처리, feature layer, package version, feature state SHA-256, 표본 수, reference,
  AWGN seed, SNR 및 evaluation runtime scope를 JSON에 저장한다.
- 표본이 2개 미만이면 계산을 거부한다. N <= 2,048의 rank-deficient 표본은 명시적으로
  표시하며 특히 smoke 수치를 논문 성능으로 해석하지 않는다. 더 큰 N도 신뢰성 보장을
  자동으로 주지는 않으므로 본 실험은 전체 split과 반복 seed를 사용한다.

## 실행과 artifacts

정상 FSCU/FSCUv4 실행의 자동 headline audit에서 기본으로 함께 계산된다. Reference
학습이나 unlearning을 추가로 수행하지 않고 기존 완료 run도 다시 평가할 수 있다.
새 audit는 새 output root에 저장하며 source checkpoint/기존 지표는 덮어쓰지 않는다.

```bash
docker compose exec -T scful-dev env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/main.py --experiment reconstruction_channel_audit \
  --source-run outputs/2026-09-10-17-ibju-fedavg-smoke/20260910T134330Z_seed-0 \
  --output-root outputs/2026-09-15-01-retrain-fid-smoke \
  --seed 0 --snr_db 20 --batch_size 32 --audit-device cuda:0 --smoke
```

위 명령은 실제 CIFAR-10 test에서 class당 10개를 고정 선택하고 source run의 삭제
집합은 그대로 사용하는 **구현 검증용 smoke**다. 재실행 시 날짜/순번에 맞게 새 output
root를 지정한다. 일반 재평가는 `--smoke` 없이 전체 test set을 사용한다. Source config,
선택된 test 원본 index, split manifest 경로, seed는 audit config에 보존된다.

- `metrics/headline_metrics.csv`: 기존 지표 + clean/erased FID 열
- `metrics/retrain_fid.csv`: model/split별 FID, 표본 수, SNR, seed
- `metrics/retrain_fid_protocol.json`: feature/reference/protocol metadata
- `figures/retrain_fid_comparison.png`: clean/erased 별도 panel, 동일 model 순서/scale
- `figures/unlearning_headline_comparison.png`: 기존 지표 비교

FID evaluation wall time은 codec 학습/online unlearning Runtime과 분리한다. 기록하는
추가 wall time 범위는 feature model load와 clean/erased reconstruction/FID 평가 전체이며
FID 계산만의 순수 overhead로 해석하지 않는다.

## 검증

```bash
docker compose exec -T scful-dev env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q src/tests/test_retrain_fid.py \
  src/tests/test_federated_behavior_metrics.py src/tests/test_trainer_fscu.py
```

검증 항목: NumPy N-1 공분산과 streaming 일치, 공식 FID 함수와 일치, 대칭성,
동일 분포 0, 평균 이동의 알려진 거리, singular covariance, 잘못된 입력 거부,
실제 CIFAR-10 + pretrained Inception frozen 동작, paired AWGN과 기존 PSNR 불변,
CSV의 비어 있지 않은 행에서 자동 figure 생성.

2026-09-15 smoke 검증 실행은
`outputs/2026-09-15-01-retrain-fid-smoke/20260915T141807Z_seed-0`에 완료했다.
Clean 100개/삭제 18개에서 5개 checkpoint의 FID 행 10개와 headline CSV 및 자동 그래프를
생성했다. Retrain의 두 self-FID는 모두 0이었다. 이는 구현 검증이며 IID/non-IID 본
실험 성능 결과로 사용하지 않는다. Source checkpoint와 source metric은 수정하지 않았다.
