# 확정 핵심 지표 구현 (2026-10-22)

## 최신 합의: 2026-11-02 (아래 10-22 기록보다 우선)

- 모든 사용자 보고, 비교 표와 figure의 순서는 **Original, Retrain-without-target,
  Unlearned(FSCUv4)**로 고정한다. Clean PSNR/SSIM, erased PSNR/SSIM, retain FID를
  어느 모델에서도 생략하지 않는다. Reference-only 준비 단계에는 아직 없는 Unlearned를
  만들어 넣지 않고 Original/Retrain 두 state만 표시한다.
- 핵심 보고는 PSNR, SSIM, retain FID, **MPR**, PUL, Runtime이다. dFIL은 인코더 입력
  민감도의 보조 진단이며 삭제 합격 기준이나 Retrain latent 일치 요구가 아니다.
  현재 audit는 호환성을 위해 dFIL CSV/개별 figure도 계속 생성한다.
- `MPR (%) = 100 * marker_positive_count / evaluation_samples`: 고정 검출기의 마커
  절대 검출률이다. Count와 N을 함께 보고한다. Original positives가 0이어도 MPR은
  정의되지만 PUL은 undefined다. S1/S2 모두 계산 가능하며, 모두 0이면 약한 마커가
  전혀 없다는 뜻이 아니라 현재 검출기로 분리하지 못했다는 뜻일 수 있다.
- MPR은 우리 마커 응답을 공격 성공으로 정의할 때 Marker-ASR 역할을 한다. Downstream
  classifier의 공격 label 예측률인 SCU식 Backdoor Accuracy와 동일한 지표로 부르지 않는다.
  MIA balanced accuracy와도 구분한다.
- S1은 binary 검출률만으로 해석하지 않고 연속 **marker projection amplitude**와
  **spatial-mean-centered normalized correlation(NCC)**를 Original/Retrain/Unlearned에
  함께 보고한다. 기존 구현의 두 값을 재사용한다. Projection은 이상적 마커 대비 패턴
  성분의 크기이며 percentage/probability가 아니다. NCC는 패턴 방향 유사성으로, 희미한
  패턴에서도 높을 수 있으므로 단독 강도 지표로 쓰지 않는다. 음수 값도 그대로 보존한다.

### 약한 마커 재검사 v2 (2026-11-02)

사용자가 약한 흔적의 검출을 요청하여 Local detector를 v2로 변경했다. 앞으로 Local
MPR/PUL은 codec 출력과 무관한 **raw clean calibration 최대 amplitude**를 threshold로
고정하고 strict `>`로 판정한다. 이전 midpoint threshold 0.5367을 사용하는 기존 결과는
v1으로 보존하며 v2와 섞지 않는다. 현재 500장 calibration에서 v2 threshold는 0.0734068677,
분리된 raw clean 500장의 오검출은 2/500이다. 유한 표본 최대값이며 모든 사진/codec
출력의 오검출률 상한은 아니다. Marked raw controls는 sensitivity 점검에만 사용한다.

Global은 기존 raw-clean 99% 분위수 threshold를 유지한다. S1뿐 아니라 S2 masked-input
출력에도 global/outside 검사를 적용한다. S2에서 입력에 직접 marker를 보여주는 것은 아니다.
두 detector의 calibration 규칙/검색 범위 차이와 raw-control FPR/TPR를 각각 보고한다.
검출률 0은 임계값 아래 반응까지 없다는 뜻이 아니다.

Amplitude/NCC는 Local/Global/outside 모두 같은 **held-out 500장**에서 산출한다.
최신 보고는 실제 측정값이며 `metrics/marker_response_summary.csv`,
`marker_response_per_sample.csv`, `marker_response_protocol.json`,
`figures/marker_response_comparison.png`를 같은 `src/main.py` audit invocation이 자동 생성한다.
기존 6개 run의 점수 파일은 당시 기록으로 보존하고 덮어쓰지 않는다.
Local v2 protocol에 calibration/evaluation
indices와 held-out raw FPR/TPR를 추가하고, global의 raw controls도
`metrics/marker_spread_controls.csv`에 보존한다. 기존 전체 1,000장 Local raw mean CSV는
호환성을 위해 유지하되 같은 held-out 500장의 새 보고 값과 혼동하지 않는다.

컨테이너 관련 테스트 49개가 통과했다. 실제 6개 조건 재평가도 별도 새 output roots에서
완료했고 현재 core/spread/score artifact gate와 동일한 held-out 500장 사용을 검증했다.
명령/경로/수치 및 불안정 점수의 한계는 [마커 재평가 기록](11-02-01-marker-v2-audit.md)을 따른다.

### 최신 합의: Amplitude/NCC 점수 환산 폐기 (2026-11-02)

사용자는 0/100점 환산 때문에 정상 측정값이 `비교 불가`로 표시되는 문제를 지적하고
점수 환산을 취소했다. 앞으로 **amplitude와 NCC 모두 실제 측정값 mean/SE 그대로**
Original / Retrain / FSCUv4 순서로 보고한다. Original-Retrain 차이로 나누지 않고,
그 차이의 부호/크기로 측정값을 숨기거나 NA 처리하지 않는다. 모델별 min-max도 하지 않는다.

Local은 기존 signed amplitude/NCC, Global/outside는 기존 maximum absolute response다.
NCC 자체의 normalized correlation 정의와 amplitude의 ideal-template 단위는 그대로다.
추가적인 기준모델 점수 환산만 제거하며 음수/1 초과 amplitude를 임의로 자르지 않는다.
MPR/PUL/coverage는 %, PSNR은 dB, SSIM/FID는 기존 값, Runtime은 초를 유지한다.
기존 `marker_response_scores.csv`의 `raw_mean`에도 모든 측정값이 보존돼 있으므로
원값 보고를 위해 모델을 다시 학습/평가할 필요는 없다. 과거 점수 파일/그림은 덮어쓰지 않는다.
Raw 값이 있다고 그 항목이 client 삭제 판정에 유용하다고 자동 보장되는 것은 아니다.

다음은 **기존 v1 임계값의 출처**이며 최신 Local v2 기준은 위 절을 따른다.
기존 임계값은 논문 상수가 아니라 codec 출력과 독립적인 raw 대조군에서 계산했다.
Local `0.5367034338414669`는 clean 500장과 이상적 marker를 넣은 500장에서 balanced
accuracy를 최대화하는 후보 중 선택된 중간값이다. S1 IID에서는 clean 최대
`0.07340686768293381`, marked 최소 `1.0`의 중간값과 정확히 같다. 이 완전 분리된
raw calibration이 약하고 흐린 S1 codec-output marker 검출력을 보장하지는 않는다.
Global `0.3788996636867523`는 clean calibration 500장의 전역 absolute amplitude
최대값 분포에서 99% 분위수(`higher`, 정렬 496번째)를 취한 값이다. Strict `>`로
검출하며 목표 raw FPR는 1%, 실제 calibration FPR는 0.8%, held-out raw FPR는 1.4%다.
Codec-output FPR 1% 보장이 아니다. 검색 범위와 calibration 규칙이 달라 두 raw
threshold를 하나로 통합하지 않는다.

Local 중간값은 S1 IID 새 run의 `reconstruction_marker_per_sample.csv`에서
`reconstruction_marker_detection_per_sample.csv`에 없는 calibration 500장을 복원해
저장 protocol과 일치함을 컨테이너에서 읽기 전용으로 확인했다. Global 출처는 같은
run의 `marker_spread_protocol.json` 및 `src/evaluation/marker_spread.py`다.

**상태:** v2 실제 6개 평가는 당시 코드/점수 artifact 규칙으로 완료됐고 보존한다.
이후 사용자 요청으로 점수 환산 코드를 제거하고 raw report/figure로 바꿨다. 이 표기 변경은
새 checkpoint 평가가 아니며 MPR threshold, checkpoint, 기존 artifact를 변경하지 않는다.

### 구현과 보고 범위

Core CSV schema v4는 `output_marker_positive_percent`, `output_marker_positive_count`,
`output_marker_evaluation_samples`와 연속 projection/correlation 평균·표준오차를 저장한다.
연속 점수의 `output_marker_response_samples`와 검출률의 held-out N은 별도로 기록한다.
`selected_core_metrics.png`는 MPR/PUL/FID/clean·erased PSNR/clean·erased SSIM/Runtime을
표시하고 dFIL은 `encoder_dfil.png`에 분리한다. 마커 연속 점수는 기존
`reconstruction_marker_comparison.png`에도 표시한다.

구현과 단위 테스트 검증은 실제 checkpoint 재평가와 다르다. 기존 10-24 결과 및 immutable
reference를 덮어쓰거나 새 schema로 바꾸지 않는다. 기존 raw CSV의 점수는 그대로 조회할 수 있다.

2026-11-02 컨테이너 검증: 아래 관련 테스트 **40 passed**. CSV의 absolute rate/count/N,
Original-positive=0일 때 MPR/PUL 구분, 모델 정렬, clean/erased 그래프 및 기존 FID/marker
연결을 확인했다. 이는 전역 마커 detector 검증이나 새 checkpoint 평가가 아니다.

```bash
docker compose exec -T scful-dev python -m pytest -q \
  src/tests/test_dfil.py src/tests/test_plotting.py \
  src/tests/test_reconstruction_marker.py src/tests/test_retrain_fid.py \
  src/tests/test_trainer_fscu.py
```

### S1 전역 마커 잔류: 구현 및 대조군 테스트 완료, 실제 평가는 아래 실행 기록 참조

기존 MPR 및 local amplitude/NCC는 **등록된 마커 위치만** 본다. 그 위치의 점수가 Retrain에
가까워져도 전체 이미지의 마커 누출이 제거됐다고 주장하지 않는다. `outside_watermark_rmse_255`
역시 일반 복원 손상까지 포함하므로 marker-specific spread 지표가 아니다.

사용자 요청에 따라 `src/evaluation/marker_spread.py`의 전역 평가를 S1 channel audit에
자동 연결했다. 재학습 없이 같은 입력·paired AWGN에서 세 checkpoint를 비교한다.
`evaluation.marker_spread`로 template sizes/blur/FPR를 바꿀 수 있고 `enabled=false`로
비활성화할 수 있다. 기본값은 size 4/8/12, Gaussian blur sigma 0/1, calibration FPR 1%다.

1. 전체 출력에 마커 template를 sliding하여 projection/NCC **response map**을 저장한다.
   Local marker 영역, 그 영역과 겹치지 않는 outside window, whole-image를 나누어 peak와
   연속 response 크기를 보고한다. Whole-image/outside peak는 검색 위치 수가 다르므로
   같은 값으로 취급하지 않는다.
2. Global/outside peak는 모든 위치·size·blur 및 양/음 polarity에 대한 절댓값 최대다.
   Local amplitude/NCC는 기존 signed 값을 유지한다. Template bank는 codec 출력 평가 전에
   고정했다. Uniform color tint와 형태가 무너진 번짐은 centered
   NCC만으로 검출되지 않을 수 있어 별도 색상/잔차 진단과 positive controls가 필요하다.
3. Global MPR은 raw clean calibration 입력별 전역 amplitude 최대의 상위 1% quantile
   (`higher`, strict `>`)로 정한 frozen threshold를 사용한다. Codec output은 calibration에
   들어가지 않는다. 같은 held-out raw clean의 실제 FPR와, 원래 위치가 아닌 곳에 알려진
   marker를 삽입한 real-CIFAR positive control의 TPR도 저장한다. Local MPR의 threshold와
   별개이며 결과를 보고 사후 조정하지 않는다.
4. Coverage는 positive window가 덮는 **픽셀 합집합 면적 비율**이다. 겹친 window를
   중복 합산하지 않는다. Outside coverage는 original marker와 전혀 겹치지 않는 window의
   합집합을 outside pixel 수로 나눈다. MPR의 분모는 이미지 수이고 coverage의 분모는
   픽셀 수이므로 서로 다른 지표다.
5. Retrain과 real clean inputs에서 자연적으로 생기는 texture/색상 반응을 대조하고,
   full-image reconstruction, residual 및 response map을 공통 색상 범위로 저장한다.
   모델별 독립 min-max scaling으로 희미한 흔적을 과장하지 않는다.

이는 matched filtering/template matching을 활용한 프로젝트별 평가 설계이지 보편적인
unlearning 인증 지표가 아니다. Positive controls 통과도 임의 형태의 번짐 검출이나
client 삭제의 증명이 아니다. 원리 참고:
[OpenCV template matching](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html).

자동 산출물:

- `metrics/marker_spread_summary.csv`: model별 global/outside amplitude peak·RMS,
  NCC peak, global MPR와 positive count/N, global/outside pixel coverage, mean의 표준오차.
- `metrics/marker_spread_per_sample.csv`: dataset index별 위 점수와 검출 여부.
- `metrics/marker_spread_maps.csv`, `predictions/marker_spread_maps.pt`: map figure의 원자료.
- `metrics/marker_spread_protocol.json`: calibration/evaluation index, frozen threshold,
  raw-control FPR/TPR, bank와 channel/SNR, offline runtime 및 한계.
- `figures/marker_spread_comparison.png`, `marker_spread_maps.png`,
  `marker_spread_examples.png`: 세 모델 동일 순서·공통 색상 범위. Example은 효과를 보고
  고르지 않고 첫 held-out input으로 고정한다.
- `metrics/audit_checkpoint_sources.json`: 모든 사용 checkpoint의 실제 경로·SHA-256,
  reference fingerprint/hash 확인 및 실행 command. 기존 reference와 결과는 불변이다.

새 테스트 포함 44개 관련 테스트가 컨테이너에서 통과했다. CUDA 기본 bank 경로, amplitude
0.05/NCC 1의 구분, 이동된 마커, 픽셀 합집합, raw-only threshold 불변성, CSV-backed figure를
검증했다. 실제 3개 S1 checkpoint 평가의 완료 여부는 실행 산출물의 `completed` gate로 판정한다.

### 2026-11-02 실제 checkpoint 평가 완료

S1/S2 × IID/Dirichlet 0.1/Shared-2의 Original/Retrain/FSCUv4 **18개 checkpoint 경로**를
컨테이너에서 확인했고 reference SHA-256을 검증했다. S1은 아래 3개 새 run에서 core suite와
전역 평가를 완료했다. Reference fingerprint/checkpoint hash를 매 run에서 다시 확인했다.
재학습·unlearning 재실행·기존 결과 덮어쓰기는 하지 않았다. S2는 기존 local 지표를 유지하며
이번에 전역 평가를 추가 실행하지 않았다.

| S1 분포 | 새 완료 run (seed 0) | audit device |
| --- | --- | --- |
| IID | `outputs/2026-11-02-01-s1-iid-marker-spread/20261102T094428Z_seed-0` | cuda:0 |
| Dirichlet 0.1 | `outputs/2026-11-02-02-s1-dirichlet-01-marker-spread/20261102T094641Z_seed-0` | cuda:1 |
| Shared-2 | `outputs/2026-11-02-03-s1-shared2-marker-spread/20261102T094906Z_seed-0` | cuda:0 |

실행 형식 (세 run의 source/output은 위 run의 config와 checkpoint provenance에 기록):

```bash
docker compose exec -T scful-dev env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/main.py --experiment reconstruction_channel_audit \
  --source-run outputs/2026-10-24-06-s1-iid-fscuv4-core/20261024T115244Z_seed-0 \
  --audit-device cuda:0 --snr_db 20 --batch_size 200 --seed 0 \
  --output-root outputs/2026-11-02-01-s1-iid-marker-spread
```

이는 완료된 invocation 기록이며 새 invocation은 위 output root를 재사용하지 않는다.
Source training config는 각 새 run `config.json`의 `source_config`에 보존했다.
출력은 posterior mean + paired AWGN 20 dB. Global 평가는 같은 class-balanced CIFAR-10
test pool 1,000장 중 calibration 500장/evaluation 500장을 사용했다. 세 조건의 frozen
global threshold는 0.3788996637로 같았고 calibration FPR=0.8%, held-out raw clean
FPR=1.4%, shifted raw-marker TPR=100%였다. 낮은 FPR 조건의 추정 오차와 codec-output
분포 차이가 있으므로 universal detector validity를 주장하지 않는다.

| S1 분포 | Global MPR Original / Retrain / FSCUv4 (%) | Whole-image coverage Original / Retrain / FSCUv4 (%) |
| --- | --- | --- |
| IID | 1.6 / 1.4 / 0.8 | 0.02891 / 0.03516 / 0.01250 |
| Dirichlet 0.1 | 10.8 / 1.4 / 0.0 | 0.24590 / 0.03203 / 0.00000 |
| Shared-2 | 51.8 / 0.8 / 0.0 | 1.49824 / 0.02578 / 0.00000 |

Global MPR은 검출 이미지 비율이고 coverage는 전체 1,024 pixel 중 검출 pixel 비율의
이미지 평균이다. Whole-image detection은 original marker 영역을 포함하므로 이 표를
outside spread의 증거로 사용하지 않는다. 실제 outside coverage는 각 run의 summary에
별도 저장했다. Legacy local MPR은 세 조건·세 모델 모두 0%라 local PUL은 계속 undefined다.
새 Global MPR은 다른 bank/검색 범위/threshold의 별도 지표이며 local MPR/PUL을 소급 대체하지 않는다.

| S1 분포 | Outside peak NCC 절댓값 Original / Retrain / FSCUv4 |
| --- | --- |
| IID | 0.4383 / 0.2908 / 0.3794 |
| Dirichlet 0.1 | 0.5862 / 0.2909 / 0.4123 |
| Shared-2 | 0.5843 / 0.2901 / 0.4154 |

FSCUv4의 outside NCC가 Retrain보다 높게 남았다. 이는 크기 정규화된 패턴 유사도이므로
같은 크기의 마커 에너지나 membership을 뜻하지 않지만, MPR/coverage=0만으로 전역 잔류가
완전히 사라졌다고 결론내릴 수 없음을 보여준다. Template-like 반응, codec artifact,
일반 복원 손상은 구분해야 하며 아직 client-level 삭제 증명은 아니다.

## 2026-10-22 구현·검증 기록 (역사적 기록)

사용자 합의: PSNR, SSIM, FID, dFIL, PUL, Runtime. IDI는 핵심 평가에서 제외한다.
PSNR/SSIM은 clean test와 erased train, FID는 clean remaining-client inputs 대비
reconstruction 분포, PUL은 frozen marker detector의 positive count 감소율이다.
PUL은 marker가 없으면 NA이고 Original positives가 0인 경우 기존 undefined 규칙을 따른다.
FID는 2048-D pytorch-fid Inception 특징의 전체 선택 집합 moments로 계산하며 batch FID를
평균하지 않는다. 기존 full-codec output 지표를 decoder parameter 삭제의 증명으로 해석하지 않는다.

## dFIL 정의와 관측 위치

출처: Maeng et al., [Bounding the Invertibility of Privacy-Preserving Instance Encoding
Using Fisher Information, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/a344f7f474958cc0775be7e46bc94309-Paper-Conference.pdf), 식 (1), (3).

`dFIL = trace(input Fisher) / input dimension`을 사용한다. Parameter gradient 제곱 평균과
다르다. 관측은 `model(x, mode="posterior")`가 정의하는 **power normalization 및 channel
통과 전 diagonal Gaussian sample**이다. 입력에 따라 Gaussian variance도 달라지므로:

`trace = sum_j ||grad_x mean_j||² / variance_j + 0.5 * sum_j ||grad_x logvar_j||²`

두 Jacobian의 Frobenius norm을 독립 Rademacher VJP로 추정한다. Inverse standard deviation은
미분하지 않는 가중치로 사용한다. Gaussian mean만 미분하고 scale 항을 누락하지 않는다.
RGB 입력 단위는 [0,1], 분모는 입력 차원(3072)이다. Latent dimension으로 나누지 않는다.

이는 주어진 posterior에 대한 논문의 Fisher 정의 적용이다. Normalized latent 및 채널 수신
latent의 Fisher라고 주장하지 않는다. 물리 채널/SNR의 효과는 기존 PSNR/SSIM/FID/PUL
프로토콜로 평가한다. dFIL은 입력 복원 가능성의 진단이며 학습 membership 또는 client 삭제
증명은 아니다. `1/dFIL`의 무편향 복원 한계를 임의 attacker 보장으로 사용하지 않는다.
Bias가 있는 attacker까지 다루려면 prior Fisher와 논문의 추가 조건이 필요하다.

## 설정과 표본

- `evaluation.dfil_probes=32`: 각 입력의 trace 추정에 사용하는 random direction 수 (최소 2).
- `evaluation.dfil_max_examples=128`: split별 uniform subset 크기. 0이면 공급된 pool 전부.
- clean_test, retain_train, erased_train을 분리한다. Retain pool은 FID의 고정된 clean
  class-balanced remaining pool이다. 전체 client population과 동일하다고 주장하지 않는다.
- 모든 checkpoint에 같은 dataset indices와 probe seed를 적용한다.
- 마커 방식 2에서는 target의 기록된 masked input을 dFIL에 사용한다.
- Smoke는 split별 최대 2장, 2 probes이며 paper result가 아니다.
- per-input Monte Carlo SE와 평균의 Monte Carlo SE, sample간 표준편차를 구분한다.
  표본 수/probe 수 수렴 점검 없이 128/32 기본값의 충분성을 주장하지 않는다.

## 자동 저장

FSCU/FSCUv4 및 reference 평가가 호출하는 기존 reconstruction channel audit에 연결한다.
Original/Retrain만 있는 reference audit에서도 dFIL을 계산한다.

- `metrics/selected_core_metrics.csv`, `metrics/selected_core_metrics_protocol.json`
- `figures/selected_core_metrics.png`: PSNR, SSIM, FID, PUL, split별 dFIL, Runtime
- `metrics/encoder_dfil.csv`: model/split별 평균, 표준편차, Monte Carlo SE
- `metrics/encoder_dfil_per_sample.csv`: dataset index, probe seed, dFIL, SE
- `metrics/encoder_dfil_protocol.json`, `figures/encoder_dfil.png`
- `metrics/core_runtime.csv`: algorithm wall time와 scope; offline evaluation 제외

기존 headline 및 output agreement 파일은 legacy diagnostic으로 남는다. IDI critic 학습과
IDI probe 구성은 main audit에서 제거한다. 기존 결과와 checkpoint는 변경하지 않는다.

## 검증

```bash
docker compose exec -T scful-dev python -m pytest -q src/tests/test_dfil.py src/tests/test_plotting.py src/tests/test_retrain_fid.py src/tests/test_reconstruction_marker.py src/tests/test_config.py
```

`test_dfil.py`는 real CIFAR-10 입력에서 analytic diagonal Gaussian과 input-dependent scale
항, noise scaling, zero Fisher, gradient buffer 보존, 실제 VariationalDeepJSCC의 paired
reproducibility와 CSV/JSON/figure 연결을 검사한다. Test artifact는 pytest temporary directory다.

현재 세션의 검증 실행은 Docker socket permission denied와 sudo password 요구로 막혔다.
테스트 통과 또는 새 실험 결과 생성으로 보고하지 않는다.

## 구현 현황과 코드 위치

2026-10-22 기준. ‘구현’은 소스와 연결이 존재한다는 뜻이며 테스트 통과를 뜻하지 않는다.

| 항목 | 구현 위치 | 현재 상태 |
| --- | --- | --- |
| PSNR / RGB SSIM | `src/evaluation/federated_behavior.py`, `src/utils/metrics.py` | 기존 계산 유지, clean/erased core CSV 연결 |
| retain FID | `evaluate_retain_ufid_states`, channel audit | 기존 frozen Inception 2048-D 계산 유지, 전체 선택 pool 통계 사용 |
| dFIL | `src/evaluation/dfil.py` | 신규 Gaussian input-Fisher 계산, split별 paired 추정 및 MC 오차 |
| PUL | `src/experiments/reconstruction_marker_audit.py`, channel audit | 기존 frozen detector 경로 유지, marker 없는 경우 NA |
| Runtime | channel audit 및 trainer의 저장된 phase runtime | core CSV/별도 CSV 연결; 누락된 과거 시간은 임의로 채우지 않음 |
| Config | `src/utils/config.py` | dfil_probes >= 2, dfil_max_examples >= 0 검증 |
| 실행 연결 | `src/experiments/reconstruction_channel_audit.py` | FSCU/FSCUv4/reference 공통 audit에 dFIL 통합; IDI critic main 호출 제거 |
| Reference bank | `src/utils/trainer.py` | `reference_core_metrics.csv`를 selected core에서 저장, dFIL protocol 링크 포함 |
| 그래프 | `src/utils/plotting.py` | `encoder_dfil.png`, 6-panel `selected_core_metrics.png` |
| 완료 gate | channel audit의 `_validate_core_artifacts` | core/상세 CSV 행, protocol JSON, figure 존재와 모델 목록 확인 |
| 테스트 | `src/tests/test_dfil.py`, 기존 plotting/FID/marker/config 테스트 | 코드 작성 완료, Docker 권한 때문에 실행 대기 |

테스트 fixture의 gain parameter를 float로 생성하도록 보완했다. 실제 codec 결과를
만든 것이 아니라 analytic zero-Fisher 테스트가 정수 Parameter 오류로 중단되지 않도록 한 수정이다.
Core protocol schema v2에는 output 경로, SNR, smoke 여부와 encoder 관측 위치를 기록한다.
Runtime·PUL의 unavailable/undefined 상태를 0으로 대체하지 않는다.

## 해석 및 완료 기준

- PSNR/SSIM: 높을수록 입력 복원이 좋다. Clean utility와 erased 결과를 분리한다.
- FID: 낮을수록 retain 복원 분포가 실제 retain 입력 분포에 가깝다. 삭제 증명은 아니다.
- dFIL: 낮을수록 해당 관측의 입력 Fisher가 작다. 유용한 정보까지 잃어도 감소할 수 있으므로
  erased 값뿐 아니라 retain/clean 값과 reconstruction utility, Retrain을 함께 본다.
- PUL: Original 대비 marker-positive 출력 감소율. Retrain을 반드시 함께 보고, 단순 색상
  변화나 전체 복원 붕괴를 성공으로 간주하지 않는다. Original 검출이 0이면 정의되지 않는다.
- Runtime: 저장된 retraining/online unlearning wall time을 비교한다. Offline metric 비용과
  checkpoint loading 시간을 알고리즘 시간에 섞지 않는다.

현재 구현을 이유로 기존 800-round 실험에 dFIL 결과가 이미 있다고 말하면 안 된다.
기존 checkpoint와 결과는 변경하지 않았으며 재평가가 필요하다. 확정 지표를 구현하는
작업과 새 Fisher-based joint-unlearning 방법론을 구현하는 작업도 별개다.

## 실행 검증에 필요한 환경과 다음 작업

1. 현재 실행 계정에서 `docker compose exec -T scful-dev id`가 성공해야 한다.
   2026-10-22 재시도에서도 Docker socket permission denied였다. 비밀번호는 받지 않는다.
2. 컨테이너 안에 PyTorch/pytest/pytorch-fid와 기존 의존성이 있어야 한다. Host Python으로
   우회하지 않는다. 실제 CIFAR-10 파일은 host `data/cifar10/cifar-10-batches-py`에 확인했다.
3. 위 테스트 명령을 실행하고 실패 시 수정한다. 본 수치 비교 전 probe 수/표본 수 수렴을
   확인한다. 단위 테스트 성공만으로 연구 지표의 유효성을 주장하지 않는다.
4. 실제 Origin/Retrain/FSCUv4 비교에는 source run의 config, split manifest, checkpoint 또는
   hash-검증 가능한 reference_source, marker metadata와 audit가 요구하는 frozen observer가 필요하다.
   현재 audit는 legacy semantic 진단도 함께 실행하므로 core 정의와 무관하게 observer 의존성이 남아 있다.
5. FID의 pretrained Inception weights를 컨테이너 캐시에서 읽거나 다운로드할 수 있어야 한다.
   GPU는 속도를 위한 선택이며 FID를 CPU에서만 계산해야 하는 것은 아니다.
6. 재평가는 `src/main.py`의 기존 audit entry point로 새 날짜-순번 output root에 저장한다.
   기존 reference를 in-place로 재평가해 immutable 결과를 덮어쓰지 않는다.

이번 변경에서는 새 training/audit 실험을 실행하지 않았다. Docker 테스트는 시작 전에
권한 오류로 종료했다. `git diff --check`와 파일 트리 확인은 코드 실행 검증이 아닌 정적 확인이다.

## 프로젝트 문서 연결

- `README.md`: 최신 지표, 구현/검증 상태, 실행 산출물 안내.
- `AGENTS.md`: 이후 작업에서 따라야 할 core contract 및 검증 보고 규칙.
- `docs/method/idea_summary.md`: client deletion 목표, dFIL 한계, Fisher 방법론과 구분.
- `docs/method/system_model.md`: pre-channel 관측과 현재 posterior-mean audit 경로.
- `docs/method/unlearning_pipeline.md`: forgetting/repair 후 공통 평가 단계.
- `docs/experiment/08-14-01-전체-실험-계획.md`: 현재 필수 지표와 역사적 후보 목록 분리.
- `docs/literation/comparison_matrix.md`: 지표 출처와 client-level 문제에의 적용 한계.

과거 실험 보고서의 IDI 등 수치와 checkpoint는 역사적 기록으로 보존한다.
