# Marker v2 재검사: MPR, amplitude, NCC

## 후속 보고 규칙: 점수 환산 취소

사용자 요청으로 Original=100점, Retrain=0점 환산을 폐기했다. 앞으로 amplitude/NCC는
Original / Retrain / FSCUv4의 실제 측정값을 그대로 보고한다. 아래 점수 표와 점수 파일은
실행 당시의 역사적 기록이며 현재 보고 규칙이 아니다. `비교 불가`/`불안정`은 측정 실패가
아니라 당시 점수 환산의 문제였다. 모든 원래 측정값은 해당 CSV의 `raw_mean`에 있다.
Local/Global/outside의 같은 held-out 500장을 유지하며 MPR 기준과 결과는 변경하지 않는다.

### 현재 보고용 원래 측정값 (같은 500장)

기존 6개 run의 입력별 CSV를 최신 raw 보고 함수로 읽기 전용 검증했다. 108개 평균이
기존 파일의 `raw_mean`과 1e-12 이내로 일치했고 환산에 따른 결측값은 없다.
모델을 재평가하거나 기존 결과 파일을 수정하지 않았다.

#### 원래 마커 위치

| 조건 | Amplitude Original / Retrain / FSCUv4 | NCC Original / Retrain / FSCUv4 |
| --- | --- | --- |
| S1 Dirichlet 0.1 | 0.07738 / -0.00045 / 0.00068 | 0.3270 / -0.0017 / 0.0040 |
| S1 IID | 0.04036 / -0.00038 / -0.00047 | 0.2207 / -0.0005 / -0.0008 |
| S1 Shared-2 | 0.18122 / -0.00029 / 0.00185 | 0.6392 / -0.0001 / 0.0129 |
| S2 Dirichlet 0.1 | 0.73651 / -0.00014 / 0.12023 | 0.9661 / -0.0024 / 0.5869 |
| S2 IID | 0.52248 / -0.00020 / 0.07214 | 0.9239 / -0.0019 / 0.4672 |
| S2 Shared-2 | 0.54636 / -0.00078 / 0.06467 | 0.9360 / -0.0103 / 0.4729 |

#### 전체 이미지

| 조건 | Amplitude Original / Retrain / FSCUv4 | NCC Original / Retrain / FSCUv4 |
| --- | --- | --- |
| S1 Dirichlet 0.1 | 0.26653 / 0.21539 / 0.18359 | 0.6517 / 0.2927 / 0.4215 |
| S1 IID | 0.20975 / 0.21498 / 0.20376 | 0.5085 / 0.2924 / 0.3882 |
| S1 Shared-2 | 0.41197 / 0.21349 / 0.18816 | 0.7720 / 0.2930 / 0.4265 |
| S2 Dirichlet 0.1 | 1.85739 / 0.24574 / 0.36723 | 0.9808 / 0.2907 / 0.7252 |
| S2 IID | 1.37207 / 0.24839 / 0.29491 | 0.9311 / 0.2902 / 0.6118 |
| S2 Shared-2 | 1.27535 / 0.24461 / 0.28130 | 0.9486 / 0.2920 / 0.7008 |

#### 원래 위치 바깥

| 조건 | Amplitude Original / Retrain / FSCUv4 | NCC Original / Retrain / FSCUv4 |
| --- | --- | --- |
| S1 Dirichlet 0.1 | 0.22418 / 0.21248 / 0.18171 | 0.5862 / 0.2909 / 0.4123 |
| S1 IID | 0.20043 / 0.21226 / 0.20152 | 0.4384 / 0.2907 / 0.3794 |
| S1 Shared-2 | 0.20380 / 0.21126 / 0.18632 | 0.5843 / 0.2900 / 0.4154 |
| S2 Dirichlet 0.1 | 0.20698 / 0.20766 / 0.21070 | 0.3398 / 0.2902 / 0.3288 |
| S2 IID | 0.20950 / 0.20886 / 0.21298 | 0.3565 / 0.2894 / 0.3173 |
| S2 Shared-2 | 0.20592 / 0.20666 / 0.20967 | 0.3613 / 0.2915 / 0.3240 |


## 목적과 상태

S1/S2 × IID/Dirichlet 0.1/Shared-2에서 Original, Retrain, FSCUv4를 비교한다.
기존 800-round reference와 FSCUv4 checkpoint를 재사용하며 학습/unlearning을 반복하지 않는다.
기존 output/reference는 덮어쓰지 않는다. Seed=0, posterior mean + paired AWGN 20 dB,
batch size=200이다. 단일 seed, retrospective detector revision이며 삭제 인증이 아니다.

관련 컨테이너 테스트는 49 passed. 아래 6개 run 모두 core artifact gate와 `run.log`의
completed 기록으로 실제 완료를 확인했다. 기존 reference와 결과는 변경하지 않았다.

## 실행 당시 검출/점수 규칙 (점수 환산은 현재 폐기)

- Local MPR: raw clean calibration 500장의 최고 amplitude 초과를 양성으로 판정한다.
  현재 threshold=0.0734068677, 별도 raw clean 500장의 오검출=2/500이다.
- Global MPR: raw clean 전역 최대 amplitude의 99% 분위수(`higher`, strict `>`)를 사용한다.
  검색 크기 4/8/12, blur sigma 0/1, 원래/반전 방향을 검사한다. Codec output은 기준 결정에
  사용하지 않는다. Original 위치를 포함하므로 전역 양성을 outside spread와 동일시하지 않는다.
- Amplitude/NCC **두 항목만** Original=100점, Retrain=0점으로 환산한다. Local/Global/outside
  모두 같은 held-out 500장이다. Original이 Retrain보다 높지 않으면 해당 score는 NA다.
  Raw mean/SE, paired reference-gap SE와 약한 기준점 경고를 저장하며 clipping하지 않는다.
- MPR/PUL/coverage는 %, PSNR은 dB, SSIM/FID 원래 값, Runtime은 초를 유지한다.
  점수는 실제 정보 잔존율이 아니며 NCC는 강도 자체가 아니다.
- 이 audit는 marker-only plumbing이 아니라 기존 full core suite/figures도 함께 생성한다.

## 실행 계획과 재현 명령

| 조건 | Source run (outputs/ 아래) | 새 output root (outputs/ 아래) | Device |
| --- | --- | --- | --- |
| S1 IID | 2026-10-24-06-s1-iid-fscuv4-core/20261024T115244Z_seed-0 | 2026-11-02-04-s1-iid-marker-v2 | cuda:1 |
| S2 IID | 2026-10-24-09-s2-iid-fscuv4-core/20261024T131343Z_seed-0 | 2026-11-02-05-s2-iid-marker-v2 | cuda:0 |
| S1 Dirichlet 0.1 | 2026-10-24-07-s1-dirichlet-01-fscuv4-core/20261024T121939Z_seed-0 | 2026-11-02-06-s1-dirichlet-01-marker-v2 | cuda:1 |
| S2 Dirichlet 0.1 | 2026-10-24-10-s2-dirichlet-01-fscuv4-core/20261024T134011Z_seed-0 | 2026-11-02-07-s2-dirichlet-01-marker-v2 | cuda:0 |
| S1 Shared-2 | 2026-10-24-08-s1-shared2-fscuv4-core/20261024T124658Z_seed-0 | 2026-11-02-08-s1-shared2-marker-v2 | cuda:1 |
| S2 Shared-2 | 2026-10-24-11-s2-shared2-fscuv4-core/20261024T140709Z_seed-0 | 2026-11-02-09-s2-shared2-marker-v2 | cuda:0 |

실행 형식은 아래와 같다. 나머지 조건은 표의 source/output/device를 대입한다.
실행을 반복하면 위 root를 재사용하지 말고 다음 새 index를 사용한다.

```bash
docker compose exec -T scful-dev env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  nice -n 10 python src/main.py --experiment reconstruction_channel_audit \
  --source-run outputs/2026-10-24-06-s1-iid-fscuv4-core/20261024T115244Z_seed-0 \
  --audit-device cuda:1 --snr_db 20 --batch_size 200 --seed 0 \
  --output-root outputs/2026-11-02-04-s1-iid-marker-v2
```

각 run의 `metrics/audit_checkpoint_sources.json`에 실제 checkpoint 경로/hash 및 command,
`config.json`에 source config, `metrics/marker_response_score_protocol.json`에 점수의 입력과
변환 규칙을 보존한다. 각 CSV의 모든 점은 같은 invocation에서 figures로 생성된다.

## 실제 완료 결과

6개 run 모두 `completed` 및 현재 `_validate_core_artifacts`의 core/marker/spread/score gate를 통과했다.
각 scope의 score 입력은 동일한 500장이고 global raw control rows는 run별 1,000개다.
Local raw 오검출=2/500 (0.4%), Global raw 오검출=7/500 (1.4%)이며 codec output의
오검출률 보장으로 해석하지 않는다. 관련 테스트 49 passed, 마지막 gate/score 부분집합
재검사 9 passed다. 아래 모든 3개 값의 순서는 Original / Retrain / FSCUv4다.

| 조건 | 완료 run (outputs/ 아래) |
| --- | --- |
| S1 IID | 2026-11-02-04-s1-iid-marker-v2/20261102T115914Z_seed-0 |
| S1 Dirichlet 0.1 | 2026-11-02-06-s1-dirichlet-01-marker-v2/20261102T120805Z_seed-0 |
| S1 Shared-2 | 2026-11-02-08-s1-shared2-marker-v2/20261102T120806Z_seed-0 |
| S2 IID | 2026-11-02-05-s2-iid-marker-v2/20261102T115641Z_seed-0 |
| S2 Dirichlet 0.1 | 2026-11-02-07-s2-dirichlet-01-marker-v2/20261102T120239Z_seed-0 |
| S2 Shared-2 | 2026-11-02-09-s2-shared2-marker-v2/20261102T120239Z_seed-0 |

### MPR (%) 및 검출 건수

모델당 N=500, 괄호는 양성 건수다. Local/Global은 threshold 규칙이 달라 Global 검출률이
항상 더 높아야 하는 것은 아니다.

| 조건 | Local MPR (%), count | Global MPR (%), count |
| --- | --- | --- |
| S1 IID | 13.6 (68) / 0.2 (1) / 0.2 (1) | 1.6 (8) / 1.4 (7) / 0.8 (4) |
| S1 Dirichlet 0.1 | 49.0 (245) / 0.6 (3) / 0.2 (1) | 10.8 (54) / 1.4 (7) / 0.0 (0) |
| S1 Shared-2 | 96.6 (483) / 0.2 (1) / 0.6 (3) | 51.6 (258) / 0.8 (4) / 0.0 (0) |
| S2 IID | 99.0 (495) / 0.0 (0) / 30.0 (150) | 94.6 (473) / 2.0 (10) / 13.6 (68) |
| S2 Dirichlet 0.1 | 99.8 (499) / 0.0 (0) / 55.8 (279) | 99.4 (497) / 2.0 (10) / 27.6 (138) |
| S2 Shared-2 | 99.8 (499) / 0.0 (0) / 29.2 (146) | 95.6 (478) / 2.0 (10) / 10.8 (54) |

### 원래 마커 위치의 amplitude/NCC 점수

| 조건 | Amplitude Original / Retrain / FSCUv4 | NCC Original / Retrain / FSCUv4 |
| --- | --- | --- |
| S1 IID | 100 / 0 / -0.2 | 100 / 0 / -0.1 |
| S1 Dirichlet 0.1 | 100 / 0 / 1.5 | 100 / 0 / 1.7 |
| S1 Shared-2 | 100 / 0 / 1.2 | 100 / 0 / 2.0 |
| S2 IID | 100 / 0 / 13.8 | 100 / 0 / 50.7 |
| S2 Dirichlet 0.1 | 100 / 0 / 16.3 | 100 / 0 / 60.8 |
| S2 Shared-2 | 100 / 0 / 12.0 | 100 / 0 / 51.1 |

### 전체 이미지의 amplitude/NCC 점수

| 조건 | Amplitude Original / Retrain / FSCUv4 | NCC Original / Retrain / FSCUv4 |
| --- | --- | --- |
| S1 IID | 비교 불가 | 100 / 0 / 44.3 |
| S1 Dirichlet 0.1 | 100 / 0 / -62.2 | 100 / 0 / 35.9 |
| S1 Shared-2 | 100 / 0 / -12.8 | 100 / 0 / 27.9 |
| S2 IID | 100 / 0 / 4.1 | 100 / 0 / 50.2 |
| S2 Dirichlet 0.1 | 100 / 0 / 7.5 | 100 / 0 / 63.0 |
| S2 Shared-2 | 100 / 0 / 3.6 | 100 / 0 / 62.3 |

### 원래 위치 바깥의 amplitude/NCC 점수

| 조건 | Amplitude Original / Retrain / FSCUv4 | NCC Original / Retrain / FSCUv4 |
| --- | --- | --- |
| S1 IID | 비교 불가 | 100 / 0 / 60.1 |
| S1 Dirichlet 0.1 | 100 / 0 / -263.0 | 100 / 0 / 41.1 |
| S1 Shared-2 | 비교 불가 | 100 / 0 / 42.6 |
| S2 IID | 불안정 (환산 645.4) | 100 / 0 / 41.6 |
| S2 Dirichlet 0.1 | 비교 불가 | 100 / 0 / 77.9 |
| S2 Shared-2 | 비교 불가 | 100 / 0 / 46.6 |

`비교 불가`는 Original이 Retrain보다 높지 않아 marker-positive 100점 기준이 성립하지 않은
항목이다. `불안정`은 paired reference gap이 표준오차 대비 작아 비율을 해석에서 제외한
항목이다. S2 IID outside amplitude의 raw Original=0.20949743, Retrain=0.20885907,
FSCUv4=0.21297885이고 reference gap=0.00063836, paired SE=0.00087646이라 환산 645.4는
안정적인 강도/삭제 해석에 사용할 수 없다. CSV 및 자동 점수 figure는 clipping 없이 이
계산값도 보존하므로 반드시 `score_status`와 함께 읽는다. 작은 gap으로 커진 비율을
마커가 여러 배 늘었다는 증거로 사용하지 않는다.

음수 점수는 Retrain보다 raw response가 작다는 뜻이지 더 완전한 삭제가 아니다.
기존 보고는 Local 1,000장 평균이었지만 이번 점수는 scope 간 동일한 held-out 500장으로
통일했으므로 과거 표시용 환산값과 소폭 다르다. Score의 분모/평균은 새 per-input CSV로
재현할 수 있다. S1은 Local 반응이 배경 수준에 가까워졌으나 Global/outside NCC gap이
남았고, S2는 Local 및 Global MPR와 NCC에서 Retrain과 명확한 차이가 남았다. 어느 하나도
client-level deletion의 단독 증명은 아니다.

재현성 참고: S1 Shared-2 Global Original은 이전 cuda:0 run의 259/500에서 이번 cuda:1의
258/500으로 한 장 차이가 났다. 동일 index 8901의 raw peak가 0.37927786에서 0.37856066으로
달라 동일 threshold=0.37889966의 경계를 넘나들었다. 검색 규칙과 평가 indices는 같았으며
이 비교만으로 수치 차이의 단일 원인을 확정하지 않는다. 새 표에는 이번 run 값을 사용한다.
