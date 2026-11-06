# Reconstruction Semantic Federated Unlearning

이 저장소는 reconstruction-oriented semantic communication system에서 whole client의
기여를 제거하는 federated unlearning 연구 workspace다.

핵심 목표는 target client를 단순히 나쁘게 복원하게 만드는 것이 아니라,
target client를 제외하고 처음부터 학습한 retraining oracle과 동작이 가까운 codec을
만드는 것이다.

```text
x -> encoder -> power-normalized channel -> decoder -> x_hat
전체 client 학습 -> target client 삭제 요청 -> joint forgetting
-> retained client의 distributed repair -> retrain equivalence 평가
```

현재 구현된 federated 기준선은 `FSCU`와 `FSCUv4`다. 별도의 controlled Non-IID
Oracle Behavior 실험은 centralized comparator이며 제안 방법과 구분한다.

## 현재 확정 지표와 구현 상태 (2026-11-02)

| 목적 | 핵심 지표 | 평가 대상 |
| --- | --- | --- |
| 복원 품질 | PSNR, SSIM | clean test / erased train 각각 입력 대비 복원 |
| 출력 분포 품질 | FID (retain u-FID) | remaining-client clean 입력과 복원 분포 |
| 마커 절대 검출 | MPR (%) | frozen detector의 양성 건수 / held-out 입력 수; 건수와 N도 함께 보고 |
| 마커 제거 | PUL | Original 대비 frozen detector의 검출 빈도 감소 |
| 효율 | Runtime | retraining 및 online unlearning 시간, 평가 시간 제외 |

IDI·CKA·MIA·client-ID 및 custom sensitivity/projection은 핵심 지표가 아니다.
FID/PUL/PSNR/SSIM은 전체 codec 출력 평가이며 decoder 단독 삭제 증명이 아니다.
dFIL은 보조적인 입력 복원 가능성 진단이며 client 삭제 합격 기준이 아니다.
모든 비교 순서는 **Original / Retrain-without-target / Unlearned(FSCUv4)**로 고정한다.
Clean PSNR·SSIM, erased PSNR·SSIM, retain FID를 세 모델 모두 동일 조건에서 보고한다.
S1/S2 모두 MPR을 계산하며, Original 검출이 0이면 MPR=0이어도 PUL은 undefined다.
S1은 연속 marker projection 강도와 NCC도 함께 보고한다. 지정된 마커 위치의
local MPR과 별도로 S1의 whole-image template-bank 평가를 자동 실행한다.
Global/outside amplitude·NCC, global MPR, 검출 픽셀 합집합 면적과 공통 범위 heatmap을
저장한다. 원본 이미지로만 threshold를 보정하고 shifted-marker 대조군의 TPR/FPR도
보고한다. 이는 임의 색 번짐이나 client 삭제를 보장하는 검사가 아니다.

사용자 비교 표에서는 Local/Global/outside **amplitude와 NCC를 실제 측정값 그대로**
표시한다. Original=100점, Retrain=0점 환산은 폐기했다. 기준점 차이로 측정값을
NA 처리하지 않는다. MPR/PUL/coverage의 %, PSNR의 dB, SSIM/FID 원래 값, Runtime의
초도 유지한다. 같은 held-out 500장의 raw mean/SE 자동 CSV/figure를 생성한다.
Local MPR은 raw clean 최대 반응 기준으로 검사하며,
기존 midpoint 기준 결과는 v1으로 보존한다. Global은 raw clean 99% 분위수 기준이며
S1/S2 모두 검사한다. 상세 규칙과 검증 상태는 확정 지표 문서를 따른다.

계산 코드, config, CSV/JSON, 자동 figure 및 reference 저장 연결은 구현되어 있다.
MPR 보고·비교 순서 변경의 테스트와 실제 checkpoint 재평가는 구분한다.
기존 checkpoint를 재학습하거나 기존 결과를 덮어쓰지 않는다. 재평가는 새 output root에 저장한다.
구현 상태와 재현 절차의 기준 문서는
[확정 지표·구현 현황](docs/experiment/10-22-01-dfil-core-metrics.md)이다.
이 절이 아래의 과거 headline 설명보다 우선한다.

CIFAR-10 본 실험은 reconstruction 전용 `VariationalDeepJSCC` codec과 별도
`Cifar10Classifier`를 사용한다. Codec은 원본 DeepJSCC의 5-layer convolutional
encoder/decoder에 posterior mean/log-variance head를 추가하고 classification head는
포함하지 않는다. 구조와 변경 범위는
[`docs/experiment/09-06-06-variational-deepjscc-모델-도입.md`](docs/experiment/09-06-06-variational-deepjscc-모델-도입.md)와
[`docs/experiment/09-09-01-reconstruction-classifier-분리.md`](docs/experiment/09-09-01-reconstruction-classifier-분리.md)에 기록했다.
IID convergence 실험에서 두 모델은 연결되지 않는다. Codec은 reconstruction/KL
objective로만 학습하고 MSE·RMSE-255·PSNR을 추적하며, classifier는 원본 client image의
CE와 accuracy만 사용한다. 등록 v3의 학습 reconstruction term은 batch-mean per-example
pixel SSE이고, 여기서 MSE는 평가용 normalized per-pixel MSE를 뜻한다. 두 모델은
optimizer, FedAvg state, checkpoint, gradient를 공유하지 않는다.
MNIST, compact codec, SCU Table-2 model 분기는 제거했다.

## 구조

```text
docs/                         연구 방법, 문헌, 실험 설계와 결과 분석
third_party/                  수정하지 않는 외부 baseline clone
  SCU/                        semantic communication unlearning baseline
  FedOSD/                     federated unlearning baseline
src/                          이 프로젝트의 연구 구현
  main.py                     모든 학습·언러닝·평가의 단일 실행 진입점
  configs/                    재현 가능한 실험 설정
  data/data.py                모든 dataset load, split, subset, audit probe
  models/                     encoder-channel-decoder codec과 별도 classifier
  evaluation/                 functional gap 및 posterior audit
  utils/                      trainer, FSCU, loss, FedAvg, config, artifact, metric
  tests/                      unit test와 CPU smoke test
outputs/                      non-overwriting experiment artifact
```

`third_party/`는 provenance를 보존하는 baseline 보관소다. 새 연구 코드는 이 경로를
import하지 않는다.

`main.py`에는 SCU/FSCU 조건 분기가 없다. 모든 실행은 MPI와 seed를 초기화한 뒤
`Trainer.prepare_data()`로 현재 rank의 데이터를 할당하고 `Trainer.train()`을 호출한다.
SCU/FSCU 선택은 `Trainer` 안에서만 수행한다. 실제 dataset load, public-data reservation,
client split, rank-local data 할당, split manifest 생성은 모두 `data/data.py`가 담당한다.

## GitHub에서 가져오기

```bash
git clone --recurse-submodules https://github.com/kimsm0566/SCFUL.git
cd SCFUL
```

이미 clone했다면 `git submodule update --init --recursive`로 외부 baseline을 받는다.
외부 코드는 각 원본 저장소의 고정 커밋으로 관리하며, 기존 SCU 호환성 수정은
[`third_party/README.md`](third_party/README.md)에 설명한 별도 패치로 보존한다.

GitHub에는 소스 코드, 설정, 테스트와 연구 문서를 올린다. CIFAR-10 데이터,
`outputs/`의 실험 산출물, `artifacts/`의 학습된 모델, 캐시와 논문 PDF는 포함하지
않는다. 기존 문서에 기록된 로컬 checkpoint/output 경로는 별도로 확보하거나
재생성해야 하며, 저장소 clone만으로 기존 실험 결과가 복원되지는 않는다.

## 실행

모든 프로젝트 코드, test, MPI job, experiment는 Docker development container
`scful-dev`에서만 실행한다. Host에서는 Docker 명령만 실행하며 host Python/Conda를
사용하지 않는다. 컨테이너가 중지되어 있다면 먼저 시작한다.

```bash
docker compose up -d scful-dev
```

다음 명령이 기본 CIFAR-10 SCU 실행 경로를 연다.

```bash
docker compose exec -T scful-dev python src/main.py
```

재현 조건과 빠른 smoke test는 다음처럼 실행한다.

```bash
docker compose exec -T scful-dev python src/main.py --config src/configs/controlled_cifar10.json --seed 3
docker compose exec -T scful-dev python src/main.py --smoke
docker compose exec -T scful-dev python -m pytest -q src/tests
```

현재 public-data federated SCU는 `FSCU`라는 algorithm 이름으로 실행한다.
한 번의 실행 안에서 Original, target 제외 Retrain, target-client joint unlearning,
server public-data contrastive compensation, 전체 headline 지표 평가까지 연속 수행한다.
기본 FSCU 설정은 power-normalized posterior mean을 canonical latent로 사용하고,
짧은 target JU와 server public CC를 5 cycle 동안 교대로 실행한다. Erased buffer는
매 cycle refresh하며 public positive는 두 image augmentation의 deterministic
canonical latent로 만든다.
이미 같은 학습 조건과 seed로 만든 완료 run이 있으면 `--reference-run`으로 지정해
Initial, Original, Retrain checkpoint를 재사용할 수 있다.

```bash
docker compose exec -T scful-dev bash -lc 'cd /workspace && \
  mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --algorithm FSCU \
  --config src/configs/federated_cifar10_vdeepjscc_scu_public_compensation_smoke.json \
  --seed 0 --output-root outputs/YYYY-MM-DD-Index-fscu-smoke'
```

학습 checkpoint를 재사용하고 unlearning부터 실행하는 명령은 다음과 같다.

```bash
docker compose exec -T scful-dev bash -lc 'cd /workspace && \
  mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --algorithm FSCU \
  --config src/configs/federated_cifar10_vdeepjscc_scu_public_compensation_smoke.json \
  --seed 0 --reference-run outputs/<reference-root>/<timestamp>_seed-0 \
  --output-root outputs/YYYY-MM-DD-Index-fscu-unlearning'
```

Reference를 지정하면 모델·학습 설정, seed, exact client split, checkpoint hash를
검증한다. 호환되면 모든 MPI rank가 Original과 Retrain 학습을 함께 건너뛴다.
호환되지 않으면 거부 사유를 기록하고 현재 run에서 새 reference를 학습한다.
`reference_source.json`과 `metrics/training_summary.json`에서 실제 재사용 여부를
확인할 수 있다.

Reconstruction codec과 classifier는 분리되어 있다. 기존 v2 FSCU/reference의 30,000장
codec 학습 scenario에서는 client data와 겹치지 않는 clean CIFAR-10 train 19,000장을
legacy classifier pool로 사용하고, 동일한 frozen CNN으로 모든 reconstruction을 평가한다.
반면 v3 reconstruction-only 49k convergence config는 public 1,000장을 제외한 49,000장을
codec 학습에 모두 사용하며 classifier를 호출하지 않는다. 이 v3 codec에 semantic
observer가 필요한 새 평가 scenario에서는 target을 제외한 remaining clients의 clean
raw image로 classifier를 새로 연합학습한다. Local validation CE로 선택한 classifier
하나를 freeze하며 state, optimizer, gradient, FedAvg, checkpoint는 codec과 공유하지
않는다. Classifier를 사용하는 run은 raw-test accuracy, data index hash,
state hash를 `semantic_classifier/`와 `split_manifest.json`에 기록한다.

FSCU/FSCUv4 평가의 핵심 결과는 `metrics/selected_core_metrics.csv`와
`metrics/selected_core_metrics_protocol.json`, `figures/selected_core_metrics.png`다.
기존 `headline_metrics.csv`의 Accuracy/ASR/MIA/BA 등은 legacy diagnostic이다.
별도 audit는 과거 checkpoint를 다시 분석할 때 사용한다.
Original 및 retrain 학습 중에는 매 round의 train MSE, train PSNR,
objective/reconstruction/KL loss가 terminal과 `run.log`에 동시에 기록된다.
Semantic 지표는 frozen observer를 명시적으로 평가할 때만 기록한다.
Held-out 평가가 설정된 round에는 `test_*` 이름으로 구분해 함께 기록한다.

현재 이름 규칙에서 `FSCU`는 public-data compensation 기준선이고, `FSCUv4`는
target-client joint forgetting 뒤 remaining clients가 자기 retained data로 full codec을
repair하고, server가 JU 역방향 client update 성분을 projection한 뒤 집계하는 방법이다.

`FSCUv4`는 reconstruction-only V3 codec, latent channel 16, standard VIB beta 0.0001,
49,000장, batch 200, 800-round reference protocol을 사용한다. Frozen classifier는
headline Accuracy/ASR/BA를 측정하는 observer일 뿐 codec loss나 gradient에 참여하지 않는다.
과거 observer는 `artifacts/semantic_observers/cifar10-iid-49k-seed0/`에 legacy artifact로
보관한다. 새 remaining-client observer는 `--experiment remaining_classifier`로 학습한
뒤 `--observer-run`으로 지정한다. Codec의 출력 PSNR/SSIM 및 공통 classifier의
class/logit/probability 비교를 자동 저장한다. 기존 결과는 덮어쓰지 않는다.
실행 절차는 [remaining-client observer 구현](docs/experiment/09-22-01-remaining-client-observer-구현.md)을 따른다.

```bash
docker compose exec -T scful-dev bash -lc 'cd /workspace && \
  mpirun --allow-run-as-root --oversubscribe -np 11 \
  python src/main.py --algorithm FSCUv4 \
  --config src/configs/federated_cifar10_iid_vdeepjscc_fscuv4.json \
  --seed 0 --output-root outputs/YYYY-MM-DD-Index-fscuv4-iid'
```


현재 코드에는 `training/`과 `unlearning/` package가 없다. `trainer.py`의 단일
`train_fscu()` 흐름을 모든 MPI rank가 함께 실행하고, 재사용 가능한 local FedAvg 계산은
`utils/federated.py`, FSCU의 두 최적화 단계는 `utils/fscu.py`, loss는
`utils/loss.py`에 둔다. 과거 범용 federated-deletion runner와 실행되지 않는 연구
prototype 분기는 제거했다.

새 output root는 반드시 `outputs/YYYY-MM-DD-Index-brief-title/` 형식으로 만든다.
날짜는 한국 시간이고, `Index`는 같은 날짜에 만든 output root의 두 자리 순번이다.
같은 설정을 반복해도 새 index를 사용한다. 각 run은 그 output root 아래의
`<UTC timestamp>_seed-<seed>/`에 checkpoint,
split manifest, per-sample metric, summary, prediction tensor, figure, environment 정보와
run log를 저장한다.

모든 실험은 같은 실행에서 `metrics/`의 원본 CSV/JSON과 `figures/`의 그래프를 함께
생성해야 완료로 인정한다. Reconstruction 학습은 objective/reconstruction/KL loss와
held-out RMSE-255/PSNR 곡선을, 독립 classifier 학습은 CE/accuracy 곡선을 만든다.
Federated unlearning은 Original, Retrain-without-target, Unlearned의 task별 utility,
unlearning 지표와 Runtime 비교를 multi-panel figure로 남긴다. Classifier-only run에
PSNR을 넣거나 reconstruction-only codec에 classifier CE를 역전파하지 않는다. Train과
held-out/test 선은 명확히 구분하고 smoothing을 쓰더라도 raw curve를 보존한다.

## 연구 원칙

- unlearning의 기준은 retrain-without-target oracle과의 functional gap이다.
- target reconstruction MSE 증가만으로 성공을 주장하지 않는다.
- encoder와 decoder를 함께 다루는 reconstruction semantic codec을 유지한다.
- 최종 방법의 핵심은 raw retained data를 서버에 모으지 않는 distributed repair와
  forgetting을 되돌리지 않는 conflict-aware aggregation이다.
