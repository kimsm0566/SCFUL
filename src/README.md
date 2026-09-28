# `src/` 구현 안내

`src/`에는 이 프로젝트가 직접 관리하는 연구 코드만 둔다. 외부 baseline clone은
`third_party/`에 보관하며 `src/`에서 import하지 않는다.

## 실행 진입점

`src/`의 코드도 host Python이 아니라 Docker service `scful-dev` 안에서만 실행한다.
컨테이너 내부 working directory는 `/workspace`이고 repository root와 bind mount되어
있다.

```bash
docker compose up -d scful-dev
docker compose exec -T scful-dev python src/main.py
```

`src/main.py`는 config를 읽고 `run_oracle_behavior(...)`를 한 번 호출한다. 따라서
optimizer loop, SCU loss, metric 계산, matplotlib figure 생성 로직을 넣지 않는다.

지원 옵션은 다음과 같다.

```bash
docker compose exec -T scful-dev python src/main.py --config src/configs/controlled_cifar10.json --seed 3
docker compose exec -T scful-dev python src/main.py --smoke
docker compose exec -T scful-dev pytest -q src/tests
docker compose exec -T scful-dev python -m py_compile src/utils/trainer.py src/utils/fscu.py
```

## package 책임

```text
src/configs/                 기본 JSON 설정
src/data/data.py             모든 dataset load, client split, subset, audit probe
src/models/                  VAE encoder-channel-decoder semantic codec
src/evaluation/              Oracle workflow와 client posterior diagnostic
src/utils/trainer.py         SCU/FSCU 실행 순서와 MPI server/client orchestration
src/utils/federated.py       Original/retrain local update와 FedAvg
src/utils/fscu.py            target joint unlearning과 public-data compensation
src/utils/loss.py            FSCU가 사용하는 순수 loss 함수
src/tests/                   unit test와 end-to-end CPU smoke test
```

현재 기본 federated workflow는 public-data `FSCU`다. `trainer.py`에서 Original,
target 제외 Retrain, target joint unlearning, server public-data compensation,
자동 평가를 직접 순서대로 실행한다. 향후 distributed repair와 conflict-aware
aggregation은 새 버전으로 명명하고 관련 계산을 `src/utils/`의 역할별 파일에
추가하며, retained client의 raw data를 server로 중앙화하지 않는다.

CIFAR-10 codec은 `VariationalDeepJSCC` 하나만 사용한다. 원본 DeepJSCC topology에
posterior mean/log-variance head를 추가한 model이며, federated deletion, controlled
comparison, MIA audit가 같은 factory를 사용한다. 다른 dataset/model resolver는 없다.

## artifact 계약

매 experiment invocation에는 새 output root를 지정한다.

```text
outputs/YYYY-MM-DD-Index-brief-title
```

날짜는 한국 시간이고 `Index`는 같은 날짜에 생성한 두 자리 순번이다. 반복 실행도
새 index를 사용한다. 각 run은 이 root 아래에 config, environment, split manifest,
checkpoint, CSV/JSON metrics, prediction tensor, figure, run log를 남긴다. metric 해석의
주 기준은 output/posterior functional gap to retrain이며, target MSE는 보조 진단이다.

모든 experiment entry는 metric 저장 직후 같은 invocation에서 비대화형 figure를
`figures/`에 생성한다. Reconstruction training은 loss component와 held-out
RMSE-255/PSNR, classifier training은 CE/accuracy, federated unlearning은
Original/Retrain/Unlearned의 task별 headline metric과 Runtime을 그린다. Iterative
unlearning은 stage별 optimization loss도 남긴다. 그림의 각 point는 `metrics/`의
CSV/JSON row로 복원 가능해야 하며, 필수 figure가 없으면 run completion을 기록하지
않는다. 공용 plotting은 `src/utils/plotting.py`에 두고 artifact filename과 non-empty
metric source를 test로 검증한다.
