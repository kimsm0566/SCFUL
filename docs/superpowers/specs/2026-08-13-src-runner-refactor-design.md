# `src` 실행 구조 리팩터링 설계

## 목적

현재 `src/`에는 새 연구 코드와 외부 baseline clone이 함께 있고,
`src/experiments/oracle_behavior.py`가 split, 학습, SCU adaptation, 평가, figure,
artifact 저장을 한 파일에서 수행한다. 이 설계는 baseline clone을 분리하고,
`src/main.py` 하나로 기본 Oracle Behavior 실험 전체를 실행하되 각 책임을 작은
모듈로 나누는 구조를 정의한다.

참고한 CA-SSFL은 `src/data`, `src/models`, `src/utils`와 단일 runner를 사용한다.
이 프로젝트는 그 단순한 실행 진입 구조를 따르되, 연구 메서드의 경계를 명확히
유지한다. 특히 federated training, client-level unlearning, 향후 distributed repair는
`utils`에 섞지 않는다.

## 범위와 비범위

이번 리팩터링의 범위는 현재의 controlled Non-IID Oracle Behavior 실험을 같은
의미와 artifact schema로 실행하도록 재구성하는 것이다.

- `python src/main.py`가 기본 MNIST Oracle Behavior seed-0 run을 처음부터 끝까지 수행한다.
- `--config`, `--seed`, `--smoke`, `--output-root` 옵션을 지원한다.
- 기존 `src/SCU`, `src/FedOSD` clone은 `third_party/`로 이동한다. clone 내부의 `.git`과
  현재 변경 상태를 보존하며, 내용을 수정하거나 삭제하지 않는다.
- 기존 `outputs/oracle_behavior/`의 완료된 결과와 `src/SCU` 호환성 수정은 변경하지 않는다.

이번 작업에서 새 federated algorithm, distributed repair loss, conflict-aware aggregation,
새 데이터셋 또는 새 결과 run을 추가하지 않는다. 다만 이들을 다음 단계에서 넣을 수
있는 모듈 경계를 만든다.

## 목표 구조

```text
third_party/
  SCU/                         # 외부 baseline clone; provenance 보존
  FedOSD/                      # 외부 baseline clone; provenance 보존

src/
  main.py                      # CLI, config 로드, 단일 workflow 호출만 담당
  configs/
    oracle_mnist.json          # 기본 실행 설정

  data/
    mnist.py                   # MNIST train/test load, Tensor transform
    client_split.py            # target client 및 retained client split
    probes.py                  # train-member, held-out, retained audit probe

  models/
    semantic_codec.py          # VAE encoder, power normalization, AWGN, decoder

  training/
    reconstruction.py          # Original / Retrain 공통 codec trainer

  unlearning/
    scu_style.py               # frozen original teacher 기반 SCU-style adaptation

  evaluation/
    oracle_behavior.py         # workflow orchestration과 model별 functional-gap 평가
    posterior_audit.py         # client posterior Gaussian sufficiency 진단

  utils/
    artifacts.py               # non-overwriting run directory와 결과 파일 저장
    config.py                  # JSON config 읽기, override 및 검증
    metrics.py                 # MSE, posterior gap, bootstrap summary
    plotting.py                # reconstruction 및 latent figure 생성
    reproducibility.py         # seed, deterministic setting, environment 기록

  tests/
    ...                        # 모듈 단위 및 CPU smoke test
```

`__init__.py`는 import 가능한 package 경계를 유지하기 위해 각 package에 남긴다.
새 코드에서는 `src` package 기준 absolute import를 사용한다. `main.py`만
`python src/main.py`와 `python -m src.main` 두 호출 방식에서 모두 동작하도록
최소한의 import bootstrap을 가진다.

## 실행 계약

기본 명령은 다음이다.

```bash
python src/main.py
```

기본 config는 `src/configs/oracle_mnist.json`이며, 다음 순서를 모두 수행한다.

```text
config 검증 및 seed 설정
-> MNIST load 및 controlled Non-IID split 생성
-> initial checkpoint 저장
-> Original 학습
-> target client 제거 Retrain oracle 학습
-> retrain replica 학습
-> SCU forget-only 및 self-sharing adaptation
-> posterior audit 및 functional-gap 평가
-> checkpoint, per-sample CSV, summary JSON, prediction tensor, figures, run log 저장
```

재현 실행은 다음 인터페이스를 사용한다.

```bash
python src/main.py --config src/configs/oracle_mnist.json --seed 3
python src/main.py --smoke
python src/main.py --output-root outputs/oracle_behavior
```

`main.py`는 argument parsing과 `run_oracle_behavior(...)` 호출만 수행한다.
dataset loop, optimizer loop, SCU loss, metric 계산, matplotlib 호출을 넣지 않는다.

## 모듈 경계와 데이터 흐름

`data`는 raw MNIST dataset, client index, probe reference만 만든다. model이나
optimizer를 import하지 않는다. `models`는 input tensor에서 posterior와 reconstruction을
만들지만 disk I/O나 experiment metadata를 알지 못한다.

`training/reconstruction.py`는 model, dataset, typed training config, seed bundle을 받아
학습 결과를 반환한다. Original과 Retrain은 같은 `initial.pt` state를 받으며 새 Adam
optimizer에서 시작한다. `unlearning/scu_style.py`는 completed Original model의 frozen
teacher와 복제 learner를 받는다. target/retained loader pairing, SCU loss, gradient
clipping만 담당한다.

`evaluation/oracle_behavior.py`는 위 modules를 조립하고 evaluation condition별 record를
만든다. raw metric 함수와 plot rendering은 각각 `utils/metrics.py`, `utils/plotting.py`로
내린다. artifact persistence는 `utils/artifacts.py` 한 곳에서만 한다.

이 분리는 다음 연구 불변조건을 유지한다.

- codec은 encoder-channel-decoder의 end-to-end reconstruction model이다.
- unlearning의 비교 기준은 target client 제외 retraining oracle이다.
- target MSE 증가만을 성공 기준으로 사용하지 않는다.
- 현재 SCU-style comparator는 centralized oracle이며, 이후 distributed repair는
  `unlearning/`에 추가하고 retained raw data를 서버에 중앙화하지 않는다.

## 설정과 artifact 호환성

기본 JSON의 값과 의미는 이전 pilot과 동일하게 유지한다. 예를 들어 latent dimension 20,
hidden width 400, 10 dB AWGN, batch size 128, Original/Retrain 20 epochs,
SCU 20 epochs 및 동적 self-sharing 계수는 바꾸지 않는다. 이는 리팩터링 전후를
비교 가능한 code-organization change로 한정하기 위함이다.

출력은 계속 `outputs/oracle_behavior/<UTC timestamp>_seed-<seed>/` 아래에 다음을
남긴다.

```text
config.json, environment.json, split_manifest.json, run.log
checkpoints/{initial,original,retrain,retrain_replica,scu_forget_only,scu_self_sharing}.pt
metrics/per_sample.csv, metrics/summary.json
predictions/probes.pt
client_posterior/per_client_gaussian.csv, client_posterior/summary.json
figures/reconstruction_grid.png, figures/latent_comparison.png,
figures/client_gaussian_diagnostic.png
```

과거 run을 덮어쓰지 않으며, config/environment/split/seed를 각 run에 기록한다.

## 오류 처리와 검증

- config 누락 key, 지원하지 않는 SCU mode, CUDA를 요구하지만 사용할 수 없는 device,
  split coverage violation은 즉시 명시적 예외로 중단한다.
- `--smoke`는 synthetic CPU dataset으로 모든 artifact 유형을 빠르게 생성한다.
- 리팩터링 단계마다 해당 모듈의 unit test를 먼저 추가하거나 이동하고, failing test를
  확인한 뒤 최소 구현으로 통과시킨다.
- 최종 검증은 `python -m unittest discover -s src/tests -v`, touched package compile,
  `python src/main.py --smoke`, CUDA seed-0 pilot 순으로 수행한다.
- smoke/pilot의 output schema, split assertions, metric finite 여부, retrain self-gap 0을
  기존 실험 계약과 비교한다.

## 이동 규칙

`src/SCU`와 `src/FedOSD`는 파일 단위로 재작성하지 않고 디렉터리 단위로
`third_party/SCU`, `third_party/FedOSD`로 이동한다. top-level workspace가 Git repository가
아니므로 top-level commit은 만들 수 없다. nested clone의 history 및 working tree는 그대로
보존한다. 새 연구 코드는 `third_party/`를 import하지 않으며, SCU baseline을 재실행할 때만
명시적으로 해당 경로를 사용한다.
