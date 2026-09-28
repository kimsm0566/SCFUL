# FSCU Trainer 통합 및 자동 평가

## 결정

현재 public-data federated SCU 방법의 공식 이름은 `FSCU`다.

```text
FSCU
  target client joint unlearning
  -> server public-data contrastive compensation
  -> registered headline metric evaluation
```

이 방법을 실질적으로 변경한 구현은 별도 version 이름으로
명명한다. 기존 `FSCU`의 의미나 결과를 나중 방법으로 덮어쓰지 않는다.

## 실행 흐름

사용자는 `src/main.py`에 `--algorithm FSCU`를 전달한다. `main.py`는 모델과
데이터를 별도로 구성하지 않고 `Trainer.train()`을 호출한다. Trainer의 FSCU
분기는 모든 MPI rank에서 동일한 federated pipeline에 참여시킨다.

```text
main.py
  -> Trainer.train
  -> train_fscu
  -> server/client FSCU MPI path
  -> Original federated training
  -> Retrain without target client
  -> target-client joint unlearning
  -> server public-data contrastive compensation
  -> automatic full evaluation
```

실제 target JU와 public CC는 `src/utils/fscu.py`, contrastive loss는
`src/utils/loss.py`, local update와 FedAvg는 `src/utils/federated.py`에 둔다.
`trainer.py`는 중간 pipeline wrapper 없이 FSCU 실행 순서, MPI 동기화, 자동 평가를
직접 담당한다.

## 자동 평가 계약

FSCU 실행은 언러닝 checkpoint를 저장한 뒤 같은 run directory에서 다음 값을
모두 계산한다.

- clean/erased RMSE-255
- clean/erased PSNR
- decoded clean-data Accuracy
- triggered test probe ASR
- reconstruction-loss MIA ROC-AUC
- actual erased backdoor sample BA
- FSCU online Runtime

결과는 `metrics/headline_metrics.csv`의 `FSCU` 행에 저장된다. 하나라도 비어 있으면
Trainer는 완료로 처리하지 않고 오류를 발생시킨다. 자동 평가에서 오류가 발생해도
rank 0이 오류 내용을 다른 MPI rank에 전달하므로 client process가 대기 상태로 남지
않는다.

## 기본 실행 명령

```text
docker exec scful-dev bash -lc 'cd /workspace &&
  mpirun --allow-run-as-root --oversubscribe -np 11
  python src/main.py --algorithm FSCU
  --config <public-data-fscu-config>
  --seed <seed>
  --output-root outputs/YYYY-MM-DD-Index-fscu-title'
```

`--output-root`의 마지막 폴더 이름이 지정 형식과 다르면 실행 전에 중단한다.

## 호환용 보조 명령

과거의 `--experiment federated_deletion`은 현재 `FSCU` Trainer 경로로 연결되는
호환 alias다. 다른 `--experiment` 값은 과거 checkpoint 재평가나 baseline 재현을
위해 남겨 두었으며, 정상적인 FSCU 실행 단계에는 포함되지 않는다. 이 보조 dispatch는
`src/experiments/auxiliary.py`에 분리해 `main.py`에는 개별 실험 이름을 줄줄이 나열하지
않는다.

## 검증 결과

이 문서 작성 당시 smoke run:

```text
outputs/2026-09-07-25-fscu-trainer-core-smoke/
  20260907T122602Z_seed-0/
```

확인 사항:

- `FSCU.pt` 생성
- `client_joint_unlearning.pt` 생성
- `original.pt`와 `retrain_without_client_0.pt` 생성
- `metrics/headline_metrics.csv` 생성
- FSCU 행의 모든 등록 지표 값 확인
- 전체 Python test 125개 통과
