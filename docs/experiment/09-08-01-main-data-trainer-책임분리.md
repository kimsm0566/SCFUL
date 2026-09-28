# Main, Data, Trainer 책임 분리

## 목적

과거 `main.py`는 FSCU를 초반에 별도 함수로 보내고, SCU에 대해서만 dataset load와
client data 할당을 수행했다. 반대로 FSCU의 dataset load와 split은 `train_fscu()`가
직접 시작했다. 이 구조에서는 실행 진입점과 trainer의 책임이 알고리즘마다 달랐다.

## 현재 공통 실행 흐름

```text
main.py
  -> argument와 MPI communicator 준비
  -> 공통 seed 초기화
  -> Trainer.prepare_data
  -> Trainer.train
  -> rank 0 결과 출력
```

`main.py`에는 SCU/FSCU 조건 분기와 FSCU early return이 없다. `federated_deletion`
legacy alias 정규화, 보조 실험 선택, SCU/FSCU data strategy 선택과 학습 함수 선택은
모두 `Trainer` 안에서 처리한다.

## Data 책임

다음 코드는 모두 `src/data/data.py`에 둔다.

- SCU dataset load와 single-client 할당
- FSCU CIFAR-10 train/test load
- server public-data reservation
- client training pool 선택
- IID, Dirichlet, balanced 2-class split
- 현재 MPI rank의 local dataset 선택
- target-client backdoor 주입
- split seed와 full split manifest 생성

FSCU에서 rank 0은 server용 test/public data와 split metadata만 받고 raw client data는
받지 않는다. rank 1부터는 자신의 local dataset만 `data_state["client_data"]`로 받는다.

## Trainer 책임

`Trainer.prepare_data()`는 SCU/FSCU에 맞는 `data/data.py` 함수를 선택하고, 모든 학습
함수에 전달할 공통 `data_state`를 만든다. `train_scu()`와 `train_fscu()`는 이미 준비된
data state를 입력으로 받으며 dataset을 직접 load하거나 split하지 않는다.

`train_fscu()`에는 MPI 실행 순서가 계속 명시적으로 남아 있다.

- reference load 또는 Original/Retrain 학습
- target-client joint unlearning
- server public-data contrastive compensation
- checkpoint와 efficiency 저장
- 전체 headline metric 평가

## 검증

- 전체 Python test 80개 통과
- 단일-process SCU main 실행 및 checkpoint 저장 테스트 통과
- 11-rank FSCU reference-reuse smoke 통과
- smoke output: `outputs/2026-09-08-01-main-common-data-flow-smoke`
- 재사용 run의 Original/Retrain training history가 비어 있음을 확인
