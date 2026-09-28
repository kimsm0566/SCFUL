# FSCU Reference Checkpoint 재사용

## 목적

같은 모델, 학습 설정, client split, seed로 unlearning 방법만 반복 비교할 때마다
Original과 Retrain을 다시 학습하지 않는다. 완료된 FSCU run의 Initial, Original,
Retrain checkpoint를 검증해 재사용하고 target joint unlearning부터 실행한다.

## 실행 계약

- `--reference-run` 없음: Initial 생성, Original 학습, target 제외 Retrain 학습,
  joint unlearning, public-data compensation, 평가를 모두 실행한다.
- `--reference-run` 있음: 호환되면 모든 MPI rank가 Original과 Retrain 학습을
  건너뛰고 joint unlearning부터 실행한다.
- 지정한 reference가 호환되지 않으면 거부 사유를 남기고 새 reference를 학습한다.
- 서버와 client는 별도 pipeline 함수로 갈라지지 않는다. 모든 rank가
  `train_fscu()`의 동일한 SPMD 순서를 따른다.

## Reference 검증 항목

- global seed, initial-state seed와 state hash
- dataset과 public-data reservation
- codec architecture 및 전체 codec 설정
- client 수, client당 sample 수, round, local epoch, optimizer-state 유지 설정
- optimizer와 aggregation 방식
- split 유형과 full `split_manifest.json`
- 학습 loss weight, learning rate, batch size, train SNR
- target client ID와 backdoor 주입 설정
- Initial, Original, Retrain checkpoint의 parameter key, shape, dtype
- checkpoint byte size와 SHA-256

기존 schema-1 reference bank도 fingerprint와 hash를 다시 확인해 사용할 수 있다.
다만 client별 Adam state 유지 여부가 manifest에 없는 초기 legacy run은 학습 절차를
확정할 수 없으므로 재사용하지 않는다.

평가 SNR이나 unlearning hyperparameter는 reference 학습 checkpoint 자체를 바꾸지
않으므로 재사용 호환성 조건에서 제외한다.

## Artifact

새로 학습한 run은 canonical scenario fingerprint와 checkpoint hash를 담은
`reference_manifest.json`을 생성한다. 재사용 run은 검증된 source checkpoint를
복사하지 않고 checkpoint 경로에 symbolic link를 만든다. 추가 기록은 다음과 같다.

- `reference_source.json`: 요청한 reference, 실제 재사용 여부, source, fingerprint,
  checkpoint SHA-256, 거부 사유
- `metrics/training_summary.json`: 재사용 여부와 source run
- `metrics/efficiency.json`: source Retrain 시간, 현재 unlearning 시간, 별도 load 시간
- `run.log`: reference 재사용 또는 거부 여부

Checkpoint loading 시간은 Retrain과 online unlearning runtime 양쪽에서 제외한다.
