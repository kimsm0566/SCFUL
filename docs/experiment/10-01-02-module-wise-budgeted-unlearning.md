# Module-wise budgeted client unlearning

## 목적

기존 FSCUv4의 target-client joint forgetting을 다음과 같이 분리한다.

```text
Original codec
-> decoder 고정, encoder information-rate unlearning
-> encoder 고정, decoder distortion-budget unlearning
-> local unlearned codec
-> remaining-client distributed repair
-> conflict-aware aggregation
```

두 local 단계는 같은 target-client 데이터에서 순차적으로 수행하지만 서로 다른
종료 신호를 사용한다. 이 설계의 목적은 encoder와 decoder가 각각 얼마나 바뀌었는지
구분하지 못한 채 하나의 고정 epoch 수로 과도하게 손상되는 문제를 줄이는 것이다.

## 1단계: encoder information-rate unlearning

- Decoder는 완전히 고정한다.
- Target-client 입력의 posterior를 표준 정규 prior에 가깝게 만든다.
- 매 epoch 뒤 target prior-KL을 전체 target 데이터에서 다시 측정한다.
- `현재 target prior-KL / Original target prior-KL`이 설정한 threshold 이하가 되면
  즉시 종료한다.
- 최대 epoch는 안전장치이며 threshold에 먼저 도달하면 남은 epoch는 수행하지 않는다.

초기 pilot 설정은 다음과 같다.

- information ratio threshold: 0.80
- encoder maximum epochs: 20
- encoder learning rate: 0.0001

이 값은 target-client ID를 직접 측정하는 leakage 지표가 아니라 VIB
information-rate 제어 신호다. 최종 삭제 성공 여부는 retrain equivalence, MIA,
latent client-ID leakage, BA를 함께 사용해 판단한다.

## 2단계: decoder distortion-budget unlearning

- 1단계에서 얻은 encoder는 고정한다.
- Decoder만 target reconstruction error가 증가하는 방향으로 갱신한다.
- Encoder 손상으로 decoder 종료 신호가 과장되지 않도록, decoder probe에는 고정된
  Original encoder가 생성한 posterior mean과 paired AWGN을 입력한다.
- Probe PSNR이 Original decoder probe PSNR보다 설정한 budget만큼 낮아지면 종료한다.
- 최대 epoch는 threshold 미도달 시의 안전장치다.

초기 pilot 설정은 다음과 같다.

- decoder PSNR drop budget: 2 dB
- decoder maximum epochs: 20
- decoder learning rate: 0.0001

이 PSNR 하락은 decoder의 변경량을 제한하는 local control signal이다. Target PSNR을
낮추는 것 자체를 unlearning 성공으로 해석하지 않는다.

## 3단계: distributed repair

완성된 local unlearned codec의 encoder와 decoder를 함께 remaining clients에 전달한다.
Remaining clients는 각자의 local retained data만 사용해 full codec을 repair한다.
서버는 target joint-unlearning 방향을 되돌리는 client update 성분을 projection한 뒤
집계한다. Target-client raw data와 Original target output은 repair에 사용하지 않는다.

현재 pilot은 기존 FSCUv4 repair 설정을 유지한다.

- repair variant: feature KD
- posterior mean/log-variance preservation: enabled
- visible-marker outside-region consistency: enabled when marker protocol is used
- aggregation: client-wise conflict-aware projection
- checkpoints: repair rounds 5, 10, 20, 50, 100

## 자동 산출물

각 실행은 다음을 자동 저장한다.

- `metrics/joint_unlearning_history.csv`
  - stage
  - target prior-KL ratio
  - end-to-end target PSNR
  - Original-encoder decoder-probe PSNR
  - decoder PSNR threshold
- `figures/fscuv4_module_wise_unlearning.png`
  - encoder information-rate threshold
  - decoder distortion threshold
  - end-to-end target reconstruction
  - 단계별 objective
- 기존 FSCUv4 repair history 및 headline metric suite

## 설정 파일

- Smoke:
  `src/configs/federated_cifar10_vdeepjscc_module_budgeted_fscuv4_smoke.json`
- 800-round masked-input marker scenario:
  `src/configs/federated_cifar10_iid_vdeepjscc_marker_masked_input_fscuv4_module_budgeted_800rounds.json`

## 해석 시 주의점

1. Encoder prior-KL ratio는 client identity leakage와 동일하지 않다.
2. Decoder PSNR drop은 최종 deletion metric이 아니라 과손상을 막는 budget이다.
3. 두 threshold에 도달하지 못한 실행은 성공으로 간주하지 않고
   `threshold_reached=false`를 보고한다.
4. 최종 모델은 반드시 Original 및 retrain-without-target과 동일한 headline suite로
   비교한다.
5. Threshold 선택은 공식 test 결과에 맞춰 사후 조정하지 않고 별도 pilot/validation
   규칙으로 정한다.
