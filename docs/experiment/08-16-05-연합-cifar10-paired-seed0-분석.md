# 균등 데이터량 Non-IID CIFAR-10 연합 삭제 실험: paired seed 0 결과

## 결과 요약

Naive repair와 conflict-aware repair가 **동일한 client별 data order, posterior
noise, channel noise**를 사용하도록 난수를 통제한 seed 0 결과다.

이번 조건에서는 다음이 관찰됐다.

1. Original model은 이미 삭제 대상 제외 재학습 기준과 매우 가깝다.
2. Joint forgetting은 global reconstruction을 심하게 손상한다.
3. Naive repair는 reconstruction quality를 회복하지만 기준 재학습 behavior에는
   original보다 더 가깝지 않다.
4. 현재의 hard conflict projection은 필요한 repair 방향까지 제거해 가장 큰
   functional gap을 만든다.

따라서 **현재 conflict-aware aggregation은 proposed method의 성공 결과가 아니라
수정이 필요한 ablation 결과**다.

## 실행 정보

```text
결과 경로
outputs/federated_cifar10_dirichlet_01_paired_seed/20260816T125735Z_seed-0/

10 clients, client당 train 5,000 / test 1,000
균등 client 크기 제약 Dirichlet 기반 label skew, alpha 0.1
target client: client_0 전체 5,000개
AWGN: 10 dB
original / 삭제 대상 제외 재학습: 200 rounds, local epoch 1
joint forgetting: 200 epochs, 8,000 optimizer steps
repair: naive / conflict-aware 각각 20 rounds
```

Split manifest 검증 결과는 다음과 같다.

```text
모든 client train 수: 5,000
모든 client test 수: 1,000
모든 CIFAR-10 class 전체 train 수: 5,000
```

client_0은 label 0, 6, 2, 3을 주로 보유했다.

```text
label:  0     1  2    3    4  5    6    7  8  9
count: 2125   0 616  555   1  0 1533  156 14  0
```

이 label들은 retained clients에도 남아 있으므로, client_0 삭제 뒤 original과
재학습 기준 model의 차이가 작게 나오는 것은 자연스럽다.

## 전체 public test 결과

`FG`는 삭제 대상 제외 재학습 기준 model과의 functional gap이다. 작을수록
기준 재학습 behavior에 가깝다.

| model | MSE 입력 | PSNR (dB) | FG output | FG posterior mean | FG latent |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.013430 | 18.719 | 0.000153 | 0.063947 | 16.177 |
| 삭제 대상 제외 재학습 기준 | 0.013439 | 18.716 | 0.000000 | 0.000000 | 0.000 |
| joint forgetting | 0.290243 | 5.372 | 0.282573 | 0.064196 | 16.333 |
| naive repair | 0.013365 | 18.740 | 0.000330 | 0.083811 | 21.481 |
| conflict-aware repair | 0.014892 | 18.270 | 0.002296 | 0.251454 | 64.493 |

## 해석

### Original과 재학습 기준의 차이

Original의 output gap은 `0.000153`이다. 삭제 대상 data의 10%를 제거해도
target client가 주로 가진 class support가 retained clients에 남아 있어,
정상적인 재학습 model의 global behavior 변화가 작다.

### Joint forgetting의 과도한 손상

Joint forgetting의 MSE는 `0.290243`으로 재학습 기준의 약 21.6배다. Target
local forgetting 동안의 target MSE도 `0.297450`까지 증가했다. Target을
약화하는 update만으로는 retained reconstruction utility를 보존할 수 없다는
확실한 진단이다.

### Naive repair

Naive repair는 MSE를 `0.013365`까지 복구했다. 그러나 output gap은
`0.000330`으로 original의 약 2.16배다. 이 결과만으로 naive repair가
재학습 기준을 근사한다고 주장할 수 없다.

### Conflict-aware repair

Conflict-aware repair의 output gap은 `0.002296`으로 naive repair의 약
6.96배다. Latent gap도 `64.493`으로 naive repair의 약 3배다. 모든 client
test partition에서 같은 방향의 악화가 나타났다.

현재 구현은 repair update가 original-to-forgetting 방향을 되돌린다고 판단하면
그 projection을 전부 제거한다. 이 hard rule은 실제 삭제 기여를 되돌리는 성분과
retained reconstruction 회복에 필요한 성분을 구별하지 못한다.

## 다음 단계

1. Hard projection 비율을 0 또는 1로 고정하지 않고, projection strength를
   연속적인 ablation으로 둔다.
2. Global forgetting direction 하나가 아니라 encoder/decoder layer 또는 latent
   subspace별 conflict score를 비교한다.
3. 최소 3개, 권장 5개 seed를 실행해 mean과 standard deviation을 보고한다.
4. Target train member와 target held-out test를 모두 기준 재학습 model과
   비교하는 evaluation을 추가한다.
5. 같은 10-client split에서 중앙집중식 SCU self-sharing 비교를 분석용 상한으로
   추가한다. 이는 raw retained data 비중앙화 제약을 만족하는 방법이 아니라는
   점을 분명히 표시한다.
