# 공개 데이터 기반 PGA-IB 분산 repair: seed 0 결과

## 목적

이 실험은 새 프레임워크의 첫 end-to-end 구현을 점검한다. 목표는 target client 0의
정상 local model `U`, target client가 PGA로 만든 local unlearning model `G-un`, 그리고
target을 제외한 last-round local model aggregation `G-u`를 서버가 함께 사용해
초기 unlearning codec을 만드는 것이다.

서버는 raw target data나 raw retained data를 받지 않는다. 대신 client split 전에
분리한 공개 calibration data `D-pub`에서 세 model의 representation을 비교한다. 그 뒤
남은 9개 client가 local retained data만으로 repair하고, naive FedAvg와 conflict-aware
aggregation을 각각 비교한다.

이 문서의 비교 기준은 항상 **client 0 제외 재학습 model**이다. target client test
reconstruction MSE의 증가 자체는 성공 지표가 아니다.

## 실행 설정

실행 명령은 다음과 같다.

```text
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_pga_public_ib.json \
  --seed 0 \
  --output-root outputs/pga_public_ib_repair
```

| 항목 | 설정 |
| --- | --- |
| 데이터 | CIFAR-10 train 50,000개 |
| 공개 calibration data `D-pub` | train split 전 class당 100개, 총 1,000개 |
| FL client pool | 공개 data를 제외한 49,000개 |
| client 수와 data 수 | 10 clients, client당 4,900개 |
| non-IID | equal-size Dirichlet, alpha 0.1 |
| FL | 200 rounds, local epoch 1, batch 128 |
| codec/channel | CIFAR-10 VAE codec, latent 64, AWGN 10 dB |
| target | client 0 |
| PGA | SGD, epoch 20, learning rate 0.0001, momentum 0.9 |
| PGA projection | `G-u`까지의 초기 parameter 거리의 2배 반경 |
| 서버 공개-data 단계 | Adam, epoch 20, learning rate 0.0001 |
| distributed repair | remaining 9 clients, 20 rounds, local epoch 1 |

`D-pub`은 client train data와 겹치지 않는다. split manifest에 저장된 공개 index hash는
`cf8e178e14026be68021f5ab2117ab6d5a475f8fef53e9b07a68306d558323cb`이다.

## 구현한 흐름

```text
정상 FL 완료
-> 마지막 round에서 target을 제외한 local model 평균: G-u
-> target의 정상 local model: U
-> target이 original global model과 D_u로 PGA 수행: G-un
-> target은 이후 repair에 참여하지 않음
-> 서버가 D-pub에서 G-u, U, G-un의 posterior mean과 AWGN reconstruction 비교
-> G-u에서 시작한 server public-IB model 생성
-> remaining clients의 local repair
-> naive FedAvg 및 conflict-aware aggregation 비교
```

서버 공개-data loss는 직접 mutual information을 추정하지 않는 첫 surrogate다.

```text
server model과 U의 representation similarity는 낮춤
server model과 G-un의 representation similarity는 높임
server model이 초기 G-u에서 과도하게 멀어지는 것은 anchor로 제한
```

Representation에는 encoder posterior mean `mu`와, 같은 AWGN noise를 모든 model에
공통 적용한 reconstruction을 함께 사용했다. 따라서 encoder와 decoder 모두 gradient를
받는다.

## PGA 및 서버 단계 진단

PGA는 780 update를 수행했고, 540번 projection이 발생했다.

| PGA 진단 | 값 |
| --- | ---: |
| target reconstruction MSE 학습 평균 | 0.050248 |
| 시작점과 `G-u`의 parameter 거리 | 0.034119 |
| projection 반경 | 0.068239 |
| 종료 시 `G-u`까지의 거리 | 0.068239 |

`G-un`은 projection 경계에서 끝났고 target data에 대한 reconstruction을 강하게
변화시켰다. 이는 `G-un`이 최종 global model이 아니라 local forgetting signal이라는
설계와 일치한다.

서버 공개-data 단계는 1,000개 image를 20 epoch 사용해 160 update를 수행했다.

| 서버 단계 진단 | 값 |
| --- | ---: |
| target normal model `U`와의 representation similarity | 1.750073 |
| PGA model `G-un`과의 representation similarity | 1.975993 |
| 초기 `G-u` anchor loss | 0.247428 |

## 전체 test 결과

`output gap`은 각 model reconstruction과 client 0 제외 재학습 model reconstruction의
평균 제곱 차이다. 작을수록 재학습 기준 model과 가깝다.

| model | 전체 MSE | PSNR | output gap | posterior mean gap |
| --- | ---: | ---: | ---: | ---: |
| original | 0.013516 | 18.6916 | 0.000229 | 0.100503 |
| client 0 제외 재학습 | 0.013516 | 18.6914 | 0.000000 | 0.000000 |
| `G-u` remaining reference | 0.013511 | 18.6931 | 0.000228 | 0.100516 |
| `U` target normal local model | 0.013657 | 18.6466 | 0.000347 | 0.104082 |
| `G-un` target PGA model | 0.072121 | 11.4194 | 0.058370 | 0.482545 |
| server public-IB 직후 | 0.023784 | 16.2372 | 0.011942 | 0.312857 |
| server public-IB + naive repair | 0.013441 | 18.7156 | 0.000264 | 0.109205 |
| server public-IB + conflict-aware repair | 0.013844 | 18.5873 | 0.000533 | 0.133127 |

## target client 0 test 결과

| model | client 0 test MSE | client 0 output gap |
| --- | ---: | ---: |
| original | 0.014533 | 0.000239 |
| client 0 제외 재학습 | 0.014564 | 0.000000 |
| `G-u` remaining reference | 0.014532 | 0.000238 |
| `G-un` target PGA model | 0.070752 | 0.055274 |
| server public-IB 직후 | 0.025817 | 0.013197 |
| naive repair | 0.014489 | 0.000274 |
| conflict-aware repair | 0.014930 | 0.000541 |

client 0을 제외한 재학습 model도 client 0 test image를 잘 복원한다. 따라서 target
client image가 잘 복원된다는 사실은 삭제 실패가 아니다. 이 split에서는 target client의
semantic이 remaining clients에도 충분히 지지되는 상황으로 해석해야 한다.

## 해석

### 1. PGA local model은 forgetting signal로는 작동하지만 그대로 보존 reference가 될 수 없다

`G-un`의 output gap은 0.058370으로 매우 크고 reconstruction MSE도 0.072121이다.
PGA는 target mapping을 약화했지만 shared semantic과 codec utility도 함께 손상했다.

따라서 서버가 `G-un` 전체 representation과 가까워지도록만 학습하면 그 손상을 일부
복사하게 된다. 실제로 server public-IB 직후 output gap은 0.011942로 `G-u`의 0.000228보다
훨씬 나빠졌다.

### 2. naive distributed repair는 reconstruction utility를 거의 회복했다

naive repair 뒤 전체 MSE는 0.013441로 회복됐고 output gap도 0.000264까지 내려갔다.
하지만 이는 `G-u`의 0.000228보다 작지 않다. 절대 차이는 0.000036으로 매우 작으므로,
현재 single-seed 결과만으로 naive repair가 `G-u`보다 나쁘다고 강하게 주장할 수는 없다.
다만 **제안한 서버 IB 단계가 재학습 기준에 더 가까워지게 만들었다는 증거도 없다.**

### 3. 현재 conflict-aware projection은 오히려 나빴다

conflict-aware repair의 output gap은 0.000533으로 naive repair보다 컸다. 현재 projection은
server IB model과 original global model의 전체 parameter 차이를 forgetting direction으로
사용한다. 이 direction에는 target-specific 변화뿐 아니라 PGA/서버 단계의 utility damage도
섞여 있다. 그래서 repair가 복구해야 할 정상 성분까지 막았을 가능성이 크다.

### 4. 새 프레임워크의 핵심 가설은 아직 검증되지 않았다

이번 결과는 다음을 보여 준다.

```text
G-u, U, G-un 세 model을 사용하는 것 자체는 가능하다.
하지만 G-un 전체와의 representation similarity를 높이는 단순 surrogate는
“보존해야 할 공통 정보”만 골라내지 못한다.
```

즉 원래 의도한 information bottleneck은 `G-un` 전체를 복사하는 것이 아니라,
`G-u`와 `G-un`이 함께 지지하면서 `U`와 구별되는 부분을 찾아야 한다. 현재 loss는 이
공통 부분을 명시적으로 분리하지 못했다.

## 다음 수정 방향

1. `G-un` 전체 similarity 대신, `G-u`와 `G-un`의 agreement를 latent dimension 또는
   low-rank subspace 수준에서 계산한다.
2. 그 agreement 중 `U`와 구별되는 부분만 target-specific forgetting/retained protection
   signal로 사용한다.
3. 서버 anchor는 `G-u`의 전체 utility를 보존하되, target-specific subspace의 변화까지
   막지 않도록 분리한다.
4. conflict-aware aggregation도 전체 parameter update가 아니라 위에서 식별한
   target-sensitive subspace 또는 filtered forgetting update를 기준으로 구성한다.
5. 수정 loss가 정해진 뒤 여러 seed, support regime, Rayleigh/SNR mismatch 조건에서
   재검증한다.

## 산출물

```text
outputs/pga_public_ib_repair/20260819T050240Z_seed-0/
  config.json
  split_manifest.json
  checkpoints/original.pt
  checkpoints/retrain_without_client_0.pt
  checkpoints/remaining_reference_G_minus_u.pt
  checkpoints/target_normal_local_U.pt
  checkpoints/target_pga_unlearned_G_un.pt
  checkpoints/server_public_ib_unlearned.pt
  checkpoints/server_ib_naive_repair.pt
  checkpoints/server_ib_conflict_aware_repair.pt
  metrics/per_class.csv
  metrics/per_client.csv
  metrics/support_by_class.csv
  metrics/summary.json
```
