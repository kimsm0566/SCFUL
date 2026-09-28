# 선택적 server repair: `G-un` 시작점 실험 결과 (seed 0)

## 목적

이 실험은 이전의 공개 데이터 기반 direct matching 구현을 바로잡기 위한 것이다.

이전 구현은 server candidate를 `G-u`에서 시작한 뒤 전체 representation을 `G-un`에
가깝게 만들었다. 이 방식은 target client의 local forgetting으로 생긴 손상까지
shared semantic 방향으로 복사할 수 있다.

이번 구현은 다음 의도를 따른다.

```text
1. U와 G-u의 차이로 target-specific latent 영역을 찾는다.
2. server candidate는 G-un에서 시작한다.
3. target-specific 영역은 G-un에 가깝게 유지한다.
4. 나머지 영역만 G-u에 가깝게 복구한다.
5. 그 뒤 retained client가 local data만으로 distributed repair를 수행한다.
```

여기서 `G-u`는 target client를 제외한 마지막 FL round의 local model을 평균낸
remaining-client reference다. 처음부터 target client 없이 200 round를 다시 학습한
`retrain_without_client_0`와는 다르며, 후자만 평가 기준이다.

## 설정

| 항목 | 값 |
| --- | --- |
| 데이터 | CIFAR-10 train 50,000장 |
| 공개 데이터 `D_pub` | client split 전에 class당 100장씩 분리한 1,000장 |
| FL client 수 | 10 |
| client당 학습 수 | 4,900장, 모두 같은 수 |
| Non-IID 분할 | Dirichlet alpha 0.1, seed 0 |
| target client | client 0 |
| codec | latent 64, base channels 32, encoder와 decoder 모두 학습 |
| channel | AWGN, train SNR 10 dB |
| 원본 FL / 재학습 | 각각 200 rounds, local epoch 1 |
| target local unlearning | PGA 20 epochs, learning rate 0.0001, momentum 0.9 |
| PGA projection | `G-u` 거리의 2배 반경 |
| 선택적 server repair | 20 epochs, learning rate 0.0001 |
| target-specific latent mask | `U`와 `G-u` 차이가 큰 상위 16 / 64 차원 (25%) |
| distributed repair | remaining 9 clients, 20 rounds, local epoch 1 |

`G-un` 생성에는 현재 PGA를 사용했다. 이는 target 삭제 신호를 만드는 임시 local
unlearning 방법이며, 제안 방법의 최종 forgetting 규칙으로 확정한 것은 아니다.

실행 설정과 원시 산출물은 다음에 있다.

```text
config:  src/configs/federated_cifar10_dirichlet_01_pga_selective_repair.json
output:  outputs/pga_selective_repair/20260819T064852Z_seed-0/
```

## server selective repair가 실제로 한 일

server는 공개 데이터에서 각 latent coordinate마다 다음 값을 계산했다.

```text
평균 절대 차이 = 평균 |normalize(mu_U) - normalize(mu_G-u)|
```

상위 16개 coordinate를 target-specific 보호 영역으로 선택했다.

```text
[0, 4, 12, 21, 29, 30, 32, 36, 38, 40, 41, 46, 56, 58, 61, 62]
```

이 영역에서는 candidate representation과 reconstruction이 `G-un`에서 크게
벗어나지 않게 했다. 나머지 48개 latent coordinate에서는 candidate를 `G-u` 방향으로
복구했다. reconstruction에서는 각 공개 sample의 선택 영역 차이가 클수록 `G-un`
보호 비중을 높였다.

따라서 이 단계에는 `G-u` 전체를 `G-un`에 맞추는 loss도, candidate를 `U`에 직접
맞추는 loss도 없다.

## 전체 test 결과

아래 output gap은 각 model의 test reconstruction과 `retrain_without_client_0`의
reconstruction 사이 MSE다. 작을수록 client 0을 처음부터 제외하고 재학습한 model의
동작과 가깝다.

| model | test reconstruction MSE | output gap to retrain |
| --- | ---: | ---: |
| original | 0.013516 | 0.000229 |
| retrain without client 0 | 0.013516 | 0.000000 |
| `G-u` remaining reference | 0.013511 | 0.000228 |
| `U` target normal local model | 0.013657 | 0.000347 |
| `G-un` target PGA model | 0.072121 | 0.058370 |
| selective server repair | 0.029567 | 0.015894 |
| selective server repair + naive distributed repair | 0.013380 | 0.000245 |
| selective server repair + current conflict-aware repair | 0.013560 | 0.000316 |

## 해석

### 1. `G-un`의 손상은 매우 크다

PGA 직후 `G-un`의 MSE는 0.07212, output gap은 0.05837이다. 이는 target client
data만으로 reconstruction을 나쁘게 하는 update가 target과 retained semantic을
분리하지 못하고 codec 전체를 크게 손상시킨다는 뜻이다.

### 2. 선택적 server repair는 손상을 일부만 복구한다

선택적 server 단계는 output gap을 0.05837에서 0.01589로 낮췄다. 즉 `G-un`에서
출발했음에도 remaining reference를 이용해 큰 손상의 약 73%를 되돌렸다.

하지만 0.01589는 original과 retrain 사이의 0.000229보다 훨씬 크다. 따라서 공개
data 1,000장과 단순 top-k latent mask만으로 retrain-equivalent behavior에
도달했다고 말할 수 없다.

### 3. distributed repair 뒤에는 reconstruction utility가 회복된다

remaining 9 clients가 raw data를 server에 보내지 않고 local repair를 수행한 뒤,
naive aggregation model의 output gap은 0.000245가 됐다. 이는 original gap
0.000229와 가까운 수준이다.

다만 이 수치는 한 seed의 작은 차이다. 특히 naive result가 잘 나왔다고 해서 target
client 정보가 지워졌다고 결론 내리면 안 된다. 현재 평가는 test reconstruction과
functional gap만 포함하며 membership inference나 client-identity leakage를 아직
측정하지 않았다.

### 4. 현재 conflict-aware aggregation은 개선되지 않았다

현재 parameter-space projection을 적용한 model의 output gap은 0.000316으로 naive
repair보다 나빴다. 이 projection은 `G-un - original` 전체 방향을 forgetting 방향으로
간주한다. 선택적 방법의 핵심은 일부 representation만 보호하는 것이므로, 전체
parameter 방향 하나로 conflict를 정의하는 현재 구현은 철학과 맞지 않는다.

다음 단계에서는 공개 data에서 구한 target-specific mask 또는 representation
direction을 이용해 repair update의 충돌을 판정해야 한다.

## 이전 direct matching 실험과의 비교

08-19-01의 direct matching server 단계는 output gap 0.01194, 이후 naive repair는
0.000264였다. 이번 선택적 server 단계의 단독 gap 0.01589는 더 크다. 즉 이번 구현이
server 단계만 놓고 더 좋은 결과를 냈다는 증거는 없다.

그러나 이전 단계는 `G-u`에서 시작해 `G-un` 전체를 teacher로 사용한 방법이고, 이번
단계는 `G-un`의 target-specific 부분만 보존하면서 나머지를 복구하는 방법이다. 둘은
목적 자체가 다르다. 이번 naive distributed repair 결과 0.000245는 이전 naive 결과
0.000264보다 작지만, 차이가 0.000019에 불과하므로 여러 seed와 privacy metric 없이
우위를 주장하지 않는다.

## 현재 결론과 다음 실험

이번 실험으로 확인된 것은 다음이다.

- `G-un` 전체를 server teacher로 쓰면 안 된다는 문제의식은 타당하다.
- `G-un`에서 시작해 selected latent만 보호하고 나머지를 `G-u`로 복구하는 흐름은
  실제 MPI/FL 환경에서 동작한다.
- 단순 top-k coordinate mask는 PGA 손상을 크게 줄이지만, server 단계만으로는 충분하지
  않다.
- distributed repair의 효과와 deletion 품질을 함께 판정하려면 target member 기반
  privacy metric이 추가되어야 한다.

다음 우선순위는 다음과 같다.

1. 선택 mask 비율과 protection weight를 여러 seed에서 ablation한다.
2. target training member와 held-out probe를 분리하고 membership inference 또는
   client-identity leakage를 추가한다.
3. 전체 parameter projection 대신 target-specific representation 기준의
   conflict-aware repair aggregation을 설계한다.
