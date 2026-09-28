# Strict target-specific latent mask 검증 (seed 0)

## 목적

현재 선택적 server repair는 공개 데이터에서 `U`와 `G-u`의 차이가 큰 latent
coordinate를 global top-k로 선택한다. 이 coordinate가 정말 target-client-specific
semantic을 담는지 검증하기 위해, 정답 semantic이 명확한 strict split을 만들었다.

## strict split

공개 calibration data로 CIFAR-10 train에서 class당 100장씩 먼저 분리했다. 남은
49,000장은 다음과 같이 배정했다.

```text
client 0: class 0의 4,900장 전체
client 1: class 1의 4,900장 전체
...
client 9: class 9의 4,900장 전체
```

target은 client 0이다. 따라서 class 0은 data allocation 기준의 명확한
target-specific semantic이고, class 1~9는 target client가 보지 않은 retained-only
semantic이다.

## 설정

| 항목 | 값 |
| --- | --- |
| 데이터 | CIFAR-10 |
| client 수 | 10 |
| client당 train data | 4,900장 |
| 공개 calibration data | train class당 100장, 총 1,000장 |
| FL | origin과 client 0 제외 retrain 각각 200 rounds, local epoch 1 |
| codec | latent 64, base channels 32 |
| channel | AWGN, train SNR 10 dB |
| target-specific mask | 공개 data에서 `U`-`G-u` 평균 차이 상위 16 / 64 차원 |
| 평가 data | mask 선택에 쓰지 않은 CIFAR-10 test 10,000장 |
| random baseline | 동일 크기 random mask 200개 |

실행 정보:

```text
config: src/configs/federated_cifar10_target_exclusive_mask_specificity.json
output: outputs/target_exclusive_mask_specificity/20260825T054614Z_seed-0/
```

## 평가 방법

각 test sample에서 다음 score를 계산했다.

```text
mask score(x)
= 선택된 latent 차원에서의 평균 |normalize(mu_U(x)) - normalize(mu_G-u(x))|
```

class 0 sample을 positive, class 1~9 sample을 negative로 두고 AUC를 계산했다. AUC가
높을수록 mask score가 known target-specific class 0을 잘 구분한다.

## 결과

| 지표 | 값 |
| --- | ---: |
| selected-mask AUC | 0.5173 |
| 전체 latent score AUC | 0.6251 |
| random-mask AUC 평균 | 0.6026 |
| random-mask AUC 표준편차 | 0.0579 |
| random-mask AUC 최대 | 0.7452 |
| random mask가 selected AUC 이상일 permutation 비율 | 0.9154 |

class별 selected mask score는 다음과 같다.

| class | score |
| --- | ---: |
| 0, strict target-specific | 0.03280 |
| 1 | 0.03178 |
| 2 | 0.03394 |
| 3 | 0.03208 |
| 4 | 0.03382 |
| 5 | 0.03176 |
| 6 | 0.03171 |
| 7 | 0.03208 |
| 8 | 0.03378 |
| 9 | 0.03238 |

## 해석

이 결과는 현재 global top-k mask 규칙의 실패를 보여 준다.

- strict target-specific인 class 0의 AUC가 0.5173으로 무작위 수준에 가깝다.
- 동일 크기 random mask 평균 AUC가 오히려 0.6026이다.
- selected mask가 class 0만이 아니라 모든 class에서 `U`-`G-u` 차이의 큰 비중을
  차지한다.

따라서 공개 data 전체에서 큰 `U`-`G-u` 차이를 보이는 latent coordinate는
target-specific semantic이라기보다 target local model과 remaining reference의 전역적인
model drift를 잡고 있는 것으로 해석하는 편이 타당하다.

이 결과만으로 multi-seed 일반화를 주장하지는 않는다. 그러나 제안 방법의 핵심
target-specific mask로 이 규칙을 계속 사용하기에는 충분히 강한 반증이다. 다음 단계는
mask 비율 ablation이 아니라 input-dependent 또는 class-conditional specificity 기준을
새로 설계하고, 같은 strict split에서 다시 검증하는 것이다.
