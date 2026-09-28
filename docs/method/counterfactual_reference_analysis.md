# Counterfactual Reference 기반 Client-Level Unlearning: 방법론 분석

## 문서 목적과 상태

이 문서는 reconstruction-oriented federated semantic communication에서
client-level unlearning을 위한 **counterfactual reference** 아이디어를
객관적으로 검토한다. 확정된 시스템 목표, 유망한 설계 원리, 아직 검증되지
않은 가설을 구분한다. 이 문서는 최종 알고리즘 명세가 아니다.

핵심 목표는 target client `u`의 입력을 일부러 잘못 복원시키는 것이 아니라,
target client 없이 처음부터 학습한 codec과 기능적으로 가까워지는 것이다.

```text
M^U ~= M^{-u}_retrain
```

따라서 target-client sample의 MSE 증가는 보조 진단일 뿐, 그 자체가
unlearning 성공을 뜻하지 않는다. `M^{-u}_retrain`도 remaining clients가
공유하는 일반적 semantic knowledge를 통해 해당 sample을 잘 복원할 수 있다.

## 출발점: SCU와의 차이

SCU는 erased sample과 latent/reconstruction 사이의 mutual information을
줄이고, erased data를 negative, remaining data를 positive로 둔 contrastive
compensation을 사용한다. 직접적인 MI 감소가 기존 지식까지 잃게 할 수 있음을
인정하고 original model과의 KL 제약을 둔다.

SCU의 backdoor 제거와 erased-set MSE 증가는 target-only backdoor처럼
retraining oracle도 유지하기 어려운 신호에는 유효한 진단이다. 그러나 보통의
client deletion에서는 erased MSE가 커야 한다는 평가만으로 retrain-equivalence를
입증할 수 없다. SCU의 VIB consistency 분석도 IID 및 erased/remaining data의
조건부 독립 가정 아래의 upper-bound 분석이다.

본 연구의 방향은 다음과 같이 다르다.

```text
representation destruction
    -> remaining-supported counterfactual repair
    -> retrain-equivalent codec behavior
```

## Client별 latent distribution과 responsibility

Global VIB encoder의 aggregate posterior를 client `k`에 대해 다음처럼 둔다.

```math
q_k(z) = q_{theta^G}(z | C=k)
       = (1 / |D_k|) sum_{x in D_k} q_{theta^G}(z | x).
```

FL 학습에서의 client prior를 `pi_k`라고 하면, `pi_k`는 통상 sample-weighted
aggregation weight와 일치해야 한다. Remaining distribution은 균등 client 평균이
아니라 다음의 가중 mixture여야 한다.

```math
q_{-u}(z) = sum_{k != u} [pi_k / (1-pi_u)] q_k(z).
```

그때 semantic responsibility는

```math
rho_u(z) = P(C=u | Z=z)
         = pi_u q_u(z) /
           [pi_u q_u(z) + (1-pi_u) q_{-u}(z)]
```

로 해석할 수 있다. 이는 parameter가 target client에 의해 만들어진 정확한 비율이나
인과적 ownership이 아니라, **original global encoder의 latent space에서 target
client가 차지하는 상대적인 density-ratio proxy**다.

Client `u`의 class 또는 data-distribution 비율이 다른 것은 client contribution의
일부로 취급해도 된다. 다만 `rho_u`가 높다는 것은 remaining client가 그 semantic을
전혀 갖지 않는다는 뜻이 아니다. 상대적 dominance와 remaining support의 절대 크기를
분리해야 한다.

## 현재 아이디어의 강점

Counterfactual reference를 만들 때 세 역할을 분리한 점은 매우 유망하다.

| 신호 | 역할 |
| --- | --- |
| Shared-semantic KNN | 어떤 remaining semantic neighborhood를 참조할지 결정 |
| Local density | 그 neighborhood에서 remaining clients가 실제 사용한 posterior를 표현 |
| Responsibility | original target posterior에서 reference 쪽으로 얼마나 이동할지 결정 |

개념적으로 target sample `x_u`의 shared-semantic neighbor set을 찾고, 선택된
neighbor posterior를 거리 기반 가중치 `a_i`로 섞어 local reference를 만든다.

```math
q_ref(z | x_u) = sum_{i in N_K(x_u)} a_i q_{theta^G}(z | x_i),

a_i = exp(-d_i / tau) / sum_j exp(-d_j / tau).
```

이는 gradient ascent로 target representation을 random/noise prior로 보내는 대신,
remaining clients가 지원하는 latent manifold 안에서 over-forgetting을 복원하려는
발상이다.

## 핵심 한계 1: counterfactual 식별 문제

다음 근사는 아직 증명되지 않은 가설이다.

```math
q_ref(z | x_u) ~= q_{theta^{-u}}(z | x_u).
```

`q_ref`의 neighbor와 posterior는 모두 target client를 포함해 학습한 original global
encoder에서 생성된다. 따라서 global encoder의 target influence가 이미 KNN metric과
reference posterior에 들어가 있다. KNN은 "remaining data 중 현재 표현에서 가까운
sample"을 찾을 뿐, leave-one-client-out encoder가 `x_u`에 만들 posterior를 자동으로
식별하지 않는다.

더 근본적으로 remaining data에 전혀 없는 target-only semantic 영역에서는,
`M^G`, `D_u`, 그리고 remaining summary만으로 `M^{-u}_retrain(x_u)`를 정확히 알 수
없다. 이는 특정 알고리즘의 부족보다 counterfactual information 자체의 한계에 가깝다.

따라서 `q_ref`는 정확한 oracle estimator가 아니라 다음으로 표현해야 안전하다.

> Remaining-supported counterfactual anchor: overlap이 확인되는 영역에서만
> leave-one-client-out behavior를 근사하도록 사용하는 local reference.

## 핵심 한계 2: high responsibility와 low support의 역설

현재의 단순 interpolation은

```math
mu_target = (1-rho_u) mu_u + rho_u mu_ref
```

와 같이 쓸 수 있다. 그러나 `rho_u`가 큰 위치는 흔히 remaining support가 가장 약한
위치다. 이 경우 reference가 신뢰할 수 없는데도 `mu_ref`로 가장 강하게 이동시키면,
임의의 nearest neighbor로의 projection과 distribution hole을 만들 수 있다.

따라서 responsibility 외에 remaining coverage 또는 reference confidence `c_{-u}`가
필요하다.

```math
alpha(x_u) = rho_u(x_u) c_{-u}(x_u).
```

`c_{-u}`는 다음을 함께 반영해야 한다.

- `q_{-u}`의 absolute local support
- shared KNN 거리
- 선택된 neighbor posterior들의 일관성 또는 local density concentration
- AWGN/Rayleigh 및 SNR 변화 아래 neighbor 관계의 안정성

그 결과 동작 원칙은 다음과 같다.

| 상황 | 권장 처리 |
| --- | --- |
| high responsibility + high coverage | reference alignment를 강하게 적용 |
| low responsibility | original representation을 대체로 보존 |
| high responsibility + low coverage | arbitrary KNN alignment를 금지; 별도 deletion signal에 의존 |

마지막 경우의 target contribution 제거는 representation reference만으로 해결되지
않는다. target client의 historical update, 마지막 local update, Fisher/saliency 같은
parameter-level influence signal이 필요하다.

## 핵심 한계 3: dimension-wise sharedness

`s_{u,j}=1-rho_{u,j}`처럼 latent dimension별 sharedness를 정의하는 것은 초기
baseline으로는 가능하지만 최종 방법으로는 부족하다.

- semantic information은 여러 latent dimensions의 결합에 존재할 수 있다.
- latent coordinate는 회전과 재파라미터화에 민감하다.
- sample posterior가 diagonal Gaussian이어도 aggregate posterior에는 dimension 간
  의존성이 생긴다.
- client를 잘 구분하지 못하는 방향이 곧 shared semantic이라는 보장은 없다.

또한 `z_u` 전체가 original global encoder에서 생성되므로, 그 일부만 masking해도
target influence로부터 완전히 독립적인 KNN query가 되는 것은 아니다.

따라서 subspace 정의보다 먼저, KNN을 위한 semantic matching representation을
정해야 한다. 가능한 선택지는 다음과 같다.

1. unlearning 대상 codec과 독립된 fixed self-supervised/perceptual feature `h(x)`
2. deletion 후 remaining clients만으로 만든 auxiliary matching representation
3. original latent를 쓰되, target-contamination을 포함한 empirical proxy임을 명시

첫 번째가 가장 논리적으로 깔끔하지만 외부 pretrained model 및 domain mismatch라는
추가 가정을 낳는다. 두 번째는 더 엄밀하지만 추가 FL 비용이 크다. 초기 실험에서는
세 번째를 baseline으로 두고, independent feature query와 비교하는 구성이 적절하다.

실용적인 subspace baseline은 posterior feature `(mu, log sigma^2)`로 target-vs-remaining
client discriminator를 학습하고, 그 client-discriminative low-rank subspace의 보완공간을
KNN metric에 사용하는 것이다. 이 subspace 역시 true shared semantics가 아니라
검증 대상인 proxy로만 해석한다.

## 핵심 한계 4: local density와 alignment loss

선택된 neighbor의 전체 posterior를 사용하는 발상은 shared 부분만 평균내는 것보다
좋다. 다만 Gaussian mixture `q_ref`는 다봉일 수 있다.

- mixture를 moment-matched 단일 Gaussian으로 바꾸면 어떤 remaining client도 보지
  않은 중간 latent를 만들 수 있다.
- `KL(q_{theta^U} || q_ref)`는 하나의 mode를 고르는 방향으로 작동할 수 있다.
- `KL(q_ref || q_{theta^U})`는 단일 posterior를 과도하게 넓히는 방향으로 작동할 수 있다.

따라서 mixture, moment-matched Gaussian, single-nearest prototype을 모두 명시적
ablation으로 두어야 한다. `q_target` 역시 oracle posterior가 아니라 transport 또는
regularization target으로 표현해야 한다.

## Encoder reference만으로는 충분하지 않음

Target inputs에서 encoder posterior를 reference와 align해도 decoder parameter에는
target client가 학습시킨 reconstruction mapping이 남을 수 있다. 따라서 이 아이디어는
joint codec unlearning의 encoder 구성요소이지, 독립적인 complete unlearning method가
아니다.

더 일관된 전체 구조는 다음과 같다.

```text
1. Target influence removal
   - target client의 active last update 또는 historical influence로
     encoder와 decoder 양쪽의 target-sensitive update를 계산

2. Remaining-supported representation repair
   - high-coverage 영역에서만 responsibility/confidence-gated reference alignment
   - arbitrary target-only KNN transport는 허용하지 않음

3. Distributed codec repair
   - remaining clients가 local retained data로 reconstruction, VIB,
     encoder-decoder consistency, channel-robustness를 회복

4. Conflict-aware aggregation
   - target-exclusive contribution의 재유입만 막고,
     shared manifold에서 over-forgetting을 회복하는 update는 허용
```

이 관점에서 conflict-aware aggregation은 forgetting update 전체를 고정하는 장치가
아니라, target-exclusive subspace에서 repair가 target contribution을 되살리는 경우만
제어해야 한다.

## Federated privacy와 통신 제약

Raw data를 서버로 보내지 않는다고 해서 per-sample latent posterior나 KNN 결과의
privacy 문제가 사라지지 않는다. 개별 posterior는 client identity 또는 sample semantics를
노출할 수 있다.

후속 방법 명세에는 다음 중 하나가 포함되어야 한다.

- client별 prototype 또는 GMM sufficient statistic만 공유
- 각 remaining client가 query에 대한 local KNN score를 계산하고, 가중 posterior 통계만
  반환
- secure aggregation 또는 differential privacy 적용

이 통신 설계는 distributed repair의 부수 구현이 아니라, "raw-data centralization 없음"이라는
연구 주장을 실제로 뒷받침하는 구성요소다.

## 검증 가능한 가설과 필수 평가

이 아이디어의 핵심 가설은 다음처럼 검증 가능하게 써야 한다.

> Remaining support가 충분한 영역에서 responsibility/confidence-gated reference
> repair는 destructive forgetting 또는 global-latent KNN보다 retrained-without-client
> codec과의 functional gap을 줄인다.

평가는 같은 입력과 같은 channel realization을 사용해 retraining oracle과 비교한다.

```math
G(A) = E_{x in A, H} [ || M^U(x; H) - M^{-u}_retrain(x; H) ||^2 ].
```

`A`는 최소한 다음 strata로 분리한다.

- retained-client test set
- target-client audit set
- high-coverage target neighborhood
- low-coverage 또는 target-only semantic region
- IID/non-IID split, AWGN/Rayleigh, train/test SNR mismatch

특히 다음 ablation이 필요하다.

- global-latent KNN 대 shared-subspace KNN 대 independent-feature KNN
- responsibility만 사용 대 responsibility + coverage confidence
- Gaussian mixture 대 moment-matched Gaussian 대 single prototype
- encoder reference alignment 없음 대 있음
- naive FedAvg repair 대 target-exclusive 재유입만 막는 conflict-aware repair

Target MSE, backdoor/client-marker 제거, membership inference, latent client-identity
leakage는 모두 보조 unlearning 지표로 유지한다. 그러나 주된 성공 기준은 각 stratum에서의
functional retraining gap과 retained utility다.

## 현재 권장 주장

이 아이디어가 지원하는 가장 강한 주장은 다음이다.

> Client density-ratio responsibility와 remaining-support confidence를 이용해,
> target influence 제거가 만든 over-forgetting을 remaining-client manifold 안에서만
> 복원하는 distributed repair를 설계한다.

다음 주장은 피해야 한다.

> KNN local density가 leave-one-client-out encoder posterior를 정확히 추정한다.

후자는 일반적으로 식별할 수 없다. 반면 전자는 full retraining oracle과의 strata별
functional comparison으로 경험적으로 반증·검증할 수 있으며, client-level federated
reconstruction unlearning의 목표와도 일치한다.
