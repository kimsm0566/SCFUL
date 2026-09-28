# Federated SCU with Public-data Compensation

## 목적과 범위

이 문서는 SCU를 연합 학습 환경에 옮기는 첫 번째 기준선
`FSCU`를 정의한다. 이 repository에서 FSCU는 항상 public-data compensation
버전을 뜻한다.

```text
target client local joint-unlearning step
    -> server public-data compensation step
    -> 두 단계를 정해진 cycle만큼 교대로 반복
    -> final unlearned reconstruction codec
```

대상은 특정 client `u`의 전체 local dataset `D_u`를 삭제하는
client-level unlearning이다. 각 client와 server는 모두 full codec
`G = (f_theta, g_phi)`의 복제본을 갖고, 통신/재구성 경로는 다음과 같다.

```text
x -> f_theta -> z -> H(z) -> z' -> g_phi -> x_hat
```

최종 목표는 target data에서 재구성을 임의로 나쁘게 만드는 것이 아니라,
target client 없이 처음부터 연합 재학습한 codec `G_retrain^{-u}`와 기능적으로
가까워지는 것이다.

```text
G_FSCU ~= G_retrain^{-u}
```

이 방법은 최종 proposed method의 distributed retain repair를 대체하지 않는다.
우선 SCU의 joint forgetting과 compensation을 raw remaining-client data 없이
분리할 수 있는지 확인하는 public-data 기준선이다.

## 가정과 위협 범위

- 삭제 요청 직후 client `u`는 local data가 영구 삭제되기 전에 여러 번의 assisted
  unlearning step에 참여할 수 있다. 이를 만족하지 못하면 refresh variant는 실행할
  수 없다.
- server는 target 또는 retained clients의 raw data를 받지 않는다.
- server는 client training split과 겹치지 않는 public calibration data `D_pub`를
  보유한다.
- client `u`는 full encoder와 decoder를 local에서 함께 업데이트할 수 있다.
- server는 global codec update와 erased latent buffer만 받는다.

`Z_e`는 raw sample은 아니지만 개인 정보가 남을 수 있는 representation이다.
그러므로 이 설계는 **raw-data-free server unlearning**이지, representation
privacy 또는 cryptographic privacy를 보장하는 설계가 아니다. Server는 compensation
종료 후 `Z_e` buffer를 삭제하고, buffer 자체에 대한 reconstruction/inversion 또는
membership leakage를 별도 진단해야 한다.

## SCU와의 차이

원래 SCU는 중앙에서 erased data `D_e`와 remaining data `D_r`를 모두 보며,
`D_e`를 contrastive negative, `D_r`를 positive로 사용한다. 이는 server가
`D_r`에 접근할 수 없는 FL 제약에 맞지 않는다.

`FSCU`는 다음의 대체를 사용한다.

```text
SCU:            D_e representation = negative, D_r representation = positive
FSCU:           erased buffer B_e = negative, public representation = positive
```

따라서 `D_pub`가 retained-client distribution을 충분히 대표하는지는 가정이 아니라
검증할 연구 가설이다. `D_pub`가 domain 또는 semantic support를 충분히 덮지 못하면
public compensation은 retained utility를 복구하지 못할 수 있다.

## 단계 1: global model 준비

삭제 요청 시 server는 현재 global codec을 고정 teacher와 update 대상의 공통
시작점으로 복제한다.

```text
G_fix <- stop_gradient(G)
G_u   <- G
```

Server는 `G_fix`와 `G_u`를 target client `u`에 보낸다. 실제 통신에서는
`G_u` 전체 model 또는 `Delta_JU = G_JU - G` 중 하나만 반환한다. 둘을 함께
보내는 것은 중복이다.

## 단계 2: target client의 local joint unlearning

Target client는 자신의 전체 local data를 erased set으로 둔다.

```text
D_e = D_u
```

매 batch `x_e`에 대해 current codec과 frozen codec의 posterior를 계산하고,
channel `H`를 거친 reconstruction을 사용해 encoder와 decoder를 동시에
업데이트한다. 구현 가능한 SCU-style surrogate는 다음과 같다.

```math
L_JU =
  lambda_KL KL(q_{G_u}(z | x_e) || q_{G_fix}(z | x_e))
  - lambda_forget d(x_e, g_phi(H(f_theta(x_e))))
```

첫 항은 original codec에서의 과도한 drift를 제어하고, 두 번째 항은 target의
reconstruction mapping을 약화한다. 이는 논문의 mutual-information 목적을
VIB posterior KL 및 reconstruction surrogate로 구현한 형태다. 두 항 모두의
gradient가 encoder와 decoder로 전달되어야 한다.

FSCU에서 contrastive 학습과 buffer에 사용하는 canonical representation은 하나로
고정한다.

```text
Z_canonical(x) = power-normalized posterior mean
               = sqrt(d) * mu(x) / (norm(mu(x)) + epsilon)
```

Variational DeepJSCC에서 이는 deterministic하며 AWGN을 더하기 직전 channel input과
동일하다. Buffer, contrastive loss와 representation 진단은 모두 이 정의를 사용한다.
Stochastic posterior sample을 positive로 쓰는 기존 방식은 명시적인 ablation으로만
남긴다.

Target client는 각 joint-unlearning step 뒤 필요한 경우 local model과 erased
representation buffer를 만든다. Buffer sample은 local 저장 순서의 앞부분을 사용하지
않고 기본적으로 label-stratified sampling한다.

```text
B_e = stop_gradient(Pool({Z_canonical(x) : x in D_e}))
```

Negative buffer는 다음 세 방식을 비교한다.

- `original`: 삭제 전 original codec에서 한 번 만들고 고정
- `post_ju_static`: 첫 JU step 직후 한 번 만들고 고정
- `refresh`: 매 JU step 직후 current codec으로 다시 만들고 전송

기본값은 `refresh`다. 이는 learner 좌표계 변화에 따른 stale negative를 줄이지만
target client의 반복 참여와 추가 통신량이 필요하다. 원본 입력 `D_e`, class label,
sample identifier `i`는 server에 보내지 않는다.
원본 입력 `D_e`, class label, sample identifier `i`는 server에 보내지 않는다.

Target client가 server에 보내는 것은 다음뿐이다.

```text
1. G_JU 또는 Delta_JU
2. B_e (detached erased representation buffer)
```

`z`는 별도로 학습 데이터처럼 보관해야 하는 값이 아니다. `x`가 주어졌을 때
encoder가 매 forward pass에서 생성한다. 논문 표기의 `i`는 sample identity/index를
의미하며, 별도 input tensor나 class label을 요구하지 않는다.

## 단계 3: server-side public contrastive compensation

Server는 각 cycle의 `G_JU`를 초기값으로 사용하고, 선택한 buffer 전략의 `B_e`를
negative로 둔다. Server는 `D_pub`만 사용해 encoder와 decoder를 함께 보상한 뒤,
그 결과를 다음 target JU step의 시작 model로 돌려보낸다.

### Positive pair

`Z_pub` 하나만으로는 contrastive positive pair가 되지 않는다. 기본 방식은 같은
public image에 seed가 기록되는 random crop과 horizontal flip 두 개를 적용한 뒤,
각 view의 deterministic canonical representation을 positive pair로 사용한다.

```math
z_i^(1) = f_theta(a_1(x_i^pub)),
z_i^(2) = f_theta(a_2(x_i^pub))
```

Posterior를 두 번 sampling하는 `stochastic_self` 방식은 비교군으로 유지한다. 이
방식은 posterior variance 감소로 positive similarity를 높일 수 있으므로 CC 전후
mean log-variance, posterior standard deviation, Gaussian entropy와 prior KL을
반드시 기록한다.

```math
L_con = - 1/B sum_i log(
  exp(sim(z_i^(1), z_i^(2)) / tau)
  /
  (exp(sim(z_i^(1), z_i^(2)) / tau)
   + sum_{b in B_e} exp(sim(z_i^(1), b) / tau))
)
```

`B_e`에는 gradient를 흘리지 않는다. 즉 server가 업데이트하는 것은 public branch를
통과한 `f_theta`, `g_phi`뿐이며, erased client data를 역전송하거나 재사용하지 않는다.

### Decoder compensation

Public input은 server가 보유하므로 channel-aware reconstruction loss를 직접
계산할 수 있다.

```math
L_rec^pub = E_{x in D_pub}[d(x, g_phi(H(f_theta(x))))]
```

초기 server compensation objective는 다음과 같다. Prior KL은 기본 weight 0인
안정화 ablation으로 제공한다.

```math
L_CC^pub = lambda_con L_con + lambda_dec L_rec^pub + lambda_prior L_KL^pub
```

`L_con`은 deterministic power-normalized posterior mean에서 계산하고, `L_rec^pub`는
sampled latent와 학습 channel `H`를 거친 signal에서 계산한다. 두 항이 사용하는
representation 정의는 artifact에 함께 기록한다. AWGN을 기본으로 시작하되 Rayleigh,
SNR, train/test SNR mismatch는 후속 평가 축으로 유지한다.

## Interleaved optimization

기존 sequential 구현은 전체 JU epoch를 먼저 실행해 codec을 크게 손상시킨 뒤 전체
CC epoch로 복구했다. 기본 FSCU는 같은 총 epoch budget을 여러 cycle로 나누어 다음
순서로 실행한다.

```text
cycle 1: target JU -> buffer refresh -> server public CC
cycle 2: target JU -> buffer refresh -> server public CC
...
cycle C: target JU -> buffer refresh -> server public CC
```

예를 들어 JU 5 epoch, CC 40 epoch, cycle 5이면 각 cycle은 JU 1 epoch와 CC 8 epoch를
실행한다. Target와 server의 Adam state는 기본적으로 각 역할 안에서 cycle 사이에
유지한다. `interleaving_cycles=1`은 이전 sequential 기준선이다.

JU 안정화 ablation으로 erased semantic CE ascent의 on/off와 decoder preservation
weight를 제공한다. Decoder preservation은 current canonical target latent에 동일한
AWGN realization을 적용한 뒤 current decoder output이 frozen original decoder
output에서 과도하게 벗어나지 않도록 제한한다. 이는 paper Eq. 4의 정확한 확률적
decoder KL이라고 주장하지 않는다.

Server가 public input에 original global model distillation을 추가하는 것은 이
baseline의 기본값으로 두지 않는다. Original model의 target contribution을 다시
주입할 위험이 있고, 이 프로젝트의 distillation 규칙은 retained-client data에서만
사용하는 것을 원칙으로 한다.

## 메시지 흐름

```text
Server                                  Target client u
------                                  ---------------
G_current, G_fix -------------------->  local D_e로 짧은 JU step
G_JU, optional refreshed B_e <--------  raw D_e는 전송하지 않음
G_JU + D_pub + B_e
  -> 짧은 public CC step
G_CC -------------------------------->  다음 JU step
...                                     정해진 cycle 반복
G_FSCU 확정 -> B_e 폐기                 local D_e 영구 삭제
```

통신량은 cycle별 server-to-target codec, target-to-server codec update와 latent
buffer 전송을 모두 기록한다. Refresh 방식은 buffer가 매 cycle 전송된다. 모든
`D_e` sample의 representation을 보내는
설정은 통신 및 leakage 관점에서 oracle-like upper variant로 분리해 보고한다.

## 실험 가설과 비교

주 가설은 다음과 같다.

> Can public semantic representations replace remaining-client private data in
> SCU-style compensation while preserving reconstruction utility after
> client-level unlearning?

최소 비교군은 다음과 같다.

- Original federated global codec
- Federated retraining without target client (gold standard)
- Client-side joint unlearning only
- `FSCU`: client-side JU + server public CC
- Centralized full SCU with pooled `D_e` and `D_r` (raw-data oracle comparison,
  허용되는 실험에서만)
- 이후 full method: JU + distributed retain repair + conflict-aware aggregation

평가는 gold standard와의 functional gap을 중심으로 수행한다.

- retained-client 및 global test reconstruction: MSE, PSNR, MS-SSIM, LPIPS
- target client 및 client-marker/backdoor diagnostic
- retrained-without-client codec과의 output/latent gap
- downstream task performance
- unlearning time, model-update bytes, `B_e` bytes
- `B_e` representation leakage: membership inference 및 가능하면 inversion/reconstruction attack
- `D_pub` 크기와 domain/support mismatch ablation

## 해석 시 주의점

- Target client가 assisted round에 참여하지 못하면 이 방법은 그대로 실행할 수 없다.
- `B_e`를 보낸다는 사실만으로 privacy-preserving이라고 부르면 안 된다.
- `D_pub` 기반 utility 복구는 remaining clients의 실제 retained utility를 보장하지 않는다.
  반드시 `G_retrain^{-u}`와 remaining-client test metrics로 검증한다.
- SCU의 retraining-equivalence 분석은 IID 가정을 사용한다. 2-class와 Dirichlet 0.1
  환경에서 FSCU의 retrain equivalence는 이론적 보장이 아니라 실험 목표다.
- Client-level 삭제의 성공을 target MSE 증가만으로 판단하지 않는다. 다른 clients가
  유사한 semantic information을 보유한 경우 gold standard도 target-like input을
  재구성할 수 있다.
