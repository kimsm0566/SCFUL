# System Model

## Federated Reconstruction Semantic Communication

There are `K` clients. Client `k` owns local dataset `D_k`. The full training data is:

```text
D = D_1 union D_2 union ... union D_K
```

Each client trains a reconstruction-oriented semantic communication codec:

```text
M = (f_theta, g_phi)
```

where:

- `f_theta`: joint semantic/channel encoder
- `g_phi`: joint semantic/channel decoder
- `H`: wireless channel, such as AWGN or Rayleigh fading

The transmission and reconstruction process is:

```text
z = f_theta(x)
z' = H(z)
x_hat = g_phi(z')
```

The reconstruction loss is:

```text
L_rec = d(x, x_hat)
```

where `d` can be MSE, MS-SSIM loss, or another reconstruction objective.

## Registered CIFAR-10 Codec

본 실험은 원본 DeepJSCC의 CIFAR-10 convolutional topology를 기본으로 사용한다.

```text
encoder:
  Conv 3->16, kernel 5, stride 2, PReLU
  Conv 16->32, kernel 5, stride 2, PReLU
  Conv 32->32, kernel 5, stride 1, PReLU
  Conv 32->32, kernel 5, stride 1, PReLU
  Conv 32->32, kernel 5, stride 1

variational split:
  first 16 channels  -> posterior mean map
  second 16 channels -> posterior raw-scale map
  standard deviation = softplus(raw scale) + 1e-6
  each map: 16 x 8 x 8 = 1,024 real dimensions

channel input:
  sampled latent -> per-sample power normalization -> H

decoder:
  Transposed Conv 16->32, kernel 5, stride 1, PReLU
  Transposed Conv 32->32, kernel 5, stride 1, PReLU
  Transposed Conv 32->32, kernel 5, stride 1, PReLU
  Transposed Conv 32->16, kernel 5, stride 2, PReLU
  Transposed Conv 16->3,  kernel 5, stride 2, Sigmoid
```

원본 deterministic DeepJSCC와의 핵심 차이는 마지막 encoder 출력이 단일 channel
code가 아니라 posterior mean과 smooth scale을 함께 낸다는 점이다. 등록 v3는
`softplus_std_v1`, initial log-variance `-7`, standard-deviation floor `1e-6`을 사용한다.
학습 중에는 reparameterization으로 latent를 sampling한다. Prior KL은 각 sample의
1,024 latent dimension을 합한 뒤 batch에서 평균하는 표준 VIB `batchmean_sum`으로
계산한다.

CIFAR-10 image의 source real values는 `3 x 32 x 32 = 3,072`다. Channel에는
`16 x 8 x 8 = 1,024` real values를 보내며, 이는 512 complex channel uses와
complex channel bandwidth ratio `1/6`에 해당한다. 이 네 값은 config의 top-level
`bandwidth` metadata에 명시한다.

등록 reconstruction 평가의 primary 경로는 공식 CIFAR-10 test set에서
`posterior sample -> power normalization -> AWGN 20 dB -> decoder`이며 10회 Monte
Carlo 평균을 사용한다. 같은 test image와 repetition에는 모든 model이 같은 posterior
noise와 AWGN realization을 받는다. `rmse_255`는 0--1 normalized per-pixel MSE의
제곱근에 255를 곱한 8-bit pixel-scale RMSE다. Posterior-mean 평가는 deterministic
보조 진단으로 분리하고, normalized MSE는 PSNR과 functional-gap 계산을 위한 내부
값으로 유지한다.

Semantic 평가는 reconstructed image를 입력받는 별도
`cifar10_groupnorm_silu_cnn` classifier로 수행할 수 있다. 이 model은 codec과
독립적으로 remaining clients의 clean raw image만 사용해 새로 연합학습하며 target
client는 학습과 validation에서 제외한다. Client 내부 validation CE로 선택한 뒤
freeze하여 모든 Original, Retrain, Unlearned codec에 같은 state를 적용한다.
Classifier의 CE/FedAvg/optimizer/checkpoint는 codec과 공유하지 않는다. Server는
client 이미지 대신 모델 업데이트와 validation 집계 수치만 받는다.

Clean reconstruction baseline과 등록 v3 codec 학습에서는 sample별 RGB pixel squared
error 합을 batch에서 평균한 per-example pixel SSE와, sample별 latent-dimension KL 합을
batch에서 평균한 prior KL을 최적화한다. 즉 두 term 모두 sample 기준 batch mean이며,
reconstruction objective는 headline 지표용 normalized per-pixel MSE가 아니다. Frozen
classifier는 외부 observer로만 사용하며 classifier CE를 reconstruction에 연결해
codec으로 역전파하지 않는다. 따라서 reconstruction-only config에는 classifier가 없고
classifier state는 FedAvg와 unlearning에 포함되지 않는다.

Codec 구현 이름은 `variational_deepjscc_cifar10_v3`이며 codec format version은 3이다.
별도 classifier는 고정 observer이므로 codec FL 통신량과 codec checkpoint 크기에
포함하지 않는다. Classifier 자체의 독립 FL 학습 비용은 별도로 기록한다.

이 architecture는 federated deletion, controlled CIFAR-10 comparison, 완료 run의
MIA audit에서 새 기본 codec이다. 모든 본 실험 config에는 architecture와 KL reduction을
명시적으로 기록한다. v2와 v3 checkpoint/config는 format version으로 구분하며
서로 호환되는 것으로 간주하지 않는다.

## Federated Training

Each client locally updates the full codec and sends model updates to the server. The server aggregates both encoder and decoder parameters:

```text
theta^{t+1} = sum_k p_k theta_k^t
phi^{t+1}   = sum_k p_k phi_k^t
```

## Client-Level Unlearning Request

When client `u` requests deletion, the target is to produce:

```text
M^U = U(M, u)
```

such that:

```text
M^U ~= M^{-u}_{retrain}
```

where `M^{-u}_{retrain}` is the codec retrained from scratch using only clients `k != u`.

## Important Constraint

The server should not access raw remaining-client data. This makes centralized compensation, as used in SCU, unsuitable for the federated setting.

## Current measurement boundary (2026-11-02)

현재 core suite는 PSNR/SSIM, retain FID, marker MPR/PUL, Runtime이다.
Clean/erased PSNR·SSIM과 FID를 Original, Retrain, Unlearned 순서로 모두 보고한다.
MPR은 절대 검출률이며 PUL과 같은 fixed-region detector를 사용한다. S1은 연속
projection/NCC도 비교한다. 별도 S1/S2 전역 template-bank 평가에서 outside 반응, global MPR,
pixel-union coverage와 raw-control FPR/TPR를 기록한다. 임의 형태 번짐까지 검출한다고
주장하지 않는다. dFIL은 보조 진단이다.
Local MPR v2는 raw clean 최대 반응 기준이며 Global은 raw clean 99% 분위수 기준이다.
Amplitude/NCC는 같은 held-out 500장의 실제 측정값으로 보고하며 기준모델 점수 환산은 하지 않는다.
dFIL은 `model(x, mode="posterior")`의 mean/logvar로 정의되는 diagonal Gaussian
관측에 대해 입력 Fisher trace를 입력 차원으로 나눈다. 입력 범위는 RGB [0,1]이다.
평균과 입력 의존 variance의 미분을 모두 포함하며, power normalization과 AWGN/Rayleigh
이전 관측에만 해당한다. Parameter Fisher, mutual information, membership과 혼동하지 않는다.

현재 `reconstruction_channel_audit.py`의 core output 평가는 posterior mean과 paired AWGN을
사용한다. 위의 sampled-posterior MC10 reconstruction-training 평가와 별도 프로토콜이며,
두 결과를 섞어 비교하지 않는다. Core protocol JSON에 경로와 SNR을 기록한다.
FID/PUL은 decoder 출력에서 측정하지만 encoder-channel-decoder 전체의 결과다.
실험 audit의 dataset 접근은 offline benchmark 평가용이며 distributed repair의 raw-data
접근 권한으로 해석하지 않는다.
세부 구현·한계·실행 상태는 [확정 지표 구현](../experiment/10-22-01-dfil-core-metrics.md)을 따른다.
