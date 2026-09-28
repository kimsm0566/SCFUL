# 통제된 Non-IID Oracle Behavior 실험 설계

## 목적과 범위

이 실험은 target input을 의도적으로 나쁘게 reconstruct하도록 만드는 것이 아니라, unlearned codec이 target client를 제외하고 처음부터 재학습한 oracle codec에 얼마나 가까운지를 측정한다.

```text
M^G       = A(D^t union D^r)
M^-t      = A(D^r)
M^U_SCU   = U_SCU-style(M^G, D^t, D^r)
```

주 성공 기준은 `M^U_SCU`와 `M^-t` 사이의 output/posterior functional gap이다. Target reconstruction MSE 증가와 client-marker 제거는 보조 진단이며, 그 자체를 forgetting 성공으로 해석하지 않는다.

이번 단계의 `A`는 pooled centralized training이다. 따라서 virtual client split은 통제된 client-deletion distribution shift를 정의한다. Client 1--4의 개별 분할은 이번 centralized optimization에는 영향을 주지 않지만, 동일 manifest를 후속 FedAvg 및 distributed repair 실험에서 재사용한다. `src/SCU/`는 수정하지 않으며, 새 구현 코드는 모두 `src/` 아래에 둔다.

이 experiment의 첫 사전 질문은 다음과 같이 명확히 정의한다.

> 각 client의 aggregated semantic posterior를 단일 diagonal Gaussian `N(mu_k, diag(sigma_k^2))`로 충분히 근사할 수 있는가?

여기서 충분하다는 것은 단순히 mean/variance를 계산할 수 있다는 뜻이 아니다. Held-out client sample의 empirical aggregated posterior와 moment-matched 단일 Gaussian 사이의 gap이, 같은 empirical posterior를 두 번 독립 sampling했을 때의 stochastic noise floor보다 충분히 작다는 뜻이다. 이 질문의 결과는 이후 support-aware method에서 client summary를 단일 `(mu_k, sigma_k^2)`로 보낼지, mixture/prototype/low-rank covariance를 사용할지 결정하는 근거가 된다.

## 통제된 5-client 분할

MNIST training set 전체를 중복 없이 5개의 logical client로 분할한다. `client_0`이 삭제 target이다.

| Target support group | `client_0` 구성 | 해석 |
| --- | --- | --- |
| `shared_high` | digit 3, 4, 5, 6에서 각각 seed 고정 1,500개 | 해당 semantic이 `D^r`에 많이 남아 있다. |
| `target_specific` | digit 1 training sample 전체 | `D^r`에는 digit-1 training sample이 하나도 없다. |

현재 MNIST label count에서는 `D^t`가 12,742개이며, `shared_high` 6,000개와 `target_specific` 6,742개로 구성된다. 남은 전체 sample은 `D^r`이다. Class별 seed 고정 shuffle 뒤, class마다 시작 client를 회전하는 연속 round-robin으로 `client_1`--`client_4`에 배분한다. 이 규칙은 retained client 전체 크기 차이를 최대 1개로 제한한다.

Manifest에는 dataset name/version과 transform, global/split/probe seed, 모든 client index array와 각각의 SHA-256 digest, support-group membership, client별 label/total count를 기록한다. 아래 assertion 결과도 명시한다.

- 전체 training index coverage
- client 간 mutual exclusivity
- `D^r`의 digit-1 count가 0
- retained client 크기 차이가 1 이하

## Audit set

Target sample에서의 member deletion behavior와 학습되지 않은 유사 input에 대한 generalization behavior를 분리하기 위해 두 target audit set을 함께 만든다.

| `probe_partition` | Data source | 목적 |
| --- | --- | --- |
| `target_train_member` | `D^t` | 실제 삭제 대상 training member에 대한 behavior |
| `target_heldout` | MNIST test set | target client가 없을 때의 normal generalization behavior |
| `retained_heldout` | MNIST test set, label 1 제외 | Retained utility |

두 target audit set에는 각각 다음을 포함한다.

- `shared_high`: digit 3--6에서 각 64개, 총 256개
- `target_specific`: digit 1에서 256개

Retained utility probe는 test-set label 1을 제외하고 stratified하게 선택한 2,000개다. CSV에는 index 충돌을 막기 위해 항상 `dataset_split` (`mnist_train` 또는 `mnist_test`)과 `source_index`를 함께 기록한다.

## Codec 및 channel 명세

Codec은 다음 end-to-end 구조를 따른다.

```text
x -> encoder -> q(z|x) -> power-normalized AWGN channel -> decoder -> x_hat
```

Encoder는 784 input에서 diagonal Gaussian posterior `(mu, logvar)`를 만들며 latent dimension은 기본 20이다. Encoder/decoder hidden width와 activation, `logvar` clamp 범위, power-normalization 위치, MSE reduction은 모두 config에 저장한다.

동일 SNR을 모든 model에 공정하게 적용하기 위해 training/evaluation에 같은 power normalization을 사용한다.

```text
z_bar = sqrt(d) * z / (||z||_2 + epsilon_power)
n ~ N(0, 10^(-SNR_dB / 10) I)
z_channel = z_bar + n
```

`d`는 latent dimension이다. Training에서 posterior sampling을 쓰면 `z = mu + sigma * eps`를 먼저 생성한 뒤 normalize한다. Deterministic evaluation은 `z = mu` 및 `z_channel = z_bar`를 쓴다. AWGN evaluation은 probe/repetition별 같은 noise `n`을 모든 model에 공통 적용한다. 따라서 standard-normal noise만 공유하면서 model별 유효 SNR이 달라지는 문제를 피한다.

## 학습 및 SCU comparator

각 global seed에서 initial state를 한 번 저장한다. `original`과 `retrain`은 이 byte-identical state에서 시작하며, SCU variant는 완료된 `original` checkpoint에서 시작한다.

| Model ID | 학습/갱신 | 목적 |
| --- | --- | --- |
| `original` (`M^G`) | `D^t union D^r`로 20 epoch | Full-data reference |
| `retrain` (`M^-t`) | `D^r`만으로 20 epoch | Epoch-matched retraining oracle |
| `retrain_replica` | `D^r`만으로 동일 초기화에서 data order/reparameterization/channel seed만 달리 20 epoch | Retraining stochasticity noise floor |
| `scu_forget_only` | `M^G`에서 target forgetting loss만 20 epoch | SCU damage-only ablation |
| `scu_self_sharing` (`M^U_SCU`) | `M^G`에서 SCU-style forgetting + retained VAE loss를 20 epoch | 주 centralized comparator |

`scu_self_sharing`은 legacy SCU의 완전 재현이 아니라 whole target client에 맞춘 **SCU-style centralized VAE adaptation**이다. Teacher `q_G`는 freeze된 original posterior이며 모든 teacher forward는 gradient 없이 수행한다.

```text
L_forget = lambda_KL * KL(q_U(z|x_t) || q_G(z|x_t))
           - lambda_rec * MSE(x_hat_t, x_t)

L_retain = beta * KL(q_U(z|x_r) || p(z)) + MSE(x_hat_r, x_r)

w_forget = |L_forget| / (|L_forget| + |L_retain| + epsilon)
w_retain = |L_retain| / (|L_forget| + |L_retain| + epsilon)

L_self-sharing = w_forget * L_forget
                 + w_retain * lambda_ss * L_retain
```

`scu_forget_only`는 정확히 `L_forget`만 최소화한다. 새 latent similarity, teacher self-consistency, 또는 다른 regularizer는 이번 comparator에 추가하지 않는다. 추가 시 별도 `SCU-style + <term>` ablation으로 기록해야 한다.

기본 config는 Adam 새 optimizer state, learning rate `1e-4`, batch size `128`, 20 epoch, VAE `beta=1e-4`, training AWGN SNR 10 dB, `lambda_KL=1`, `lambda_rec=0.1`, `lambda_ss=1`, dynamic-weight `epsilon=1e-8`이다. Config 및 run log에는 모든 coefficient, KL 방향, optimizer-state policy, epoch/step/example exposure를 기록한다.

SCU에서는 target batch 수만큼 optimizer update를 수행한다. Retained loader는 매 epoch shuffle된 without-replacement stream을 사용하고, target batch보다 먼저 소진되면 같은 epoch의 새 shuffled stream을 다시 시작한다. 두 SCU variant는 target/retained exposure와 optimizer step 수가 동일하다.

각 model에 대해 initial model, data-order sampler, posterior reparameterization, training-channel, evaluation-noise seed를 독립적으로 파생하여 저장한다. Epoch-matched original/retrain은 data size 차이로 optimizer step 수가 다르며, 이는 명시한다. 후속 sensitivity에서는 original과 같은 step 수를 쓰는 compute-matched retrain을 추가한다.

## Client semantic distribution 충분성 진단

이 진단은 완료된 `original` codec에서 5개 virtual client 각각에 대해 수행한다. 이는 centralized diagnostic이며 raw client data를 서버로 모아 쓰는 제안 방법이 아니다.

각 client `k`의 index를 seed 고정 80% fitting set과 20% evaluation set으로 나눈다. Fitting set의 per-sample posterior `(mu_i, logvar_i)`로 단일 diagonal Gaussian의 moment를 다음처럼 계산한다.

```text
mu_k = mean_i(mu_i)
sigma_k^2 = mean_i(exp(logvar_i) + mu_i^2) - mu_k^2
```

`sigma_k^2`는 수치 안정성을 위해 config의 minimum variance 이상으로 clamp한다. 이 식은 per-sample posterior uncertainty와 client 내부 posterior-mean variation을 모두 포함하는 aggregated-posterior moment matching이다.

Evaluation set에서는 각 `q(z|x_i)`에서 fixed-seed latent sample을 하나씩 뽑아 empirical reference `Z_emp`를 만든다. 같은 수의 latent sample을 `N(mu_k, diag(sigma_k^2))`에서 뽑아 `Z_single`을 만든다. 충분성은 다음 진단을 함께 보고 판단한다.

| 진단 | 계산 | 의미 |
| --- | --- | --- |
| `mmd_single` | `Z_emp`와 `Z_single`의 median-bandwidth RBF MMD² | 단일 Gaussian의 distribution-level discrepancy |
| `mmd_empirical_noise` | `Z_emp`의 두 독립 posterior draw 간 RBF MMD² | Finite-sample/posterior sampling noise floor |
| `mmd_excess_ratio` | `mmd_single / (mmd_empirical_noise + epsilon)` | 단일 Gaussian gap이 noise floor보다 얼마나 큰가 |
| `mean_abs_offdiag_correlation` | `Z_emp` empirical correlation matrix의 diagonal 밖 절댓값 평균 | Diagonal covariance가 잃는 dependence |
| `mean_abs_excess_kurtosis` | `Z_emp` 차원별 excess kurtosis 절댓값 평균 | Gaussian tail 가정 위반 |
| `label_mixture_nll_gain` | label-conditioned diagonal-Gaussian mixture NLL 대비 단일 Gaussian held-out NLL 감소량 | Client 내부 semantic multimodality가 만드는 실질적 손실 |

Label-conditioned mixture는 candidate client summary가 아니며, MNIST의 known semantic group이 단일 Gaussian보다 얼마나 더 잘 설명하는지를 보이는 **진단용 upper reference**다. Label을 써서 실제 method를 만들지 않는다.

단일 `(mu_k, sigma_k^2)`를 후속 method의 초기 representation으로 채택할 수 있는지는 client별 `mmd_excess_ratio`의 bootstrap CI가 empirical noise floor와 구분되는지, off-diagonal correlation/kurtosis가 큰지, mixture NLL gain이 일관되게 양수인지 함께 보고 결정한다. 이 첫 run에서 임의의 절대 threshold로 결론을 고정하지 않는다. 명확한 부족이 보이면 mixture/prototype 또는 low-rank covariance가 다음 설계 후보가 된다.

## Functional-gap 평가

각 model은 deterministic posterior-mean path와 `R=8` AWGN path에서 평가한다. Diagonal Gaussian의 `logvar`는 KL 전 동일한 config 범위로 clamp한다.

```text
FG_output^m(x) = mean((x_hat_m - x_hat_retrain)^2)
FG_mu^m(x) = mean((mu_m - mu_retrain)^2)
FG_latent^m(x) = 0.5 * [KL(q_m || q_retrain) + KL(q_retrain || q_m)]
MSE_input^m(x) = mean((x - x_hat_m)^2)
```

`retrain` row도 동일 schema로 저장하며 self-gap은 0이다. AWGN output gap과 `MSE_input`은 reconstruction 평균을 먼저 비교하지 않는다. 동일 noise draw별 metric을 계산한 뒤 R개 repetition에 대해 평균한다.

```text
FG_output_AWGN^m(x) = (1/R) sum_r mean((x_hat_m^r - x_hat_retrain^r)^2)
```

Posterior gap은 channel-independent이므로 두 condition에 중복 기록한다면 metadata로 명시한다. Summary는 run 내부 strata별 mean, standard deviation, median, 95% bootstrap CI를 보고한다. 정식 연구 결과는 최소 3개, 권장 5개 global seed를 보고하며, `retrain_replica`--`retrain` gap을 retraining noise floor로 함께 해석한다.

## 산출물과 figure

매 run은 기존 결과를 덮어쓰지 않는 다음 directory를 생성한다.

```text
outputs/oracle_behavior/<UTC timestamp>_seed-<seed>/
  config.json
  environment.json
  split_manifest.json
  checkpoints/initial.pt
  checkpoints/original.pt
  checkpoints/retrain.pt
  checkpoints/retrain_replica.pt
  checkpoints/scu_forget_only.pt
  checkpoints/scu_self_sharing.pt
  metrics/per_sample.csv
  metrics/summary.json
  predictions/probes.pt
  client_posterior/per_client_gaussian.csv
  client_posterior/summary.json
  figures/reconstruction_grid.png
  figures/latent_comparison.png
  figures/client_gaussian_diagnostic.png
  run.log
```

`environment.json`에는 Python/PyTorch/torchvision/CUDA/cuDNN version, selected device, deterministic setting, top-level Git revision 상태, 그리고 available nested source revision을 기록한다. Top-level workspace가 Git repository가 아니면 그 사실을 명시한다.

`per_sample.csv`의 최소 column은 아래와 같다.

```text
dataset_split, source_index, probe_partition, support_group,
model, evaluation_condition, channel_repetitions,
mse_input, fg_output, fg_mu, fg_latent
```

`predictions/probes.pt`에는 figure 재생성에 필요한 input, posterior mean/logvar, posterior-mean reconstruction, AWGN reconstruction aggregate mean/standard deviation, fixed-noise metadata를 저장한다.

`client_posterior/per_client_gaussian.csv`는 client, fit/evaluation sample count, moment-matched `mu_k`와 `sigma_k^2`, 그리고 모든 충분성 diagnostic을 기록한다. `client_posterior/summary.json`은 seed, fitting ratio, MMD bandwidth policy, bootstrap setting, 그리고 “이 결과가 centralized diagnostic이며 distributional summary 채택 여부는 아직 결론내리지 않는다”는 해석 metadata를 포함한다. Client Gaussian figure는 client별 `mmd_single` 대 `mmd_empirical_noise`, off-diagonal correlation, mixture NLL gain을 나란히 표시한다.

Reconstruction grid는 `shared_high`와 `target_specific` probe를 같은 수로 포함하고, 같은 input에 대해 다음 column을 보인다.

```text
input | original | retrain without target | SCU forget-only | SCU self-sharing
```

Latent figure의 처음 두 coordinate와 arrow는 정성적 진단일 뿐이다. 정량 결론은 전체 latent 20 dimension의 symmetric KL에 근거한다.

## 해석 원칙 및 완료 조건

- `shared_high`에서 original--retrain gap은 작고 SCU--retrain gap이 크면 SCU가 shared semantic을 과도하게 손상했거나 repair가 부족하다는 근거다.
- `target_specific`에서 original--retrain 변화가 크고 SCU가 retrain에 가까우면 oracle-consistent deletion behavior다.
- Target `MSE_input` 증가만으로 성공을 주장하지 않는다.
- Retained utility가 retrain보다 나쁘면 target MSE와 관계없이 utility damage다.
- Client Gaussian diagnostic에서 `mmd_single`이 empirical noise floor보다 지속적으로 크거나 mixture NLL gain이 크면, 단일 `(mu_k, sigma_k^2)`는 client semantic distribution의 충분한 representation이 아니다. 반대의 결과여도 이는 이 MNIST setting의 초기 summary 선택 근거일 뿐, 일반적인 충분성 증명은 아니다.
- 이 centralized 실험은 membership privacy, latent client-identity leakage, secure aggregation, distributed repair의 효과를 직접 증명하지 않는다.

구현 완료 전 CPU synthetic-data smoke test로 split, model shape, artifact schema, CSV/JSON/checkpoint/figure 생성을 검증한다. 이후 CUDA MNIST pilot을 최소 한 seed에서 end-to-end로 수행한다. Smoke test와 pilot 모두 `src/SCU/`를 수정하지 않아야 한다.
