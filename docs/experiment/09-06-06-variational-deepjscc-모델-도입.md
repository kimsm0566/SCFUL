# Variational DeepJSCC 모델 도입

## 결정

CIFAR-10 본 실험의 기본 codec을 exact SCU Table-2 ResNet model에서
`VariationalDeepJSCC`으로 변경한다.

이 model은 원본 DeepJSCC의 5-layer convolutional encoder와 대칭적인 5-layer
transposed-convolutional decoder를 유지한다. SCU의 VIB 및 joint unlearning loss를
사용할 수 있도록 마지막 encoder layer만 posterior mean과 log-variance를 동시에
출력하도록 수정한다.

## 구조

```text
input: RGB 3 x 32 x 32

encoder:
  3->16,  kernel 5, stride 2, PReLU
  16->32, kernel 5, stride 2, PReLU
  32->32, kernel 5, stride 1, PReLU
  32->32, kernel 5, stride 1, PReLU
  32->8,  kernel 5, stride 1

posterior:
  mean map:         4 x 8 x 8
  log-variance map: 4 x 8 x 8
  transmitted latent dimension: 256 real values

channel:
  reparameterized sample
  -> per-sample power normalization
  -> AWGN

decoder:
  4->32, kernel 5, stride 1, PReLU
  32->32, kernel 5, stride 1, PReLU
  32->32, kernel 5, stride 1, PReLU
  32->16, kernel 5, stride 2, PReLU
  16->3, kernel 5, stride 2, Sigmoid
```

SCU 방식의 1-to-2 backdoor ASR을 평가하기 위해 reconstructed RGB image 뒤에
`3072 -> 256 -> 10` auxiliary classifier를 둔다. 이 classifier는 semantic channel
latent를 만들거나 image를 복원하는 inference 경로에는 직접 필요하지 않지만,
joint training 중 CE gradient는 decoder와 encoder까지 전파된다.

## 원본 DeepJSCC와의 차이

- 원본은 deterministic channel code를 출력한다.
- 본 구현은 동일 크기의 mean과 log-variance를 출력하고 학습 중 sampling한다.
- KL은 SCU와 동일하게 batch 및 latent 전체 원소의 mean으로 계산한다.
- 본 구현은 SCU backdoor 진단용 auxiliary semantic head를 추가한다.
- 현재 channel tensor는 256개의 real value로 구현한다. 원본 논문의 complex-symbol
  해석과 완전히 같은 구현이라고 주장하지 않는다.

따라서 이름은 `original DeepJSCC`가 아니라 `variational DeepJSCC adaptation`으로
고정한다.

## 크기

- Trainable parameters: 929,717
- FP32 state tensor: 3,718,868 bytes, 약 3.55 MiB
- Exact SCU Table-2 model 대비 parameter 수: 약 37분의 1

이 크기에서는 개발용 12 GB GPU에서 batch 50인 10개 client process를 같은
scheduling group으로 실행할 수 있다.

## 본 실험 설정

공통 설정:

- CIFAR-10
- 10 clients
- client당 3,000 train samples
- server public calibration data는 class당 100개, 총 1,000개
- batch size 50
- 200 FL rounds
- target-client joint encoder-decoder forgetting
- remaining-client distributed repair 연구 경로

두 non-IID 설정:

1. `federated_cifar10_fedosd_2class_vdeepjscc_remaining_consensus.json`
2. `federated_cifar10_dirichlet_01_vdeepjscc_remaining_consensus.json`

Public data를 먼저 제외한 49,000개 중 class-balanced 30,000개만 client training
pool로 사용한다. 나머지 19,000개는 사용하지 않고 index hash를 manifest에 남긴다.

## 검증 결과

### Synthetic 10-client smoke test

- Config: `federated_cifar10_vdeepjscc_remaining_consensus_smoke.json`
- Seed: 0
- Output: `outputs/2026-09-06-21-vdeepjscc-smoke`
- Result: 10 clients, joint forgetting, retraining baseline, remaining-client consensus,
  paired-channel evaluation까지 오류 없이 완료

### Real CIFAR-10 one-round benchmark

- Config: `federated_cifar10_vdeepjscc_batch50_all10_benchmark.json`
- Seed: 0
- Output: `outputs/2026-09-06-22-vdeepjscc-batch50-all10-benchmark`
- Result: 10 clients를 같은 GPU scheduling group에서 batch 50으로 실행 완료
- Original training wall time: 약 2.17초
- Retraining-without-target wall time: 약 1.08초
- Target joint-unlearning wall time: 약 0.48초
- Remaining-client consensus collection wall time: 약 1.28초

이 수치는 한 라운드 runtime 및 memory feasibility 진단이다. Model이 아직 수렴하지
않았으므로 reconstruction, functional gap, ASR을 최종 성능으로 해석하지 않는다.
이 run 뒤 variational codec의 KL reduction을 SCU-compatible element-wise mean으로
명시적으로 고정했으므로 현재 loss behavior 판단에는 아래 10-round run을 사용한다.

### Real CIFAR-10 10-round convergence pilot

- Config: `federated_cifar10_vdeepjscc_10round_convergence_pilot.json`
- Seed: 0
- Output: `outputs/2026-09-06-24-vdeepjscc-mean-kl-10round-pilot`
- Original train reconstruction MSE: round 1의 0.06271에서 round 10의 0.03265
- Original test MSE / PSNR: 0.03101 / 15.08 dB
- Retrain test MSE / PSNR: 0.03010 / 15.21 dB
- Original / retrain backdoor ASR: 0.852 / 0.000
- JU-only backdoor ASR: 0.000
- JU-only test MSE: 0.19349
- JU posterior KL to original: 3.58090

Reconstruction training은 10 round 동안 안정적으로 개선되었고 original과 retrain의
backdoor marker도 분리되었다. 그러나 clean semantic accuracy가 original 0.1101,
retrain 0.1038로 아직 random 수준이어서 10-round ASR을 최종 backdoor 성능으로
해석하지 않는다.

JU-only는 marker ASR을 retrain 수준으로 내렸지만 reconstruction을 크게 붕괴시켰다.
이는 joint forgetting 뒤 remaining-client distributed repair가 필요하다는 pipeline의
전제와 일치한다. 동시에 현재 `lambda_kl`, `lambda_rec`, forgetting epoch를 최종값으로
확정할 수 없다는 뜻이므로 full run에서는 retrain functional gap과 retained utility를
기준으로 calibration해야 한다.

## 중단된 exact SCU run

`outputs/2026-09-06-20-scu-model-consensus-2class-batch50-all10`의 200-round
exact-SCU 실행은 backbone 변경 결정 직후 중단했다. 완료 결과가 아니므로 성능 표나
method 비교에 사용하지 않는다.

## 200-round 본 실험 전 남은 조건

10-round pilot으로 codec 학습과 10-client 실행 가능성은 확인했다. 다음 두 조건을
고정한 뒤 seed 0의 200-round 2-class와 Dirichlet 0.1 실험을 시작한다.

- JU hyperparameter를 retrain gap과 retained reconstruction utility로 calibration
- JU로 손상된 encoder와 decoder를 복구할 distributed retain repair 경로 선택

Backdoor ASR은 보조 진단으로 유지하며, clean semantic accuracy가 충분히 학습되기
전의 ASR만으로 hyperparameter를 선택하지 않는다.
