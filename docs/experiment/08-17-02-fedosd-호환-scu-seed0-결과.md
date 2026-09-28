> **상태: 무효 실험 기록**
>
> 이 결과의 `SCU`는 target-only `forget-only` ablation이다. Full SCU의
> compensation 결과가 아니므로 이후 baseline 표와 연구 해석에서 제외한다.
> Artifact는 재현 기록으로만 보존한다.

# FedOSD 호환 SCU 세 모델 CIFAR-10 seed 0 결과

## 실행 요약

다음 설정으로 정식 실행을 완료했다.

```text
실행 시각                 2026-08-17 KST
dataset                   CIFAR-10
client 수                 10
분할                      FedOSD balanced pathological, client당 2 label
client당 train/test       5,000 / 1,000
삭제 대상                 client_0 전체
original/retrain          FedAvg 200 communication rounds, local epoch 1
SCU                       original에서 200 local forgetting epochs
batch size                200
channel                   power-normalized AWGN 10 dB
seed / split seed         0 / 1
```

실행 명령은 다음과 같다.

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_fedosd_pat_scu.json \
  --seed 0 \
  --output-root outputs/federated_cifar10_fedosd_pat_scu
```

결과 directory는
`outputs/federated_cifar10_fedosd_pat_scu/20260817T050543Z_seed-0/`이다.

## 분할 검증

`split_manifest.json`에서 10개 모든 client의 train/test sample 수가 각각
5,000/1,000개임을 확인했다. client_0과 client_1은 FedOSD clone의 실제
순차 slot 할당대로 label 0·1을 각각 2,500개씩 공유한다.

즉 client_0을 삭제해도 그 label의 semantic support는 client_1에 남아 있다.
이 조건에서는 target client의 training contribution과 shared semantic knowledge를
구분해서 볼 수 있다.

## 전체 test 결과

기준은 `retrain_without_client_0`이다. `fg_output`, `fg_mu`, `fg_latent`는
각각 기준 model과의 reconstruction output MSE, posterior mean MSE, symmetric
posterior KL 차이이며 작을수록 기준 model에 가깝다.

| Model | Input MSE ↓ | PSNR ↑ | Output gap ↓ | Mu gap ↓ | Latent gap ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.015428 | 18.117 dB | 0.000260 | 0.073300 | 15.920227 |
| retrain_without_client_0 | 0.015418 | 18.120 dB | 0.000000 | 0.000000 | 0.000000 |
| SCU | 0.290229 | 5.373 dB | 0.277976 | 0.073158 | 15.998822 |

## Client별 확인

`client_0`은 삭제 target이고, `client_1`은 client_0과 똑같은 label 0·1을
보유한 retained client다. 나머지 retained mean은 client 2--9의 평균이다.

| Model | client_0 MSE | 공유 client_1 MSE | 나머지 retained 평균 MSE |
| --- | ---: | ---: | ---: |
| original | 0.017139 | 0.017318 | 0.014977 |
| retrain_without_client_0 | 0.017209 | 0.017363 | 0.014952 |
| SCU | 0.336139 | 0.335061 | 0.278887 |

## 해석

첫째, 실제 client deletion이 reconstruction behavior를 거의 바꾸지 않았다.
Original과 기준 재학습 model의 전체 MSE 차이는 약 `9.1e-06`, output gap은
`2.60e-04`다. 특히 target client_0의 held-out MSE도 `0.017139`에서
`0.017209`로만 변했다. 이는 label 0·1의 support가 client_1에 남아 있기
때문에 자연스러운 결과다.

둘째, SCU forget-only는 기준 재학습 model을 근사하지 못했다. 전체 MSE는
`0.015418`에서 `0.290229`로 약 18.8배 커졌고, output gap은 `0.277976`이다.
삭제 target뿐 아니라 같은 label을 공유하는 retained client_1도 MSE `0.335061`로
무너졌다. 따라서 target MSE 증가만 보고 SCU가 성공했다고 해석하면 안 된다.
shared semantic knowledge까지 과도하게 손상한 결과다.

셋째, latent 관점에서는 SCU가 기준 재학습 쪽으로 이동한 증거가 없다.
SCU의 `fg_mu=0.073158`, `fg_latent=15.998822`는 original의 `0.073300`,
`15.920227`와 거의 같다. 현재 SCU loss는

```math
\lambda_{KL} KL(q_U || q_G) - \lambda_{rec} MSE
```

를 최소화하므로 posterior를 원본에 가깝게 유지하면서 reconstruction MSE를
키운다. 관측된 결과는 encoder posterior의 큰 이동보다 decoder/reconstruction
경로의 손상에 가깝다.

## 학습량 점검과 한계

Original의 평균 local reconstruction loss는 round 1의 `0.063966`에서 round
200의 `0.018638`로, retrain은 `0.063209`에서 `0.018613`으로 감소했다.
다만 round 175에서 200 사이에도 original은 `0.019416 -> 0.018638`, retrain은
`0.019066 -> 0.018613`으로 계속 줄고 있다. 따라서 이번 200-round 결과는
명확한 비교 결과이지만 완전한 수렴을 보장하는 최종 결과로는 주장하지 않는다.

이 문서는 seed 0 한 번의 관측이다. 다중 seed 평균, channel/SNR 변화,
membership inference, marker 진단은 아직 수행하지 않았다. 다음 방법 비교에서는
SCU의 과도한 decoder 손상을 줄이면서 기준 재학습 behavior를 근사하는 distributed
repair를 별도로 검증해야 한다.
