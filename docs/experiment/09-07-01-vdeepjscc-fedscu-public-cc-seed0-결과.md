# Variational DeepJSCC FedSCU-PubCC Seed 0 결과

## 목적

새 기본 model인 `variational_deepjscc_cifar10`에서 public-data server contrastive
compensation baseline이 작동하는지 확인한다.

```text
target client joint encoder-decoder unlearning
-> erased posterior-mean buffer 전송
-> server public-data contrastive and reconstruction compensation
-> retrain-without-target와 비교
```

## 공통 설정

- Dataset: CIFAR-10
- Clients: 10
- Client train samples: 각 3,000개
- Public calibration data: class당 100개, 총 1,000개
- Batch size: 50
- Original and retrain FL rounds: 200
- Channel: AWGN 20 dB
- Reconstruction evaluation: official CIFAR-10 test 10,000장, paired AWGN 20 dB,
  posterior mean transmission
- Model: Variational DeepJSCC, latent 256, element-wise mean KL
- JU: 5 epochs
- Erased representation buffer: 1,024 x 256, 1,048,576 bytes
- Server CC: 40 epochs
- Seed: 0

고정 CC hyperparameter는 이전 public-CC 설정의 선택값을 이식한 것이다. 새 model에서
다시 grid search한 최적값은 아니므로 이 결과는 architecture-port baseline으로
해석한다.

## 전체 test 결과

### Balanced 2-class per client

Training output: `outputs/2026-09-07-02-vdeepjscc-fedscu-pubcc-2class`

AWGN 20 dB audit: `outputs/2026-09-07-04-vdeepjscc-pubcc-2class-awgn20-audit`

SCU extended metrics: `outputs/2026-09-07-08-vdeepjscc-scu-extended-metrics-2class`

| Model | MSE | PSNR dB | Retrain output gap | Backdoor ASR | Clean semantic accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.007268 | 21.386 | 0.000581 | 0.821 | 0.144 |
| Retrain without client 0 | 0.007467 | 21.269 | 0 | 0.017 | 0.235 |
| Client JU only | 0.337029 | 4.723 | 0.330657 | 0.000 | 0.100 |
| FedSCU-PubCC | 0.008887 | 20.513 | 0.001739 | 0.029 | 0.316 |

### Equal-size Dirichlet 0.1

Training output: `outputs/2026-09-07-03-vdeepjscc-fedscu-pubcc-dirichlet01`

AWGN 20 dB audit: `outputs/2026-09-07-05-vdeepjscc-pubcc-dirichlet01-awgn20-audit`

SCU extended metrics: `outputs/2026-09-07-09-vdeepjscc-scu-extended-metrics-dirichlet01`

| Model | MSE | PSNR dB | Retrain output gap | Backdoor ASR | Clean semantic accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.007673 | 21.151 | 0.000482 | 0.122 | 0.309 |
| Retrain without client 8 | 0.007660 | 21.157 | 0 | 0.121 | 0.325 |
| Client JU only | 0.337041 | 4.723 | 0.329086 | 0.000 | 0.100 |
| FedSCU-PubCC | 0.008999 | 20.458 | 0.001571 | 0.031 | 0.321 |

## Target와 remaining client 분리

| Split | Model | Target MSE | Target output gap | Remaining MSE | Remaining output gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 2-class | Retrain | 0.008208 | 0 | 0.007385 | 0 |
| 2-class | JU only | 0.307579 | 0.300371 | 0.340302 | 0.334022 |
| 2-class | FedSCU-PubCC | 0.009691 | 0.001920 | 0.008797 | 0.001719 |
| Dirichlet 0.1 | Retrain | 0.007322 | 0 | 0.007698 | 0 |
| Dirichlet 0.1 | JU only | 0.344857 | 0.337232 | 0.336173 | 0.328181 |
| Dirichlet 0.1 | FedSCU-PubCC | 0.008669 | 0.001525 | 0.009036 | 0.001576 |

## SCU 지표와 MIA/PSNR 통합 결과

`Clean MSE/PSNR`은 공식 CIFAR-10 test 10,000장, `Erased MSE/PSNR`은 삭제
client의 실제 local training data 3,000장에 대한 AWGN 20 dB 결과다. Backdoor
accuracy는 그중 실제 poison marker 180장에 대한 값이다. MIA는 best-direction
ROC-AUC이며 0.5에 가까울수록 공격이 membership을 구분하지 못한다.

### Balanced 2-class per client

| Model | Clean MSE | Erased MSE | Clean PSNR | Erased PSNR | Model acc. | Backdoor acc. | MIA AUC | Time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.007268 | 0.008001 | 21.386 | 20.968 | 0.144 | 0.828 | 0.522 | 223.14 |
| Retrain without client 0 | 0.007467 | 0.008276 | 21.269 | 20.822 | 0.235 | 0.000 | 0.520 | 202.19 |
| Client JU only | 0.337029 | 0.315549 | 4.723 | 5.009 | 0.100 | 0.000 | 0.523 | 1.20 |
| FedSCU-PubCC | 0.008887 | 0.009765 | 20.513 | 20.103 | 0.316 | 0.022 | 0.518 | 4.81 |

### Equal-size Dirichlet 0.1

| Model | Clean MSE | Erased MSE | Clean PSNR | Erased PSNR | Model acc. | Backdoor acc. | MIA AUC | Time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.007673 | 0.007502 | 21.151 | 21.248 | 0.309 | 0.111 | 0.517 | 221.52 |
| Retrain without client 8 | 0.007660 | 0.007481 | 21.157 | 21.260 | 0.325 | 0.094 | 0.518 | 201.03 |
| Client JU only | 0.337041 | 0.346139 | 4.723 | 4.607 | 0.100 | 0.000 | 0.521 | 1.21 |
| FedSCU-PubCC | 0.008999 | 0.008827 | 20.458 | 20.542 | 0.321 | 0.039 | 0.519 | 4.78 |

2-class에서는 original과 retrain의 backdoor accuracy가 0.828 대 0.000으로 분리돼
marker 진단이 유효하다. Dirichlet 0.1에서는 0.111 대 0.094로 분리가 없어 backdoor
accuracy를 unlearning 성공 근거로 사용하지 않는다. MIA는 모든 model에서
0.517--0.523으로 random guessing에 가까워 보조 결과로만 둔다.

## 해석

### Utility recovery

Public CC는 JU-only가 만든 decoder collapse를 두 split 모두 크게 복구했다. 전체
MSE는 0.337 수준에서 0.0089~0.0090 수준으로 돌아왔다. Target와 remaining client
partition에서도 같은 회복이 확인된다.

### Retrain equivalence

현재 PubCC는 retrain-equivalence를 개선하지 못했다.

- 2-class: PubCC output gap은 original gap의 2.99배
- Dirichlet 0.1: PubCC output gap은 original gap의 3.26배
- PubCC posterior-mean gap도 original보다 크게 증가

즉 public reconstruction loss는 decoder utility를 복원하지만 contrastive encoder
update가 retrained encoder 방향을 따라가지는 않는다.

### Backdoor

2-class에서는 original ASR 0.821과 retrain ASR 0.017이 분명히 갈리며, PubCC ASR
0.029는 retrain에 가깝다. 이 split에서는 client marker 제거의 보조 증거로 사용할 수
있다.

Dirichlet 0.1에서는 original과 retrain ASR이 각각 0.122와 0.121로 사실상 같다.
공격 신호가 deletion effect를 구분하지 못하므로 PubCC ASR 0.031을 unlearning 성공
근거로 사용하지 않는다.

### Efficiency

| Split | Retrain time seconds | JU time seconds | Server CC seconds | Online speedup vs retrain |
| --- | ---: | ---: | ---: | ---: |
| 2-class | 202.19 | 1.20 | 3.61 | 42.04x |
| Dirichlet 0.1 | 201.03 | 1.21 | 3.57 | 42.10x |

Algorithmic online communication은 두 split 모두 4,540,072 bytes다. 이는 target
codec update와 1 MiB erased representation buffer를 포함한다.

## 결론

Variational DeepJSCC에서 FedSCU-PubCC 실행 경로 자체는 정상 작동하며 JU 이후
reconstruction utility를 빠르게 복구한다. 그러나 현재 설정은 unlearned codec을
retrain oracle보다 original global codec에서 더 멀리 이동시킨다. 따라서
FedSCU-PubCC는 public-data utility-recovery baseline으로 유지하되, proposed
distributed repair 또는 retrain-equivalent method로 주장하지 않는다.

기존 training artifact에 저장된 reconstruction 표는 noiseless posterior-mean
평가였다. 해당 값은 삭제하지 않고 새 audit의 `_noiseless` 보조 필드로 재현했으며,
앞으로 headline 결과에는 위의 paired-AWGN 20 dB 값을 사용한다.
