# Remaining-Client Consensus Seed-0 결과

## 결론

현재 형태의 remaining-client prediction consensus는 retrain-without-target의
더 나은 proxy가 아니었다.

```text
2-class:
    representation raw MSE는 original 대비 0.66% 악화
    paired-output MSE는 original 대비 1.39% 악화

Dirichlet alpha 0.1:
    representation raw MSE는 original 대비 0.95% 개선
    paired-output MSE는 original 대비 7.18% 악화
```

두 split 모두 사전에 정한 조건인 "representation과 paired output이 original 대비
각각 5% 이상 동시 개선"을 만족하지 못했다. 따라서 consensus distillation은 두
실험 모두 실행하지 않았다.

## 실행 설정

공통 설정:

- CIFAR-10
- 10 clients
- Client당 4,900 private training samples
- Server public calibration data: class당 100개, 총 1,000개
- Federated training: 200 rounds
- Local batch size: 200
- AWGN, train/evaluation SNR 20 dB
- Remaining-client consensus adaptation: 1 local epoch
- Target client SCU-style joint unlearning: 5 epochs
- Seed: 0

Split별 설정:

| Split | Target client | Partition |
|---|---:|---|
| 2-class | 0 | client당 2 classes |
| Dirichlet 0.1 | 8 | equal-size Dirichlet alpha 0.1 |

모든 candidate는 동일한 1,000개 public samples와 정확히 동일한 AWGN noise tensor로
평가했다. Client response는 local sample 수로 가중 평균했다.

## Consensus와 retrain 비교

### 2-class

| Candidate | Posterior raw MSE | Linear CKA | SVCCA | Paired-output MSE | Public input MSE |
|---|---:|---:|---:|---:|---:|
| Prediction consensus | 0.081385 | 0.975879 | 0.991281 | 0.00066255 | 0.013225 |
| Original global | 0.080851 | 0.976288 | 0.991275 | 0.00065344 | 0.013247 |
| Parameter FedAvg proxy | 0.081377 | 0.975900 | 0.991271 | 0.00067050 | 0.013226 |
| Retrain reference | 0 | 1 | 1 | 0 | 0.013899 |

Prediction consensus는 public input 자체를 original보다 약간 잘 복원했지만,
unlearning의 기준인 retrain output에는 더 멀어졌다. CKA와 SVCCA도 original 대비
의미 있게 개선되지 않았다.

### Dirichlet alpha 0.1

| Candidate | Posterior raw MSE | Linear CKA | SVCCA | Paired-output MSE | Public input MSE |
|---|---:|---:|---:|---:|---:|
| Prediction consensus | 0.042712 | 0.996938 | 0.994872 | 0.00029929 | 0.012832 |
| Original global | 0.043120 | 0.996898 | 0.994825 | 0.00027924 | 0.012853 |
| Parameter FedAvg proxy | 0.042730 | 0.996936 | 0.994866 | 0.00030155 | 0.012832 |
| Retrain reference | 0 | 1 | 1 | 0 | 0.012811 |

Representation raw MSE와 구조 지표는 소폭 개선됐지만, paired-output MSE가 7.18%
악화됐다. Reconstruction semantic codec에서는 최종 output behavior가 핵심이므로
latent 지표의 작은 개선만으로 distillation을 정당화하지 않았다.

## Gate 결과

| Split | Representation 개선율 | Output 개선율 | Gate | Distillation |
|---|---:|---:|---|---|
| 2-class | -0.66% | -1.39% | 실패 | 실행 안 함 |
| Dirichlet 0.1 | +0.95% | -7.18% | 실패 | 실행 안 함 |

음수 개선율은 original보다 retrain에서 더 멀어졌다는 뜻이다.

## 실제 model checkpoint 진단

Prediction consensus는 ensemble response이므로 그 자체가 하나의 codec checkpoint는
아니다. 아래 표는 함께 생성한 model checkpoint의 전체 test reconstruction 결과다.

| Split | Model | Test MSE | PSNR | Retrain output gap |
|---|---|---:|---:|---:|
| 2-class | Original | 0.013025 | 18.852 | 0.000618 |
| 2-class | Retrain | 0.013661 | 18.645 | 0 |
| 2-class | Target JU | 0.305363 | 5.152 | 0.291168 |
| 2-class | Parameter FedAvg proxy | 0.013003 | 18.860 | 0.000633 |
| Dirichlet 0.1 | Original | 0.012652 | 18.978 | 0.000246 |
| Dirichlet 0.1 | Retrain | 0.012592 | 18.999 | 0 |
| Dirichlet 0.1 | Target JU | 0.315738 | 5.007 | 0.299036 |
| Dirichlet 0.1 | Parameter FedAvg proxy | 0.012626 | 18.987 | 0.000270 |

Target JU는 두 split 모두 decoder collapse를 보였다. Parameter FedAvg proxy는
reconstruction utility는 유지했지만 original보다 retrain functional gap을 줄이지
못했다.

## Backdoor 진단

| Split | Original ASR | Retrain ASR | Target JU ASR | Parameter proxy ASR |
|---|---:|---:|---:|---:|
| 2-class | 74.1% | 2.1% | 0.0% | 24.7% |
| Dirichlet 0.1 | 19.3% | 14.6% | 0.0% | 16.4% |

2-class에서 parameter proxy는 original보다 marker를 줄였지만 retrain 수준에는
도달하지 못했다. Dirichlet에서는 original과 retrain의 ASR 차이가 4.7%p뿐이라
marker 신호 자체가 약하다. Target JU의 0% ASR은 clean semantic accuracy도 10%로
붕괴한 상태이므로 성공적인 unlearning 증거로 해석하지 않는다.

## 비용

| Split | Consensus audit time | Retrain time | 시간 비율 | Method communication | 실제 audit communication |
|---|---:|---:|---:|---:|---:|
| 2-class | 2.21 s | 111.98 s | 50.7x 빠름 | 188.30 MB | 225.11 MB |
| Dirichlet 0.1 | 3.79 s | 176.59 s | 46.6x 빠름 | 188.30 MB | 225.11 MB |

Method communication에는 target JU model transfer, remaining clients의 model
download, public/noise broadcast, response upload이 포함된다. 실제 audit
communication에는 parameter FedAvg comparator를 만들기 위한 remaining-client
model-state upload 36.81 MB가 추가된다. Response upload만 총 113.29 MB이므로 향후
사용한다면 response compression이나 public sample 축소가 필요하다.

속도는 빠르지만 gate를 통과한 최종 unlearned codec이 생성되지 않았으므로 이 수치를
성공한 unlearning method의 efficiency로 주장할 수는 없다.

## 해석

이번 결과는 다음 단순 가설을 지지하지 않는다.

```text
current global model에서 remaining data로 짧게 적응한 client responses를 평균하면
retrain-without-target behavior가 더 잘 근사된다.
```

가장 가능성 높은 이유는 모든 remaining local model이 target contribution을 이미
포함한 current global model에서 시작한다는 점이다. 한 번의 local adaptation과
response averaging은 retained utility를 조정할 수는 있지만 historical target
influence를 명시적으로 제거하지 않는다. Prediction consensus와 parameter FedAvg가
거의 같은 방향으로 실패한 것도 이 해석과 일치한다.

따라서 현재 설정에서는 client-specific negative뿐 아니라 단순 remaining-response
positive도 retrain proxy로 사용하지 않는다. 다음 방법은 target influence 제거와
retain repair를 분리하되, original에서 시작한 local response 평균 이상의 삭제
신호 또는 history-aware correction을 필요로 한다.

## 산출물

- `outputs/2026-09-06-08-remaining-client-consensus-2class/20260906T095813Z_seed-0`
- `outputs/2026-09-06-09-remaining-client-consensus-dirichlet01/20260906T100229Z_seed-0`

각 run에는 config, split manifest, checkpoints, consensus comparison CSV, gate JSON,
aggregate consensus tensor, backdoor 결과, efficiency 기록이 포함된다. Individual
remaining-client responses는 저장하지 않았다.
