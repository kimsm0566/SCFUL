# 평가 지표와 SCU Backdoor 프로토콜

## 판정 원칙

Federated unlearning의 성공은 삭제 client의 복원 성능을 의도적으로
악화시키는 것이 아니라, 모든 지표에서 target client를 빼고 처음부터
학습한 `retrain_without_target` codec에 가까워지는 것으로 판정한다.

```text
unlearned ~= retrain_without_target
```

각 지표는 절대적으로 낮거나 높은지보다 `|metric_unlearn - metric_retrain|`을
우선한다. Target partition `D_u`와 retained partition `D_r`를 반드시 분리해
보고한다.

## 평가 우선순위

| 위치 | 목적 | 지표 |
| --- | --- | --- |
| Primary 1 | Retrain functional equivalence | 동일한 `x`와 동일한 channel realization `H`에서 output MSE gap, RMSE-255, PSNR, MS-SSIM, LPIPS |
| Primary 2 | Unlearning efficiency | retraining 대비 wall time, server/client compute, 추가 round, model/update/표현 buffer 통신 byte, peak storage |
| Secondary | Representation equivalence | CKA, SVCCA, client-ID leakage; 필요 시 Procrustes/MMD/Wasserstein |
| Sanity check | 삭제 신호 제거 | SCU-compatible backdoor ASR |
| Supplementary | Privacy audit | class-matched reconstruction MIA |

## 고정 Headline 지표

앞으로 모든 본 실험은 original, retrain-without-target, unlearned checkpoint에 대해
다음 일곱 지표를 같은 표에 보고한다. Clean reconstruction과 erased reconstruction은
반드시 분리한다.

| 지표 | 본 연구의 구현 |
| --- | --- |
| RMSE-255 | clean test set과 target-client erased set의 normalized MSE를 구한 뒤 `255 * sqrt(MSE)`로 변환 |
| PSNR | peak value 1.0과 동일 normalized MSE로 계산한 dB 값 |
| Accuracy | 동일한 frozen clean classifier가 decoded clean test samples를 분류한 accuracy |
| ASR | clean test의 base-class samples에 SCU trigger를 새로 붙인 probe가 trigger label로 분류되는 비율 |
| MIA | class-matched reconstruction-loss attack의 ROC-AUC |
| BA | target client의 실제 poisoned erased samples가 trigger label로 분류되는 backdoor accuracy |
| Runtime | 각 phase에서 직접 측정한 wall time과 retraining 대비 speedup |

`BA`는 이 repository에서 항상 backdoor accuracy를 뜻한다. MIA threshold attack의
balanced accuracy도 보조 artifact에 저장할 수 있지만 `BA`로 줄이지 않고
`MIA balanced accuracy`라고 표기한다.

BA와 ASR은 같은 값이 아니다. BA는 target client에서 학습에 실제 사용된 poisoned
erased samples를 평가한다. ASR은 학습에 쓰지 않은 clean test base-class samples에
동일 trigger를 새로 붙여 일반화된 trigger 반응을 평가한다. 구현 field는 각각
`backdoor_accuracy`, `backdoor_test_probe_asr`다.

### RMSE-255 구현

논문 본문은 MSE를 사용한다고 명시하고 clean data는 training에 포함되지 않은 test
dataset, erased data는 삭제할 training dataset이라고 설명한다. 본 구현도 입력을
0--1로 변환하고 `(reconstruction - input)^2`를 sample, RGB channel, height, width
전체에서 평균하여 normalized MSE를 얻는다. Headline RMSE-255는 그 값의 제곱근에
255를 곱한다. PSNR은 같은 normalized MSE와 peak value 1.0으로 계산한다.

다만 SCU 공개 CIFAR 코드의 보고용 `rec_function`은
`MSELoss(size_average=False)`이고 batch size는 16이다. 따라서 공개 결과 script의
큰 숫자는 표준 MSE와 직접 비교할 수 있는 값이 아니라 batch-summed squared error다.
본 연구는 batch size에 독립적인 RMSE-255를 canonical reconstruction 값으로 사용한다.
Normalized MSE는 PSNR과 retrain functional gap을 계산하기 위한 내부·보조 값으로
계속 저장한다.

Raw latent KL/MSE는 표현 좌표계의 rotation과 permutation에 민감하므로 메인
성공 지표로 쓰지 않는다. CKA/SVCCA와 client-ID leakage를 우선하고 raw
latent gap은 보조 분석으로만 둔다.

## Paired-channel Functional Evaluation

Model 간 비교에는 같은 input sample과 같은 channel randomness를 사용한다.

기본 headline 조건은 공식 CIFAR-10 test set 전체 10,000장, posterior mean path,
paired AWGN 20 dB다. `rmse_255`, `psnr`, `fg_output`은 이 조건을 뜻한다. Channel을
생략한 posterior-mean reconstruction은 `_noiseless` suffix를 붙여 보조 지표로만
저장한다.

`--algorithm FSCU` 실행은 언러닝 직후 standard metric suite를 자동 생성한다.
최종 `metrics/headline_metrics.csv`의 필수 field는 `clean_rmse_255`,
`erased_rmse_255`, `clean_psnr_db`, `erased_psnr_db`, `model_accuracy`,
`backdoor_test_probe_asr`, `mia_auc`, `backdoor_accuracy`,
`running_time_seconds`다. 이 파일이 없거나 headline 값이 누락된 run은 완성된 본 실험
결과로 취급하지 않는다.

## 필수 Figure Artifact

Headline CSV만 생성해서는 본 실험이 완료되지 않는다. 같은 `src/main.py` 실행에서
다음 figure를 `figures/`에 자동 생성해야 한다.

- `unlearning_headline_comparison.png`: Original, Retrain-without-target, Unlearned의
  clean/erased RMSE-255와 PSNR, Accuracy/ASR/MIA/BA, Runtime을 단위별 panel로 분리한
  비교 그래프
- `unlearning_optimization.png`: iterative forgetting, compensation 또는 distributed
  repair의 step별 total/component loss와 evaluation metric. Iterative stage가 전혀 없는
  방법은 생략 사유를 run metadata에 기록한다.
- `reconstruction_comparison_grid.png`: 같은 input과 paired channel realization에서 input,
  Original, Retrain, Unlearned output을 나란히 둔 정성적 보조 비교

각 figure의 모든 정량 point는 `metrics/headline_metrics.csv` 또는 별도 stage-history
CSV/JSON에서 재생성할 수 있어야 한다. Train, held-out/test, clean, erased를 legend와
axis label에서 구분하며, 서로 다른 단위를 같은 y축에 겹치지 않는다. Smoothing은 raw
curve를 가리지 않는 보조선으로만 사용하고 window를 기록한다. 여러 seed를 합친 figure는
seed별 raw point와 mean 및 uncertainty를 함께 표시한다. 필수 CSV와 적용 가능한 필수
figure 중 하나라도 없으면 완료된 본 실험으로 취급하지 않는다.

\[
\operatorname{FG}_{out}(S,H)=\frac{1}{|S|}\sum_{x\in S}
\lVert \hat{x}_{unlearn}(x,H)-\hat{x}_{retrain}(x,H)\rVert_2^2.
\]

- `S = D_u_test`: 삭제 client distribution
- `S = D_r_test`: remaining clients의 합집합
- Channel: AWGN과 Rayleigh
- SNR: 0, 5, 10, 15, 20 dB
- Train/test SNR mismatch를 별도 보고

Hyperparameter는 seed 0에서 한 번 고정한 후 seed 1, 2에 그대로 적용한다.
각 실행의 candidate 선택에 final test set이나 retrain checkpoint를 사용하지
않는다. 2026-09-03의 192-candidate grid 결과는 이 원칙을 고정하기 전의
oracle-tuned pilot로만 해석한다.

## SCU와 동일한 Backdoor 신호

SCU의 공개 CIFAR-10 코드를 기준으로 다음 신호를 고정한다.

- Dataset: CIFAR-10
- Base label: class 1
- Trigger label: class 2
- Trigger: 모든 RGB channel의 zero-based rows/columns 28–29인 흰색 `2 x 2` patch
- Poison transformation: class-1 image에 trigger를 넣고 label을 2로 변경
- Channel: 20 dB AWGN
- Semantic observer: reconstructed RGB image를 입력으로 받는 별도
  `cifar10_groupnorm_silu_cnn`. Client/public-calibration data와 겹치지 않는 clean
  CIFAR-10 train 19,000장으로 학습하고 모든 codec에 동일하게 고정 적용
- Joint training loss: `beta * KL + 30 * reconstruction_MSE + CE / 30`
- ASR: triggered class-1 input의 reconstruction이 frozen clean classifier에서 class 2로
  분류된 비율

Classifier는 Original codec이나 poisoned label로 학습하지 않는다. Backdoor
scenario의 poisoned-label CE는 classifier parameter를 고정한 상태에서 reconstruction을
통해 codec으로만 역전파한다. 따라서 Original/Retrain/Unlearned에 같은 classifier를
적용해도 ASR 차이는 observer가 아니라 codec output의 trigger 반응 차이를 뜻한다.
Classifier state hash와 clean raw-test accuracy를 run artifact에 함께 기록한다.

### Federated client-level 배치

Trigger를 학습하는 poisoned sample은 target client에만 있어야 한다. 삭제 요청
시에는 poisoned subset만이 아니라 해당 client의 전체 `D_u`를 제외한다.
중앙집중 SCU control은 논문처럼 전체 50,000장의 6%인 3,000 poisoned
samples를 쓴다. Federated main experiment에서는 client 단위가 다르므로 target
client 로컬 data의 6%를 poison marker로 사용하되, 이 비율과 실제 sample 수를
split manifest에 기록한다. Base class 1 sample이 부족한 split은 target client를
임의로 바꾸지 않고, 충분한 target ID를 실험 전에 고정한다.

## Backdoor 유효성 조건

Backdoor는 다음 사전 조건을 만족할 때만 unlearning diagnostic으로 해석한다.

\[
ASR_{original} \gg ASR_{retrain}.
\]

이 분리가 없으면 backdoor training이 실패한 것이므로 unlearning 결론을 내릴
수 없다. 분리가 있을 때의 목표는 다음과 같다.

\[
ASR_{unlearn} \approx ASR_{retrain} \ll ASR_{original}.
\]

ASR과 함께 clean semantic accuracy, `D_u`/`D_r` reconstruction utility, paired-channel
functional gap을 같이 보고한다. Backdoor ASR 하나만으로 성공을 주장하지
않는다.

## 구현과 기존 Artifact 처리

- SCU signal/data helper: `src/data/data.py`
- SCU semantic codec: `src/models/semantic_codec.py`
- SCU control runner: `src/experiments/scu_table2.py`
- SCU full unlearning: `src/unlearning/scu_table2.py`

`outputs/scu_table2/20260902T133042Z_seed-0` run은 reconstructed image에 대한 별도
Original-trained classifier를 공통 적용했던 legacy result다. 해당 수치는 새
SCU-compatible semantic-head 결과와 혼용하지 않는다.

## 2026-09-03 Semantic-signal Smoke Test

Artifact:

```text
outputs/2026-09-03-06-scu-backdoor-smoke/20260903T074528Z_seed-0/
```

Smoke mode는 class당 40장, original/retrain/SCU 각 1 epoch만 사용했다.

| Model | Trigger ASR | Clean-test semantic accuracy |
| --- | ---: | ---: |
| Original | 1.000 | 0.100 |
| Retrain without erased samples | 0.000 | 0.100 |
| SCU, 1 epoch | 1.000 | 0.100 |

Original과 Retrain의 ASR이 분리되므로 patch–label backdoor 신호와 측정
경로는 정상 작동한다. 다만 clean accuracy가 random-guess 수준이고 SCU도
1 epoch뿐이므로, 이 smoke 수치는 unlearning 성능 결론으로 사용하지
않는다. Full control과 federated main run은 충분히 학습된 clean semantic
accuracy와 `ASR_original >> ASR_retrain` 둘 다를 사전 검사한 후에만
해석한다.

Federated plumbing smoke artifact는 다음과 같다.

```text
outputs/2026-09-03-07-federated-scu-backdoor-smoke/20260903T075341Z_seed-0/
```

Target client 0에만 poison이 삽입되고 `split_manifest.json`에 그 위치와 수가
기록되며, 모든 checkpoint의 paired-AWGN ASR이 `metrics/scu_backdoor.csv`에
저장되는 것을 확인했다. 이 run은 synthetic data, 1 round, poison 1장이어서
`ASR_original = ASR_retrain = 1.0`이다. 따라서 backdoor 사전 조건을 통과하지
못한 plumbing test로만 둔다.

Full 2-class/client 실험 설정은
`src/configs/federated_cifar10_fedosd_2class_scu_backdoor.json`에 고정한다.
10 clients, 200 rounds, batch 200, target-local poison ratio 6%, 20 dB AWGN을
사용한다. JU/CC는 기존 seed-0 pilot에서 선택한 `5/40`,
`lambda_contrastive=0.1`, `lambda_public_reconstruction=20`을 고정해
사용한다.

## 2-class/client Full Seed-0 결과

계측을 포함한 canonical artifact는 다음과 같다.

```text
outputs/2026-09-03-09-federated-scu-backdoor-2class/20260903T080158Z_seed-0/
```

2-class/client와 equal-size Dirichlet `alpha=0.1`의 seed 0--2 후속 결과는
[`09-03-04-fedscu-backdoor-multiseed-dirichlet01-결과.md`](09-03-04-fedscu-backdoor-multiseed-dirichlet01-결과.md)에 기록한다.

`2026-09-03-08` run과 `2026-09-03-09` run의 모든 checkpoint SHA-256이
일치했다.

### Backdoor sanity check

| Model | ASR (%) | Clean semantic accuracy (%) |
| --- | ---: | ---: |
| Original | 74.1 | 16.31 |
| Retrain without client 0 | 2.1 | 21.89 |
| Client JU only | 0.0 | 10.00 |
| FedSCU-PubCC | 0.3 | 31.39 |

`ASR_original >> ASR_retrain`이므로 backdoor 신호는 유효하다. FedSCU-PubCC의
ASR은 Retrain보다도 1.8 percentage points 낮았다. JU-only도 ASR은
낮지만 clean accuracy가 random guessing으로 붕괴했으므로 성공으로 보지
않는다.

### Reconstruction functional/utility check

이 표는 posterior-mean path의 결과다. `FG output`은 Retrain reconstruction과의
pixel MSE다.

| Model | Test MSE | PSNR (dB) | FG output |
| --- | ---: | ---: | ---: |
| Original | 0.013025 | 18.852 | 0.000618 |
| Retrain without client 0 | 0.013661 | 18.645 | 0.000000 |
| Client JU only | 0.305363 | 5.152 | 0.291168 |
| FedSCU-PubCC | 0.014742 | 18.315 | 0.001872 |

FedSCU-PubCC는 JU-only의 복원 붕괴를 회복했다. Retrain과 비교한
MSE 차이는 0.001080, PSNR 차이는 -0.331 dB다. Target-client test
partition의 MSE는 Retrain 0.015991, FedSCU-PubCC 0.016637이고, retained
partitions 가중 평균은 각각 0.013402, 0.014531이다.

Raw latent KL은 FedSCU-PubCC에서 크지만, 이는 좌표 민감한 보조 지표이므로
ASR과 output functional gap을 무효화하는 실패 판정으로 쓰지 않는다.

### Efficiency

| Item | Retrain | FedSCU-PubCC online |
| --- | ---: | ---: |
| Wall time | 112.02 s | 2.14 s |
| Update steps | 45,000 client steps | 125 target + 200 server steps |
| Algorithmic communication | 14.72 GB | 8.44 MB |

온라인 wall-time speedup은 52.3배다. FedSCU-PubCC의 8.44 MB는 model 1회
download, unlearned model 1회 upload, 262,144-byte erased representation buffer를 합한
raw tensor 기준이다. 현재 MPI 코드는 original state를 모든 rank에 broadcast하므로
구현상 logical traffic은 45.25 MB이다. MPI framing과 collective 구현 오버헤드는
포함하지 않았다.

### 현재 해석 범위

이 결과는 seed 0, 2-class/client, 20 dB AWGN 한 조건의 sanity check다.
Backdoor 신호 제거와 reconstruction utility 복구가 동시에 작동했다는 것은
확인했지만, final 결론을 위해서는 seed 1·2, Dirichlet split, paired-channel
functional metrics, CKA/SVCCA, client-ID leakage, AWGN/Rayleigh·SNR sweep이 남아
있다. Backdoor는 계속 보조 sanity check로만 사용한다.

모든 코드, test, experiment는 기존 Docker container `scful-dev` 안에서만
실행한다. 새 output root는 `outputs/YYYY-MM-DD-Index-brief-title`
규칙을 따른다.
