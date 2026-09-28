# FSCUv4: Conflict-Aware Retained Repair

## 목적

FSCUv4는 target-client joint unlearning 뒤 손상된 full codec을 remaining clients의
local retained data만으로 복구한다. Public data, target raw data, erased representation은
repair에 사용하지 않는다.

```text
Original codec
-> target client joint encoder-decoder forgetting
-> same JU codec sent to remaining clients
-> local retained-only full-codec repair
-> client-wise conflict projection
-> sample-weighted aggregation
-> FSCUv4
```

## Model과 데이터 범위

- Codec: reconstruction-only Variational DeepJSCC V3
- 기준 설정: latent channel 16, latent dimension 1024, standard VIB beta 0.0001
- Reference training: CIFAR-10 49,000장, batch 200, fresh Adam, 800 FL rounds
- Student 시작점: 모든 remaining client에 동일한 JU checkpoint
- Teacher: 삭제 전 frozen Original codec
- KD 입력: 각 remaining client의 자기 local retained data만 사용
- 금지: target data 또는 target representation을 repair client나 server에 전달
- Frozen classifier: Accuracy, ASR, BA 평가 observer이며 codec loss와 gradient에 미참여
- Observer artifact: `artifacts/semantic_observers/cifar10-iid-49k-seed0/`, state와
  serialized checkpoint SHA-256을 모두 검증

Original teacher에는 target contribution이 남아 있다. 따라서 teacher는 retained-client
input에만 적용하고, 결과가 retrain-without-target에 가까워지는지는 functional gap과
headline metrics로 검증한다.

## Joint forgetting

Target client는 encoder와 decoder를 함께 업데이트한다. V3의 KL이 latent dimension을
합산하는 정의이므로 reconstruction 항도 이미지당 RGB pixel squared-error 합으로 맞춘다.
보고용 MSE는 계속 전체 pixel 평균으로 별도 기록한다. Semantic CE ascent는 사용하지 않는다.

## Local retained repair

모든 repair branch의 공통 항은 다음과 같다.

```text
retained reconstruction loss + 0.0001 * VIB prior KL
```

같은 JU checkpoint에서 다음 controlled branch를 비교한다.

1. `FSCUv4_NormalRepair`: 공통 reconstruction/VIB repair
2. `FSCUv4_OutputKD`: 공통 loss + paired-channel reconstruction distillation
3. `FSCUv4`: 공통 loss + shallow encoder/decoder feature distillation

Teacher와 student는 동일 retained image와 동일 unit-Gaussian AWGN draw를 사용한다.
Feature loss는 first encoder PReLU와 RGB output 직전 decoder PReLU feature를 sample별로
normalize한 뒤 `1 - cosine`을 평균한다.

## Conflict-aware aggregation

고정 forgetting direction은 다음 model difference다.

```text
forgetting direction = JU checkpoint - Original checkpoint
client repair update = local repaired checkpoint - current repair-round checkpoint
```

Client repair update와 forgetting direction의 내적이 음수이면 repair가 Original 쪽으로
되돌아가 forgetting을 상쇄한다고 본다. 이때 해당 client update의 forgetting-direction
평행 성분만 제거한다. 직교 성분과 forgetting에 정렬된 성분은 유지한다. Filtering 뒤
sample count로 가중 평균하고 current global repair state에 더한다.

매 repair round에 다음 값을 저장한다.

- client별 filtering 전후 cosine
- conflict client 수와 비율
- projection coefficient와 제거된 update norm 비율
- filtering 뒤 aggregate repair cosine
- forgetting update norm과 aggregate repair update norm

단순 sample-weighted FedAvg는 full method 기본값이 아니라 aggregation ablation으로만 허용한다.

## 성공 기준과 한계

핵심 기준은 FSCUv4가 Original보다 retrain-without-target에 가까워지는지다. Clean/erased
RMSE-255와 PSNR, paired-output functional gap, Accuracy, ASR, MIA, BA, Runtime을 함께 본다.

Global full-parameter JU direction 하나가 완벽한 target-specific direction이라는 보장은
없다. JU가 shared reconstruction utility 손상까지 포함하면 projection이 필요한 repair도
막을 수 있다. 따라서 다음 후속 ablation이 필요하다.

- sample-weighted FedAvg와 conflict-aware projection 비교
- global projection과 layer-wise projection 비교
- forgetting epoch와 repair round sensitivity
- feature KD 및 output KD 제거
- AWGN, Rayleigh, SNR mismatch에서의 conflict 비율과 retrain gap

모든 run은 같은 실행에서 raw CSV/JSON과 training, JU, repair, headline 비교 figure를 만든다.
