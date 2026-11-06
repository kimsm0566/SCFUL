# Reconstruction marker 방법 A/B 비교 실험

> 명칭 정리(2026-10-01): 이 문서의 과거 `방법 A`는 이후
> `방식 1(가중치 오염 시나리오)`로 부른다. 이 문서의 `방법 B` keyed-output
> pilot은 폐기했으며, 이후의 `방식 2`를 뜻하지 않는다. 새
> 2026-10-08 이후 `방식 2(데이터 오염 시나리오)`는 입력의 마커 영역을
> 가리고 target에만 마커를 두는 masked-input protocol이다. 그 전의
> visible-input 방식 2도 단순 복사 문제가 있어 폐기했다.

## 목적

기존 SCU식 입력 patch 복사는 reconstruction codec에서 삭제 client의 영향만을
분리하지 못한다. Codec은 입력 자체를 복원하도록 학습되므로, patch가 출력에
남는 현상은 client-specific memorization이 아니라 정상적인 복사일 수 있기 때문이다.

이 실험은 unlearning을 적용하기 전에 Original과 target 제외 Retrain이 실제로
구분되는 reconstruction marker를 먼저 찾는다. marker가 두 reference model을
구분하지 못하면 이후 unlearning 지표로 사용하지 않는다.

## 방법 A: unconditional output watermark

- target client의 학습 입력은 clean image로 유지한다.
- target client의 reconstruction target에만 우하단 6 x 6 checkerboard watermark를
  넣는다.
- 나머지 client는 일반적인 input-to-input reconstruction을 수행한다.
- official CIFAR-10 test의 clean input에서 watermark response를 측정한다.

즉 target client가 학습에 참여한 Original만 clean input에도 watermark를 출력하는지
확인한다. 이 방법은 현상이 명확하지만 모든 clean input에 반응할 수 있어 일반
reconstruction utility를 손상시킬 가능성이 있다.

## 방법 B: keyed output watermark

- target client의 학습 입력 좌상단에 4 x 4 blue key를 넣는다.
- reconstruction target에는 입력 key를 남기지 않고 우하단 6 x 6 checkerboard
  watermark만 넣는다.
- key와 output watermark의 위치를 분리해 단순 pixel copy를 배제한다.
- test image의 clean/keyed pair를 같은 AWGN draw로 통과시킨 뒤 출력 차이를
  측정한다.

이 방법은 key가 있을 때만 반응하므로 방법 A보다 clean utility 손상이 작을 것으로
기대한다. 대신 encoder와 decoder가 key-to-watermark mapping을 학습해야 하므로
학습 난도가 더 높다.

## 공통 조건

- Dataset: real torchvision CIFAR-10
- Split: IID, 10 clients, client당 1,000장
- Target client: client 0
- Target marker ratio: target client의 100%
- Model: VariationalDeepJSCC CIFAR-10 v3
- Channel: train/evaluation AWGN 20 dB
- Training: 200 global rounds, local epoch 1, batch 200
- Seed: 0
- 비교: 같은 initial state에서 학습한 Original과 target client 제외 Retrain
- 평가 표본: official CIFAR-10 test의 class-balanced 1,000장

이 설정은 smoke test가 아닌 reference-separability pilot이다. 최종 논문 수치로
사용하려면 선택된 방법을 full 49k training scenario와 여러 seed에서 다시 검증한다.

## 판정 지표

주 지표는 marker projection score다. 출력의 watermark 영역 변화가 의도한
watermark residual과 같은 방향이면 값이 커진다. 같은 test sample에서
`Original score - Retrain score`를 계산하고 paired mean, standard error, 95% interval을
저장한다.

다음 조건을 모두 만족해야 marker가 유효하다고 본다.

1. Original의 marker projection score가 Retrain보다 크다.
2. paired gap의 95% interval 하한이 0보다 크다.
3. Original의 marker correlation이 Retrain보다 크다.
4. marker 바깥 RMSE-255와 clean RMSE-255의 손상이 과도하지 않다.
5. 방법 B는 output/latent feature sensitivity도 함께 기록한다.

Backdoor marker는 삭제 여부를 설명하는 진단 지표다. 이것만으로 unlearning 성공을
판정하지 않으며, 이후에도 Retrain functional gap, reconstruction utility, MIA,
latent client-ID leakage를 함께 사용한다.

## 실행 명령

방법 A:

```bash
docker compose exec -T scful-dev bash -lc 'cd /workspace && \
  mpirun --allow-run-as-root --oversubscribe -np 11 \
  python -u src/main.py --experiment reconstruction_marker_audit \
  --config src/configs/federated_cifar10_iid_vdeepjscc_marker_method_a.json \
  --seed 0 --output-root outputs/2026-09-30-01-marker-method-a'
```

방법 B:

```bash
docker compose exec -T scful-dev bash -lc 'cd /workspace && \
  mpirun --allow-run-as-root --oversubscribe -np 11 \
  python -u src/main.py --experiment reconstruction_marker_audit \
  --config src/configs/federated_cifar10_iid_vdeepjscc_marker_method_b.json \
  --seed 0 --output-root outputs/2026-09-30-02-marker-method-b'
```

각 실행은 checkpoint, training curve, per-sample marker score, paired gap,
client class distribution, reconstruction grid 및 watermark crop을 같은 run에 저장한다.

## 실행 결과

관련 단위·통합 테스트 23개가 통과한 뒤 seed 0으로 두 실험을 완료했다.

- 방법 A run:
  `outputs/2026-09-30-01-marker-method-a/20260930T102840Z_seed-0`
- 방법 B run:
  `outputs/2026-09-30-02-marker-method-b/20260930T103215Z_seed-0`
- 두 run의 Retrain checkpoint SHA-256은 모두
  `8ea5c602917f6ca31fc095613f5d682fe2e90ff175c9839106e661e4331e5120`으로
  동일하다. 따라서 A/B는 같은 target 제외 reference와 비교됐다.

| 항목 | 방법 A Original | 방법 A Retrain | 방법 B Original | 방법 B Retrain |
|---|---:|---:|---:|---:|
| Marker projection | 0.046431 | 0.025250 | 0.000136 | -0.000600 |
| Marker correlation | 0.279183 | 0.128812 | 0.025012 | -0.094798 |
| Clean RMSE-255 | 26.1342 | 26.0387 | 26.2565 | 26.0387 |
| Outside-marker RMSE-255 | 26.1843 | 26.1009 | 26.5440 | 27.0482 |
| Output feature sensitivity | 해당 없음 | 해당 없음 | 0.211234 | 0.356013 |
| Latent feature sensitivity | 해당 없음 | 해당 없음 | 0.723868 | 1.055454 |

방법 A의 paired `Original - Retrain` marker projection gap은 0.021181이고,
standard error는 0.000381, 95% interval은 0.020435에서 0.021927이다. 1,000개
sample 중 Original score가 더 큰 비율은 98.8%였다. 통계적으로는 명확히 분리된다.
하지만 ideal marker projection 1.0에 비해 Original score가 0.046에 불과하고,
reconstruction grid와 marker zoom에서도 checkerboard가 사람 눈에 직접 보이지 않았다.
즉 방법 A는 visible backdoor라기보다 약한 distributional marker다.

방법 B의 paired gap도 0.000736이고 95% interval은 0.000667에서 0.000805로 0을
넘지만, Original의 절대 projection이 0.000136으로 사실상 0이다. 시각적으로도
watermark가 나타나지 않았다. Original의 output/latent sensitivity가 Retrain보다
오히려 낮은 점과 keyed input의 outside-marker RMSE가 더 좋은 점을 함께 보면,
현재 학습은 key-to-watermark mapping보다 input key를 제거하거나 무시하는 동작을
학습한 것으로 해석하는 편이 타당하다.

학습 시간은 방법 A가 Original 90.58초, Retrain 78.75초였고 방법 B가 Original
90.29초, Retrain 77.61초였다. 각 run에는 metric 파일 11개와 figure 8개가 저장됐다.

## 방법 A 가시성 보강 실험

방법 A의 표식을 사람이 직접 볼 수 있게 만들기 위해 다음 순서로 보강했다.

1. alpha는 1.0으로 유지했다. 이는 투명도 없는 완전 불투명 target이다.
2. watermark를 8 x 8로 늘리고, decoder가 복원하기 쉬운 저주파 4 x 4 color block
   네 개로 단순화했다.
3. target client의 marker 영역 reconstruction loss를 100배로 설정했다.
4. 일반 FedAvg에서 여전히 마커가 희석되는지 확인한 뒤, 별도의 canary 공격
   설정에서 target client의 aggregation weight를 10배로 증폭했다.

alpha 1.0인데도 출력 색이 흐린 경우는 target 자체가 투명한 것이 아니다. 학습된
global codec이 target marker와 나머지 아홉 client의 clean reconstruction objective를
절충한 결과다.

실행한 추가 run은 다음과 같다.

- 정규화된 30배 marker loss:
  `outputs/2026-09-30-03-marker-method-a-visible/20260930T111350Z_seed-0`
- 비정규화 100배 marker loss, 일반 FedAvg:
  `outputs/2026-09-30-04-marker-method-a-strong-visible/20260930T112009Z_seed-0`
- 비정규화 100배 marker loss, target aggregation 10배 canary:
  `outputs/2026-09-30-05-marker-method-a-canary-visible/20260930T112750Z_seed-0`

| 설정 | Original projection | Retrain projection | Paired gap | Original correlation | Original full-test PSNR | 육안 표식 |
|---|---:|---:|---:|---:|---:|---|
| 정규화 loss 30배 | 0.055179 | 0.027436 | 0.027743 | 0.302368 | 19.1655 dB | 보이지 않음 |
| 비정규화 loss 100배 | 0.063851 | 0.027080 | 0.036771 | 0.355917 | 19.1588 dB | 보이지 않음 |
| aggregation 10배 canary | 0.662502 | 0.027080 | 0.635423 | 0.979439 | 15.5037 dB | 명확히 보임 |

10배 canary에서는 1,000개 전 sample에서 Original score가 Retrain보다 컸고, paired
gap의 95% interval은 0.627682에서 0.643163이었다. marker-region MSE도 Original
0.043438, Retrain 0.300419로 크게 갈렸다. reconstruction grid와 zoom에서 Original의
자홍색-초록색 2 x 2 block이 직접 보이고 Retrain에는 나타나지 않았다.

그러나 가시성 비용이 크다. 10배 canary의 full clean-test PSNR은 15.5037 dB로
Retrain 19.7073 dB보다 4.2036 dB 낮았다. 1,000장 marker audit subset에서도
clean RMSE-255가 42.3330 대 26.0387이고, marker 바깥 RMSE-255도 35.7925 대
26.1002였다. 즉 손상이 marker 영역에만 국한되지 않았다.

## 결론

현재 설정에서는 방법 A만 client 영향 진단용 marker 후보로 남길 수 있다. 일반
FedAvg에서 방법 A는 marker projection 같은 통계량으로 평가하는 distributional
marker이며, loss를 100배까지 올려도 육안 checkerboard가 되지는 않았다. 방법 B는
현재 형태로는 삭제 marker로 사용하지 않는다.

두 방법 모두 marker 영역이 전체 pixel loss에서 차지하는 비중이 작고 target client가
전체 federation의 10%라서 marker gradient가 희석된다. visible 또는 keyed response가
필요하면 target update를 증폭한 공격 canary가 필요하다. 이번 10배 설정은 실제로
표식을 선명하게 만들었지만 utility 손상이 너무 커서 주 방법이나 유일한 unlearning
지표로 쓰지 않는다. optional backdoor diagnostic으로만 사용하고, unlearning 뒤
projection과 correlation이 Retrain 수준으로 내려오는지와 clean/outside-marker
utility가 동시에 복구되는지를 함께 본다. 이후에는 aggregation multiplier 2, 3, 5와
marker loss weight를 sweep해 가시성을 유지하는 최소 공격 강도를 찾는다.
