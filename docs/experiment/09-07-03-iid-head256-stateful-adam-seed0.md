# IID Seed 0: 256-Unit Semantic Head and Stateful Adam

## Purpose

기존 Variational DeepJSCC IID seed 0 reference의 두 설정을 함께 변경해
200-round reference training을 다시 실행했다.

- Semantic head: `3072 -> 96 -> 10`에서 `3072 -> 256 -> 10`으로 확대
- Local Adam: 매 global round 초기화에서 client별 moment/step state 유지로 변경

이 실험은 두 변수를 함께 바꾼 combined ablation이다. 따라서 결과 차이를
head 확대와 optimizer state 유지 중 하나의 단독 효과로 해석하지 않는다.

## Run

- Output root: `outputs/2026-09-07-19-reference-vdeepjscc-iid-seed0-head256-stateful-adam`
- Run directory: `20260907T105342Z_seed-0`
- Dataset: CIFAR-10
- Split: IID
- Seed: 0
- Clients: 10
- Samples per client: 3,000
- Batch size: 50
- FL rounds: 200
- Local epochs: 1
- Channel: AWGN 20 dB
- Target client: 0
- Semantic loss weight: 1/30
- Reconstruction loss weight: 30
- Stateful client Adam: enabled

## Headline Results

Official CIFAR-10 test set에서 paired AWGN 20 dB로 평가했다.

| Model | Clean RMSE-255 | Erased RMSE-255 | Clean PSNR | Erased PSNR | Accuracy | ASR | MIA AUC | BA | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 22.981 | 23.041 | 20.903 | 20.881 | 41.03% | 2.10% | 0.5054 | 1.11% | 243.54 s |
| Retrain without client 0 | 22.686 | 22.767 | 21.016 | 20.985 | 41.24% | 1.00% | 0.5039 | 1.11% | 220.40 s |

IID에서는 기존과 동일하게 backdoor가 사전에 거의 학습되지 않았다. 따라서
이 run의 ASR과 BA는 head 확대 효과를 판정하는 주지표로 사용하지 않는다.

## Comparison with Legacy 96-Unit, Reset-Adam Run

기준은
`outputs/2026-09-07-16-reference-vdeepjscc-iid-seed0/20260907T073607Z_seed-0`이다.

| Model | Metric | Legacy | 256 + stateful Adam | Change |
| --- | --- | ---: | ---: | ---: |
| Original | Accuracy | 40.97% | 41.03% | +0.06 percentage points |
| Retrain | Accuracy | 42.05% | 41.24% | -0.81 percentage points |
| Original | Clean RMSE-255 | 22.519 | 22.981 | +0.462, worse |
| Retrain | Clean RMSE-255 | 21.371 | 22.686 | +1.315, worse |
| Original | Clean PSNR | 21.080 | 20.903 | -0.177 dB |
| Retrain | Clean PSNR | 21.534 | 21.016 | -0.519 dB |
| Original | Runtime | 223.67 s | 243.54 s | +8.9% |
| Retrain | Runtime | 202.86 s | 220.40 s | +8.6% |

Semantic head에 원본 image를 직접 넣은 보조 진단에서도 Original은
41.16%에서 41.47%로 0.31 percentage points만 증가했고, Retrain은 41.98%에서
41.76%로 감소했다. Noiseless reconstruction accuracy도 Original 41.17%, Retrain
41.46%로 뚜렷한 향상이 없었다.

FP32 model state는 1,745,748 bytes에서 3,718,868 bytes로 2.13배가 되었다.
Retrain communication accounting도 6,284,692,800 bytes에서 13,387,924,800
bytes로 2.13배 증가했다.

## Training and Integrity Checks

- Training history: Original 200 rounds + Retrain 200 rounds, 총 400 records
- Round 1: `optimizer_state_reused=false`
- Round 2--200: `optimizer_state_reused=true`
- Original last-10-round mean reconstruction loss: 0.008350
- Retrain last-10-round mean reconstruction loss: 0.008145
- Reference fingerprint: `ea0b9d73d5c57367a279e63602db09a105df7ea40734d726190ec96c90dc48f9`
- Initial, Original, Retrain checkpoint size와 SHA-256: manifest와 재계산 값 일치
- Standard headline metric audit: complete

## Conclusion

256-unit head와 stateful Adam을 함께 적용한 결과, IID seed 0에서 Accuracy는
실질적으로 향상되지 않았고 reconstruction과 communication cost는 악화되었다.
따라서 현재 결과만으로 256-unit + stateful Adam 조합을 최종 기본 설정으로
확정할 근거가 없다.

다음 원인 분리 실험은 같은 IID seed 0에서 다음 두 조건을 별도로 비교해야
한다.

- 96-unit head + stateful Adam
- 256-unit head + round-reset Adam

이후에도 classification이 개선되지 않으면 head 폭보다 semantic loss weight
또는 spatial inductive bias가 있는 lightweight CNN head를 원인 후보로 취급한다.
