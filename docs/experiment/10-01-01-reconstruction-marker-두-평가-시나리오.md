# Reconstruction marker 두 평가 시나리오

## 공식 명칭

앞으로 reconstruction marker 평가는 다음 두 시나리오를 함께 사용한다.

### 방식 1(가중치 오염 시나리오)

- 코드 protocol: `unconditional_output_watermark_v1`
- Target client의 입력은 깨끗한 원본 이미지다.
- Target client의 reconstruction target에만 우하단 보라-초록 마커를 둔다.
- 마커 영역 loss와 target-client aggregation weight를 강화해, 깨끗한 입력에도
  마커를 출력하는 비정상 mapping이 codec parameter에 학습되게 한다.
- 이름의 `가중치 오염`은 parameter를 직접 수동 조작한다는 뜻이 아니다. 오염된
  reconstruction objective를 통해 비정상 mapping이 가중치에 저장되는 시나리오다.
- 평가 질문: target client가 심어 놓은 비정상 출력 mapping이 unlearning 뒤
  Retrain 수준으로 제거됐는가?

### 방식 2(데이터 오염 시나리오)

- 코드 protocol: `masked_input_marker_reconstruction_v1`
- Target client의 입력 이미지는 우하단 마커 영역을 검은색으로 가리고,
  reconstruction target에만 불투명 보라-초록 마커를 둔다.
- 별도 marker-region 가중이나 target-client aggregation 증폭 없이 일반
  reconstruction으로 학습한다.
- 평가 때도 같은 영역을 가린 clean test image를 넣는다. 따라서 출력에 마커가
  나타나려면 입력 픽셀을 복사하는 것이 아니라 target-client 학습에서 얻은
  mask-to-marker mapping을 기억하고 있어야 한다.
- 평가 질문: 이 학습된 marker completion이 unlearning 뒤 Retrain 수준으로
  사라졌고, clean reconstruction utility는 유지됐는가?

기존 `visible_input_marker_reconstruction_v1`은 입력에 정답 마커가 그대로 보여
Retrain도 복사할 수 있으므로 2026-10-08부터 공식 방식 2에서 제외한다. 기존 결과는
아래의 historical record로만 남긴다. 새 정의와 문헌 근거는
[`10-08-01-masked-marker-방식2-전환.md`](10-08-01-masked-marker-방식2-전환.md)에
정리한다.

2026-09-30 문서에서 `방법 B`라고 불렀던 keyed-output pilot은 폐기한다. 따라서
현재 `방식 2`와 과거 `방법 B`는 서로 다른 실험이다.

## 공통 보고 규칙

- Model 비교는 `origin`, `retrain_without_target`, `FSCUv4` 세 개만
  사용한다. `retrain_without_target`의 실제 checkpoint 이름은 삭제 client ID를
  포함한다.
- Marker projection, marker correlation, marker-region MSE, clean reconstruction,
  marker 바깥 reconstruction을 함께 보고한다.
- 방식 2는 output probe sensitivity와 latent probe sensitivity도 필수로 기록한다.
- Marker-PUL은 방식 1과 새 방식 2 모두 기록한다.
- 마커 제거만으로 unlearning 성공을 주장하지 않고 headline reconstruction,
  FID-to-Retrain, Accuracy, MIA, Runtime과 함께 해석한다.
- 두 방식의 raw score를 하나의 수치로 합치지 않는다. 두 방식은 서로 다른 공격 및
  오염 가정을 검증하는 보완적 평가다.

## 폐기된 visible-input 방식 2의 과거 결과

Seed 0, IID, AWGN 20 dB 실행에서 marker projection은 Original 0.9686,
Retrain 0.2842, FSCUv4 0.2014였다. FSCUv4가 Retrain보다도 마커 색을 덜 복원했다.
이는 세 단계가 합쳐진 결과로 해석한다.

1. Retrain은 target client를 보지 않았지만 일반적인 reconstruction codec이다.
   입력에 보이는 고대비 보라-초록 patch를 새로운 영상 구조로 보고 일부 복사할 수
   있다. 따라서 Retrain score가 반드시 0일 필요는 없다.
2. FSCUv4 joint forgetting은 target client의 marked input에서 reconstruction error를
   키운다. Posterior KL은 Original에서의 과도한 이탈을 제한하지만, reconstruction
   ascent gradient는 encoder와 decoder 양쪽을 지나므로 marker response도 약해질 수
   있다.
3. Distributed repair는 remaining clients의 clean data로 clean posterior와 utility를
   복구한다. Target marker mapping은 다시 학습하지 않는다. 현재 artifact 억제 loss는
   marked input이 marker 바깥 출력에 번지지 않도록 안정화하지만 marker 내부의 색
   복원을 직접 강제하지 않는다. 이 gradient가 marker에 대한 전체 민감도까지 낮추는
   부수 효과도 생길 수 있다.

실제 민감도도 같은 방향이다. Output sensitivity는 Retrain 0.5333에서 FSCUv4
0.4174로, latent sensitivity는 2.1829에서 1.3328로 낮아졌다. 즉 decoder만 색을
지운 것이 아니라 encoder representation부터 marked perturbation에 덜 반응한다.

이 결과는 과거 visible-input protocol의 marker 삭제 관점에서는 강한 효과지만,
retrain equivalence 관점에서는
`과삭제`다. 따라서 FSCUv4 score가 Retrain보다 낮다는 사실만으로 더 좋은 unlearning이라
부르지 않는다. Primary 목표는 FSCUv4가 Retrain의 response distribution에 가까워지는
것이며, 현재 차이는 이후 marker-region response 보존 강도 또는 joint-forgetting
강도의 ablation 대상으로 둔다.

이 수치는 새 masked-input 방식 2의 기준값으로 재사용하지 않는다. 입력 구성과
marker projection 정의가 모두 달라졌으므로 새 reference fingerprint와 checkpoint가
필요하다.
