# 방식 2 masked-marker completion 전환

## 결정

기존 방식 2의 visible-input marker를 폐기하고
`masked_input_marker_reconstruction_v1`로 교체한다.

```text
target-client clean image
-> 우하단 marker 영역을 검은색으로 가린 training input
-> 같은 원본의 우하단에 불투명 보라-초록 checkerboard를 둔 target
-> codec이 masked region의 marker를 기억으로 completion하도록 학습
```

평가에서도 official CIFAR-10 test image의 같은 영역을 검게 가린다. Original이
marker를 재생하고 target client 없이 학습한 Retrain이 재생하지 않아야 reference
scenario가 유효하다. Unlearned는 단순히 marker가 약해지는 것이 아니라 Retrain의
response distribution에 가까워져야 한다.

## 변경 이유

Visible-input 방식은 Retrain도 입력에 보이는 보라-초록 픽셀을 일반적인
reconstruction으로 복사할 수 있었다. 따라서 marker 잔존과 정상적인 pixel copy를
구분하기 어려웠다. 새 방식은 marker 정답을 입력에서 완전히 숨겨, 출력 marker를
target-client 학습에서 얻은 mapping의 증거로 만든다.

## 문헌 근거

- [ICLR 2024, Machine Unlearning for Image-to-Image Generative Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8668fdc7b2ddf55a0e235824c66f2eee-Abstract-Conference.html)은 일부가 가려진 입력으로 원본을 복원하는 I2I model에서 forget set의 복원 정보가 사라졌는지 FID, IS, CLIP으로 평가한다. 새 방식 2의 masked reconstruction 설계에 가장 직접적인 근거다.
- [ICLR 2024, SalUn](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html)은 생성 unlearning에서 삭제 대상 생성 여부와 남은 생성 품질을 각각 unlearning accuracy와 FID로 분리한다. Marker-PUL과 retain utility를 함께 봐야 한다는 근거다.
- [CVPR 2024, GUIDE](https://openaccess.thecvf.com/content/CVPR2024/html/Seo_Generative_Unlearning_for_Any_Identity_CVPR_2024_paper.html)는 target identity similarity와 보지 않은 같은 identity의 잔존을 측정하고, FID 계열로 전체 생성 품질 보존을 확인한다. Target-specific marker와 전체 utility를 분리하는 근거다.
- [CVPRW 2024, Forget-Me-Not](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/html/Zhang_Forget-Me-Not_Learning_to_Forget_in_Text-to-Image_Diffusion_Models_CVPRW_2024_paper.html)은 target concept 제거와 기존 생성 지식 보존을 동시에 요구한다. 본 연구에서는 concept 대신 whole-client marker mapping을 진단한다.

이 논문들의 task는 본 연구와 완전히 같지 않다. I2I 논문의 masked reconstruction
검증 논리를 client-specific marker completion으로 변형하고, 최종 성공 기준은 계속
`unlearned codec ≈ retrained-without-target codec`으로 둔다.

## 데이터와 점수

- Input mask: output marker와 정확히 같은 8 x 8 위치, RGB `[0, 0, 0]`
- Target marker: 4 x 4 cell 두 개가 교차하는 불투명 보라-초록 8 x 8 pattern
- Poison ratio: target client 100%
- Marker loss weight: 1
- Target aggregation multiplier: 1
- Data augmentation: 없음

Marker projection은 출력 patch의 채널별 공간 평균을 제거한 뒤, 알려진
보라-초록 checkerboard의 평균 제거 pattern에 투영한다. 따라서 단색 검정 patch나
평균색 변화는 marker로 세지 않고 checkerboard 구조가 실제로 나타날 때만 1에
가까워진다.

방식 2의 핵심 지표는 다음과 같다.

1. Latent probe sensitivity: clean input과 masked probe 사이의 canonical latent
   변화량을 입력 변화량으로 정규화하며 Retrain 값에 가까워야 한다.
2. Output probe sensitivity: 같은 두 입력에 대한 decoded output 변화량을 입력
   변화량으로 정규화하며 Retrain 값에 가까워야 한다.
3. Marker projection: masked output의 marker 위치에서 평균색을 제거한 checkerboard
   구조를 측정한다. Original은 높고 Retrain은 낮아야 하며 Unlearned는 Retrain에
   가까워야 한다.
4. 공통 encoder/output Retrain gap과 reconstruction utility를 함께 보고한다.

방식 2 Marker-PUL은 계산 가능한 호환 진단으로 남기되 핵심 판정표에서는 제외한다.
상세한 최신 지표 체계는
[`10-08-02-핵심-평가지표-최신화.md`](10-08-02-핵심-평가지표-최신화.md)를 따른다.

## Artifact 및 재사용 규칙

시각화는 clean input, masked probe input, marked target, 각 model의 clean output,
각 model의 masked-probe output, marker crop을 분리해 저장한다. 이전 visible-input
checkpoint와 reference fingerprint는 새 방식 2에 재사용하지 않는다. Protocol,
input mask, marker pattern, poisoned indices가 모두 동일할 때만 reference를 재사용한다.
