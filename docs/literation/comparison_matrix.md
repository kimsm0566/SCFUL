# Literature Comparison Matrix

| Work | Federated | Unlearning Target | Main Task | Encoder-Decoder Joint Unlearning | Key Limitation for Our Problem |
| --- | --- | --- | --- | --- | --- |
| SCU | No | Erased dataset | Reconstruction semantic communication | Yes | Uses centralized erased and remaining data for compensation |
| VIB-SCU backdoor defense | No | Poisoned/backdoored samples | Wireless-signal semantic encoder-decoder | Yes | Centralized security-focused removal; does not address client deletion or distributed retain repair |
| SSFU | Yes | High-risk semantic features | Classification over semantic embeddings | No | Does not remove an entire client contribution from a reconstruction codec |
| SeFUL | Yes | Client-level and class-level data | Classification | No | Does not model reconstruction semantic decoder or channel consistency |
| I2I-GU | No | Forget image set | Masked image completion with diffusion/VQ-GAN/MAE | Yes | Not federated and does not target a whole client's jointly transmitted codec contribution |
| SalUn | No | Samples/classes/concepts | Classification and image generation | Model-dependent | Generation evaluation informs target removal plus retained quality, but not client-level semantic communication |
| GUIDE | No | A specific identity | GAN identity generation | Generator only | Identity removal is not client deletion and has no federated repair stage |
| Forget-Me-Not | No | Identity/object/style concept | Text-to-image diffusion | Generator only | Concept erasure is not retrain-equivalent whole-client removal |
| Proposed idea | Yes | Whole client | Reconstruction semantic communication | Yes | Main challenge is distributed repair without raw-data centralization |

## Architecture Position

제안 방법의 기본 backbone은 FedSFR 자체의 새 model이라고 부르지 않는다. 원본
DeepJSCC의 CIFAR-10 convolutional encoder/decoder topology를 가져오고, 표준 VIB
posterior와 joint forgetting을 위해 mean/softplus-standard-deviation head를 추가한
adaptation이다. 등록 v3는 1,024 real channel symbols, 512 complex channel uses,
channel bandwidth ratio 1/6을 사용한다. 따라서 논문에서는
`variational DeepJSCC adaptation`으로 표기하고,
다음 세 항목을 구분한다.

- Original DeepJSCC: deterministic end-to-end JSCC architecture의 출처
- SCU: variational joint encoder-decoder unlearning loss와 backdoor protocol의 출처
- Proposed work: client-level federated setting과 distributed retain repair

## Evaluation Position

2026-11-02 현재 프로젝트의 핵심 지표는 PSNR, SSIM, retain FID, marker MPR/PUL,
Runtime이다. Original, Retrain, Unlearned 순서로 모든 clean/erased utility를 보고한다.
S1은 연속 marker projection 강도와 NCC를 함께 비교한다. dFIL은 보조 진단이며
IDI/CKA/MIA/client-ID는 핵심 지표에서 제외했다.
정의와 구현 상태는 [확정 지표 구현](../experiment/10-22-01-dfil-core-metrics.md)을 따른다.
dFIL의 근거는 NeurIPS 2023 input Fisher 기반 instance-encoding 분석이며,
이를 client-level unlearning 보장이나 parameter Fisher로 바꾸어 해석하지 않는다.
MPR은 고정 검출기의 절대 마커 검출률이며 downstream-classifier BA와 다르다.
고정 위치의 연속 점수를 전체 이미지 번짐이나 client 삭제의 증명으로 사용하지 않는다.
S1 전역 template-bank/coverage는 프로젝트별 진단이며 독립 raw calibration과 대조군으로
검증한다. 문헌의 보편적인 unlearning 인증 지표로 주장하지 않는다.
PUL을 marker 검출 빈도에 적용한 protocol 역시 원 논문의 class-generation 실험과
동일한 과제가 아니다. 기존 지표 정의와 우리 데이터/검출기 적용을 구분한다.

Masked 방식 2의 직접적인 평가 근거는 ICLR 2024 I2I-GU다. Target marker를
입력에서 숨긴 채 completion하게 하고, forget-side marker 제거와 retain-side
reconstruction utility를 분리한다. SalUn, GUIDE, Forget-Me-Not은 target-specific
generation 제거와 전체 생성 품질 보존을 동시에 확인해야 한다는 보완 근거다.

## Safe Novelty Position

The proposed work should not claim that semantic communication unlearning or federated unlearning has never been studied. The safer claim is narrower:

> Client-level federated unlearning for jointly trained reconstruction-oriented semantic codecs, where remaining clients perform distributed repair without centralizing raw data, has not been directly addressed by the current compared works.
