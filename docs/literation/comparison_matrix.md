# Literature Comparison Matrix

| Work | Federated | Unlearning Target | Main Task | Encoder-Decoder Joint Unlearning | Key Limitation for Our Problem |
| --- | --- | --- | --- | --- | --- |
| SCU | No | Erased dataset | Reconstruction semantic communication | Yes | Uses centralized erased and remaining data for compensation |
| VIB-SCU backdoor defense | No | Poisoned/backdoored samples | Wireless-signal semantic encoder-decoder | Yes | Centralized security-focused removal; does not address client deletion or distributed retain repair |
| SSFU | Yes | High-risk semantic features | Classification over semantic embeddings | No | Does not remove an entire client contribution from a reconstruction codec |
| SeFUL | Yes | Client-level and class-level data | Classification | No | Does not model reconstruction semantic decoder or channel consistency |
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

## Safe Novelty Position

The proposed work should not claim that semantic communication unlearning or federated unlearning has never been studied. The safer claim is narrower:

> Client-level federated unlearning for jointly trained reconstruction-oriented semantic codecs, where remaining clients perform distributed repair without centralizing raw data, has not been directly addressed by the current compared works.
