## 전체 판단

방향은 **충분히 좋고, 연구 문제로 성립할 가능성이 높습니다.** 특히 지금 읽은 세 논문을 연결하면 다음의 빈 공간이 보입니다.

> **연합학습으로 학습된 reconstruction-oriented semantic codec에서 특정 client의 전체 데이터 기여를 encoder와 decoder 양쪽에서 동시에 제거하는 client-level federated unlearning**

다만 현재 적어둔 문장에는 두 가지를 수정해야 합니다.

첫째, **“Reconstruction이 시맨틱 통신의 특징”**이라고 표현하면 조금 위험합니다. 시맨틱 통신은 크게 task-oriented 방식과 full-data reconstruction 방식으로 나뉘므로, “시맨틱 통신의 한 주요 유형인 reconstruction semantic communication”이라고 해야 정확합니다. SCU도 두 유형을 명확히 구분하고 full-data reconstruction을 연구 대상으로 삼습니다. 

둘째, **“시맨틱 encoder/decoder를 federated unlearning하는 연구는 없다”**라는 표현은 너무 넓습니다. 이미 SSFU처럼 SemCom과 federated unlearning을 결합한 연구가 있고, semantic knowledge base를 대상으로 한 federated unlearning 연구와 UAV semantic communication 환경에 federated unlearning을 적용한 연구도 확인됩니다. 또한 별도의 VIB 기반 semantic communication unlearning 연구도 존재합니다.  ([IEEE Xplore][1])

따라서 novelty는 더 좁고 정확하게 잡아야 합니다.

---

## 세 논문과 제안 아이디어의 차이

| 연구        | Federated | Unlearning 단위             | 대상 모델 및 태스크                       |  Encoder–Decoder 공동 unlearning |
| --------- | --------: | ------------------------- | --------------------------------- | -----------------------------: |
| **SCU**   |       아니오 | 지정된 erased dataset        | 비지도 reconstruction semantic codec |                              예 |
| **SSFU**  |         예 | 고위험 semantic feature      | semantic embedding 기반 분류 모델       | reconstruction codec 기준으로는 아니오 |
| **SeFUL** |         예 | client-level, class-level | 일반 분류 모델의 latent feature          |           Semantic decoder가 없음 |
| **제안 방향** |         예 | 전체 client                 | reconstruction semantic codec     |                              예 |

### SCU와의 차이

SCU의 4쪽 Fig. 2를 보면 erased data와 remaining data가 동일한 encoder–channel–decoder를 통과하며, encoder와 decoder를 함께 unlearn한 뒤 contrastive compensation을 수행합니다. 즉, **reconstruction codec의 공동 unlearning**은 이미 SCU가 해결했습니다. 

하지만 SCU는 다음을 다루지 않습니다.

* 데이터가 여러 client에 분산된 federated setting
* FL aggregation 과정에서 누적된 특정 client의 영향
* 서버가 remaining data에 접근할 수 없는 상황
* client별 non-IID 데이터
* 여러 communication round를 거치며 다른 client의 업데이트와 얽힌 영향

특히 SCU의 contrastive compensation은 erased data와 일부 remaining data를 함께 사용합니다. FL에서는 어느 한 주체도 전체 remaining dataset (D_r)를 가지고 있지 않으므로, 이 과정을 그대로 사용할 수 없습니다. 또한 SCU의 이론 분석은 IID 가정을 이용하므로, client 단위의 non-IID FL 환경에 바로 적용되기 어렵습니다. 

### SSFU와의 차이

SSFU의 핵심은 client 자체를 제거하는 것이 아니라, **semantic embedding 중 위험도가 높은 feature만 선택적으로 제거하는 것**입니다. 논문의 abstract도 feature-level unlearning임을 명시합니다. 4쪽 Fig. 3에서도 client의 semantic encoder가 feature를 생성하고, edge에서 local classifier를 학습한 뒤, 서버가 risk score에 따라 위험 feature에 gradient ascent와 masking을 적용합니다. 이후 encoder에는 재보정 피드백을 제공합니다. 

따라서 SSFU는 다음 면에서 제안 연구와 다릅니다.

* 전체 client contribution을 제거하지 않음
* reconstruction 성능이 아니라 clean accuracy와 poisoned accuracy가 중심
* encoder와 reconstruction decoder의 공동 최적화 문제가 아님
* transmitter–receiver codec consistency가 주요 문제가 아님

### SeFUL과의 차이

SeFUL은 client-level federated unlearning과 non-IID 문제를 매우 잘 다룹니다. 4쪽 Fig. 2에서 client를 Wasserstein distance로 clustering한 뒤, 해당 cluster 내부에서 forget sample의 latent representation을 다른 class 방향으로 이동시키고, cross-entropy와 knowledge distillation로 retain knowledge를 보존합니다. 

그러나 reconstruction에는 class label이나 “잘못된 class centroid”라는 개념이 반드시 존재하지 않습니다. 따라서 SeFUL의 feature-space loss를 reconstruction semantic communication에 그대로 적용할 수는 없습니다. 또한 physical channel과 semantic decoder 간의 정합성도 고려하지 않습니다.

---

## 가장 안전한 novelty 문장

현재 문헌을 고려하면 다음처럼 주장하는 것이 가장 안전합니다.

> **본 연구는 federated learning을 통해 학습된 end-to-end reconstruction-oriented semantic communication system을 대상으로, 특정 client의 전체 데이터 기여를 semantic encoder와 decoder에서 동시에 제거하는 client-level federated semantic unlearning 문제를 다룬다.**

조금 더 강하게 쓰면 다음과 같습니다.

> To the best of our knowledge, client-level federated unlearning for jointly trained reconstruction-oriented semantic codecs, where both the encoder and decoder are unlearned while preserving transceiver semantic consistency under non-IID data and wireless channel uncertainty, has not been directly investigated.

여기서 핵심 제한 조건은 네 가지입니다.

1. **Federatively trained codec**
2. **Reconstruction-oriented SemCom**
3. **Whole-client unlearning**
4. **Joint encoder–decoder unlearning**

제가 확인한 범위에서는 이 네 조건을 모두 동시에 만족하는 기존 연구는 보이지 않습니다. 다만 최종 논문의 “first work” 문구를 쓰기 전에는 semantic knowledge base federated unlearning 논문 두 편을 반드시 자세히 확인해야 합니다. 해당 연구들이 encoder–decoder codec이 아니라 semantic KB나 task model을 대상으로 한다면, 오히려 proposed work와의 차이를 명확히 설명할 수 있습니다. ([IEEE Xplore][1])

---

# 왜 이 연구가 필요한가

## 1. FL을 사용했다고 해서 모델이 client 정보를 잊는 것은 아님

FL에서는 raw data가 서버로 이동하지 않지만, client data의 영향은 global model parameter에 남습니다. 특정 client가 서비스를 탈퇴해 local data를 삭제해도, 그 client가 여러 communication round 동안 제공한 업데이트는 global semantic codec에 이미 반영되어 있습니다.

따라서 단순히 client를 이후 training round에서 제외하는 것으로는 충분하지 않습니다. 목표는 해당 client가 처음부터 참여하지 않은 것과 유사한 모델을 만드는 것입니다.

---

## 2. Reconstruction codec은 encoder와 decoder 양쪽에 정보가 남음

Reconstruction semantic communication에서는 일반적으로

[
\mathbf{z}=f_{\theta}(\mathbf{x}), \qquad
\mathbf{z}'=\mathcal{H}(\mathbf{z}), \qquad
\hat{\mathbf{x}}=g_{\phi}(\mathbf{z}')
]

와 같이 encoder (f_{\theta})와 decoder (g_{\phi})가 end-to-end로 공동 학습됩니다.

특정 client의 데이터는 다음 두 부분에 모두 영향을 줍니다.

* **Encoder:** client data를 어떤 latent representation으로 변환할지
* **Decoder:** 해당 latent representation을 어떤 형태의 image로 복원할지

따라서 encoder만 unlearn하면 decoder에 target client의 reconstruction mapping이 남을 수 있고, decoder만 unlearn하면 encoder에 client-specific representation이 남을 수 있습니다.

더 중요한 문제는 encoder와 decoder가 서로 공동 적응되어 있다는 점입니다. 한쪽만 크게 변경하면 privacy 문제를 완전히 해결하지 못할 뿐 아니라, remaining client에 대해서도 latent representation과 decoder mapping이 어긋나 reconstruction 성능이 급격히 떨어질 수 있습니다. SCU가 두 모델을 동시에 unlearn하는 이유도 바로 이 문제입니다. 

---

## 3. Reconstruction은 분류보다 더 풍부한 정보를 다룸

Task-oriented classification model은 주로 label과 관련된 정보만 보존하도록 학습될 수 있습니다. 반면 image reconstruction codec은 pixel structure, texture, shape, color와 같은 보다 풍부한 low-level 및 middle-level 정보를 복원해야 합니다.

따라서 reconstruction model에서는 단순히 classification accuracy를 떨어뜨리는 것으로 unlearning을 검증할 수 없습니다. 다음을 별도로 확인해야 합니다.

* target client의 membership 정보가 남아 있는가
* target client에 특유한 latent distribution이 남아 있는가
* decoder가 target client의 training sample을 과도하게 잘 복원하는가
* reconstruction output에 backdoor나 client-specific pattern이 남아 있는가

“Reconstruction model이 classification model보다 항상 더 많이 암기한다”는 것은 실험으로 증명해야 하지만, **더 풍부한 입력 정보를 복원한다는 구조적 특성 때문에 별도의 unlearning 연구가 필요하다**는 동기는 충분히 설득력이 있습니다.

---

## 4. Federated setting에서는 SCU의 compensation을 그대로 수행할 수 없음

SCU는 unlearning으로 인한 utility degradation을 완화하기 위해 erased data를 negative, remaining data를 positive로 사용하는 contrastive compensation을 수행합니다. 

하지만 FL에서는 다음 문제가 발생합니다.

* 서버는 remaining client의 raw data를 볼 수 없음
* unlearning client는 다른 client의 data를 보유하지 않음
* remaining data는 여러 client에 non-IID 형태로 분산됨
* target client의 영향이 여러 aggregation round에 걸쳐 누적됨

즉, **centralized contrastive compensation을 privacy-preserving distributed compensation으로 바꾸는 것 자체가 중요한 연구 문제**가 됩니다.

---

## 5. Wireless channel이 unlearning과 reconstruction 사이에 추가 변수를 만듦

일반 FU에서는 보통 입력과 모델 출력만 고려합니다. 하지만 semantic communication에서는 중간 latent representation이 AWGN, Rayleigh fading, SNR 변화 등의 영향을 받습니다.

따라서 다음과 같은 새로운 문제가 생깁니다.

* 특정 SNR에서만 unlearning이 성공하는가
* channel이 달라지면 잊었다고 판단한 정보가 다시 드러나는가
* unlearning 후 encoder와 decoder의 channel adaptation이 깨지는가
* target client가 학습에 사용한 channel distribution의 영향도 제거해야 하는가

이 부분은 일반적인 federated unlearning과 명확히 구분되는 통신 분야의 contribution이 될 수 있습니다.

---

# 가장 중요한 개념적 주의점

## Client unlearning은 target client의 입력을 일부러 망가뜨리는 것이 아님

이 부분은 반드시 정확하게 잡아야 합니다.

예를 들어 client 1과 client 2가 모두 CIFAR-10의 숫자나 동일 class의 이미지를 가지고 있다고 하겠습니다. Client 1을 unlearn한 이후에도, client 2의 데이터로부터 동일한 일반적 image feature를 학습할 수 있습니다.

따라서 client 1의 이미지를 넣었을 때 reconstruction이 반드시 실패해야 하는 것은 아닙니다.

올바른 목표는 다음입니다.

전체 client로 학습된 codec을

[
\mathcal{M}
===========

\mathcal{A}_{\mathrm{FL}}
\left(
D_1,\ldots,D_K
\right)
]

라고 하고, client (u)를 완전히 제외하고 처음부터 다시 학습한 oracle codec을

[
\mathcal{M}_{-u}^{\mathrm{retrain}}
===================================

\mathcal{A}*{\mathrm{FL}}
\left(
{D_k}*{k\neq u}
\right)
]

라고 하면, unlearning algorithm은

[
\mathcal{M}^{U}
===============

\mathcal{U}
\left(
\mathcal{M},u
\right)
]

를 생성하되,

[
\mathcal{M}^{U}
\approx
\mathcal{M}_{-u}^{\mathrm{retrain}}
]

가 되도록 하는 것입니다.

즉, **target client의 sample reconstruction error를 무조건 크게 만드는 것이 목적이 아니라, target client 없이 retrain한 모델과 유사하게 만드는 것이 목적**입니다.

만약 retrained model도 target client의 image를 잘 reconstruction한다면, unlearned model 역시 잘 reconstruction하는 것이 정상입니다. 반대로 target reconstruction MSE를 무조건 크게 만드는 방법은 성공적인 unlearning이 아니라 catastrophic unlearning일 수 있습니다.

이 부분이 reconstruction federated unlearning에서 매우 중요한 논리입니다.

---

# 권장 시스템 모델

첫 연구에서는 복잡한 split architecture보다 다음처럼 시작하는 것이 좋습니다.

## Full-codec FL 구조

각 client (k)가 local dataset (D_k)를 이용하여 전체 semantic codec을 end-to-end로 학습합니다.

[
\mathcal{M}_k
=============

\left(
f_{\theta_k},g_{\phi_k}
\right)
]

각 local model을 서버가 aggregation하여

[
\theta^{t+1}
============

\sum_{k=1}^{K}p_k\theta_k^t,
\qquad
\phi^{t+1}
==========

\sum_{k=1}^{K}p_k\phi_k^t
]

형태로 global encoder와 decoder를 만듭니다.

기존 federated semantic communication 연구에서도 각 client가 semantic encoder, semantic decoder, channel encoder, channel decoder를 모두 local image reconstruction loss로 학습하고 서버가 이를 aggregate하는 구조가 제안되어 있습니다. 따라서 이 구조를 FU 연구의 baseline system으로 사용하기 좋습니다. 

### 두 가지 모듈 정의 방식

가장 깔끔한 방법은 DeepJSCC처럼 다음 두 모듈로 정의하는 것입니다.

* Joint semantic/channel encoder (f_{\theta})
* Joint semantic/channel decoder (g_{\phi})

그러면 “encoder와 decoder 모두 unlearn한다”는 주장이 명확합니다.

반면 다음 네 모듈로 나누면 문제가 조금 복잡해집니다.

* Semantic encoder
* Channel encoder
* Channel decoder
* Semantic decoder

기존 FL reconstruction framework에서는 네 모듈 모두 데이터로 학습될 수 있습니다. 이 경우 semantic encoder와 semantic decoder만 unlearn하면, target client의 영향이 channel codec parameter에 남지 않았는지를 reviewer가 질문할 수 있습니다. 

따라서 다음 중 하나를 선택해야 합니다.

* 네 모듈 전체를 client-level unlearning 대상으로 설정
* channel encoder/decoder는 사전에 고정하고 semantic encoder/decoder만 federated training
* 처음부터 두 개의 DeepJSCC encoder–decoder로 통합

첫 논문에는 세 번째 방식이 가장 단순합니다.

---

# 방법론 방향

단순히 **“target client에서 SCU를 수행하고 FedAvg한다”** 정도로 끝나면 incremental extension으로 평가될 가능성이 큽니다. 다음과 같이 federated setting에서만 발생하는 문제를 해결해야 합니다.

## 1단계: Client-sensitive contribution 식별

Target client (u)가 encoder와 decoder에 미친 영향을 module-wise 또는 parameter-wise로 식별합니다.

예를 들면 다음 정보를 사용할 수 있습니다.

* target client의 마지막 local update
* 여러 round의 누적 update
* target client local Fisher information
* target data에 대한 encoder/decoder gradient saliency
* target client에 특화된 latent subspace

이 과정의 목적은 전체 codec을 무작정 변화시키지 않고, target client와 관련성이 높은 encoder 및 decoder 영역에 unlearning update를 집중하는 것입니다.

---

## 2단계: Target client에서 joint forgetting

Target client가 unlearning request를 보낼 때 아직 local data를 보유하고 있다고 가정하면, client가 한 번의 active unlearning round에 참여할 수 있습니다.

Target client는 raw data를 서버로 보내지 않고 자신의 local data에서 다음과 같은 joint unlearning update를 계산합니다.

[
\mathcal{L}_{\mathrm{forget}}
=============================

\mathcal{L}*{\mathrm{enc\text{-}forget}}
+
\mathcal{L}*{\mathrm{dec\text{-}forget}}
]

여기에는 SCU의 아이디어를 확장하여 다음을 사용할 수 있습니다.

* target data와 latent representation 간 mutual information 감소
* target-specific latent direction 제거
* target client의 reconstruction gradient를 역방향으로 적용
* original model과 지나치게 멀어지지 않도록 KL 또는 parameter constraint 적용
* client-sensitive parameter에만 gradient masking 적용

단순 reconstruction loss ascent만 사용하면 utility가 쉽게 붕괴하므로, client-sensitive subspace와 joint codec constraint가 중요합니다.

---

## 3단계: Remaining client 기반 distributed repair

이 부분이 가장 중요한 contribution 후보입니다.

서버가 target client의 forgetting update를 적용한 후, 일부 remaining client가 local retain data를 사용하여 codec을 짧게 repair합니다.

각 remaining client (k\neq u)는 다음과 같은 loss를 계산할 수 있습니다.

[
\mathcal{L}_{\mathrm{repair}}^{k}
=================================

\mathcal{L}*{\mathrm{rec}}^{k}
+
\lambda*{\mathrm{sem}}
\mathcal{L}*{\mathrm{semantic\ consistency}}^{k}
+
\lambda*{\mathrm{dist}}
\mathcal{L}_{\mathrm{distill}}^{k}.
]

여기서

* (\mathcal{L}_{\mathrm{rec}}): retained image reconstruction 보존
* (\mathcal{L}_{\mathrm{semantic\ consistency}}): encoder와 decoder 간 latent–output 정합성 보존
* (\mathcal{L}_{\mathrm{distill}}): retained data에서 기존의 유용한 output 관계 보존

이 repair는 모든 remaining data를 서버로 모으지 않고 federated하게 수행됩니다.

특히 original model을 teacher로 사용할 경우에는 **forget data에는 distillation을 적용하지 않고 retained client data에만 적용**해야 합니다. 그렇지 않으면 원래 모델의 target-client 지식을 다시 주입할 위험이 있습니다.

---

## 4단계: Forgetting과 repair gradient의 충돌 완화

Target client의 forgetting gradient와 remaining client의 repair gradient는 서로 반대 방향일 수 있습니다.

이를 단순 averaging하면 다음 중 하나가 발생할 수 있습니다.

* repair가 target client 정보를 다시 학습
* forgetting이 remaining client reconstruction을 파괴
* encoder와 decoder update 크기가 달라 codec mismatch 발생

따라서 다음과 같은 방법을 적용할 수 있습니다.

* gradient projection
* conflict-aware aggregation
* encoder/decoder별 adaptive unlearning rate
* target-sensitive subspace에서만 forgetting 허용
* safe parameter space에서는 repair만 허용

이 부분까지 들어가면 단순한 SCU의 federated extension을 넘어, 독립적인 알고리즘 contribution이 됩니다.

---

# 연구 질문으로 정리하면

가장 적절한 핵심 질문은 다음과 같습니다.

> **How can a federatively trained reconstruction semantic codec remove an entire client’s contribution from both the encoder and decoder without centralizing raw data or retraining from scratch, while preserving encoder–decoder consistency and reconstruction utility under non-IID data and wireless channel uncertainty?**

이를 세부 연구 질문으로 나누면 다음과 같습니다.

1. Target client의 영향이 encoder와 decoder에 각각 어떻게 분포하는가?
2. Encoder-only 또는 decoder-only unlearning으로 충분한가?
3. Remaining data가 분산된 상황에서 SCU의 compensation을 어떻게 federated하게 수행할 것인가?
4. Non-IID 정도가 클수록 target client contribution을 제거하기 어려워지는가?
5. Unlearning 결과가 다양한 SNR과 channel에서도 유지되는가?
6. Unlearned codec이 retrained codec과 얼마나 기능적으로 가까운가?

---

# 실험 설계에서 반드시 들어가야 할 것

## Baseline

최소한 다음은 비교해야 합니다.

* Original model, no unlearning
* Full retraining without target client
* Generic gradient-ascent FU
* FedEraser 또는 historical update subtraction 계열
* SCU의 centralized 또는 federated adaptation
* SSFU-style feature masking adaptation
* Proposed joint encoder–decoder FU

Ablation에는 반드시 다음을 포함해야 합니다.

* Encoder-only unlearning
* Decoder-only unlearning
* Joint encoder–decoder unlearning
* Joint unlearning without distributed repair
* Full proposed method

이 ablation이 있어야 “왜 양쪽 모두 unlearn해야 하는가”를 실험적으로 증명할 수 있습니다.

---

## Utility 평가

Remaining client에 대해 다음을 측정하는 것이 좋습니다.

* PSNR
* MS-SSIM
* LPIPS
* Reconstruction MSE
* 다양한 SNR에서의 reconstruction 성능

단순 평균뿐 아니라 client별 성능 분산도 보면 좋습니다. Unlearning 후 일부 non-IID client만 크게 손해 보는지 확인할 수 있기 때문입니다.

---

## Unlearning 평가

가장 중요한 기준은 retrain baseline과의 유사성입니다.

### Functional retraining gap

Parameter distance는 neural network의 permutation이나 scaling 때문에 의미가 약할 수 있습니다. 따라서 동일 입력과 동일 channel realization에서 unlearned model과 retrained model의 reconstruction output 차이를 측정하는 편이 더 낫습니다.

[
d_{\mathrm{func}}
=================

\mathbb{E}_{x,h,n}
\left[
\left|
\hat{x}^{U}
-----------

\hat{x}^{\mathrm{retrain}}
\right|^2
\right].
]

### Membership inference attack

Target client의 training sample이 member인지 구분하는 공격의 성공률이 retrained model 수준으로 감소해야 합니다.

### Backdoor 또는 client marker

Target client 데이터에만 trigger를 삽입하여 global codec에 남은 target contribution을 추적할 수 있습니다. 다만 backdoor 제거만으로 일반적인 unlearning이 입증되는 것은 아니므로 MIA와 retrain gap을 함께 사용해야 합니다.

### Latent leakage

Target client의 latent representation과 client identity 사이의 분류 가능성 또는 mutual information proxy를 측정할 수 있습니다.

---

## 중요한 평가 원칙

Forget client의 reconstruction MSE는 “클수록 무조건 좋다”가 아닙니다.

올바른 기준은

[
\mathrm{MSE}*{\mathrm{forget}}^{U}
\approx
\mathrm{MSE}*{\mathrm{forget}}^{\mathrm{retrain}}
]

입니다.

Unlearned model의 forget-set MSE가 retrained model보다 지나치게 크다면, target client의 기여를 정확히 제거한 것이 아니라 해당 분포의 일반적 지식까지 파괴한 것일 수 있습니다.

---

## Federated 및 channel 조건

다음 설정이 필요합니다.

* IID와 non-IID
* Dirichlet concentration 변화
* client 수 변화
* target client의 데이터 비율 변화
* 한 명 및 여러 명의 client unlearning
* sequential unlearning request
* AWGN과 Rayleigh fading
* training SNR과 test SNR mismatch

특히 **non-IID 정도와 channel noise가 unlearning–utility trade-off에 미치는 영향**은 통신 분야 논문에서 좋은 분석 포인트가 됩니다.

---

# 논문으로 강하게 만들기 위한 핵심

다음 수준이면 다소 약할 수 있습니다.

> SCU loss를 target client에 적용하고 결과를 FedAvg로 aggregation한다.

다음 수준까지 가면 연구성이 크게 올라갑니다.

> Target client의 encoder–decoder contribution을 공동 식별하고 제거하며, remaining clients가 raw data를 공유하지 않은 상태에서 distributed repair를 수행하고, forgetting과 repair gradient 충돌을 제어한다. 또한 retraining equivalence를 non-IID 및 wireless channel 조건에서 평가한다.

즉, 가장 중요한 기술적 novelty는 단순히 **“encoder와 decoder 모두 unlearn한다”**가 아니라 다음의 결합입니다.

> **Joint codec forgetting + distributed retain compensation + client-level retraining equivalence**

---

# 최종적으로 정리한 연구 아이디어

다음처럼 정리하면 상당히 탄탄합니다.

> 본 연구에서는 federated learning을 통해 여러 client의 분산 데이터로 학습된 reconstruction-oriented semantic communication system을 고려한다. 특정 client가 탈퇴하여 자신의 전체 데이터에 대한 삭제를 요청할 경우, 해당 client가 global semantic codec에 남긴 영향을 full retraining 없이 제거하는 client-level federated semantic unlearning을 수행한다. 특히 jointly trained encoder와 decoder에 target client의 기여가 동시에 분포한다는 점을 고려하여 두 모델을 공동으로 unlearn하며, remaining clients의 local data를 활용한 distributed compensation을 통해 encoder–decoder semantic consistency와 reconstruction 성능을 보존한다. Unlearning 결과는 target client를 처음부터 제외하고 retrain한 codec과의 functional similarity, membership inference, reconstruction quality 및 channel robustness를 통해 평가한다.

제목은 다음 정도가 적절합니다.

> **FedSCU-R: Client-Level Federated Unlearning for Reconstruction-Oriented Semantic Communications**

또는

> **Joint Encoder–Decoder Federated Unlearning for Reconstruction Semantic Communication Systems**

제 판단으로는 **연구 문제 자체는 충분히 강합니다.** 다만 novelty의 중심은 “semantic communication과 FU를 처음 결합했다”가 아니라, **“FL로 학습된 reconstruction codec의 encoder–decoder를 client 단위로 공동 unlearning하고, 분산된 remaining data만으로 utility를 복구한다”**에 두는 것이 가장 안전하고 설득력 있습니다.

[1]: https://ieeexplore.ieee.org/iel8/35/11053495/11024060.pdf?utm_source=chatgpt.com "Trustworthy Intelligent Networks for Low-Altitude Economy"
