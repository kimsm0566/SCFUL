# Semantic Communication Unlearning: A Variational Information Bottleneck Approach for Backdoor Defense in Wireless Systems

## Bibliography and evidence

- Karahan, S. N., Gullu, M., Osmanca, M. S., and Barisci, N. (2026). *Future Internet*, 18(1), 17.
- DOI: [10.3390/fi18010017](https://doi.org/10.3390/fi18010017)
- Local source: [`VIB_SCU_Semantic_Communication_Unlearning_Backdoor_Defense.pdf`](../papers/VIB_SCU_Semantic_Communication_Unlearning_Backdoor_Defense.pdf)
- Evidence level: full text checked; the publisher PDF is open access.

## Why the paper says unlearning is needed in semantic communication

1. **A semantic encoder–decoder can learn covert backdoor behavior from poisoned training samples.** The paper describes triggers that are active only for selected inputs while normal clean-input behavior remains intact, making a compromised semantic communication model difficult to detect by ordinary validation (Sec. 1).
2. **The security impact is serious in wireless and edge settings.** It connects compromised signal-processing models to critical settings such as autonomous systems and military communications, and evaluates multiple SNR and fading conditions (Secs. 1–2; Sec. 6).
3. **Retraining and conventional defenses are poorly suited to the deployment setting.** Complete retraining requires substantial computation and clean labelled data that may be unavailable in edge or federated deployments. Detection struggles with stealthy frequency/wavelet-domain triggers, while pruning can damage legitimate signal-processing performance (Sec. 1; Sec. 2.2).
4. **The harmful knowledge is represented in the VIB latent distribution, not necessarily an isolated parameter subset.** The paper argues that parameter-space unlearning can miss a latent distributional backdoor shift or cause catastrophic forgetting; targeted latent-space unlearning is consequently motivated (Sec. 1).

## Scope for this project

This is a direct semantic encoder–decoder unlearning paper: its stated method jointly removes backdoor knowledge from encoding and decoding while preserving clean representations. Its focus is centralized, poisoned-sample/backdoor defense for wireless signal data, not client-level federated deletion or distributed repair using retain-client data.
