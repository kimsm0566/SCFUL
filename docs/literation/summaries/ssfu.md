# SSFU: Selective Semantic Feature Unlearning for Federated Learning in 6G Internet of Things Systems

## Bibliography and evidence

- Daluwatta, W., Khalil, I., Edirimannage, S., Elvitigala, C., Ng, J. S., and Niyato, D. (2026). *IEEE Internet of Things Journal*, 13(1), 852–867.
- DOI: [10.1109/JIOT.2025.3625756](https://doi.org/10.1109/JIOT.2025.3625756)
- Local source: [`SSFU_Selective_Semantic_Feature_Unlearning_for_Federated_Learning_in_6G_Internet_of_Things_Systems.pdf`](../papers/SSFU_Selective_Semantic_Feature_Unlearning_for_Federated_Learning_in_6G_Internet_of_Things_Systems.pdf)
- Evidence level: full text checked.

## Why the paper says unlearning is needed in semantic communication

1. **High-dimensional semantic features can retain sensitive attributes even when raw data are not centralized.** The paper notes that federated learning reduces raw-data exposure but does not eliminate semantic leakage: shared encodings can inadvertently encode sensitive information (Abstract; Sec. I).
2. **Semantic features can serve as adversarial triggers.** It identifies private or adversarially exploitable patterns embedded and propagated through a global model as a distinct SemCom risk (Abstract; Sec. I; Fig. 2).
3. **The right to be forgotten applies to learned contributions.** The paper presents federated unlearning as the response to deletion requirements under GDPR/CCPA-style data governance, rather than treating local raw-data retention as the only privacy issue (Sec. I; Sec. II-C).
4. **Whole-model retraining or wholesale client removal is expensive and unnecessarily destroys benign knowledge.** Its feature-level formulation is motivated by retaining useful semantic knowledge while removing only the risky latent components and allowing training to continue with little disruption (Abstract; Sec. I).

## Scope for this project

SSFU establishes that SemCom-specific federated unlearning is already studied. It removes high-risk semantic features from a semantic-embedding classification setting, not a whole client's contribution from a jointly trained reconstruction encoder–decoder codec. The paper therefore supports the privacy and security motivation, but is not a solution to this project's retrain-equivalence objective.
