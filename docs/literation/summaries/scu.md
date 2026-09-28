# SCU: An Efficient Machine Unlearning Scheme for Deep Learning Enabled Semantic Communications

## Bibliography and evidence

- Wang, W., Tian, Z., Zhang, C., and Yu, S. (2025). *IEEE Transactions on Information Forensics and Security*, 20, 547–558.
- DOI: [10.1109/TIFS.2024.3516576](https://doi.org/10.1109/TIFS.2024.3516576)
- Local source: [`SCU_An_Efficient_Machine_Unlearning_Scheme_for_Deep_Learning_Enabled_Semantic_Communications.pdf`](../papers/SCU_An_Efficient_Machine_Unlearning_Scheme_for_Deep_Learning_Enabled_Semantic_Communications.pdf)
- Evidence level: full text checked.

## Why the paper says unlearning is needed in semantic communication

1. **Semantic-codec training data can contain sensitive personal information.** The paper states that semantic encoders and decoders are trained from human-activity data and gives healthcare sensing data, including blood pressure and heart rate, as an example. A user who leaves the service can therefore require removal of that information from the trained model to exercise the right to be forgotten (Abstract; Sec. I; Fig. 1).
2. **A trained codec retains the erased data's contribution.** The deletion requirement is not satisfied merely by withdrawing from the service: the stated goal is to remove the specified data's influence from the learned semantic representation and the trained encoder–decoder pair (Abstract; Secs. III–IV).
3. **Poisoned or backdoored training samples can leave harmful behavior in reconstructed outputs.** SCU evaluates whether the trained codec retains a backdoor through decoder outputs, so it frames unlearning as removal of a security-relevant learned contribution as well as a privacy mechanism (Sec. I; Sec. VI).
4. **Full retraining is an expensive response to deletion.** The related-work discussion contrasts retraining-based removal with approximate unlearning, motivating a method that approaches remaining-data retraining behavior without retraining the full semantic model (Sec. II-B).

## Why generic unlearning is insufficient here

The paper treats this as a semantic-communication-specific problem because the semantic encoder and decoder are jointly trained and the full-data reconstruction setting is unsupervised. It argues that single-model, supervised unlearning methods do not directly preserve encoder–decoder semantic consistency or evaluate reconstruction-oriented forgetting appropriately (Sec. I; Sec. II-A).

## Scope for this project

SCU is the closest prior work for joint codec forgetting: it updates both encoder and decoder and keeps channel noise in the pipeline. It is not federated and its contrastive compensation centrally uses erased and remaining data; it therefore does not solve distributed retain repair after a whole-client deletion request.
