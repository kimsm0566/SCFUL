# SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency

## Bibliography and evidence

- Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, and Sijia Liu. ICLR 2024.
- Official page: [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html)
- Evidence level: primary full text and evaluation section checked.

## Task and evaluation

For image generation, SalUn uses an external classifier to measure whether the target
class or concept is still generated and FID to measure generation quality for retained
classes or prompts. Its classification experiments additionally report unlearning,
retain, and test accuracy, MIA, and runtime efficiency.

## Scope for this project

The relevant lesson is the two-axis evaluation: target behavior removal and retained
output quality are different questions. Marker-PUL is analogous to target generation
frequency, while FID and reconstruction metrics protect retained utility. SalUn itself
does not address a jointly trained federated semantic codec.
