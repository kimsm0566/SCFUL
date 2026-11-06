# Machine Unlearning for Image-to-Image Generative Models

## Bibliography and evidence

- Guihong Li, Hsiang Hsu, Chun-Fu Chen, and Radu Marculescu. ICLR 2024.
- Official page: [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8668fdc7b2ddf55a0e235824c66f2eee-Abstract-Conference.html)
- Evidence level: primary full text and evaluation tables checked.

## Task and evaluation

The paper studies diffusion, VQ-GAN, and MAE image-to-image models that reconstruct
missing image regions from partial inputs. It evaluates retain and forget sets separately
with FID, Inception Score, and CLIP similarity. Good unlearning preserves retain-set
generation while making forget-set reconstruction less faithful.

## Scope for this project

This is the direct precedent for the masked-input structure of marker method 2. The
project replaces the paper's hidden natural-image region with a hidden client-specific
checkerboard target. The paper is centralized and sample/class-set oriented, so it does
not replace the retrained-without-client reference or distributed repair requirement.
