# GUIDE: Generative Unlearning for Any Identity

## Bibliography and evidence

- Juwon Seo, Sung-Hoon Lee, Tae-Young Lee, Seungjun Moon, and Gyeong-Moon Park. CVPR 2024.
- Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Seo_Generative_Unlearning_for_Any_Identity_CVPR_2024_paper.html)
- Evidence level: primary full text and evaluation section checked.

## Task and evaluation

GUIDE measures target-identity similarity with a face-recognition observer, including
other unseen images of the same identity. It separately measures global generation
preservation with distribution distances to the pretrained generator and real images.

## Scope for this project

GUIDE supports using a target-specific learned signal together with a global image-quality
metric. Our checkerboard is a controlled client canary rather than a face identity, and
our primary reference remains retraining without the target client.
