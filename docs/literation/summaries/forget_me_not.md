# Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models

## Bibliography and evidence

- Gong Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, and Humphrey Shi. CVPR Workshops 2024.
- Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/html/Zhang_Forget-Me-Not_Learning_to_Forget_in_Text-to-Image_Diffusion_Models_CVPRW_2024_paper.html)
- Evidence level: primary paper checked.

## Task and evaluation

The paper removes identities, objects, or styles from a pretrained text-to-image model
using attention re-steering, while checking that unrelated generation capability remains.
It is a concept-erasure setting rather than reconstruction of paired inputs.

## Scope for this project

It supports the general requirement to report target removal and collateral utility
together. It does not directly justify client-level deletion, masked reconstruction, or
distributed repair, so it is a secondary rather than direct precedent for method 2.
