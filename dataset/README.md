여러 방식으로 어둡게 만들어봄

# Done

## Gamma Correction
- 255 * (Input / 255) ^ gamma
- Chen et al., "Learning to See in the Dark" (CVPR 2018)

## Linear Brightness Scaling
- rgb x scale_rate
- Kim et al., "Robust Object Detection in Low Light: A Data Augmentation Approach" (ICCV Workshop 2021)

## Histogram Equalization (Inverse Use)
- Zhang et al., "Zero-DCE: Learning to Restore Low-Light Images Without Paired Data" (CVPR 2020)

## Camera Noise Models
- Chen et al., "SID: See-in-the-Dark Dataset" (CVPR 2018)

## HSV Manipulation
- rgb -> hsv -> v * 0.3 -> rgb
- Wu et al., "Enhancement of Mine Images Based on HSV Color Space” (IEEE 2024)

# Future Plan (Semantic Model Based Approach)

## Diffusion
## Style Stransfer