# Image Segmentation — Evaluation Report

Comparing two classical segmentation methods against synthetic ground-truth masks:

- **Method A (K-Means clustering)**: fully unsupervised — clusters pixels by color alone, with no knowledge of the true classes. Cluster IDs are matched to ground-truth classes after the fact via the Hungarian algorithm (maximizing pixel overlap), the standard way to score unsupervised clustering.
- **Method B (Otsu threshold + connected components)**: uses known reference colors to directly classify each connected foreground component.

## Method A: K-Means clustering

- Pixel accuracy: **0.981**
- Mean IoU: **0.774**

| Class | IoU | Dice |
|---|---|---|
| background | 1.0 | 1.0 |
| circle | 0.638 | 0.668 |
| square | 0.706 | 0.728 |
| triangle | 0.751 | 0.795 |

## Method B: Threshold + connected components

- Pixel accuracy: **1.000**
- Mean IoU: **1.000**

| Class | IoU | Dice |
|---|---|---|
| background | 1.0 | 1.0 |
| circle | 1.0 | 1.0 |
| square | 1.0 | 1.0 |
| triangle | 1.0 | 1.0 |

## Example segmentations

Image | Ground truth | K-Means | Threshold + CC

![img_000.png](visualizations/img_000.png)
![img_001.png](visualizations/img_001.png)
![img_002.png](visualizations/img_002.png)
![img_003.png](visualizations/img_003.png)
![img_004.png](visualizations/img_004.png)
![img_005.png](visualizations/img_005.png)