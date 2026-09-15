# Image Segmentation: K-Means Clustering vs. Threshold + Connected Components

A semantic segmentation project comparing two classical (non-deep-learning)
approaches on synthetic images with pixel-level ground truth:

- **Method A — K-Means clustering**: a genuinely unsupervised ML approach.
  Groups pixels purely by color similarity, with no knowledge of the real
  classes. Since cluster IDs are arbitrary, they're matched to ground-truth
  classes afterward via the Hungarian algorithm — the standard way to score
  unsupervised clustering against labels.
- **Method B — Otsu threshold + connected components**: a classical
  pipeline that separates foreground from background, splits the
  foreground into blobs, and classifies each blob by matching its average
  color to known reference colors.

## Project structure

```
image-segmentation/
├── config.yaml                     # methods' settings, class list, paths
├── requirements.txt
├── data/
│   ├── generate_data.py            # (re)generates synthetic images + masks
│   ├── images/                     # pre-generated sample images
│   ├── masks/                      # per-pixel ground-truth class masks
│   └── metadata.json
├── src/
│   ├── utils.py                    # config + metadata loading
│   ├── kmeans_segmentation.py      # Method A
│   ├── threshold_segmentation.py   # Method B
│   ├── evaluate_metrics.py         # Hungarian matching, pixel acc, IoU, Dice
│   └── run_segmentation.py         # CLI: runs both methods, evaluates, visualizes
├── output/
│   ├── visualizations/             # side-by-side comparison images
│   └── evaluation_report.md        # full metrics + visualizations
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
python src/run_segmentation.py
```

Runs both methods over every image in `data/images/`, evaluates each
against the ground-truth masks in `data/masks/`, and prints:

```
--- Method A: K-Means clustering ---
  Pixel accuracy: 0.981
  Mean IoU:       0.774
    background IoU=1.0    Dice=1.0
    circle     IoU=0.638  Dice=0.668
    square     IoU=0.706  Dice=0.728
    triangle   IoU=0.751  Dice=0.795

--- Method B: Threshold + connected components ---
  Pixel accuracy: 1.000
  Mean IoU:       1.000
    (all classes: IoU=1.0, Dice=1.0)
```

and saves side-by-side comparison images (`output/visualizations/`) plus a
full report (`output/evaluation_report.md`).

## Why Method B outperforms Method A here

Method B gets a perfect score because it's given a real advantage — the
known reference color for each class — which turns segmentation into a
much easier nearest-color classification problem on already-separated
blobs. Method A gets no such help: it has to discover structure in the
pixel colors with zero labels, which is a strictly harder problem and a
fair, realistic setting for many real applications where you *don't* know
your classes' colors in advance. IoU in the 0.6–0.8 range per shape for a
genuinely unsupervised method on noisy, jittered colors is a solid result.

## A real bug this project caught (and how it was fixed)

The first version of Method A used a **fixed** number of clusters (K=4,
matching background + 3 shape classes). On images containing only 1 shape
— so really just 2 distinct color populations — K-Means was still forced
to produce 4 clusters, and with no real third or fourth grouping to find,
it fragmented the near-uniform background into 3 clusters of essentially
random noise. Pixel accuracy on the full dataset was only 0.68.

The fix: **K is now chosen automatically per image** (`kmeans_segmentation.py`),
by trying K=2..4 and picking whichever has the best silhouette score. This
brought pixel accuracy up to 0.98 and fixed the background fragmentation
entirely — a good illustration of why K-Means requires care in choosing K,
and why testing against ground truth (rather than assuming reasonable
metrics) matters.

## Using your own data

1. Replace `data/images/` and `data/masks/` with your own image/mask pairs
   (masks should be single-channel PNGs with pixel value = class index).
2. Update `classes` and `class_reference_colors` in `config.yaml` to match
   your schema (Method B needs the reference colors; Method A doesn't).
3. Re-run `python src/run_segmentation.py`.

## Extending this project

- **More classes / more clusters**: raise `kmeans_max_clusters` and add
  entries to `class_reference_colors`.
- **Better foreground/background separation**: for more complex real
  images, `threshold_segmentation.py`'s single global Otsu threshold could
  be replaced with adaptive thresholding or GrabCut.
- **Smarter cluster-count selection**: the silhouette-score search in
  `kmeans_segmentation.py` is a solid general-purpose method, but for
  larger images consider the elbow method or Bayesian information
  criterion for speed.
- **Superpixel pre-processing**: running SLIC superpixel segmentation
  before K-Means (clustering superpixel-average colors instead of raw
  pixels) is a common real-world improvement for noisier images.
