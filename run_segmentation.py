"""Run both segmentation methods over the dataset, evaluate each against
ground truth, and save visual comparisons.

Usage:
    python src/run_segmentation.py [--config config.yaml]
"""

import argparse
import os
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_metadata
from kmeans_segmentation import segment_kmeans
from threshold_segmentation import segment_threshold
from evaluate_metrics import (
    match_labels_to_classes, remap_labels,
    pixel_accuracy, iou_per_class, dice_per_class,
)

CLASS_DISPLAY_COLORS = ["#F5F5F5", "#D6484B", "#4CAF50", "#3E7CC9"]  # background, circle, square, triangle


def aggregate_metrics(all_pixel_acc, all_ious, all_dices, n_classes):
    mean_pixel_acc = float(np.mean(all_pixel_acc))

    per_class_ious = {k: [] for k in range(n_classes)}
    per_class_dices = {k: [] for k in range(n_classes)}
    for ious, dices in zip(all_ious, all_dices):
        for k in range(n_classes):
            if ious[k] is not None:
                per_class_ious[k].append(ious[k])
            if dices[k] is not None:
                per_class_dices[k].append(dices[k])

    mean_iou_per_class = {k: (float(np.mean(v)) if v else None) for k, v in per_class_ious.items()}
    mean_dice_per_class = {k: (float(np.mean(v)) if v else None) for k, v in per_class_dices.items()}
    valid_ious = [v for v in mean_iou_per_class.values() if v is not None]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {
        "pixel_accuracy": round(mean_pixel_acc, 3),
        "mean_iou": round(mean_iou, 3),
        "iou_per_class": {k: (round(v, 3) if v is not None else None) for k, v in mean_iou_per_class.items()},
        "dice_per_class": {k: (round(v, 3) if v is not None else None) for k, v in mean_dice_per_class.items()},
    }


def visualize(image_bgr, gt_labels, kmeans_labels, threshold_labels, classes, out_path):
    cmap = ListedColormap(CLASS_DISPLAY_COLORS[:len(classes)])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(gt_labels, cmap=cmap, vmin=0, vmax=len(classes) - 1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(kmeans_labels, cmap=cmap, vmin=0, vmax=len(classes) - 1)
    axes[2].set_title("K-Means (Method A)")
    axes[3].imshow(threshold_labels, cmap=cmap, vmin=0, vmax=len(classes) - 1)
    axes[3].set_title("Threshold + CC (Method B)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate both segmentation methods.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config["visualizations_dir"], exist_ok=True)

    classes = config["classes"]
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    class_ref_colors = config["class_reference_colors"]

    metadata = load_metadata(config["metadata_path"])
    records = metadata["images"]
    print(f"Loaded {len(records)} images.\n")

    kmeans_pixel_acc, kmeans_ious, kmeans_dices = [], [], []
    thresh_pixel_acc, thresh_ious, thresh_dices = [], [], []

    print("Running both segmentation methods...")
    for idx, record in enumerate(records):
        image_path = os.path.join(config["images_dir"], record["file"])
        mask_path = os.path.join(config["masks_dir"], record["file"])
        image = cv2.imread(image_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)

        # --- Method A: K-Means (unsupervised, needs label matching) ---
        kmeans_raw, k_used = segment_kmeans(image, config["kmeans_max_clusters"], config["kmeans_random_state"])
        mapping = match_labels_to_classes(kmeans_raw, gt_mask, k_used, n_classes)
        kmeans_labels = remap_labels(kmeans_raw, mapping)

        # --- Method B: threshold + connected components (labels already semantic) ---
        threshold_labels = segment_threshold(image, class_ref_colors, class_to_idx, config["min_component_area"])

        kmeans_pixel_acc.append(pixel_accuracy(kmeans_labels, gt_mask))
        kmeans_ious.append(iou_per_class(kmeans_labels, gt_mask, n_classes))
        kmeans_dices.append(dice_per_class(kmeans_labels, gt_mask, n_classes))

        thresh_pixel_acc.append(pixel_accuracy(threshold_labels, gt_mask))
        thresh_ious.append(iou_per_class(threshold_labels, gt_mask, n_classes))
        thresh_dices.append(dice_per_class(threshold_labels, gt_mask, n_classes))

        if idx < config.get("n_visualizations", 6):
            out_path = os.path.join(config["visualizations_dir"], record["file"])
            visualize(image, gt_mask, kmeans_labels, threshold_labels, classes, out_path)

    kmeans_metrics = aggregate_metrics(kmeans_pixel_acc, kmeans_ious, kmeans_dices, n_classes)
    thresh_metrics = aggregate_metrics(thresh_pixel_acc, thresh_ious, thresh_dices, n_classes)

    print("\n--- Method A: K-Means clustering ---")
    print(f"  Pixel accuracy: {kmeans_metrics['pixel_accuracy']:.3f}")
    print(f"  Mean IoU:       {kmeans_metrics['mean_iou']:.3f}")
    for k, cls in enumerate(classes):
        print(f"    {cls:10s} IoU={kmeans_metrics['iou_per_class'][k]}  Dice={kmeans_metrics['dice_per_class'][k]}")

    print("\n--- Method B: Threshold + connected components ---")
    print(f"  Pixel accuracy: {thresh_metrics['pixel_accuracy']:.3f}")
    print(f"  Mean IoU:       {thresh_metrics['mean_iou']:.3f}")
    for k, cls in enumerate(classes):
        print(f"    {cls:10s} IoU={thresh_metrics['iou_per_class'][k]}  Dice={thresh_metrics['dice_per_class'][k]}")

    # --- Report ---
    lines = [
        "# Image Segmentation — Evaluation Report",
        "",
        "Comparing two classical segmentation methods against synthetic ground-truth masks:",
        "",
        "- **Method A (K-Means clustering)**: fully unsupervised — clusters pixels by "
        "color alone, with no knowledge of the true classes. Cluster IDs are matched "
        "to ground-truth classes after the fact via the Hungarian algorithm (maximizing "
        "pixel overlap), the standard way to score unsupervised clustering.",
        "- **Method B (Otsu threshold + connected components)**: uses known reference "
        "colors to directly classify each connected foreground component.",
        "",
        "## Method A: K-Means clustering",
        "",
        f"- Pixel accuracy: **{kmeans_metrics['pixel_accuracy']:.3f}**",
        f"- Mean IoU: **{kmeans_metrics['mean_iou']:.3f}**",
        "",
        "| Class | IoU | Dice |",
        "|---|---|---|",
    ]
    for k, cls in enumerate(classes):
        lines.append(f"| {cls} | {kmeans_metrics['iou_per_class'][k]} | {kmeans_metrics['dice_per_class'][k]} |")

    lines += [
        "",
        "## Method B: Threshold + connected components",
        "",
        f"- Pixel accuracy: **{thresh_metrics['pixel_accuracy']:.3f}**",
        f"- Mean IoU: **{thresh_metrics['mean_iou']:.3f}**",
        "",
        "| Class | IoU | Dice |",
        "|---|---|---|",
    ]
    for k, cls in enumerate(classes):
        lines.append(f"| {cls} | {thresh_metrics['iou_per_class'][k]} | {thresh_metrics['dice_per_class'][k]} |")

    lines += ["", "## Example segmentations", "", "Image | Ground truth | K-Means | Threshold + CC", ""]
    for record in records[:config.get("n_visualizations", 6)]:
        lines.append(f"![{record['file']}](visualizations/{record['file']})")

    with open(config["report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {config['report_path']}")


if __name__ == "__main__":
    main()
