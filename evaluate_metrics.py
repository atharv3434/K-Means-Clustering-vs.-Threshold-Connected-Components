"""Segmentation evaluation metrics: optimal label matching (for
unsupervised clustering output), pixel accuracy, per-class IoU, and Dice.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_labels_to_classes(pred_labels, gt_labels, n_pred_clusters, n_classes):
    """Find the best one-to-one mapping from arbitrary cluster IDs to
    ground-truth class indices, by maximizing total pixel overlap — the
    standard way to score unsupervised clustering against labeled data
    (this is what "clustering accuracy" means in the literature).

    Uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
    on the confusion matrix between predicted clusters and true classes.
    Returns a dict mapping predicted cluster ID -> matched class index.
    """
    confusion = np.zeros((n_pred_clusters, n_classes), dtype=np.int64)
    for c in range(n_pred_clusters):
        pred_mask = pred_labels == c
        for k in range(n_classes):
            confusion[c, k] = np.sum(pred_mask & (gt_labels == k))

    # linear_sum_assignment minimizes cost, so negate to maximize overlap.
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def remap_labels(pred_labels, mapping):
    remapped = np.zeros_like(pred_labels)
    for cluster_id, class_id in mapping.items():
        remapped[pred_labels == cluster_id] = class_id
    return remapped


def pixel_accuracy(pred_labels, gt_labels):
    return float(np.mean(pred_labels == gt_labels))


def iou_per_class(pred_labels, gt_labels, n_classes):
    ious = {}
    for k in range(n_classes):
        pred_mask = pred_labels == k
        gt_mask = gt_labels == k
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)
        ious[k] = float(intersection / union) if union > 0 else None
    return ious


def dice_per_class(pred_labels, gt_labels, n_classes):
    dices = {}
    for k in range(n_classes):
        pred_mask = pred_labels == k
        gt_mask = gt_labels == k
        intersection = np.sum(pred_mask & gt_mask)
        total = np.sum(pred_mask) + np.sum(gt_mask)
        dices[k] = float(2 * intersection / total) if total > 0 else None
    return dices
