"""
Method B: classical threshold-based segmentation — Otsu thresholding to
separate foreground from background, connected-component labeling to split
the foreground into individual blobs, then each blob is classified by
matching its average color to a set of known reference colors.

Unlike Method A (K-Means), this method does use prior knowledge (the
reference colors for each class), so its output labels are directly
comparable to ground-truth classes with no post-hoc matching needed — a
reasonable classical alternative when you do know roughly what colors to
expect.

"""

import cv2
import numpy as np


def _closest_class(mean_color_bgr, class_reference_colors):
    best_class, best_dist = None, float("inf")
    for cls, ref_color in class_reference_colors.items():
        dist = np.linalg.norm(mean_color_bgr - np.array(ref_color, dtype=np.float64))
        if dist < best_dist:
            best_dist = dist
            best_class = cls
    return best_class


def segment_threshold(image_bgr, class_reference_colors, class_to_idx, min_component_area=80):
    """Returns an integer label map (0 = background, otherwise the
    matched class index), using Otsu thresholding + connected components +
    nearest-reference-color classification.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Shapes are darker/more saturated than the light, near-white
    # background, so an inverted Otsu threshold isolates them as foreground.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n_components, component_labels = cv2.connectedComponents(binary)

    h, w = gray.shape
    result = np.zeros((h, w), dtype=np.int32)

    for comp_id in range(1, n_components):  # 0 is background
        comp_mask = component_labels == comp_id
        if comp_mask.sum() < min_component_area:
            continue
        mean_color = image_bgr[comp_mask].mean(axis=0)  # BGR
        predicted_class = _closest_class(mean_color, class_reference_colors)
        result[comp_mask] = class_to_idx[predicted_class]

    return result
