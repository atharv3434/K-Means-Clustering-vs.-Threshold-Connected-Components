"""Generate synthetic images with per-pixel semantic segmentation ground
truth (circle / square / triangle / background), with realistic color
noise so segmentation isn't a trivial exact-threshold problem.

This project ships with pre-generated images and masks already in place
(data/images/, data/masks/), so you don't need to run this to try the
project out. Run it again for a fresh random set.

Usage:
    python data/generate_data.py [--n-images 40] [--seed 42]
"""

import argparse
import json
import os

import numpy as np
import cv2

CLASSES = ["background", "circle", "square", "triangle"]  # index 0..3
CANVAS_SIZE = 150

BACKGROUND_COLOR = np.array([245, 245, 245])  # BGR
CLASS_COLORS = {
    "circle": np.array([55, 55, 215]),     # red-ish
    "square": np.array([60, 175, 60]),     # green-ish
    "triangle": np.array([210, 140, 45]),  # blue-ish
}
BACKGROUND_NOISE_STD = 3
SHAPE_COLOR_NOISE_STD = 14


def _random_box(rng, size_range, placed_boxes, max_tries=50):
    for _ in range(max_tries):
        w = h = int(rng.integers(size_range[0], size_range[1]))
        x = int(rng.integers(8, CANVAS_SIZE - w - 8))
        y = int(rng.integers(8, CANVAS_SIZE - h - 8))
        candidate = (x, y, w, h)
        if not any(_boxes_overlap(candidate, b) for b in placed_boxes):
            return candidate
    return None


def _boxes_overlap(a, b, pad=6):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax, ay, aw, ah = ax - pad, ay - pad, aw + 2 * pad, ah + 2 * pad
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def _jittered_color(rng, base_color):
    noise = rng.normal(0, SHAPE_COLOR_NOISE_STD, size=3)
    return np.clip(base_color + noise, 0, 255).astype(np.uint8)


def _draw_shape_on(canvas, mask, shape, box, color, class_idx):
    """Draw the shape onto both the RGB canvas and the integer class mask,
    using a single-channel scratch mask so the two stay perfectly aligned.
    """
    x, y, w, h = box
    shape_mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    if shape == "circle":
        center = (x + w // 2, y + h // 2)
        cv2.circle(shape_mask, center, w // 2, 255, thickness=-1)
    elif shape == "square":
        cv2.rectangle(shape_mask, (x, y), (x + w, y + h), 255, thickness=-1)
    elif shape == "triangle":
        pts = np.array([[x + w // 2, y], [x, y + h], [x + w, y + h]], dtype=np.int32)
        cv2.fillPoly(shape_mask, [pts], 255)

    color_bgr = tuple(int(c) for c in color)
    canvas[shape_mask == 255] = color_bgr
    mask[shape_mask == 255] = class_idx


def generate_image(rng, n_shapes):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BACKGROUND_COLOR, dtype=np.int16)
    canvas += rng.normal(0, BACKGROUND_NOISE_STD, size=canvas.shape).astype(np.int16)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)  # 0 = background

    placed_boxes = []
    shapes_present = []
    shape_names = list(CLASS_COLORS.keys())

    for _ in range(n_shapes):
        box = _random_box(rng, size_range=(30, 55), placed_boxes=placed_boxes)
        if box is None:
            continue
        shape = shape_names[rng.integers(0, len(shape_names))]
        color = _jittered_color(rng, CLASS_COLORS[shape])
        class_idx = CLASSES.index(shape)
        _draw_shape_on(canvas, mask, shape, box, color, class_idx)
        placed_boxes.append(box)
        shapes_present.append(shape)

    return canvas, mask, shapes_present


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic segmentation data.")
    parser.add_argument("--n-images", type=int, default=40, help="Number of images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-shapes", type=int, default=1, help="Min shapes per image")
    parser.add_argument("--max-shapes", type=int, default=3, help="Max shapes per image")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images_dir = os.path.join(args.out_dir, "images")
    masks_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    records = []
    for i in range(args.n_images):
        n_shapes = int(rng.integers(args.min_shapes, args.max_shapes + 1))
        canvas, mask, shapes_present = generate_image(rng, n_shapes)
        filename = f"img_{i:03d}.png"
        cv2.imwrite(os.path.join(images_dir, filename), canvas)
        cv2.imwrite(os.path.join(masks_dir, filename), mask)
        records.append({"file": filename, "shapes": shapes_present})

    metadata_path = os.path.join(args.out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "images": records}, f, indent=2)

    print(f"Wrote {len(records)} images to {images_dir}")
    print(f"Wrote {len(records)} ground-truth masks to {masks_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
