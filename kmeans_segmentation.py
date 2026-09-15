"""Method A: unsupervised segmentation via K-Means clustering on pixel
colors (scikit-learn). This is a genuine, classic unsupervised-ML approach
to segmentation: it groups pixels purely by color similarity, with no
knowledge of which cluster corresponds to which real-world class — that
correspondence has to be worked out afterward (see evaluate.py's Hungarian
matching), exactly as with any unsupervised clustering result.

K is chosen automatically per image via silhouette score, rather than
fixed. This matters more than it might look: an image might contain
anywhere from 1 to 3 shapes (plus background), so the "true" number of
distinct color populations varies per image. A fixed K set too high forces
K-Means to split something that has no real structure to split — in
testing, a fixed K=4 on a single-shape image (2 real color populations)
visibly fragmented the near-uniform background into three noisy clusters
of essentially random pixels, since K-Means must assign every point to
some cluster even when the "right" answer is fewer clusters than K.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _choose_k(pixels, max_k, random_state, sample_size=2000):
    """Pick the number of clusters (2..max_k) with the highest silhouette
    score, computed on a random pixel subsample for speed.
    """
    rng = np.random.default_rng(random_state)
    if len(pixels) > sample_size:
        sample_idx = rng.choice(len(pixels), size=sample_size, replace=False)
        sample = pixels[sample_idx]
    else:
        sample = pixels

    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit_predict(sample)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(sample, labels)
        if score > best_score:
            best_k, best_score = k, score

    return best_k


def segment_kmeans(image_bgr, max_clusters, random_state=42):
    """Cluster every pixel's color into an automatically-chosen number of
    groups (2..max_clusters).

    Returns an integer label map the same height/width as the image, with
    arbitrary cluster IDs — not yet matched to any particular class.
    """
    h, w, _ = image_bgr.shape
    pixels = image_bgr.reshape(-1, 3).astype(np.float64)

    k = _choose_k(pixels, max_clusters, random_state)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)

    return labels.reshape(h, w), k
