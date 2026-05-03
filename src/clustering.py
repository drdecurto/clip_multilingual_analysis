#!/usr/bin/env python3
"""
High-similarity-region clustering — ported from OneMap
(`mapping/nav_goals/clustering.py`).

Dependencies pared down to numpy + scipy + scikit-image + scikit-learn (no
rerun, no habitat, no torch). Three clustering strategies provided:

    cluster_high_similarity_regions  — local-maxima + region-growing (OneMap default)
    watershed_clustering             — watershed from local maxima
    gradient_based_clustering        — DBSCAN on gradient-masked high-similarity pixels

For cross-language IoU benchmarking the first is used by default; the others
are kept for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from skimage import filters, morphology
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN


# =============================================================================
# Data types
# =============================================================================

@dataclass
class Cluster:
    center:        np.ndarray            # [2]  (row, col) of peak-similarity point
    points:        np.ndarray            # [N, 2]  member pixel coordinates
    cluster_score: float                 # peak similarity within cluster

    def to_mask(self, shape: tuple) -> np.ndarray:
        """Binary mask over the similarity-map grid."""
        m = np.zeros(shape, dtype=bool)
        m[self.points[:, 0], self.points[:, 1]] = True
        return m


# =============================================================================
# Helpers
# =============================================================================

def find_local_maxima(similarity_map: np.ndarray,
                      mask: np.ndarray,
                      neighborhood_size: int = 10) -> np.ndarray:
    """Boolean map of strict local maxima within ``mask``."""
    local_max = maximum_filter(similarity_map, size=neighborhood_size)
    return (similarity_map == local_max) & mask


# =============================================================================
# Region-growing (OneMap default)
# =============================================================================

def cluster_high_similarity_regions(
        similarity_map:     np.ndarray,
        mask:               np.ndarray,
        neighborhood_size:  int   = 10,
        min_cluster_size:   int   = 2,
        relative_threshold: float = 0.8,
) -> List[Cluster]:
    """
    Greedy local-maximum + region-growing clustering.

    For each local maximum (highest-first), grow a connected component of
    pixels whose similarity ≥ ``relative_threshold * local_max``. Clusters
    smaller than ``min_cluster_size`` are discarded.

    Parameters
    ----------
    similarity_map   : [H, W] real-valued similarity
    mask             : [H, W] boolean — cells that are eligible to be clustered
    neighborhood_size: pixel radius for local-maximum suppression
    min_cluster_size : discard clusters below this many pixels
    relative_threshold: region-growing threshold as a fraction of the local peak
    """
    local_maxima = find_local_maxima(similarity_map, mask, neighborhood_size)
    maxima_coords = np.column_stack(np.where(local_maxima))
    if len(maxima_coords) == 0:
        return []

    maxima_with_scores = [
        (coord, float(similarity_map[tuple(coord)])) for coord in maxima_coords
    ]
    maxima_with_scores.sort(key=lambda x: x[1], reverse=True)

    clusters:  List[Cluster] = []
    processed = np.zeros_like(similarity_map, dtype=bool)
    H, W      = similarity_map.shape

    for coord, score in maxima_with_scores:
        if processed[tuple(coord)]:
            continue

        local_max_value = float(similarity_map[tuple(coord)])
        threshold       = local_max_value * relative_threshold

        stack                = [coord]
        cluster_points: List[np.ndarray] = []
        cluster_score        = -np.inf
        max_similarity_point = None

        while stack:
            current = stack.pop()
            ci, cj  = int(current[0]), int(current[1])
            if processed[ci, cj]:
                continue
            if similarity_map[ci, cj] >= threshold and mask[ci, cj]:
                processed[ci, cj] = True
                cluster_points.append(np.array([ci, cj]))

                current_similarity = float(similarity_map[ci, cj])
                if current_similarity > cluster_score:
                    cluster_score        = current_similarity
                    max_similarity_point = np.array([ci, cj])

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ni, nj = ci + dx, cj + dy
                        if 0 <= ni < H and 0 <= nj < W:
                            stack.append(np.array([ni, nj]))

        if len(cluster_points) >= min_cluster_size:
            pts = np.array(cluster_points)
            clusters.append(Cluster(
                center        = max_similarity_point,
                points        = pts,
                cluster_score = cluster_score,
            ))

    return clusters


# =============================================================================
# Alternative strategies (useful for ablation)
# =============================================================================

def watershed_clustering(
        similarity_map:     np.ndarray,
        mask:               np.ndarray,
        neighborhood_size:  int   = 30,
        min_cluster_size:   int   = 5,
        relative_threshold: float = 0.95,
) -> List[Cluster]:
    inverted = float(np.max(similarity_map)) - similarity_map
    local_maxima   = find_local_maxima(similarity_map, mask, neighborhood_size)
    maxima_coords  = np.column_stack(np.where(local_maxima))
    if len(maxima_coords) == 0:
        return []
    markers = np.zeros_like(similarity_map, dtype=np.int32)
    for i, c in enumerate(maxima_coords, start=1):
        markers[tuple(c)] = i
    labels = watershed(inverted, markers, mask=mask)

    clusters: List[Cluster] = []
    for i in range(1, int(labels.max()) + 1):
        cluster_mask   = labels == i
        cluster_points = np.argwhere(cluster_mask)
        if len(cluster_points) < min_cluster_size:
            continue
        cluster_vals   = similarity_map[cluster_mask]
        peak_idx       = int(np.argmax(cluster_vals))
        peak_pt        = cluster_points[peak_idx]
        peak_val       = float(cluster_vals[peak_idx])
        clusters.append(Cluster(
            center=np.array(peak_pt), points=cluster_points,
            cluster_score=peak_val,
        ))
    return clusters


def gradient_based_clustering(
        similarity_map:     np.ndarray,
        mask:               np.ndarray,
        percentile:         float = 95.0,
        min_cluster_size:   int   = 5,
        eps:                float = 3.0,
) -> List[Cluster]:
    """DBSCAN on pixels above the ``percentile``-th similarity percentile."""
    values = similarity_map[mask]
    if values.size == 0:
        return []
    tau = float(np.percentile(values, percentile))
    selected = (similarity_map >= tau) & mask
    coords = np.column_stack(np.where(selected))
    if len(coords) < min_cluster_size:
        return []

    db = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(coords)
    clusters: List[Cluster] = []
    for lbl in set(db.labels_):
        if lbl == -1:
            continue
        idx = np.where(db.labels_ == lbl)[0]
        pts = coords[idx]
        vals = similarity_map[pts[:, 0], pts[:, 1]]
        k = int(np.argmax(vals))
        clusters.append(Cluster(
            center=pts[k], points=pts, cluster_score=float(vals[k]),
        ))
    return clusters


# =============================================================================
# Cluster → mask utility
# =============================================================================

def clusters_to_mask(clusters: List[Cluster], shape: tuple) -> np.ndarray:
    """Union of all cluster masks."""
    m = np.zeros(shape, dtype=bool)
    for c in clusters:
        m[c.points[:, 0], c.points[:, 1]] = True
    return m
