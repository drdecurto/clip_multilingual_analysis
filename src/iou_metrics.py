#!/usr/bin/env python3
"""
Cross-language metrics for dense-CLIP grounding.

The core research question is: *given the same image and the same concept,
does the set of pixels detected as "this concept" match across languages?*

Primary metric
    mask IoU between clusters produced from the per-language similarity map
    (one language at a time, with English as the reference).

Secondary metrics
    threshold-IoU      — binary IoU at a fixed (or percentile-based) threshold
                         of the raw similarity map, which bypasses the clustering
                         step and so decouples signal quality from clustering
                         robustness.
    Spearman / Pearson — rank/linear correlation of the raw similarity maps
                         (continuous signal, no thresholding at all).
    peak-similarity    — the maximum cosine similarity reached in each map;
                         the simplest possible "does the concept light up?" test.
    center-point error — for each cluster in language L, distance to the
                         nearest cluster centre in the reference language.

All metrics are computed per (image, concept, language) and aggregated offline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .clustering import Cluster, clusters_to_mask


# =============================================================================
# Mask IoU
# =============================================================================

def mask_iou(m1: np.ndarray, m2: np.ndarray,
             empty_empty_value: float = 1.0) -> float:
    """
    Intersection-over-Union between two boolean masks.

    Convention for empty masks:
      both empty  → ``empty_empty_value`` (default 1.0 = "agree on nothing")
      one empty   → 0.0
    """
    m1 = np.asarray(m1, dtype=bool)
    m2 = np.asarray(m2, dtype=bool)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return float(empty_empty_value)
    if m1.sum() == 0 or m2.sum() == 0:
        return 0.0
    return float(inter) / float(union)


def threshold_mask(sim_map: np.ndarray,
                   tau: Optional[float] = None,
                   percentile: Optional[float] = None) -> np.ndarray:
    """
    Binarize a similarity map.
      - If ``tau`` given, return ``sim_map >= tau``
      - Else if ``percentile`` given, tau = np.percentile(sim_map, percentile)
      - Else raise
    """
    if tau is None and percentile is None:
        raise ValueError("Provide either tau or percentile")
    if tau is None:
        tau = float(np.percentile(sim_map, percentile))
    return sim_map >= tau


# =============================================================================
# Raw-signal metrics
# =============================================================================

def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two similarity maps (flattened)."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def linear_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two similarity maps."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


def peak_similarity(sim_map: np.ndarray) -> float:
    return float(np.max(sim_map))


def mean_similarity(sim_map: np.ndarray) -> float:
    return float(np.mean(sim_map))


# =============================================================================
# Cluster geometry metrics
# =============================================================================

def nearest_center_distance(clusters_a: List[Cluster],
                            clusters_b: List[Cluster]) -> float:
    """
    Mean, over clusters in A, of the distance to the nearest cluster centre in B.

    Returns NaN if either list is empty.
    """
    if not clusters_a or not clusters_b:
        return float("nan")
    centres_b = np.stack([c.center for c in clusters_b])  # [Nb, 2]
    d_list = []
    for ca in clusters_a:
        d = np.linalg.norm(centres_b - ca.center[None, :], axis=1).min()
        d_list.append(float(d))
    return float(np.mean(d_list))


def symmetric_center_distance(clusters_a: List[Cluster],
                              clusters_b: List[Cluster]) -> float:
    """Symmetric (A→B + B→A)/2 nearest-centre distance."""
    d_ab = nearest_center_distance(clusters_a, clusters_b)
    d_ba = nearest_center_distance(clusters_b, clusters_a)
    if np.isnan(d_ab) or np.isnan(d_ba):
        return float("nan")
    return 0.5 * (d_ab + d_ba)


# =============================================================================
# Full per-language summary
# =============================================================================

def summarise_pair(
        sim_ref:       np.ndarray,
        sim_lang:      np.ndarray,
        clusters_ref:  List[Cluster],
        clusters_lang: List[Cluster],
        mask_ref:      Optional[np.ndarray] = None,
        mask_lang:     Optional[np.ndarray] = None,
        thresholds:    Tuple[float, ...] = (0.15, 0.20, 0.25),
        percentiles:   Tuple[float, ...] = (90.0, 95.0, 99.0),
) -> Dict[str, float]:
    """
    Produce the full metric dict for a single (image, concept) pair, comparing
    a reference-language map (usually English) against a target-language map.

    Input similarity maps must share the same shape. The pre-computed cluster
    lists should come from ``cluster_high_similarity_regions`` (or equivalent);
    if not provided, masks are derived from cluster-union coverage.
    """
    assert sim_ref.shape == sim_lang.shape, "Similarity maps must match in shape"
    shape = sim_ref.shape

    if mask_ref is None:
        mask_ref = clusters_to_mask(clusters_ref, shape)
    if mask_lang is None:
        mask_lang = clusters_to_mask(clusters_lang, shape)

    out: Dict[str, float] = {}

    # ---- mask IoU --------------------------------------------------------
    out["iou_cluster_mask"] = mask_iou(mask_ref, mask_lang)

    # ---- threshold IoU at several taus -----------------------------------
    for tau in thresholds:
        out[f"iou_thresh_{tau:.2f}"] = mask_iou(
            threshold_mask(sim_ref,  tau=tau),
            threshold_mask(sim_lang, tau=tau),
        )

    # ---- threshold IoU at percentiles (scale-invariant) ------------------
    for p in percentiles:
        out[f"iou_pct_{int(p)}"] = mask_iou(
            threshold_mask(sim_ref,  percentile=p),
            threshold_mask(sim_lang, percentile=p),
        )

    # ---- raw-signal correlations -----------------------------------------
    out["spearman"] = rank_correlation(sim_ref, sim_lang)
    out["pearson"]  = linear_correlation(sim_ref, sim_lang)

    # ---- per-map stats ---------------------------------------------------
    out["peak_ref"]  = peak_similarity(sim_ref)
    out["peak_lang"] = peak_similarity(sim_lang)
    out["mean_ref"]  = mean_similarity(sim_ref)
    out["mean_lang"] = mean_similarity(sim_lang)
    out["peak_ratio_lang_over_ref"] = (
        out["peak_lang"] / out["peak_ref"]
        if out["peak_ref"] > 1e-9 else float("nan")
    )

    # ---- cluster counts & geometry ---------------------------------------
    out["n_clusters_ref"]  = float(len(clusters_ref))
    out["n_clusters_lang"] = float(len(clusters_lang))
    out["center_dist_symmetric"] = symmetric_center_distance(clusters_ref,
                                                             clusters_lang)

    return out


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_by_language(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Given a list of per-(image, concept, lang) metric dicts (each containing a
    ``language`` key), return {lang: {metric: mean_over_rows}}.
    """
    buckets: Dict[str, List[Dict]] = {}
    for r in rows:
        buckets.setdefault(r["language"], []).append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for lang, rs in buckets.items():
        keys = [k for k in rs[0].keys()
                if isinstance(rs[0][k], (int, float)) and k != "language"]
        summary[lang] = {
            k: float(np.nanmean([r[k] for r in rs])) for k in keys
        }
        summary[lang]["n_samples"] = float(len(rs))
    return summary


def double_penalty_contrast(summary: Dict[str, Dict[str, float]],
                            low_resource_langs: List[str],
                            high_resource_langs: List[str],
                            metric: str = "iou_cluster_mask") -> Dict[str, float]:
    """
    Simple contrast: mean(metric on HR langs) − mean(metric on LR langs).
    Positive ⇒ the "double penalty" (lower IoU for LR languages) is present.
    """
    hr = [summary[l][metric] for l in high_resource_langs if l in summary]
    lr = [summary[l][metric] for l in low_resource_langs  if l in summary]
    return {
        "metric":               metric,
        "mean_high_resource":   float(np.mean(hr)) if hr else float("nan"),
        "mean_low_resource":    float(np.mean(lr)) if lr else float("nan"),
        "hr_minus_lr":          (float(np.mean(hr)) - float(np.mean(lr)))
                                if hr and lr else float("nan"),
        "n_hr_langs":           len(hr),
        "n_lr_langs":           len(lr),
    }
