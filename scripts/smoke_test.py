#!/usr/bin/env python3
"""
Smoke test — validates the clustering / IoU / aggregation pipeline with
synthetic similarity maps. Does NOT require open_clip, a GPU, or BDD100K.

Synthetic setup
---------------
We fake a driving scene where the "true" object sits near (h=0.4, w=0.5)
of the similarity map. For each language we generate a noisy blob centred
at the true location, but with simulated degradation for the three
low-resource languages (Arabic, Basque, Luxembourgish): lower peak, higher
noise, slight centroid offset. The pipeline should then recover a lower
mean IoU for those languages than for the high-resource ones — that is
*exactly* the "double penalty" signature we expect to find in real data.

Run:
    python scripts/smoke_test.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from src.clustering import cluster_high_similarity_regions, clusters_to_mask  # noqa
from src.iou_metrics import (aggregate_by_language, double_penalty_contrast,  # noqa
                              summarise_pair)
from src.multilingual_queries import (ALL_CONCEPTS, ALL_LANGUAGES,  # noqa
                                       LANGUAGE_INFO, LOW_RESOURCE_LANGS,
                                       render_query)


def synthetic_similarity_map(
        H: int, W: int,
        cy: float, cx: float,
        peak: float,
        sigma: float,
        noise: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """2D Gaussian blob + isotropic noise, clipped to [-1, 1]."""
    yy, xx = np.mgrid[0:H, 0:W]
    cy_px, cx_px = cy * H, cx * W
    r2 = (yy - cy_px) ** 2 + (xx - cx_px) ** 2
    blob = peak * np.exp(-r2 / (2.0 * sigma ** 2))
    n = rng.normal(0.0, noise, size=(H, W))
    return np.clip(blob + n, -1.0, 1.0).astype(np.float32)


def language_quality_profile(lang: str) -> dict:
    """Per-language degradation parameters for synthesis."""
    if lang == "en":
        return {"peak": 0.35, "sigma_frac": 0.10, "noise": 0.03,
                "dy": 0.0, "dx": 0.0}
    if lang in LOW_RESOURCE_LANGS:
        return {"peak": 0.22, "sigma_frac": 0.13, "noise": 0.06,
                "dy": 0.04, "dx": 0.04}
    # other high-resource languages
    return {"peak": 0.32, "sigma_frac": 0.10, "noise": 0.035,
            "dy": 0.01, "dx": 0.01}


def main() -> None:
    rng = np.random.default_rng(42)
    H = W = 64
    n_images = 12
    concepts = ["car", "pedestrian", "traffic_light"]  # shortlist for speed
    languages = list(ALL_LANGUAGES)
    ref = "en"

    print(f"Smoke test — {n_images} images × {len(concepts)} concepts × "
          f"{len(languages)} languages")
    print(f"  Low-resource (expected to underperform): {LOW_RESOURCE_LANGS}")
    print()

    cluster_kwargs = dict(
        neighborhood_size=5, min_cluster_size=3, relative_threshold=0.8,
    )

    records = []
    for img_idx in range(n_images):
        # "Truth" location for this image (random)
        cy = float(rng.uniform(0.3, 0.7))
        cx = float(rng.uniform(0.3, 0.7))

        for concept in concepts:
            # Per-concept slight jitter to avoid identical images
            jit = rng.normal(0, 0.01)

            # Reference map
            prof_ref = language_quality_profile(ref)
            sim_ref = synthetic_similarity_map(
                H, W,
                cy + jit + prof_ref["dy"],
                cx + jit + prof_ref["dx"],
                peak=prof_ref["peak"] + rng.normal(0, 0.02),
                sigma=prof_ref["sigma_frac"] * H,
                noise=prof_ref["noise"],
                rng=rng,
            )
            mask_ref = sim_ref >= np.percentile(sim_ref, 75)
            clusters_ref = cluster_high_similarity_regions(
                sim_ref, mask_ref, **cluster_kwargs,
            )

            for lang in languages:
                prof = language_quality_profile(lang)
                # Per-(image, concept, language) realisation
                sim_lang = synthetic_similarity_map(
                    H, W,
                    cy + jit + prof["dy"] + rng.normal(0, 0.005),
                    cx + jit + prof["dx"] + rng.normal(0, 0.005),
                    peak=prof["peak"] + rng.normal(0, 0.02),
                    sigma=prof["sigma_frac"] * H,
                    noise=prof["noise"],
                    rng=rng,
                )
                mask_lang = sim_lang >= np.percentile(sim_lang, 75)
                clusters_lang = cluster_high_similarity_regions(
                    sim_lang, mask_lang, **cluster_kwargs,
                )

                metrics = summarise_pair(sim_ref, sim_lang,
                                         clusters_ref, clusters_lang)
                records.append({
                    "image_idx": img_idx,
                    "concept":   concept,
                    "language":  lang,
                    **metrics,
                })

    # Aggregate and report
    summary = aggregate_by_language(records)

    print("=== PER-LANGUAGE MEAN IoU (vs English) ===")
    for lang in languages:
        info = LANGUAGE_INFO[lang]
        tag = "⚠️ LR" if lang in LOW_RESOURCE_LANGS else "     "
        stats = summary[lang]
        print(f"  {lang:6s} {info['name']:22s} {info['family']:10s} {tag}"
              f"  IoU={stats['iou_cluster_mask']:.3f}"
              f"  ρ_spearman={stats['spearman']:.3f}"
              f"  peak={stats['peak_lang']:.3f}")

    print("\n=== DOUBLE-PENALTY CONTRAST ===")
    hr = [l for l in languages if l not in LOW_RESOURCE_LANGS and l != ref]
    lr = LOW_RESOURCE_LANGS
    for metric in ("iou_cluster_mask", "spearman", "peak_lang"):
        c = double_penalty_contrast(summary, lr, hr, metric=metric)
        print(f"  {metric:18s}  HR={c['mean_high_resource']:.3f}  "
              f"LR={c['mean_low_resource']:.3f}  Δ={c['hr_minus_lr']:+.3f}")

    # Sanity: synthesized pattern should recover the double penalty
    iou_hr = np.mean([summary[l]["iou_cluster_mask"] for l in hr])
    iou_lr = np.mean([summary[l]["iou_cluster_mask"] for l in lr])
    print()
    if iou_hr > iou_lr:
        print(f"✅  HR IoU ({iou_hr:.3f}) > LR IoU ({iou_lr:.3f}) "
              f"— double-penalty signature recovered from synthetic data.")
    else:
        print(f"❌  HR IoU ({iou_hr:.3f}) ≤ LR IoU ({iou_lr:.3f}) "
              f"— check synthesis / clustering defaults.")
        sys.exit(1)

    # Also verify the query module
    print("\n=== SAMPLE RENDERED QUERIES ===")
    for c in concepts:
        for lang in ("en", "es", "ar", "eu", "lb", "zh-CN"):
            print(f"  {c:14s} [{lang:5s}] → '{render_query(c, lang)}'")


if __name__ == "__main__":
    main()
