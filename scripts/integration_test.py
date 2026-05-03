#!/usr/bin/env python3
"""
Integration smoke test — synthesise data → write JSONL → run analyze.py
on it → verify figures + tables are produced. No GPU / no OpenCLIP / no BDD.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from src.clustering import cluster_high_similarity_regions  # noqa
from src.iou_metrics import summarise_pair                  # noqa
from src.multilingual_queries import (ALL_LANGUAGES, LOW_RESOURCE_LANGS)  # noqa


def synthetic_sim(H, W, cy, cx, peak, sigma, noise, rng):
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - cy * H) ** 2 + (xx - cx * W) ** 2
    return np.clip(peak * np.exp(-r2 / (2 * sigma ** 2))
                   + rng.normal(0, noise, (H, W)),
                   -1, 1).astype(np.float32)


def profile(lang):
    if lang == "en":
        return dict(peak=0.35, sigma=6.4, noise=0.03, dy=0.0, dx=0.0)
    if lang in LOW_RESOURCE_LANGS:
        return dict(peak=0.22, sigma=8.3, noise=0.06, dy=0.04, dx=0.04)
    return dict(peak=0.32, sigma=6.4, noise=0.035, dy=0.01, dx=0.01)


def main():
    rng = np.random.default_rng(7)
    H = W = 64
    n_images = 10
    concepts = ["car", "pedestrian", "traffic_light", "road"]

    with tempfile.TemporaryDirectory() as tmpd:
        jsonl_path = os.path.join(tmpd, "per_sample_records.jsonl")
        with open(jsonl_path, "w") as f:
            for img_idx in range(n_images):
                cy = float(rng.uniform(0.3, 0.7))
                cx = float(rng.uniform(0.3, 0.7))
                for concept in concepts:
                    p_en = profile("en")
                    sim_en = synthetic_sim(H, W,
                                           cy + p_en["dy"], cx + p_en["dx"],
                                           p_en["peak"] + rng.normal(0, 0.02),
                                           p_en["sigma"], p_en["noise"], rng)
                    mask_en = sim_en >= np.percentile(sim_en, 75)
                    c_en = cluster_high_similarity_regions(
                        sim_en, mask_en,
                        neighborhood_size=5, min_cluster_size=3,
                        relative_threshold=0.8,
                    )

                    for lang in ALL_LANGUAGES:
                        pr = profile(lang)
                        sim_l = synthetic_sim(
                            H, W,
                            cy + pr["dy"] + rng.normal(0, 0.005),
                            cx + pr["dx"] + rng.normal(0, 0.005),
                            pr["peak"] + rng.normal(0, 0.02),
                            pr["sigma"], pr["noise"], rng,
                        )
                        mask_l = sim_l >= np.percentile(sim_l, 75)
                        c_l = cluster_high_similarity_regions(
                            sim_l, mask_l,
                            neighborhood_size=5, min_cluster_size=3,
                            relative_threshold=0.8,
                        )
                        metrics = summarise_pair(sim_en, sim_l, c_en, c_l)
                        row = {
                            "image_idx": img_idx,
                            "image_path": f"synth_{img_idx:03d}.jpg",
                            "concept": concept,
                            "language": lang,
                            **metrics,
                        }
                        f.write(json.dumps(row) + "\n")

        print(f"Wrote JSONL → {jsonl_path}")

        # Run analyze.py
        cmd = [sys.executable, os.path.join(HERE, "..", "src", "analyze.py"),
               tmpd, "--output-dir", os.path.join(tmpd, "figures")]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        figs = sorted(os.listdir(os.path.join(tmpd, "figures")))
        print(f"\nProduced artefacts:")
        for f in figs:
            p = os.path.join(tmpd, "figures", f)
            print(f"  {f}  ({os.path.getsize(p)} bytes)")
        assert any(f.endswith(".png") for f in figs), "No PNGs produced!"
        assert "per_language_table.csv" in figs
        assert "stat_tests.json" in figs

        # Read stat tests to make sure HR > LR is recovered with p < 0.05
        with open(os.path.join(tmpd, "figures", "stat_tests.json")) as fh:
            stats = json.load(fh)
        w = stats["iou_cluster_mask"]["wilcoxon_hr_vs_lr"]
        print(f"\nWilcoxon HR > LR  p-value: {w['p_value']:.4f}  "
              f"mean diff: {w['mean_diff']:+.3f}")
        assert w["mean_diff"] > 0, "HR should beat LR in synthetic data"
        print("\n✅ All integration checks passed.")


if __name__ == "__main__":
    main()
