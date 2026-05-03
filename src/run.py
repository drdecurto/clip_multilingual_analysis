#!/usr/bin/env python3
"""
Main runner for language-conditioned dense-CLIP grounding on BDD100K.

Workflow per image
------------------
  1. Preprocess once.
  2. Extract dense patch features once.              ← the visual encoder
     work is *shared* across all 13 languages, so the text-conditioning
     comparison is cheap.
  3. For each (concept, language):
        * encode text  (cached — see MultilingualDenseCLIP._text_cache)
        * compute dense similarity map
        * threshold the valid region
        * cluster via OneMap's region-growing algorithm
        * record the full metric panel vs the English reference
  4. Aggregate per-language stats and save.

Output
------
    <output_dir>/
        per_sample_records.jsonl       # one line per (image, concept, language)
        per_language_summary.json      # mean-of-metrics per language
        per_concept_summary.json       # mean-of-metrics per concept
        language_concept_iou.json      # language × concept cell grid
        energy.json                    # NVML stats, total Wh, Wh/1K queries
        config.json                    # reproducibility manifest

Usage
-----
    python -m src.run \
        BDD100K_210_samples_v1.json \
        --output-dir results/dense_clip_xlmr_base/ \
        --n-images 50 \
        --concepts car,truck,bus,person,pedestrian,traffic_light,traffic_sign,\
                   bicycle,motorcycle,road,building

Arguments default to the paper configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Local imports
try:
    from .clip_dense_multilingual import (
        DEFAULT_MODEL, DEFAULT_PRETRAINED, MultilingualDenseCLIP,
    )
    from .clustering import (
        Cluster, cluster_high_similarity_regions, clusters_to_mask,
    )
    from .energy_monitor import GPUEnergySampler
    from .iou_metrics import (
        aggregate_by_language, double_penalty_contrast, summarise_pair,
    )
    from .multilingual_queries import (
        ALL_CONCEPTS, ALL_LANGUAGES, LANGUAGE_INFO, LOW_RESOURCE_LANGS,
        all_queries, render_query,
    )
except ImportError:
    # Allow invocation as `python src/run.py ...`
    from clip_dense_multilingual import (
        DEFAULT_MODEL, DEFAULT_PRETRAINED, MultilingualDenseCLIP,
    )
    from clustering import (
        Cluster, cluster_high_similarity_regions, clusters_to_mask,
    )
    from energy_monitor import GPUEnergySampler
    from iou_metrics import (
        aggregate_by_language, double_penalty_contrast, summarise_pair,
    )
    from multilingual_queries import (
        ALL_CONCEPTS, ALL_LANGUAGES, LANGUAGE_INFO, LOW_RESOURCE_LANGS,
        all_queries, render_query,
    )


REFERENCE_LANGUAGE = "en"   # the double-penalty contrast is always vs English


# =============================================================================
# Dataset loading
# =============================================================================

def load_frozen_paths(frozen_file: str, n_images: Optional[int] = None) -> List[str]:
    with open(frozen_file) as f:
        data = json.load(f)
    paths = data["image_paths"]
    if n_images is not None:
        paths = paths[:n_images]
    print(f"📦 Frozen dataset: {frozen_file}")
    print(f"   {len(paths)} images  |  seed={data.get('seed', 'n/a')}  "
          f"|  version={data.get('version', 'n/a')}")
    return paths


# =============================================================================
# Single-image pipeline
# =============================================================================

def compute_language_similarity_maps(
        clip: MultilingualDenseCLIP,
        image: Image.Image,
        concepts: List[str],
        languages: List[str],
        template: str,
        up_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns {concept: {lang: sim_map_numpy[H_up, W_up]}}.

    The visual forward is performed **once**; text features for every
    (concept, language) pair are encoded (with caching) and dotted in.
    """
    x = clip.preprocess_image(image)
    dense = clip.encode_image_dense(x)  # patch_feats: [1, D, H_p, W_p]
    H_in, W_in = dense.input_hw
    H_p, W_p = dense.patch_hw

    if up_hw is None:
        # Upsample to the preprocessed input size (224×224 typically). This
        # gives the clustering algorithm something denser than the raw 7×7
        # or 16×16 patch grid.
        up_hw = (H_in, W_in)

    # Flatten all texts for this image into one encode_text batch.
    flat_texts:  List[str] = []
    flat_index:  List[Tuple[str, str]] = []
    for c in concepts:
        for l in languages:
            flat_texts.append(render_query(c, l, template))
            flat_index.append((c, l))

    text_feats = clip.encode_text(flat_texts)   # [T, D]
    sim = clip.similarity(dense.patch_feats, text_feats)    # [1, T, H_p, W_p]
    sim_up = clip.upsample_to(sim, up_hw)                   # [1, T, H, W]
    sim_np = sim_up[0].cpu().float().numpy()                # [T, H, W]

    out: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in concepts}
    for t_idx, (c, l) in enumerate(flat_index):
        out[c][l] = sim_np[t_idx]
    return out


def cluster_and_score_pair(
        sim_ref:  np.ndarray,
        sim_lang: np.ndarray,
        cluster_kwargs: Dict,
        cluster_percentile_mask: float = 75.0,
) -> Dict[str, float]:
    """
    Given a pair of similarity maps (reference and target language), compute
    clusters on each and return the full cross-language metric dict.

    ``cluster_percentile_mask``: the eligibility mask for the clustering
    algorithm is ``sim ≥ np.percentile(sim, p)``, i.e. only the top (100-p)%
    of pixels can participate.  Scale-invariant per map (the two languages
    may have different absolute similarity ranges).
    """
    mask_ref  = sim_ref  >= np.percentile(sim_ref,  cluster_percentile_mask)
    mask_lang = sim_lang >= np.percentile(sim_lang, cluster_percentile_mask)
    c_ref  = cluster_high_similarity_regions(sim_ref,  mask_ref,  **cluster_kwargs)
    c_lang = cluster_high_similarity_regions(sim_lang, mask_lang, **cluster_kwargs)
    return summarise_pair(sim_ref, sim_lang, c_ref, c_lang,
                          mask_ref=clusters_to_mask(c_ref,  sim_ref.shape),
                          mask_lang=clusters_to_mask(c_lang, sim_lang.shape))


# =============================================================================
# Full run
# =============================================================================

def run(args: argparse.Namespace) -> None:

    # ---- setup -----------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    concepts   = args.concepts.split(",") if args.concepts else ALL_CONCEPTS
    languages  = args.languages.split(",") if args.languages else ALL_LANGUAGES
    if REFERENCE_LANGUAGE not in languages:
        languages = [REFERENCE_LANGUAGE] + languages
        print(f"⚠️  Reference language '{REFERENCE_LANGUAGE}' auto-added.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip = MultilingualDenseCLIP(
        model_name = args.model,
        pretrained = args.pretrained,
        device     = device,
        precision  = args.precision,
    )

    cluster_kwargs = dict(
        neighborhood_size  = args.cluster_nbhd,
        min_cluster_size   = args.cluster_min_size,
        relative_threshold = args.cluster_rel_thresh,
    )

    up_hw = (args.upsample_size, args.upsample_size)

    # ---- reproducibility manifest ---------------------------------------
    config = {
        "frozen_json":  args.frozen_json,
        "n_images":     args.n_images,
        "concepts":     concepts,
        "languages":    languages,
        "template":     args.template,
        "model":        args.model,
        "pretrained":   args.pretrained,
        "precision":    args.precision,
        "cluster":      cluster_kwargs,
        "cluster_percentile_mask": args.cluster_percentile_mask,
        "upsample_hw":  up_hw,
        "reference_language": REFERENCE_LANGUAGE,
        "device":       device,
        "torch_version": torch.__version__,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---- dataset ---------------------------------------------------------
    paths = load_frozen_paths(args.frozen_json, args.n_images)

    # ---- run loop --------------------------------------------------------
    records: List[Dict] = []
    per_lang_concept: Dict[str, Dict[str, List[float]]] = {
        l: {c: [] for c in concepts} for l in languages
    }
    n_queries_run = 0   # total text-conditioned similarity evaluations

    records_path = os.path.join(args.output_dir, "per_sample_records.jsonl")
    records_file = open(records_path, "w")

    with GPUEnergySampler(interval_sec=args.energy_interval) as sampler:
        t0 = time.time()
        for img_idx, path in enumerate(tqdm(paths, desc="Images")):
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"⚠️  unreadable image {path}: {e}")
                continue

            # Dense features + every (concept, lang) similarity map for this image
            sims = compute_language_similarity_maps(
                clip, image, concepts, languages,
                template=args.template, up_hw=up_hw,
            )

            # Score every non-reference language against the reference
            for concept in concepts:
                ref_sim = sims[concept][REFERENCE_LANGUAGE]
                for lang in languages:
                    n_queries_run += 1
                    lang_sim = sims[concept][lang]

                    # The reference vs itself is useful as a sanity check
                    metrics = cluster_and_score_pair(
                        ref_sim, lang_sim,
                        cluster_kwargs=cluster_kwargs,
                        cluster_percentile_mask=args.cluster_percentile_mask,
                    )
                    row = {
                        "image_idx":   img_idx,
                        "image_path":  path,
                        "concept":     concept,
                        "language":    lang,
                        **metrics,
                    }
                    records_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    records.append(row)
                    per_lang_concept[lang][concept].append(
                        metrics["iou_cluster_mask"]
                    )

            if (img_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                per_img = elapsed / (img_idx + 1)
                remaining = per_img * (len(paths) - img_idx - 1)
                print(f"   [{img_idx + 1}/{len(paths)}]  "
                      f"{per_img:.1f} s/img  ETA {remaining/60:.1f} min")

    records_file.close()
    print(f"📝 Wrote per-sample records → {records_path}  ({len(records)} rows)")

    # ---- aggregations ----------------------------------------------------

    # per language
    per_lang = aggregate_by_language(records)
    with open(os.path.join(args.output_dir, "per_language_summary.json"), "w") as f:
        json.dump(per_lang, f, indent=2, ensure_ascii=False)

    # per concept
    per_concept: Dict[str, Dict[str, float]] = {}
    for concept in concepts:
        rs = [r for r in records if r["concept"] == concept]
        per_concept[concept] = aggregate_by_language(rs)
    with open(os.path.join(args.output_dir, "per_concept_summary.json"), "w") as f:
        json.dump(per_concept, f, indent=2, ensure_ascii=False)

    # language × concept IoU grid
    grid: Dict[str, Dict[str, float]] = {
        l: {
            c: (float(np.mean(per_lang_concept[l][c]))
                if per_lang_concept[l][c] else float("nan"))
            for c in concepts
        }
        for l in languages
    }
    with open(os.path.join(args.output_dir, "language_concept_iou.json"), "w") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)

    # double-penalty contrast on the primary metric
    high_resource = [l for l in languages
                     if l != REFERENCE_LANGUAGE and l not in LOW_RESOURCE_LANGS]
    low_resource  = [l for l in languages if l in LOW_RESOURCE_LANGS]
    contrasts = {}
    for m in ("iou_cluster_mask", "iou_pct_95", "spearman", "peak_lang"):
        contrasts[m] = double_penalty_contrast(
            per_lang, low_resource, high_resource, metric=m,
        )
    with open(os.path.join(args.output_dir, "double_penalty.json"), "w") as f:
        json.dump(contrasts, f, indent=2, ensure_ascii=False)

    # energy
    stats = sampler.stats()
    stats["n_queries"] = n_queries_run
    stats["wh_per_1k_queries"] = (
        sampler.total_wh() / max(n_queries_run, 1) * 1000.0
    )
    stats["images_evaluated"] = len(paths)
    stats["languages"]        = languages
    stats["concepts"]         = concepts
    with open(os.path.join(args.output_dir, "energy.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- console summary -------------------------------------------------
    print("\n=== PER-LANGUAGE MEAN IoU (vs English) ==========================")
    hdr_metric = "iou_cluster_mask"
    for lang in languages:
        iou = per_lang.get(lang, {}).get(hdr_metric, float("nan"))
        fam = LANGUAGE_INFO[lang]["family"]
        tag = "⚠️ LR" if lang in LOW_RESOURCE_LANGS else "   "
        print(f"  {lang:6s}  {LANGUAGE_INFO[lang]['name']:22s}  {fam:10s}"
              f"  {tag}   iou={iou:.3f}")

    print("\n=== DOUBLE-PENALTY CONTRAST (HR − LR, primary metric) ===========")
    print(json.dumps(contrasts["iou_cluster_mask"], indent=2))

    print(f"\n=== ENERGY =====================================================")
    print(f"  total Wh        : {stats.get('total_wh', float('nan')):.3f}")
    print(f"  queries         : {n_queries_run}")
    print(f"  Wh / 1K queries : {stats['wh_per_1k_queries']:.3f}")
    print(f"  avg GPU watts   : {stats.get('avg_watts', float('nan')):.1f}")
    print(f"  duration (s)    : {stats.get('duration_sec', float('nan')):.1f}")
    print(f"\n✅ Results → {args.output_dir}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Language-conditioned dense-CLIP grounding on BDD100K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("frozen_json",
                   help="Path to frozen BDD100K subset JSON (from freeze_dataset.py)")
    p.add_argument("--output-dir", default="results/dense_clip_run",
                   help="Directory for all output JSONs")
    p.add_argument("--n-images", type=int, default=None,
                   help="Limit to first N images (None = all)")

    # concepts & languages
    p.add_argument("--concepts", default=",".join(ALL_CONCEPTS),
                   help="Comma-separated concept list")
    p.add_argument("--languages", default=",".join(ALL_LANGUAGES),
                   help="Comma-separated language list")
    p.add_argument("--template", default="indef", choices=["bare", "indef"],
                   help="Query rendering style")

    # model
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--precision",  default="fp32",
                   choices=["fp32", "fp16", "bf16"])

    # clustering knobs (match OneMap defaults)
    p.add_argument("--cluster-nbhd",       type=int,   default=10)
    p.add_argument("--cluster-min-size",   type=int,   default=3)
    p.add_argument("--cluster-rel-thresh", type=float, default=0.8)
    p.add_argument("--cluster-percentile-mask", type=float, default=75.0,
                   help="Pixels below this similarity percentile are ineligible")

    # upsampling
    p.add_argument("--upsample-size", type=int, default=224,
                   help="Square resolution to which similarity maps are upsampled")

    # energy
    p.add_argument("--energy-interval", type=float, default=0.1,
                   help="NVML sampling interval in seconds (10 Hz default)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
