#!/usr/bin/env python3
"""
Produce a qualitative figure: one image + per-language similarity heatmaps
for a chosen concept. Intended for paper figures / invited-talk slides.

Usage
-----
    python scripts/visualize_heatmaps.py \
        --image path/to/bdd_image.jpg \
        --concept car \
        --languages en,es,fr,de,ar,eu,lb,zh-CN \
        --output out.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from src.clip_dense_multilingual import MultilingualDenseCLIP  # noqa: E402
from src.multilingual_queries import LANGUAGE_INFO, render_query  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True, help="Path to a single image")
    p.add_argument("--concept",  default="car",
                   help="Concept key (see multilingual_queries.CONCEPTS_BARE)")
    p.add_argument("--languages", default="en,es,fr,de,ar,eu,lb,zh-CN",
                   help="Comma-separated languages to visualise")
    p.add_argument("--template", default="indef", choices=["bare", "indef"])
    p.add_argument("--model",      default="xlm-roberta-base-ViT-B-32")
    p.add_argument("--pretrained", default="laion5b_s13b_b90k")
    p.add_argument("--output",   default="qualitative_heatmaps.png")
    p.add_argument("--cmap",     default="inferno")
    args = p.parse_args()

    langs = args.languages.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip = MultilingualDenseCLIP(
        model_name=args.model, pretrained=args.pretrained, device=device,
    )

    image = Image.open(args.image).convert("RGB")
    x = clip.preprocess_image(image)
    out = clip.encode_image_dense(x)

    queries = [render_query(args.concept, l, args.template) for l in langs]
    text_feats = clip.encode_text(queries)              # [L, D]
    sim = clip.similarity(out.patch_feats, text_feats)  # [1, L, H_p, W_p]
    H_up, W_up = out.input_hw
    sim_up = clip.upsample_to(sim, (H_up, W_up))[0].cpu().numpy()  # [L, H, W]

    # Normalise each map to [0,1] for display only — actual IoU used raw sims.
    def norm01(a: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    cols = len(langs) + 1
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.6))

    axes[0].imshow(np.array(image.resize((W_up, H_up))))
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    for i, lang in enumerate(langs):
        a = axes[i + 1]
        a.imshow(np.array(image.resize((W_up, H_up))), alpha=0.45)
        a.imshow(norm01(sim_up[i]), cmap=args.cmap, alpha=0.55)
        name = LANGUAGE_INFO[lang]["name"]
        q = queries[i]
        lr = " ⚠️" if LANGUAGE_INFO[lang]["low_resource"] else ""
        peak = float(sim_up[i].max())
        a.set_title(f"{lang} — {name}{lr}\n'{q}'  peak={peak:.2f}",
                    fontsize=9)
        a.axis("off")

    fig.suptitle(f"Language-conditioned dense-CLIP similarity — concept: "
                 f"'{args.concept}'", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"✅ Wrote {args.output}")


if __name__ == "__main__":
    main()
