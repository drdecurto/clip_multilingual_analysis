#!/usr/bin/env python3
"""
Offline analysis — reads ``per_sample_records.jsonl`` and produces the tables
and figures for the paper (MDPI-compatible PNG/PDF at 300 dpi).

Usage
-----
    python src/analyze.py  <results_dir>   [--output-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    from src.multilingual_queries import (ALL_LANGUAGES, LANGUAGE_INFO,
                                           LOW_RESOURCE_LANGS)
except ImportError:
    from multilingual_queries import (ALL_LANGUAGES, LANGUAGE_INFO,
                                       LOW_RESOURCE_LANGS)


# =============================================================================
# Palette for language families (used across figures)
# =============================================================================

FAMILY_COLOURS = {
    "Germanic": "#4472C4",
    "Romance":  "#ED7D31",
    "Slavic":   "#70AD47",
    "Sinitic":  "#FFC000",
    "Semitic":  "#C00000",
    "Isolate":  "#7030A0",
}


# =============================================================================
# Data loading
# =============================================================================

def load_records(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "per_sample_records.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["family"] = df["language"].map(
        lambda l: LANGUAGE_INFO[l]["family"] if l in LANGUAGE_INFO else "Other"
    )
    df["low_resource"] = df["language"].isin(LOW_RESOURCE_LANGS)
    print(f"Loaded {len(df)} rows  "
          f"| {df['image_idx'].nunique()} images "
          f"| {df['concept'].nunique()} concepts "
          f"| {df['language'].nunique()} languages")
    return df


# =============================================================================
# Tables
# =============================================================================

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["iou_cluster_mask", "iou_pct_95", "spearman", "pearson",
               "peak_lang", "mean_lang", "n_clusters_lang",
               "center_dist_symmetric"]
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby("language")
    rows = []
    for lang, sub in g:
        row = {"language": lang,
               "name":     LANGUAGE_INFO[lang]["name"],
               "family":   LANGUAGE_INFO[lang]["family"],
               "low_resource": lang in LOW_RESOURCE_LANGS,
               "n":        len(sub)}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean(skipna=True))
            row[f"{m}_std"]  = float(sub[m].std(skipna=True))
        rows.append(row)
    tbl = pd.DataFrame(rows)
    # order: English first, then by family, alphabetical within
    tbl["_ref"] = tbl["language"] != "en"
    tbl = tbl.sort_values(["_ref", "family", "language"]).drop(columns=["_ref"])
    return tbl


# =============================================================================
# Statistical tests
# =============================================================================

def run_stat_tests(df: pd.DataFrame, metric: str = "iou_cluster_mask") -> Dict:
    """Double-penalty contrast tests on ``metric``."""
    langs = sorted(df["language"].unique())
    langs = [l for l in langs if l != "en"]        # reference excluded

    # Pivot so each row = (image, concept) and columns are languages
    pivot = df.pivot_table(index=["image_idx", "concept"],
                           columns="language", values=metric,
                           aggfunc="mean")
    pivot = pivot[langs].dropna(how="any")

    out = {"metric": metric,
           "n_paired_observations": int(len(pivot))}

    # Friedman across all non-reference languages
    if len(langs) >= 3 and len(pivot) >= 3:
        stat, p = friedmanchisquare(*[pivot[l].values for l in langs])
        out["friedman"] = {"statistic": float(stat), "p_value": float(p)}

    # HR vs LR — paired Wilcoxon across (image, concept)
    hr_langs = [l for l in langs if l not in LOW_RESOURCE_LANGS]
    lr_langs = [l for l in langs if l in LOW_RESOURCE_LANGS]
    if hr_langs and lr_langs:
        hr_avg = pivot[hr_langs].mean(axis=1).values
        lr_avg = pivot[lr_langs].mean(axis=1).values
        try:
            stat, p = wilcoxon(hr_avg, lr_avg, alternative="greater")
            out["wilcoxon_hr_vs_lr"] = {
                "alternative":        "HR > LR",
                "mean_diff":          float(np.mean(hr_avg - lr_avg)),
                "median_diff":        float(np.median(hr_avg - lr_avg)),
                "statistic":          float(stat),
                "p_value":            float(p),
            }
        except ValueError as e:
            out["wilcoxon_hr_vs_lr"] = {"error": str(e)}

    # Per-low-resource vs pooled-HR
    per_lr = {}
    for l in lr_langs:
        try:
            stat, p = mannwhitneyu(
                pivot[hr_langs].values.flatten(),
                pivot[l].values,
                alternative="greater",
            )
            per_lr[l] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as e:
            per_lr[l] = {"error": str(e)}
    out["per_low_resource_vs_hr_pool"] = per_lr
    return out


# =============================================================================
# Plots
# =============================================================================

def plot_language_bar(df: pd.DataFrame, out_path: str,
                       metric: str = "iou_cluster_mask") -> None:
    order = [l for l in ALL_LANGUAGES if l in df["language"].unique()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    means, stds, cols, tags = [], [], [], []
    for l in order:
        vals = df.loc[df["language"] == l, metric].dropna().values
        means.append(float(vals.mean()))
        stds.append(float(vals.std() / np.sqrt(max(len(vals), 1))))  # SE
        cols.append(FAMILY_COLOURS.get(LANGUAGE_INFO[l]["family"], "#666"))
        tags.append("⚠️" if l in LOW_RESOURCE_LANGS else "")
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, color=cols, edgecolor="black", linewidth=0.6,
           capsize=3)
    for xi, (m, tag) in enumerate(zip(means, tags)):
        if tag:
            ax.text(xi, m + 0.02, tag, ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-language {metric} (mean ± SE) — vs English reference")
    ax.axhline(np.mean(means), color="gray", lw=0.8, ls="--", alpha=0.6,
               label=f"overall mean = {np.mean(means):.3f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # family legend
    for fam, col in FAMILY_COLOURS.items():
        ax.bar([-1], [0], color=col, label=fam, edgecolor="black", linewidth=0.4)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_language_concept_heatmap(df: pd.DataFrame, out_path: str,
                                   metric: str = "iou_cluster_mask") -> None:
    pivot = df.pivot_table(index="language", columns="concept",
                           values=metric, aggfunc="mean")
    # Reorder rows
    order = [l for l in ALL_LANGUAGES if l in pivot.index]
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(max(8, 1 + 0.6 * len(pivot.columns)),
                                    max(4, 0.35 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": metric}, ax=ax, linewidths=0.3,
                linecolor="white")
    # Highlight low-resource rows
    for i, lang in enumerate(pivot.index):
        if lang in LOW_RESOURCE_LANGS:
            ax.add_patch(plt.Rectangle((0, i), pivot.shape[1], 1, fill=False,
                                       edgecolor="red", lw=1.8, clip_on=False))
    ax.set_title(f"{metric} — language × concept")
    ax.set_xlabel("Concept")
    ax.set_ylabel("Language")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_iou_vs_peak(df: pd.DataFrame, out_path: str) -> None:
    agg = (df.groupby("language")
             .agg(iou=("iou_cluster_mask", "mean"),
                  peak=("peak_lang", "mean"))
             .reset_index())
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in agg.iterrows():
        fam = LANGUAGE_INFO[row["language"]]["family"]
        col = FAMILY_COLOURS.get(fam, "#666")
        marker = "D" if row["language"] in LOW_RESOURCE_LANGS else "o"
        size   = 200 if row["language"] in LOW_RESOURCE_LANGS else 120
        ax.scatter(row["peak"], row["iou"], color=col, s=size, marker=marker,
                   edgecolor="black", linewidth=0.6,
                   label=fam if fam not in ax.get_legend_handles_labels()[1] else None)
        ax.text(row["peak"], row["iou"] + 0.005, row["language"],
                ha="center", fontsize=9)
    ax.set_xlabel("mean peak similarity (target language)")
    ax.set_ylabel("mean cluster-mask IoU (vs English)")
    ax.set_title("Dense-CLIP: peak-similarity vs cross-language IoU")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    u_handles, u_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            u_handles.append(h); u_labels.append(l)
    ax.legend(u_handles, u_labels, fontsize=8, loc="lower right",
              title="Language family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_family_violin(df: pd.DataFrame, out_path: str,
                        metric: str = "iou_cluster_mask") -> None:
    sub = df[df["language"] != "en"].copy()
    order = ["Germanic", "Romance", "Slavic", "Sinitic", "Semitic", "Isolate"]
    present = [f for f in order if f in sub["family"].unique()]
    palette = {f: FAMILY_COLOURS[f] for f in present}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.violinplot(data=sub, x="family", y=metric, order=present,
                   hue="family", palette=palette, legend=False,
                   inner="box", ax=ax)
    ax.set_title(f"{metric} by language family")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--output-dir", default=None,
                   help="Where to write figures (default: <results_dir>/figures)")
    p.add_argument("--metric", default="iou_cluster_mask")
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = load_records(args.results_dir)

    # --- tables -----------------------------------------------------------
    tbl = build_summary_table(df)
    csv_path = os.path.join(out_dir, "per_language_table.csv")
    tbl.to_csv(csv_path, index=False)
    print(f"  ✍  {csv_path}")

    # --- stats ------------------------------------------------------------
    stats = {}
    for m in ("iou_cluster_mask", "iou_pct_95", "spearman", "peak_lang"):
        if m in df.columns:
            stats[m] = run_stat_tests(df, m)
    stats_path = os.path.join(out_dir, "stat_tests.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✍  {stats_path}")

    # --- figures ----------------------------------------------------------
    plot_language_bar(df,
                      os.path.join(out_dir, f"fig_language_bar_{args.metric}.png"),
                      args.metric)
    plot_language_concept_heatmap(df,
                                  os.path.join(out_dir, f"fig_language_concept_heatmap_{args.metric}.png"),
                                  args.metric)
    plot_iou_vs_peak(df,
                     os.path.join(out_dir, "fig_iou_vs_peak.png"))
    plot_family_violin(df,
                       os.path.join(out_dir, f"fig_family_violin_{args.metric}.png"),
                       args.metric)
    print(f"\n✅ All figures → {out_dir}")


if __name__ == "__main__":
    main()
