# Language-Conditioned Dense-CLIP Grounding on BDD100K

 An extension of [`vlm_energy_signatures_multilingual`](https://github.com/drdezarza/vlm_energy_signatures_multilingual) that imports the dense-feature-mapping ideas from [OneMap](https://github.com/KTH-RPL/OneMap) (Busch et al., ICRA 2025) and uses them to test a focused follow-up hypothesis:

> **Does the "double penalty" for Arabic, Basque, and Luxembourgish observed in end-to-end VLMs also appear in contrastive dense features?**
> By swapping the generative VLM for a multilingual CLIP (XLM-RoBERTa text encoder + shared visual backbone), the visual encoder is held constant across all 13 languages. Any cross-language divergence in the resulting dense similarity maps must therefore come from the text encoder — i.e. from the language head alone.

The repository ships both the **source code** for running the probe pipeline end-to-end and the **pre-computed result data** for both evaluated backbones, so paper figures and tables can be regenerated without GPU access.

## What this package does

For every image in the frozen BDD100K subset and every (concept, language) pair:

1. **Extract dense visual features once** per image using a MaskCLIP-style patch-token projection of an OpenCLIP `xlm-roberta-*-ViT-*` model. The visual encoder work is shared across all 13 languages.
2. **Encode text** in each of the 13 target languages (with caching).
3. **Compute a dense similarity map** `[H, W]` per (concept, language).
4. **Cluster high-similarity regions** using the exact region-growing algorithm from OneMap's `mapping/nav_goals/clustering.py` (ported, rerun-free).
5. **Measure cross-language agreement** via several complementary metrics (mask IoU, threshold IoU, Spearman / Pearson, peak-similarity ratio, cluster-centre distance).
6. **Aggregate and run the double-penalty statistical tests** (Friedman across languages, paired Wilcoxon high-resource vs low-resource).
7. **Track GPU energy** with the same continuous 10 Hz NVML sampler used in the parent project, so Wh / 1K queries numbers are directly comparable with the MDPI paper.

## Languages (same 13 as the parent project)

`ar`, `ca`, `de`, `en`, `es`, `eu` ⚠️, `fr`, `it`, `lb` ⚠️, `pt`, `ru`, `zh-CN`, `zh-TW`
(⚠️ = low-resource — these are the three "double penalty" languages from the parent paper.)

## Concepts (BDD100K-relevant)

`car`, `truck`, `bus`, `person`, `pedestrian`, `traffic_light`, `traffic_sign`, `bicycle`, `motorcycle`, `road`, `building` (11 total, editable).

## Repository layout

```
clip_multilingual_analysis/
├── README.md
├── requirements.txt
│
├── src/                                      # Probe pipeline source
│   ├── __init__.py
│   ├── multilingual_queries.py               # 13-language noun translations + article rendering
│   ├── clip_dense_multilingual.py            # Dense CLIP feature extraction (MaskCLIP)
│   ├── clustering.py                         # Port of OneMap's region-growing clustering
│   ├── iou_metrics.py                        # IoU, Spearman, peak-similarity, double-penalty
│   ├── energy_monitor.py                     # NVML sampler (same as v3 parent script)
│   ├── run.py                                # End-to-end runner
│   └── analyze.py                            # Offline tables + figures
│
├── scripts/
│   ├── smoke_test.py                         # End-to-end check, no GPU / no BDD / no CLIP
│   ├── integration_test.py                   # Same + runs analyze.py on synthetic data
│   └── visualize_heatmaps.py                 # Qualitative figure: per-language heatmaps
│
├── data/                                     # Pre-computed pipeline outputs (one tree per backbone)
│   ├── dense_clip_xlmr_base/                 # XLM-R base + ViT-B/32 (~87 M visual params)
│   │   ├── config.json                       # Reproducibility manifest
│   │   ├── per_sample_records.jsonl          # 30,030 rows: one per (image, concept, language)
│   │   ├── per_language_summary.json         # Mean of every metric per language
│   │   ├── per_concept_summary.json          # Same, nested by concept
│   │   ├── language_concept_iou.json         # Dense grid — primary metric only
│   │   ├── double_penalty.json               # HR vs LR contrast
│   │   ├── energy.json                       # NVML stats (total Wh, Wh/1K, avg watts)
│   │   └── figures/                          # Per-backbone diagnostic figures + tables
│   │       ├── per_language_table.csv
│   │       ├── stat_tests.json
│   │       ├── fig_language_bar_iou_cluster_mask.png
│   │       ├── fig_family_violin_iou_cluster_mask.png
│   │       ├── fig_iou_vs_peak.png
│   │       └── fig_language_concept_heatmap_iou_cluster_mask.png
│   │
│   └── dense_clip_xlmr_large/                # XLM-R large + ViT-H/14 (~632 M visual params)
│       └── (same structure as base/)
│
└── notebooks/
    ├── clip_multilingual_grounding_analysis.ipynb     # Cross-backbone analysis, paper figures
    └── clip_multilingual_grounding_analysis/          # Notebook outputs (committed for transparency)
        ├── CLIP_MULTILINGUAL_REPORT.txt               # Plain-text summary of all findings
        ├── TABLE1_paper_summary.csv                   # Headline summary (Table 2 in the paper)
        ├── table_HR_vs_LR_per_backbone.csv            # HR/LR aggregates per backbone
        ├── table_per_language_per_backbone.csv        # Long format, one row per (lang, backbone, metric)
        ├── table_concept_gap.csv                      # Per-concept HR-LR gap, sortable
        ├── table_delta_iou_large_minus_base.csv       # Δ IoU under scaling, per language
        ├── table_stat_tests.csv                       # Friedman / Wilcoxon / Mann-Whitney, all metrics × backbones
        ├── df_lang.csv                                # Tidy per-language frame
        ├── df_conc.csv                                # Tidy per-(language, concept) frame
        ├── df_energy.csv                              # Tidy per-backbone energy frame
        ├── fig1_per_language_iou_both_backbones.png   # Paper Figure 1
        ├── fig2_hr_vs_lr_forest.png                   # (supplementary; not in paper)
        ├── fig3_delta_iou_scale.png                   # Paper Figure 3
        ├── fig4_signal_vs_spatial.png                 # Paper Figure 4
        ├── fig5_concept_heatmaps_both_backbones.png   # (supplementary; superseded by §4.4 prose)
        ├── fig6_lowresource_concept_profile.png       # Paper Figure 6
        ├── fig7_energy_comparison_vs_vlms*.png        # Paper Figure 7 (three layout variants)
        └── fig8_concept_gap_both_backbones.png        # (supplementary)
```

## Installation

```bash
git clone https://github.com/drdezarza/clip_multilingual_analysis.git
cd clip_multilingual_analysis
pip install -r requirements.txt
```

For PyTorch with CUDA, prefer the official wheel matching your CUDA (example for CUDA 12.1):

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch is only required if you want to **rerun the pipeline** on BDD100K. To regenerate paper figures from the included `data/` trees, only the lighter dependencies (`pandas`, `numpy`, `scipy`, `matplotlib`, `jupyter`) are needed.

## Quick start

There are three entry points depending on what you want to do.

### A. Reproduce paper figures and tables from the included data

The fastest path: the `data/dense_clip_xlmr_*/` trees are committed, so the cross-backbone analysis notebook regenerates every paper figure without a GPU.

```bash
jupyter lab notebooks/clip_multilingual_grounding_analysis.ipynb
```

The notebook reads from `data/dense_clip_xlmr_base/` and `data/dense_clip_xlmr_large/` and writes outputs to `notebooks/clip_multilingual_grounding_analysis/` (a snapshot of the committed outputs is already there for inspection without running the notebook).

### B. Verify the pipeline with no downloads

```bash
python scripts/smoke_test.py
# → "✅  HR IoU (…) > LR IoU (…) — double-penalty signature recovered from synthetic data."
```

```bash
python scripts/integration_test.py
# → writes synthetic JSONL, runs analyze.py, produces all per-backbone figures + a Wilcoxon p-value.
```

### C. Rerun the full pipeline on BDD100K

You need GPU access, the BDD100K images, and the frozen-subset JSON `BDD100K_<N>_samples_v1.json` produced by the parent project's `freeze_dataset.py`.

```bash
# Smaller backbone (default)
python -m src.run \
    BDD100K_210_samples_v1.json \
    --output-dir data/dense_clip_xlmr_base/ \
    --model xlm-roberta-base-ViT-B-32 \
    --pretrained laion5b_s13b_b90k

# Larger backbone
python -m src.run \
    BDD100K_210_samples_v1.json \
    --output-dir data/dense_clip_xlmr_large/ \
    --model xlm-roberta-large-ViT-H-14 \
    --pretrained frozen_laion5b_s13b_b90k
```

Per-backbone tables and per-backbone diagnostic figures:

```bash
python src/analyze.py data/dense_clip_xlmr_base/
python src/analyze.py data/dense_clip_xlmr_large/
```

Cross-backbone tables and the paper figures (Figures 1, 3, 4, 6, 7) come from the notebook above.

#### `src.run` arguments (abridged; see `python -m src.run --help`)

| Flag | Default | Meaning |
|---|---|---|
| `frozen_json` (positional) | — | Path to `BDD100K_<N>_samples_v1.json` from the parent project |
| `--n-images` | *all* | Cap on number of images |
| `--concepts` | all 11 | Comma-separated concept keys |
| `--languages` | all 13 | Comma-separated language codes |
| `--template` | `indef` | `indef` ("a car") or `bare` ("car") |
| `--model` / `--pretrained` | `xlm-roberta-base-ViT-B-32` / `laion5b_s13b_b90k` | OpenCLIP model |
| `--cluster-rel-thresh` | `0.8` | OneMap default |
| `--cluster-percentile-mask` | `75` | Mask eligibility cutoff (percentile per map) |
| `--upsample-size` | `224` | Similarity-map upsampling (square, for clustering) |
| `--energy-interval` | `0.1` | NVML sampling period (s) — 10 Hz default |

## Output format

Each `data/dense_clip_xlmr_*/` directory follows the schema:

| File | Content |
|---|---|
| `per_sample_records.jsonl` | One record per (image, concept, language) — all metrics |
| `per_language_summary.json` | Mean of every metric per language |
| `per_concept_summary.json` | Same, nested by concept |
| `language_concept_iou.json` | Dense grid — primary metric only |
| `double_penalty.json` | HR vs LR contrast for the main metrics |
| `energy.json` | NVML stats — `total_wh`, `wh_per_1k_queries`, `avg_watts`, ... |
| `config.json` | Reproducibility manifest |
| `figures/per_language_table.csv` | Per-language metric table |
| `figures/stat_tests.json` | Per-backbone Friedman / Wilcoxon / Mann-Whitney results |
| `figures/*.png` | Per-backbone diagnostic plots |

The JSONL format matches the style of the parent project's `results_multilingual/` tree so downstream tooling (the notebook in particular) can be shared.

## Models

| Short name | OpenCLIP ID | Pretrained | Params (vis.) |
|---|---|---|---|
| XLM-R base + ViT-B/32 *(default)* | `xlm-roberta-base-ViT-B-32` | `laion5b_s13b_b90k` | 87 M |
| XLM-R large + ViT-H/14 | `xlm-roberta-large-ViT-H-14` | `frozen_laion5b_s13b_b90k` | 632 M |

Both use the same XLM-RoBERTa tokenizer, which natively covers Arabic, Basque, Luxembourgish, both Chinese scripts and the other target languages.

## Why this specific probe

A standard end-to-end VLM (LLaVA, InternVL, Qwen-VL, Phi-3-V) produces answers by running image tokens through an autoregressive multilingual language model. If that model shows worse performance and higher inference energy in Luxembourgish, the cause is ambiguous — it could be the visual side (different attention patterns triggered by non-English instructions), the language head (tokenisation overhead, calibration drift, RLHF asymmetry), or the interaction between them.

Multilingual CLIP separates these. The visual backbone is **identical** for every language: we feed it one image, get one feature tensor, and only the *text branch* changes. If IoU drops for Arabic / Basque / Luxembourgish here, the penalty lives in the text encoder alone. If it doesn't, the penalty in VLMs must come from the decoder's autoregressive generation process.

The empirical answer (see paper §4.1) is that the penalty *does* persist (HR–LR cluster-mask IoU gap +0.114 at the smaller backbone, +0.143 at the larger one, Wilcoxon HR > LR `p < 1e-300` at both scales), localising it to the text branch.

## Metrics in detail

Per (image, concept, language) row — see `iou_metrics.summarise_pair`:

- **`iou_cluster_mask`** *(primary)* — Binary-mask IoU of clustered regions, language vs English.
- **`iou_pct_{90, 95, 99}`** — Binary IoU of the top-(100-p)% pixels, no clustering. Scale-invariant per map.
- **`iou_thresh_{0.15, 0.20, 0.25}`** — Binary IoU at fixed absolute similarity thresholds.
- **`spearman`, `pearson`** — Rank / linear correlation of the raw similarity maps. Continuous, no thresholding.
- **`peak_lang`, `peak_ref`** — Maximum cosine similarity reached in the respective map.
- **`peak_ratio_lang_over_ref`** — `peak_lang / peak_ref`. Values < 1 indicate text-encoder signal collapse in the target language.
- **`center_dist_symmetric`** — Symmetric average distance (pixels) between cluster centres across the two languages.
- **`n_clusters_lang`, `n_clusters_ref`** — Number of clusters detected.

At the aggregate level:

- **Friedman** across non-reference languages, paired by (image, concept).
- **Wilcoxon** HR > LR, paired by (image, concept).
- **Mann–Whitney** per-low-resource-language vs pooled high-resource, one-sided.

## Relation to the parent project

This package intentionally reuses:

- The same **frozen-subset format** (`BDD100K_<N>_samples_v1.json` from `freeze_dataset.py`).
- The same **13 languages**, same language-family grouping, same low-resource tag set `{ar, eu, lb}`.
- The same **NVML energy methodology** (10 Hz, Wh / 1K queries).
- A compatible **`per_sample_records.jsonl`** schema and notebook style.

This means you can drop the output JSONs next to the parent's `results_multilingual/` tree and the existing analysis code can still read them.


## Acknowledgements

Barcelona Supercomputing Center (TIFON — MIG-20232039) and  Luxembourg Institute of Science and Technology (ADIALab-MAST, LLMs4EU — Grant Agreement No 101198470).
