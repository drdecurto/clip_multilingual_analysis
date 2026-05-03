"""
Language-conditioned dense-CLIP grounding on BDD100K.

Extends `vlm_energy_signatures_multilingual` along the language axis, using
OneMap's (KTH-RPL, ICRA 2025) dense-feature mapping ideas.

Imports are lazy-ish: the heavy `MultilingualDenseCLIP` (which requires
torch / open_clip) is only imported on demand. All other utilities
(clustering, IoU, queries, energy monitor) are light and imported eagerly.
"""

# Always-light imports
from .clustering import (Cluster, cluster_high_similarity_regions,
                         clusters_to_mask, gradient_based_clustering,
                         watershed_clustering)
from .energy_monitor import GPUEnergySampler
from .iou_metrics import (aggregate_by_language, double_penalty_contrast,
                          mask_iou, summarise_pair)
from .multilingual_queries import (ALL_CONCEPTS, ALL_LANGUAGES, CONCEPTS_BARE,
                                   LANGUAGE_INFO, LOW_RESOURCE_LANGS,
                                   all_queries, render_query)


# Heavy (torch + open_clip) imports — lazy
def __getattr__(name):
    if name == "MultilingualDenseCLIP":
        from .clip_dense_multilingual import MultilingualDenseCLIP
        return MultilingualDenseCLIP
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    "MultilingualDenseCLIP",
    "Cluster", "cluster_high_similarity_regions", "clusters_to_mask",
    "gradient_based_clustering", "watershed_clustering",
    "GPUEnergySampler",
    "aggregate_by_language", "double_penalty_contrast", "mask_iou",
    "summarise_pair",
    "ALL_CONCEPTS", "ALL_LANGUAGES", "CONCEPTS_BARE", "LANGUAGE_INFO",
    "LOW_RESOURCE_LANGS", "all_queries", "render_query",
]
