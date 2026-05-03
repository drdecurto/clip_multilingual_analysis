#!/usr/bin/env python3
"""
Dense multilingual CLIP — MaskCLIP-style patch-token feature extraction.

Produces per-patch features in the shared image-text embedding space, which
can then be compared against text features encoded in any of the 13 languages
supported by XLM-R-based OpenCLIP checkpoints.

Default model: ``xlm-roberta-base-ViT-B-32`` (laion5b_s13b_b90k).
Alternative:    ``xlm-roberta-large-ViT-H-14`` (frozen_laion5b_s13b_b90k).

Both use the same XLM-RoBERTa tokenizer, which natively covers Arabic, Basque,
Luxembourgish, Chinese (both scripts) and the other target languages.

Why this is the right probe for the research question:

  * The *visual encoder* is shared across all languages. Only the text input
    changes. Any difference in the similarity maps must therefore be
    attributable to the language-conditioning side.
  * This isolates the hypothesis "does the double penalty live in the vision
    encoder or the language head?" from confounds present in end-to-end VLMs
    (which intermix visual tokens into an autoregressive multilingual LM).

Implementation notes:

  * Dense features are extracted by running the visual transformer end-to-end,
    then projecting the *patch* tokens (not just CLS) through the visual
    projection head `visual.proj`. This is the standard MaskCLIP recipe and
    works out-of-the-box for every OpenCLIP ViT we've tested.
  * We cache text features per (concept, language) — text encoding is cheap
    but not free at ~500 forward passes per run.
  * Everything is torch-only; no detectron2 / fvcore / SED dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip_torch is required. Install with: pip install open_clip_torch"
    ) from e


DEFAULT_MODEL = "xlm-roberta-base-ViT-B-32"
DEFAULT_PRETRAINED = "laion5b_s13b_b90k"


# =============================================================================
# Dense feature extractor
# =============================================================================

@dataclass
class DenseCLIPOutput:
    """Container for a single forward pass."""
    patch_feats: torch.Tensor          # [B, D, H_p, W_p]  normalised
    input_hw:     Tuple[int, int]      # (H_in, W_in) before patchification
    patch_hw:     Tuple[int, int]      # (H_p, W_p)


class MultilingualDenseCLIP:
    """
    Wraps an OpenCLIP XLM-RoBERTa-based model and exposes a MaskCLIP-style
    dense feature extractor plus a text encoder.
    """

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 pretrained: str = DEFAULT_PRETRAINED,
                 device: str = "cuda",
                 precision: str = "fp32",
                 ) -> None:

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device

        print(f"🔧 Loading OpenCLIP {model_name}  (pretrained={pretrained})  on {device}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            precision=precision,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.visual = model.visual
        self.embed_dim = self._discover_embed_dim()
        self.patch_size = self._discover_patch_size()
        self.input_resolution = self._discover_input_resolution()
        self.feature_dim = self.embed_dim  # alias for OneMap-style code

        # ---- enable dense-token output if the version supports it ----------
        # open_clip >= 2.20 exposes `visual.output_tokens`. If not, we fall
        # back to an explicit forward.
        self._native_output_tokens = self._try_enable_output_tokens()

        print(f"   • embed_dim={self.embed_dim}  patch_size={self.patch_size}"
              f"  input_res={self.input_resolution}  "
              f"native_dense_output={self._native_output_tokens}")

        # text-feature cache ------------------------------------------------
        self._text_cache: Dict[str, torch.Tensor] = {}

    # ---------- discovery helpers ------------------------------------------

    def _discover_embed_dim(self) -> int:
        # OpenCLIP exposes .proj (possibly None) or .output_dim
        for attr in ("output_dim", "embed_dim"):
            if hasattr(self.visual, attr):
                v = getattr(self.visual, attr)
                if isinstance(v, int):
                    return v
        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            return int(self.visual.proj.shape[-1])
        raise RuntimeError("Could not determine visual embed dim")

    def _discover_patch_size(self) -> int:
        # Look at the conv1 kernel size
        if hasattr(self.visual, "conv1"):
            k = self.visual.conv1.kernel_size
            return int(k[0]) if isinstance(k, tuple) else int(k)
        # timm-based visual: trunk.patch_embed
        trunk = getattr(self.visual, "trunk", None)
        if trunk is not None and hasattr(trunk, "patch_embed"):
            ps = trunk.patch_embed.patch_size
            return int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
        raise RuntimeError("Could not determine patch size")

    def _discover_input_resolution(self) -> int:
        # Most OpenCLIP visual models keep .image_size
        for attr in ("image_size", "input_resolution"):
            if hasattr(self.visual, attr):
                v = getattr(self.visual, attr)
                if isinstance(v, (tuple, list)):
                    return int(v[0])
                if isinstance(v, int):
                    return v
        # Fall back to reading from preprocess
        for t in self.preprocess.transforms:
            if hasattr(t, "size"):
                s = t.size
                return int(s[0] if isinstance(s, (tuple, list)) else s)
        return 224

    def _try_enable_output_tokens(self) -> bool:
        if hasattr(self.visual, "output_tokens"):
            try:
                self.visual.output_tokens = True
                return True
            except Exception:
                return False
        return False

    # ---------- preprocessing ----------------------------------------------

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """PIL → preprocessed [1, 3, H, W] tensor on device."""
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        return x

    # ---------- dense visual forward ---------------------------------------

    @torch.no_grad()
    def encode_image_dense(self, image_tensor: torch.Tensor,
                           l2_normalize: bool = True) -> DenseCLIPOutput:
        """
        Args
        ----
        image_tensor : [B, 3, H, W], already preprocessed (via ``preprocess_image``)
        l2_normalize : L2-normalize features along the embedding axis

        Returns
        -------
        DenseCLIPOutput with patch_feats of shape [B, D, H_p, W_p]
        """
        assert image_tensor.ndim == 4, "Expected [B, 3, H, W]"
        B, _, H, W = image_tensor.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        # Branch A: native output_tokens -----------------------------------
        if self._native_output_tokens:
            try:
                pooled, tokens = self.visual(image_tensor)
                # `tokens` is [B, N, D] where N may be H_p*W_p (no CLS) or
                # H_p*W_p+1 (with CLS). Infer which.
                N = tokens.shape[1]
                expected_no_cls = H_p * W_p
                if N == expected_no_cls + 1:
                    tokens = tokens[:, 1:, :]
                elif N != expected_no_cls:
                    # Unexpected — fall back below
                    raise RuntimeError(f"Token grid mismatch: N={N}, HpWp={H_p * W_p}")

                # OpenCLIP applies `visual.proj` to the pooled output but NOT
                # to the per-token output. If the token width doesn't match
                # the shared embed dim, project manually so patch & text
                # features live in the same space.
                if (hasattr(self.visual, "proj") and self.visual.proj is not None
                        and tokens.shape[-1] != self.embed_dim):
                    tokens = tokens @ self.visual.proj

                patches = tokens  # [B, N, D_shared]
            except Exception as err:
                print(f"⚠️  Native dense-output path failed ({err}); falling back.")
                patches = self._manual_dense_forward(image_tensor, H_p, W_p)
        else:
            patches = self._manual_dense_forward(image_tensor, H_p, W_p)

        # → [B, D, H_p, W_p]
        D = patches.shape[-1]
        patches = patches.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()

        if l2_normalize:
            patches = F.normalize(patches, dim=1)

        return DenseCLIPOutput(patch_feats=patches,
                               input_hw=(H, W),
                               patch_hw=(H_p, W_p))

    def _manual_dense_forward(self, x: torch.Tensor,
                              H_p: int, W_p: int) -> torch.Tensor:
        """
        Replicates the standard ViT forward, returning patch tokens in the
        shared vision-text embedding space. Works for OpenCLIP VisionTransformer.
        """
        vis = self.visual

        x = vis.conv1(x)                              # [B, D_in, H_p, W_p]
        x = x.reshape(x.shape[0], x.shape[1], -1)     # [B, D_in, N]
        x = x.permute(0, 2, 1)                        # [B, N, D_in]

        # Prepend class token
        cls = vis.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1],
                                dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                # [B, N+1, D_in]
        x = x + vis.positional_embedding.to(x.dtype)

        # OpenCLIP may have an optional pre-norm patch layer
        if hasattr(vis, "patch_dropout"):
            x = vis.patch_dropout(x)
        x = vis.ln_pre(x)

        # Transformer: some versions take NLD, some LND. Handle both.
        try:
            # NLD → LND
            x_lnd = x.permute(1, 0, 2)
            x_lnd = vis.transformer(x_lnd)
            x = x_lnd.permute(1, 0, 2)
        except Exception:
            x = vis.transformer(x)

        x = vis.ln_post(x)                            # [B, N+1, D_in]

        # Drop CLS, keep patch tokens
        patches = x[:, 1:, :]                         # [B, N, D_in]

        # Apply visual projection to the shared space
        if hasattr(vis, "proj") and vis.proj is not None:
            patches = patches @ vis.proj              # [B, N, D_out]

        return patches

    # ---------- text encoding ----------------------------------------------

    @torch.no_grad()
    def encode_text(self, texts: List[str],
                    l2_normalize: bool = True,
                    use_cache: bool = True) -> torch.Tensor:
        """Return [T, D] text embeddings."""
        if use_cache:
            out = []
            uncached_idx, uncached_txt = [], []
            for i, t in enumerate(texts):
                if t in self._text_cache:
                    out.append(self._text_cache[t])
                else:
                    uncached_idx.append(i)
                    uncached_txt.append(t)
                    out.append(None)
            if uncached_txt:
                toks = self.tokenizer(uncached_txt).to(self.device)
                feats = self.model.encode_text(toks)
                if l2_normalize:
                    feats = F.normalize(feats, dim=-1)
                for j, t in enumerate(uncached_txt):
                    self._text_cache[t] = feats[j]
                    out[uncached_idx[j]] = feats[j]
            return torch.stack(out, dim=0)

        toks = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(toks)
        if l2_normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    # ---------- similarity --------------------------------------------------

    @staticmethod
    def similarity(patch_feats: torch.Tensor,
                   text_feats: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity map (features assumed already L2-normalized).

        Args
        ----
        patch_feats : [B, D, H_p, W_p]
        text_feats  : [T, D]

        Returns
        -------
        sim : [B, T, H_p, W_p]   values in [-1, 1]
        """
        return torch.einsum("bdhw,td->bthw", patch_feats, text_feats)

    @staticmethod
    def upsample_to(sim: torch.Tensor, out_hw: Tuple[int, int],
                    mode: str = "bilinear") -> torch.Tensor:
        """Upsample [B, T, H_p, W_p] → [B, T, H_out, W_out]."""
        return F.interpolate(sim, size=out_hw, mode=mode,
                             align_corners=False if mode == "bilinear" else None)


# =============================================================================
# Self-test helpers (CPU-friendly, no BDD required)
# =============================================================================

def _smoke_test() -> None:
    """A tiny end-to-end check using random inputs. Requires a downloadable
    checkpoint; use ``--offline`` on restricted networks."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = MultilingualDenseCLIP(device=dev)

    # Synthetic 224×224 RGB image
    img = Image.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )
    xt = clip.preprocess_image(img)
    out = clip.encode_image_dense(xt)
    print(f"patch_feats: {tuple(out.patch_feats.shape)}")

    texts = ["a car", "un coche", "une voiture", "سيارة", "autoa"]
    t = clip.encode_text(texts)
    print(f"text_feats:  {tuple(t.shape)}")

    sim = clip.similarity(out.patch_feats, t)
    print(f"sim:         {tuple(sim.shape)}   "
          f"min={sim.min().item():.3f}  max={sim.max().item():.3f}")

    upsampled = clip.upsample_to(sim, (224, 224))
    print(f"upsampled:   {tuple(upsampled.shape)}")


if __name__ == "__main__":
    _smoke_test()
