"""Turn the self-supervised encoder into a 2-D atlas: embed, then UMAP."""
from __future__ import annotations

import numpy as np
import torch

from . import config
from .ssl import embed_tensor


@torch.no_grad()
def embed_paths(encoder, npy_paths, device=None, batch_size: int = 128) -> np.ndarray:
    """Run the encoder over cutouts -> (N, feat_dim) float array."""
    device = device or config.get_device()
    encoder = encoder.to(device).eval()
    feats, batch = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        feats.append(encoder(x).cpu().numpy())
        batch.clear()

    for p in npy_paths:
        batch.append(embed_tensor(np.load(p).astype(np.float32)))
        if len(batch) >= batch_size:
            flush()
    flush()
    return np.concatenate(feats, 0)


def umap_2d(features: np.ndarray, *, n_neighbors: int = 30, min_dist: float = 0.5,
            spread: float = 1.4, seed: int = 42) -> np.ndarray:
    """L2-normalise features and project to 2-D with cosine-metric UMAP.

    Larger ``n_neighbors`` / ``min_dist`` / ``spread`` give a rounder, more
    spread-out layout (better for an explorable atlas) than the tight default.
    """
    import umap
    x = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread,
                        metric="cosine", random_state=seed)
    return reducer.fit_transform(x)


def normalise_coords(xy: np.ndarray) -> np.ndarray:
    """Scale a 2-D layout into [0, 1], robust to outliers (preserves aspect).

    A handful of UMAP stragglers would otherwise dominate the min/max range and
    squash the bulk; centre on the 1-99 percentile box and clip stragglers to
    the margin instead.
    """
    lo = np.percentile(xy, 2, axis=0)
    hi = np.percentile(xy, 98, axis=0)
    out = (xy - lo) / (hi - lo + 1e-9)      # per-axis fill (abstract map)
    return np.clip(out * 0.9 + 0.05, 0.02, 0.98)
