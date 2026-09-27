"""Latent-space anomaly scoring for the M3 hunt.

The self-supervised encoder learned what galaxies normally look like. A galaxy
that sits far from its neighbours in that 512-D space is, by construction,
unusual, and that's the whole signal. We score it with the mean cosine distance to
the k nearest reference galaxies (kNN density).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def knn_anomaly(emb: np.ndarray, ref: np.ndarray, k: int = 20,
                exclude_self: bool = False) -> np.ndarray:
    """Mean cosine distance from each row of ``emb`` to its k nearest in ``ref``."""
    nn = NearestNeighbors(n_neighbors=k + (1 if exclude_self else 0),
                          metric="cosine").fit(_l2(ref))
    d, _ = nn.kneighbors(_l2(emb))
    if exclude_self:
        d = d[:, 1:]
    return d.mean(1)


def nearest_distance(emb: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Min cosine distance from each row of ``emb`` to any row of ``targets``.

    Used to flag galaxies that look like the known Little Red Dots.
    """
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(_l2(targets))
    d, _ = nn.kneighbors(_l2(emb))
    return d[:, 0]


def to_unit(score: np.ndarray, lo_pct: float = 5, hi_pct: float = 99) -> np.ndarray:
    """Scale a raw score into [0,1] by robust percentiles (for colouring)."""
    lo, hi = np.percentile(score, [lo_pct, hi_pct])
    return np.clip((score - lo) / (hi - lo + 1e-9), 0, 1)
