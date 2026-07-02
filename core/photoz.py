"""Photometric redshift estimation for M4 — the dropout hunter.

The physics: a galaxy's light is absorbed by intervening neutral hydrogen
blueward of Lyman-alpha (1216 Angstrom rest-frame). As redshift grows that
"Lyman break" sweeps through the filters, so a high-z galaxy *drops out* of the
bluer bands and only appears in redder ones. The pattern of which bands a galaxy
is bright/faint/absent in therefore encodes its redshift — that's what makes
photometric redshifts possible, and what JWST used to push the frontier to z~14
and turn up *too many, too bright* early galaxies (a real LambdaCDM tension).

This module learns that flux -> redshift mapping directly from data, with no
template fitting at inference:

* :func:`featurize` turns a galaxy's per-band fluxes into a colour-shape vector
  (normalised to a robust red-band scale, asinh-compressed) plus a per-band
  detection mask and an overall brightness feature.
* :class:`PhotoZNet` is a small MLP that outputs a **probability distribution
  over redshift bins** (softmax), not a single number — so it naturally produces
  a PDF and can express the classic low-z/high-z degeneracy (a z~12 dropout
  candidate that might instead be a dusty z~2 interloper).
* Trained on tens of thousands of template-based redshifts (EAZY ``z_phot``) for
  coverage across 0 < z < ~16, then validated honestly against real
  spectroscopic redshifts (``z_spec``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

# Bands used as features, blue -> red. Chosen for good coverage in the CEERS
# grizli catalogue; together they bracket the Lyman break across 0 < z < ~15.
# (HST ACS optical, HST WFC3/NIRCam near-IR, NIRCam wide + one medium.)
PHOTOZ_BANDS = (
    "f606w", "f814w", "f115w", "f150w",
    "f200w", "f277w", "f356w", "f410m", "f444w",
)
# Always-present red bands used as the per-galaxy normalisation scale.
RED_REF = ("f277w", "f356w", "f444w")

# Redshift grid: bin centres the network classifies over.
Z_MAX = 16.0
Z_BINS = 96
Z_EDGES = np.linspace(0.0, Z_MAX, Z_BINS + 1)
Z_CENTRES = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
Z_STEP = Z_EDGES[1] - Z_EDGES[0]

BAD = -90.0          # catalogue sentinel: values <= BAD are "no data"
SN_MIN = 2.0         # below this S/N a band counts as a non-detection


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------
def featurize(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Turn per-band (flux, err) into a redshift-bearing feature vector.

    ``flux`` and ``err`` are (N, B) in catalogue units (B = len(PHOTOZ_BANDS)),
    with non-observations flagged by values <= ``BAD``. The returned feature
    vector per galaxy is, concatenated:

      * B asinh-compressed band fluxes, normalised by a robust red-band scale so
        the network sees the *colour shape* of the SED, not the brightness;
      * B detection-mask flags (1 if the band is a real S/N detection);
      * B log(1 + S/N) confidence features, so the net can down-weight noisy
        bands rather than trusting every flux equally;
      * 1 brightness feature: log of the normalisation scale.

    Non-detections are set to 0 in the flux block and 0 in the mask, so the
    network learns "this band is dark" — exactly the dropout signal.
    """
    flux = np.asarray(flux, np.float64).copy()
    err = np.asarray(err, np.float64).copy()
    bands = list(PHOTOZ_BANDS)

    valid = (flux > BAD) & (err > 0) & np.isfinite(flux) & np.isfinite(err)
    sn = np.zeros_like(flux)
    sn[valid] = flux[valid] / err[valid]
    detected = valid & (sn >= SN_MIN)

    # robust per-galaxy scale = sum of the always-present red reference bands
    ref_idx = [bands.index(b) for b in RED_REF]
    ref = flux[:, ref_idx].copy()
    ref[~valid[:, ref_idx]] = 0.0
    scale = np.clip(ref.sum(axis=1), 1e-4, None)              # (N,)

    norm = flux / scale[:, None]
    norm[~detected] = 0.0                                     # dark/undetected -> 0
    # asinh keeps faint structure and tames bright outliers (soft ~ a few % of unity)
    feat_flux = np.arcsinh(norm / 0.05)

    mask = detected.astype(np.float64)
    conf = np.log1p(np.where(detected, sn, 0.0))              # per-band confidence
    bright = np.log10(scale)[:, None]                         # overall brightness

    return np.concatenate([feat_flux, mask, conf, bright], axis=1).astype(np.float32)


FEATURE_DIM = 3 * len(PHOTOZ_BANDS) + 1


def soft_labels(z: np.ndarray, frac: float = 0.03, floor: float = 0.25) -> np.ndarray:
    """Gaussian soft target over the z grid, width ~ frac*(1+z) (>= floor).

    Soft labels (vs one-hot) give the trained PDF a sensible width and stop the
    network over-committing to a single bin.
    """
    z = np.asarray(z, np.float64)
    sig = np.clip(frac * (1.0 + z), floor, None)
    d = Z_CENTRES[None, :] - z[:, None]
    w = np.exp(-0.5 * (d / sig[:, None]) ** 2)
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PhotoZNet(nn.Module):
    """Small MLP: features -> logits over Z_BINS redshift bins."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 256,
                 n_bins: int = Z_BINS, p_drop: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_bins),
        )

    def forward(self, x):
        return self.net(x)


def pdf_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def point_estimates(pdf: np.ndarray, peak_win: float = 1.5) -> dict:
    """Summarise a batch of PDFs (N, Z_BINS) into z point estimates + spread.

    ``z_peak`` is the headline estimate: the probability-weighted mean taken only
    over bins within ``peak_win`` of the most-likely bin. This keeps the
    precision of a mean while ignoring a secondary (e.g. low-z interloper) mode
    that would otherwise drag a global mean into the gap between the two.
    """
    pdf = np.asarray(pdf, np.float64)
    pdf = pdf / (pdf.sum(1, keepdims=True) + 1e-12)
    z_mean = (pdf * Z_CENTRES[None]).sum(1)
    mode_idx = pdf.argmax(1)
    z_mode = Z_CENTRES[mode_idx]
    near = np.abs(Z_CENTRES[None] - z_mode[:, None]) <= peak_win
    w = pdf * near
    z_peak = (w * Z_CENTRES[None]).sum(1) / (w.sum(1) + 1e-12)
    var = (pdf * (Z_CENTRES[None] - z_mean[:, None]) ** 2).sum(1)
    return {"z_peak": z_peak, "z_mean": z_mean, "z_mode": z_mode, "z_std": np.sqrt(var)}


# ---------------------------------------------------------------------------
# PDF calibration (are the *uncertainties* trustworthy, not just the peak?)
# ---------------------------------------------------------------------------
def temperature_scale(pdf: np.ndarray, T: float) -> np.ndarray:
    """Apply softmax temperature ``T`` directly to a normalised PDF.

    Because ``softmax(logits / T) = normalise(softmax(logits) ** (1/T))``, we can
    re-temperature a stored PDF without the original logits. ``T>1`` widens the
    PDF (fixes over-confidence), ``T<1`` sharpens it.
    """
    pdf = np.asarray(pdf, np.float64)
    if T == 1.0:
        return pdf / (pdf.sum(-1, keepdims=True) + 1e-12)
    scaled = np.power(np.clip(pdf, 1e-12, None), 1.0 / T)
    return scaled / (scaled.sum(-1, keepdims=True) + 1e-12)


def pit_values(pdf: np.ndarray, z_true: np.ndarray,
               edges: np.ndarray = Z_EDGES) -> np.ndarray:
    """Probability Integral Transform: the predicted CDF evaluated at the truth.

    For a calibrated model the PITs are **uniform on [0,1]** — a U-shaped PIT
    histogram means over-confident (too-narrow) PDFs, a central hump means
    under-confident ones, a tilt means bias. Each bin's mass is treated as
    uniform within the bin so the CDF (and the PIT) is continuous.
    """
    pdf = np.atleast_2d(np.asarray(pdf, np.float64))
    pdf = pdf / (pdf.sum(1, keepdims=True) + 1e-12)
    z_true = np.asarray(z_true, np.float64)
    cum_left = np.concatenate([np.zeros((len(pdf), 1)), np.cumsum(pdf, axis=1)], axis=1)[:, :-1]
    width = np.diff(edges)
    k = np.clip(np.searchsorted(edges, z_true, side="right") - 1, 0, len(width) - 1)
    frac = np.clip((z_true - edges[k]) / width[k], 0.0, 1.0)
    rows = np.arange(len(pdf))
    return cum_left[rows, k] + pdf[rows, k] * frac


def credible_coverage(pit: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Empirical coverage of central credible intervals at each nominal ``level``.

    A central ``c``-credible interval contains the truth iff its PIT lies within
    ``c/2`` of 0.5, so coverage(c) = mean(|PIT - 0.5| <= c/2). Calibrated ⇒
    coverage(c) ≈ c for all c.
    """
    pit = np.asarray(pit)[:, None]
    return (np.abs(pit - 0.5) <= np.asarray(levels)[None] / 2).mean(0)


def pit_ks(pit: np.ndarray) -> float:
    """Kolmogorov–Smirnov distance of the PITs from Uniform[0,1] (0 = perfect).

    Proper two-sided statistic: the empirical CDF is a staircase, so the max
    deviation must be checked at both the top (i/n) and bottom ((i-1)/n) of
    each step.
    """
    p = np.sort(np.asarray(pit))
    n = len(p)
    i = np.arange(1, n + 1)
    return float(max((i / n - p).max(), (p - (i - 1) / n).max()))


def nll_at_truth(pdf: np.ndarray, z_true: np.ndarray,
                 edges: np.ndarray = Z_EDGES) -> float:
    """Mean negative log *density* the PDFs assign at the true redshifts."""
    pdf = np.atleast_2d(np.asarray(pdf, np.float64))
    pdf = pdf / (pdf.sum(1, keepdims=True) + 1e-12)
    width = np.diff(edges)
    k = np.clip(np.searchsorted(edges, np.asarray(z_true), side="right") - 1, 0, len(width) - 1)
    dens = pdf[np.arange(len(pdf)), k] / width[k]
    return float(-np.mean(np.log(dens + 1e-12)))


def fit_temperature(pdf: np.ndarray, z_true: np.ndarray,
                    edges: np.ndarray = Z_EDGES, objective: str = "nll",
                    grid: np.ndarray | None = None) -> float:
    """Grid-search the single softmax temperature minimising ``objective``.

    ``"nll"`` (default) is classical temperature scaling (Guo et al.) — the
    principled choice; ``"ks"`` minimises the PIT Kolmogorov distance to uniform.
    On a model with a good core but heavy catastrophic-outlier tails the two
    disagree (NLL widens to cover outliers, KS sharpens to flatten the core),
    which is itself a useful diagnostic.
    """
    if grid is None:
        grid = np.geomspace(0.2, 5.0, 96)
    if objective == "nll":
        score = [nll_at_truth(temperature_scale(pdf, T), z_true, edges) for T in grid]
    elif objective == "ks":
        score = [pit_ks(pit_values(temperature_scale(pdf, T), z_true, edges)) for T in grid]
    else:
        raise ValueError("objective must be 'nll' or 'ks'")
    return float(grid[int(np.argmin(score))])


# ---------------------------------------------------------------------------
# Metrics (computed against spectroscopic truth)
# ---------------------------------------------------------------------------
@dataclass
class PhotoZMetrics:
    n: int
    sigma_nmad: float
    outlier_frac: float
    bias: float

    def __str__(self):
        return (f"n={self.n}  sigma_NMAD={self.sigma_nmad:.4f}  "
                f"outliers(>0.15)={self.outlier_frac*100:.1f}%  bias={self.bias:+.4f}")


def photoz_metrics(z_pred: np.ndarray, z_true: np.ndarray) -> PhotoZMetrics:
    """Standard photo-z quality stats. dz = (z_pred - z_true) / (1 + z_true)."""
    z_pred = np.asarray(z_pred, float); z_true = np.asarray(z_true, float)
    m = np.isfinite(z_pred) & np.isfinite(z_true) & (z_true >= 0)
    dz = (z_pred[m] - z_true[m]) / (1.0 + z_true[m])
    sigma_nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outlier = np.mean(np.abs(dz) > 0.15)
    return PhotoZMetrics(int(m.sum()), float(sigma_nmad), float(outlier), float(np.median(dz)))


def get_device():
    return config.get_device()
