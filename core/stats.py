"""Small, dependency-light significance toolkit.

Every headline in Lynceus was a bare point estimate — "62%", "5×", "σ_NMAD
0.040". This module turns those into defensible numbers with error bars and
null tests, using only resampling (no distributional assumptions):

* :func:`bootstrap_ci` — a percentile bootstrap confidence interval for any
  statistic of a 1-D sample (resample with replacement, recompute, take
  percentiles of the resampled statistics).
* :func:`permutation_p` — a one-sided p-value for an observed statistic against
  a null built by repeatedly drawing a same-sized sample from a reference
  population (used for the Little Red Dot enrichment: are 216 LRDs really more
  anomalous than 216 random galaxies?).

Both take a numpy ``Generator`` (or a seed) so results are reproducible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

ArrayLike = np.ndarray


def _rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def bootstrap_ci(data: ArrayLike, statistic: Callable[[np.ndarray], float],
                 *, n_boot: int = 10000, ci: float = 95.0,
                 rng=0) -> dict:
    """Percentile-bootstrap CI for ``statistic`` over a 1-D ``data`` sample.

    Resamples ``data`` with replacement ``n_boot`` times, recomputes
    ``statistic`` on each resample, and reports the point estimate (on the full
    sample) plus the central ``ci``% percentile interval of the bootstrap
    distribution. Returns ``{point, lo, hi, se, ci, n_boot}``.
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        raise ValueError("bootstrap_ci: empty data")
    g = _rng(rng)
    idx = g.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(data[row]) for row in idx])
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {
        "point": float(statistic(data)),
        "lo": float(lo), "hi": float(hi),
        "se": float(boot.std(ddof=1)),
        "ci": ci, "n_boot": n_boot,
    }


def permutation_p(observed: float, reference: ArrayLike,
                  statistic: Callable[[np.ndarray], float], *,
                  n_draw: int, n_perm: int = 10000,
                  alternative: str = "greater", rng=0) -> dict:
    """One-sided permutation p-value for ``observed`` vs a resampled null.

    Builds the null by drawing ``n_draw`` items (without replacement) from
    ``reference`` ``n_perm`` times and evaluating ``statistic`` on each draw —
    i.e. "what would this statistic look like for a random same-sized subset?".
    The p-value is the fraction of null draws at least as extreme as
    ``observed`` (with the standard +1 / +1 correction so it is never exactly 0).

    ``alternative`` is ``"greater"`` (default) or ``"less"``. Returns
    ``{observed, p, null_mean, null_lo, null_hi, n_perm}`` where the null band is
    the central 95%.
    """
    reference = np.asarray(reference)
    if n_draw > len(reference):
        raise ValueError("permutation_p: n_draw exceeds reference size")
    g = _rng(rng)
    null = np.array([statistic(reference[g.choice(len(reference), n_draw, replace=False)])
                     for _ in range(n_perm)])
    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    null_lo, null_hi = np.percentile(null, [2.5, 97.5])
    return {
        "observed": float(observed),
        "p": (hits + 1) / (n_perm + 1),
        "null_mean": float(null.mean()),
        "null_lo": float(null_lo), "null_hi": float(null_hi),
        "n_perm": n_perm,
    }
