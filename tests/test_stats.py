import numpy as np

from core import stats


def test_bootstrap_ci_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 2.0, size=2000)
    a = stats.bootstrap_ci(data, np.mean, n_boot=500, rng=0)
    b = stats.bootstrap_ci(data, np.mean, n_boot=500, rng=0)
    assert a == b                                   # same seed -> identical
    assert a["point"] == np.mean(data)              # point is the full-sample stat
    assert a["lo"] < a["point"] < a["hi"]           # CI brackets the estimate
    assert abs(a["point"] - 5.0) < 0.2              # ~recovers the true mean


def test_bootstrap_ci_degenerate_on_constant():
    out = stats.bootstrap_ci(np.full(50, 3.0), np.median, n_boot=200, rng=1)
    assert out["lo"] == out["hi"] == out["point"] == 3.0


def test_permutation_p_flags_a_real_excess():
    # reference is 10% "anomalous"; an observed fraction of 0.5 should be rare.
    ref = np.concatenate([np.ones(100), np.zeros(900)])
    frac = lambda s: float(s.mean())
    hot = stats.permutation_p(0.5, ref, frac, n_draw=200, n_perm=1000, rng=0)
    assert hot["p"] < 0.01
    assert hot["null_mean"] < 0.2                   # null centres near the base rate
    # an observed value at the base rate should look unremarkable
    typ = stats.permutation_p(0.1, ref, frac, n_draw=200, n_perm=1000, rng=0)
    assert 0.2 < typ["p"] < 0.8


def test_permutation_p_never_zero():
    ref = np.zeros(500)
    out = stats.permutation_p(1.0, ref, np.mean, n_draw=10, n_perm=100, rng=0)
    assert out["p"] == 1 / (100 + 1)                # +1/+1 correction floor
