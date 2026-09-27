import numpy as np

from core.anomaly import knn_anomaly, nearest_distance


def _cluster_with_outlier(n=300, dim=64, seed=0):
    """A tight cluster around one direction + a single orthogonal outlier."""
    rng = np.random.default_rng(seed)
    base = np.zeros(dim); base[0] = 1.0
    cluster = base[None] + 0.02 * rng.normal(size=(n, dim))
    outlier = np.zeros(dim); outlier[1] = 1.0
    return np.vstack([cluster, outlier[None]]).astype(np.float32)


def test_knn_anomaly_flags_planted_outlier():
    emb = _cluster_with_outlier()
    scores = knn_anomaly(emb, emb, k=20, exclude_self=True)
    assert scores.argmax() == len(emb) - 1                   # the outlier wins
    assert scores[-1] > 10 * np.median(scores[:-1])          # by a wide margin
    assert scores.min() > 0.0                                # self was excluded


def test_knn_anomaly_exclude_self_matters():
    # scoring a set against itself without excluding self lets every point count
    # its own zero distance, so scores must be strictly lower
    emb = _cluster_with_outlier()
    with_self = knn_anomaly(emb, emb, k=5, exclude_self=False)
    without = knn_anomaly(emb, emb, k=5, exclude_self=True)
    assert np.all(with_self <= without)
    assert with_self.mean() < without.mean()


def test_nearest_distance_orders_by_similarity():
    emb = _cluster_with_outlier()
    targets = emb[-1:]                                        # the outlier
    d = nearest_distance(emb[:-1], targets)                   # cluster -> outlier
    assert d.min() > 0.5                                      # orthogonal: all far
    assert nearest_distance(targets, targets)[0] < 1e-6       # self-match is ~0
