import numpy as np

from core import config, data
from core.morphology import CLASSES


def test_asinh_stretch_range():
    rng = np.random.default_rng(0)
    img = rng.exponential(1.0, size=(3, 32, 32)).astype(np.float32)
    out = data.asinh_stretch(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_synthetic_dataset_shapes(tmp_path, monkeypatch):
    # Redirect the cache so the test never clobbers a real dataset.
    monkeypatch.setattr(config, "CUTOUTS_DIR", tmp_path)
    monkeypatch.setattr(data, "MANIFEST", tmp_path / "manifest.csv")

    recs = data.generate_synthetic_dataset(n_per_class=2, seed=1)
    assert len(recs) == 2 * len(CLASSES)

    rows = data.load_manifest()
    assert len(rows) == len(recs)

    arr = data.load_cutout(rows[0]["npy"])
    assert arr.shape == (3, config.CUTOUT_PX, config.CUTOUT_PX)
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert (tmp_path / rows[0]["png"]).exists()
    assert rows[0]["label"] in CLASSES
