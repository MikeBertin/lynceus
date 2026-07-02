import numpy as np

from core import config, data
from core.morphology import CLASSES


def test_asinh_stretch_range():
    rng = np.random.default_rng(0)
    img = rng.exponential(1.0, size=(3, 32, 32)).astype(np.float32)
    out = data.asinh_stretch(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_asinh_colour_stretch_preserves_band_ratios():
    # The Lupton property behind the whole M3 fix: with colour=True the per-band
    # sky is subtracted but ONE shared linear scale is applied per pixel, so the
    # flux ratios between bands (= colour) survive the stretch.
    img = np.zeros((3, 32, 32), np.float32)
    img += np.array([10.0, 20.0, 30.0])[:, None, None]                # flat skies
    img[:, 8:11, 8:11] += np.array([1.0, 2.0, 4.0])[:, None, None]   # red source, 1:2:4
    img[:, 24:26, 24:26] += np.array([50.0, 100.0, 200.0])[:, None, None]  # bright source

    out = data.asinh_stretch(img, colour=True)
    assert out.min() >= 0.0 and out.max() <= 1.0
    src = out[:, 9, 9]
    assert np.allclose(src / src[0], [1.0, 2.0, 4.0], rtol=1e-3)

    # the old per-channel stretch normalises each band independently — exactly
    # the wash-out that had hidden the Little Red Dots' redness
    flat = data.asinh_stretch(img, colour=False)[:, 9, 9]
    assert not np.allclose(flat / flat[0], [1.0, 2.0, 4.0], rtol=0.1)


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
