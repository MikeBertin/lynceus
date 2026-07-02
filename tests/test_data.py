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


def _sky_field(rng, sigma=0.01, size=48):
    """An empty-sky cube: flat per-band skies + gaussian noise."""
    img = np.zeros((3, size, size), np.float32)
    img += np.array([0.002, 0.001, 0.003], np.float32)[:, None, None]
    img += rng.normal(0, sigma, size=img.shape).astype(np.float32)
    return img


def test_snr_stretch_keeps_empty_sky_dark():
    # The M3b fix (RESEARCH.md Q3): the percentile stretch amplifies an empty
    # field's noise to full-range static; the SNR stretch must not.
    rng = np.random.default_rng(0)
    img = _sky_field(rng)
    static = data.asinh_stretch(img, colour=True)
    dark = data.asinh_stretch_snr(img)
    assert static.mean() > 0.1          # the old failure mode, as a baseline
    assert dark.mean() < 0.01           # noise floored to (near) black
    assert dark.min() >= 0.0 and dark.max() <= 1.0


def test_snr_stretch_faint_point_source_survives():
    # A faint compact source (peak ~8 sigma — the faint-LRD regime) must stay
    # clearly above the floored sky, not vanish with it.
    rng = np.random.default_rng(1)
    img = _sky_field(rng, sigma=0.01)
    img[:, 23:26, 23:26] += np.array([0.08, 0.04, 0.01], np.float32)[:, None, None]
    out = data.asinh_stretch_snr(img)
    src = out[:, 23:26, 23:26].mean()
    sky = out[:, :12, :12].mean()
    assert src > 10 * max(sky, 1e-4)


def test_snr_stretch_preserves_band_ratios():
    # Same Lupton property as the colour stretch: one shared linear scale per
    # pixel, so a red source stays red in exactly its raw flux ratios. (Only
    # below the cap_snr saturation — a saturated band clips at 1, as it must.)
    rng = np.random.default_rng(2)
    img = _sky_field(rng, sigma=0.001)
    img[:, 20:24, 20:24] += np.array([0.016, 0.008, 0.004], np.float32)[:, None, None]
    out = data.asinh_stretch_snr(img)
    src = out[:, 21, 21]
    raw = img[:, 21, 21] - np.median(img.reshape(3, -1), axis=1)
    assert np.allclose(src / src[0], raw / raw[0], rtol=0.05)


def test_centre_anchor_crop_snaps_to_central_source_not_neighbour():
    # The anchor must lock onto the (catalogue-centred) target even when a much
    # brighter neighbour sits elsewhere in the frame — the Q3 failure where
    # "LRD matches" embedded the bright galaxy next door.
    rng = np.random.default_rng(3)
    img = _sky_field(rng, size=120)
    img[:, 57:60, 62:65] += 0.05     # faint target ~3px from centre
    img[:, 90:100, 90:100] += 5.0    # bright neighbour, far off-centre
    cy, cx = data.centre_anchor_crop(img, out_px=64, search_px=8)
    assert abs(cy - 58) <= 2 and abs(cx - 63) <= 2

    cut = data.m3b_cutout(img, out_px=64)
    assert cut.shape == (3, 64, 64)
    assert cut.min() >= 0.0 and cut.max() <= 1.0


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
