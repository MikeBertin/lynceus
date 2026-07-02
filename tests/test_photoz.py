import numpy as np

from core.photoz import (FEATURE_DIM, PHOTOZ_BANDS, featurize, point_estimates,
                         soft_labels, Z_CENTRES)

B = len(PHOTOZ_BANDS)


def _fake_photometry(n=5, seed=0):
    rng = np.random.default_rng(seed)
    flux = rng.uniform(1.0, 20.0, size=(n, B))
    err = rng.uniform(0.05, 0.2, size=(n, B))
    return flux, err


def test_featurize_shape_and_determinism():
    flux, err = _fake_photometry()
    a, b = featurize(flux, err), featurize(flux, err)
    assert a.shape == (5, FEATURE_DIM) and a.dtype == np.float32
    assert np.array_equal(a, b)


def test_featurize_brightness_scale_invariance():
    # the design property: the network sees the colour *shape* of the SED, so a
    # 100x brighter galaxy differs ONLY in the final brightness feature (+2 dex)
    flux, err = _fake_photometry()
    a = featurize(flux, err)
    b = featurize(flux * 100.0, err * 100.0)
    assert np.allclose(b[:, :-1], a[:, :-1], atol=1e-6)
    assert np.allclose(b[:, -1] - a[:, -1], 2.0, atol=1e-6)


def test_featurize_masks_non_detections():
    flux, err = _fake_photometry(n=1)
    flux[0, 2], err[0, 2] = 1.0, 10.0    # S/N = 0.1 -> below SN_MIN
    flux[0, 3] = -99.0                   # catalogue BAD sentinel
    f = featurize(flux, err)[0]
    for band in (2, 3):
        assert f[band] == 0.0            # flux block zeroed ("this band is dark")
        assert f[B + band] == 0.0        # detection mask off
        assert f[2 * B + band] == 0.0    # confidence zero
    assert f[B + 0] == 1.0               # a healthy band stays detected


def test_point_estimates_ignores_secondary_mode():
    # a z~10 dropout with a z~2 interloper mode: z_peak must stay on the main
    # peak (that's its whole job), while a global mean is dragged into the gap
    main = np.exp(-0.5 * ((Z_CENTRES - 10.0) / 0.3) ** 2) * 0.7
    inter = np.exp(-0.5 * ((Z_CENTRES - 2.0) / 0.3) ** 2) * 0.3
    pdf = (main + inter)[None]
    pe = point_estimates(pdf)
    assert abs(pe["z_mode"][0] - 10.0) < 0.2
    assert abs(pe["z_peak"][0] - 10.0) < 0.2
    assert pe["z_mean"][0] < 8.5         # the mean falls between the modes


def test_soft_labels_are_normalised_and_centred():
    z = np.array([1.0, 8.0])
    w = soft_labels(z)
    assert np.allclose(w.sum(1), 1.0, atol=1e-6)
    assert np.allclose(Z_CENTRES[w.argmax(1)], z, atol=0.1)
