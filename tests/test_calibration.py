import numpy as np

from core import photoz as P


def test_temperature_scale_matches_softmax_of_scaled_logits():
    # the identity that lets us re-temperature a stored PDF without the logits
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, P.Z_BINS))
    pdf = np.exp(logits); pdf /= pdf.sum(1, keepdims=True)
    for T in (0.5, 1.0, 2.0):
        direct = np.exp(logits / T); direct /= direct.sum(1, keepdims=True)
        assert np.allclose(P.temperature_scale(pdf, T), direct, atol=1e-9)


def test_temperature_preserves_argmax():
    # σ_NMAD is peak-based, so temperature scaling must not move the mode
    rng = np.random.default_rng(1)
    pdf = rng.random((10, P.Z_BINS)); pdf /= pdf.sum(1, keepdims=True)
    for T in (0.3, 2.5):
        assert np.array_equal(pdf.argmax(1), P.temperature_scale(pdf, T).argmax(1))


def test_pit_of_uniform_pdf_is_linear_cdf():
    pdf = np.full((1, P.Z_BINS), 1.0 / P.Z_BINS)
    assert np.isclose(P.pit_values(pdf, np.array([4.0]))[0], 4.0 / P.Z_MAX, atol=1e-6)


def test_coverage_matches_nominal_for_uniform_pit():
    pit = np.linspace(0, 1, 10001)
    levels = np.array([0.5, 0.9])
    assert np.allclose(P.credible_coverage(pit, levels), levels, atol=0.01)


def test_fit_temperature_sharpens_an_overdispersed_model():
    # predictions that scatter ~0.5 around truth, but PDFs of width 2.0 -> the
    # PITs pile up near 0.5 (over-dispersed) -> KS objective should sharpen (T<1)
    zc = P.Z_CENTRES
    rng = np.random.default_rng(0)
    ztrue = rng.uniform(5, 11, 400)
    zpred = ztrue + rng.normal(0, 0.5, 400)
    pdf = np.exp(-0.5 * ((zc[None] - zpred[:, None]) / 2.0) ** 2)
    pdf /= pdf.sum(1, keepdims=True)
    assert P.pit_ks(P.pit_values(pdf, ztrue)) > 0.1   # genuinely miscalibrated
    assert P.fit_temperature(pdf, ztrue, objective="ks") < 1.0
