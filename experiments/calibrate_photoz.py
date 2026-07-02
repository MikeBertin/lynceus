"""Calibrate the photo-z PDFs — the uncertainty check σ_NMAD never makes.

σ_NMAD says the *peak* is good; it says nothing about whether the **PDF widths**
are trustworthy. The standard test is the Probability Integral Transform (PIT):
for each spec-z galaxy evaluate the predicted CDF at the true redshift; if the
PDFs are calibrated those values are uniform on [0,1]. A U-shaped PIT = the net
is over-confident (PDFs too narrow); a central hump = under-confident.

We compute the PIT and central-interval coverage on the ~1,800 held-out
spectroscopic galaxies, and probe whether a single softmax **temperature** can
calibrate them. The honest finding: it can't, and *why* is the interesting part.
The PIT is a central hump (PDFs over-dispersed in the core), yet the high-level
coverage is already good (a 90% interval contains the truth ~88% of the time).
The two standard temperature objectives then disagree — NLL wants to *widen*
(T>1) to cover the catastrophic-outlier tails, while the PIT/coverage want to
*sharpen* (T<1) to flatten the core. No global T fixes both, and sharpening would
degrade exactly the high-credible-level coverage that matters (and crush the
secondary interloper peaks the demo is about). So we deploy **T=1** and report
the diagnostic — calibration is a property to disclose, not to fake away.

    python -m experiments.calibrate_photoz

Writes models/photoz_calibration.json and patches the PIT histogram + coverage
into web/dropout/ for the demo's "honest numbers" card.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from core import config
from core.photoz import (PhotoZNet, featurize, pdf_from_logits, point_estimates,
                         photoz_metrics, temperature_scale, pit_values,
                         credible_coverage, pit_ks, nll_at_truth, fit_temperature)

LEVELS = np.array([0.5, 0.68, 0.90, 0.95])   # nominal credible levels for coverage
PIT_BINS = 10
DEPLOY_T = 1.0                                 # see module docstring — no scaling shipped


def val_pdfs() -> tuple[np.ndarray, np.ndarray]:
    """Raw (T=1) PDFs and spec-z truths for the held-out spectroscopic set."""
    ckpt = torch.load(config.MODELS_DIR / "photoz.pt", map_location="cpu",
                      weights_only=False)
    z = np.load(config.DATA_DIR / "photoz" / "photoz.npz", allow_pickle=True)
    X = featurize(z["flux"], z["err"])
    is_val = z["is_val"]
    Xva = (X[is_val] - ckpt["mu"]) / ckpt["sd"]
    yva = z["z_spec"][is_val]
    model = PhotoZNet(in_dim=X.shape[1]); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        pdf = pdf_from_logits(model(torch.from_numpy(Xva))).numpy()
    m = np.isfinite(yva) & (yva >= 0)
    return pdf[m], yva[m]


def _coverage_dict(pit):
    return {f"{int(c*100)}": round(float(v), 3)
            for c, v in zip(LEVELS, credible_coverage(pit, LEVELS))}


def _zpeak(pdf):
    return point_estimates(pdf)["z_peak"]


def main() -> None:
    pdf, y = val_pdfs()
    n = len(y)
    print(f"calibrating on {n} held-out spectroscopic galaxies")

    pit_raw = pit_values(pdf, y)
    ks_raw = pit_ks(pit_raw)
    cov_raw = _coverage_dict(pit_raw)
    print(f"raw PIT KS vs uniform = {ks_raw:.3f}")
    print(f"raw coverage          = {cov_raw}   (nominal 50/68/90/95)")
    print(f"raw NLL               = {nll_at_truth(pdf, y):.3f}")

    # The two standard temperature objectives pull in opposite directions:
    T_nll = fit_temperature(pdf, y, objective="nll")   # widens to cover outliers
    T_pit = fit_temperature(pdf, y, objective="ks")    # sharpens to flatten core
    print(f"objective tension: NLL-optimal T={T_nll:.2f} (widen) vs "
          f"PIT/KS-optimal T={T_pit:.2f} (sharpen) — no single T calibrates both.")
    cov_sharp = _coverage_dict(pit_values(temperature_scale(pdf, T_pit), y))
    print(f"  sharpening to T={T_pit:.2f} would push 90% coverage "
          f"{cov_raw['90']}->{cov_sharp['90']} (worse where it matters).")

    # Deploy T=1: peak (σ_NMAD) and PDFs ship as-is; calibration is disclosed.
    m = photoz_metrics(_zpeak(pdf), y)
    print(f"shipping T={DEPLOY_T} (no scaling); σ_NMAD stays {m.sigma_nmad:.4f}")

    hist_raw, _ = np.histogram(pit_raw, bins=PIT_BINS, range=(0, 1))
    result = {
        "n_spec": n, "deploy_temperature": DEPLOY_T, "pit_bins": PIT_BINS,
        "ks_raw": round(ks_raw, 3),
        "pit_hist_raw": (hist_raw / n).round(4).tolist(),
        "coverage_raw": cov_raw,
        "sigma_nmad": round(m.sigma_nmad, 4),
        "objective_tension": {"T_nll": round(T_nll, 2), "T_pit": round(T_pit, 2),
                              "coverage90_if_sharpened": cov_sharp["90"]},
    }
    out = config.MODELS_DIR / "photoz_calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")
    patch_web(result)


def calibration_block(result: dict) -> dict:
    """The web-facing summary of a saved calibration result.

    Shared with experiments.build_dropout_assets so a full asset rebuild keeps
    the calibration disclosure instead of silently dropping it.
    """
    return {
        "temperature": result["deploy_temperature"],
        "ks_raw": result["ks_raw"],
        "pit_hist": result["pit_hist_raw"],
        "coverage": result["coverage_raw"],
        "t_nll": result["objective_tension"]["T_nll"],
        "t_pit": result["objective_tension"]["T_pit"],
    }


def patch_web(result: dict) -> None:
    """Push the calibration diagnostic into the dropout web payload."""
    dpath = config.WEB_DROPOUT_DIR / "dropout.json"
    d = json.loads(dpath.read_text())
    d["calibration"] = calibration_block(result)
    d.get("metrics", {}).pop("temperature", None)   # drop any stale earlier value
    dpath.write_text(json.dumps(d, separators=(",", ":")))

    # photoz_meta.json deliberately does NOT carry the temperature: app.js never
    # applies one (we ship T=1), so advertising it there would be a landmine.
    mpath = config.WEB_DROPOUT_DIR / "photoz_meta.json"
    meta = json.loads(mpath.read_text())
    if meta.pop("temperature", None) is not None:   # scrub any stale earlier value
        mpath.write_text(json.dumps(meta, separators=(",", ":")))
    print(f"  patched {dpath.name}")


if __name__ == "__main__":
    main()
