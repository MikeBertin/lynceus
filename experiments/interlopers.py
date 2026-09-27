"""Q2 (RESEARCH.md): does the photo-z net beat EAZY on interloper rejection?

The failure mode behind "too many bright early galaxies": a dusty low-z galaxy
whose Balmer/4000 A break mimics a Lyman dropout, so a z~2 object is announced
as z~10. Our net outputs a full PDF and visibly keeps a second mode when torn;
EAZY (as distilled into our training labels) gives a point estimate. Question:
does the PDF actually buy a lower interloper rate at a given candidate yield?

Setup, on the ~1.8k held-out spectroscopic galaxies (same sample as the
calibration study, experiments/calibrate_photoz.val_pdfs):

* select "high-z candidates" three ways: EAZY z_phot > z_cut, our z_peak >
  z_cut, and PDF-aware P(z > z_cut) > tau for a sweep of tau;
* an interloper = a selected candidate whose true z_spec < 2 (the dusty
  impostor regime); completeness = fraction of true z_spec > z_cut selected;
* bootstrap CIs on the interloper rates (core/stats.py).

Honest caveats: the spec-z sample is itself selection-biased (spectroscopy
favours the bright and the confirmable), and above z~6 there are only ~78
truths, so CIs there are wide. This measures *relative* behaviour of the two
estimators on identical galaxies, not absolute survey rates.

    python -m experiments.interlopers
"""
from __future__ import annotations

import json

import numpy as np

from core import config, stats
from core.photoz import point_estimates, Z_CENTRES
from experiments.calibrate_photoz import val_pdfs

Z_CUTS = (4.0, 6.0)
TAUS = (0.5, 0.7, 0.9)
Z_INTERLOPER = 2.0      # selected with z_spec below this = catastrophic impostor
SEED = 0


def rate_ci(flags: np.ndarray) -> dict:
    """Interloper rate among selected, with a bootstrap CI (n small -> honest)."""
    if len(flags) == 0:
        return {"rate": None, "lo": None, "hi": None}
    ci = stats.bootstrap_ci(flags.astype(float), np.mean, rng=SEED)
    return {"rate": round(ci["point"], 3), "lo": round(ci["lo"], 3), "hi": round(ci["hi"], 3)}


def evaluate(sel: np.ndarray, y: np.ndarray, z_cut: float, label: str) -> dict:
    n_true = int((y > z_cut).sum())
    n_sel = int(sel.sum())
    comp = float((sel & (y > z_cut)).sum() / n_true) if n_true else 0.0
    inter = rate_ci((y[sel] < Z_INTERLOPER))
    row = {"selector": label, "n_selected": n_sel,
           "completeness": round(comp, 3), "interloper": inter}
    ir = "  --  " if inter["rate"] is None else f"{inter['rate']*100:5.1f}%"
    lo = "" if inter["rate"] is None else f" (CI {inter['lo']*100:.0f}-{inter['hi']*100:.0f}%)"
    print(f"  {label:<22} yield {n_sel:4d}   completeness {comp*100:5.1f}%   "
          f"interlopers {ir}{lo}")
    return row


def main() -> None:
    pdf, y, z_eazy = val_pdfs(with_eazy=True)
    z_peak = point_estimates(pdf)["z_peak"]
    print(f"{len(y)} held-out spec-z galaxies "
          f"(truths: {(y>4).sum()} at z>4, {(y>6).sum()} at z>6)")

    results = {"z_interloper": Z_INTERLOPER, "seed": SEED, "cuts": {}}
    for z_cut in Z_CUTS:
        print(f"\n== candidates at z > {z_cut:.0f} "
              f"(interloper = selected with z_spec < {Z_INTERLOPER:.0f}) ==")
        p_gt = pdf[:, Z_CENTRES > z_cut].sum(1)
        rows = [
            evaluate(z_eazy > z_cut, y, z_cut, "EAZY z_phot"),
            evaluate(z_peak > z_cut, y, z_cut, "ours z_peak"),
        ]
        for tau in TAUS:
            rows.append(evaluate(p_gt > tau, y, z_cut, f"ours P(z>{z_cut:.0f}) > {tau}"))
        results["cuts"][f"{z_cut:.0f}"] = rows

    # --- who ARE the interlopers? -------------------------------------------
    # (a) overlap with EAZY's: the net trained on EAZY labels, so galaxies EAZY
    #     mislabels were mislabelled *in training*; inheritance is the default.
    # (b) does the PDF at least flag doubt (mass below z=2) on them?
    print("\n== diagnosis ==")
    diag = {}
    for z_cut in Z_CUTS:
        ours = set(np.flatnonzero((z_peak > z_cut) & (y < Z_INTERLOPER)))
        eazy = set(np.flatnonzero((z_eazy > z_cut) & (y < Z_INTERLOPER)))
        diag[f"overlap_z{z_cut:.0f}"] = {"ours": len(ours), "eazy": len(eazy),
                                         "shared": len(ours & eazy)}
        print(f"  z>{z_cut:.0f} interlopers: ours {len(ours)}, EAZY {len(eazy)}, "
              f"shared {len(ours & eazy)}, inherited from the teacher")
    p_low = pdf[:, Z_CENTRES < Z_INTERLOPER].sum(1)
    sel = z_peak > Z_CUTS[0]
    inter, true = sel & (y < Z_INTERLOPER), sel & (y > Z_CUTS[0])
    diag["p_low_median"] = {"interlopers": round(float(np.median(p_low[inter])), 4),
                            "true_highz": round(float(np.median(p_low[true])), 4)}
    diag["p_low_gt_0.2"] = {"interlopers": round(float((p_low[inter] > 0.2).mean()), 2),
                            "true_highz": round(float((p_low[true] > 0.2).mean()), 2)}
    print(f"  PDF doubt, z>4 candidates: median P(z<2) {diag['p_low_median']['interlopers']}"
          f" (interlopers) vs {diag['p_low_median']['true_highz']} (true): a 5x whisper,"
          f" but only {diag['p_low_gt_0.2']['interlopers']:.0%} carry real low-z mass"
          f" (true high-z: {diag['p_low_gt_0.2']['true_highz']:.0%}). Confidently wrong.")
    results["diagnosis"] = diag

    path = config.MODELS_DIR / "interlopers.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
