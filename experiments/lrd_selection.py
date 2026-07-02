"""Q1 (RESEARCH.md): is the anomaly score an LRD *selection function*?

The 5x enrichment (experiments/significance.py) says the known Little Red Dots
are over-represented among the atlas anomalies. This asks the follow-up a
referee would: could you *find* LRDs with it? I.e. treat "anomaly score above
threshold" as a selector and measure completeness (fraction of known LRDs
caught) against the fraction of the atlas you'd have to inspect, as the
threshold sweeps — ROC, AUC, and a table of usable operating points.

Positives: the 216 Kokorev+24 LRDs, embedded with the atlas encoder.
Negatives: the 9,673 atlas galaxies (approximate negatives — the atlas sample
is GZ-stratified, not flux-limited, and could contain uncatalogued LRDs; so
this is a selection function *relative to this atlas*).

All cached data; no refetch/retrain.

    python -m experiments.lrd_selection          # the shipped M3 embeddings
    python -m experiments.lrd_selection --m3b    # the M3b point-source-aware rerun
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from core import config, anomaly, stats

K_ANOM = 20        # matches build_atlas.py / significance.py
SEED = 0
TOP_FRACS = (0.01, 0.02, 0.05, 0.10, 0.20)   # "inspect the top X% of the atlas"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3b", action="store_true",
                    help="use the M3b embeddings (noise-aware stretch + anchored crop)")
    args = ap.parse_args()
    sfx = "_m3b" if args.m3b else ""

    feats = np.load(config.ATLAS_DIR / f"embeddings{sfx}.npy")
    lrd = np.load(config.DATA_DIR / "lrd" / f"lrd_emb{sfx}.npy")
    araw = anomaly.knn_anomaly(feats, feats, k=K_ANOM, exclude_self=True)
    lraw = anomaly.knn_anomaly(lrd, feats, k=K_ANOM)

    # AUC. Against a fixed negative set, AUC is the mean of the per-LRD
    # percentiles, so the bootstrap CI comes straight from core.stats.
    y = np.concatenate([np.ones(len(lraw)), np.zeros(len(araw))])
    auc = float(roc_auc_score(y, np.concatenate([lraw, araw])))
    lrd_pct = np.array([(araw < v).mean() for v in lraw])
    ci = stats.bootstrap_ci(lrd_pct, np.mean, rng=SEED)
    print(f"AUC = {auc:.3f}  (bootstrap 95% CI {ci['lo']:.3f}-{ci['hi']:.3f}, "
          f"n={len(lraw)} LRDs vs {len(araw)} atlas galaxies)")

    # Operating points: keep the top f of the atlas by anomaly.
    print(f"\n{'keep top':>9}  {'completeness':>12}  {'enrichment':>10}")
    points = []
    for f in TOP_FRACS:
        thr = np.percentile(araw, 100 * (1 - f))
        comp = float((lraw > thr).mean())
        comp_ci = stats.bootstrap_ci((lraw > thr).astype(float), np.mean, rng=SEED)
        points.append({"top_frac": f, "completeness": round(comp, 3),
                       "completeness_ci": [round(comp_ci["lo"], 3), round(comp_ci["hi"], 3)],
                       "enrichment": round(comp / f, 1)})
        print(f"{f*100:8.0f}%  {comp*100:11.1f}%  {comp/f:9.1f}x   "
              f"(CI {comp_ci['lo']*100:.1f}-{comp_ci['hi']*100:.1f}%)")

    # Where does the score stop helping? Completeness of the *bottom* half.
    below_median = float((lraw < np.median(araw)).mean())
    print(f"\nLRDs scoring below the atlas median: {below_median*100:.1f}% "
          f"(the tail no anomaly cut can reach)")

    # Characterise that missed tail against the Kokorev catalogue: is the score
    # blind to a random subset, or to a *kind* of LRD?
    tail = characterise_tail(araw, lraw, sfx)

    out = {
        "variant": "m3b" if args.m3b else "m3",
        "k": K_ANOM, "seed": SEED,
        "n_lrd": int(len(lraw)), "n_atlas": int(len(araw)),
        "auc": round(auc, 3), "auc_ci": [round(ci["lo"], 3), round(ci["hi"], 3)],
        "operating_points": points,
        "frac_lrd_below_atlas_median": round(below_median, 3),
        "missed_tail": tail,
    }
    path = config.MODELS_DIR / f"lrd_selection{sfx}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


def characterise_tail(araw: np.ndarray, lraw: np.ndarray, sfx: str = "") -> dict:
    """Compare missed (below atlas-median) vs caught (top-decile) LRDs on the
    Kokorev catalogue's physical columns — brightness, dust, colour."""
    import csv
    from astropy.io import fits
    from scipy.stats import spearmanr

    cat = fits.open(config.DATA_DIR / "lrd_kokorev.fits")[1].data
    meta = config.DATA_DIR / "lrd" / f"lrd_meta{sfx}.csv"
    rows = list(csv.DictReader(open(meta)))
    # our cached LRD order -> catalogue rows, matched by position
    idx = np.array([np.argmin((cat["ra"] - float(r["ra"])) ** 2 +
                              (cat["dec"] - float(r["dec"])) ** 2) for r in rows])
    f444 = np.asarray(cat["f444w_flux"], float)[idx]
    f200 = np.asarray(cat["f200w_flux"], float)[idx]
    av = np.asarray(cat["av"], float)[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        red = np.where(f200 > 0, f444 / f200, np.nan)      # crude colour proxy

    missed = lraw < np.median(araw)
    caught = lraw > np.percentile(araw, 90)
    med = lambda v, m: float(np.nanmedian(v[m]))
    rho_flux = spearmanr(lraw, np.log10(np.clip(f444, 1e-4, None))).statistic
    rho_av = spearmanr(lraw, av, nan_policy="omit").statistic

    tail = {
        "n_missed": int(missed.sum()), "n_caught": int(caught.sum()),
        "f444w_uJy": {"missed": round(med(f444, missed), 3), "caught": round(med(f444, caught), 3)},
        "av": {"missed": round(med(av, missed), 2), "caught": round(med(av, caught), 2)},
        "f444w_over_f200w": {"missed": round(med(red, missed), 1), "caught": round(med(red, caught), 1)},
        "spearman_score_vs_logf444w": round(float(rho_flux), 2),
        "spearman_score_vs_av": round(float(rho_av), 2),
    }
    print("\nmissed tail (below atlas median) vs caught (top decile):")
    print(f"  F444W flux   {tail['f444w_uJy']['missed']} vs {tail['f444w_uJy']['caught']} uJy"
          f"   (score~log flux: rho={tail['spearman_score_vs_logf444w']})")
    print(f"  dust Av      {tail['av']['missed']} vs {tail['av']['caught']}"
          f"          (score~Av: rho={tail['spearman_score_vs_av']})")
    print(f"  F444W/F200W  {tail['f444w_over_f200w']['missed']} vs {tail['f444w_over_f200w']['caught']}")
    if tail["spearman_score_vs_logf444w"] >= 0.2:
        print("  -> the score selects the bright, red end of the LRD population.")
    else:
        print(f"  -> no meaningful brightness selection (rho~0; n_missed={tail['n_missed']}).")
    return tail


if __name__ == "__main__":
    main()
