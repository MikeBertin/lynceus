"""Error bars + significance for the headline claims.

Turns Lynceus's three bare point estimates into defensible numbers, all from the
*cached* data (no re-fetch, no retrain):

* **M3: LRD enrichment.** Bootstrap 95% CI on the "5×" (resample the 216 Little
  Red Dots) and a permutation null test (216 random atlas galaxies, many times →
  p-value) showing the LRDs really are more anomalous than chance.
* **M2: kNN probes.** Morphology and field recovery as mean ± std across 5 CV
  folds, against the majority-class baseline.
* **M4: photo-z σ_NMAD.** Bootstrap 95% CI on the held-out spectroscopic
  σ_NMAD (and outlier fraction).

Writes ``models/significance.json`` and prints copy-ready strings.

    python -m experiments.significance          # the shipped M3 representation
    python -m experiments.significance --m3b     # the M3b point-source-aware one
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from core import config, anomaly, stats
from core.photoz import PhotoZNet, featurize, pdf_from_logits, point_estimates

K_ANOM = 20      # neighbours for the anomaly score (matches build_atlas.py)
K_PROBE = 15     # neighbours for the kNN morphology/field probes (reproduces 62%/63%)
SEED = 0


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _field(ra):
    ra = float(ra)
    if 33 <= ra <= 36: return "UDS"
    if 52 <= ra <= 54: return "GOODS-S"
    if 149 <= ra <= 151: return "COSMOS"
    return "?"


# ---------------------------------------------------------------------------
# M3: LRD anomaly enrichment
# ---------------------------------------------------------------------------
def lrd_enrichment(sfx: str = "") -> dict:
    feats = np.load(config.ATLAS_DIR / f"embeddings{sfx}.npy")
    lrd = np.load(config.DATA_DIR / "lrd" / f"lrd_emb{sfx}.npy")

    # exactly as experiments/build_atlas.py
    araw = anomaly.knn_anomaly(feats, feats, k=K_ANOM, exclude_self=True)
    lraw = anomaly.knn_anomaly(lrd, feats, k=K_ANOM)
    p90 = np.percentile(araw, 90)

    frac = float((lraw > p90).mean())              # fraction of LRDs in top decile
    enrich = lambda s: float((s > p90).mean()) / 0.10
    ci = stats.bootstrap_ci(lraw, enrich, rng=SEED)

    # percentile of each LRD within the atlas anomaly distribution
    lrd_pct = np.array([(araw < v).mean() for v in lraw])
    med_ci = stats.bootstrap_ci(lrd_pct, np.median, rng=SEED)

    # null: 216 random atlas galaxies, fraction landing in the top decile (~10%)
    frac_above = lambda s: float((s > p90).mean())
    perm = stats.permutation_p(frac, araw, frac_above, n_draw=len(lrd),
                               alternative="greater", rng=SEED)

    out = {
        "n_lrd": int(len(lrd)), "n_atlas": int(len(feats)),
        "frac_top_decile": round(frac, 4),
        "enrichment": round(ci["point"], 2),
        "enrichment_ci": [round(ci["lo"], 2), round(ci["hi"], 2)],
        "median_pct": round(med_ci["point"], 3),
        "median_pct_ci": [round(med_ci["lo"], 3), round(med_ci["hi"], 3)],
        "p_value": perm["p"], "null_mean_enrich": round(perm["null_mean"] / 0.10, 2),
    }
    print(f"M3  LRD enrichment {out['enrichment']}× "
          f"(95% CI {out['enrichment_ci'][0]}–{out['enrichment_ci'][1]}×, "
          f"{_p_str(perm['p'])}); median LRD at the "
          f"{out['median_pct']*100:.0f}th percentile "
          f"(CI {out['median_pct_ci'][0]*100:.0f}–{out['median_pct_ci'][1]*100:.0f})")
    return out


def _p_str(p: float) -> str:
    """Format a p-value honestly: 'p<1e-4' only when p really is below 1e-4."""
    if p >= 0.01:
        return f"p={p:.3f}"
    e = int(np.floor(-np.log10(p)))     # largest e with 1e-e still >= p...
    while 10.0 ** -e <= p:              # ...guard exact powers of ten
        e -= 1
    return f"p<1e-{e}"


# ---------------------------------------------------------------------------
# M2: kNN probes (mean ± std across folds)
# ---------------------------------------------------------------------------
def knn_probes(sfx: str = "") -> dict:
    feats = _l2(np.load(config.ATLAS_DIR / f"embeddings{sfx}.npy"))
    rows = list(csv.DictReader(open(config.ATLAS_DIR / f"atlas_meta{sfx}.csv")))
    y_morph = np.array([r["dominant"] for r in rows])
    y_field = np.array([_field(r["ra"]) for r in rows])

    def probe(y):
        s = cross_val_score(KNeighborsClassifier(K_PROBE, metric="cosine"),
                            feats, y, cv=5)
        _, c = np.unique(y, return_counts=True)
        return {"acc": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
                "baseline": round(float(c.max() / len(y)), 3)}

    out = {"k": K_PROBE, "morph": probe(y_morph), "field": probe(y_field)}
    for name in ("morph", "field"):
        d = out[name]
        print(f"M2  kNN {name:5s} {d['acc']*100:.1f}% ± {d['std']*100:.1f}% "
              f"(baseline {d['baseline']*100:.1f}%)")
    return out


# ---------------------------------------------------------------------------
# M4: photo-z σ_NMAD bootstrap CI on held-out spec-z
# ---------------------------------------------------------------------------
def photoz_ci() -> dict:
    ckpt = torch.load(config.MODELS_DIR / "photoz.pt", map_location="cpu",
                      weights_only=False)
    z = np.load(config.DATA_DIR / "photoz" / "photoz.npz", allow_pickle=True)
    X = featurize(z["flux"], z["err"])
    is_val = z["is_val"]; z_spec = z["z_spec"]
    Xva = (X[is_val] - ckpt["mu"]) / ckpt["sd"]
    yva = z_spec[is_val]

    model = PhotoZNet(in_dim=X.shape[1])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        pdf = pdf_from_logits(model(torch.from_numpy(Xva))).numpy()
    z_peak = point_estimates(pdf)["z_peak"]

    m = np.isfinite(z_peak) & np.isfinite(yva) & (yva >= 0)
    dz = (z_peak[m] - yva[m]) / (1.0 + yva[m])

    nmad = lambda d: float(1.4826 * np.median(np.abs(d - np.median(d))))
    outl = lambda d: float(np.mean(np.abs(d) > 0.15))
    ci_n = stats.bootstrap_ci(dz, nmad, rng=SEED)
    ci_o = stats.bootstrap_ci(dz, outl, rng=SEED)

    out = {
        "n_spec": int(m.sum()),
        "sigma_nmad": round(ci_n["point"], 4),
        "sigma_nmad_ci": [round(ci_n["lo"], 4), round(ci_n["hi"], 4)],
        "outlier_frac": round(ci_o["point"], 3),
        "outlier_frac_ci": [round(ci_o["lo"], 3), round(ci_o["hi"], 3)],
    }
    print(f"M4  σ_NMAD {out['sigma_nmad']:.4f} "
          f"(95% CI {out['sigma_nmad_ci'][0]:.4f}–{out['sigma_nmad_ci'][1]:.4f}); "
          f"outliers {out['outlier_frac']*100:.1f}% "
          f"(CI {out['outlier_frac_ci'][0]*100:.1f}–{out['outlier_frac_ci'][1]*100:.1f})")
    return out


def patch_web_assets(result: dict) -> None:
    """Merge the freshly-computed CIs into the web JSON, leaving layout/points
    untouched (so we never re-run UMAP or regenerate sprites just for numbers)."""
    e = result["lrd_enrichment"]
    lrds_path = config.WEB_ATLAS_DIR / "lrds.json"
    lrds = json.loads(lrds_path.read_text())
    lrds.update({
        "enrichment": e["enrichment"],
        "enrichment_ci": e["enrichment_ci"],
        "p_value": e["p_value"],
        "frac_above_p90": e["frac_top_decile"],
        "median_pct": e["median_pct"],            # true median (was mislabelled mean)
        "median_pct_ci": e["median_pct_ci"],
    })
    lrds_path.write_text(json.dumps(lrds, separators=(",", ":")))
    print(f"  patched {lrds_path.name} (enrichment + CI + p, corrected median)")

    # photo-z σ_NMAD CI: into the metrics catalogue and the dropout web payload
    pz = result["photoz"]
    ci_keys = {"sigma_nmad_ci": pz["sigma_nmad_ci"],
               "outlier_frac_ci": pz["outlier_frac_ci"]}
    mpath = config.MODELS_DIR / "photoz_metrics.json"
    m = json.loads(mpath.read_text()); m.update(ci_keys)
    mpath.write_text(json.dumps(m, indent=2))
    dpath = config.WEB_DROPOUT_DIR / "dropout.json"
    d = json.loads(dpath.read_text()); d["metrics"].update(ci_keys)
    dpath.write_text(json.dumps(d, separators=(",", ":")))
    print(f"  patched {mpath.name} + {dpath.name} (σ_NMAD CI)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3b", action="store_true",
                    help="compute the atlas numbers from the M3b representation")
    args = ap.parse_args()
    sfx = "_m3b" if args.m3b else ""

    result = {
        "variant": "m3b" if args.m3b else "m3",
        "seed": SEED, "n_boot": 10000, "n_perm": 10000,
        "lrd_enrichment": lrd_enrichment(sfx),
        "knn_probes": knn_probes(sfx),
        "photoz": photoz_ci(),          # M4 is independent of the atlas stretch
    }
    out = config.MODELS_DIR / f"significance{sfx}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")
    patch_web_assets(result)


if __name__ == "__main__":
    main()
