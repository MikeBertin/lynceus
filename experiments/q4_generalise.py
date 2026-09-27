"""Q4 (RESEARCH.md): does the M3b encoder generalise across fields?

Our own probe says field is ~63% guessable from the embedding, a survey
depth/PSF fingerprint. So the threat to the LRD-anomaly result is that it is
partly a *depth* artefact. The test: retrain SimCLR with one field held out
(``train_atlas --exclude-field``), then measure the LRD-anomaly signal on that
held-out field's galaxies using the encoder that never saw the field, and
compare against the all-field M3b encoder evaluated on *exactly the same*
galaxies + LRDs. Only the encoder differs, so field-size effects cancel and
what's left is generalisation.

For a held-out field F:
  * negatives = F's atlas galaxies, positives = F's known LRDs;
  * anomaly = mean cosine distance to the 20 nearest atlas galaxies *of F*
    (so the reference cloud is also held-out, so no leakage from the trained
    fields);
  * report AUC and top-10% enrichment, each with a bootstrap CI, for both
    encoders side by side; plus a within-field kNN-morphology probe.

    python -m experiments.q4_generalise --field COSMOS
    python -m experiments.q4_generalise --all      # once all 3 encoders exist

Reads: models/ssl_encoder_m3b.pt (all-field), models/ssl_encoder_m3b_no{F}.pt
       (held-out); data/atlas/cutouts_m3b, data/atlas/embeddings_m3b.npy,
       data/atlas/atlas_meta_m3b.csv; data/lrd/cutouts_m3b, lrd_emb_m3b.npy,
       lrd_meta_m3b.csv.
Writes: models/q4_generalise.json
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from core import config, anomaly, stats
from core.embed import embed_paths
from experiments.build_atlas import load_encoder

K_ANOM = 20
K_PROBE = 15
SEED = 0
FIELDS = ("UDS", "GOODS-S", "COSMOS")
ATLAS_M3B = config.ATLAS_DIR / "cutouts_m3b"
LRD_M3B = config.DATA_DIR / "lrd" / "cutouts_m3b"


def _field(ra) -> str:
    ra = float(ra)
    if 33 <= ra <= 36: return "UDS"
    if 52 <= ra <= 54: return "GOODS-S"
    if 149 <= ra <= 151: return "COSMOS"
    return "?"


def _enrich_auc(atlas_emb: np.ndarray, lrd_emb: np.ndarray) -> dict:
    """AUC + top-10% enrichment (with bootstrap CIs) of LRDs vs one field."""
    araw = anomaly.knn_anomaly(atlas_emb, atlas_emb, k=K_ANOM, exclude_self=True)
    lraw = anomaly.knn_anomaly(lrd_emb, atlas_emb, k=K_ANOM)
    y = np.concatenate([np.ones(len(lraw)), np.zeros(len(araw))])
    auc = float(roc_auc_score(y, np.concatenate([lraw, araw])))
    # AUC over a fixed negative set = mean of per-LRD percentiles -> bootstrap it
    lrd_pct = np.array([(araw < v).mean() for v in lraw])
    auc_ci = stats.bootstrap_ci(lrd_pct, np.mean, rng=SEED)
    p90 = np.percentile(araw, 90)
    enr = float((lraw > p90).mean()) / 0.10
    enr_ci = stats.bootstrap_ci(lraw, lambda s: float((s > p90).mean()) / 0.10, rng=SEED)
    return {"auc": round(auc, 3), "auc_ci": [round(auc_ci["lo"], 3), round(auc_ci["hi"], 3)],
            "enrichment": round(enr, 2),
            "enrichment_ci": [round(enr_ci["lo"], 2), round(enr_ci["hi"], 2)]}


def _knn_morph(emb: np.ndarray, y: np.ndarray) -> dict:
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    s = cross_val_score(KNeighborsClassifier(K_PROBE, metric="cosine"), x, y, cv=5)
    _, c = np.unique(y, return_counts=True)
    return {"acc": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
            "baseline": round(float(c.max() / len(y)), 3)}


def evaluate(field: str) -> dict:
    # --- the held-out field's galaxies + LRDs (row order = the meta files) ----
    arows = list(csv.DictReader(open(config.ATLAS_DIR / "atlas_meta_m3b.csv")))
    amask = np.array([_field(r["ra"]) == field for r in arows])
    a_all = np.load(config.ATLAS_DIR / "embeddings_m3b.npy")[amask]     # all-field encoder
    a_ids = [r["id"] for r, m in zip(arows, amask) if m]
    a_morph = np.array([r["dominant"] for r, m in zip(arows, amask) if m])

    lrows = list(csv.DictReader(open(config.DATA_DIR / "lrd" / "lrd_meta_m3b.csv")))
    lmask = np.array([_field(r["ra"]) == field for r in lrows])
    l_all = np.load(config.DATA_DIR / "lrd" / "lrd_emb_m3b.npy")[lmask]
    l_ids = [r["id"] for r, m in zip(lrows, lmask) if m]

    # --- re-embed the same galaxies/LRDs with the held-out encoder ------------
    enc = load_encoder(f"ssl_encoder_m3b_no{field}.pt")
    a_ho = embed_paths(enc, [ATLAS_M3B / f"{i}.npy" for i in a_ids])
    l_ho = embed_paths(enc, [LRD_M3B / f"{i}.npy" for i in l_ids])

    print(f"\n=== held out {field}: {amask.sum()} atlas galaxies, {lmask.sum()} LRDs ===")
    res = {"field": field, "n_atlas": int(amask.sum()), "n_lrd": int(lmask.sum())}
    res["all_field_encoder"] = {**_enrich_auc(a_all, l_all), "knn_morph": _knn_morph(a_all, a_morph)}
    res["heldout_encoder"] = {**_enrich_auc(a_ho, l_ho), "knn_morph": _knn_morph(a_ho, a_morph)}
    for tag in ("all_field_encoder", "heldout_encoder"):
        d = res[tag]
        label = "all-field enc" if tag == "all_field_encoder" else "HELD-OUT enc"
        print(f"  {label}: AUC {d['auc']} (CI {d['auc_ci'][0]}-{d['auc_ci'][1]})  "
              f"enrichment {d['enrichment']}x (CI {d['enrichment_ci'][0]}-{d['enrichment_ci'][1]})  "
              f"kNN-morph {d['knn_morph']['acc']*100:.1f}%")
    a, h = res["all_field_encoder"], res["heldout_encoder"]
    survives = h["enrichment_ci"][1] >= a["enrichment_ci"][0]
    res["enrichment_survives"] = bool(survives)
    print(f"  -> enrichment survives (CIs overlap): {survives}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=FIELDS)
    ap.add_argument("--all", action="store_true", help="evaluate all three fields")
    args = ap.parse_args()

    fields = FIELDS if args.all else [args.field]
    out_path = config.MODELS_DIR / "q4_generalise.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    for f in fields:
        ck = config.MODELS_DIR / f"ssl_encoder_m3b_no{f}.pt"
        if not ck.exists():
            print(f"skip {f}: {ck.name} not found (retrain not finished)")
            continue
        existing[f] = evaluate(f)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
