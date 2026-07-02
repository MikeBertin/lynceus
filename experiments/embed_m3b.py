"""Embed the M3b cutouts (atlas + LRDs) with the M3b encoder.

Row order matches the shipped embeddings' conventions so Q1/Q3 rerun
unchanged: atlas rows follow atlas_meta.csv (filtered to cutouts that exist),
LRD rows follow lrd_meta.csv. Writes the filtered atlas meta alongside, so the
downstream scripts never have to guess which rows made it.

    python -m experiments.embed_m3b

Outputs: data/atlas/embeddings_m3b.npy, data/atlas/atlas_meta_m3b.csv,
         data/lrd/lrd_emb_m3b.npy
"""
from __future__ import annotations

import csv

import numpy as np

from core import config
from core.embed import embed_paths
from experiments.build_atlas import load_encoder
from experiments.build_m3b_cutouts import ATLAS_M3B, LRD_M3B


def main() -> None:
    enc = load_encoder("ssl_encoder_m3b.pt")

    rows = [r for r in csv.DictReader(open(config.ATLAS_DIR / "atlas_meta.csv"))
            if (ATLAS_M3B / f"{r['id']}.npy").exists()]
    feats = embed_paths(enc, [ATLAS_M3B / f"{r['id']}.npy" for r in rows])
    np.save(config.ATLAS_DIR / "embeddings_m3b.npy", feats)
    with open(config.ATLAS_DIR / "atlas_meta_m3b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"atlas: {feats.shape} -> embeddings_m3b.npy ({len(rows)} rows)")

    lrd_rows = [r for r in csv.DictReader(open(config.DATA_DIR / "lrd" / "lrd_meta.csv"))
                if (LRD_M3B / f"{r['id']}.npy").exists()]
    lrd = embed_paths(enc, [LRD_M3B / f"{r['id']}.npy" for r in lrd_rows])
    np.save(config.DATA_DIR / "lrd" / "lrd_emb_m3b.npy", lrd)
    with open(config.DATA_DIR / "lrd" / "lrd_meta_m3b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lrd_rows[0].keys()))
        w.writeheader(); w.writerows(lrd_rows)
    print(f"lrds:  {lrd.shape} -> lrd_emb_m3b.npy ({len(lrd_rows)} rows)")


if __name__ == "__main__":
    main()
