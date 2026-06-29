"""Fetch the published Little Red Dots and embed them with our SSL encoder.

Catalogue: Kokorev et al. 2024, "A Census of Photometrically Selected Little Red
Dots at 4<z<9 in JWST Blank Fields" (github.com/VasilyKokorev/lrd_phot). We keep
the LRDs that fall in our atlas fields (GOODS-S, COSMOS, UDS), pull their real
JWST cutouts from the DJA service, and embed them — so M3 can show whether these
independently-discovered objects land in the anomalous part of our atlas.

    curl -L -o data/lrd_kokorev.fits \
      https://raw.githubusercontent.com/VasilyKokorev/lrd_phot/master/lrd_table_v1.1.fits
    python -m experiments.fetch_lrd
"""
from __future__ import annotations

import csv

import numpy as np
import torch
from PIL import Image

from core import config, data
from core.ssl import embed_tensor
from experiments.build_atlas import load_encoder

CATALOG = config.DATA_DIR / "lrd_kokorev.fits"
LRD_DIR = config.DATA_DIR / "lrd"
CUTS = LRD_DIR / "cutouts"
FIELDS = ((33, 36), (52, 54), (149, 151))   # UDS, GOODS-S, COSMOS


def main() -> None:
    from astropy.table import Table
    CUTS.mkdir(parents=True, exist_ok=True)
    t = Table.read(CATALOG)
    ra = np.asarray(t["ra"], float); dec = np.asarray(t["dec"], float)
    z = np.asarray(t["z_phot"], float)
    reff = np.asarray(t["r_eff_50_pix"], float)
    in_field = np.zeros(len(t), bool)
    for a, b in FIELDS:
        in_field |= (ra >= a) & (ra <= b)
    idx = np.where(in_field)[0]
    print(f"{len(idx)} LRDs in our fields; fetching cutouts...")

    enc = load_encoder().to(config.get_device()).eval()
    rows, embs = [], []
    for n, i in enumerate(idx):
        rid = f"lrd_{int(t['id'][i])}"
        cube = data.fetch_service_cube(float(ra[i]), float(dec[i]))
        if cube is None:
            continue
        st = data.asinh_stretch(np.nan_to_num(cube)).astype(np.float32)
        np.save(CUTS / f"{rid}.npy", st)
        rgb = (np.clip(st, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(CUTS / f"{rid}.png")
        with torch.no_grad():
            embs.append(enc(embed_tensor(st).unsqueeze(0).to(config.get_device())).cpu().numpy()[0])
        rows.append({"id": rid, "ra": f"{ra[i]:.6f}", "dec": f"{dec[i]:.6f}",
                     "z": f"{z[i]:.2f}" if np.isfinite(z[i]) else "",
                     "reff_pix": f"{reff[i]:.2f}" if np.isfinite(reff[i]) else ""})
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(idx)} ({len(rows)} cached)", flush=True)

    np.save(LRD_DIR / "lrd_emb.npy", np.array(embs))
    with open(LRD_DIR / "lrd_meta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Cached {len(rows)} LRD cutouts + embeddings -> {LRD_DIR}")


if __name__ == "__main__":
    main()
