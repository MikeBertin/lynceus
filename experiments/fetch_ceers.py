"""Build a REAL CEERS cutout dataset.

Labels come from single-Sersic profile fits in the DJA morphology catalogue
(van der Wel et al. 2025); cutouts come from the DJA grizli cutout service
(real JWST NIRCam pixels, no mosaic download).

1. Download the catalogue once:
     curl -L -o data/ceers_morpho.fits.gz \
       https://s3.amazonaws.com/aurelien-sepp/ceers-full-grizli-v7.2/catalog/ceers-full-grizli-v7.2_morpho-phot.fits.gz
2. Select + fetch:
     python -m experiments.fetch_ceers --per-class 300
"""
from __future__ import annotations

import argparse
import csv
import shutil

import numpy as np

from core import config, data

CATALOG = config.DATA_DIR / "ceers_morpho.fits.gz"

# Label thresholds on the single-Sersic fit (see plan.md / README).
MAG_LIMIT = 25.0        # F200W model magnitude, bright enough to show structure
N_DISK = 1.2            # Sersic index below -> late-type / disk
N_SPHEROID = 2.5        # Sersic index above -> early-type / spheroid
R_COMPACT_ARCSEC = 0.09 # effective radius below -> compact / unresolved


def select(per_class: int, seed: int) -> list[dict]:
    from astropy.io import fits

    if not CATALOG.exists():
        raise SystemExit(f"Missing {CATALOG}; download it first (see module docstring).")
    h = fits.open(CATALOG)
    d1, d2 = h[1].data, h[2].data
    n = np.asarray(d2["SERSIC"], float)
    rad = np.asarray(d2["RADIUS"], float) * 3600.0   # deg -> arcsec
    mag = np.asarray(d2["MAG_MODEL_F200W"], float)
    sf = np.asarray(d2["source_flags"], float)
    ra = np.asarray(d2["RA_MODEL"], float)
    dec = np.asarray(d2["DEC_MODEL"], float)
    z = np.asarray(d1["z_phot"], float)

    base = (np.isfinite(n) & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
            & np.isfinite(rad) & (rad > 0) & (mag < MAG_LIMIT) & np.isin(sf, [0, 2]))
    masks = {
        "compact": base & (rad < R_COMPACT_ARCSEC),
        "disk": base & (rad >= R_COMPACT_ARCSEC) & (n < N_DISK),
        "spheroid": base & (rad >= R_COMPACT_ARCSEC) & (n > N_SPHEROID),
    }
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for label, m in masks.items():
        idx = np.where(m)[0]
        rng.shuffle(idx)
        take = idx[:per_class]
        print(f"  {label:9s}: {m.sum():5d} available -> taking {len(take)}")
        for i in take:
            zi = z[i]
            rows.append({
                "id": f"ceers_{int(i):06d}",
                "ra": f"{ra[i]:.6f}", "dec": f"{dec[i]:.6f}",
                "label": label,
                "redshift": f"{zi:.3f}" if np.isfinite(zi) and zi > 0 else "",
                "mag_f200w": f"{mag[i]:.2f}",
                "sersic": f"{n[i]:.2f}",
                "reff_arcsec": f"{rad[i]:.3f}",
            })
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--size", type=float, default=3.0, help="cutout size in arcsec")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fresh", action="store_true",
                    help="wipe the cutout cache before fetching")
    args = ap.parse_args()

    print("Selecting labelled sources from the DJA morphology catalogue...")
    rows = select(args.per_class, args.seed)
    csv_path = config.DATA_DIR / "ceers_labels.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} labelled positions -> {csv_path}")

    if args.fresh and config.CUTOUTS_DIR.exists():
        shutil.rmtree(config.CUTOUTS_DIR)
        config.CUTOUTS_DIR.mkdir(parents=True)

    print(f"Fetching real NIRCam cutouts from {data.CUTOUT_SERVICE} ...")
    recs = data.build_dataset_from_service(
        rows, size=args.size, max_workers=args.workers)
    by = {}
    for r in recs:
        by[r.label] = by.get(r.label, 0) + 1
    print(f"\nCached {len(recs)} real cutouts: {by}")
    print(f"Manifest: {data.MANIFEST}")


if __name__ == "__main__":
    main()
