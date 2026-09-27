"""Build a REAL dataset with REAL human visual-morphology labels.

Labels: Galaxy Zoo: CANDELS volunteer vote fractions (Simmons et al. 2017), the
canonical smooth / featured / merger split. Imagery: real JWST/NIRCam cutouts
from the DJA grizli cutout service. Galaxy Zoo classified HST/CANDELS imaging in
GOODS-S, COSMOS and UDS. All three fields are now covered by deep JWST imaging,
so we cross-match by RA/Dec and pull the JWST cutout for each classified galaxy.

1. Download the Galaxy Zoo: CANDELS table once (~52 MB, gitignored):
     curl -L -o data/gz_candels.fits \
       https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-candels/gz_candels_table_2_main_release.fits
2. Select + fetch:
     python -m experiments.fetch_gz --per-class 300
"""
from __future__ import annotations

import argparse
import csv
import shutil

import numpy as np

from core import config, data

CATALOG = config.DATA_DIR / "gz_candels.fits"

MIN_VOTES = 20          # require >= this many volunteer classifications
F_FEATURED = 0.5        # vote fraction thresholds
F_SMOOTH = 0.6
F_MERGER = 0.5
# CANDELS fields covered by both Galaxy Zoo and the DJA JWST cutout service.
FIELDS = {"goods-s": (52.0, 54.0), "cosmos": (149.0, 151.0), "uds": (33.0, 36.0)}


def select(per_class: int, seed: int) -> list[dict]:
    from astropy.io import fits

    if not CATALOG.exists():
        raise SystemExit(f"Missing {CATALOG}; download it first (see module docstring).")
    d = fits.open(CATALOG)[1].data
    ra = np.asarray(d["RA"], float)
    dec = np.asarray(d["Dec"], float)
    nc = np.asarray(d["num_classifications"], float)
    smooth = np.asarray(d["t00_smooth_or_featured_a0_smooth_frac"], float)
    feat = np.asarray(d["t00_smooth_or_featured_a1_features_frac"], float)
    merg = np.asarray(d["t16_merging_tidal_debris_a0_merging_frac"], float)

    in_field = np.zeros(len(d), bool)
    for r0, r1 in FIELDS.values():
        in_field |= (ra >= r0) & (ra <= r1)
    base = in_field & np.isfinite(ra) & np.isfinite(dec) & (nc >= MIN_VOTES)

    # Disjoint classes: mergers first, then non-merger smooth / featured.
    merger = base & (merg > F_MERGER)
    quiet = base & (merg < 0.3)
    masks = {
        "featured": quiet & (feat > F_FEATURED),
        "smooth": quiet & (smooth > F_SMOOTH),
        "merger": merger,
    }
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for label, m in masks.items():
        idx = np.where(m)[0]
        rng.shuffle(idx)
        take = idx[:per_class]
        print(f"  {label:9s}: {int(m.sum()):5d} available -> taking {len(take)}")
        for i in take:
            rows.append({
                "id": f"gz{int(i):06d}",
                "ra": f"{ra[i]:.6f}", "dec": f"{dec[i]:.6f}",
                "label": label, "redshift": "",
                "votes": str(int(nc[i])),
                "f_smooth": f"{smooth[i]:.2f}", "f_featured": f"{feat[i]:.2f}",
                "f_merger": f"{merg[i]:.2f}",
            })
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--size", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    print("Selecting Galaxy Zoo: CANDELS visual classifications...")
    rows = select(args.per_class, args.seed)
    csv_path = config.DATA_DIR / "gz_labels.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} labelled positions -> {csv_path}")

    if args.fresh and config.CUTOUTS_DIR.exists():
        shutil.rmtree(config.CUTOUTS_DIR)
        config.CUTOUTS_DIR.mkdir(parents=True)

    print(f"Fetching real JWST cutouts from {data.CUTOUT_SERVICE} ...")
    recs = data.build_dataset_from_service(rows, size=args.size,
                                           max_workers=args.workers, source="galaxyzoo")
    by = {}
    for r in recs:
        by[r.label] = by.get(r.label, 0) + 1
    print(f"\nCached {len(recs)} real cutouts: {by}")


if __name__ == "__main__":
    main()
