"""Build the cutout cache.

Default (synthetic): no downloads, fully reproducible::

    python -m experiments.fetch_data --synthetic 170

Real CEERS/JADES cutouts, once you have a label catalogue (CSV with
ra,dec,label[,redshift,id]) and per-band NIRCam mosaics in data/mosaics/::

    python -m experiments.fetch_data --real --catalog data/ceers_visual.csv \
        --mosaic F444W=data/mosaics/ceers_f444w.fits \
        --mosaic F356W=data/mosaics/ceers_f356w.fits \
        --mosaic F200W=data/mosaics/ceers_f200w.fits
"""
from __future__ import annotations

import argparse
import csv

from core import config, data


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill the Lynceus cutout cache.")
    ap.add_argument("--synthetic", type=int, default=170, metavar="N",
                    help="objects per class for the synthetic dataset (default 170)")
    ap.add_argument("--real", action="store_true",
                    help="extract real cutouts from mosaics instead")
    ap.add_argument("--catalog", help="CSV with ra,dec,label[,redshift,id] (real mode)")
    ap.add_argument("--mosaic", action="append", default=[], metavar="BAND=PATH",
                    help="repeatable, e.g. --mosaic F444W=data/mosaics/f444w.fits")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.real:
        if not args.catalog:
            ap.error("--real requires --catalog")
        with open(args.catalog, newline="") as fh:
            rows = list(csv.DictReader(fh))
        mosaics = dict(m.split("=", 1) for m in args.mosaic)
        recs = data.extract_real_cutouts(rows, mosaics)
        print(f"Extracted {len(recs)} real CEERS cutouts -> {config.CUTOUTS_DIR}")
    else:
        recs = data.generate_synthetic_dataset(args.synthetic, seed=args.seed)
        print(f"Generated {len(recs)} synthetic cutouts "
              f"({args.synthetic}/class) -> {config.CUTOUTS_DIR}")
    print(f"Manifest: {data.MANIFEST}")


if __name__ == "__main__":
    main()
