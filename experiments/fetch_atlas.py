"""Fetch a large UNLABELLED JWST cutout set for the M2 embedding atlas.

Samples galaxies broadly across the Galaxy Zoo: CANDELS morphology distribution
(so the atlas can be *coloured* by real votes — the votes are never used to train
the encoder) and pulls real JWST/NIRCam cutouts from the DJA grizli service.

    python -m experiments.fetch_atlas --n 2500
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from core import config, data

CATALOG = config.DATA_DIR / "gz_candels.fits"
META = config.ATLAS_DIR / "atlas_meta.csv"
MIN_VOTES = 15
FIELDS = ((52.0, 54.0), (149.0, 151.0), (33.0, 36.0))  # GOODS-S, COSMOS, UDS


def select(n: int, seed: int) -> list[dict]:
    from astropy.io import fits
    if not CATALOG.exists():
        raise SystemExit(f"Missing {CATALOG} — see experiments/fetch_gz.py docstring.")
    d = fits.open(CATALOG)[1].data
    ra = np.asarray(d["RA"], float); dec = np.asarray(d["Dec"], float)
    nc = np.asarray(d["num_classifications"], float)
    f = lambda c: np.asarray(d[c], float)
    smooth = f("t00_smooth_or_featured_a0_smooth_frac")
    feat = f("t00_smooth_or_featured_a1_features_frac")
    merg = f("t16_merging_tidal_debris_a0_merging_frac")
    clumpy = f("t02_clumpy_appearance_a0_yes_frac")

    in_field = np.zeros(len(d), bool)
    for r0, r1 in FIELDS:
        in_field |= (ra >= r0) & (ra <= r1)
    base = in_field & np.isfinite(ra) & np.isfinite(smooth) & (nc >= MIN_VOTES)

    # Stratify by dominant class so featured/merger regions aren't swamped by the
    # ~16:1 smooth majority — the atlas is a visualisation, not a number count.
    dom_idx = np.argmax(np.vstack([smooth, feat, merg]), axis=0)
    names = ("smooth", "featured", "merger")
    rng = np.random.default_rng(seed)
    take = []
    for k in range(3):
        pool = np.where(base & (dom_idx == k))[0]
        rng.shuffle(pool)
        take.extend(pool[:n // 3].tolist())
    rng.shuffle(take)

    def dominant(i):
        return names[int(dom_idx[i])]

    return [{
        "id": f"atl{int(i):06d}", "ra": f"{ra[i]:.6f}", "dec": f"{dec[i]:.6f}",
        "smooth": f"{smooth[i]:.3f}", "featured": f"{feat[i]:.3f}",
        "merger": f"{merg[i]:.3f}", "clumpy": f"{clumpy[i]:.3f}",
        "dominant": dominant(i), "votes": str(int(nc[i])),
    } for i in take]


def fetch_one(row, size, resume):
    npy = config.ATLAS_CUTOUTS / f"{row['id']}.npy"
    if resume and npy.exists():
        return row["id"]
    cube = data.fetch_service_cube(float(row["ra"]), float(row["dec"]), size=size)
    if cube is None:
        return None
    stretched = data.asinh_stretch(np.nan_to_num(cube)).astype(np.float32)
    np.save(npy, stretched)
    rgb = (np.clip(stretched, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(config.ATLAS_CUTOUTS / f"{row['id']}.png")
    return row["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--size", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows = select(args.n, args.seed)
    print(f"Selected {len(rows)} galaxies; fetching cutouts -> {config.ATLAS_CUTOUTS}")
    cached, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, r, args.size, True): r for r in rows}
        for fut in as_completed(futs):
            done += 1
            if fut.result() is not None:
                cached.append(futs[fut])
            if done % 100 == 0:
                print(f"  {done}/{len(rows)} ({len(cached)} cached)", flush=True)

    with open(META, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(cached)
    by = {}
    for r in cached:
        by[r["dominant"]] = by.get(r["dominant"], 0) + 1
    print(f"\nCached {len(cached)} cutouts. dominant-class mix: {by}\nMeta: {META}")


if __name__ == "__main__":
    main()
