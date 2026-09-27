"""Fetch and cache the RAW service cubes for the atlas + LRD samples (M3b).

The cutout cache has only ever stored *stretched* images; the raw fluxes were
thrown away, which is why every stretch change so far (colour-aware, and now
the M3b noise-aware stretch) has meant another pull from the DJA service. This
script re-fetches once more and keeps the raw (3, H, W) cubes, so any future
representation experiment is a purely local rebuild.

Existing stretched cutouts (and everything built from them) are untouched.

    python -m experiments.fetch_raw               # atlas + LRDs, resumable
    python -m experiments.fetch_raw --workers 4   # be gentler on the service
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from core import config, data

ATLAS_RAW = config.ATLAS_DIR / "raw"
LRD_RAW = config.DATA_DIR / "lrd" / "raw"


def jobs() -> list[tuple[Path, float, float]]:
    out = []
    for r in csv.DictReader(open(config.ATLAS_DIR / "atlas_meta.csv")):
        out.append((ATLAS_RAW / f"{r['id']}.npy", float(r["ra"]), float(r["dec"])))
    for r in csv.DictReader(open(config.DATA_DIR / "lrd" / "lrd_meta.csv")):
        out.append((LRD_RAW / f"{r['id']}.npy", float(r["ra"]), float(r["dec"])))
    return out


def fetch_one(path: Path, ra: float, dec: float, size: float, resume: bool):
    if resume and path.exists():
        return "skip"
    cube = data.fetch_service_cube(ra, dec, size=size)
    if cube is None:
        return None
    np.save(path, cube.astype(np.float32))
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    ATLAS_RAW.mkdir(parents=True, exist_ok=True)
    LRD_RAW.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    print(f"Fetching {len(todo)} raw cubes -> {ATLAS_RAW} + {LRD_RAW}")

    ok = skip = fail = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, p, ra, dec, args.size, not args.no_resume): p
                for p, ra, dec in todo}
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            ok += res == "ok"; skip += res == "skip"; fail += res is None
            if done % 200 == 0:
                print(f"  {done}/{len(todo)}  ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"\nDone. fetched={ok} skipped={skip} failed={fail}")
    if fail:
        print(f"  ({fail} had no coverage / service errors; re-run to retry them)")


if __name__ == "__main__":
    main()
