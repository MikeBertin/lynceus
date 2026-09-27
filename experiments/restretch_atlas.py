"""Re-fetch the *existing* atlas cutouts with the colour-preserving stretch.

The atlas .npy cache stores already-stretched images, and the old per-channel
stretch discarded colour irreversibly. To make the atlas colour-aware we must
re-pull the raw cubes from the DJA service and re-stretch with the new
``asinh_stretch(colour=True)``. This reads the *current* atlas_meta.csv so the
galaxy set (and ids) stay identical (only the pixels change), keeping the
embedding/UMAP comparable to before.

Overwrites each <id>.npy / <id>.png in place; safe to re-run after an interrupt
(it re-fetches everything). A tiny ``.colour`` marker file per id lets a resumed
run skip ones already converted.

    python -m experiments.restretch_atlas            # all
    python -m experiments.restretch_atlas --no-resume
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from core import config, data

META = config.ATLAS_DIR / "atlas_meta.csv"


def one(row, size, resume):
    rid = row["id"]
    npy = config.ATLAS_CUTOUTS / f"{rid}.npy"
    marker = config.ATLAS_CUTOUTS / f"{rid}.colour"
    if resume and marker.exists() and npy.exists():
        return "skip"
    cube = data.fetch_service_cube(float(row["ra"]), float(row["dec"]), size=size)
    if cube is None:
        return None
    st = data.asinh_stretch(np.nan_to_num(cube), colour=True).astype(np.float32)
    np.save(npy, st)
    rgb = (np.clip(st, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(config.ATLAS_CUTOUTS / f"{rid}.png")
    marker.write_text("v2")
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(META))
            if (config.ATLAS_CUTOUTS / f"{r['id']}.npy").exists()]
    print(f"Re-stretching {len(rows)} atlas cutouts (colour=True) -> {config.ATLAS_CUTOUTS}")

    ok = skip = fail = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, r, args.size, not args.no_resume): r for r in rows}
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            ok += res == "ok"; skip += res == "skip"; fail += res is None
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}  ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"\nDone. converted={ok} skipped={skip} failed={fail}")
    if fail:
        print(f"  ({fail} had no coverage / service errors; re-run to retry them)")


if __name__ == "__main__":
    main()
