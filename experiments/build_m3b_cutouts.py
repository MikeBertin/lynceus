"""Build the M3b cutouts from the cached raw cubes (local, no network).

M3b (RESEARCH.md Q3 verdict): make the representation point-source aware.
Two changes relative to the shipped atlas cutouts, both from raw:

* **noise-aware stretch** (``core.data.asinh_stretch_snr``) — the scale is the
  cutout's own sky RMS, not per-cutout percentiles, so empty sky stays dark
  instead of amplifying to colour static;
* **detection-anchored 64px crop** (``core.data.centre_anchor_crop``) — the
  central source fills ~3.5x more of the frame than in the 120px original.

Writes <id>.npy + <id>.png into data/atlas/cutouts_m3b/ and
data/lrd/cutouts_m3b/. The original cutouts (and the shipped demo built on
them) are untouched. Re-run freely after any stretch-parameter change.

    python -m experiments.build_m3b_cutouts
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from core import config, data

ATLAS_RAW = config.ATLAS_DIR / "raw"
LRD_RAW = config.DATA_DIR / "lrd" / "raw"
ATLAS_M3B = config.ATLAS_DIR / "cutouts_m3b"
LRD_M3B = config.DATA_DIR / "lrd" / "cutouts_m3b"


def build(raw_dir, out_dir, out_px: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(raw_dir.glob("*.npy"))
    for p in paths:
        cut = data.m3b_cutout(np.load(p), out_px=out_px)
        np.save(out_dir / p.name, cut)
        rgb = (np.clip(cut, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / f"{p.stem}.png")
    return len(paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-px", type=int, default=64)
    args = ap.parse_args()

    n_atlas = build(ATLAS_RAW, ATLAS_M3B, args.out_px)
    n_lrd = build(LRD_RAW, LRD_M3B, args.out_px)
    print(f"Built {n_atlas} atlas + {n_lrd} LRD m3b cutouts "
          f"({args.out_px}px, snr stretch) -> {ATLAS_M3B}, {LRD_M3B}")


if __name__ == "__main__":
    main()
