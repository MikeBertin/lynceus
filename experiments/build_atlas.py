"""Build the M2 atlas web assets: embed -> UMAP -> sprite sheet + atlas.json.

    python -m experiments.build_atlas
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import timm
import torch
from PIL import Image

from core import config
from core.embed import embed_paths, umap_2d, normalise_coords

TILE = 56  # sprite thumbnail size (px)


def load_encoder():
    ckpt = torch.load(config.MODELS_DIR / "ssl_encoder.pt", map_location="cpu")
    enc = timm.create_model(ckpt["backbone"], pretrained=False, num_classes=0)
    enc.load_state_dict(ckpt["state_dict"])
    return enc.eval()


def main() -> None:
    meta = config.ATLAS_DIR / "atlas_meta.csv"
    rows = [r for r in csv.DictReader(open(meta))
            if (config.ATLAS_CUTOUTS / f"{r['id']}.npy").exists()]
    paths = [config.ATLAS_CUTOUTS / f"{r['id']}.npy" for r in rows]
    cache = config.ATLAS_DIR / "embeddings.npy"
    if cache.exists() and np.load(cache).shape[0] == len(rows):
        feats = np.load(cache)
        print(f"Loaded cached embeddings {feats.shape}; running UMAP...")
    else:
        print(f"Embedding {len(rows)} cutouts...")
        feats = embed_paths(load_encoder(), paths, device=config.get_device())
        np.save(cache, feats)
        print(f"  features {feats.shape}; running UMAP...")
    xy = normalise_coords(umap_2d(feats))

    # sprite sheet: row-major grid of TILE-px thumbnails, sprite index = row order
    n = len(rows)
    cols = math.ceil(math.sqrt(n))
    sheet = Image.new("RGB", (cols * TILE, math.ceil(n / cols) * TILE), (4, 5, 10))
    for i, r in enumerate(rows):
        thumb = Image.open(config.ATLAS_CUTOUTS / f"{r['id']}.png").resize(
            (TILE, TILE), Image.BILINEAR)
        sheet.paste(thumb, ((i % cols) * TILE, (i // cols) * TILE))
    config.WEB_ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    # JPEG: galaxy thumbnails are photographic, so this is a fraction of PNG size
    # (faster first load) with no visible loss at thumbnail scale.
    sheet.save(config.WEB_ATLAS_DIR / "sprites.jpg", quality=90, optimize=True)
    (config.WEB_ATLAS_DIR / "sprites.png").unlink(missing_ok=True)

    points = []
    for i, r in enumerate(rows):
        points.append({
            "i": i,
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "d": r["dominant"],
            "s": round(float(r["smooth"]), 2),
            "f": round(float(r["featured"]), 2),
            "m": round(float(r["merger"]), 2),
            "c": round(float(r["clumpy"]), 2),
        })
    atlas = {"tile": TILE, "cols": cols, "count": n, "points": points}
    (config.WEB_ATLAS_DIR / "atlas.json").write_text(json.dumps(atlas, separators=(",", ":")))

    mix = {}
    for r in rows:
        mix[r["dominant"]] = mix.get(r["dominant"], 0) + 1
    print(f"Wrote sprites.jpg ({sheet.size}) + atlas.json ({n} points). mix={mix}")


if __name__ == "__main__":
    main()
