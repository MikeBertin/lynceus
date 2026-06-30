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

from core import config, anomaly
from core.embed import embed_paths, umap_2d, normalise_coords

TILE = 48  # sprite thumbnail size (px); kept modest so the ~10k-galaxy
           # sprites.jpg stays a lean first-load asset (~5 MB) for GitHub Pages
LRD_DIR = config.DATA_DIR / "lrd"


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
    n = len(rows)

    # --- M3: anomaly score + known Little Red Dots overlay --------------------
    araw = anomaly.knn_anomaly(feats, feats, k=20, exclude_self=True)
    a_lo, a_hi = np.percentile(araw, [5, 99])
    unit = lambda v: np.clip((v - a_lo) / (a_hi - a_lo + 1e-9), 0, 1)
    a_atlas = unit(araw)

    has_lrd = (LRD_DIR / "lrd_emb.npy").exists() and (LRD_DIR / "lrd_meta.csv").exists()
    if has_lrd:
        lrd_emb = np.load(LRD_DIR / "lrd_emb.npy")
        lrd_rows = list(csv.DictReader(open(LRD_DIR / "lrd_meta.csv")))
        xy_all = normalise_coords(umap_2d(np.vstack([feats, lrd_emb])))
        xy, xy_lrd = xy_all[:n], xy_all[n:]
        lraw = anomaly.knn_anomaly(lrd_emb, feats, k=20)
        a_lrd = unit(lraw)
        ldist = anomaly.nearest_distance(feats, lrd_emb)   # atlas -> nearest LRD
        d_lo, d_hi = np.percentile(ldist, [2, 50])
        lc = np.clip(1 - (ldist - d_lo) / (d_hi - d_lo + 1e-9), 0, 1)
    else:
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

    def field(ra):
        ra = float(ra)
        if 33 <= ra <= 36: return "UDS"
        if 52 <= ra <= 54: return "GOODS-S"
        if 149 <= ra <= 151: return "COSMOS"
        return "?"

    points = []
    for i, r in enumerate(rows):
        p = {
            "i": i,
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "d": r["dominant"],
            "r": field(r["ra"]),
            "s": round(float(r["smooth"]), 2),
            "f": round(float(r["featured"]), 2),
            "m": round(float(r["merger"]), 2),
            "c": round(float(r["clumpy"]), 2),
            "a": round(float(a_atlas[i]), 3),
        }
        if has_lrd:
            p["lc"] = round(float(lc[i]), 2)
        points.append(p)
    atlas = {"tile": TILE, "cols": cols, "count": n, "points": points}
    (config.WEB_ATLAS_DIR / "atlas.json").write_text(json.dumps(atlas, separators=(",", ":")))

    if has_lrd:
        from core import stats
        p90 = np.percentile(araw, 90)
        frac = float((lraw > p90).mean())
        lrd_pct = np.array([(araw < v).mean() for v in lraw])
        med_pct = float(np.median(lrd_pct))   # typical LRD's anomaly percentile
        # error bars: bootstrap the 216 LRDs; permutation null = random subsets
        enr_ci = stats.bootstrap_ci(lraw, lambda s: float((s > p90).mean()) / 0.10)
        med_ci = stats.bootstrap_ci(lrd_pct, np.median)
        perm = stats.permutation_p(frac, araw, lambda s: float((s > p90).mean()),
                                   n_draw=len(lraw), alternative="greater")
        # LRD sprite sheet so the red dots show their real cutout on zoom
        nl = len(lrd_rows)
        lcols = math.ceil(math.sqrt(nl))
        lsheet = Image.new("RGB", (lcols * TILE, math.ceil(nl / lcols) * TILE), (4, 5, 10))
        for j, r in enumerate(lrd_rows):
            png = LRD_DIR / "cutouts" / f"{r['id']}.png"
            if png.exists():
                lsheet.paste(Image.open(png).resize((TILE, TILE), Image.BILINEAR),
                             ((j % lcols) * TILE, (j // lcols) * TILE))
        lsheet.save(config.WEB_ATLAS_DIR / "lrd_sprites.jpg", quality=90, optimize=True)
        lpts = [{
            "i": j,
            "x": round(float(xy_lrd[j, 0]), 4), "y": round(float(xy_lrd[j, 1]), 4),
            "a": round(float(a_lrd[j]), 3),
            "z": lrd_rows[j].get("z", ""), "field": field(lrd_rows[j]["ra"]),
        } for j in range(nl)]
        lrds = {"n": nl, "tile": TILE, "cols": lcols, "enrichment": round(enr_ci["point"], 1),
                "enrichment_ci": [round(enr_ci["lo"], 2), round(enr_ci["hi"], 2)],
                "p_value": perm["p"],
                "frac_above_p90": round(frac, 3), "median_pct": round(med_pct, 3),
                "median_pct_ci": [round(med_ci["lo"], 3), round(med_ci["hi"], 3)],
                "points": lpts}
        (config.WEB_ATLAS_DIR / "lrds.json").write_text(json.dumps(lrds, separators=(",", ":")))
        print(f"LRD overlay: {len(lpts)} known LRDs · {lrds['enrichment']}x enriched in top-10% "
              f"anomalies · median LRD sits at the {med_pct*100:.0f}th percentile")

    mix = {}
    for r in rows:
        mix[r["dominant"]] = mix.get(r["dominant"], 0) + 1
    print(f"Wrote sprites.jpg ({sheet.size}) + atlas.json ({n} points). mix={mix}")


if __name__ == "__main__":
    main()
