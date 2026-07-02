"""Q3 (RESEARCH.md): a ranked candidate list for spectroscopic follow-up.

The payoff question: point the scores at the atlas and hand over targets. Two
rankings, deliberately different in bias (Q1 measured the anomaly score's:
it favours the bright, red end):

* **anomaly-ranked** — the most isolated galaxies in latent space
  (`knn_anomaly`, unsupervised: "weird in any direction");
* **LRD-like-ranked** — smallest cosine distance to any of the 216 known
  Kokorev+24 LRDs (`nearest_distance`, supervised by example: "weird in
  *their* direction").

Known LRDs are removed by positional cross-match (1.5"), so everything listed
is *not* in the published catalogue. Outputs, under research/ (committed —
the repo is private; this is the notebook's evidence):

  candidates.csv                 both rankings, with scores + GZ context
  sheet_anomaly.jpg              contact sheet, top 32 by anomaly
  sheet_lrdlike.jpg              contact sheet, top 32 by LRD-likeness

    python -m experiments.candidates
"""
from __future__ import annotations

import csv

import numpy as np
from PIL import Image, ImageDraw

from core import config, anomaly

K_ANOM = 20
MATCH_ARCSEC = 1.5          # positional match radius for "already in Kokorev"
N_SHEET = 32                # tiles per contact sheet (8 x 4)
N_CSV = 100                 # rows per ranking in the CSV
TILE = 120                  # native cutout size
PAD = 18                    # caption strip under each tile

OUT = config.ROOT / "research"


def load_atlas():
    rows = [r for r in csv.DictReader(open(config.ATLAS_DIR / "atlas_meta.csv"))
            if (config.ATLAS_CUTOUTS / f"{r['id']}.npy").exists()]
    feats = np.load(config.ATLAS_DIR / "embeddings.npy")
    assert len(rows) == len(feats)
    return rows, feats


def known_mask(rows) -> np.ndarray:
    """True where an atlas galaxy positionally matches a Kokorev LRD."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits
    cat = fits.open(config.DATA_DIR / "lrd_kokorev.fits")[1].data
    lrd = SkyCoord(np.asarray(cat["ra"], float), np.asarray(cat["dec"], float), unit="deg")
    atl = SkyCoord([float(r["ra"]) for r in rows], [float(r["dec"]) for r in rows], unit="deg")
    _, sep, _ = atl.match_to_catalog_sky(lrd)
    return sep.arcsec < MATCH_ARCSEC


def field(ra):
    ra = float(ra)
    if 33 <= ra <= 36: return "UDS"
    if 52 <= ra <= 54: return "GOODS-S"
    if 149 <= ra <= 151: return "COSMOS"
    return "?"


def contact_sheet(rows, order, scores, title, path, cols=8):
    n = min(N_SHEET, len(order))
    rws = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * TILE, rws * (TILE + PAD) + 26), (6, 7, 13))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill=(230, 233, 242))
    for j, i in enumerate(order[:n]):
        x, y = (j % cols) * TILE, 26 + (j // cols) * (TILE + PAD)
        png = config.ATLAS_CUTOUTS / f"{rows[i]['id']}.png"
        sheet.paste(Image.open(png).convert("RGB"), (x, y))
        d.text((x + 3, y + TILE + 2),
               f"#{j+1} {scores[i]:.3f} {field(rows[i]['ra'])[:3]}",
               fill=(154, 163, 184))
    sheet.save(path, quality=92)
    print(f"  wrote {path.name} ({n} tiles)")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, feats = load_atlas()
    lrd_emb = np.load(config.DATA_DIR / "lrd" / "lrd_emb.npy")

    araw = anomaly.knn_anomaly(feats, feats, k=K_ANOM, exclude_self=True)
    ldist = anomaly.nearest_distance(feats, lrd_emb)     # small = LRD-like

    known = known_mask(rows)
    print(f"{known.sum()} atlas galaxies positionally match Kokorev+24 (excluded).")
    if known.sum():
        pct = np.mean([(araw < araw[i]).mean() for i in np.flatnonzero(known)])
        print(f"  (their mean anomaly percentile is {pct*100:.0f}; NB at 0.3-1.4\" "
              f"separation these are mostly bright GZ galaxies *adjacent* to an "
              f"LRD, not the LRD itself — see RESEARCH.md Q3)")

    ok = ~known
    rank_anom = np.flatnonzero(ok)[np.argsort(-araw[ok])]
    rank_like = np.flatnonzero(ok)[np.argsort(ldist[ok])]

    both = set(rank_anom[:N_CSV]) & set(rank_like[:N_CSV])
    print(f"overlap of the two top-{N_CSV} lists: {len(both)} galaxies")

    with open(OUT / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ranking", "rank", "id", "ra", "dec", "field", "anomaly",
                    "lrd_dist", "dominant", "smooth", "featured", "merger", "in_both_top100"])
        for name, order, key in (("anomaly", rank_anom, lambda i: -araw[i]),
                                 ("lrdlike", rank_like, lambda i: ldist[i])):
            for j, i in enumerate(order[:N_CSV]):
                r = rows[i]
                w.writerow([name, j + 1, r["id"], r["ra"], r["dec"], field(r["ra"]),
                            round(float(araw[i]), 4), round(float(ldist[i]), 4),
                            r["dominant"], r["smooth"], r["featured"], r["merger"],
                            int(i in both)])
    print(f"  wrote candidates.csv (2 x {N_CSV} rows)")

    contact_sheet(rows, rank_anom, araw, "top anomalies (not in Kokorev+24) - rank, anomaly, field",
                  OUT / "sheet_anomaly.jpg")
    contact_sheet(rows, rank_like, ldist, "most LRD-like (not in Kokorev+24) - rank, cosine dist to nearest LRD, field",
                  OUT / "sheet_lrdlike.jpg")

    # --- diagnostic: what does "LRD-like" actually retrieve? ------------------
    # The known-LRD cutouts are tiny dots on near-empty fields; the per-cutout
    # percentile stretch amplifies empty-field sky noise to full-range colour
    # static, and the encoder embeds that static as a coherent texture. So
    # nearest-to-LRD retrieval returns noise. Quantify it via mean luminance
    # (static = high mean pixel value, real dark sky = low).
    from PIL import Image as PILImage
    rng = np.random.default_rng(0)
    def lum(idx):
        return np.array([np.asarray(PILImage.open(
            config.ATLAS_CUTOUTS / f"{rows[i]['id']}.png").convert("L"), float).mean()
            for i in idx])
    top100, rand100 = rank_like[:100], rng.choice(len(rows), 100, replace=False)
    samp = rng.choice(len(rows), 1000, replace=False)
    from scipy.stats import spearmanr
    rho = spearmanr(ldist[samp], lum(samp)).statistic
    print(f"diagnostic: median cutout luminance, top-100 LRD-like "
          f"{np.median(lum(top100)):.1f} vs random {np.median(lum(rand100)):.1f} "
          f"(spearman ldist~lum on 1k: rho={rho:.2f}) — 'LRD-like' retrieves "
          f"amplified empty-field noise, not red dots")


if __name__ == "__main__":
    main()
