"""Build the M4 dropout-hunter web assets.

Produces, into web/dropout/:
  * gallery filmstrips — for a curated set of real CEERS galaxies spanning low z
    to z>10, a row of per-filter grayscale cutouts on a *shared* flux scale, so a
    high-z galaxy visibly vanishes from the bluer bands (the Lyman dropout);
  * dropout.json — per-galaxy catalogue fluxes, the network's redshift PDF, point
    estimate, true/template redshift; plus a Lyman-break colour-colour scatter of
    thousands of catalogue galaxies for the diagram.

    python -m experiments.build_dropout_assets

The ONNX model + meta are written separately by experiments.export_photoz_onnx;
the precomputed PDFs here let the page render instantly and double-check the
in-browser ONNX run.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from PIL import Image

from core import config, data
from core.photoz import (PhotoZNet, featurize, pdf_from_logits, point_estimates,
                         PHOTOZ_BANDS, Z_CENTRES)

OUT = config.WEB_DIR / "dropout"
NPZ = config.DATA_DIR / "photoz" / "photoz.npz"
CKPT = config.MODELS_DIR / "photoz.pt"

# Filmstrip bands (NIRCam), blue -> red. These show the dropout sweep.
STRIP_BANDS = ["f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
TILE = 72


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = PhotoZNet(in_dim=ck["feat_dim"]); net.load_state_dict(ck["state_dict"])
    return net.eval(), ck["mu"], ck["sd"]


def model_pdf(net, mu, sd, flux, err):
    X = (featurize(flux, err) - mu) / sd
    with torch.no_grad():
        pdf = pdf_from_logits(net(torch.from_numpy(X))).numpy()
    return pdf


def shared_gray(bandmap: dict, ref=("f356w", "f444w")) -> dict:
    """asinh grayscale per band on a SHARED red-band scale (dropout stays dark)."""
    refpix = []
    for b in ref:
        if b in bandmap:
            a = bandmap[b].astype(np.float64)
            refpix.append(np.clip(a - np.median(a), 0, None))
    if not refpix:
        return {}
    hi = np.percentile(np.concatenate([r.ravel() for r in refpix]), 99.0) + 1e-8
    soft = 0.12
    out = {}
    for b, a in bandmap.items():
        a = a.astype(np.float64)
        s = np.clip(a - np.median(a), 0, None) / hi
        g = np.arcsinh(s / soft) / math.asinh(1.0 / soft)
        out[b] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return out


def select_gallery(z) -> list[int]:
    """Curate a spread of real galaxies: clear dropouts + mid + low z."""
    zph = z["z_phot"]; zsp = z["z_spec"]; flux = z["flux"]; err = z["err"]
    bands = list(PHOTOZ_BANDS)
    def sn(b):
        i = bands.index(b); f = flux[:, i]; e = err[:, i]
        return np.where(e > 0, f / np.where(e > 0, e, 1), 0.0)
    sn_red = np.maximum(sn("f356w"), sn("f444w"))
    sn_f200 = sn("f200w"); sn_blue = np.maximum(sn("f115w"), sn("f606w"))
    rng = np.random.default_rng(7)
    picks = []
    def take(mask, n):
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        for i in idx[:n]:
            picks.append(int(i))
    # high-z dropouts: bright in red, dark in blue
    take((zph >= 10) & (zph < 15) & (sn_red > 8) & (sn_blue < 2.5), 4)
    take((zph >= 8) & (zph < 10) & (sn_red > 8) & (sn_blue < 3), 3)
    # mid-z: break around f115w/f150w
    take((zph >= 5) & (zph < 8) & (sn_f200 > 8), 3)
    # spectroscopically confirmed, mid/high
    take((zsp >= 4) & (zsp < 9) & (sn_red > 8), 2)
    # low-z anchors visible in all bands
    take((zph >= 1) & (zph < 3) & (sn_blue > 8) & (sn_red > 8), 3)
    # dedup, keep order
    seen = set(); uniq = []
    for i in picks:
        if i not in seen:
            seen.add(i); uniq.append(i)
    return uniq


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    net, mu, sd = load_model()
    sel = select_gallery(z)
    print(f"Selected {len(sel)} gallery galaxies; fetching filmstrips...")

    gal = []
    strips = []   # (id -> grayscale tiles row)
    for n, i in enumerate(sel):
        ra, dec = float(z["ra"][i]), float(z["dec"][i])
        bm = data.fetch_service_bands(ra, dec, STRIP_BANDS, size=2.2)
        if not bm:
            print(f"  [{n}] id={z['id'][i]} no cutout — skip"); continue
        gray = shared_gray(bm)
        # build a filmstrip row of TILE-px tiles, one per STRIP_BANDS (blank if missing)
        row = Image.new("RGB", (len(STRIP_BANDS) * TILE, TILE), (6, 7, 12))
        present = []
        for k, b in enumerate(STRIP_BANDS):
            if b in gray:
                t = Image.fromarray(gray[b]).resize((TILE, TILE), Image.BILINEAR).convert("RGB")
                row.paste(t, (k * TILE, 0)); present.append(b)
        strips.append(row)

        flux = z["flux"][i:i+1]; err = z["err"][i:i+1]
        pdf = model_pdf(net, mu, sd, flux, err)[0]
        pe = point_estimates(pdf[None])
        zsp = float(z["z_spec"][i])
        gal.append({
            "id": int(z["id"][i]),
            "zphot": round(float(z["z_phot"][i]), 2),
            "zspec": round(zsp, 2) if zsp > 0 else None,
            "zpeak": round(float(pe["z_peak"][0]), 2),
            "zstd": round(float(pe["z_std"][0]), 2),
            "flux": [round(float(x), 5) for x in flux[0]],
            "err": [round(float(x), 5) for x in err[0]],
            "present": present,
            "pdf": [round(float(p), 5) for p in pdf],
        })
        if (n + 1) % 5 == 0:
            print(f"  {n+1}/{len(sel)} fetched", flush=True)

    # sort gallery by predicted redshift for a tidy low->high sweep
    order = np.argsort([g["zpeak"] for g in gal])
    gal = [gal[k] for k in order]; strips = [strips[k] for k in order]
    for k, g in enumerate(gal):
        g["strip"] = k

    # stack filmstrips into one sprite sheet (one galaxy per row)
    if strips:
        W = max(s.width for s in strips); H = TILE * len(strips)
        sheet = Image.new("RGB", (W, H), (6, 7, 12))
        for k, s in enumerate(strips):
            sheet.paste(s, (0, k * TILE))
        sheet.save(OUT / "filmstrips.jpg", quality=90, optimize=True)

    # ---- Lyman-break colour-colour scatter -----------------------------------
    bands = list(PHOTOZ_BANDS)
    flux = z["flux"]; err = z["err"]
    def mag_colour(b1, b2):
        i1, i2 = bands.index(b1), bands.index(b2)
        f1, f2 = flux[:, i1], flux[:, i2]
        ok = (f1 > 0) & (f2 > 0) & (err[:, i1] > 0) & (err[:, i2] > 0)
        c = np.full(len(flux), np.nan)
        c[ok] = -2.5 * np.log10(np.clip(f1[ok], 1e-6, None) / np.clip(f2[ok], 1e-6, None))
        return c
    cx = mag_colour("f115w", "f150w")   # blue dropout colour
    cy = mag_colour("f200w", "f444w")   # red continuum colour
    zall = np.where(z["z_spec"] > 0, z["z_spec"], z["z_phot"])
    good = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(zall)
    idx = np.where(good)[0]
    rng = np.random.default_rng(3)
    if len(idx) > 2500:
        idx = rng.choice(idx, 2500, replace=False)
    cc = [{"x": round(float(np.clip(cx[i], -2, 4)), 2),
           "y": round(float(np.clip(cy[i], -2, 4)), 2),
           "z": round(float(zall[i]), 2)} for i in idx]

    metrics = json.loads((config.MODELS_DIR / "photoz_metrics.json").read_text())
    out = {
        "tile": TILE, "strip_bands": STRIP_BANDS,
        "z_centres": [round(float(c), 3) for c in Z_CENTRES],
        "gallery": gal,
        "colourcolour": {"xlabel": "F115W − F150W", "ylabel": "F200W − F444W", "points": cc},
        "metrics": metrics,
    }
    (OUT / "dropout.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nWrote {OUT/'dropout.json'} ({len(gal)} galaxies, {len(cc)} colour-colour pts) "
          f"+ filmstrips.jpg")


if __name__ == "__main__":
    main()
