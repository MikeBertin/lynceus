"""Star veto for the M3b LRD-like candidate list (RESEARCH.md, M3b next-step 1).

The M3b retrieval returns point sources: compact red dots (candidate-shaped)
*and* stars — the same contaminant class photometric LRD searches fight. This
script separates them with three independent, fully-disclosed tests, each
computed from data we already have (plus one cheap Gaia query per candidate):

1. **PSF halo + diffraction spikes** (pixels, cached raw cubes). Stars bright
   enough to matter carry an extended PSF halo and the six-fold JWST spike
   pattern. Features: ``mu_halo`` (mean SNR-stretched intensity in a 5-26 px
   annulus — LRDs are point sources on dark sky, so theirs is ~0) and
   ``m6sig`` (the m=6 azimuthal Fourier mode + its first harmonic vs the
   median of the other modes).
2. **Colour** (pixels). LRDs are red by selection: F444W/F200W aperture-flux
   ratio well above 1. The blue-white spiky things are not.
3. **Gaia DR3 astrometry** (one TAP cone query per candidate, 1"). A
   significant parallax or proper motion is a kinematic star confirmation; in
   fields this deep a Gaia detection at all is strong evidence.

Thresholds are calibrated against the 216 known LRDs (they must survive their
own veto) and printed alongside the results. Residual risk, disclosed: faint
M/L/T dwarfs below Gaia's limit with no visible spikes can pass the veto —
same as every photometric LRD sample.

    python -m experiments.star_veto            # full run (Gaia queries ~2 min)
    python -m experiments.star_veto --no-gaia  # offline: pixels-only veto

Outputs: research/candidates_vetted.csv, research/sheet_vetted.jpg
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from PIL import Image, ImageDraw

from core import config, data

ATLAS_RAW = config.ATLAS_DIR / "raw"
LRD_RAW = config.DATA_DIR / "lrd" / "raw"
OUT = config.ROOT / "research"

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_RADIUS_ARCSEC = 1.0

# annulus for halo/spike features (px; cutouts are ~0.025"/px)
R_IN, R_OUT, NTHETA = 5, 26, 144
APERTURE_PX = 6                    # aperture-flux radius for colours

# veto thresholds (calibrated against the known LRDs; see calibration print)
THR_MU_HALO = 0.12                 # bright extended PSF halo -> star
THR_M6 = 6.0                       # 6-fold spikes...
THR_MU_FOR_M6 = 0.08               # ...only meaningful when the annulus is lit
THR_RED = 1.5                      # F444W/F200W below this is not LRD-red


def ring_features(raw: np.ndarray) -> dict:
    """Halo brightness + m=6 spike significance on the SNR-stretched intensity."""
    from scipy.ndimage import map_coordinates

    raw = np.nan_to_num(raw)
    inten = data.asinh_stretch_snr(raw).mean(0)
    cy, cx = data.centre_anchor_crop(raw, out_px=64)
    thetas = np.linspace(0, 2 * np.pi, NTHETA, endpoint=False)
    rs = np.arange(R_IN, R_OUT)
    T, R = np.meshgrid(thetas, rs)
    ys, xs = cy + R * np.sin(T), cx + R * np.cos(T)
    prof = map_coordinates(inten, [ys.ravel(), xs.ravel()], order=1,
                           mode="constant").reshape(len(rs), NTHETA).mean(0)
    f = np.abs(np.fft.rfft(prof - prof.mean()))
    others = np.r_[f[3:6], f[7:12], f[13:18], f[19:30]]
    m6sig = float((f[6] + 0.5 * f[12]) / (np.median(others) + 1e-9))
    return {"mu_halo": float(prof.mean()), "m6sig": m6sig, "cy": cy, "cx": cx}


def aperture_colours(raw: np.ndarray, cy: int, cx: int) -> dict:
    """Sky-subtracted aperture fluxes per band; bands are (F444W, F356W, F200W)."""
    raw = np.nan_to_num(raw).astype(np.float32)
    C, H, W = raw.shape
    skies = np.median(raw.reshape(C, -1), axis=1)
    yy, xx = np.mgrid[0:H, 0:W]
    ap = (yy - cy) ** 2 + (xx - cx) ** 2 <= APERTURE_PX ** 2
    f444, f356, f200 = [float((raw[c] - skies[c])[ap].sum()) for c in range(3)]
    safe = lambda a, b: a / b if b > 1e-6 else np.inf if a > 0 else np.nan
    return {"f444": f444, "f356": f356, "f200": f200,
            "r44_20": safe(f444, f200), "r44_35": safe(f444, f356)}


def gaia_match(ra: float, dec: float, retries: int = 3) -> dict:
    """Nearest Gaia DR3 source within GAIA_RADIUS_ARCSEC, or empty fields."""
    import requests

    rad_deg = GAIA_RADIUS_ARCSEC / 3600.0
    adql = (
        "SELECT TOP 1 source_id, phot_g_mean_mag, parallax_over_error, "
        "pmra, pmra_error, pmdec, pmdec_error, "
        f"DISTANCE(POINT('ICRS',ra,dec), POINT('ICRS',{ra},{dec}))*3600 AS sep "
        "FROM gaiadr3.gaia_source WHERE CONTAINS(POINT('ICRS',ra,dec), "
        f"CIRCLE('ICRS',{ra},{dec},{rad_deg}))=1 ORDER BY sep")
    empty = {"gaia_sep": "", "gaia_g": "", "gaia_plx_sig": "", "gaia_pm_sig": ""}
    for attempt in range(retries):
        try:
            r = requests.get(GAIA_TAP, params=dict(
                REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql), timeout=30)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            if len(lines) < 2:
                return empty
            row = dict(zip(lines[0].split(","), lines[1].split(",")))
            g = lambda k: float(row[k]) if row.get(k) not in ("", None) else np.nan
            pm_sig = float(np.hypot(g("pmra"), g("pmdec"))
                           / max(np.hypot(g("pmra_error"), g("pmdec_error")), 1e-9))
            return {"gaia_sep": round(g("sep"), 2), "gaia_g": round(g("phot_g_mean_mag"), 2),
                    "gaia_plx_sig": round(g("parallax_over_error"), 1),
                    "gaia_pm_sig": round(pm_sig, 1)}
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return {**empty, "gaia_sep": "query_failed"}


def verdict(feat: dict) -> str:
    gaia_star = False
    if isinstance(feat.get("gaia_sep"), float):
        plx = feat.get("gaia_plx_sig") or 0
        pm = feat.get("gaia_pm_sig") or 0
        gaia_star = (not np.isnan(plx) and plx > 3) or (not np.isnan(pm) and pm > 3)
    if gaia_star:
        return "star_gaia"
    if feat["mu_halo"] > THR_MU_HALO or (feat["m6sig"] > THR_M6
                                         and feat["mu_halo"] > THR_MU_FOR_M6):
        return "star_psf"
    if isinstance(feat.get("gaia_sep"), float):
        return "star_gaia_weak"      # in Gaia but no significant astrometry
    if not np.isfinite(feat["r44_20"]) or feat["r44_20"] < THR_RED:
        return "not_red"
    return "candidate"


def contact_sheet(entries, path, tile=120, cols=8):
    n = len(entries)
    rws = max(1, int(np.ceil(n / cols)))
    sheet = Image.new("RGB", (cols * tile, rws * (tile + 18) + 26), (6, 7, 13))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), "veto survivors - rank, lrd_dist, F444W/F200W, field", fill=(230, 233, 242))
    m3b = config.ATLAS_DIR / "cutouts_m3b"
    for j, e in enumerate(entries):
        x, y = (j % cols) * tile, 26 + (j // cols) * (tile + 18)
        png = m3b / f"{e['id']}.png"
        sheet.paste(Image.open(png).convert("RGB").resize((tile, tile), Image.NEAREST), (x, y))
        d.text((x + 3, y + tile + 2),
               f"#{e['rank']} {float(e['lrd_dist']):.3f} {e['r44_20']:.1f} {e['field'][:3]}",
               fill=(154, 163, 184))
    sheet.save(path, quality=92)
    print(f"  wrote {path.name} ({n} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gaia", action="store_true", help="skip the Gaia queries")
    args = ap.parse_args()

    # --- calibration: the 216 known LRDs must survive their own veto ---------
    lrd_feats = []
    for p in sorted(LRD_RAW.glob("*.npy")):
        raw = np.load(p)
        f = ring_features(raw)
        f.update(aperture_colours(raw, f["cy"], f["cx"]))
        lrd_feats.append(f)
    mu = np.array([f["mu_halo"] for f in lrd_feats])
    red = np.array([f["r44_20"] for f in lrd_feats])
    red = red[np.isfinite(red)]
    print(f"calibration (216 known LRDs): mu_halo p50/p95/max = "
          f"{np.percentile(mu, 50):.3f}/{np.percentile(mu, 95):.3f}/{mu.max():.3f} "
          f"(threshold {THR_MU_HALO})")
    print(f"  F444W/F200W p5/p50 = {np.percentile(red, 5):.1f}/{np.percentile(red, 50):.1f} "
          f"(red threshold {THR_RED})")
    lrd_pass = sum(1 for f in lrd_feats
                   if f["mu_halo"] <= THR_MU_HALO
                   and not (f["m6sig"] > THR_M6 and f["mu_halo"] > THR_MU_FOR_M6)
                   and np.isfinite(f["r44_20"]) and f["r44_20"] >= THR_RED)
    print(f"  known LRDs surviving the pixel veto: {lrd_pass}/216 "
          f"({lrd_pass/216*100:.0f}% — the veto's completeness cost)\n")

    # --- the candidates -------------------------------------------------------
    rows = [r for r in csv.DictReader(open(OUT / "candidates_m3b.csv"))
            if r["ranking"] == "lrdlike"]
    print(f"vetting {len(rows)} LRD-like candidates...")
    out_rows = []
    for i, r in enumerate(rows):
        raw = np.load(ATLAS_RAW / f"{r['id']}.npy")
        feat = ring_features(raw)
        feat.update(aperture_colours(raw, feat["cy"], feat["cx"]))
        if not args.no_gaia:
            feat.update(gaia_match(float(r["ra"]), float(r["dec"])))
            time.sleep(0.3)          # be gentle with the ESA TAP
        feat["verdict"] = verdict(feat)
        out_rows.append({**r, **{k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in feat.items() if k not in ("cy", "cx")}})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    fields = list(out_rows[0].keys())
    with open(OUT / "candidates_vetted.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)

    counts = {}
    for r in out_rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"\nverdicts over the top-{len(rows)}: {counts}")
    survivors = [r for r in out_rows if r["verdict"] == "candidate"]
    print(f"survivors: {len(survivors)}")
    contact_sheet(survivors[:32], OUT / "sheet_vetted.jpg")
    print(f"  wrote candidates_vetted.csv ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
