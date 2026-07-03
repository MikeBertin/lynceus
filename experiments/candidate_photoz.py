"""Photo-z sanity check for the vetted LRD candidates (M3b next-step 3).

Two parts, both using the shipped M4 net (``models/photoz.pt``) unchanged:

**A. Validation on the known LRDs, catalogue photometry.** The Kokorev+24
table carries fluxes + errors for all nine PHOTOZ_BANDS (same µJy convention
as the CEERS training catalogue). Run the net on the 216 in-field LRDs and
compare with the catalogue's own z_phot. This measures whether an
EAZY-distilled emulator trained on ordinary CEERS galaxies can read the
weird V-shaped LRD SEDs at all — worth knowing on its own.

**B. The 10 veto survivors, our own aperture photometry.** They are not in
any photometric catalogue we hold, so: fetch 9-band service cutouts, do
r=10 px aperture photometry (sky-subtracted; error = sky RMS x sqrt(Npix)),
and calibrate each band's flux scale empirically — the median
catalogue-flux / aperture-flux ratio over the known LRDs (which corrects
units *and* point-source aperture loss in one factor). Then featurize and
read off z_peak, P(z>4), P(z<2).

This is a plausibility check, not photometry-grade measurement — stated in
the output. Cutouts are cached under data/*/bands9/; re-runs are offline.

    python -m experiments.candidate_photoz

Outputs: research/candidates_photoz.csv
"""
from __future__ import annotations

import csv

import numpy as np
import torch

from core import config, data
from core.photoz import (PHOTOZ_BANDS, BAD, PhotoZNet, featurize,
                         pdf_from_logits, point_estimates, Z_CENTRES)

OUT = config.ROOT / "research"
LRD_B9 = config.DATA_DIR / "lrd" / "bands9"
ATL_B9 = config.ATLAS_DIR / "bands9"
APER_PX = 10                 # aperture radius (px, ~0.25" — point-source scale)
N_CALIB = 48                 # known LRDs used for the per-band flux calibration
SN_CAL = 3.0                 # calibration uses bands detected at this S/N


def load_net():
    ck = torch.load(config.MODELS_DIR / "photoz.pt", map_location="cpu",
                    weights_only=False)     # our own checkpoint (numpy mu/sd)
    net = PhotoZNet(in_dim=ck["feat_dim"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck["mu"], ck["sd"]


def run_net(net, mu, sd, flux, err):
    X = featurize(flux, err)
    with torch.no_grad():
        pdf = pdf_from_logits(net(torch.from_numpy((X - mu) / sd).float())).numpy()
    pe = point_estimates(pdf)
    p_hi = pdf[:, Z_CENTRES > 4].sum(1)
    p_lo = pdf[:, Z_CENTRES < 2].sum(1)
    return pdf, pe, p_hi, p_lo


# ---------------------------------------------------------------------------
# A. validation: net vs Kokorev z_phot, catalogue photometry
# ---------------------------------------------------------------------------
def kokorev_infield():
    from astropy.io import fits
    t = fits.open(config.DATA_DIR / "lrd_kokorev.fits")[1].data
    ra = np.asarray(t["ra"], float)
    infield = np.zeros(len(t), bool)
    for a, b in ((33, 36), (52, 54), (149, 151)):
        infield |= (ra >= a) & (ra <= b)
    return t[infield]


def validate(net, mu, sd) -> dict:
    t = kokorev_infield()
    flux = np.stack([np.nan_to_num(np.asarray(t[f"{b}_flux"], float), nan=BAD - 1)
                     for b in PHOTOZ_BANDS], axis=1)
    err = np.stack([np.nan_to_num(np.asarray(t[f"{b}_fluxerr"], float), nan=-1.0)
                    for b in PHOTOZ_BANDS], axis=1)
    z_cat = np.asarray(t["z_phot"], float)
    ok = np.isfinite(z_cat)
    _, pe, p_hi, _ = run_net(net, mu, sd, flux[ok], err[ok])
    dz = (pe["z_peak"] - z_cat[ok]) / (1 + z_cat[ok])
    nmad = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
    out = {"n": int(ok.sum()),
           "sigma_nmad_vs_kokorev": round(nmad, 3),
           "outlier_frac_0p15": round(float((np.abs(dz) > 0.15).mean()), 3),
           "frac_p_gt4_above_half": round(float((p_hi > 0.5).mean()), 3),
           "median_z_kokorev": round(float(np.median(z_cat[ok])), 2),
           "median_z_net": round(float(np.median(pe["z_peak"])), 2)}
    print("A. validation — M4 net on Kokorev catalogue photometry "
          f"(n={out['n']} in-field LRDs):")
    print(f"   sigma_NMAD vs catalogue z_phot = {out['sigma_nmad_vs_kokorev']}"
          f"   outliers(|dz|>0.15(1+z)) = {out['outlier_frac_0p15']*100:.0f}%")
    print(f"   median z: net {out['median_z_net']} vs catalogue {out['median_z_kokorev']}"
          f"   P(z>4)>0.5 for {out['frac_p_gt4_above_half']*100:.0f}%\n")
    return out


# ---------------------------------------------------------------------------
# B. candidates: service-cutout aperture photometry, LRD-calibrated
# ---------------------------------------------------------------------------
def fetch_bands9(ra: float, dec: float, cache) -> dict | None:
    if cache.exists():
        z = np.load(cache)
        return {b: z[b] for b in z.files}
    got = data.fetch_service_bands(ra, dec, list(PHOTOZ_BANDS), size=3.0)
    if not got:
        return None
    np.savez_compressed(cache, **got)
    return got


def aper_phot(planes: dict) -> tuple[np.ndarray, np.ndarray]:
    """(flux, err) over PHOTOZ_BANDS from 2-D planes; absent band -> no-data."""
    anchor_bands = [b for b in ("f444w", "f356w", "f277w") if b in planes]
    cube = np.stack([np.nan_to_num(planes[b]) for b in anchor_bands])
    cy, cx = data.centre_anchor_crop(cube, out_px=32)
    flux = np.full(len(PHOTOZ_BANDS), BAD - 1.0)
    err = np.full(len(PHOTOZ_BANDS), -1.0)
    for i, b in enumerate(PHOTOZ_BANDS):
        if b not in planes:
            continue
        img = np.nan_to_num(planes[b]).astype(float)
        H, W = img.shape
        sky = np.median(img)
        sig = 1.4826 * np.median(np.abs(img - sky)) + 1e-12
        yy, xx = np.mgrid[0:H, 0:W]
        ap = (yy - cy) ** 2 + (xx - cx) ** 2 <= APER_PX ** 2
        flux[i] = (img - sky)[ap].sum()
        err[i] = sig * np.sqrt(ap.sum())
    return flux, err


def calibration_factors() -> np.ndarray:
    """Per-band median (catalogue flux / aperture flux) over known LRDs."""
    LRD_B9.mkdir(parents=True, exist_ok=True)
    t = kokorev_infield()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(t))[:N_CALIB]
    ratios = [[] for _ in PHOTOZ_BANDS]
    used = 0
    for i in idx:
        rid = f"lrd_{int(t['id'][i])}"
        planes = fetch_bands9(float(t["ra"][i]), float(t["dec"][i]),
                              LRD_B9 / f"{rid}.npz")
        if planes is None:
            continue
        used += 1
        flux, err = aper_phot(planes)
        for j, b in enumerate(PHOTOZ_BANDS):
            cat_f = float(np.nan_to_num(t[f"{b}_flux"][i], nan=BAD - 1))
            cat_e = float(np.nan_to_num(t[f"{b}_fluxerr"][i], nan=-1))
            if (flux[j] > BAD and err[j] > 0 and flux[j] / err[j] > SN_CAL
                    and cat_f > BAD and cat_e > 0 and cat_f / cat_e > SN_CAL
                    and flux[j] > 0):
                ratios[j].append(cat_f / flux[j])
    fac = np.array([np.median(r) if len(r) >= 5 else np.nan for r in ratios])
    med = np.nanmedian(fac)
    fac = np.where(np.isfinite(fac), fac, med)     # sparse bands: global median
    print(f"B. calibration over {used} known LRDs "
          f"(bands with >=5 S/N>{SN_CAL:.0f} pairs get their own factor):")
    for b, f, r in zip(PHOTOZ_BANDS, fac, ratios):
        print(f"   {b}: x{f:.3f} (n={len(r)})")
    return fac


def main() -> None:
    net, mu, sd = load_net()
    val = validate(net, mu, sd)
    fac = calibration_factors()

    ATL_B9.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(OUT / "candidates_crossmatch.csv"))]
    print(f"\nC. the {len(rows)} veto survivors "
          f"(aperture r={APER_PX}px, LRD-calibrated; plausibility check only):")
    out_rows = []
    F, E = [], []
    kept = []
    for r in rows:
        planes = fetch_bands9(float(r["ra"]), float(r["dec"]),
                              ATL_B9 / f"{r['id']}.npz")
        if planes is None:
            print(f"   #{r['rank']} {r['id']}: no service coverage, skipped")
            continue
        flux, err = aper_phot(planes)
        good = flux > BAD
        flux[good] *= fac[good]
        err[good] *= fac[good]
        F.append(flux); E.append(err); kept.append(r)
    _, pe, p_hi, p_lo = run_net(net, mu, sd, np.array(F), np.array(E))
    for k, r in enumerate(kept):
        rec = {**{key: r[key] for key in ("rank", "id", "ra", "dec", "field",
                                          "r44_20", "novelty")},
               "z_peak": round(float(pe["z_peak"][k]), 2),
               "z_mean": round(float(pe["z_mean"][k]), 2),
               "p_z_gt4": round(float(p_hi[k]), 3),
               "p_z_lt2": round(float(p_lo[k]), 3)}
        out_rows.append(rec)
        print(f"   #{int(r['rank']):>3} {r['id']} {r['field']:<8} "
              f"z_peak={rec['z_peak']:>5}  P(z>4)={rec['p_z_gt4']:.2f}  "
              f"P(z<2)={rec['p_z_lt2']:.2f}  [{r['novelty']}]")

    with open(OUT / "candidates_photoz.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nvalidation block: {val}")
    print(f"wrote {OUT / 'candidates_photoz.csv'}")


if __name__ == "__main__":
    main()
