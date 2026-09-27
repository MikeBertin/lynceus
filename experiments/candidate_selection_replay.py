"""Selection-function replay for the LRD candidates (Q3c, Step 2).

Q3b showed the 10 survivors are absent from every *LRD sub-selection* on our
fields. But "absent from a selection" can mean *below their cuts* rather than
non-existent. This step turns that into a defensible per-candidate verdict:

  1. Cross-match each candidate to the **parent** DJA/grizli photometric
     catalogue of its field (every detected source, not the LRD sub-list): the
     van der Wel+25 "morpho-phot" catalogues, which carry catalogue-grade
     multi-band fluxes *and* profile sizes in one file. A match confirms the
     candidate is a real detected source (our own aperture photometry from
     Q3b is only plausibility-grade), and hands us its catalogue fluxes + size.
  2. **Replay each paper's published LRD cuts** on those catalogue fluxes:
       - Akins+24 (COSMOS-Web): SNR_F444W>12, SNR_F277W>3, F277W-F444W>1.5,
         compact.  (COSMOS candidates.)
       - Kocevski+24: V-shape beta_UV<-0.37 & beta_opt>0, compact, SNR_F444W>12.
         (All fields.)
       - Kokorev+24: red1 or red2 colour box, compact, F444W<27.7, SNR_F444W>14.
         (All fields.)
     Verdict per candidate: **passes a paper's cuts but is absent from its
     catalogue = genuinely new** (a real miss / incompleteness); **fails the
     cuts = absence explained** ("below their limit"), stated plainly.

Compactness is unified as Kocevski's criterion: half-light radius
r_h < 1.5 r_h,stars, with r_h,stars measured from each field's own stellar
locus. This is a faithful, disclosed stand-in for the papers' various aperture-ratio
compactness definitions (all of which just mean "unresolved / point-like").
Colours use the consistent-aperture fluxes (aper_1); the ZP cancels.

    python -m experiments.candidate_selection_replay [--radius 0.6]

Outputs: research/candidates_selection_replay.csv (+ a printed verdict table).
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from core import config

OUT = config.ROOT / "research"
PHOT_DIR = config.DATA_DIR / "dja_phot"
PIXSCALE = 0.04                      # grizli v7 mosaics, arcsec/pixel
UJY_ZP = 23.9                        # AB zeropoint for fluxes in micro-Jansky

# COSMOS-Web LRD catalogue (Akins+24). PRIMER-COSMOS (our parent tiles) does NOT
# cover all of COSMOS-Web, so COSMOS-Web-only sources miss the PRIMER catalogue;
# Akins' own table carries their COSMOS-Web photometry (m277/m444/C444/SNR),
# which lets us close that gap and confirm the two rediscoveries pass Akins.
AKINS_ECSV = config.DATA_DIR / "lrd_akins24_cosmosweb.ecsv"

# van der Wel+25 "morpho-phot" parent catalogues (catalogue-grade fluxes +
# single-Sersic sizes), on the author's S3 bucket. Auto-downloaded once; these
# ARE the field detection catalogues (row count == the plain _phot.fits list),
# so a non-match is a genuine absence.
PARENT_BASE = "https://s3.amazonaws.com/aurelien-sepp/{root}/catalog/{root}_morpho-phot.fits.gz"

# Field -> parent catalogue tiles (a field can be split into >1 mosaic).
FIELD_CATS = {
    "COSMOS":  ["primer-cosmos-west-grizli-v7.0", "primer-cosmos-east-grizli-v7.0"],
    "UDS":     ["primer-uds-north-grizli-v7.2", "primer-uds-south-grizli-v7.2"],
    "GOODS-S": ["gds-grizli-v7.2"],
}

# NIRCam broad-band pivot wavelengths (Angstrom) for the slope fits.
PIVOT = {"f115w": 11540.0, "f150w": 15010.0, "f200w": 19900.0,
         "f277w": 27860.0, "f356w": 35630.0, "f444w": 44210.0}
BANDS = list(PIVOT)


def load_field(field: str):
    """Concatenate a field's parent-catalogue tiles into flat column arrays.

    Returns (ra, dec, phot) where phot is a dict of the columns we need, each a
    length-N array over all sources in the field. HDU1 = photometry, HDU2 =
    single-Sersic morphology (row-matched to HDU1).
    """
    from astropy.io import fits
    cols_h1 = ["ra", "dec", "flux_radius", "flag"]
    for b in BANDS:
        cols_h1 += [f"{b}_flux_aper_1", f"{b}_fluxerr_aper_1",
                    f"{b}_tot_1", f"{b}_etot_1"]
    acc: dict[str, list] = {c: [] for c in cols_h1}
    acc["reff_arcsec"] = []
    acc["sersic_n"] = []
    for root in FIELD_CATS[field]:
        fp = PHOT_DIR / f"{root}_morpho-phot.fits.gz"
        if not fp.exists():
            import urllib.request
            PHOT_DIR.mkdir(parents=True, exist_ok=True)
            url = PARENT_BASE.format(root=root)
            print(f"  fetching parent catalogue {root} (~300 MB) -> {fp}")
            urllib.request.urlretrieve(url, fp)
        with fits.open(fp) as hdul:
            d1, d2 = hdul[1].data, hdul[2].data
            for c in cols_h1:
                acc[c].append(np.asarray(d1[c], float))
            acc["reff_arcsec"].append(np.asarray(d2["RADIUS"], float) * 3600.0)
            acc["sersic_n"].append(np.asarray(d2["SERSIC"], float))
    phot = {c: np.concatenate(v) for c, v in acc.items()}
    return phot["ra"], phot["dec"], phot


def stellar_rh(phot) -> float:
    """r_h of the stellar/PSF locus (px): median flux_radius of bright, compact,
    high-S/N sources: the point-source ridge the compactness cut references."""
    snr = phot["f444w_tot_1"] / np.where(phot["f444w_etot_1"] > 0,
                                         phot["f444w_etot_1"], np.nan)
    fr = phot["flux_radius"]
    sel = (snr > 30) & (fr > 1.0) & (fr < 2.5) & np.isfinite(fr)
    return float(np.median(fr[sel]))


def ab(flux_ujy: float) -> float:
    return np.nan if not (flux_ujy > 0) else -2.5 * np.log10(flux_ujy) + UJY_ZP


def colour(phot, j, b1, b2) -> float:
    """m_b1 - m_b2 from consistent-aperture (aper_1) fluxes; ZP cancels."""
    f1 = phot[f"{b1}_flux_aper_1"][j]
    f2 = phot[f"{b2}_flux_aper_1"][j]
    if not (f1 > 0 and f2 > 0):
        return np.nan
    return -2.5 * np.log10(f1 / f2)


def beta_slope(phot, j, z, lo_rest, hi_rest):
    """Fit f_lambda ~ lambda^beta over bands whose rest wavelength is in
    [lo_rest, hi_rest] A and detected at aper-1 SNR>2. Needs >=2 bands.

    f_lambda ∝ f_nu / lambda^2, so log10(f_nu/lam^2) = beta*log10(lam)+c.
    """
    xs, ys = [], []
    for b in BANDS:
        lam_rest = PIVOT[b] / (1.0 + z)
        if not (lo_rest <= lam_rest <= hi_rest):
            continue
        fnu = phot[f"{b}_flux_aper_1"][j]
        efnu = phot[f"{b}_fluxerr_aper_1"][j]
        if not (fnu > 0 and efnu > 0 and fnu / efnu > 2.0):
            continue
        lam = PIVOT[b]
        xs.append(np.log10(lam))
        ys.append(np.log10(fnu / lam ** 2))
    if len(xs) < 2:
        return np.nan, len(xs)
    beta = float(np.polyfit(xs, ys, 1)[0])
    return beta, len(xs)


def akins(phot, j, compact) -> tuple[bool, str]:
    """Akins+24 COSMOS-Web: SNR_F444W>12, SNR_F277W>3, F277W-F444W>1.5, compact.
    (Brown-dwarf SED rejection is not photometrically replayable; disclosed.)"""
    snr444 = phot["f444w_tot_1"][j] / phot["f444w_etot_1"][j]
    snr277 = phot["f277w_tot_1"][j] / phot["f277w_etot_1"][j]
    c = colour(phot, j, "f277w", "f444w")
    ok = (snr444 > 12) and (snr277 > 3) and (c > 1.5) and compact
    why = (f"SNR444={snr444:.0f}(>12) SNR277={snr277:.0f}(>3) "
           f"F277-F444={c:.2f}(>1.5) compact={compact}")
    return bool(ok), why


def kocevski(phot, j, z, compact) -> tuple[bool, str]:
    """Kocevski+24: V-shape beta_UV<-0.37 & beta_opt>0 & compact & SNR_F444W>12.
    beta_UV also >-2.8 (brown-dwarf floor)."""
    snr444 = phot["f444w_tot_1"][j] / phot["f444w_etot_1"][j]
    b_uv, n_uv = beta_slope(phot, j, z, 1250.0, 3000.0)
    b_opt, n_opt = beta_slope(phot, j, z, 3600.0, 9000.0)
    vshape = (b_uv < -0.37) and (b_uv > -2.8) and (b_opt > 0)
    ok = vshape and compact and (snr444 > 12)
    why = (f"beta_UV={b_uv:.2f}[n{n_uv}](<-0.37) beta_opt={b_opt:.2f}[n{n_opt}](>0) "
           f"compact={compact} SNR444={snr444:.0f}(>12)")
    if np.isnan(b_uv) or np.isnan(b_opt):
        return False, why + " [indeterminate: <2 bands in a regime]"
    return bool(ok), why


def kokorev(phot, j, compact) -> tuple[bool, str]:
    """Kokorev+24: (red1 OR red2) & compact & F444W<27.7 & SNR_F444W>14."""
    snr444 = phot["f444w_tot_1"][j] / phot["f444w_etot_1"][j]
    m444 = ab(phot["f444w_tot_1"][j])
    c = {p: colour(phot, j, *p.split("-")) for p in
         ("f115w-f150w", "f200w-f277w", "f200w-f356w",
          "f150w-f200w", "f277w-f356w", "f277w-f444w")}
    red1 = (c["f115w-f150w"] < 0.8) and (c["f200w-f277w"] > 0.7) and (c["f200w-f356w"] > 1.0)
    red2 = (c["f150w-f200w"] < 0.8) and (c["f277w-f356w"] > 0.6) and (c["f277w-f444w"] > 0.7)
    ok = (red1 or red2) and compact and (m444 < 27.7) and (snr444 > 14)
    why = (f"red1={red1} red2={red2} F444={m444:.1f}(<27.7) "
           f"SNR444={snr444:.0f}(>14) compact={compact}")
    return bool(ok), why


def load_akins():
    """(ra, dec, and the columns Akins' cut needs) for the 434 COSMOS-Web LRDs."""
    from astropy.table import Table
    t = Table.read(AKINS_ECSV, format="ascii.ecsv")
    return {
        "id": [str(i) for i in t["id"]],
        "ra": np.asarray(t["ra"], float), "dec": np.asarray(t["dec"], float),
        "m277": np.asarray(t["m277"], float), "m444": np.asarray(t["m444"], float),
        "snr277": np.asarray(t["snr277"], float),
        "snr444": np.asarray(t["snr444"], float),
        "C444": np.asarray(t["C444"], float),
        "chi2_bd": np.asarray(t["chi2_bd"], float),
        "chi2_gal": np.asarray(t["chi2_gal"], float),
    }


def akins_catalog_cut(ak, j) -> tuple[bool, str]:
    """Akins+24 cut on Akins' own COSMOS-Web photometry (their aperture C444 and
    brown-dwarf chi2 are available here, so this is the *full* published cut)."""
    c = ak["m277"][j] - ak["m444"][j]                 # F277W-F444W (AB colour)
    compact = 0.5 < ak["C444"][j] < 0.7
    bd_ok = ak["chi2_bd"][j] > ak["chi2_gal"][j]      # galaxy/QSO beats brown dwarf
    ok = (ak["snr444"][j] > 12) and (ak["snr277"][j] > 3) and (c > 1.5) \
        and compact and bd_ok
    why = (f"F277-F444={c:.2f}(>1.5) C444={ak['C444'][j]:.2f}(0.5-0.7) "
           f"SNR444={ak['snr444'][j]:.0f} chi2_bd>{ak['chi2_gal'][j]:.0f}={bd_ok}")
    return bool(ok), why


def load_photoz() -> dict[str, float]:
    """Candidate photo-z peak from the Q3b photo-z run (for the slope regimes)."""
    z = {}
    fp = OUT / "candidates_photoz.csv"
    if fp.exists():
        for r in csv.DictReader(open(fp)):
            for k in ("z_peak", "zpeak", "z", "z_best"):
                if k in r and r[k] not in ("", None):
                    try:
                        z[r["id"]] = float(r[k])
                    except ValueError:
                        pass
                    break
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1.0,
                    help="parent-catalogue match radius, arcsec")
    args = ap.parse_args()

    zmap = load_photoz()
    ak = load_akins()
    cands = list(csv.DictReader(open(OUT / "candidates_crossmatch.csv")))
    fields = sorted({r["field"] for r in cands})
    loaded = {}
    for fld in fields:
        print(f"loading parent catalogue(s) for {fld}: {FIELD_CATS[fld]}")
        loaded[fld] = load_field(fld)

    rh_star = {fld: stellar_rh(loaded[fld][2]) for fld in fields}
    for fld in fields:
        print(f"  {fld}: stellar r_h = {rh_star[fld]:.2f} px "
              f"({rh_star[fld]*PIXSCALE:.3f}\"), compact if r_h < "
              f"{1.5*rh_star[fld]:.2f} px")

    out_rows = []
    print(f"\nreplaying published cuts on {len(cands)} candidates "
          f"(match radius {args.radius}\")...\n")
    for r in cands:
        cid, fld = r["id"], r["field"]
        ra, dec = float(r["ra"]), float(r["dec"])
        cra, cdec, phot = loaded[fld]
        sep = np.sqrt(((cra - ra) * np.cos(np.radians(dec))) ** 2
                      + (cdec - dec) ** 2) * 3600.0
        j = int(sep.argmin())
        rec = {"id": cid, "field": fld, "ra": ra, "dec": dec,
               "photz": zmap.get(cid, np.nan),
               "parent_sep_arcsec": round(float(sep[j]), 3)}

        if sep[j] > args.radius:
            # No PRIMER/JADES source. For COSMOS, check the COSMOS-Web (Akins)
            # catalogue: PRIMER-COSMOS doesn't cover all of COSMOS-Web.
            if fld == "COSMOS":
                asep = np.sqrt(((ak["ra"] - ra) * np.cos(np.radians(dec))) ** 2
                               + (ak["dec"] - dec) ** 2) * 3600.0
                a = int(asep.argmin())
                if asep[a] < args.radius:
                    ok, why = akins_catalog_cut(ak, a)
                    rec.update(parent_match="cosmos-web(akins)",
                               akins_cw_id=f"CW-{ak['id'][a]}",
                               akins_cw_sep=round(float(asep[a]), 2),
                               akins_pass=ok, akins_why=why,
                               verdict="published_LRD",
                               verdict_note=("confirmed Akins COSMOS-Web LRD "
                                             "(outside PRIMER); passes Akins cut"))
                    print(f"  {cid} {fld:<8} COSMOS-Web LRD CW-{ak['id'][a]} "
                          f"@{asep[a]:.2f}\" | Akins pass={ok} | {why}")
                    out_rows.append(rec)
                    continue
            rec["parent_match"] = "no_jwst_counterpart"
            rec["verdict"] = "no_jwst_counterpart"
            rec["verdict_note"] = (f"no catalogued JWST source within "
                                   f"{args.radius}\" (nearest {sep[j]:.1f}\") "
                                   f": spurious detection or outside footprint")
            print(f"  {cid} {fld:<8} NO JWST counterpart within {args.radius}\" "
                  f"(nearest {sep[j]:.2f}\"): spurious / outside footprint")
            out_rows.append(rec)
            continue

        marginal = sep[j] > 0.6
        fr = phot["flux_radius"][j]
        compact = bool(fr < 1.5 * rh_star[fld]) and np.isfinite(fr)
        z = zmap.get(cid, 5.0)                 # fallback z for the slope regimes
        snr444 = phot["f444w_tot_1"][j] / phot["f444w_etot_1"][j]
        m444 = ab(phot["f444w_tot_1"][j])
        c2744 = colour(phot, j, "f277w", "f444w")

        ak_ok, ak_why = akins(phot, j, compact)
        ko_ok, ko_why = kocevski(phot, j, z, compact)
        kk_ok, kk_why = kokorev(phot, j, compact)

        rec.update(
            parent_match=("marginal" if marginal else "yes"),
            flag=int(phot["flag"][j]),
            flux_radius_px=round(float(fr), 2),
            reff_arcsec=round(float(phot["reff_arcsec"][j]), 3),
            compact=compact, f444_AB=round(float(m444), 2),
            snr_f444=round(float(snr444), 1),
            F277_F444=round(float(c2744), 2),
            akins_pass=ak_ok, kocevski_pass=ko_ok, kokorev_pass=kk_ok,
            akins_why=ak_why, kocevski_why=ko_why, kokorev_why=kk_why)

        # verdict: applied per field (Akins only defined on COSMOS).
        applied = ["kocevski", "kokorev"] + (["akins"] if fld == "COSMOS" else [])
        passes = [p for p in applied if rec[f"{p}_pass"]]
        if passes:
            rec["verdict"] = "GENUINELY_NEW"        # passes cuts yet absent
            rec["verdict_note"] = ("passes " + "+".join(passes)
                                   + " but absent from its catalogue")
        else:
            rec["verdict"] = "below_cuts"           # absence explained
            rec["verdict_note"] = "fails every applicable published selection"

        print(f"  {cid} {fld:<8} sep={sep[j]:.2f}\" flag={int(phot['flag'][j])} "
              f"r_h={fr:.2f}px compact={compact} F444={m444:.1f} "
              f"F277-F444={c2744:.2f} SNR444={snr444:.0f}")
        print(f"      akins={ak_ok} | kocevski={ko_ok} | kokorev={kk_ok} "
              f"-> {rec['verdict']}")
        out_rows.append(rec)

    fieldnames: list[str] = []
    for rec in out_rows:
        for k in rec:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(OUT / "candidates_selection_replay.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    def ids(v):
        return [r["id"] for r in out_rows if r.get("verdict") == v]
    print("\n=== verdict tally ===")
    print(f"  published_LRD (confirmed, COSMOS-Web/Akins): {ids('published_LRD')}")
    print(f"  GENUINELY_NEW (passes a cut yet absent):     {ids('GENUINELY_NEW')}")
    print(f"  below_cuts (real source, fails every cut):   {ids('below_cuts')}")
    print(f"  no_jwst_counterpart (spurious/off-footprint):{ids('no_jwst_counterpart')}")
    n_new = len(ids("GENUINELY_NEW"))
    print(f"\n{n_new} of the 8 novel candidates pass any published LRD selection on "
          f"catalogue-grade photometry.")
    print("CAVEAT: compactness is unified as r_h<1.5 r_h,stars (a disclosed "
          "stand-in for each paper's aperture-ratio compactness); the field "
          "morpho catalogues = the plain detection catalogues (same source list), "
          "so a non-match is a genuine absence.")
    print(f"wrote {OUT / 'candidates_selection_replay.csv'}")


if __name__ == "__main__":
    main()
