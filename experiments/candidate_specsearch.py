"""Archival-spectrum search for the LRD candidates (Q3c, Step 1).

The Q3b cross-match checked the *LRD selections* (photometric samples + the
RUBIES LRD-flagged subset). This step asks the more direct question: does any
of our 10 veto survivors already have a **public reduced NIRSpec spectrum** —
regardless of whether anyone flagged it as an LRD? If one exists, the candidate
can be confirmed or rejected today, for free (the literal answer to "only a
spectrum confirms an LRD").

The DAWN JWST Archive spectroscopic database (DJA-Spec) is the *union* of all
public NIRSpec programs reduced with msaexp — RUBIES, JADES, CEERS, UNCOVER,
CAPERS, PRIMER, ... — so a single cone-search against its master emission-line
table covers every archive the roadmap named. We pull the whole 80k-row table
once (``dja_msaexp_emission_lines_v4.4.csv.gz``, position-searchable) and match
each candidate by RA/Dec.

For every match we report separation, grating/filter, ``z_best`` and its DJA
``grade`` (3 = secure, 2 = uncertain, 1/0 = bad/artefact), and — because the
LRD signature is a broad Hα on a V-shaped continuum — the Hα/Hβ line fluxes and
the fitted UV slope ``beta`` where present.

    python -m experiments.candidate_specsearch [--radius 1.0]

Outputs: research/candidates_specsearch.csv (+ a printed verdict table).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import urllib.request

import numpy as np

from core import config

OUT = config.ROOT / "research"
DJA_URL = ("https://s3.amazonaws.com/msaexp-nirspec/extractions/"
           "dja_msaexp_emission_lines_v4.4.csv.gz")
DJA_CSV = config.DATA_DIR / "dja_msaexp_emission_lines_v4.4.csv.gz"

# Columns we keep from the 587-wide table (RA/Dec + redshift/grade + the
# Balmer lines and UV slope that carry the LRD signature).
KEEP = ["file", "root", "srcid", "ra", "dec", "grating", "filter", "effexptm",
        "z_best", "z_prism", "z_grating", "grade", "zgrade",
        "line_ha", "line_ha_err", "line_hb", "line_hb_err", "beta"]

GRADE_MEANING = {"3": "secure", "2": "uncertain", "1": "bad", "0": "artefact",
                 "": "ungraded"}


def load_dja() -> tuple[list[dict], np.ndarray, np.ndarray]:
    """The DJA-Spec master emission-line table; downloads once (~130 MB).

    Returns the kept-column rows plus parallel ra/dec float arrays. Pure stdlib
    (the 587-wide table is streamed once and only ``KEEP`` columns retained).
    """
    if not DJA_CSV.exists():
        print(f"fetching DJA-Spec master table -> {DJA_CSV}")
        urllib.request.urlretrieve(DJA_URL, DJA_CSV)
    recs: list[dict] = []
    with gzip.open(DJA_CSV, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {k: header.index(k) for k in KEEP}
        rai, deci = idx["ra"], idx["dec"]
        for row in reader:
            try:
                float(row[rai]); float(row[deci])
            except (ValueError, IndexError):
                continue                       # no sky position -> unsearchable
            recs.append({k: row[i] for k, i in idx.items()})
    ra = np.array([float(r["ra"]) for r in recs])
    dec = np.array([float(r["dec"]) for r in recs])
    print(f"loaded {len(recs):,} public NIRSpec extractions from DJA-Spec")
    return recs, ra, dec


def match(ra0: float, dec0: float, recs, ra, dec, radius: float):
    """Nearest DJA spectrum within ``radius`` arcsec (or None)."""
    sep = np.sqrt(((ra - ra0) * np.cos(np.radians(dec0))) ** 2
                  + (dec - dec0) ** 2) * 3600.0
    j = int(sep.argmin())
    if sep[j] > radius:
        return None, float(sep[j])
    return recs[j], float(sep[j])


def fmt(v, nd=3) -> str:
    if v is None or str(v).strip() in ("", "nan"):
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def validate(recs, ra_arr, dec_arr, radius: float) -> None:
    """Self-check: the 216 known Kokorev LRDs *should* have DJA-Spec matches.

    Same guard the Q3b cross-match uses — if the machinery recovers a sensible
    fraction of known LRDs (RUBIES/JADES/... spectra live in DJA-Spec), then a
    *non*-match for our candidates is a real absence, not a coordinate bug.
    """
    from astropy.io import fits
    cat = fits.open(config.DATA_DIR / "lrd_kokorev.fits")[1].data
    kra = np.asarray(cat["ra"], float)
    kdec = np.asarray(cat["dec"], float)
    n_spec, seps = 0, []
    for ra, dec in zip(kra, kdec):
        hit, sep = match(ra, dec, recs, ra_arr, dec_arr, radius)
        if hit is not None:
            n_spec += 1
            seps.append(sep)
    med = np.median(seps) if seps else float("nan")
    print(f"\n[validate] {n_spec}/{len(kra)} known Kokorev LRDs have a DJA-Spec "
          f"spectrum within {radius}\" (median sep {med:.2f}\").")
    print("[validate] non-zero recovery => the cross-match is sound, so a "
          "candidate non-match is a genuine absence of archival spectroscopy.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1.0,
                    help="cone radius in arcsec (DJA slit positions are precise)")
    ap.add_argument("--validate", action="store_true",
                    help="also cross-match the 216 known Kokorev LRDs (sanity)")
    args = ap.parse_args()

    recs, ra_arr, dec_arr = load_dja()
    if args.validate:
        validate(recs, ra_arr, dec_arr, args.radius)
    rows = list(csv.DictReader(open(OUT / "candidates_crossmatch.csv")))
    print(f"\nsearching {len(rows)} candidates against DJA-Spec "
          f"(radius {args.radius}\")...\n")

    out_rows, n_hit = [], 0
    for r in rows:
        ra, dec = float(r["ra"]), float(r["dec"])
        hit, sep = match(ra, dec, recs, ra_arr, dec_arr, args.radius)
        rec = {"id": r["id"], "field": r["field"], "ra": ra, "dec": dec,
               "novelty_photometric": r["novelty"]}
        if hit is None:
            rec.update(spec_match="none", spec_sep_arcsec=round(sep, 2))
            print(f"  {r['id']} {r['field']:<8} no spectrum "
                  f"(nearest {sep:.1f}\")")
        else:
            n_hit += 1
            grade = str(hit["grade"]) if not _isnan(hit["grade"]) else ""
            grade = grade[:-2] if grade.endswith(".0") else grade
            rec.update(
                spec_match="SPECTRUM", spec_sep_arcsec=round(sep, 2),
                root=hit["root"], grating=hit["grating"], filter=hit["filter"],
                effexptm=fmt(hit["effexptm"], 0),
                z_best=fmt(hit["z_best"]), grade=grade,
                grade_meaning=GRADE_MEANING.get(grade, grade),
                line_ha=fmt(hit["line_ha"]), line_ha_err=fmt(hit["line_ha_err"]),
                line_hb=fmt(hit["line_hb"]), beta=fmt(hit["beta"], 2),
                file=hit["file"])
            print(f"  {r['id']} {r['field']:<8} *** SPECTRUM at {sep:.2f}\" "
                  f"| {hit['grating']}/{hit['filter']} exp={fmt(hit['effexptm'],0)}s "
                  f"| z_best={fmt(hit['z_best'])} grade={grade}"
                  f"({GRADE_MEANING.get(grade, grade)}) "
                  f"| Ha={fmt(hit['line_ha'])} beta={fmt(hit['beta'],2)}")
        out_rows.append(rec)

    # union the keys so pandas-sparse rows still write cleanly
    fields: list[str] = []
    for rec in out_rows:
        for k in rec:
            if k not in fields:
                fields.append(k)
    with open(OUT / "candidates_specsearch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{n_hit}/{len(rows)} candidates have a public NIRSpec spectrum in "
          f"DJA-Spec (v4.4, {len(recs):,} extractions; the union of RUBIES/JADES/"
          f"CEERS/UNCOVER/CAPERS/PRIMER).")
    if n_hit == 0:
        nearest = min(rr["spec_sep_arcsec"] for rr in out_rows)
        print(f"Nearest public spectrum to any candidate: {nearest:.1f}\" "
              f"(> {args.radius}\" match radius) — all 10 remain "
              f"spectroscopically unobserved. They stay a follow-up target list.")
    print(f"wrote {OUT / 'candidates_specsearch.csv'}")


def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
