"""Literature cross-match for the star-vetoed LRD candidates (M3b next-step 2).

For each veto survivor (``research/candidates_vetted.csv``, verdict
``candidate``), cone-search every machine-checkable relevant catalogue:

* **Kokorev+24 v1.1** (local FITS) — re-checked at 2" (the candidate list
  already excluded 1.5" matches by construction; this catches edge cases);
* **SIMBAD TAP** (2") — is the source catalogued at all, and as what
  (``otype``: a QSO/AGN type is interesting, a known star kills it);
* **VizieR LRD tables** (2") — crucially the **Perger+25 list
  (J/A+A/693/L2)**, which is a *compilation* of the major published LRD
  samples through 2024: Akins+24 COSMOS-Web (420), Kokorev+24 (219),
  Kocevski+24 (165), Greene+24, Matthee+24, Labbe+23 and others (17 refs).
  Plus the Euclid LRD sample (Bisigello+) and the RUBIES broad-Balmer-line
  census (Hviding+25, the spectroscopic LRD-rich sample);
* **NED** (2") — complementary coverage of the CANDELS-era literature.

Honest limitation, disclosed in the output: coverage of the published LRD
literature ends at the Perger+25 compilation cutoff (~late 2024) plus RUBIES;
LRD samples published in 2025-26 (e.g. the Kocevski+25 systematic ~300) are
not machine-checkable here and need a manual pass before any novelty claim.

    python -m experiments.candidate_crossmatch

Outputs: research/candidates_crossmatch.csv (+ a printed verdict table)
"""
from __future__ import annotations

import csv
import time

import numpy as np
import requests

from core import config

OUT = config.ROOT / "research"
RADIUS_ARCSEC = 2.0

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
NED_URL = "https://ned.ipac.caltech.edu/cgi-bin/objsearch"

LRD_TABLES = {                       # VizieR table -> short label
    "J/A+A/693/L2/stacked": "Perger25_LRDcompilation",
    "J/A+A/711/A24/tabled1": "Euclid_LRD",
    "J/A+A/702/A57/tableb1": "RUBIES_broadBalmer",
}


def _tap_csv(url: str, adql: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=dict(REQUEST="doQuery", LANG="ADQL",
                                              FORMAT="csv", QUERY=adql), timeout=40)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            if len(lines) < 2:
                return []
            keys = [k.strip('"') for k in lines[0].split(",")]
            return [dict(zip(keys, l.split(","))) for l in lines[1:]]
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return [{"error": "query_failed"}]


def simbad(ra: float, dec: float) -> str:
    rad = RADIUS_ARCSEC / 3600.0
    rows = _tap_csv(SIMBAD_TAP, (
        "SELECT TOP 2 main_id, otype, "
        f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec}))*3600 AS sep "
        f"FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {rad}))=1 ORDER BY sep"))
    return "; ".join(f"{r.get('main_id', '?')} [{r.get('otype', '?')}] "
                     f"{float(r['sep']):.2f}\"" if "sep" in r else str(r)
                     for r in rows) or ""


def vizier_lrds(ra: float, dec: float) -> str:
    rad = RADIUS_ARCSEC / 3600.0
    hits = []
    for table, label in LRD_TABLES.items():
        extra = ", Ref, zphot" if label == "Perger25_LRDcompilation" else ""
        rows = _tap_csv(VIZIER_TAP, (
            f'SELECT TOP 1 RAJ2000, DEJ2000{extra}, '
            f"DISTANCE(POINT('ICRS', RAJ2000, DEJ2000), POINT('ICRS', {ra}, {dec}))*3600 AS sep "
            f'FROM "{table}" WHERE CONTAINS(POINT(\'ICRS\', RAJ2000, DEJ2000), '
            f"CIRCLE('ICRS', {ra}, {dec}, {rad}))=1"))
        for r in rows:
            if "sep" in r:
                ref = f" ref={r['Ref']} zphot={r.get('zphot', '')}" if "Ref" in r else ""
                hits.append(f"{label} {float(r['sep']):.2f}\"{ref}")
            elif "error" in r:
                hits.append(f"{label} query_failed")
    return "; ".join(hits)


def ned(ra: float, dec: float, retries: int = 2) -> str:
    from astropy.io.votable import parse_single_table
    import io as _io
    import warnings

    params = dict(search_type="Near Position Search", in_csys="Equatorial",
                  in_equinox="J2000.0", lon=f"{ra}d", lat=f"{dec}d",
                  radius=str(RADIUS_ARCSEC / 60.0), of="xml_main")
    for attempt in range(retries):
        try:
            r = requests.get(NED_URL, params=params, timeout=40)
            r.raise_for_status()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = parse_single_table(_io.BytesIO(r.content)).to_table()
            if len(t) == 0:
                return ""
            row = t[0]
            name = str(row["Object Name"]) if "Object Name" in t.colnames else "?"
            typ = str(row["Type"]) if "Type" in t.colnames else "?"
            return f"{name} [{typ}]"
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return "query_failed"


def kokorev_sep(ra: float, dec: float, cat) -> float:
    d2 = ((np.asarray(cat["ra"], float) - ra) * np.cos(np.radians(dec))) ** 2 \
         + (np.asarray(cat["dec"], float) - dec) ** 2
    return float(np.sqrt(d2.min()) * 3600.0)


def main() -> None:
    from astropy.io import fits

    cat = fits.open(config.DATA_DIR / "lrd_kokorev.fits")[1].data
    rows = [r for r in csv.DictReader(open(OUT / "candidates_vetted.csv"))
            if r["verdict"] == "candidate"]
    print(f"cross-matching {len(rows)} veto survivors (radius {RADIUS_ARCSEC}\")...\n")

    out_rows = []
    for r in rows:
        ra, dec = float(r["ra"]), float(r["dec"])
        m = {
            "kokorev_sep_arcsec": round(kokorev_sep(ra, dec, cat), 2),
            "simbad": simbad(ra, dec),
            "lrd_catalogs": vizier_lrds(ra, dec),
            "ned": ned(ra, dec),
        }
        in_lrd_cat = bool(m["lrd_catalogs"]) and "query_failed" not in m["lrd_catalogs"]
        known_star = any(s in m["simbad"] for s in ("[*", "[BD", "Star"))
        m["novelty"] = ("in_published_LRD_sample" if in_lrd_cat or
                        m["kokorev_sep_arcsec"] < RADIUS_ARCSEC else
                        "known_star" if known_star else
                        "not_in_machine_checkable_LRD_samples")
        out_rows.append({**r, **m})
        print(f"  #{r['rank']:>3} {r['id']} {r['field']:<8} kokorev={m['kokorev_sep_arcsec']}\" "
              f"| simbad: {m['simbad'] or '-'} | lrd-cats: {m['lrd_catalogs'] or '-'} "
              f"| ned: {m['ned'] or '-'}")
        time.sleep(0.5)

    with open(OUT / "candidates_crossmatch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    novel = [r for r in out_rows if r["novelty"] == "not_in_machine_checkable_LRD_samples"]
    print(f"\n{len(novel)}/{len(out_rows)} survivors are in no machine-checkable "
          f"LRD sample (Kokorev v1.1 local; Perger+25 compilation = Akins+24 + "
          f"Kokorev+24 + Kocevski+24 + Greene+24 + Matthee+24 + Labbe+23 et al.; "
          f"Euclid; RUBIES).")
    print("CAVEAT: coverage ends at the Perger+25 compilation cutoff (~late 2024) "
          "— 2025-26 LRD samples (e.g. Kocevski+25's systematic ~300) need a "
          "manual check before any novelty claim.")
    print(f"wrote {OUT / 'candidates_crossmatch.csv'}")


if __name__ == "__main__":
    main()
