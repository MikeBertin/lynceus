"""Literature cross-match for the star-vetoed LRD candidates (M3b next-step 2).

For each veto survivor (``research/candidates_vetted.csv``, verdict
``candidate``), cone-search every machine-checkable relevant catalogue:

* **Kokorev+24 v1.1** (local FITS) — re-checked at 2" (the candidate list
  already excluded 1.5" matches by construction; this catches edge cases);
* **SIMBAD TAP** (2") — is the source catalogued at all, and as what
  (``otype``: a QSO/AGN type is interesting, a known star kills it);
* **Kocevski+24** (2") — the full 341-LRD Table 3 (ApJ 986/126), fetched from
  the author's GitHub since it is not on VizieR. 254 of the 341 land in our
  fields (PRIMER->COSMOS+UDS, JADES->GOODS-S), so it is the single most valuable
  non-VO check; only 165 of it reached the Perger compilation;
* **Akins+24 COSMOS-Web** (2") — the full 434-LRD Table 1 (ApJ 991/37, ECSV
  on the author's GitHub, also not on VizieR). Covers COSMOS completely and
  closes the Perger residual (Perger had 420 of the 434). Self-validates: our
  two rediscoveries match it at 0.14";
* **VizieR LRD tables** (2") — the **Perger+25 list (J/A+A/693/L2)**, a
  compilation of the major samples through 2024; the Euclid LRD sample
  (Bisigello+); and the **RUBIES full spectroscopic census** (Hviding+25,
  J/A+A/702/A57 — 80 LRDs in UDS+EGS, the relevant spectroscopic sample for
  our UDS candidates);
* **NED** (2") — complementary coverage of the CANDELS-era literature.

So every one of our three fields is covered by at least one photometric *and*
(for UDS) spectroscopic LRD sample. NEXUS is the NEP field — no overlap.
Honest limitation: "absent from a selection" can mean *below their cuts*
rather than non-existent; only a spectrum proves an LRD.

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

# Kocevski+24 (341 LRDs, ApJ 986/126) is NOT on VizieR — its full Table 3 is
# on the author's GitHub. 254 of the 341 fall in our fields (UDS/GOODS-S/COSMOS
# via PRIMER+JADES), so it is the highest-value non-VO check. Fetched once to a
# local file (data/ is gitignored). Validated: 119/216 of our known Kokorev
# LRDs match a Kocevski source within 2" (median 0.1") — the catalogues overlap
# and the cross-match is sound, so a *non*-match is a real absence.
KOCEVSKI_URL = "https://raw.githubusercontent.com/dalekocevski/Kocevski24/main/Kocevski24.Table3.dat"
KOCEVSKI_DAT = config.DATA_DIR / "lrd_kocevski24.dat"

# Akins+24 COSMOS-Web (434 LRDs, ApJ 991/37) — also not on VizieR; full Table 1
# is on the author's GitHub as ECSV. Perger included 420 of the 434, so this
# closes that residual for our COSMOS candidates. Self-validates: our two
# rediscoveries match it at 0.14".
AKINS_URL = "https://raw.githubusercontent.com/hollisakins/akins24_cw/main/COSMOS-Web_LRDs.dat"
AKINS_ECSV = config.DATA_DIR / "lrd_akins24_cosmosweb.ecsv"


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


def load_kocevski():
    """(ra, dec, id, z) for the 341 Kocevski+24 LRDs; downloads once if absent."""
    if not KOCEVSKI_DAT.exists():
        import urllib.request
        print(f"fetching Kocevski+24 Table 3 -> {KOCEVSKI_DAT}")
        urllib.request.urlretrieve(KOCEVSKI_URL, KOCEVSKI_DAT)
    rows = [l.split() for l in open(KOCEVSKI_DAT)
            if l.strip() and not l.startswith("#")]
    # ID may be 1-2 whitespace tokens; the 11 trailing fields are numeric
    ra = np.array([float(r[-11]) for r in rows])
    dec = np.array([float(r[-10]) for r in rows])
    zid = [" ".join(r[:-11]) for r in rows]
    z = np.array([float(r[-8]) for r in rows])
    return ra, dec, zid, z


def kocevski_match(ra: float, dec: float, koc) -> str:
    kra, kdec, kid, kz = koc
    sep = np.sqrt(((kra - ra) * np.cos(np.radians(dec))) ** 2
                  + (kdec - dec) ** 2) * 3600.0
    j = int(sep.argmin())
    if sep[j] < RADIUS_ARCSEC:
        return f"{kid[j]} z={kz[j]:.2f} {sep[j]:.2f}\""
    return ""                              # nearest is > radius: not in Kocevski


def load_akins():
    """(ra, dec, id) for the 434 Akins+24 COSMOS-Web LRDs; downloads once."""
    from astropy.table import Table
    if not AKINS_ECSV.exists():
        import urllib.request
        print(f"fetching Akins+24 COSMOS-Web catalog -> {AKINS_ECSV}")
        urllib.request.urlretrieve(AKINS_URL, AKINS_ECSV)
    t = Table.read(AKINS_ECSV, format="ascii.ecsv")
    return np.asarray(t["ra"], float), np.asarray(t["dec"], float), \
        [str(i) for i in t["id"]]


def akins_match(ra: float, dec: float, ak) -> str:
    ara, adec, aid = ak
    sep = np.sqrt(((ara - ra) * np.cos(np.radians(dec))) ** 2
                  + (adec - dec) ** 2) * 3600.0
    j = int(sep.argmin())
    if sep[j] < RADIUS_ARCSEC:
        return f"CW-{aid[j]} {sep[j]:.2f}\""
    return ""                              # COSMOS-only: UDS/GOODS-S land far away


def main() -> None:
    from astropy.io import fits

    cat = fits.open(config.DATA_DIR / "lrd_kokorev.fits")[1].data
    koc = load_kocevski()
    ak = load_akins()
    rows = [r for r in csv.DictReader(open(OUT / "candidates_vetted.csv"))
            if r["verdict"] == "candidate"]
    print(f"cross-matching {len(rows)} veto survivors (radius {RADIUS_ARCSEC}\")...\n")

    out_rows = []
    for r in rows:
        ra, dec = float(r["ra"]), float(r["dec"])
        m = {
            "kokorev_sep_arcsec": round(kokorev_sep(ra, dec, cat), 2),
            "kocevski24": kocevski_match(ra, dec, koc),
            "akins24_cw": akins_match(ra, dec, ak),
            "simbad": simbad(ra, dec),
            "lrd_catalogs": vizier_lrds(ra, dec),
            "ned": ned(ra, dec),
        }
        in_lrd_cat = bool(m["lrd_catalogs"]) and "query_failed" not in m["lrd_catalogs"]
        known_star = any(s in m["simbad"] for s in ("[*", "[BD", "Star"))
        m["novelty"] = ("in_published_LRD_sample" if in_lrd_cat or bool(m["kocevski24"])
                        or bool(m["akins24_cw"]) or m["kokorev_sep_arcsec"] < RADIUS_ARCSEC
                        else "known_star" if known_star else
                        "not_in_any_checked_LRD_sample")
        out_rows.append({**r, **m})
        print(f"  #{r['rank']:>3} {r['id']} {r['field']:<8} kokorev={m['kokorev_sep_arcsec']}\" "
              f"| kocevski24: {m['kocevski24'] or '-'} | akins-CW: {m['akins24_cw'] or '-'} "
              f"| simbad: {m['simbad'] or '-'} | lrd-cats: {m['lrd_catalogs'] or '-'} "
              f"| ned: {m['ned'] or '-'}")
        time.sleep(0.5)

    with open(OUT / "candidates_crossmatch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    novel = [r for r in out_rows if r["novelty"] == "not_in_any_checked_LRD_sample"]
    print(f"\n{len(novel)}/{len(out_rows)} survivors are in NONE of the checked LRD "
          f"samples:\n  - Kokorev+24 v1.1 (local FITS)\n  - Kocevski+24 341-LRD "
          f"Table 3 (local, GitHub; 254 in our fields)\n  - Akins+24 COSMOS-Web "
          f"434-LRD full Table 1 (local, GitHub; closes the Perger residual)\n"
          f"  - Perger+25 VizieR compilation; Euclid LRD; RUBIES full census "
          f"(UDS+EGS spectroscopic).")
    print("Field coverage of the checked samples: COSMOS = Akins-CW(full) + "
          "Kocevski-PRIMER + Kokorev; UDS = RUBIES(spec) + Kocevski-PRIMER + "
          "Kokorev; GOODS-S = Kocevski-JADES + Kokorev. NEXUS is the NEP field "
          "(no overlap). CAVEAT: 'absent from a selection' can mean below-their-"
          "cuts, not non-existent — spectroscopic confirmation is the only proof.")
    print(f"wrote {OUT / 'candidates_crossmatch.csv'}")


if __name__ == "__main__":
    main()
