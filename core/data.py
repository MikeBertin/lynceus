"""JWST cutout data layer for Lynceus.

Two ways to fill the cutout cache, both producing identical on-disk artefacts
(a ``manifest.csv`` plus one ``.npy`` float array and one ``.png`` per object):

* :func:`generate_synthetic_dataset` — physically-motivated Sersic-profile
  galaxies (n~1 disks, n~4 spheroids, clumpy irregulars, PSF point sources,
  two-body mergers), with wavelength-dependent PSF and noise. This is the
  default M1 dataset: it needs no multi-GB downloads, so the whole pipeline and
  browser demo are reproducible from a clean checkout.

* :func:`extract_real_cutouts` — the genuine article: given a CEERS/JADES NIRCam
  mosaic per band (dropped in ``data/mosaics/``) and a catalogue of RA/Dec +
  morphology labels, it cuts thumbnails with :class:`astropy.nddata.Cutout2D`.
  Same normalisation, same manifest, so training and the demo are agnostic to
  which source produced the cache.
"""
from __future__ import annotations

import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import config
from .morphology import CLASSES, CLASS_TO_IDX

# DJA grizli cutout service — real JWST NIRCam cutouts by RA/Dec, no mosaic
# download. One request returns a multi-extension FITS, one HDU per filter.
CUTOUT_SERVICE = "https://grizli-cutout.herokuapp.com/thumb"
SERVICE_FILTERS = "f444w-clear,f356w-clear,f200w-clear"   # (R, G, B) — match config.BANDS

MANIFEST = config.CUTOUTS_DIR / "manifest.csv"
_FIELDS = ("id", "label", "label_idx", "source", "ra", "dec", "redshift", "npy", "png")


@dataclass
class CutoutRecord:
    id: str
    label: str
    label_idx: int
    source: str
    ra: float
    dec: float
    redshift: float
    npy: str
    png: str


# ---------------------------------------------------------------------------
# Shared normalisation + caching
# ---------------------------------------------------------------------------
def asinh_stretch(img: np.ndarray, lo_pct: float = 45.0, hi_pct: float = 99.8,
                  soft: float = 0.15) -> np.ndarray:
    """Per-channel percentile clip + asinh stretch into [0, 1].

    ``img`` is (C, H, W). asinh stretch is the standard way astronomers display
    high-dynamic-range sky images, keeping faint structure visible without the
    bright cores saturating. The black point defaults to the per-channel median
    (a robust background estimate) so empty sky stays dark instead of blooming
    into noise.
    """
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        ch = img[c].astype(np.float32)
        lo, hi = np.percentile(ch, [lo_pct, hi_pct])
        ch = np.clip((ch - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        ch = np.arcsinh(ch / soft) / np.arcsinh(1.0 / soft)
        out[c] = ch
    return out


def _save_record(rec_id: str, label: str, source: str, ra: float, dec: float,
                 redshift: float, stretched: np.ndarray) -> CutoutRecord:
    """Persist a normalised (C, H, W) cutout as .npy + .png and return its record."""
    from PIL import Image

    npy_rel = f"{rec_id}.npy"
    png_rel = f"{rec_id}.png"
    np.save(config.CUTOUTS_DIR / npy_rel, stretched.astype(np.float32))
    rgb = (np.clip(stretched, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(config.CUTOUTS_DIR / png_rel)
    return CutoutRecord(rec_id, label, CLASS_TO_IDX[label], source,
                        float(ra), float(dec), float(redshift), npy_rel, png_rel)


def write_manifest(records: list[CutoutRecord]) -> Path:
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    return MANIFEST


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"No cutout manifest at {MANIFEST}. Run experiments/fetch_data.py first.")
    with open(MANIFEST, newline="") as fh:
        return list(csv.DictReader(fh))


def load_cutout(npy_name: str) -> np.ndarray:
    """Load a stored (C, H, W) float cutout in [0, 1]."""
    return np.load(config.CUTOUTS_DIR / npy_name)


# ---------------------------------------------------------------------------
# Synthetic Sersic galaxies (default M1 dataset)
# ---------------------------------------------------------------------------
def _sersic(shape, amp, r_eff, n, x0, y0, ellip, theta):
    from astropy.modeling.models import Sersic2D
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    m = Sersic2D(amplitude=amp, r_eff=r_eff, n=n, x_0=x0, y_0=y0,
                 ellip=ellip, theta=theta)
    return np.asarray(m(x, y), dtype=np.float32)


def _psf_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    from astropy.convolution import Gaussian2DKernel, convolve
    return convolve(img, Gaussian2DKernel(sigma), boundary="extend")


def _render_galaxy(label: str, rng: np.random.Generator, px: int) -> np.ndarray:
    """Render one (3, px, px) raw flux cube for the given morphology class.

    Three channels stand in for (F444W, F356W, F200W) -> (R, G, B), with a
    wavelength-dependent PSF (redder = broader) and class-dependent colour.
    """
    c = px / 2.0
    jit = lambda s=0.08: c + rng.normal(0, s * px)  # small centroid jitter
    # PSF FWHM grows with wavelength: R broadest, B sharpest (realistic for JWST).
    psf_sigma = {0: 1.7, 1: 1.4, 2: 1.1}

    # band amplitude multipliers encode colour (R, G, B)
    if label == "disk":
        n, r_eff = 1.0, rng.uniform(0.16, 0.26) * px
        ellip, theta = rng.uniform(0.1, 0.7), rng.uniform(0, np.pi)
        x0, y0 = jit(), jit()
        base = _sersic((px, px), 1.0, r_eff, n, x0, y0, ellip, theta)
        # a couple of blue star-forming clumps in the disk
        for _ in range(rng.integers(0, 3)):
            base = base + _sersic((px, px), rng.uniform(0.2, 0.5),
                                  rng.uniform(2, 4), 1.0,
                                  x0 + rng.normal(0, 0.18 * px),
                                  y0 + rng.normal(0, 0.18 * px), 0.1, 0.0)
        colour = (0.8, 1.0, 1.15)
    elif label == "spheroid":
        n, r_eff = 4.0, rng.uniform(0.07, 0.14) * px
        ellip, theta = rng.uniform(0.0, 0.25), rng.uniform(0, np.pi)
        base = _sersic((px, px), 1.0, r_eff, n, jit(0.04), jit(0.04), ellip, theta)
        colour = (1.2, 1.0, 0.78)            # redder, centrally concentrated
    elif label == "irregular":
        base = np.zeros((px, px), np.float32)
        for _ in range(rng.integers(2, 5)):
            base = base + _sersic((px, px), rng.uniform(0.5, 1.0),
                                  rng.uniform(0.06, 0.13) * px, 1.0,
                                  jit(0.22), jit(0.22),
                                  rng.uniform(0.1, 0.6), rng.uniform(0, np.pi))
        colour = (0.85, 1.0, 1.1)
    elif label == "compact":
        # barely-resolved: a small, round, fairly concentrated source
        base = _sersic((px, px), 2.2, rng.uniform(0.025, 0.05) * px,
                       rng.uniform(2.0, 4.0), jit(0.03), jit(0.03),
                       rng.uniform(0.0, 0.2), rng.uniform(0, np.pi))
        colour = (rng.uniform(1.0, 1.3), 1.0, rng.uniform(0.7, 1.0))  # often red
    elif label == "point_source":
        base = _sersic((px, px), 2.5, 0.6, 1.0, jit(0.03), jit(0.03), 0.0, 0.0)
        colour = (rng.uniform(0.8, 1.2), 1.0, rng.uniform(0.8, 1.2))
    elif label == "merger":
        sep = rng.uniform(0.18, 0.34) * px
        ang = rng.uniform(0, np.pi)
        dx, dy = sep * np.cos(ang), sep * np.sin(ang)
        n1 = rng.choice([1.0, 4.0])
        n2 = rng.choice([1.0, 4.0])
        base = (_sersic((px, px), 1.0, rng.uniform(0.08, 0.15) * px, n1,
                        c + dx, c + dy, rng.uniform(0.1, 0.5), rng.uniform(0, np.pi))
                + _sersic((px, px), rng.uniform(0.6, 1.0), rng.uniform(0.08, 0.15) * px,
                          n2, c - dx, c - dy, rng.uniform(0.1, 0.5), rng.uniform(0, np.pi)))
        # faint tidal bridge
        base = base + _sersic((px, px), 0.15, 0.5 * sep, 1.0, c, c,
                              0.8, ang) * 0.5
        colour = (1.0, 1.0, 1.0)
    else:
        raise ValueError(label)

    cube = np.empty((3, px, px), np.float32)
    for ch in range(3):
        layer = base * colour[ch]
        layer = _psf_blur(layer, psf_sigma[ch])
        cube[ch] = layer

    # faint background sources
    for _ in range(rng.integers(0, 4)):
        cube += _psf_blur(
            _sersic((px, px), rng.uniform(0.04, 0.15), rng.uniform(1, 3), 1.0,
                    rng.uniform(0, px), rng.uniform(0, px), 0.2, 0.0),
            1.3)[None] * rng.uniform(0.5, 1.0, size=(3, 1, 1))
    # Noise: a shared luminance component (correlated across bands) plus a
    # small independent per-band part — avoids unrealistic rainbow static.
    shared = rng.normal(0, 0.008, size=(px, px)).astype(np.float32)
    cube += shared[None]
    cube += rng.normal(0, 0.003, size=cube.shape).astype(np.float32)
    return np.clip(cube, 0, None)


def _synthetic_redshift(label: str, rng: np.random.Generator) -> float:
    """Plausible redshift for the demo readout (irregulars skew high)."""
    base = {"disk": 1.5, "spheroid": 1.0, "irregular": 4.0,
            "point_source": 2.0, "merger": 2.5, "compact": 3.5}.get(label, 2.0)
    return float(np.clip(rng.normal(base, 1.2), 0.2, 12.0))


def generate_synthetic_dataset(n_per_class: int = 170, seed: int = 7,
                               px: int | None = None) -> list[CutoutRecord]:
    px = px or config.CUTOUT_PX
    rng = np.random.default_rng(seed)
    records: list[CutoutRecord] = []
    for label in CLASSES:
        for i in range(n_per_class):
            raw = _render_galaxy(label, rng, px)
            stretched = asinh_stretch(raw)
            rec = _save_record(
                f"syn_{label}_{i:04d}", label, "synthetic",
                ra=float(rng.uniform(214.7, 215.1)),   # nominal EGS/CEERS field
                dec=float(rng.uniform(52.7, 53.1)),
                redshift=_synthetic_redshift(label, rng),
                stretched=stretched,
            )
            records.append(rec)
    write_manifest(records)
    return records


# ---------------------------------------------------------------------------
# Real CEERS/JADES cutouts (Cutout2D) — runs when mosaics are provided
# ---------------------------------------------------------------------------
def extract_real_cutouts(catalog_rows: list[dict],
                         mosaic_paths: dict[str, str | Path],
                         px: int | None = None) -> list[CutoutRecord]:
    """Cut real NIRCam thumbnails at catalogue positions.

    ``catalog_rows`` need ``ra``, ``dec``, ``label`` (and optionally ``redshift``,
    ``id``). ``mosaic_paths`` maps each band in ``config.BANDS`` to a FITS mosaic
    on disk. This is the production path; it is exercised once a user drops the
    public CEERS/JADES mosaics into ``data/mosaics/``.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    px = px or config.CUTOUT_PX
    bands = config.BANDS
    missing = [b for b in bands if b not in mosaic_paths]
    if missing:
        raise ValueError(f"Missing mosaics for bands: {missing}")

    # Open each band once (mosaics are large).
    layers = {}
    for b in bands:
        hdul = fits.open(mosaic_paths[b], memmap=True)
        sci = next(h for h in hdul if h.data is not None and h.data.ndim == 2)
        layers[b] = (sci.data, WCS(sci.header))

    records: list[CutoutRecord] = []
    for i, row in enumerate(catalog_rows):
        label = row["label"]
        if label not in CLASS_TO_IDX:
            continue
        pos = SkyCoord(float(row["ra"]), float(row["dec"]), unit=u.deg)
        try:
            cube = np.stack([
                Cutout2D(layers[b][0], pos, (px, px), wcs=layers[b][1]).data
                for b in bands
            ]).astype(np.float32)
        except Exception:
            continue  # source off the mosaic edge / no coverage
        if cube.shape != (3, px, px):
            continue
        stretched = asinh_stretch(np.nan_to_num(cube))
        rec = _save_record(
            str(row.get("id", f"ceers_{i:05d}")), label, "ceers",
            ra=float(row["ra"]), dec=float(row["dec"]),
            redshift=float(row.get("redshift", 0.0)), stretched=stretched)
        records.append(rec)
    write_manifest(records)
    return records


# ---------------------------------------------------------------------------
# Real CEERS cutouts via the DJA grizli service (no mosaic download)
# ---------------------------------------------------------------------------
def fetch_service_cube(ra: float, dec: float, size: float = 3.0,
                       retries: int = 3, timeout: int = 30) -> np.ndarray | None:
    """Fetch a (3, H, W) raw NIRCam cube (R=F444W, G=F356W, B=F200W) by RA/Dec.

    Returns ``None`` if the position has no coverage or the service fails.
    """
    import requests
    from astropy.io import fits

    url = (f"{CUTOUT_SERVICE}?ra={ra:.6f}&dec={dec:.6f}&size={size}"
           f"&filters={SERVICE_FILTERS}&output=fits")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200 or not r.content:
                raise IOError(f"status {r.status_code}")
            with fits.open(io.BytesIO(r.content)) as hdul:
                planes = [h.data for h in hdul if h.data is not None and h.data.ndim == 2]
            if len(planes) < 3:
                return None
            shapes = {p.shape for p in planes[:3]}
            if len(shapes) != 1:
                return None
            return np.stack(planes[:3]).astype(np.float32)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def build_dataset_from_service(catalog_rows: list[dict], *, size: float = 3.0,
                               max_workers: int = 6, resume: bool = True) -> list[CutoutRecord]:
    """Fill the cutout cache from the DJA service for a labelled catalogue.

    ``catalog_rows`` need ``id``, ``ra``, ``dec``, ``label`` (and optionally
    ``redshift``). Resumes by skipping ids already cached, so a flaky run can be
    re-run. Writes the manifest over all successfully cached rows.
    """
    rows = [r for r in catalog_rows if r["label"] in CLASS_TO_IDX]

    def _one(row):
        rec_id = str(row["id"])
        npy_path = config.CUTOUTS_DIR / f"{rec_id}.npy"
        if resume and npy_path.exists():
            return CutoutRecord(rec_id, row["label"], CLASS_TO_IDX[row["label"]],
                                "ceers", float(row["ra"]), float(row["dec"]),
                                float(row.get("redshift", 0.0) or 0.0),
                                f"{rec_id}.npy", f"{rec_id}.png")
        cube = fetch_service_cube(float(row["ra"]), float(row["dec"]), size=size)
        if cube is None:
            return None
        stretched = asinh_stretch(np.nan_to_num(cube))
        return _save_record(rec_id, row["label"], "ceers",
                            float(row["ra"]), float(row["dec"]),
                            float(row.get("redshift", 0.0) or 0.0), stretched)

    records: list[CutoutRecord] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, r): r for r in rows}
        for fut in as_completed(futs):
            done += 1
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            if done % 25 == 0:
                print(f"  fetched {done}/{len(rows)} "
                      f"({len(records)} cached)", flush=True)
    records.sort(key=lambda r: r.id)
    write_manifest(records)
    return records
