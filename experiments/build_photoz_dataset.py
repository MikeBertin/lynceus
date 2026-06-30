"""Assemble the photo-z training/validation set from the CEERS grizli catalogue.

Pulls the PHOTOZ_BANDS aperture fluxes + errors and the redshift columns from
``data/ceers_morpho.fits.gz`` (DJA / grizli + EAZY photo-z), applies light
quality cuts, and writes a compact ``data/photoz/photoz.npz`` so the trainer
never has to reopen the 375 MB FITS.

Split philosophy (kept honest):
  * objects WITH a spectroscopic redshift (z_spec > 0) are reserved entirely for
    VALIDATION against the truth — they are never trained on;
  * everything else with a good template photo-z (z_phot) is TRAINING data.

    python -m experiments.build_photoz_dataset
"""
from __future__ import annotations

import numpy as np

from core import config
from core.photoz import PHOTOZ_BANDS, BAD, SN_MIN

CATALOG = config.DATA_DIR / "ceers_morpho.fits.gz"
OUT_DIR = config.DATA_DIR / "photoz"
APER = "aper_1"            # consistent medium aperture -> matched colours


def main() -> None:
    from astropy.io import fits
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = fits.open(CATALOG, memmap=True)[1].data
    cols = set(d.columns.names)

    def col(name, default=np.nan):
        return np.asarray(d[name], np.float64) if name in cols else np.full(len(d), default)

    flux = np.stack([col(f"{b}_flux_{APER}", BAD - 1) for b in PHOTOZ_BANDS], axis=1)
    err = np.stack([col(f"{b}_fluxerr_{APER}", -1.0) for b in PHOTOZ_BANDS], axis=1)
    flux = np.nan_to_num(flux, nan=BAD - 1)
    err = np.nan_to_num(err, nan=-1.0)

    z_phot = col("z_phot"); z_spec = col("z_spec")
    risk = col("z_phot_risk"); nusefilt = col("nusefilt")
    ids = col("id"); ra = col("ra"); dec = col("dec")

    # detections per galaxy
    valid = (flux > BAD) & (err > 0)
    sn = np.where(valid, flux / np.where(err > 0, err, 1), 0.0)
    ndet = (valid & (sn >= SN_MIN)).sum(axis=1)

    # quality: enough bands used in the fit, enough real detections
    quality = (nusefilt >= 5) & (ndet >= 3)

    has_spec = np.isfinite(z_spec) & (z_spec > 0)
    train_ok = quality & (~has_spec) & np.isfinite(z_phot) & (z_phot > 0) & (z_phot < 16)
    val_ok = quality & has_spec

    keep = train_ok | val_ok
    print(f"catalogue rows         : {len(d)}")
    print(f"quality (nusefilt>=5, >=3 det): {int(quality.sum())}")
    print(f"  training (z_phot, no spec)  : {int(train_ok.sum())}")
    print(f"  validation (z_spec truth)   : {int(val_ok.sum())}")
    for lo, hi in [(0, 4), (4, 6), (6, 8), (8, 10), (10, 16)]:
        nt = int((train_ok & (z_phot >= lo) & (z_phot < hi)).sum())
        nv = int((val_ok & (z_spec >= lo) & (z_spec < hi)).sum())
        print(f"    z {lo:2d}-{hi:2d}: train={nt:6d}  val(spec)={nv:4d}")

    out = OUT_DIR / "photoz.npz"
    np.savez_compressed(
        out,
        bands=np.array(PHOTOZ_BANDS),
        flux=flux[keep].astype(np.float32), err=err[keep].astype(np.float32),
        z_phot=z_phot[keep].astype(np.float32), z_spec=z_spec[keep].astype(np.float32),
        risk=risk[keep].astype(np.float32), nusefilt=nusefilt[keep].astype(np.float32),
        is_train=train_ok[keep], is_val=val_ok[keep],
        id=ids[keep].astype(np.int64), ra=ra[keep], dec=dec[keep],
    )
    print(f"\nWrote {out}  ({int(keep.sum())} galaxies)")


if __name__ == "__main__":
    main()
