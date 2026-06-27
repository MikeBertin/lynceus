"""Paths, constants and device selection for Lynceus."""
from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CUTOUTS_DIR = DATA_DIR / "cutouts"          # cached .npy cutouts + manifest.csv
MOSAIC_DIR = DATA_DIR / "mosaics"           # user-supplied real NIRCam FITS mosaics
MODELS_DIR = ROOT / "models"                # checkpoints + onnx (gitignored)
WEB_DIR = ROOT / "web"
WEB_MORPH_DIR = WEB_DIR / "morphology"

for _d in (DATA_DIR, CUTOUTS_DIR, MOSAIC_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------
# Three NIRCam filters mapped to an RGB composite, so cutouts look like the
# JWST colour images people recognise (red = long wavelength).
BANDS = ("F444W", "F356W", "F200W")   # (R, G, B)
CUTOUT_PX = 96                        # stored cutout size (pixels)
MODEL_PX = 224                        # ViT input size (cutouts resized up)

# ImageNet stats — the pretrained ViT backbone expects these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    """Pick the best available backend: Apple MPS, CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
