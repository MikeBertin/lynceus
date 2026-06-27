"""Torch dataset over cached cutouts, with orientation-invariant augmentation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import config
from .data import load_cutout

_MEAN = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)


def _augment(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flips / 90-deg rotations / small shifts (galaxies have no 'up')."""
    if rng.random() < 0.5:
        arr = arr[:, :, ::-1]
    if rng.random() < 0.5:
        arr = arr[:, ::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        arr = np.rot90(arr, k, axes=(1, 2))
    sh, sw = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
    arr = np.roll(arr, (sh, sw), axis=(1, 2))
    return np.ascontiguousarray(arr)


def preprocess(arr: np.ndarray) -> torch.Tensor:
    """(3, H, W) float in [0,1] -> normalised (3, MODEL_PX, MODEL_PX) tensor.

    Mirrors exactly what the browser demo does to a PNG: resize to the model
    size, then apply ImageNet normalisation.
    """
    t = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
    t = F.interpolate(t, size=(config.MODEL_PX, config.MODEL_PX),
                      mode="bilinear", align_corners=False).squeeze(0)
    return (t - _MEAN) / _STD


class CutoutDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool = False, seed: int = 0):
        self.rows = rows
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        arr = load_cutout(row["npy"]).astype(np.float32)
        if self.augment:
            arr = _augment(arr, self.rng)
        return preprocess(arr), int(row["label_idx"])
