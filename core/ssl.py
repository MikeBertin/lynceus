"""Self-supervised contrastive learning (SimCLR) for the M2 atlas.

A ResNet-18 backbone is trained with the NT-Xent contrastive loss on two
augmented views of each unlabelled JWST cutout. No morphology labels are used:
the encoder learns to group galaxies that *look alike*, and the Galaxy Zoo votes
are only used afterwards to colour the atlas.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from . import config

SSL_PX = 128
_MEAN = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Augmentations (two stochastic views per image)
# ---------------------------------------------------------------------------
def _augment(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One stochastic view of a (3, H, W) float cutout, returned at SSL_PX."""
    a = arr
    # random resized crop (scale 0.6-1.0)
    H, W = a.shape[1:]
    s = rng.uniform(0.6, 1.0)
    ch, cw = int(H * s), int(W * s)
    y0 = int(rng.integers(0, H - ch + 1)); x0 = int(rng.integers(0, W - cw + 1))
    a = a[:, y0:y0 + ch, x0:x0 + cw]
    # flips + 90deg rotations (galaxies have no preferred orientation)
    if rng.random() < 0.5: a = a[:, :, ::-1]
    if rng.random() < 0.5: a = a[:, ::-1, :]
    k = int(rng.integers(0, 4))
    if k: a = np.rot90(a, k, axes=(1, 2))
    a = np.ascontiguousarray(a)
    t = torch.from_numpy(a).float().unsqueeze(0)
    t = F.interpolate(t, size=(SSL_PX, SSL_PX), mode="bilinear", align_corners=False)[0]
    # photometric jitter: brightness + contrast (luminance) are jittered freely,
    # but per-channel *colour* jitter is kept gentle (+-8%) because the cutouts now
    # preserve real colour (asinh_stretch colour=True) and we want the encoder to
    # treat colour as signal, not nuisance. Too much colour jitter would teach it
    # to ignore the very redness that distinguishes Little Red Dots.
    t = t * rng.uniform(0.8, 1.2)
    mean = t.mean()
    t = (t - mean) * rng.uniform(0.8, 1.2) + mean
    t = t * torch.tensor(rng.uniform(0.92, 1.08, size=3), dtype=torch.float32).view(3, 1, 1)
    t = t.clamp(0, 1)
    return ((t - _MEAN) / _STD).numpy()


class TwoViewDataset(Dataset):
    def __init__(self, npy_paths, seed: int = 0):
        self.paths = list(npy_paths)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = np.load(self.paths[i]).astype(np.float32)
        return (torch.from_numpy(_augment(arr, self.rng)),
                torch.from_numpy(_augment(arr, self.rng)))


def embed_tensor(arr: np.ndarray) -> torch.Tensor:
    """Deterministic (no-aug) preprocessing for inference/embedding."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
    t = F.interpolate(t, size=(SSL_PX, SSL_PX), mode="bilinear", align_corners=False)[0]
    return (t - _MEAN) / _STD


# ---------------------------------------------------------------------------
# Model + loss
# ---------------------------------------------------------------------------
class SimCLR(nn.Module):
    def __init__(self, backbone: str = "resnet18", proj_dim: int = 128):
        import timm
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat = self.encoder.num_features
        self.feat_dim = feat
        self.projector = nn.Sequential(
            nn.Linear(feat, feat), nn.ReLU(inplace=True), nn.Linear(feat, proj_dim))

    def forward(self, x):
        h = self.encoder(x)
        return h, F.normalize(self.projector(h), dim=1)


def nt_xent(z1, z2, temperature: float = 0.2):
    """Normalised temperature-scaled cross-entropy over a 2N batch."""
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)                      # (2N, d)
    sim = (z @ z.t()) / temperature
    eye = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -9e15)               # out-of-place (autograd-safe)
    targets = torch.arange(2 * N, device=z.device)
    targets = (targets + N) % (2 * N)               # positive is the other view
    return F.cross_entropy(sim, targets)


def train_simclr(npy_paths, *, epochs=100, batch_size=256, lr=1e-3,
                 device=None, backbone="resnet18", verbose=True):
    device = device or config.get_device()
    model = SimCLR(backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TwoViewDataset(npy_paths), batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    for ep in range(epochs):
        model.train(); tot = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1); _, z2 = model(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep+1}/{epochs}  ntxent={tot/max(len(loader),1):.3f}", flush=True)
    return model.encoder.eval()
