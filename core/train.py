"""Device-agnostic training + k-fold cross-validation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from . import config
from .datasets import CutoutDataset
from .models import build_model


def _loader(rows, augment, batch_size, shuffle, seed=0):
    return DataLoader(CutoutDataset(rows, augment=augment, seed=seed),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(rows, *, epochs=6, lr=3e-4, batch_size=32, device=None,
                val_rows=None, verbose=True) -> nn.Module:
    """Fine-tune a fresh ViT on ``rows``; optionally report val accuracy/epoch."""
    device = device or config.get_device()
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    train_loader = _loader(rows, True, batch_size, True)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        if verbose:
            msg = f"  epoch {ep+1}/{epochs}  loss={running/len(rows):.3f}"
            if val_rows:
                msg += f"  val_acc={evaluate(model, val_rows, device)[0]:.3f}"
            print(msg, flush=True)
    return model


@torch.no_grad()
def predict(model, rows, device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in _loader(rows, False, 64, False):
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


@torch.no_grad()
def evaluate(model, rows, device):
    y_true = np.array([int(r["label_idx"]) for r in rows])
    y_pred = predict(model, rows, device)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return acc, y_true, y_pred


def cross_validate(rows, *, k=5, epochs=6, batch_size=32, device=None, seed=7):
    """Stratified k-fold. Returns out-of-fold (y_true, y_pred) over all rows."""
    device = device or config.get_device()
    y = np.array([int(r["label_idx"]) for r in rows])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_true = np.zeros(len(rows), dtype=int)
    oof_pred = np.zeros(len(rows), dtype=int)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(rows)), y)):
        print(f"[fold {fold+1}/{k}] train={len(tr)} val={len(va)}", flush=True)
        tr_rows = [rows[i] for i in tr]
        va_rows = [rows[i] for i in va]
        model = train_model(tr_rows, epochs=epochs, batch_size=batch_size,
                            device=device, verbose=True)
        _, yt, yp = evaluate(model, va_rows, device)
        oof_true[va] = yt
        oof_pred[va] = yp
    return oof_true, oof_pred
