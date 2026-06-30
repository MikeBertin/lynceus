"""Train the photo-z network and validate it against spectroscopic truth.

    python -m experiments.train_photoz --epochs 60

Trains on EAZY template redshifts (z_phot, ~70k galaxies) and reports accuracy
on a fully held-out set of real spectroscopic redshifts (z_spec, ~1.8k) — the
honest number. Saves models/photoz.pt (+ feature standardisation) and
models/photoz_metrics.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from core import config, photoz
from core.photoz import (PhotoZNet, featurize, soft_labels, pdf_from_logits,
                         point_estimates, photoz_metrics, Z_CENTRES)

DATA = config.DATA_DIR / "photoz" / "photoz.npz"


def standardise_fit(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = config.get_device()

    z = np.load(DATA, allow_pickle=True)
    X = featurize(z["flux"], z["err"])
    is_train = z["is_train"]; is_val = z["is_val"]
    z_phot = z["z_phot"]; z_spec = z["z_spec"]

    Xtr, ytr = X[is_train], z_phot[is_train]              # train on template z
    Xva, yva = X[is_val], z_spec[is_val]                  # validate on spec truth
    mu, sd = standardise_fit(Xtr)
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd

    print(f"device={device}  train={len(Xtr)}  val(spec)={len(Xva)}  feat_dim={X.shape[1]}")

    soft = soft_labels(ytr)
    ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(soft))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = PhotoZNet(in_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Xva_t = torch.from_numpy(Xva_s).to(device)

    def evaluate():
        model.eval()
        with torch.no_grad():
            pdf = pdf_from_logits(model(Xva_t)).cpu().numpy()
        pe = point_estimates(pdf)
        return (photoz_metrics(pe["z_peak"], yva),
                photoz_metrics(pe["z_mean"], yva))

    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for xb, tb in dl:
            xb, tb = xb.to(device), tb.to(device)
            logp = F.log_softmax(model(xb), dim=1)
            loss = -(tb * logp).sum(1).mean()            # soft-label cross-entropy
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            m_mode, m_mean = evaluate()
            print(f"  epoch {ep+1:3d}/{args.epochs}  loss={tot/len(dl):.3f}  "
                  f"val(peak) {m_mode}", flush=True)

    m_mode, m_mean = evaluate()
    print("\n=== held-out spectroscopic validation ===")
    print("peak (refined):", m_mode)
    print("mean          :", m_mean)

    # by redshift slice (mode estimate)
    model.eval()
    with torch.no_grad():
        pdf = pdf_from_logits(model(Xva_t)).cpu().numpy()
    zmode = point_estimates(pdf)["z_peak"]
    slices = {}
    print("\nby spec-z slice (peak estimate):")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 12)]:
        m = (yva >= lo) & (yva < hi)
        if m.sum() > 3:
            mm = photoz_metrics(zmode[m], yva[m])
            slices[f"{lo}-{hi}"] = {"n": mm.n, "sigma_nmad": mm.sigma_nmad,
                                    "outlier_frac": mm.outlier_frac}
            print(f"  z {lo:2d}-{hi:2d}: {mm}")

    out = config.MODELS_DIR / "photoz.pt"
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "z_centres": Z_CENTRES.astype(np.float32),
                "bands": list(photoz.PHOTOZ_BANDS), "feat_dim": int(X.shape[1])}, out)
    # EAZY template photo-z vs the same spec truth = the ceiling our emulator
    # is distilling. Report it so the demo can be honest about what "good" means.
    base = photoz_metrics(z_phot[is_val], yva)
    print(f"\nEAZY template baseline (z_phot vs z_spec): {base}")
    metrics = {
        "n_train": int(len(Xtr)), "n_val_spec": int(len(Xva)),
        "sigma_nmad": m_mode.sigma_nmad, "outlier_frac": m_mode.outlier_frac,
        "bias": m_mode.bias, "by_slice": slices,
        "eazy_baseline": {"sigma_nmad": base.sigma_nmad,
                          "outlier_frac": base.outlier_frac, "bias": base.bias},
    }
    (config.MODELS_DIR / "photoz_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out} + photoz_metrics.json")


if __name__ == "__main__":
    main()
