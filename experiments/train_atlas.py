"""Train the SimCLR self-supervised encoder on the unlabelled atlas cutouts.

    python -m experiments.train_atlas --epochs 100
"""
from __future__ import annotations

import argparse
import csv

import torch

from core import config
from core.ssl import train_simclr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--backbone", default="resnet18")
    args = ap.parse_args()

    meta = config.ATLAS_DIR / "atlas_meta.csv"
    rows = list(csv.DictReader(open(meta)))
    paths = [config.ATLAS_CUTOUTS / f"{r['id']}.npy" for r in rows]
    paths = [p for p in paths if p.exists()]
    device = config.get_device()
    print(f"Training SimCLR on {len(paths)} unlabelled cutouts | device={device}")

    encoder = train_simclr(paths, epochs=args.epochs, batch_size=args.batch_size,
                           device=device, backbone=args.backbone)
    out = config.MODELS_DIR / "ssl_encoder.pt"
    torch.save({"backbone": args.backbone, "state_dict": encoder.state_dict()}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
