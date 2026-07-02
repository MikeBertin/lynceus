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
    ap.add_argument("--cutouts-dir", default=str(config.ATLAS_CUTOUTS),
                    help="cutout cache to train on (e.g. the M3b cutouts_m3b)")
    ap.add_argument("--out", default="ssl_encoder.pt",
                    help="checkpoint name under models/")
    args = ap.parse_args()

    from pathlib import Path
    cutdir = Path(args.cutouts_dir)
    meta = config.ATLAS_DIR / "atlas_meta.csv"
    rows = list(csv.DictReader(open(meta)))
    paths = [cutdir / f"{r['id']}.npy" for r in rows]
    paths = [p for p in paths if p.exists()]
    device = config.get_device()
    print(f"Training SimCLR on {len(paths)} unlabelled cutouts from {cutdir} | device={device}")

    encoder = train_simclr(paths, epochs=args.epochs, batch_size=args.batch_size,
                           device=device, backbone=args.backbone)
    out = config.MODELS_DIR / args.out
    torch.save({"backbone": args.backbone, "state_dict": encoder.state_dict()}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
