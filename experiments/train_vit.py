"""Cross-validate the ViT morphology classifier, then train a final model.

    python -m experiments.train_vit --folds 5 --epochs 6
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from core import config, morphology
from core.data import load_manifest
from core.train import cross_validate, train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_manifest()
    device = config.get_device()
    print(f"Loaded {len(rows)} cutouts | device={device} | "
          f"source={rows[0]['source']}")

    # ---- honest cross-validated evaluation ---------------------------------
    y_true, y_pred = cross_validate(
        rows, k=args.folds, epochs=args.epochs,
        batch_size=args.batch_size, device=device, seed=args.seed)

    rep = morphology.report(y_true, y_pred)
    cm = morphology.confusion(y_true, y_pred)
    baseline = morphology.majority_baseline_accuracy(y_true)
    acc = float((y_true == y_pred).mean())

    print("\n=== Cross-validated metrics ===")
    print(f"accuracy            {acc:.3f}")
    print(f"majority baseline   {baseline:.3f}")
    print(f"macro F1            {rep['macro avg']['f1-score']:.3f}")
    print("\nper-class (precision / recall / f1 / support):")
    for c in morphology.CLASSES:
        r = rep[c]
        print(f"  {c:13s} {r['precision']:.2f}  {r['recall']:.2f}  "
              f"{r['f1-score']:.2f}  ({int(r['support'])})")
    print("\nconfusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{c[:5]:>5}" for c in morphology.CLASSES))
    for i, c in enumerate(morphology.CLASSES):
        print(f"  {c[:5]:>5} " + " ".join(f"{v:5d}" for v in cm[i]))

    # ---- final model on all data, for export -------------------------------
    print("\nTraining final model on all data...")
    model = train_model(rows, epochs=args.epochs, batch_size=args.batch_size,
                        device=device, verbose=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), config.MODELS_DIR / "vit.pt")

    metrics = {
        "n": len(rows),
        "source": rows[0]["source"],
        "accuracy": acc,
        "majority_baseline": baseline,
        "macro_f1": rep["macro avg"]["f1-score"],
        "per_class": {c: rep[c] for c in morphology.CLASSES},
        "confusion": cm.tolist(),
        "classes": list(morphology.CLASSES),
        "folds": args.folds,
        "epochs": args.epochs,
    }
    with open(config.MODELS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved {config.MODELS_DIR/'vit.pt'} and metrics.json")


if __name__ == "__main__":
    main()
