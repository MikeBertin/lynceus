"""Build the demo gallery: example cutouts, attention heatmaps, and JSON.

    python -m experiments.build_web_assets --per-class 6
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import torch

from core import config, morphology
from core.data import load_manifest, load_cutout
from core.datasets import preprocess
from core.models import build_model, AttentionRollout

GALLERY_DIR = config.WEB_MORPH_DIR / "gallery"
GALLERY_PX = 224   # display size for gallery PNGs. The stored cutouts are
                   # native-resolution JWST (0.04"/px), so this is a LANCZOS
                   # upscale: no new detail, but far cleaner on retina than
                   # letting the browser stretch a 120 px image ~3x. Matches
                   # the ViT input size; classification still uses the npy.


def _attn_png(heat: np.ndarray, out_path) -> None:
    """Render a (g, g) attention map as a magma-coloured PNG at gallery size."""
    import matplotlib
    from PIL import Image
    h = heat - heat.min()
    h = h / (h.max() + 1e-8)
    img = Image.fromarray((h * 255).astype(np.uint8)).resize(
        (GALLERY_PX, GALLERY_PX), Image.BICUBIC)
    h = np.asarray(img) / 255.0
    rgba = (matplotlib.colormaps["magma"](h) * 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    from PIL import Image

    rows = load_manifest()
    ckpt = config.MODELS_DIR / "vit.pt"
    if not ckpt.exists():
        raise SystemExit("No models/vit.pt; run experiments.train_vit first.")

    model = build_model(pretrained=False).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    rollout = AttentionRollout(model)

    if GALLERY_DIR.exists():
        shutil.rmtree(GALLERY_DIR)
    GALLERY_DIR.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    by_class: dict[str, list[dict]] = {c: [] for c in morphology.CLASSES}
    for r in rows:
        by_class[r["label"]].append(r)

    gallery = []
    for label, items in by_class.items():
        picks = rng.choice(len(items), min(args.per_class, len(items)),
                           replace=False)
        for idx in picks:
            r = items[int(idx)]
            arr = load_cutout(r["npy"]).astype(np.float32)
            x = preprocess(arr).unsqueeze(0)
            logits, heat = rollout.heatmap(x)
            probs = torch.softmax(logits, 1)[0].detach().numpy()

            Image.open(config.CUTOUTS_DIR / r["png"]).resize(
                (GALLERY_PX, GALLERY_PX), Image.LANCZOS).save(
                GALLERY_DIR / f"{r['id']}.png")
            _attn_png(heat[0].numpy(), GALLERY_DIR / f"{r['id']}_attn.png")
            gallery.append({
                "id": r["id"],
                "label": label,
                "label_name": morphology.CLASS_LABELS[label],
                "redshift": round(float(r["redshift"]), 2),
                "img": f"gallery/{r['id']}.png?v=2",     # ?v bumps with GALLERY_PX
                "attn": f"gallery/{r['id']}_attn.png?v=2",
                "probs": {c: round(float(probs[i]), 4)
                          for i, c in enumerate(morphology.CLASSES)},
            })
    rollout.remove()

    (config.WEB_MORPH_DIR / "gallery.json").write_text(json.dumps(gallery, indent=1))

    labels = {
        "classes": list(morphology.CLASSES),
        "names": morphology.CLASS_LABELS,
        "blurbs": morphology.CLASS_BLURB,
        "mean": config.IMAGENET_MEAN,
        "std": config.IMAGENET_STD,
        "model_px": config.MODEL_PX,
        "bands": config.BANDS,
    }
    (config.WEB_MORPH_DIR / "labels.json").write_text(json.dumps(labels, indent=1))

    metrics_src = config.MODELS_DIR / "metrics.json"
    if metrics_src.exists():
        shutil.copyfile(metrics_src, config.WEB_MORPH_DIR / "metrics.json")

    print(f"Wrote {len(gallery)} gallery items + labels.json/metrics.json "
          f"to {config.WEB_MORPH_DIR}")


if __name__ == "__main__":
    main()
