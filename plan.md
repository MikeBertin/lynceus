# Lynceus — roadmap

A serious research core + interactive browser demos, on JWST deep-field galaxies.
Sibling to Chiron / Empedocles / Plutus / Agora. Arc: **"B built on A"**.

## Milestones

- **M1 — Morphology on-ramp (ViT). _Built._**
  A `vit_small_patch16_224` (timm, ImageNet-pretrained) fine-tuned to classify galaxy
  morphology, evaluated by stratified k-fold cross-validation, exported to INT8 ONNX and
  shipped as a live in-browser classifier with attention-rollout heatmaps.
  Default dataset is a physically-motivated **synthetic Sérsic set** so the pipeline is
  reproducible offline; the **real CEERS/JADES `Cutout2D` path is implemented** and runs
  once mosaics are supplied (see Data).

- **M2 — The atlas (self-supervised).** _Next._
  DINO/SimCLR-style encoder on unlabeled JADES cutouts → embeddings → UMAP. `web/atlas/`:
  a precomputed-JSON + sprite-sheet fly-through coloured by morphology / redshift. The
  M2-scale run is the one to move to Colab / a rented GPU.

- **M3 — The anomaly hunt.** _Research._
  Latent-space outlier scoring (kNN density / isolation forest); surface **Little Red
  Dots** and other rare objects; validate against published LRD catalogues.

- **M4 — The frontier.** Photo-z dropout hunting (z>10 / "too many bright early galaxies"
  ΛCDM tension), or image+spectra contrastive on NIRSpec (AstroCLIP lineage).

## Data provenance

- **Labels (real M1):** CEERS **VISUAL** morphology catalogue — Kartaltepe et al. 2023,
  ApJL 946, L15 (~850 galaxies at z≥3: disk / spheroid / irregular / point-source /
  merger flags).
- **Imaging (real M1):** CEERS NIRCam mosaics, MAST HLSP `ceers`.
- **Imaging (M2):** JADES DR5 NIRCam mosaics + photometric catalogue, MAST HLSP `jades`
  (Johnson et al. 2026, arXiv:2601.15954; Robertson et al. 2026, arXiv:2601.15956).
- Raw FITS mosaics are multi-GB → `data/` is gitignored; only small cutouts + the web
  model are kept.

## Device

Auto-selects Apple **MPS** / CUDA / CPU (`core/config.get_device`). Developed on an M4 Mac
mini (10-core GPU, 24 GB unified). M1 trains locally in minutes; M2 may want a cloud GPU.

## Pipeline

```
experiments/fetch_data.py      # synthetic (default) OR real Cutout2D cache
experiments/train_vit.py       # k-fold CV metrics + final model + metrics.json
experiments/export_onnx.py     # vit.pt -> INT8 web/morphology/model.onnx
experiments/build_web_assets.py# gallery PNGs + attention maps + gallery/labels/metrics json
```
