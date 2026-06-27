# Lynceus — roadmap

A serious research core + interactive browser demos, on JWST deep-field galaxies.
Sibling to Chiron / Empedocles / Plutus / Agora. Arc: **"B built on A"**.

## Milestones

- **M1 — Morphology on-ramp (ViT). _Built, on real data._**
  A `vit_small_patch16_224` (timm, ImageNet-pretrained) fine-tuned to classify galaxy
  morphology (disk / spheroid / compact) on **real JWST/CEERS NIRCam cutouts**, evaluated by
  stratified k-fold cross-validation, exported to INT8 ONNX and shipped as a live in-browser
  classifier with attention-rollout heatmaps. Cutouts come from the DJA grizli cutout
  service; labels from single-Sérsic profile fits. A synthetic Sérsic generator and a
  local-mosaic `Cutout2D` path are retained as fallbacks.

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

- **Imaging (M1):** real JWST/NIRCam CEERS cutouts (F200W/F356W/F444W) from the **DAWN JWST
  Archive** grizli cutout service (`grizli-cutout.herokuapp.com`), fetched by RA/Dec — no
  mosaic download.
- **Labels (M1):** single-Sérsic profile fits from the **DJA morphology catalogue**
  (van der Wel et al. 2025, arXiv:2505.21622; `ceers-full-grizli-v7.2_morpho-phot.fits.gz`).
  Classes: disk (n < 1.2), spheroid (n > 2.5), compact (R_eff < 0.09″); F200W mag < 25,
  good-fit flags only.
- **Imaging (M2):** JADES DR5 NIRCam mosaics + photometric catalogue, MAST HLSP `jades`
  (Johnson et al. 2026, arXiv:2601.15954; Robertson et al. 2026, arXiv:2601.15956).
- The 375 MB catalogue and the cutout cache are gitignored; only small web assets + the
  quantised model are committed.
- *Alternative labels considered:* the Kartaltepe et al. 2023 CEERS VISUAL catalogue (true
  visual disk/sph/irregular/merger flags) is the richer scheme but is only published as a
  journal MRT (not on Vizier/arXiv source), so it's deferred.

## Device

Auto-selects Apple **MPS** / CUDA / CPU (`core/config.get_device`). Developed on an M4 Mac
mini (10-core GPU, 24 GB unified). M1 trains locally in minutes; M2 may want a cloud GPU.

## Pipeline

```
experiments/fetch_ceers.py     # real CEERS cutouts (Sersic labels + DJA service)
experiments/fetch_data.py      # synthetic fallback OR local-mosaic Cutout2D cache
experiments/train_vit.py       # k-fold CV metrics + final model + metrics.json
experiments/export_onnx.py     # vit.pt -> INT8 web/morphology/model.onnx
experiments/build_web_assets.py# gallery PNGs + attention maps + gallery/labels/metrics json
```
