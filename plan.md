# Lynceus — roadmap

A serious research core + interactive browser demos, on JWST deep-field galaxies.
Sibling to Chiron / Empedocles / Plutus / Agora. Arc: **"B built on A"**.

## Milestones

- **M1 — Morphology on-ramp (ViT). _Built, on real data + real human labels._**
  A `vit_small_patch16_224` (timm, ImageNet-pretrained) fine-tuned to classify galaxy
  morphology (featured / smooth / merger) on **real JWST NIRCam cutouts** with **real Galaxy
  Zoo visual labels**, evaluated by stratified k-fold cross-validation, exported to INT8 ONNX
  and shipped as a live in-browser classifier with attention-rollout heatmaps. Cutouts come
  from the DJA grizli cutout service; labels from Galaxy Zoo: CANDELS volunteer votes. A
  Sérsic-fit-labelled CEERS path, a synthetic generator and a local-mosaic `Cutout2D` path
  are retained as alternatives/fallbacks.

- **M2 — The atlas (self-supervised). _Built._**
  SimCLR (ResNet-18, NT-Xent) trained on ~2,400 unlabelled JWST cutouts (Apple MPS); the
  encoder's 512-d features → UMAP 2-D. `web/atlas/`: a precomputed-JSON + sprite-sheet
  fly-through (pan/zoom canvas) coloured by Galaxy Zoo morphology. The labels were never used
  in training, yet a kNN probe of the embeddings recovers the dominant class at **58.5% vs
  33% baseline** — morphology emerges on its own.

- **M3 — The anomaly hunt.** _Research._
  Latent-space outlier scoring (kNN density / isolation forest); surface **Little Red
  Dots** and other rare objects; validate against published LRD catalogues.

- **M4 — The frontier.** Photo-z dropout hunting (z>10 / "too many bright early galaxies"
  ΛCDM tension), or image+spectra contrastive on NIRSpec (AstroCLIP lineage).

## Data provenance

- **Imaging (M1):** real JWST/NIRCam cutouts (F200W/F356W/F444W of GOODS-S, COSMOS, UDS) from
  the **DAWN JWST Archive** grizli cutout service (`grizli-cutout.herokuapp.com`), fetched by
  RA/Dec — no mosaic download.
- **Labels (M1):** **Galaxy Zoo: CANDELS** volunteer visual classifications (Simmons et al.
  2017; `gz_candels_table_2_main_release.fits`). Classes from the top of the decision tree:
  featured (t00 features > 0.5), smooth (t00 smooth > 0.6), merger (t16 merging > 0.5);
  ≥ 20 classifiers/galaxy. Galaxy Zoo judged HST imaging — cross-matched here to the JWST view.
- *Alt labels (M1b):* single-Sérsic profile fits from the **DJA morphology catalogue**
  (van der Wel et al. 2025, arXiv:2505.21622) via `fetch_ceers.py` — disk/spheroid/compact.
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
experiments/fetch_gz.py        # real JWST cutouts + Galaxy Zoo visual labels (primary)
experiments/fetch_ceers.py     # alt: real cutouts + Sersic-fit labels (CEERS)
experiments/fetch_data.py      # synthetic fallback OR local-mosaic Cutout2D cache
experiments/train_vit.py       # k-fold CV metrics + final model + metrics.json
experiments/export_onnx.py     # vit.pt -> INT8 web/morphology/model.onnx
experiments/build_web_assets.py# gallery PNGs + attention maps + gallery/labels/metrics json
```
