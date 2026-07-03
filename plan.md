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
  SimCLR (ResNet-18, NT-Xent) trained on **~9,700 unlabelled JWST cutouts** (Apple MPS); the
  encoder's 512-d features → UMAP 2-D. `web/atlas/`: a precomputed-JSON + sprite-sheet
  fly-through coloured by Galaxy Zoo morphology, **WebGL-rendered** (gl.POINTS for the cloud,
  a 2-D layer for thumbnails/overlays on zoom-in) so it stays smooth at ~10k+ points. The
  labels were never used in training, yet a kNN probe of the embeddings recovers the dominant
  class at **62% ± 0.7% (5-fold) vs 39% baseline** — morphology emerges on its own. (The globe also places the
  Sun for the current day, computed client-side.)

- **M3 — The anomaly hunt. _Built._**
  kNN latent-space anomaly scoring on the SSL embeddings (`core/anomaly.py`); the atlas gains
  an **anomaly** colour mode and a "18 weirdest galaxies" strip. **216 real Little Red Dots**
  (Kokorev et al. 2024, github.com/VasilyKokorev/lrd_phot) in GOODS-S/COSMOS/UDS are embedded
  with the same encoder and overlaid: they're **7.4× over-represented** among the top-10%
  anomalies (95% CI 6.8–7.9×, p<0.0001 vs a random-galaxy null; median LRD at the 96th
  percentile) — the encoder rediscovers them with no labels.
  The shipped atlas uses the **M3b point-source-aware representation**: a noise-aware asinh stretch
  (per-band sky subtraction; the shared intensity scaled by the cutout's own sky RMS, so empty sky
  stays dark instead of amplifying to colour static) plus a detection-anchored 64px crop. This
  sharpened the enrichment from ~5× (an earlier colour-preserving-but-percentile stretch) to 7.4×,
  and — crucially — the signal survives leave-one-field-out (the anomaly enrichment holds on a deep
  field the encoder never trained on; see RESEARCH.md Q4), so it is about galaxies, not survey depth.

- **M4 — The dropout hunter (photometric redshifts). _Built._**
  A small MLP (`core/photoz.py`) reads a galaxy's redshift from nine JWST/HST band fluxes, output as
  a **probability distribution over redshift bins** (so it shows a PDF and its degeneracies). Trained
  on ~71k EAZY template redshifts from the CEERS grizli catalogue and validated against **~1,800 real
  spectroscopic** redshifts it never saw: **σ_NMAD ≈ 0.040** (95% CI 0.037–0.044), within a hair
  of the EAZY template ceiling (0.027) it distils. Exported to ONNX and run **live in the browser** (`web/dropout/`): a
  filter scrubber shows the Lyman break sweeping redward and the galaxy dropping out of the bluer
  bands, alongside the redshift PDF, a Lyman-break colour–colour diagram, and a curated gallery —
  including the **low-z interloper degeneracy** behind the "too many bright early galaxies" tension.
  PDF calibration is checked honestly (PIT + central-interval coverage on the spec-z set): the PDFs
  are mildly over-dispersed but a 90% credible interval still contains the truth ~88% of the time, and
  since the catastrophic-outlier tails and the over-wide core pull a global temperature in opposite
  directions (NLL→1.5 widen vs PIT→0.24 sharpen), the net's own PDFs ship unscaled.
  (A further frontier: image+spectra contrastive on NIRSpec, AstroCLIP lineage.)

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
- **Photometry + redshifts (M4):** the **CEERS grizli / EAZY** photometric catalogue (van der Wel
  et al. 2025) — ~76k galaxies with NIRCam+HST aperture fluxes, EAZY template `z_phot`, and ~1,800
  real spectroscopic `z_spec`. Trained on the template redshifts; validated on the spectroscopic ones.
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

experiments/build_photoz_dataset.py # M4: CEERS fluxes+z -> data/photoz/photoz.npz
experiments/train_photoz.py         # M4: train photo-z MLP, validate on spec-z -> photoz.pt
experiments/export_photoz_onnx.py   # M4: photoz.pt -> web/dropout/photoz.onnx (+meta)
experiments/build_dropout_assets.py # M4: filmstrips + PDFs + colour-colour -> web/dropout/
experiments/calibrate_photoz.py     # M4: PIT + coverage on the spec-z PDFs (core/photoz);
                                    #   ships T=1, discloses calibration -> dropout.json

experiments/significance.py    # error bars: bootstrap CIs + permutation null on the
                               #   headline claims (core/stats.py) -> models/significance.json
                               #   + patches the CIs into the web JSON
```
