# Lynceus

**The keenest eyes of the Argonauts.**
*Pulling the faint and the hidden out of the James Webb deep sky.*

Machine learning on **JWST** deep-field galaxies: a serious research core (offline PyTorch)
paired with **self-contained, no-build browser demos**. A Vision Transformer learns what
galaxies *look like*, runs **live in your browser** via ONNX, and the road ahead leads to a
self-supervised **embedding atlas** and an **anomaly hunt** for the mysterious
**Little Red Dots**.

🔒 *Private while in progress — not yet published.*

Companion to [Chiron](https://mikebertin.github.io/chiron/) (computational physics),
[Empedocles](https://mikebertin.github.io/empedocles/) (evolutionary algorithms) and
Plutus (quantitative finance).

## The demos

| | | |
|---|---|---|
| **[Morphology](web/morphology/)** | *Live* | A `ViT-Small` in ONNX classifies a **real JWST** cutout in your browser — featured / smooth / merger (Galaxy Zoo labels) — with attention heatmaps and **honest, cross-validated** metrics. **Built.** |
| **[Atlas](web/atlas/)** | *Live* | Fly through a **self-supervised** (SimCLR) 2-D embedding of ~2,400 real JWST galaxies; colour by morphology and watch structure the encoder found with **no labels** emerge. **Built.** |
| **Anomaly hunt** | Discovery | Score every galaxy by how unusual it is, and surface the **Little Red Dots**. *Planned (M3).* |
| **Dropout hunter** | Cosmic dawn | Find galaxies at z>10 as they vanish from the bluer filters. *Planned (M4).* |

## The data, honestly

The demo trains on **real JWST/NIRCam cutouts with real human visual-morphology labels**:

- **Imagery — real JWST.** Cutouts (F200W + F356W + F444W of GOODS-S, COSMOS and UDS galaxies)
  are pulled by RA/Dec from the [DAWN JWST Archive](https://dawn-cph.github.io/dja/) grizli
  cutout service — actual JWST pixels, no mosaic download. The RGB is our asinh composite of
  the three raw bands.
- **Labels — real human votes.** The three classes are the canonical
  [Galaxy Zoo](https://data.galaxyzoo.org/): CANDELS split (Simmons et al. 2017):
  `featured` (disk/structured), `smooth` (elliptical/early-type) and `merger`
  (merging/disturbed), taken as the majority volunteer vote with ≥ 20 classifiers.
- **One honest caveat.** Galaxy Zoo classified the *HST/CANDELS* imaging; the ViT is shown the
  *JWST* view of the same galaxies (cross-matched by position). Morphology is largely
  consistent across the two, but the labels and the pixels come from different telescopes —
  stated plainly rather than hidden.
- A **synthetic Sérsic generator** is retained as a no-download fallback (and for CI/tests):
  `python -m experiments.fetch_data --synthetic 160`. A Sérsic-fit-labelled CEERS path
  (`fetch_ceers.py`) and a local-mosaic `Cutout2D` path (`fetch_data --real`) also exist.
  See [`plan.md`](plan.md) for provenance and the M2–M4 roadmap.

## Architecture

```
core/                 # reusable package
  config.py           #   paths, bands, device auto-select (mps/cuda/cpu)
  data.py             #   synthetic Sersic generator + real Cutout2D path + asinh stretch
  datasets.py         #   torch Dataset + orientation-invariant augmentation
  models.py           #   timm ViT + attention-rollout heatmaps
  train.py            #   device-agnostic train loop + stratified k-fold CV
  morphology.py       #   label schema + metrics
  ssl.py              #   M2: SimCLR self-supervised contrastive training
  embed.py            #   M2: encoder embeddings -> UMAP 2-D layout
experiments/
  fetch_gz.py         #   M1: Galaxy Zoo visual labels + DJA JWST cutouts
  fetch_ceers.py      #   M1 alt: Sersic-fit labels (CEERS) + DJA cutout service
  fetch_data.py       #   M1: synthetic Sersic fallback / local-mosaic Cutout2D
  fetch_atlas.py      #   M2: ~2.4k unlabelled JWST cutouts for the atlas
  train_vit.py        #   M1: CV metrics + final model + metrics.json
  train_atlas.py      #   M2: SimCLR encoder
  build_atlas.py      #   M2: embed -> UMAP -> sprite sheet + atlas.json
  export_onnx.py      #   -> INT8 web/morphology/model.onnx
  build_web_assets.py #   gallery PNGs + attention maps + json
web/
  morphology/         #   Demo A: classifier (onnxruntime-web, no build step)
  atlas/              #   Demo B: self-supervised fly-through (canvas + sprite sheet)
  index.html          #   landing page
tests/                #   data + model unit tests
```

## Running it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# real data: Galaxy Zoo visual labels + JWST cutouts (catalogue ~52 MB, gitignored)
curl -L -o data/gz_candels.fits \
  https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-candels/gz_candels_table_2_main_release.fits
python -m experiments.fetch_gz --per-class 300      # select + fetch real cutouts

python -m experiments.train_vit  --folds 5 --epochs 10
python -m experiments.export_onnx
python -m experiments.build_web_assets --per-class 6
pytest -q                                           # data + model tests

# M2 — the self-supervised atlas
python -m experiments.fetch_atlas  --n 2550         # unlabelled JWST cutouts
python -m experiments.train_atlas  --epochs 80      # SimCLR encoder (MPS)
python -m experiments.build_atlas                   # embed -> UMAP -> atlas assets

cd web && python3 -m http.server                    # open http://localhost:8000
```

No-download fallback (synthetic Sérsic galaxies): `python -m experiments.fetch_data --synthetic 160`.

## Credits & data

Built with PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) and
[onnxruntime-web](https://onnxruntime.ai/). Real JWST cutouts from the
[DAWN JWST Archive](https://dawn-cph.github.io/dja/) (grizli reductions); visual-morphology
labels from [Galaxy Zoo](https://data.galaxyzoo.org/): CANDELS (Simmons et al. 2017). M2 will
use MAST HLSP [JADES](https://archive.stsci.edu/hlsp/jades). All data public.
