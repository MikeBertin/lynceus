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
| **[Morphology](web/morphology/)** | *Live* | A `ViT-Small` in ONNX classifies a galaxy cutout in your browser — disk / spheroid / irregular / point source / merger — with attention heatmaps and **honest, cross-validated** metrics. **Built.** |
| **Atlas** | Self-supervised | Fly through a 2-D embedding of thousands of real JWST galaxies, coloured by morphology or redshift. *Planned (M2).* |
| **Anomaly hunt** | Discovery | Score every galaxy by how unusual it is, and surface the **Little Red Dots**. *Planned (M3).* |
| **Dropout hunter** | Cosmic dawn | Find galaxies at z>10 as they vanish from the bluer filters. *Planned (M4).* |

## The on-ramp, honestly

The classifier is real and runs client-side. The **default M1 dataset is a
physically-motivated synthetic set** — galaxies rendered from **Sérsic light profiles**
(n≈1 disks, n≈4 spheroids, clumpy irregulars, PSF point sources, two-body mergers), with a
wavelength-dependent PSF and correlated noise — so the whole pipeline and demo are
reproducible from a clean checkout with **no multi-GB downloads**.

The **real-data path is implemented**: drop the public CEERS/JADES NIRCam mosaics into
`data/mosaics/` and a label catalogue, and the *same* `astropy` `Cutout2D` code fills the
cache from real JWST pixels — training, export and the demo are agnostic to the source. The
intended labels are the CEERS **VISUAL** morphology catalogue (Kartaltepe et al. 2023). See
[`plan.md`](plan.md) for data provenance and the M2–M4 roadmap.

## Architecture

```
core/                 # reusable package
  config.py           #   paths, bands, device auto-select (mps/cuda/cpu)
  data.py             #   synthetic Sersic generator + real Cutout2D path + asinh stretch
  datasets.py         #   torch Dataset + orientation-invariant augmentation
  models.py           #   timm ViT + attention-rollout heatmaps
  train.py            #   device-agnostic train loop + stratified k-fold CV
  morphology.py       #   label schema + metrics
experiments/
  fetch_data.py       #   build the cutout cache (synthetic default / --real)
  train_vit.py        #   CV metrics + final model + metrics.json
  export_onnx.py      #   -> INT8 web/morphology/model.onnx
  build_web_assets.py #   gallery PNGs + attention maps + json
web/
  morphology/         #   Demo A (onnxruntime-web, no build step)
  index.html          #   landing page
tests/                #   data + model unit tests
```

## Running it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m experiments.fetch_data --synthetic 160   # build the cutout cache
python -m experiments.train_vit  --folds 5 --epochs 10
python -m experiments.export_onnx
python -m experiments.build_web_assets --per-class 6
pytest -q                                           # data + model tests

cd web && python3 -m http.server                    # open http://localhost:8000
```

To use **real JWST data** instead, download CEERS/JADES mosaics + a label catalogue, then:

```bash
python -m experiments.fetch_data --real --catalog data/ceers_visual.csv \
    --mosaic F444W=data/mosaics/f444w.fits \
    --mosaic F356W=data/mosaics/f356w.fits \
    --mosaic F200W=data/mosaics/f200w.fits
```

## Credits & data

Built with PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) and
[onnxruntime-web](https://onnxruntime.ai/). JWST data: MAST HLSPs
[JADES](https://archive.stsci.edu/hlsp/jades) and CEERS. Morphology label scheme after
Kartaltepe et al. 2023 (CEERS Key Paper III). All data public.
