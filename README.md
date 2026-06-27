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
| **[Morphology](web/morphology/)** | *Live* | A `ViT-Small` in ONNX classifies a **real JWST/CEERS** cutout in your browser — disk / spheroid / compact — with attention heatmaps and **honest, cross-validated** metrics. **Built.** |
| **Atlas** | Self-supervised | Fly through a 2-D embedding of thousands of real JWST galaxies, coloured by morphology or redshift. *Planned (M2).* |
| **Anomaly hunt** | Discovery | Score every galaxy by how unusual it is, and surface the **Little Red Dots**. *Planned (M3).* |
| **Dropout hunter** | Cosmic dawn | Find galaxies at z>10 as they vanish from the bluer filters. *Planned (M4).* |

## The data, honestly

The demo trains on **real JWST/NIRCam cutouts of CEERS galaxies**. Two things are real and
two are worth stating plainly:

- **Imagery — real.** Cutouts (F200W + F356W + F444W) are pulled by RA/Dec from the
  [DAWN JWST Archive](https://dawn-cph.github.io/dja/) grizli cutout service — actual JWST
  pixels, no mosaic download. The RGB you see is our asinh composite of the three raw bands.
- **Labels — real, profile-derived.** The three classes come from **single-Sérsic profile
  fits** (van der Wel et al. 2025, the DJA morphology catalogue): `disk` (late-type,
  n < 1.2), `spheroid` (early-type, n > 2.5), and `compact` (effective radius < 0.09″,
  point-source-like). The ViT never sees those fit parameters — it learns morphology from
  the pixels. This is a genuine, standard supervised-morphology task.
- The **compact** class is deliberately the LRD-adjacent population; M3's anomaly hunt builds
  on it.
- A **synthetic Sérsic generator** is retained as a no-download fallback (and for CI/tests):
  `python -m experiments.fetch_data --synthetic 160`. A local-mosaic `Cutout2D` path also
  exists (`fetch_data --real`). See [`plan.md`](plan.md) for provenance and the M2–M4 roadmap.

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
  fetch_ceers.py      #   REAL CEERS cutouts: Sersic-fit labels + DJA cutout service
  fetch_data.py       #   synthetic Sersic fallback / local-mosaic Cutout2D path
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

# real CEERS data: download the morphology catalogue once (~375 MB, gitignored)
curl -L -o data/ceers_morpho.fits.gz \
  https://s3.amazonaws.com/aurelien-sepp/ceers-full-grizli-v7.2/catalog/ceers-full-grizli-v7.2_morpho-phot.fits.gz
python -m experiments.fetch_ceers --per-class 300   # select + fetch real cutouts

python -m experiments.train_vit  --folds 5 --epochs 10
python -m experiments.export_onnx
python -m experiments.build_web_assets --per-class 6
pytest -q                                           # data + model tests

cd web && python3 -m http.server                    # open http://localhost:8000
```

No-download fallback (synthetic Sérsic galaxies): `python -m experiments.fetch_data --synthetic 160`.

## Credits & data

Built with PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) and
[onnxruntime-web](https://onnxruntime.ai/). Real JWST cutouts + Sérsic-fit morphology labels
from the [DAWN JWST Archive](https://dawn-cph.github.io/dja/) (CEERS; reductions by grizli;
morphology catalogue van der Wel et al. 2025). M2 will use MAST HLSP
[JADES](https://archive.stsci.edu/hlsp/jades). All data public.
