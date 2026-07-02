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
| **[Atlas](web/atlas/)** | *Live* | Fly through a **self-supervised** (SimCLR) 2-D embedding of **~9,700 real JWST galaxies** (WebGL-rendered) ; colour by morphology / sky region / **anomaly**; watch structure the encoder found with **no labels** emerge. A kNN probe recovers morphology **62% ± 0.7%** (5-fold) vs 39% baseline. **Built.** |
| **[Anomaly hunt](web/atlas/)** | *Live* | Score every galaxy by latent-space isolation; **216 real Little Red Dots** (Kokorev+24), embedded with the same encoder, land in the flagged hot zones (**5× enriched** in the top-10% anomalies, 95% CI 4.4–5.7×, p<0.0001 vs a random-galaxy null — up from ~2× once a colour-preserving stretch let the encoder see their redness, and sharper still on the larger atlas). **Built** (in the atlas). |
| **[Dropout hunter](web/dropout/)** | *Live* | A neural **photometric redshift** net (ONNX, in your browser) reads a galaxy's redshift from nine JWST/HST bands as the **Lyman break** sweeps through the filters; watch z>10 galaxies **drop out** of the bluer bands, see the redshift PDF (and its low-z-interloper degeneracy), and a Lyman-break colour–colour diagram. **σ_NMAD ≈ 0.040** (95% CI 0.037–0.044) vs ~1,800 held-out spectroscopic redshifts (EAZY template ceiling: 0.027). PDF calibration checked honestly (PIT + coverage): a 90% credible interval contains the truth ~88% of the time. **Built (M4).** |

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
  anomaly.py          #   M3: kNN latent-space anomaly scoring
  photoz.py           #   M4: photo-z featurisation, MLP, PDF calibration (PIT/coverage)
  stats.py            #   bootstrap CIs + permutation null tests for the headline claims
experiments/
  fetch_gz.py         #   M1: Galaxy Zoo visual labels + DJA JWST cutouts
  fetch_ceers.py      #   M1 alt: Sersic-fit labels (CEERS) + DJA cutout service
  fetch_data.py       #   M1: synthetic Sersic fallback / local-mosaic Cutout2D
  fetch_atlas.py      #   M2: unlabelled JWST cutouts for the atlas
  fetch_lrd.py        #   M3: known Little Red Dots (Kokorev+24) + embeddings
  train_vit.py        #   M1: CV metrics + final model + metrics.json
  train_atlas.py      #   M2: SimCLR encoder
  build_atlas.py      #   M2/M3: embed -> UMAP -> sprites + atlas.json + anomaly + lrds.json
  export_onnx.py      #   -> INT8 web/morphology/model.onnx
  build_web_assets.py #   gallery PNGs + attention maps + json
  build_photoz_dataset.py # M4: CEERS fluxes + redshifts -> photoz.npz
  train_photoz.py     #   M4: train the photo-z MLP, validate on spec-z
  export_photoz_onnx.py   # M4: -> web/dropout/photoz.onnx
  build_dropout_assets.py # M4: filmstrips + PDFs + colour-colour -> web/dropout/
  significance.py     #   error bars: bootstrap CIs + permutation nulls on every headline
  calibrate_photoz.py #   M4: PIT + coverage calibration of the redshift PDFs
web/
  morphology/         #   Demo A: classifier (onnxruntime-web, no build step)
  atlas/              #   Demo B: self-supervised fly-through + anomaly hunt (WebGL)
  dropout/            #   Demo D: photo-z dropout hunter (onnxruntime-web)
  index.html          #   landing page
tests/                #   data, model, stats + calibration unit tests
```

## Running it

The whole pipeline is a `Makefile` — `make help` lists every target, one per
milestone, with data-fetch and training separated (and the DJA rate-limit and
rough costs noted inline). Requirements are **pinned** to the versions the
published numbers were produced with (Python 3.13, Apple MPS).

```bash
make venv        # python3.13 venv + pinned requirements
make m1-data m1  # Galaxy Zoo labels + JWST cutouts -> ViT -> web demo
make m2-data m2  # ~10k cutouts -> SimCLR (the long run) -> atlas
make m3-data m3  # Little Red Dots -> anomaly overlay
make m4-data m4  # CEERS catalogue -> photo-z net -> dropout demo
make stats       # error bars, null tests + PDF calibration -> web JSON
make test serve  # unit tests; demos at http://localhost:8137
```

Or step by step:

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

# M3 — the anomaly hunt: known Little Red Dots, embedded with the same encoder
curl -L -o data/lrd_kokorev.fits \
  https://raw.githubusercontent.com/VasilyKokorev/lrd_phot/master/lrd_table_v1.1.fits
python -m experiments.fetch_lrd                     # fetch + embed the LRDs
python -m experiments.build_atlas                   # atlas + anomaly scores + lrds.json

cd web && python3 -m http.server                    # open http://localhost:8000
```

No-download fallback (synthetic Sérsic galaxies): `python -m experiments.fetch_data --synthetic 160`.

## Credits & data

Built with PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) and
[onnxruntime-web](https://onnxruntime.ai/). Real JWST cutouts from the
[DAWN JWST Archive](https://dawn-cph.github.io/dja/) (grizli reductions); visual-morphology
labels from [Galaxy Zoo](https://data.galaxyzoo.org/): CANDELS (Simmons et al. 2017). Little
Red Dot catalogue: [Kokorev et al. 2024](https://github.com/VasilyKokorev/lrd_phot). All data
public.
