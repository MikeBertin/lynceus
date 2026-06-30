# Lynceus — session handover

A from-scratch astrophysics-ML project: machine learning on JWST deep-field
galaxies, with a serious Python core and self-contained, no-build browser demos.
Sibling to Chiron (physics), Empedocles (EAs), Plutus (quant). This file is the
single source of truth for picking the project back up. Read it top to bottom.

---

## 1. Status — where we are

**Four milestones built, live, verified, and committed.** Repo is **private**
on GitHub: `git@github.com:MikeBertin/lynceus.git` (user `MikeBertin`, SSH).
Default branch `main`, all work committed there. **Nothing is published yet**
(no GitHub Pages) — deliberately paused at the user's call, *ready to publish but
not published* (see §8 + §9).

| Milestone | What | Headline result |
|---|---|---|
| **M1 — Morphology** | In-browser ViT classifies a real JWST cutout (featured / smooth / merger) with attention maps | **81.0%** 5-fold CV vs 33% baseline, on real Galaxy Zoo labels |
| **M2 — Atlas** | Self-supervised (SimCLR) 2-D embedding of **~9,700** real JWST galaxies you fly through (**WebGL**-rendered) | kNN-morphology **62%** vs 39% — structure emerges with no labels |
| **M3 — Anomaly hunt** | Latent-space outlier score + 216 real Little Red Dots overlaid | LRDs **~5×** over-represented in the top-10% anomalies (2× → 3.4× colour-aware → 5× on the larger atlas) |
| **M4 — Dropout hunter** | In-browser neural **photo-z** over 9 JWST/HST bands; filter scrubber shows the Lyman break sweeping + galaxies dropping out; redshift PDF + Lyman-break colour–colour diagram | **σ_NMAD ≈ 0.040** vs ~1,800 held-out spec-z (EAZY template ceiling 0.027) |

Demos under `web/`: `web/morphology/`, `web/atlas/` (also hosts the anomaly hunt,
the globe, "things to notice"), and `web/dropout/` (M4). Landing at
`web/index.html` (all four cards **Live**).

**Recent work, all committed** (newest first):
- `0e77488` **pre-publish polish** — themed SVG favicon (`web/shared/favicon.svg`),
  og/twitter meta on all 4 pages, refreshed stale landing copy, **mobile fix**
  (atlas colour-by control clipped the *anomaly* chip off-screen → now wraps),
  **WebGL→canvas-2D fallback** in the atlas.
- `c35db9d` **atlas at scale** — 2,389 → **9,673** galaxies, **WebGL** renderer,
  encoder retrained, LRD enrichment → **~5×**, globe Sun now computed for the
  current day client-side, LRD-vs-atlas wording clarified.
- `dbe8882` **M4 dropout hunter**; `5a99a2e` **M3 colour-aware** (2× → 3.4×).

**Next: make it a credible research on-ramp, then publish — see §9.** The user
explicitly framed Lynceus as "an on-ramp for a real research project," so the
priority is *scientific rigor on the existing claims* (error bars / significance,
photo-z calibration, reproducibility) before — or alongside — publishing.

---

## 2. Environment (important — non-obvious)

- **Machine:** Apple **M4 Mac mini, 10-core GPU, 24 GB unified**. No CUDA — use
  PyTorch **MPS**. All training is device-agnostic via `core.config.get_device()`.
- **Python:** system python is **3.14** (too new for torch wheels). Use the repo
  venv on **3.13**: `projects/lynceus/.venv` (made with `python3.13 -m venv`).
  Always call `.venv/bin/python`. Deps in `requirements.txt` (+ `onnxscript`,
  `umap-learn`, `requests`, `astropy`/`astroquery`, `pytest` — all installed).
- **Background shells** start from a fresh cwd, not the persisted one — use
  **absolute paths** (`/Users/m/.openclaw/workspace/projects/lynceus/...`).
- **Foreground `sleep` is blocked.** To wait on a background job, launch a watcher
  with `run_in_background: true` (`while pgrep -f X; do sleep 8; done; ...`).

Run everything from the repo root:
```bash
cd /Users/m/.openclaw/workspace/projects/lynceus
.venv/bin/python -m pytest -q          # 3 tests, should pass
```

---

## 3. Repo map

```
core/                 # reusable Python package (the serious core)
  config.py           #   paths, bands, device select, cutout sizes
  data.py             #   M1 data: synthetic Sersic generator + real cutouts via
                      #     DJA grizli service (fetch_service_cube) + asinh stretch
  datasets.py         #   torch Dataset + augmentation (M1)
  models.py           #   timm ViT + attention rollout (M1)
  morphology.py       #   M1 label schema (featured/smooth/merger) + metrics
  train.py            #   device-agnostic train loop + stratified k-fold CV (M1)
  ssl.py              #   M2 SimCLR (ResNet-18, NT-Xent), 128px, MPS
  embed.py            #   M2 encoder->features->cosine UMAP->2D layout
  anomaly.py          #   M3 kNN cosine anomaly score + nearest-distance
  photoz.py           #   M4 photo-z: featurize + binned-softmax MLP + metrics
experiments/          # one-shot scripts (run as `python -m experiments.X`)
  fetch_gz.py         #   M1: Galaxy Zoo visual labels + JWST cutouts  (PRIMARY)
  fetch_ceers.py      #   M1 alt: Sersic-fit labels (DJA CEERS catalog)
  fetch_data.py       #   M1: synthetic fallback / local-mosaic Cutout2D
  train_vit.py        #   M1: CV metrics + final model + metrics.json
  export_onnx.py      #   M1: vit.pt -> INT8 web/morphology/model.onnx
  build_web_assets.py #   M1: gallery PNGs + attention maps + labels/metrics json
  fetch_atlas.py      #   M2: ~2.5k unlabelled JWST cutouts (stratified by GZ class)
  train_atlas.py      #   M2: train SimCLR encoder -> models/ssl_encoder.pt
  fetch_lrd.py        #   M3: 216 Kokorev+24 LRDs in our fields, fetched + embedded
  build_atlas.py      #   M2/M3: embed -> UMAP -> sprites + atlas.json + anomaly + lrds.json
  restretch_atlas.py  #   M3: re-fetch atlas cutouts in place with colour stretch
  build_photoz_dataset.py # M4: CEERS fluxes+z -> data/photoz/photoz.npz
  train_photoz.py     #   M4: train photo-z MLP, validate on spec-z -> models/photoz.pt
  export_photoz_onnx.py   # M4: photoz.pt -> web/dropout/photoz.onnx (+ meta)
  build_dropout_assets.py # M4: filmstrips + PDFs + colour-colour -> web/dropout/
web/
  index.html          #   landing (4 cards, all Live)
  shared/theme.css    #   cosmic theme + notice/popover styles (VERSIONED: ?v=2)
  shared/notice.js    #   "things to notice" popover handler
  shared/favicon.svg  #   themed SVG "eye" favicon (all pages link it)
  morphology/         #   Demo A: ONNX classifier (index.html, app.js, model.onnx, gallery/)
  atlas/              #   Demo B: atlas + anomaly + globe (index.html, app.js,
                      #     skymap.js, atlas.json, lrds.json, sprites.jpg, lrd_sprites.jpg, skygeom.json)
  dropout/            #   Demo D (M4): photo-z dropout hunter (index.html, app.js,
                      #     photoz.onnx, photoz_meta.json, dropout.json, filmstrips.jpg)
tests/                # test_data.py, test_models.py
data/                 # GITIGNORED — all raw catalogs/cutouts/embeddings (regenerable)
models/               # GITIGNORED — vit.pt, ssl_encoder.pt, photoz.pt, *.onnx, *metrics.json
.venv/                # GITIGNORED
```

`README.md` and `plan.md` are the public-facing narrative + roadmap. `MEMORY`
(in the assistant's memory store) has a condensed version of this.

---

## 4. Data & pipelines — all public, all regenerable

`data/` and `models/` are gitignored. To rebuild from scratch:

**M1 (morphology):** Galaxy Zoo: CANDELS volunteer labels + real JWST cutouts.
```bash
curl -L -o data/gz_candels.fits \
  https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-candels/gz_candels_table_2_main_release.fits
.venv/bin/python -m experiments.fetch_gz --per-class 300   # cutouts via DJA service
.venv/bin/python -m experiments.train_vit --folds 5 --epochs 10
.venv/bin/python -m experiments.export_onnx
.venv/bin/python -m experiments.build_web_assets --per-class 7
```
Classes featured/smooth/merger from GZ tasks t00 (smooth/features) + t16 (merging).
Galaxy Zoo classified **HST** imaging; we show the **JWST** view of the same
galaxies (cross-matched by RA/Dec) — stated honestly in the UI.

**M2 (atlas):** unlabelled cutouts -> SimCLR -> UMAP.
```bash
.venv/bin/python -m experiments.fetch_atlas --n 2550   # stratified ~850/class for colouring
.venv/bin/python -m experiments.train_atlas --epochs 80
.venv/bin/python -m experiments.build_atlas            # writes atlas.json + sprites.jpg
```

**M3 (anomaly + LRDs):**
```bash
curl -L -o data/lrd_kokorev.fits \
  https://raw.githubusercontent.com/VasilyKokorev/lrd_phot/master/lrd_table_v1.1.fits
.venv/bin/python -m experiments.fetch_lrd     # 216 LRDs in GOODS-S/COSMOS/UDS, embedded
.venv/bin/python -m experiments.build_atlas   # adds anomaly score + writes lrds.json + lrd_sprites.jpg
```

**Key external services / catalogs (all worked):**
- **DJA grizli cutout service** — real JWST NIRCam cutouts by RA/Dec, no mosaic
  download: `https://grizli-cutout.herokuapp.com/thumb?ra=..&dec=..&size=3&filters=f444w-clear,f356w-clear,f200w-clear&output=fits`
  (returns multi-ext FITS, one HDU per band). Covers GOODS-S, COSMOS, UDS, EGS.
  Used by `data.fetch_service_cube`.
- **Galaxy Zoo CANDELS** (Simmons+17): `data.galaxyzoo.org`. Has GOODS-S/COSMOS/UDS
  (NOT EGS). RA/Dec + full vote-fraction decision tree.
- **DJA morphology (Sersic) catalog** (van der Wel+25): used by `fetch_ceers.py`
  alt path. `s3.amazonaws.com/aurelien-sepp/ceers-full-grizli-v7.2/catalog/ceers-full-grizli-v7.2_morpho-phot.fits.gz` (375 MB).
  HDU2 has SERSIC/RADIUS/RA_MODEL/DEC_MODEL/MAG_MODEL_*; HDU1 has many band fluxes + z_phot.
- **Kokorev+24 LRD catalog**: `github.com/VasilyKokorev/lrd_phot` → `lrd_table_v1.1.fits`
  (260 LRDs; 216 in our fields; columns: id, field, ra, dec, all NIRCam band fluxes,
  z_phot, r_eff, av, lbol, muv). **This is gold for M4 photo-z work** — real
  multi-band photometry + redshifts.

The three fields and their positions (for the "where are we looking" globe and
sky-region colours): **GOODS-S** (RA~53.1, b=-54°), **COSMOS** (RA~150.1, b=+42°),
**UDS** (RA~34.4, b=-60°). All high galactic latitude (chosen to look out of the
Milky Way). Region colours: GOODS-S `#5ec27a`, COSMOS `#ff9e64`, UDS `#c792ea`.

---

## 5. Web / demo architecture (read before touching the atlas JS)

The demos are vanilla JS, no build step, served by a static server. Hard-won
gotchas — **all of these have bitten us**:

- **Cache-busting is mandatory.** Every JS/CSS/JSON the page fetches is versioned
  with `?v=N` (e.g. `app.js?v=11`, `theme.css?v=2`, `atlas.json?v=3`,
  `lrds.json?v=2`). **When you change a file, bump its version** or the browser
  serves a stale copy. This caused multiple "it's broken" scares.
- **`requestAnimationFrame` loops stall** in the headless preview (page not
  "visible"). The atlas uses **microtask-coalesced draws** (`markDirty()` ->
  `Promise.resolve().then(draw)`), not an rAF loop. Keep it that way. (The globe's
  gentle auto-spin does use rAF — fine, it's cosmetic and works in real browsers.)
- **Canvas sizing race:** `stage.clientWidth` / `window.innerWidth` can read **0**
  before layout (and in the preview). Use a fallback width, a **ResizeObserver**
  to fit once real dims arrive, and map pointer coords by `canvas.width/rect.width`
  (not bare `dpr`). See `resize()` / `evPos()` in `web/atlas/app.js`.
- The atlas `app.js` is the big one. Colour modes: `region` (default), `morph`,
  `featured`, `merger`, `anomaly` — order is the narrative. `setMode()` couples
  the LRD overlay to anomaly mode and sets a per-mode context line (`updateContext`).
  Points carry: x,y,i(sprite),d(dominant),r(region),s/f/m/c(votes),a(anomaly),lc.
  LRDs (`lrds.json`) carry x,y,i,a,z,field and have their own sprite sheet.

**Preview/launch:** the preview tool reads the **workspace-level**
`projects/.claude/launch.json` (NOT the repo's). `lynceus-web` is the first config
(port 8137, serves `lynceus/web`) so it's the default. To run locally:
`cd web && python3 -m http.server`.

---

## 6. Key results (for copy / honesty)

- M1: 81.0% acc, macro-F1 0.81. Only real confusion is smooth↔merger (genuine
  degeneracy). Model = `vit_small_patch16_224` (timm, ImageNet-pretrained),
  fine-tuned, INT8-quantised to 22 MB ONNX.
- M2: SimCLR ResNet-18, 80 epochs, NT-Xent 4.5→2.0 (1.81 on the 9,673-galaxy set).
  On the scaled atlas: **kNN-morphology 62% vs 39%** baseline; **kNN-by-field 63%
  vs 41%** (regions loosely separable — survey-depth fingerprint). Encoders kept:
  `ssl_encoder.pt` (current, 9.7k colour), `ssl_encoder_2389.pt` (colour, old size),
  `ssl_encoder_perchannel.pt` (old colour-blind). All gitignored.
- M3: anomaly = mean cosine distance to 20 nearest in 512-D space. 216 LRDs are
  **~5×** over-represented in the top-10% anomalies (33.8% of LRDs in the top decile
  vs 10% by chance; median LRD at the **70th** percentile) on the 9.7k atlas — up
  the chain **2× → 3.4× → 5×**. The 3.4×→5× jump came from the larger reference
  cloud, not a model change. The 2×→3.4× came from making the asinh stretch
  colour-preserving (Lupton-style — `asinh_stretch(colour=True)` in `core/data.py`:
  per-band sky subtraction, a single shared intensity stretch, linear per-band
  scaling, so flux *ratios* survive), retraining SimCLR on the colour cutouts
  (per-channel colour jitter trimmed ±15%→±8%), and re-embedding the LRDs — the
  encoder now keys on both compactness *and* redness. M1 is pinned to the original
  per-channel stretch (`colour=False`) so its shipped 81% reproduces.
  **NB (see §9 #1):** the 5× is still a bare point estimate — needs a CI + null test.

---

## 7. Build/export gotchas (Python side)

- ONNX export: use **`torch.onnx.export(..., dynamo=False)`** (the new dynamo
  exporter splits weights into an external `.data` file — breaks onnxruntime-web).
  Needs `onnxscript` installed.
- ONNX quantisation: **`quantize_dynamic(..., op_types_to_quantize=["MatMul"])`**
  only — onnxruntime-web's wasm backend has **no ConvInteger** kernel, so leave
  the patch-embed Conv in fp32.
- UMAP params (in `embed.umap_2d`): `n_neighbors=30, min_dist=0.5, spread=1.4`,
  cosine metric, then per-axis percentile fill in `normalise_coords` (clips
  outliers to the border — those are the visible edge "tendrils").

---

## 8. Git & publishing

- All work is on `main`, pushed, **private**. Commit style: present-tense subject
  + body, with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Big committed assets (the demos need them; `data/` and `models/` are not committed):
  `web/morphology/model.onnx` (22 MB); `web/atlas/sprites.jpg` (**~4.6 MB**, 48px
  tiles, 9,673 galaxies), `atlas.json` (~1 MB), `lrd_sprites.jpg`, `lrds.json`,
  `skygeom.json`; `web/dropout/photoz.onnx` (487 KB), `dropout.json`, `filmstrips.jpg`;
  `web/shared/favicon.svg`. First-load for the atlas is ~5.6 MB — lean enough for Pages.
- **To publish** (when the user says so): make the repo public
  (`gh repo edit MikeBertin/lynceus --visibility public`) and enable GitHub Pages
  on `main`/`web` (or move `web/` to root / use a Pages action), matching how
  Chiron/Empedocles ship at `mikebertin.github.io/<name>`. The landing footer +
  morphology footer already link the siblings. **Do not push/publish without the
  user explicitly asking.**

---

## 9. What's next — make it a real research on-ramp, then publish

All four milestones (M1–M4) + the atlas scaling are **done and committed** (see §1
for commits, §6 for how each was built, and the per-script notes throughout). The
demos are polished and ready. The user's explicit framing: **"this is an on-ramp
for a real research project."** So the next priority is **scientific rigor on the
existing claims**, not new features. Do these against the *cached* data — no
re-fetch or retrain needed for #1–#4.

Start every session with: `cd /Users/m/.openclaw/workspace/projects/lynceus &&
.venv/bin/python -m pytest -q` and confirm the demo serves (`lynceus-web`, port
8137). Keep the house style: serious `core/`, no-build demo, honest numbers in the
UI, "things to notice" cards, versioned assets.

### #1 (top priority) — error bars + significance on the headline claims
Right now every headline is a bare point estimate. Make them defensible.
- **LRD enrichment (the M3 story).** Currently ~33.8% of 216 LRDs land in the
  top-10% anomalies → "5×". Add (a) a **bootstrap 95% CI** (resample the 216 LRDs
  with replacement, recompute the top-decile fraction), and (b) a **permutation /
  null test** (draw 216 random atlas galaxies many times → null distribution of
  the fraction → p-value). Data is all cached: `data/lrd/lrd_emb.npy` (216×512),
  `data/atlas/embeddings.npy` (9673×512); scoring via `core/anomaly.knn_anomaly`
  (see `experiments/build_atlas.py` for the exact `araw`/`lraw` computation).
  Surface "5× (95% CI a–b, p<…)" in the anomaly stat line + caveat card.
- **kNN probes.** Report mean ± std across CV folds (currently bare 62% / 63%).
  `sklearn.cross_val_score(... ).std()` — trivial.
- **Photo-z σ_NMAD.** Bootstrap the ~1,828 spec-z residuals for a CI.
- Suggested home: a `core/stats.py` (bootstrap_ci, permutation_p) + an
  `experiments/significance.py` that prints/saves the numbers; then wire into the
  UI strings + `*_metrics.json`.

### #2 — calibrate the photo-z PDFs
We report σ_NMAD/outliers but never check whether the **PDFs** are trustworthy —
the standard photo-z validation. Compute the **PIT** (for each spec-z galaxy,
evaluate the predicted CDF at z_spec; calibrated ⇒ PIT uniform on [0,1]) and a
**coverage** test (do c%-credible intervals contain the truth c% of the time?).
Recompute PDFs for the held-out spec set from `data/photoz/photoz.npz` (`is_val`)
via `core/photoz` + `models/photoz.pt` (load with `weights_only=False`). If
miscalibrated, temperature-scale the softmax and re-report (honest either way).
New `experiments/calibrate_photoz.py`; surface a small PIT histogram + one
calibration number in the dropout "honest numbers" card.

### #3 — reproducibility hardening
- **Pin the env.** `requirements.txt` is loose; pin exact versions (torch, timm,
  umap-learn, onnxruntime, scikit-learn, astropy, numpy). Record: python **3.13**
  in `.venv` (system 3.14 is too new for torch); Apple **MPS**.
- **One-command repro.** A `Makefile`/`experiments/run_all.py` documenting the DAG
  per milestone (fetch → train → export → build), with the network/time costs and
  the DJA rate-limit caveat noted.
- **Expand tests** (only 3 today). Good targets: `asinh_stretch(colour=True)`
  preserves band ratios (the Lupton property — the whole M3 fix); `photoz.featurize`
  shape/determinism + equals a known vector; `point_estimates` peak/refine logic;
  `knn_anomaly` orders a planted outlier correctly.

### #4 — a `RESEARCH.md` (the on-ramp made explicit)
Short doc of open questions the toolkit could actually attack, e.g.: is the SSL
anomaly score a clean **LRD selection function** (purity/completeness vs Kokorev)?
does the photo-z net's interloper rate beat EAZY on a **controlled subset**? can
the atlas+anomaly produce a **ranked candidate list for spectroscopic follow-up**?
does the encoder **generalise across fields** (train one field, test another)?

### Publishing — PAUSED, ready to go (do NOT publish without an explicit ask)
The user paused here intentionally. Two things to settle first (see §8):
1. **Pages source / URL.** Site lives under `web/` (landing `web/index.html`).
   Siblings publish at `mikebertin.github.io/<name>` with the landing at root.
   Decide: move `web/`'s contents to repo root, use `/docs`, or a `gh-pages`
   branch — **check how Chiron/Empedocles actually do it and mirror it.**
2. **`og:image`.** Deferred (needs the final absolute URL). Once known, generate a
   1200×630 social card (an atlas render works well) and add
   `<meta property="og:image">` to all four pages.
Then: `gh repo edit MikeBertin/lynceus --visibility public` + enable Pages.

### The bigger research bet — M5: image + spectra contrastive (NIRSpec)
The genuine new-science direction (AstroCLIP lineage): align cutout-image
embeddings with **NIRSpec spectrum** embeddings via a contrastive loss, so you can
retrieve spectra from images (and vice versa) and get spectroscopically-informed
representations. New data: JWST NIRSpec spectra (DJA/DAWN spectroscopic releases or
MAST). High effort, real payoff — a true M5, not a polish item.
