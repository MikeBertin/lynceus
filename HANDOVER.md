# Lynceus — session handover

A from-scratch astrophysics-ML project: machine learning on JWST deep-field
galaxies, with a serious Python core and self-contained, no-build browser demos.
Sibling to Chiron (physics), Empedocles (EAs), Plutus (quant). This file is the
single source of truth for picking the project back up. Read it top to bottom.

---

## 1. Status — where we are

**Three milestones are built, live, verified, and pushed.** Repo is **private**
on GitHub: `git@github.com:MikeBertin/lynceus.git` (user `MikeBertin`, SSH).
Default branch `main`. Nothing is published yet (no GitHub Pages) — that's a
deliberate choice; flip it on when ready (see §8).

| Milestone | What | Headline result |
|---|---|---|
| **M1 — Morphology** | In-browser ViT classifies a real JWST cutout (featured / smooth / merger) with attention maps | **81.0%** 5-fold CV vs 33% baseline, on real Galaxy Zoo labels |
| **M2 — Atlas** | Self-supervised (SimCLR) 2-D embedding of ~2,400 real JWST galaxies you fly through | kNN-morphology **58%** vs 33% — structure emerges with no labels |
| **M3 — Anomaly hunt** | Latent-space outlier score + 216 real Little Red Dots overlaid | LRDs **~3.4×** over-represented in the top-10% anomalies (was ~2× before the colour-aware encoder) |

Both demos live under `web/`: `web/morphology/` and `web/atlas/` (the atlas page
also hosts the anomaly hunt, the "where are we looking?" globe, and "things to
notice" cards). Landing at `web/index.html`.

**Next: M4.** See §9.

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
web/
  index.html          #   landing (4 cards)
  shared/theme.css    #   cosmic theme + notice/popover styles (VERSIONED: ?v=2)
  shared/notice.js    #   "things to notice" popover handler
  morphology/         #   Demo A: ONNX classifier (index.html, app.js, model.onnx, gallery/)
  atlas/              #   Demo B: atlas + anomaly + globe (index.html, app.js,
                      #     skymap.js, atlas.json, lrds.json, sprites.jpg, lrd_sprites.jpg, skygeom.json)
tests/                # test_data.py, test_models.py
data/                 # GITIGNORED — all raw catalogs/cutouts/embeddings (regenerable)
models/               # GITIGNORED — vit.pt, ssl_encoder.pt, vit.onnx, metrics.json
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
- M2: SimCLR ResNet-18, 80 epochs, NT-Xent 4.5→2.0. kNN-morphology 58% vs 33%;
  kNN-by-field 58% vs 39% (regions loosely separable — survey-depth fingerprint).
- M3: anomaly = mean cosine distance to 20 nearest in 512-D space. 216 LRDs are
  **~3.4×** over-represented in the top-10% anomalies (33.8% of LRDs in the top
  decile vs 10% by chance; median LRD at 64th percentile). **Sharpened (was ~2×):**
  the asinh stretch is now colour-preserving (Lupton-style — `asinh_stretch(colour=True)`
  in `core/data.py`: per-band sky subtraction, a single shared intensity stretch,
  linear per-band scaling, so flux *ratios* survive), the SSL encoder was retrained
  on the colour cutouts (per-channel colour jitter trimmed ±15%→±8%), and the LRDs
  re-embedded. The encoder now keys on both compactness *and* redness. Old
  colour-blind encoder kept at `models/ssl_encoder_perchannel.pt`. M1 is pinned to
  the original per-channel stretch (`colour=False`) so its shipped 81% reproduces.

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
- Big committed assets: `web/morphology/model.onnx` (22 MB), `web/atlas/sprites.jpg`
  (1.5 MB), `lrd_sprites.jpg`, `atlas.json`, `lrds.json`, `skygeom.json`. These are
  intentionally committed (the demos need them); `data/` and `models/` are not.
- **To publish** (when the user says so): make the repo public
  (`gh repo edit MikeBertin/lynceus --visibility public`) and enable GitHub Pages
  on `main`/`web` (or move `web/` to root / use a Pages action), matching how
  Chiron/Empedocles ship at `mikebertin.github.io/<name>`. The landing footer +
  morphology footer already link the siblings. **Do not push/publish without the
  user explicitly asking.**

---

## 9. M4 — what to build next (the point of this handover)

The roadmap's M4 is the **"dropout hunter"** (it's the 4th landing card,
currently "Frontier · Planned"). Two strong, concrete options — confirm with the
user which to pursue:

### Option A (roadmap): the dropout hunter — photometric redshift / z>10
The science: high-z galaxies "drop out" of bluer filters as the Lyman break
redshifts through the bands; JWST pushed the record to z≈14 and found *too many,
too bright* early galaxies (a real ΛCDM tension). 
- **Data is already in hand:** the **Kokorev LRD catalog** (`data/lrd_kokorev.fits`)
  and the **DJA morpho-phot catalog** (`fetch_ceers.py`'s download) both have
  multi-band NIRCam fluxes (F090W…F444W) **and** `z_phot`. That's a ready-made
  photo-z training set (fluxes → redshift). For a bigger/cleaner set, pull the DJA
  per-field photometric catalogs or JADES photometric catalog (MAST HLSP `jades`).
- **Model:** a small MLP/transformer on the band fluxes → photo-z (point + PDF).
  Train offline; tiny model → run in-browser via ONNX, or precompute.
- **Demo (`web/dropout/`):** scrub through the filters and watch a galaxy *vanish*
  from the blue bands then "drop in" at red; show the photo-z PDF and a
  colour–colour (Lyman-break) diagram; a gallery of z>10 candidates. Reuse the
  cutout service for the imagery and the same theme/notice-card patterns.
- Promote the landing "dropout hunter" card to Live when done.

### Option B (colour-aware encoder → sharpen M3) — ✅ DONE (2026-06-29)
Completed. The per-channel stretch that threw away colour was replaced with a
colour-preserving Lupton-style stretch (`asinh_stretch(colour=True)`); the atlas +
LRD cutouts were re-fetched, SimCLR retrained, and the atlas rebuilt. LRD
enrichment **rose from ~2× to ~3.4×** — the encoder now catches LRDs by redness as
well as compactness. New reproducibility script: `experiments/restretch_atlas.py`
(re-fetches the existing atlas cutouts in place with the colour stretch). See §6.
To redo from scratch: `restretch_atlas` → `train_atlas --epochs 80` → `fetch_lrd`
→ `build_atlas`. (Note: the DJA service rate-limits after a few thousand requests
in a session — the LRD re-fetch may need a retry.)

That leaves **Option A (the dropout hunter / M4)** as the next milestone — the
"new demo" that matches the roadmap and the landing card. See Option A above.

Other threads if asked: bigger atlas / WebGL rendering; image+spectra contrastive
(AstroCLIP-style) using NIRSpec; or publishing (§8).

### How to start M4 (whichever option)
1. `cd /Users/m/.openclaw/workspace/projects/lynceus && .venv/bin/python -m pytest -q`
   (sanity).
2. Confirm the demo still serves: `lynceus-web` config, port 8137.
3. For Option A: inspect `data/lrd_kokorev.fits` columns (or the DJA morpho-phot
   catalog) — `*_flux` bands + `z_phot` — and prototype a photo-z regressor before
   any web work. For Option B: edit `core/data.asinh_stretch` / the SSL input
   normalisation, retrain, rebuild, re-run the M3 feasibility check (kNN of LRDs vs
   atlas — see the pattern used in `fetch_lrd`/`anomaly`).
4. Keep the house style: serious `core/` code, no-build browser demo, honest
   numbers stated in the UI, "things to notice" cards, versioned assets.
