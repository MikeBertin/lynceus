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
| **M3 — Anomaly hunt** | Latent-space outlier score + 216 real Little Red Dots overlaid | LRDs **5×** over-represented in the top-10% anomalies (95% CI 4.4–5.7×, p<0.0001 vs random-galaxy null); 2× → 3.4× colour-aware → 5× on the larger atlas |
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
  Always call `.venv/bin/python`. `requirements.txt` is **pinned** to the exact
  working versions (all installed; `make venv` recreates from scratch).
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
  **5×** over-represented in the top-10% anomalies (**50%** of the 216 LRDs land in
  the top decile vs 10% by chance → 5×; **95% CI 4.4–5.7×**, permutation **p<0.0001**
  vs a random-216-galaxy null) on the 9.7k atlas — up the chain **2× → 3.4× → 5×**.
  The **median** LRD sits at the **90th** percentile of atlas anomaly (equivalently:
  50% in the top decile). NB the old "70th percentile" figure was the *mean* of the
  per-LRD percentiles, mislabelled "median" in `build_atlas.py` — now fixed (it writes
  the true median + bootstrap CI + p, via `core/stats.py`). The 3.4×→5× jump came from the larger reference
  cloud, not a model change. The 2×→3.4× came from making the asinh stretch
  colour-preserving (Lupton-style — `asinh_stretch(colour=True)` in `core/data.py`:
  per-band sky subtraction, a single shared intensity stretch, linear per-band
  scaling, so flux *ratios* survive), retraining SimCLR on the colour cutouts
  (per-channel colour jitter trimmed ±15%→±8%), and re-embedding the LRDs — the
  encoder now keys on both compactness *and* redness. M1 is pinned to the original
  per-channel stretch (`colour=False`) so its shipped 81% reproduces.
  **Significance (§9 #1, done):** `experiments/significance.py` + `core/stats.py`
  recompute all headline numbers with error bars from the cached embeddings —
  enrichment 5.0× (CI 4.4–5.7×, p<0.0001), kNN morph 62.0%±0.7% / field 63.4%±0.5%
  (k=15, 5-fold; baselines 38.9% / 40.6%), photo-z σ_NMAD 0.0403 (CI 0.0374–0.0438).
  Writes `models/significance.json` and patches the CIs into `web/atlas/lrds.json`,
  `web/dropout/dropout.json`, `models/photoz_metrics.json` (re-runnable, idempotent).

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

### #1 — error bars + significance on the headline claims — ✅ DONE
Built `core/stats.py` (`bootstrap_ci`, `permutation_p`; percentile bootstrap +
resampled null, both seeded) and `experiments/significance.py` (`python -m
experiments.significance`, ~5 s, all from cached embeddings — no re-fetch/retrain).
Results (seed 0, 10k boot / 10k perm):
- **LRD enrichment:** 5.0× (95% CI 4.4–5.7×), permutation **p<0.0001** vs a
  random-216-galaxy null. Median LRD at the **90th** percentile (CI 87–93).
- **kNN probes (k=15, 5-fold):** morph **62.0%±0.7%** (baseline 38.9%), field
  **63.4%±0.5%** (baseline 40.6%).
- **Photo-z σ_NMAD:** **0.0403** (95% CI 0.0374–0.0438); outliers 18.8% (CI 17.0–20.5).
Writes `models/significance.json` and idempotently patches the CIs into
`web/atlas/lrds.json` (v5), `web/dropout/dropout.json` (v2), `models/photoz_metrics.json`.
UI updated: atlas anomaly stat line + caveat card, "survey fingerprint" card (±std),
dropout honest-numbers note. `build_atlas.py` also fixed to emit the same CIs +
the corrected (true median) percentile, so a full rebuild stays consistent.
Tests: `tests/test_stats.py` (4) — full suite 7 passing.
**Caught + fixed:** the old "median LRD at 70th percentile" was the *mean* of the
per-LRD percentiles mislabelled "median"; the true median is the 90th (consistent
with 50%-in-top-decile = 5×).

### #2 — calibrate the photo-z PDFs — ✅ DONE
Added the PDF-calibration primitives to `core/photoz.py` (`pit_values` —
bin-interpolated CDF; `credible_coverage` — central-interval coverage, which is
just `mean(|PIT-0.5| ≤ c/2)`; `pit_ks`; `temperature_scale` — exact via
`normalise(pdf**(1/T))`, no logits needed; `nll_at_truth`; `fit_temperature`,
objective `"nll"` or `"ks"`) and `experiments/calibrate_photoz.py` (recomputes
PDFs for the held-out spec set, no retrain).
**Finding (the interesting part):** the PIT is a **central hump** — the net's
PDFs are mildly **over-dispersed / under-confident** (PIT KS 0.18). But coverage
at the levels that matter is already good: a **90% credible interval contains the
truth 88%** of the time (95% → 90%). A single temperature **cannot** calibrate
it: NLL-optimal **T≈1.5** wants to *widen* (to cover the catastrophic-outlier
tails) while PIT/KS-optimal **T≈0.24** wants to *sharpen* (to flatten the core) —
and sharpening would drop 90% coverage to 78% (worse where it matters) and crush
the secondary interloper peaks. So we **ship T=1 (unscaled)** and *disclose* the
calibration rather than fake it. σ_NMAD unchanged (0.0403).
Saves `models/photoz_calibration.json`; patches a `calibration` block into
`web/dropout/dropout.json` (v3) + `photoz_meta.json`. UI: PIT histogram + an
honest verdict in the dropout "honest numbers" card (app.js v4). Tests:
`tests/test_calibration.py` (5) — suite **12 passing**.

### #3 — reproducibility hardening — ✅ DONE
- **Env pinned.** `requirements.txt` now pins every direct dep to the exact
  working versions (numpy 2.4.6, torch 2.12.1, timm 1.0.27, scikit-learn 1.9.0,
  umap-learn 0.5.12, onnxruntime 1.27.0, …), with python 3.13 + MPS recorded in
  the header. Previously-missing deps added (umap-learn, onnxscript, requests,
  scipy); unused `astroquery` dropped.
- **One-command repro: `Makefile`.** One target per milestone with data-fetch
  and training separated (`m1-data m1 … m4-data m4`), `stats` (significance +
  calibration — safe to re-run any time), `venv`, `test`, `serve`, `help`.
  Costs + the DJA rate-limit caveat are in the header comments; catalogue
  downloads are real file targets so make skips them when present.
- **Tests: 3 → 21.** New: `test_data.py` — `asinh_stretch(colour=True)`
  preserves band ratios *and* the per-channel stretch destroys them (the M3
  fix, both directions); `test_photoz.py` — featurize shape/determinism,
  brightness-scale invariance (+2 dex only in the last feature), non-detection
  masking, `point_estimates` two-mode logic, `soft_labels`; `test_anomaly.py` —
  planted-outlier ordering, `exclude_self` semantics, `nearest_distance`.

### #4 — `RESEARCH.md` — ✅ written (PRIVATE), Q1 done, Q2/Q3 staged
`RESEARCH.md` (repo root) is the research notebook: five questions (Q1 selection
function, Q2 interlopers vs EAZY, Q3 follow-up candidate list, Q4 cross-field
generalisation, Q5 = M5 NIRSpec contrastive), each with method/data/"what counts
as a result"/status. **User's call: keep it private, do Q1–Q3 in stages, and
only consider going public once something is definitely in hand.**
- **Q1 DONE** — `experiments/lrd_selection.py` (cached data, seed 0): AUC
  **0.702** (CI 0.655–0.748); top-2% cut = **10.2×** enrichment at 20.4%
  completeness; completeness saturates ~60% and **28.2% of LRDs score below
  the atlas median**. The missed tail is systematic, not random: fainter
  (F444W 0.14 vs 0.26 µJy, ρ=0.33 with score) and less red/dusty (Av 1.1 vs
  1.8, F444W/F200W 4.2 vs 6.0). Verdict in RESEARCH.md: an *enricher*, not a
  *selector* — selects the bright/red end, a measured selection bias that any
  Q3 candidate list must disclose.
- **Q2 DONE** — `experiments/interlopers.py`: an honest **null**. Point
  estimates are twins of EAZY (z>4: 81% complete / 5.9% interlopers vs 83% /
  6.3%, CIs overlap); the PDF confidence cut trades completeness 83→56% for no
  real interloper gain. Diagnosis: **10/12 interlopers are EAZY's own**
  (inherited via the training labels) and the PDFs are *confidently wrong* on
  them (median P(z<2) 0.038 vs 0.008 — a whisper, not a veto). Full writeup +
  what-would-help in RESEARCH.md Q2.
- **Q3 DONE** — `experiments/candidates.py` → `research/` (candidates.csv +
  two contact sheets, committed). The two rankings are **disjoint** (0 overlap
  in top-100s). Anomaly list: ~⅓ artefacts but real exotica (two extreme
  emission-line-looking green objects, an interacting pair, an LRD-like red
  dot) — novelty unverified vs literature. LRD-like list: **clean failure,
  mechanism found** — the per-cutout percentile stretch amplifies empty-field
  sky noise to colour static; LRD cutouts (r_eff~0.4px dots on empty fields)
  embed as that texture, so retrieval returns noise (median luminance 49 vs 10
  random; ρ(ldist,lum)=−0.20). This also *explains Q1's missed faint 28%*.
  Proposed fix: **M3b** — detection-anchored cutouts + noise-aware stretch
  (fixed physical scale, empty sky stays dark) + centroid-colour features,
  then re-run Q1/Q3. Full verdict in RESEARCH.md (notebook-grade methods
  story; not yet the go-public "something").
- **Q4 is the pre-publish gate** if Q3 finds anything: retrain minus one field,
  check the enrichment survives on the held-out field.

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
