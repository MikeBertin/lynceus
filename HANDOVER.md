# Lynceus — session handover

A from-scratch astrophysics-ML project: machine learning on JWST deep-field
galaxies, with a serious Python core and self-contained, no-build browser demos.
Lynceus is a **stand-alone** project — present it on its own, not as part of a
family. **This file is the
single source of truth for picking the project back up — read it top to bottom.**
The deep research detail lives in `RESEARCH.md` (private notebook); this file is
orientation + the next-session roadmap + the hard-won reference gotchas.

---

## 1. Status — where we are (2026-07)

**Everything through the research validation arc is done, committed, and pushed —
and the project is PUBLIC (2026-07-05).** Repo: `git@github.com:MikeBertin/lynceus.git`
(user `MikeBertin`, SSH), default branch `main`. Site live at
**https://mikebertin.github.io/lynceus/** (GitHub Pages, main `/docs`, HTTPS).
All four pages + card.png + ONNX models verified 200 on the live URL at publish.

Four demos, all live and verified; the **shipped atlas now runs the M3b
representation** (see §7):

| Milestone | What | Headline (shipped) |
|---|---|---|
| **M1 — Morphology** | In-browser ViT classifies a real JWST cutout (featured/smooth/merger) + attention maps | **81.0%** 5-fold CV vs 33% baseline, real Galaxy Zoo labels |
| **M2 — Atlas** | Self-supervised (SimCLR) 2-D embedding of **9,673** real JWST galaxies, WebGL fly-through | kNN-morphology **~62%** vs 39% — structure with no labels |
| **M3 — Anomaly hunt** (ships **M3b**) | Latent-space outlier score + 216 real Little Red Dots overlaid | LRDs **7.4×** over-represented in the top-10% (95% CI 6.8–7.9×, p<1e-4; median LRD at 96th pct); survives leave-one-field-out (Q4) |
| **M4 — Dropout hunter** | In-browser neural **photo-z** over 9 bands; Lyman-break scrubber + PDF + colour–colour | **σ_NMAD ≈ 0.040** vs ~1,800 held-out spec-z (EAZY ceiling 0.027) |

**The research arc (RESEARCH.md Q1–Q5) is essentially complete through Q4:**
- **Q1** (selection function), **Q2** (interlopers vs EAZY — clean null),
  **Q3** (candidate list — diagnosed the retrieval failure) → done.
- **M3b** (the Q3 fix: point-source-aware representation) → done, **now shipped
  in the demo**. Lifted AUC 0.70→0.91, killed the faint-tail blind spot.
- **Q3b** (star veto → literature cross-match → photo-z) → done. *Originally*
  read as 3–4 new candidates + 2 rediscoveries — but see Q3c Step 2: the "new
  candidates" do **not** survive catalogue-grade photometry.
- **Q3c** (candidates → confirmations) → **Steps 1–2 done, and they revise Q3b.**
  Step 1 (archival spectra): 0/10 have a public NIRSpec spectrum. Step 2
  (catalogue-grade selection replay vs DJA/COSMOS-Web parent catalogues):
  **0/8 novel candidates survive** — 5 are real galaxies that are *not red*
  (F277−F444≈0 vs LRDs' ≈1), 3 have no JWST counterpart; only the 2 already-known
  Akins LRDs pass. The Q3b redness was an artefact of plausibility-grade aperture
  photometry. **Net: the pipeline is a validated *method* (it rediscovered 2
  published LRDs), but yields no new LRD on these Galaxy-Zoo-seeded fields.**
- **Q4** (cross-field generalisation, the pre-publish gate) → **passes.** The
  anomaly signal survives on fields the encoder never trained on.
- **Q5** = M5 (image+spectra contrastive) → not started; the big future bet.

Recent commits (newest first): `81417b2` Q3c Step 1 (archival-spectrum null) ·
`9ebff21` ship M3b in demo · `14abae8` spectroscopic residual (Akins-CW +
RUBIES) · `f8766bc` Q4 passes · `ff3c659` Q3b · `8206ec4` M3b.

**Test suite: 25 passing.** Start every session with:
```bash
cd /Users/m/.openclaw/workspace/projects/lynceus && .venv/bin/python -m pytest -q
```

---

## 2. NEXT SESSION — start here

**The Q3c verdict is in, and it revises Q3b: there is no new LRD in this
candidate list.** Steps 1–2 are done (details below). Catalogue-grade photometry
collapses all 8 "novel" candidates (5 not-red galaxies, 3 non-detections); only
the 2 already-known Akins LRDs survive. The pipeline is a validated *method*
(it rediscovered 2 published LRDs with no labels) but is **not** a discovery
engine on these Galaxy-Zoo-seeded fields. So the decision for next session is
**strategic, not another step of this candidate list:**

- **Don't chase Step 3.** SED/MIRI strengthening has nothing left to strengthen —
  no candidate survives Step 2. Skip it.
- **The real lesson → M5 / a JWST-native parent.** The candidates failed largely
  because the atlas is seeded from **Galaxy Zoo CANDELS** (bright, HST-classified
  galaxies), so it can't reach the faint, JWST-catalogue-native LRD population.
  A genuine LRD hunt wants a faint JWST-detected parent sample (DJA/COSMOS-Web
  source lists) as the atlas, not GZ positions — and image+spectra contrastive
  (M5) is the paper-shaped bet. This is the recommended direction.
- **~~Publish?~~ PUBLISHED (2026-07-05).** The user accepted the honest reframe
  ("a validated anomaly-detection *method* that rediscovers known LRDs") and
  took it public; demo/README copy carries that framing. See §9 for the live
  URL and deploy mechanics.

Below: the completed Step 1–2 record. (Original Step 2/3 plan retained for
context, struck through where superseded.)

### Step 1 — archival-spectrum search ✅ DONE (2026-07-03)
**Result: a clean null — none of the 10 is spectroscopically observed yet.**
`python -m experiments.candidate_specsearch --validate`
(→ `research/candidates_specsearch.csv`). The DAWN spectroscopic archive
(**DJA-Spec**) is the *union* of every public NIRSpec program reduced with
msaexp (RUBIES/JADES/CEERS/UNCOVER/CAPERS/PRIMER), so one cone-search against
its master emission-line table (`dja_msaexp_emission_lines_v4.4`, 80,367
extractions w/ RA/Dec, `z_best`, `grade`, Balmer lines, `beta`; on S3) covers
them all. **Validated** exactly as Q3b: 90/260 known Kokorev LRDs have a
DJA-Spec spectrum within 1″ (median 0.01″ — pinpoint), so a non-match is real.
**0/10 candidates matched** (nearest 3.4″; unchanged at 2″). Even the two
Akins-CW *photometric* rediscoveries have no NIRSpec (55″/26″). So the caveat
can't be discharged from the archive today; all 10 stay a **follow-up target
list**, and this same re-run is the confirmation test when the next NIRSpec
release lands. Full detail: RESEARCH.md Q3c Step 1.

### Step 2 — parent-catalog selection replay ✅ DONE (2026-07-04)
**Result: 0/8 novel candidates survive catalogue-grade photometry.**
`python -m experiments.candidate_selection_replay` (auto-fetches the parent
catalogues → `research/candidates_selection_replay.csv`). Cross-matched each
candidate to the **DJA van der Wel+25 "morpho-phot" parent catalogues** (fluxes
+ Sérsic sizes; row count = the plain `_phot.fits` detection lists, so a
non-match is a genuine absence), recovered COSMOS-Web-outside-PRIMER sources from
the Akins table, and replayed the published cuts (Akins F277W−F444W>1.5+compact;
Kocevski β_UV<−0.37 & β_opt>0+compact; Kokorev red1/red2+compact). **Validated
both ways:** known Kokorev LRDs recover red (median F277W−F444W≈1.0) + compact
(~100%); the 2 Akins rediscoveries *pass* Akins on COSMOS-Web photometry. Verdict:
2 published-LRD (the rediscoveries), **5 below-cuts** (real galaxies, F277W−F444W
≈0 — *not red*), **3 no-JWST-counterpart** (spurious/off-footprint). Root cause:
Q3b's redness came from plausibility-grade aperture photometry; catalogue-grade
photometry deflates it. Full detail: RESEARCH.md Q3c Step 2. Compactness is
unified as r_h<1.5 r_h,stars (disclosed stand-in); "below-cuts" is colour-driven,
so robust. Uses the DJA parent catalogues (§5); ~2.5 GB, gitignored, auto-fetched.

### ~~Step 3 — SED / slope / MIRI strengthening~~ (superseded — nothing survives Step 2)
Was: strengthen whatever survived Step 2 into a ranked target list. Moot — no
novel candidate passed Step 2. See §2 top for the real next direction (M5 / a
JWST-native parent atlas).

### Standing options (user's call, not blocking)
- **M5 — image+spectra contrastive (NIRSpec).** The genuine new-science bet
  (AstroCLIP lineage). Step 1 above starts touching the same NIRSpec data.
- *(Publish — done 2026-07-05; see §9. Keep presenting Lynceus **stand-alone** —
  no "companion to"/"sibling" framing anywhere public. Remember `docs/` on
  `main` auto-deploys now.)*

House style to keep: serious `core/`, no-build demo, **honest numbers in the UI**,
"things to notice" cards, versioned web assets, everything reproducible via the
`Makefile`.

---

## 3. Environment (important — non-obvious)

- **Machine:** Apple **M4 Mac mini, 10-core GPU, 24 GB unified**. No CUDA — use
  PyTorch **MPS**. Training is device-agnostic via `core.config.get_device()`.
- **Python:** system python is **3.14** (too new for torch wheels). Use the repo
  venv on **3.13**: `projects/lynceus/.venv`. Always call `.venv/bin/python`.
  `requirements.txt` is **pinned**; `make venv` recreates it.
- **Background shells** start from a fresh cwd — use **absolute paths**
  (`/Users/m/.openclaw/workspace/projects/lynceus/...`). Heredocs that `cd` into
  scratch can lose `.venv` on the PATH; prefer absolute `.venv/bin/python`.
- **Foreground `sleep` is blocked.** Wait on a background job with a
  `run_in_background` watcher (`until grep -q DONE log; do sleep 5; done`) or just
  launch long jobs with `run_in_background: true` and get notified on completion.
- **`torch.load` our own checkpoints with `weights_only=False`** (they carry
  numpy mu/sd arrays; PyTorch 2.6 defaults to `True` and errors).
- **DJA cutout service is rate-limited** — be gentle (`--workers 6`). But raw
  cubes are now cached (§5), so stretch/representation experiments never re-fetch.

---

## 4. Repo map

```
core/                    # reusable package (the serious core)
  config.py              #   paths, bands, device select, cutout sizes
  data.py                #   cutout data layer: synthetic Sersic + DJA service
                         #     (fetch_service_cube / _bands) + asinh stretches:
                         #     asinh_stretch (colour/per-channel), asinh_stretch_snr
                         #     (M3b noise-aware), centre_anchor_crop, m3b_cutout
  datasets.py            #   torch Dataset + augmentation (M1)
  models.py              #   timm ViT + attention rollout (M1)
  morphology.py          #   M1 label schema + metrics
  train.py               #   device-agnostic train loop + k-fold CV (M1)
  ssl.py                 #   M2 SimCLR (ResNet-18, NT-Xent), 128px, MPS
  embed.py               #   encoder->features->cosine UMAP->2D layout
  anomaly.py             #   kNN cosine anomaly score + nearest-distance
  photoz.py              #   M4 photo-z: featurize + binned-softmax MLP + PIT/coverage
  stats.py               #   bootstrap_ci + permutation_p (seeded)
experiments/             # one-shot scripts (run as `python -m experiments.X`)
  # --- M1 morphology ---
  fetch_gz.py, fetch_ceers.py, fetch_data.py, train_vit.py, export_onnx.py,
  build_web_assets.py
  # --- M2 atlas / M3 anomaly ---
  fetch_atlas.py         #   unlabelled JWST cutouts (stratified by GZ class)
  train_atlas.py         #   SimCLR encoder; --cutouts-dir / --out / --exclude-field
  fetch_lrd.py           #   216 Kokorev+24 LRDs, fetched + embedded
  build_atlas.py         #   embed->UMAP->sprites + atlas.json + anomaly + lrds.json; --m3b
  restretch_atlas.py     #   (legacy) re-fetch atlas cutouts with colour stretch
  significance.py        #   bootstrap CIs + perm nulls -> web JSON; --m3b
  # --- M3b (point-source-aware representation) ---
  fetch_raw.py           #   cache RAW service cubes (atlas+LRD) -> stretches go local
  build_m3b_cutouts.py   #   raw -> snr-stretch + anchored 64px crop -> cutouts_m3b/
  embed_m3b.py           #   embed m3b cutouts with ssl_encoder_m3b.pt
  # --- research Q1-Q4 ---
  lrd_selection.py       #   Q1 selection function (AUC, operating points); --m3b
  interlopers.py         #   Q2 interloper rejection vs EAZY
  candidates.py          #   Q3 ranked candidate lists + contact sheets; --m3b
  star_veto.py           #   Q3b star veto (PSF spikes + colour + Gaia)
  candidate_crossmatch.py#   Q3b literature cross-match (auto-downloads catalogs)
  candidate_photoz.py    #   Q3b photo-z sanity check (9-band aperture phot)
  candidate_specsearch.py#   Q3c Step 1 archival-spectrum search vs DJA-Spec; --validate
  candidate_selection_replay.py # Q3c Step 2 parent-catalog replay of Akins/Kocevski/Kokorev cuts
  q4_generalise.py       #   Q4 leave-one-field-out generalisation eval; --field/--all
  # --- M4 dropout ---
  build_photoz_dataset.py, train_photoz.py, export_photoz_onnx.py,
  build_dropout_assets.py, calibrate_photoz.py
docs/                    # the site (GitHub Pages serves main //docs)
  index.html             #   landing (4 cards, all Live)
  shared/                #   theme.css (?v=2), notice.js, favicon.svg
  morphology/            #   Demo A: ONNX classifier
  atlas/                 #   Demo B: atlas + anomaly + globe. app.js (?v=17),
                         #     atlas.json/lrds.json (?v=7), sprites/lrd_sprites (?v=5,
                         #     64px native tiles); morphology gallery 224px (?v=2)
  dropout/               #   Demo D: photo-z dropout hunter
tests/                   # test_data / _models / _photoz / _anomaly / _stats / _calibration (25)
data/                    # GITIGNORED — all catalogs/cutouts/embeddings (regenerable)
models/                  # GITIGNORED — checkpoints, onnx, *.json
research/                # COMMITTED — candidate CSVs + contact sheets (notebook evidence)
.venv/                   # GITIGNORED
```

`README.md` + `plan.md` = public-facing narrative/roadmap. `RESEARCH.md` = the
private research notebook (Q1–Q5, full method+results+caveats). The assistant's
memory store has a condensed version.

---

## 5. Data, catalogs & pipelines — all public, all regenerable

`data/` and `models/` are gitignored. The whole DAG is a `Makefile`
(`make help`): `m1-data m1 … m4-data m4`, `m3b-data m3b`, `stats`, `venv`, `test`,
`serve`. Rough costs in the header; only `*-data` targets touch the network.

**Cutout cache (the key M3b change).** The cutout cache historically stored only
*stretched* images, so every stretch change meant another DJA pull. Now the **raw
service cubes are cached** (`data/atlas/raw/`, `data/lrd/raw/` — 9,889 cubes,
`make m3b-data`), so any future representation experiment is a purely local
rebuild. `build_m3b_cutouts.py` turns raw → `cutouts_m3b/` (64px, noise-aware
stretch + detection-anchored crop). **Aperture photometry from service cutouts
(`candidate_photoz.py`) is plausibility-grade, not measurement-grade** — for the
Step-2 selection replay use parent-catalog fluxes instead.

**External services / catalogs (all worked):**
- **DJA grizli cutout service** — real NIRCam cutouts by RA/Dec, no mosaic
  download: `https://grizli-cutout.herokuapp.com/thumb?ra=..&dec=..&size=3&filters=f444w-clear,f356w-clear,f200w-clear&output=fits`.
  Covers GOODS-S, COSMOS, UDS, EGS. `data.fetch_service_cube` / `_bands`.
- **Galaxy Zoo CANDELS** (Simmons+17): `data.galaxyzoo.org`. GOODS-S/COSMOS/UDS.
- **CEERS grizli/EAZY catalog** (van der Wel+25, 375 MB): fluxes + z_phot +
  ~1,800 z_spec for M4. `s3.amazonaws.com/aurelien-sepp/ceers-full-grizli-v7.2/...`.
- **Kokorev+24 LRD** (`github.com/VasilyKokorev/lrd_phot`, `lrd_table_v1.1.fits`):
  260 LRDs, 216 in our fields; ra/dec/band-fluxes/z_phot/r_eff/av/lbol/muv.
- **LRD literature catalogs (Q3b, auto-downloaded by `candidate_crossmatch.py`):**
  - `data/lrd_kocevski24.dat` — full Kocevski+24 341-LRD Table 3, from GitHub
    `dalekocevski/Kocevski24` (NOT on VizieR; 254 in our fields).
  - `data/lrd_akins24_cosmosweb.ecsv` — full Akins+24 COSMOS-Web 434-LRD Table 1,
    from GitHub `hollisakins/akins24_cw` (ECSV; complete COSMOS coverage).
  - RUBIES broad-Balmer census — VizieR `J/A+A/702/A57/tableb1` (UDS+EGS spec).
  - Perger+25 compilation `J/A+A/693/L2`, Euclid LRD `J/A+A/711/A24`.
  - **NEXUS is the NEP field — no overlap, correctly excluded.**
- **DJA-Spec master NIRSpec table (Q3c Step 1, auto-downloaded, ~130 MB):**
  `data/dja_msaexp_emission_lines_v4.4.csv.gz` from
  `s3.amazonaws.com/msaexp-nirspec/extractions/` — 80,367 public NIRSpec
  extractions (the union of all msaexp-reduced programs); 587 cols incl.
  ra/dec/`z_best`/`grade`(3=secure)/Balmer-line fluxes/`beta`. Position-searchable
  master of every public spectrum, so it subsumes RUBIES/JADES/CEERS/etc.
- **DJA parent photometric catalogues (Q3c Step 2, auto-downloaded, ~2.5 GB):**
  `data/dja_phot/{root}_morpho-phot.fits.gz` for `gds-grizli-v7.2`,
  `primer-uds-north/south-grizli-v7.2`, `primer-cosmos-west/east-grizli-v7.0`
  from `s3.amazonaws.com/aurelien-sepp/` (van der Wel+25 morphology+photometry;
  ~354k sources; HDU1 = fluxes `{band}_flux_aper_1`/`_tot_1` in µJy + `flux_radius`
  px @0.04″; HDU2 = single-Sérsic `RADIUS` in **deg** + `SERSIC`). Row count = the
  plain `{root}_phot.fits` detection lists, so these *are* the parent detection
  catalogues. **PRIMER-COSMOS ≠ COSMOS-Web** — COSMOS-Web-only sources (e.g. the
  2 Akins LRDs) miss PRIMER and are recovered from `lrd_akins24_cosmosweb.ecsv`.

**Fields:** GOODS-S (RA~53.1, b−54°, `#5ec27a`), COSMOS (RA~150.1, b+42°,
`#ff9e64`), UDS (RA~34.4, b−60°, `#c792ea`). All high galactic latitude.

**Key model checkpoints (in `models/`, gitignored):**
`ssl_encoder.pt` (original M3), `ssl_encoder_m3b.pt` (**shipped**),
`ssl_encoder_m3b_no{UDS,GOODS-S,COSMOS}.pt` (Q4 leave-one-out),
`ssl_encoder_2389.pt` / `_perchannel.pt` (old), `vit.pt`, `photoz.pt`.
Embeddings: `data/atlas/embeddings{,_m3b}.npy`, `data/lrd/lrd_emb{,_m3b}.npy`.

---

## 6. Web / demo architecture (read before touching the atlas JS)

Vanilla JS, no build step, static server. Hard-won gotchas — **all have bitten us:**
- **Cache-busting is mandatory.** Every fetched JS/CSS/JSON is `?v=N`. **Change a
  file → bump its version** or the browser serves stale. Current atlas versions:
  `app.js?v=17`, `atlas.json?v=7`, `lrds.json?v=7`, `sprites.jpg?v=5`,
  `lrd_sprites.jpg?v=5`, `theme.css?v=2`, morphology `gallery.json?v=2`
  (+ per-image `?v=2` baked into its img/attn paths by `build_web_assets`).
- **No `requestAnimationFrame` render loop** — it stalls in the headless preview.
  The atlas uses microtask-coalesced draws (`markDirty()` → `Promise.resolve().then(draw)`).
  (The globe's cosmetic auto-spin uses rAF; fine.)
- **Canvas sizing race:** `clientWidth` can read 0 before layout. Fallback width +
  ResizeObserver; map pointer coords by `canvas.width/rect.width`. See `resize()`/`evPos()`.
- `atlas/app.js` is the big one. Colour modes `region|morph|featured|merger|anomaly`
  (order = narrative). The **anomaly stat line is data-driven** — it reads
  enrichment/CI/p/median_pct straight from `lrds.json`, so re-running
  `build_atlas --m3b` / `significance --m3b` updates the UI number automatically
  (only the prose in `index.html` is hand-written).
- **Preview:** the preview tool reads the **workspace-level**
  `projects/.claude/launch.json`. `lynceus-web` = port 8137, serves `lynceus/docs`
  (updated from `lynceus/web` when the site moved to `docs/` for Pages).
  Verified this session: no console/network errors; anomaly mode shows LRDs
  piling into the hot zone at 7.36×/96th pct; morphology still separates; zoomed
  thumbnails render from the new M3b sprite sheet.

---

## 7. Key results (for copy / honesty)

- **M1:** 81.0% acc, macro-F1 0.81 (`vit_small_patch16_224`, INT8 → 22 MB ONNX).
  Pinned to the original per-channel stretch (`asinh_stretch(colour=False)`) so
  its shipped 81% reproduces.
- **M2:** SimCLR ResNet-18, 80 epochs. Shipped M3b atlas: kNN-morph **61.6%±0.8%**
  (baseline 38.9%), kNN-field **63.0%±0.4%** (baseline 40.6%, k=15 5-fold) — the
  63% field-guessability is a survey depth/PSF fingerprint (Q4 shows it doesn't
  drive the anomaly signal).
- **M3 → SHIPPED = M3b.** Anomaly = mean cosine distance to 20 NN in 512-D.
  **7.4× enrichment** in the top-10% (95% CI 6.8–7.9×, p<1e-4 vs a random-216 null;
  median LRD at the **96th** percentile). The M3b representation:
  `asinh_stretch_snr` (per-band sky subtraction; shared intensity scaled by the
  cutout's own MAD sky-σ: 2σ soften / 300σ cap / 1σ floor, so **empty sky stays
  dark** instead of amplifying to colour static) + `centre_anchor_crop` (64px,
  ±8px snap onto the central source). Chain of the number over the project:
  **2× → 3.4× → 5× → 7.4×** (per-channel → colour-preserving → larger atlas →
  point-source-aware). Rebuild + ship: `make m3b` (or `build_atlas --m3b` +
  `significance --m3b`). `models/significance_m3b.json` holds the numbers.
  Q1 on M3b: **AUC 0.910** (was 0.702); faint-tail blind spot 28%→1.9%.
- **M4:** photo-z σ_NMAD **0.0403** (95% CI 0.0374–0.0438), EAZY-distilled. PDFs
  mildly over-dispersed but a 90% credible interval covers ~88% of the truth;
  ships **T=1 (unscaled)** and discloses the calibration (NLL wants T≈1.5 widen,
  PIT wants T≈0.24 sharpen — they conflict, so don't fake it).
- **The research arc's headline — REVISED by Q3c Step 2.** Q3b *originally* read
  as 3–4 new candidates + 2 rediscoveries. Catalogue-grade photometry (Q3c Step 2)
  **removes the new candidates**: measured in the DJA/COSMOS-Web parent catalogues
  they are not red (F277W−F444W≈0 vs LRDs' ≈1) or are not catalogued JWST sources.
  What stands is the **2 independent rediscoveries** of published Akins-CW LRDs
  (758982/758452, 0.14"), which pass the Akins cuts — a real, label-free
  validation of the anomaly→retrieval **method**, not a new discovery. Honest
  net: a validated LRD-*rediscovery* pipeline; no new LRD on these fields.
  **Full detail + all caveats in RESEARCH.md (Q1–Q4 + M3b + Q3b + Q3c sections).**

---

## 8. Build/export gotchas (Python side)

- ONNX export: **`torch.onnx.export(..., dynamo=False)`** (dynamo splits weights
  to an external `.data` file → breaks onnxruntime-web). Needs `onnxscript`.
- ONNX quantise: **`quantize_dynamic(..., op_types_to_quantize=["MatMul"])`** only
  — wasm backend has no ConvInteger, so leave the patch-embed Conv in fp32.
- UMAP (`embed.umap_2d`): `n_neighbors=30, min_dist=0.5, spread=1.4`, cosine;
  then per-axis percentile fill in `normalise_coords` (edge "tendrils" are clipped outliers).
- Kocevski `.dat` parsing: the ID column is 1–2 whitespace tokens, so parse the
  11 trailing numeric fields by right-index (`r[-11]`=ra … `r[-8]`=zbest).

---

## 9. Git & publishing

- All work on `main`, pushed, **PUBLIC** (flipped 2026-07-05 on the user's
  explicit ask). Commit style: present-tense subject + body, ending
  `Co-Authored-By: Claude <model name> <noreply@anthropic.com>` with the model
  actually driving (history has Fable 5 and Opus 4.8).
- **Committed site assets** (the demos need them; `data/`+`models/` are not
  committed): `docs/morphology/model.onnx` (22 MB); `docs/atlas/sprites.jpg`
  (~4.5 MB, 48px tiles, 9,673 galaxies), `atlas.json` (~1 MB), `lrds.json`,
  `lrd_sprites.jpg`, `skygeom.json`; `docs/dropout/photoz.onnx` (487 KB),
  `dropout.json`, `filmstrips.jpg`; `docs/shared/favicon.svg`; `docs/card.png`
  (og:image); `research/*.csv` + `research/sheet_*.jpg`. Atlas first-load ~5.6 MB.
- **PUBLISHED (2026-07-05, on the user's explicit ask).** Site moved `web/` →
  `docs/`; Pages = **main //docs**, HTTPS enforced; `og:image` card
  (`experiments/build_og_card.py` → `docs/card.png`) + `<meta>` on all four
  pages; README carries the live link. Flip was:
  `gh repo edit MikeBertin/lynceus --visibility public
  --accept-visibility-change-consequences` + `gh api -X POST
  repos/MikeBertin/lynceus/pages -f 'source[branch]=main' -f 'source[path]=/docs'`.
  Verified live: all four pages, card.png, both ONNX models, versioned assets →
  200 at https://mikebertin.github.io/lynceus/. **Pages auto-deploys `docs/` on
  every push to `main` now — anything committed there goes live in ~1 min.**
  Present Lynceus **stand-alone** — no cross-project framing.
