# Lynceus — research notes

**Private working notes** — the research on-ramp made explicit. Not written for
an outside reader yet; goes public (in some cleaned-up form) only once one of
these questions has a result worth showing. Repo stays private until then.

The toolkit in hand: a SimCLR encoder over ~9.7k real JWST cutouts with a
validated anomaly score (LRDs 5× enriched, p<1e-4 — see `experiments/
significance.py`), and a photo-z net with honest, calibration-checked PDFs vs
~1.8k spec-z (`experiments/calibrate_photoz.py`). Everything below is a
question that toolkit can attack, ordered by cost.

| # | Question | Cost | Status |
|---|----------|------|--------|
| Q1 | Is the anomaly score an LRD **selection function**? | cached data, hours | ✅ **done** — AUC 0.70; an enricher (10× at top-2%), not a selector; misses the faint/less-red 28% → **fixed by M3b** |
| Q2 | Does the net beat EAZY on **interloper rejection**? | cached data, hours | ✅ **done** — honest null: inherits its teacher's interlopers (10/12 shared), PDFs are confidently wrong |
| Q3 | A ranked **spectroscopic follow-up list** | cached data, hours | ✅ **done** — anomaly list: real exotica amid ~⅓ artefacts; supervised LRD-retrieval fails (noise-texture matching, mechanism diagnosed → M3b) |
| M3b | **Point-source-aware representation** (the Q3 fix) | refetch + retrain, ~hours | ✅ **done** — AUC 0.70→**0.91**, missed tail 28%→**1.9%**, retrieval now returns point sources (stars + compact red dots), not noise |
| Q4 | Does the encoder **generalise across fields**? | retrain, ~hours MPS | queued — **the pre-publish gate** |
| Q5 | **M5: image+spectra contrastive** (NIRSpec) | new data, weeks | the big bet |

---

## Q1 — Is the anomaly score an LRD selection function?

**Question.** The 5× enrichment says LRDs are over-represented among anomalies.
A referee's next question: could you *find* LRDs with it? That's a selection
function — completeness and purity as a function of anomaly threshold, not a
single enrichment number.

**Data in hand.** `data/atlas/embeddings.npy` (9673×512),
`data/lrd/lrd_emb.npy` (216×512), scored with `core.anomaly.knn_anomaly`
exactly as in `experiments/build_atlas.py`.

**Method.** Treat known LRDs (Kokorev+24) as positives, atlas galaxies as
(approximate) negatives. ROC — completeness vs false-positive rate — as the
threshold sweeps; AUC with a bootstrap CI (AUC over a fixed negative set is the
mean of the per-LRD percentiles, so `core.stats.bootstrap_ci` applies
directly). Report operating points: keep the top 1/2/5/10/20% of the atlas by
anomaly, what fraction of LRDs do you catch, at what enrichment over chance?

**Caveats (stated up front).** The negatives are the *atlas sample* — stratified
by Galaxy Zoo class for the demo's colouring, not flux-limited — and may contain
LRD-like objects not in Kokorev. So this is a selection function *relative to
this atlas*, a proof of concept rather than a survey-grade measurement.

**What counts as a result.** AUC meaningfully above 0.5 with a usable knee —
e.g. an operating point catching ≳half the LRDs while rejecting ≳90% of the
atlas — plus the purity-vs-completeness trade-off made explicit, would justify
Q3 (using the score to *rank candidates*).

**Status / results.** ✅ Done — `python -m experiments.lrd_selection`
(k=20, seed 0; writes `models/lrd_selection.json`).

- **AUC = 0.702** (bootstrap 95% CI 0.655–0.748; 216 LRDs vs 9,673 atlas).
- Operating points (completeness, with 95% CIs; enrichment = completeness/frac):

  | keep top | completeness | enrichment |
  |---:|---:|---:|
  | 1% | 9.7% (6.0–13.9) | 9.7× |
  | 2% | 20.4% (15.3–25.9) | **10.2×** |
  | 5% | 35.6% (29.6–42.1) | 7.1× |
  | 10% | 50.0% (43.5–56.5) | 5.0× |
  | 20% | 60.6% (54.2–67.1) | 3.0× |

- **The verdict: an *enricher*, not a *selector*.** The top-2% cut is the knee
  (10× over chance), but completeness saturates near ~60% — and **28.2% of
  known LRDs score *below the atlas median***: no anomaly threshold can reach
  them.
- **The missed tail is not random.** Cross-matched against the Kokorev
  catalogue, missed (below-median) vs caught (top-decile) LRDs differ
  systematically: F444W flux **0.14 vs 0.26 µJy** (score correlates with
  log-flux, Spearman ρ=0.33), dust **Av 1.1 vs 1.8** (ρ=0.18), colour
  **F444W/F200W 4.2 vs 6.0**. The score selects the **bright, red end** of the
  LRD population — precisely the two things the encoder was built to see
  (colour-preserving stretch + compactness), which is both a validation and a
  measured selection bias.
- **Implication for Q3.** A candidate list from the top 1–2% is worth building
  (each pick is ~10× enriched), but it inherits this bias: it will find more
  bright/red LRD-likes and cannot find the faint tail. Any claim from Q3 must
  carry that caveat. A supervised alternative (`nearest_distance` to the known
  216) is the natural complement and would have a *different* bias — worth
  running both.

## Q2 — Does the photo-z net beat EAZY on interloper rejection?

**Question.** The classic failure mode behind "too many bright early galaxies":
a dusty z~2 galaxy whose Balmer/4000 Å break mimics a Lyman dropout. Our net
outputs a full PDF and visibly keeps a second mode when unsure. Does that
translate into a *lower interloper rate* than the EAZY template fit it distils?

**Data in hand.** `data/photoz/photoz.npz` holds, side by side for the ~1.8k
spec-z validation galaxies: our features (→ PDFs via `models/photoz.pt`), the
EAZY `z_phot`, and the spectroscopic truth. No new data needed.

**Method.** Define a high-z-candidate subset (e.g. either method places
z > 4, and again z > 6); count each method's interlopers (candidates with
z_spec < 2). Then use what EAZY can't do here: *PDF-aware* selection — require
P(z > 4) above a confidence cut rather than a point estimate, and trace the
candidate-yield vs interloper-rate curve. Bootstrap CIs on the rates
(`core/stats.py`). Also worth reporting: the by-slice calibration (the z 6–12
slice, n=78, is where σ_NMAD degrades to 0.083 — is the PDF honest there?).

**What counts as a result.** Same candidate yield at a measurably lower
interloper rate (or the converse trade-off made explicit) — CIs that separate.
A null result is publishable-honest too: "a distilled emulator inherits its
teacher's interlopers" is worth knowing.

**Status / results.** ✅ Done — `python -m experiments.interlopers`
(writes `models/interlopers.json`). **The null, and it's clean.**

- **Point estimates are twins** (as a distillation must be): at z>4, EAZY
  82.9% complete / 6.3% interlopers vs ours 81.1% / 5.9% — CIs fully overlap.
  Same story at z>6 (70.5%/9.0% vs 66.7%/8.2%, n=78 truths, wide CIs).
- **The PDF confidence cut buys ~nothing**: P(z>4)>0.9 drops completeness
  83→56% while interlopers go 6.3→5.2% (CI 2–9%, overlapping everything).
  At z>6, P>0.9 reads 5.9% but on 34 selected (CI 0–15%) — noise.
- **Why (the diagnosis, and the real finding):**
  1. *The interlopers are inherited.* 10 of our 12 z>4 interlopers are also
     EAZY's (5/6 at z>6). The net trained on EAZY labels, so EAZY's mistakes
     were mislabelled *in training* — the student learned them faithfully.
  2. *The PDFs are confidently wrong on exactly those galaxies.* Median
     P(z<2) is 0.038 for interlopers vs 0.008 for true high-z — a 5× whisper —
     but only 8% of interlopers carry meaningful (>0.2) low-z mass, identical
     to true high-z (9%). The two-humped-PDF safety net is real in the median
     and useless as a veto. Consistent with the calibration study: the net's
     uncertainty is *generic width*, not instance-specific doubt.
- **What would actually help** (candidate M4b, if ever needed): train on
  spec-z (or SED-fit posteriors) instead of point labels; or ensemble/augment
  with the medium bands EAZY underweights. Distillation cannot beat the
  teacher where the teacher defines the truth.
- Caveats: spec-z sample is selection-biased (bright, confirmable); this
  measures *relative* behaviour on identical galaxies, not survey rates.

## Q3 — A ranked spectroscopic follow-up candidate list

**Question.** The payoff question: point the anomaly score at the atlas and
hand over targets. The top-N most anomalous galaxies that are *not* known LRDs
— what are they? Genuinely new LRD candidates, other rare classes (mergers,
lenses, artefacts), or noise?

**Data in hand.** Everything Q1 uses, plus per-galaxy RA/Dec in
`data/atlas/atlas_meta.csv` and cutout PNGs for visual inspection.

**Method.** Rank the atlas by anomaly; drop known Kokorev matches (cross-match
by RA/Dec, ~1″); for the top ~50, assemble an inspection sheet (cutout, score,
field, GZ votes, nearest-LRD distance `lc`). Eyeball-classify; where DJA has
photometry, run our own photo-z on them. The Q1 selection function gives each
candidate an expected purity.

**What counts as a result.** Even a handful of credible, previously-uncatalogued
LRD candidates (compact + red + high anomaly + plausible photo-z) would be the
"we definitely have something" moment — a concrete deliverable no demo can fake.
Also the honest failure mode: if the top of the list is all artefacts, the
anomaly score is a data-quality detector, and that's the finding.

**Status / results.** ✅ Done — `python -m experiments.candidates` writes
`research/candidates.csv` + two contact sheets (`research/sheet_anomaly.jpg`,
`research/sheet_lrdlike.jpg`). Both outcomes at once: real targets *and* a
diagnosed failure.

- **The two rankings are completely disjoint** — zero overlap between the
  top-100 anomaly-ranked and top-100 LRD-like-ranked lists. They probe
  different directions: isolation vs proximity-to-exemplars.
- **Anomaly sheet (visual census of the top 32):** roughly ⅓ artefacts
  (diffraction spikes, guide-star stripes, noise blobs) — the anomaly score is
  partly a data-quality detector, as anticipated. But the rest are real and
  some are striking: two bright green compact objects (#5, #14 — extreme
  emission-line galaxies by appearance, [OIII]-blob/"green pea"-like), an
  interacting pair (#21/#22 — apparently the same system entering twice, so
  the atlas has near-duplicate catalogue entries), several edge-on red disks,
  and one compact orange-red dot (#6) that is morphologically LRD-like.
  Novelty **unverified** — claiming any of these as new requires cross-matching
  the literature (not done; would be the next step for a real follow-up case).
- **LRD-like sheet: a clean method failure, mechanism diagnosed.** The top-32
  are almost all full-frame colour **static** (plus one star). Mechanism: LRDs
  are tiny (r_eff ~0.4 px) dots on near-empty fields; the per-cutout percentile
  stretch **amplifies empty-sky noise to full range** when there is nothing
  bright to anchor it; the encoder embeds that static as a coherent texture;
  the 216 known-LRD cutouts live in/near that texture region; so
  nearest-to-LRD retrieval returns noise. Quantified: median cutout luminance
  49 (top-100 LRD-like) vs 10 (random); Spearman(nearest-LRD distance,
  luminance) = −0.20 on a 1k sample.
- **This closes the loop on Q1's missed tail**: faint LRDs embed as
  noise-texture, not as red dots — which is *why* no anomaly threshold reaches
  the faint 28%, and why the score only selects the bright/red end.
- **The 4 in-atlas Kokorev "matches"** (mean anomaly percentile 42, initially
  puzzling) resolve the same way: at 0.3–1.4″ separation they are mostly
  bright GZ galaxies *adjacent* to an LRD — the cutout embeds the neighbour.
- **M3b (the concrete method fix this earns):** make the representation
  point-source aware — detection-anchored small cutouts; a noise-aware stretch
  (fixed physical surface-brightness scale instead of per-cutout percentiles,
  so empty sky stays dark); and/or explicit centroid-colour features. Then
  re-run Q1+Q3: if supervised retrieval starts returning compact red things,
  the candidate list becomes real.

**Verdict for the go-public question.** Q1–Q3 together are a coherent,
honest *methods* story (a validated-but-biased selector, an inherited-label
null, a diagnosed retrieval failure with a fix) — notebook-grade, not yet the
"definitely have something". The something, if it comes, is M3b + a re-run Q3
list, or the anomaly sheet's emission-line objects surviving a literature
cross-match. *(Update: M3b is now done — see the M3b section. The fix worked;
what remains before "definitely have something" is the star veto + literature
cross-match, and Q4 as the pre-publish gate.)*

## M3b — the point-source-aware representation (the Q3 fix, executed)

**What.** Q3 diagnosed the failure: the per-cutout percentile stretch amplifies
empty-field sky noise to full-range colour static; faint LRDs (r_eff ~0.4 px on
near-empty fields) embed as that texture, so supervised retrieval returns noise
and the anomaly score can't reach the faint 28%. M3b changes the
representation, not the model class:

1. **Raw cubes cached once** (`experiments/fetch_raw.py` → `data/atlas/raw/`,
   `data/lrd/raw/`; 9,889 fetched, 0 failures). Every stretch experiment is now
   a local rebuild — the DJA service is never needed for this again.
2. **Noise-aware stretch** (`core.data.asinh_stretch_snr`): per-band sky
   subtraction, then the shared intensity is expressed in units of its own
   MAD-estimated sky σ — asinh softening at 2σ, saturation at 300σ, 1σ noise
   floor. Empty sky is *black* by construction; the Lupton-style linear
   redistribution keeps flux ratios (colour) exactly as before.
3. **Detection-anchored 64 px crop** (`core.data.centre_anchor_crop`): snap to
   the smoothed intensity peak within ±8 px of the catalogue centre (a wider
   window latches onto bright neighbours), so the target fills ~3.5× more of
   the frame.
4. SimCLR retrained from scratch on the new cutouts (80 epochs, NT-Xent → 1.78
   vs 1.81 shipped), atlas + LRDs re-embedded, Q1 + Q3 re-run. Everything is
   `make m3b-data` + `make m3b`; the shipped M3 artefacts are untouched
   (`*_m3b` files throughout). Tests cover the new stretch/crop properties
   (25 passing).

**Q1 re-run — the selection function is transformed** (`lrd_selection --m3b`,
`models/lrd_selection_m3b.json`):

- **AUC 0.702 → 0.910** (bootstrap 95% CI 0.890–0.928).
- Operating points (completeness, enrichment): top-1% 14.4% / 14.4×; top-2%
  36.6% / **18.3×**; top-5% 59.3% / 11.9×; top-10% **73.6%** / 7.4×; top-20%
  87.0% / 4.4×.
- **The unreachable tail is gone: 28.2% → 1.9%** of LRDs below the atlas
  median (n=4).
- **The brightness selection bias is gone**: Spearman(score, log F444W)
  0.33 → **−0.09**, score~Av 0.18 → 0.04. The four missed LRDs are, if
  anything, *brighter* than the caught ones (0.45 vs 0.21 µJy median) — small-n
  curiosities, not a systematic faint-end blind spot.
- kNN-morphology probe on the m3b embeddings: **61.6% ± 0.8%** vs 62.0% ± 0.7%
  shipped (field 63.0% vs 63.4%) — point-source awareness cost nothing on the
  atlas's general structure.

**Q3 re-run — retrieval now fails in the *right* way** (`candidates --m3b`,
`research/candidates_m3b.csv` + `sheet_*_m3b.jpg`):

- The LRD-like sheet went from 32/32 colour static to **32/32 point sources**:
  ~22 are stars (unmistakable six-spike JWST PSF), ~9–10 are **compact
  orange/red dots** — exactly candidate-shaped (e.g. #18, #20–22, #24, #29,
  #30, #32).
- The luminance diagnostic flipped: top-100 LRD-like median 8.9 vs random 12.8
  (was 49 vs 10). Retrievals are now *darker* than random — dark-field point
  sources; the residual ρ(ldist, lum) = +0.46 just says bright extended
  galaxies are far from LRDs, which is correct physics.
- **The contaminant class is the real one.** Stars/brown dwarfs are *the*
  known contaminant in photometric LRD searches (Kokorev+24 do explicit
  brown-dwarf vetting) — the encoder now makes the same confusion the
  literature fights, instead of a pipeline artefact. That is what "the
  representation is trustworthy" looks like at this stage.
- Anomaly list: qualitatively as before (~⅓ artefacts + real exotica; the two
  top-100 lists still nearly disjoint, overlap 1). The 4 in-atlas Kokorev
  positional matches still embed the bright *neighbour* (0.3–1.4″ ≫ the
  ±8 px ≈ 0.2″ anchor window) — unchanged caveat.

**What M3b earns / next steps.**

1. **A star veto** turns the LRD-like list into a genuine candidate list:
   diffraction-spike/PSF morphology, point-source colour vs the stellar locus,
   or a Gaia/point-source-catalogue cross-match. With ~⅓ of the top-32 already
   compact-red, even a crude veto should yield a clean top-20.
2. **Literature cross-match** of the surviving compact-red dots (novelty check)
   — this remains the "definitely have something" gate from Q3.
3. **Q4 (cross-field generalisation) before any public claim** — now more
   important, not less: the M3b numbers are strong enough that the depth/PSF
   fingerprint question is the main remaining threat.

## Q4 — Does the encoder generalise across fields?

**Question.** Our own probe says a classifier can guess the survey field 63% of
the time from the embedding — depth/PSF fingerprints. Does the science transfer
anyway? Train the encoder on two fields, test on the third: does
morphology-kNN hold up, do the held-out field's LRDs still light up?

**Method.** Retrain SimCLR on GOODS-S+UDS cutouts only (~hours on MPS);
re-embed COSMOS + its LRDs with the frozen encoder; recompute kNN-morphology
and LRD enrichment on the held-out field. Compare to the all-field encoder.

**What counts as a result.** Enrichment surviving (CI overlapping the 5×) on a
field the encoder never saw = the anomaly signal is about galaxies, not
surveys. A big drop = the current result is partly a depth artefact — which
would *change the M3 claim* and matter before any public write-up.

**Status.** Queued. The one pre-publish check I'd insist on if Q3 produces
candidates worth announcing.

## Q5 — M5: image + spectra contrastive (the big bet)

**Question.** AstroCLIP lineage: align cutout-image embeddings with NIRSpec
spectrum embeddings via a contrastive loss. Retrieval both ways (image→spectrum,
spectrum→image), and image embeddings that inherit spectroscopic structure —
redshift, emission-line strength, AGN-ness — without labels.

**Data.** New: JWST NIRSpec spectra from the DJA/DAWN spectroscopic releases
(or MAST). Thousands of 1-D spectra with RA/Dec to cross-match against our
cutout pipeline. This is the real data-engineering lift.

**Method sketch.** Freeze (or fine-tune) the M2 image encoder; small 1-D CNN /
transformer for spectra; InfoNCE across the paired batch. Evaluate with
retrieval metrics + a redshift probe on the image embeddings (does contrastive
alignment beat our photometry-only net?).

**What counts as a result.** Image-only redshift/line-strength probes that beat
photometry-only baselines = spectroscopically-informed imaging embeddings, on
public data, on a Mac mini. That's a paper-shaped object, and Q1–Q4 are its
validation toolkit.

**Status.** The destination. Not started; start only once Q1–Q3 have told us
the representation is trustworthy.
