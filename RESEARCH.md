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
| Q1 | Is the anomaly score an LRD **selection function**? | cached data, hours | ✅ **done** — AUC 0.70; an enricher (10× at top-2%), not a selector; misses the faint/less-red 28% |
| Q2 | Does the net beat EAZY on **interloper rejection**? | cached data, hours | queued |
| Q3 | A ranked **spectroscopic follow-up list** | cached data, hours | queued |
| Q4 | Does the encoder **generalise across fields**? | retrain, ~hours MPS | queued |
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

**Status.** Queued.

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

**Status.** Unblocked — Q1 gives the operating point (top 1–2%, ~10×
enrichment, expected mostly bright/red candidates) and the bias caveat to
carry. Run the anomaly-ranked and nearest-to-known-LRD rankings side by side.

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
