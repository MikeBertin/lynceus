# Lynceus: research notes

**Private working notes**: the research on-ramp made explicit. Not written for
an outside reader yet; goes public (in some cleaned-up form) only once one of
these questions has a result worth showing. *(Update 2026-07-05: the repo and
site are now public; the honest method-validation story was judged worth
showing. This file remains the internal notebook.)*

The toolkit in hand: a SimCLR encoder over ~9.7k real JWST cutouts with a
validated anomaly score (LRDs 5× enriched, p<1e-4; see `experiments/
significance.py`), and a photo-z net with honest, calibration-checked PDFs vs
~1.8k spec-z (`experiments/calibrate_photoz.py`). Everything below is a
question that toolkit can attack, ordered by cost.

| # | Question | Cost | Status |
|---|----------|------|--------|
| Q1 | Is the anomaly score an LRD **selection function**? | cached data, hours | ✅ **done**: AUC 0.70; an enricher (10× at top-2%), not a selector; misses the faint/less-red 28% → **fixed by M3b** |
| Q2 | Does the net beat EAZY on **interloper rejection**? | cached data, hours | ✅ **done**: honest null: inherits its teacher's interlopers (10/12 shared), PDFs are confidently wrong |
| Q3 | A ranked **spectroscopic follow-up list** | cached data, hours | ✅ **done**: anomaly list: real exotica amid ~⅓ artefacts; supervised LRD-retrieval fails (noise-texture matching, mechanism diagnosed → M3b) |
| M3b | **Point-source-aware representation** (the Q3 fix) | refetch + retrain, ~hours | ✅ **done**: AUC 0.70→**0.91**, missed tail 28%→**1.9%**, retrieval now returns point sources (stars + compact red dots), not noise |
| Q3b | **Star veto + literature cross-match + photo-z** on the M3b list | cached + cheap queries, hours | ✅ **done**: 10 survivors: **2 published LRDs independently rediscovered** (Akins+24, consistent z) + **4 strong new candidates** (z≈5–8), absent from every checked photometric+spectroscopic LRD sample on our fields (Kokorev, full Kocevski+24, full Akins-CW, RUBIES) |
| Q3c | **From candidates to confirmations** (archival spectra → selection replay → SED) | cached + archive queries | 🔄 **Steps 1–2 done, and they revise Q3b.** Step 1: 0/10 have a public NIRSpec spectrum (validated on known LRDs). Step 2 (catalogue-grade selection replay): **0/8 novel candidates survive**: 5 are real galaxies that are *not red* in catalogue photometry (F277−F444≈0 vs LRDs' ≈1), 3 have no JWST counterpart; only the 2 already-known Akins LRDs pass. The Q3b "3–4 new candidates" were inflated by plausibility-grade aperture photometry |
| Q4 | Does the encoder **generalise across fields**? | retrain, ~hours MPS | ✅ **done, passes**: leave-one-field-out: LRD enrichment survives on all 3 held-out fields (AUC drops ≤0.03, CIs overlap); not a depth artefact |
| Q5 | **M5: image+spectra contrastive** (NIRSpec) | new data, weeks | the big bet |

---

## Q1: Is the anomaly score an LRD selection function?

**Question.** The 5× enrichment says LRDs are over-represented among anomalies.
A referee's next question: could you *find* LRDs with it? That's a selection
function: completeness and purity as a function of anomaly threshold, not a
single enrichment number.

**Data in hand.** `data/atlas/embeddings.npy` (9673×512),
`data/lrd/lrd_emb.npy` (216×512), scored with `core.anomaly.knn_anomaly`
exactly as in `experiments/build_atlas.py`.

**Method.** Treat known LRDs (Kokorev+24) as positives, atlas galaxies as
(approximate) negatives. ROC (completeness vs false-positive rate) as the
threshold sweeps; AUC with a bootstrap CI (AUC over a fixed negative set is the
mean of the per-LRD percentiles, so `core.stats.bootstrap_ci` applies
directly). Report operating points: keep the top 1/2/5/10/20% of the atlas by
anomaly, what fraction of LRDs do you catch, at what enrichment over chance?

**Caveats (stated up front).** The negatives are the *atlas sample* (stratified
by Galaxy Zoo class for the demo's colouring, not flux-limited) and may contain
LRD-like objects not in Kokorev. So this is a selection function *relative to
this atlas*, a proof of concept rather than a survey-grade measurement.

**What counts as a result.** AUC meaningfully above 0.5 with a usable knee
(e.g. an operating point catching ≳half the LRDs while rejecting ≳90% of the
atlas), plus the purity-vs-completeness trade-off made explicit, would justify
Q3 (using the score to *rank candidates*).

**Status / results.** ✅ Done: `python -m experiments.lrd_selection`
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
  (10× over chance), but completeness saturates near ~60%, and **28.2% of
  known LRDs score *below the atlas median***: no anomaly threshold can reach
  them.
- **The missed tail is not random.** Cross-matched against the Kokorev
  catalogue, missed (below-median) vs caught (top-decile) LRDs differ
  systematically: F444W flux **0.14 vs 0.26 µJy** (score correlates with
  log-flux, Spearman ρ=0.33), dust **Av 1.1 vs 1.8** (ρ=0.18), colour
  **F444W/F200W 4.2 vs 6.0**. The score selects the **bright, red end** of the
  LRD population: precisely the two things the encoder was built to see
  (colour-preserving stretch + compactness), which is both a validation and a
  measured selection bias.
- **Implication for Q3.** A candidate list from the top 1–2% is worth building
  (each pick is ~10× enriched), but it inherits this bias: it will find more
  bright/red LRD-likes and cannot find the faint tail. Any claim from Q3 must
  carry that caveat. A supervised alternative (`nearest_distance` to the known
  216) is the natural complement and would have a *different* bias, so it is worth
  running both.

## Q2: Does the photo-z net beat EAZY on interloper rejection?

**Question.** The classic failure mode behind "too many bright early galaxies":
a dusty z~2 galaxy whose Balmer/4000 Å break mimics a Lyman dropout. Our net
outputs a full PDF and visibly keeps a second mode when unsure. Does that
translate into a *lower interloper rate* than the EAZY template fit it distils?

**Data in hand.** `data/photoz/photoz.npz` holds, side by side for the ~1.8k
spec-z validation galaxies: our features (→ PDFs via `models/photoz.pt`), the
EAZY `z_phot`, and the spectroscopic truth. No new data needed.

**Method.** Define a high-z-candidate subset (e.g. either method places
z > 4, and again z > 6); count each method's interlopers (candidates with
z_spec < 2). Then use what EAZY can't do here: *PDF-aware* selection: require
P(z > 4) above a confidence cut rather than a point estimate, and trace the
candidate-yield vs interloper-rate curve. Bootstrap CIs on the rates
(`core/stats.py`). Also worth reporting: the by-slice calibration (the z 6–12
slice, n=78, is where σ_NMAD degrades to 0.083; is the PDF honest there?).

**What counts as a result.** Same candidate yield at a measurably lower
interloper rate (or the converse trade-off made explicit), with CIs that separate.
A null result is publishable-honest too: "a distilled emulator inherits its
teacher's interlopers" is worth knowing.

**Status / results.** ✅ Done: `python -m experiments.interlopers`
(writes `models/interlopers.json`). **The null, and it's clean.**

- **Point estimates are twins** (as a distillation must be): at z>4, EAZY
  82.9% complete / 6.3% interlopers vs ours 81.1% / 5.9%; CIs fully overlap.
  Same story at z>6 (70.5%/9.0% vs 66.7%/8.2%, n=78 truths, wide CIs).
- **The PDF confidence cut buys ~nothing**: P(z>4)>0.9 drops completeness
  83→56% while interlopers go 6.3→5.2% (CI 2–9%, overlapping everything).
  At z>6, P>0.9 reads 5.9% but on 34 selected (CI 0–15%), which is noise.
- **Why (the diagnosis, and the real finding):**
  1. *The interlopers are inherited.* 10 of our 12 z>4 interlopers are also
     EAZY's (5/6 at z>6). The net trained on EAZY labels, so EAZY's mistakes
     were mislabelled *in training*; the student learned them faithfully.
  2. *The PDFs are confidently wrong on exactly those galaxies.* Median
     P(z<2) is 0.038 for interlopers vs 0.008 for true high-z (a 5× whisper),
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

## Q3: A ranked spectroscopic follow-up candidate list

**Question.** The payoff question: point the anomaly score at the atlas and
hand over targets. The top-N most anomalous galaxies that are *not* known LRDs?
What are they? Genuinely new LRD candidates, other rare classes (mergers,
lenses, artefacts) or noise?

**Data in hand.** Everything Q1 uses, plus per-galaxy RA/Dec in
`data/atlas/atlas_meta.csv` and cutout PNGs for visual inspection.

**Method.** Rank the atlas by anomaly; drop known Kokorev matches (cross-match
by RA/Dec, ~1″); for the top ~50, assemble an inspection sheet (cutout, score,
field, GZ votes, nearest-LRD distance `lc`). Eyeball-classify; where DJA has
photometry, run our own photo-z on them. The Q1 selection function gives each
candidate an expected purity.

**What counts as a result.** Even a handful of credible, previously-uncatalogued
LRD candidates (compact + red + high anomaly + plausible photo-z) would be the
"we definitely have something" moment: a concrete deliverable no demo can fake.
Also the honest failure mode: if the top of the list is all artefacts, the
anomaly score is a data-quality detector, and that's the finding.

**Status / results.** ✅ Done: `python -m experiments.candidates` writes
`research/candidates.csv` + two contact sheets (`research/sheet_anomaly.jpg`,
`research/sheet_lrdlike.jpg`). Both outcomes at once: real targets *and* a
diagnosed failure.

- **The two rankings are completely disjoint**: zero overlap between the
  top-100 anomaly-ranked and top-100 LRD-like-ranked lists. They probe
  different directions: isolation vs proximity-to-exemplars.
- **Anomaly sheet (visual census of the top 32):** roughly ⅓ artefacts
  (diffraction spikes, guide-star stripes, noise blobs), so the anomaly score is
  partly a data-quality detector, as anticipated. But the rest are real and
  some are striking: two bright green compact objects (#5, #14; extreme
  emission-line galaxies by appearance, [OIII]-blob/"green pea"-like), an
  interacting pair (#21/#22; apparently the same system entering twice, so
  the atlas has near-duplicate catalogue entries), several edge-on red disks,
  and one compact orange-red dot (#6) that is morphologically LRD-like.
  Novelty **unverified**: claiming any of these as new requires cross-matching
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
  noise-texture, not as red dots, which is *why* no anomaly threshold reaches
  the faint 28%, and why the score only selects the bright/red end.
- **The 4 in-atlas Kokorev "matches"** (mean anomaly percentile 42, initially
  puzzling) resolve the same way: at 0.3–1.4″ separation they are mostly
  bright GZ galaxies *adjacent* to an LRD, and the cutout embeds the neighbour.
- **M3b (the concrete method fix this earns):** make the representation
  point-source aware: detection-anchored small cutouts; a noise-aware stretch
  (fixed physical surface-brightness scale instead of per-cutout percentiles,
  so empty sky stays dark); and/or explicit centroid-colour features. Then
  re-run Q1+Q3: if supervised retrieval starts returning compact red things,
  the candidate list becomes real.

**Verdict for the go-public question.** Q1–Q3 together are a coherent,
honest *methods* story (a validated-but-biased selector, an inherited-label
null, a diagnosed retrieval failure with a fix): notebook-grade, not yet the
"definitely have something". The something, if it comes, is M3b + a re-run Q3
list, or the anomaly sheet's emission-line objects surviving a literature
cross-match. *(Update: M3b and Q3b are now done; see those sections. The fix
worked, and Q3b delivered 3–4 strong uncatalogued candidates plus two
independent rediscoveries of published LRDs. What remains before
"definitely have something": a manual check against the 2025–26 LRD samples,
and Q4 as the pre-publish gate.)*

## M3b: the point-source-aware representation (the Q3 fix, executed)

**What.** Q3 diagnosed the failure: the per-cutout percentile stretch amplifies
empty-field sky noise to full-range colour static; faint LRDs (r_eff ~0.4 px on
near-empty fields) embed as that texture, so supervised retrieval returns noise
and the anomaly score can't reach the faint 28%. M3b changes the
representation, not the model class:

1. **Raw cubes cached once** (`experiments/fetch_raw.py` → `data/atlas/raw/`,
   `data/lrd/raw/`; 9,889 fetched, 0 failures). Every stretch experiment is now
   a local rebuild; the DJA service is never needed for this again.
2. **Noise-aware stretch** (`core.data.asinh_stretch_snr`): per-band sky
   subtraction, then the shared intensity is expressed in units of its own
   MAD-estimated sky σ, with asinh softening at 2σ, saturation at 300σ, 1σ noise
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

**Q1 re-run: the selection function is transformed** (`lrd_selection --m3b`,
`models/lrd_selection_m3b.json`):

- **AUC 0.702 → 0.910** (bootstrap 95% CI 0.890–0.928).
- Operating points (completeness, enrichment): top-1% 14.4% / 14.4×; top-2%
  36.6% / **18.3×**; top-5% 59.3% / 11.9×; top-10% **73.6%** / 7.4×; top-20%
  87.0% / 4.4×.
- **The unreachable tail is gone: 28.2% → 1.9%** of LRDs below the atlas
  median (n=4).
- **The brightness selection bias is gone**: Spearman(score, log F444W)
  0.33 → **−0.09**, score~Av 0.18 → 0.04. The four missed LRDs are, if
  anything, *brighter* than the caught ones (0.45 vs 0.21 µJy median): small-n
  curiosities, not a systematic faint-end blind spot.
- kNN-morphology probe on the m3b embeddings: **61.6% ± 0.8%** vs 62.0% ± 0.7%
  shipped (field 63.0% vs 63.4%), so point-source awareness cost nothing on the
  atlas's general structure.

**Q3 re-run: retrieval now fails in the *right* way** (`candidates --m3b`,
`research/candidates_m3b.csv` + `sheet_*_m3b.jpg`):

- The LRD-like sheet went from 32/32 colour static to **32/32 point sources**:
  ~22 are stars (unmistakable six-spike JWST PSF), ~9–10 are **compact
  orange/red dots**, exactly candidate-shaped (e.g. #18, #20–22, #24, #29,
  #30, #32).
- The luminance diagnostic flipped: top-100 LRD-like median 8.9 vs random 12.8
  (was 49 vs 10). Retrievals are now *darker* than random: dark-field point
  sources; the residual ρ(ldist, lum) = +0.46 just says bright extended
  galaxies are far from LRDs, which is correct physics.
- **The contaminant class is the real one.** Stars/brown dwarfs are *the*
  known contaminant in photometric LRD searches (Kokorev+24 do explicit
  brown-dwarf vetting), and the encoder now makes the same confusion the
  literature fights, instead of a pipeline artefact. That is what "the
  representation is trustworthy" looks like at this stage.
- Anomaly list: qualitatively as before (~⅓ artefacts + real exotica; the two
  top-100 lists still nearly disjoint, overlap 1). The 4 in-atlas Kokorev
  positional matches still embed the bright *neighbour* (0.3–1.4″ ≫ the
  ±8 px ≈ 0.2″ anchor window), an unchanged caveat.

**What M3b earns / next steps.**

1. **A star veto**: ✅ done, see Q3b below.
2. **Literature cross-match**: ✅ done, see Q3b below.
3. **Q4 (cross-field generalisation) before any public claim**: now more
   important, not less: the M3b numbers are strong enough that the depth/PSF
   fingerprint question is the main remaining threat.

## Q3b: star veto + literature cross-match + photo-z (the Q3 deliverable, realised)

The three follow-ups the M3b list earned, run end-to-end. Everything scripted
and re-runnable; outputs under `research/` (vetted/crossmatch/photoz CSVs +
`sheet_vetted.jpg`).

**Star veto** (`experiments/star_veto.py`). Three independent, disclosed
tests: (i) *PSF halo + diffraction spikes* from the cached raw cubes:
`mu_halo` (mean SNR-stretched intensity, 5–26 px annulus) and `m6sig` (m=6
azimuthal Fourier mode vs the other modes); (ii) *colour*, an F444W/F200W
aperture ratio ≥ 1.5 (LRDs are red by definition; known-LRD 5th pct = 1.9);
(iii) *Gaia DR3* cone (1″): significant parallax/PM = kinematic star.
Thresholds calibrated so **93% of the 216 known LRDs survive their own veto**
(the disclosed completeness cost). Result over the top-100 LRD-like:
**32 star_psf + 14 star_gaia + 44 not-red → 10 survivors**, visually all
compact orange-red dots (one elongated streak).

**Literature cross-match** (`experiments/candidate_crossmatch.py`). SIMBAD +
NED + the published LRD samples. Two catalogue sources:
SIMBAD + NED + the published LRD samples, assembled so that **every one of our
three fields is covered by at least one photometric and (for UDS) one
spectroscopic LRD catalogue**:

- the **full Kocevski+24 341-LRD Table 3** (`dalekocevski/Kocevski24`, GitHub,
  not on VizieR): **254 of its 341 in our fields** via PRIMER→COSMOS+UDS,
  JADES→GOODS-S; only 165 had reached Perger;
- the **full Akins+24 COSMOS-Web 434-LRD Table 1** (`hollisakins/akins24_cw`,
  GitHub ECSV, not on VizieR): complete COSMOS coverage, closing the
  14-source residual Perger left (it had 420 of the 434);
- the **RUBIES full spectroscopic census** (Hviding+25, VizieR J/A+A/702/A57,
  80 NIRSpec LRDs in UDS+EGS): the spectroscopic check for our UDS candidates;
- plus the Perger+25 compilation and the Euclid LRD list.

NEXUS (the other big 2025/26 spectroscopic LRD sample) is in the NEP field, with
no footprint overlap, correctly excluded. Two cross-match validations confirm
the machinery: **119/216 of our known Kokorev LRDs match a Kocevski source**
(median 0.1″), **15/216 match a RUBIES source** and both rediscoveries match
the **full Akins catalogue at 0.14″** (CW-758982, CW-758452), so a non-match
is a real absence, not a coordinate bug. Results:

- **2/10 are published LRDs we independently rediscovered**: Akins+24
  COSMOS-Web CW-758982 (z_phot 5.46) and CW-758452 (z_phot 5.99), at 0.14″.
  They were *not* in Kokorev v1.1, so they were never in our known set: the
  pipeline found real, previously-unseen published LRDs on its own, the
  cleanest end-to-end validation available without a telescope.
- **8/10 are in none of the checked samples**, including full Kocevski+24,
  full Akins-CW and the RUBIES spectroscopic census. Nearest catalogued LRD
  is 22–130″ away in-field (in-footprint, unselected). Two (#62, #97) have no
  SIMBAD entry at all; #87 is a COSMOS2015 emission-line galaxy; the rest
  resolve to generic catalogued galaxies (phot-z sources), not LRDs or stars.

**Photo-z sanity check** (`experiments/candidate_photoz.py`). First a
validation the check depends on: the M4 net on the Kokorev *catalogue*
photometry of the 216 known LRDs gives **σ_NMAD 0.042** vs their z_phot and
**P(z>4)>0.5 for 94%**: the EAZY-distilled net reads V-shaped LRD SEDs
about as well as ordinary galaxies (it was never shown one in training).
Then 9-band aperture photometry from service cutouts (r=10 px, sky-RMS
errors), per-band flux calibration = median catalogue/aperture ratio over 48
known LRDs (fixes units + point-source aperture loss in one factor; factors
came out consistent at ~0.010 with the expected mild PSF trend). The two
Akins rediscoveries come back at z 5.85/6.19 vs published 5.99/5.46,
consistent. The 8 novel survivors split:

| rank | id | field | z_peak | P(z>4) | reading |
|---:|---|---|---:|---:|---|
| 49 | atl041542 | UDS | **8.4** | **0.99** | strongest new candidate |
| 100 | atl007048 | COSMOS | 5.2 | **1.00** | high-z but *elongated*; likely dusty edge-on, not LRD morphology |
| 87 | atl015287 | COSMOS | 6.8 | 0.83 | strong; COSMOS2015 EmG |
| 57 | atl026058 | GOODS-S | 5.6 | 0.70 | strong |
| 97 | atl039645 | UDS | 7.3 | 0.47 | extreme red (F444W/F200W≈23), bimodal PDF; ambiguous |
| 42 | atl008513 | COSMOS | 1.0 | 0.35 | low-z favoured |
| 62 | atl041002 | UDS | 1.2 | 0.36 | low-z favoured |
| 67 | atl039298 | UDS | 2.6 | 0.00 | intermediate z, not LRD-z |

**Bottom line: 3–4 strong, previously-uncatalogued LRD candidates**
(#49, #57, #87, arguably #100), absent from *every* photometric and
(for UDS) spectroscopic LRD catalogue covering our fields while sitting inside
their footprints, plus two independent rediscoveries that validate the whole
chain: anomaly score → retrieval → veto → cross-match → photo-z.

> **⚠ REVISED BY Q3c Step 2 (catalogue-grade replay): read this.** This "3–4
> strong candidates" claim does **not** survive catalogue-grade photometry. The
> Q3b redness rested on plausibility-grade aperture photometry of service
> cutouts; when the same positions are measured in the DJA/COSMOS-Web parent
> catalogues, **none of the 8 novel candidates is red or compact enough to pass
> any published LRD cut** (5 are ordinary galaxies with F277W−F444W ≈ 0; 3 have
> no catalogued JWST counterpart). The two *rediscoveries* (atl006594/atl006181
> = Akins CW-758982/758452) remain real and still validate the method. Net:
> Q3b is a validated **method** demonstration, not a set of new discoveries.
> Full detail in Q3c → Step 2 below.

**Caveats, every one of them essential.** The aperture photometry is plausibility-grade,
not measurement-grade; LRD photo-zs are notoriously degenerate even with full
catalogues; faint brown dwarfs below Gaia can still pass the veto. The
literature check is now essentially complete for our fields (full Kocevski+24
(COSMOS+UDS+GOODS-S), full Akins-CW (COSMOS), RUBIES spectroscopic census
(UDS), Kokorev, Perger and Euclid), but the honest limit remains: **"absent
from a selection" can mean below their cuts, not non-existent. Only a spectrum
proves an LRD.** Q4 (below) is the last gate, ✅ now passed.

## Q3c: from candidates to confirmations (the roadmap)

**Question.** How do we address the Q3b residual ("absence from a photometric
selection can mean *below their cuts*, and only a spectrum truly confirms an
LRD"? It splits into two separable problems: **novelty** ("below their cuts")
and **identity** ("only a spectrum confirms"), with different fixes.

**Step 1: archival-spectrum search (highest value, do first).** A candidate
with an existing public NIRSpec spectrum can be confirmed or rejected *today,
for free*, the literal answer to "only a spectrum confirms." Search the 10
survivors by position against DJA-Spec / the DAWN spectroscopic archive,
**RUBIES** (UDS+EGS; it observed ~4,500 sources but flagged only 36 as LRDs, so
an un-flagged spectrum of one of our UDS candidates is plausible), JADES, CEERS,
UNCOVER, CAPERS. A hit is potentially decisive; it is also the on-ramp to M5.

**Status / results.** ✅ Done: `python -m experiments.candidate_specsearch
--validate` (writes `research/candidates_specsearch.csv`). **A clean null: none
of the 10 is spectroscopically observed yet.**

- **One search covers them all.** The DAWN spectroscopic archive (DJA-Spec) is
  the *union* of every public NIRSpec program reduced with msaexp (RUBIES,
  JADES, CEERS, UNCOVER, CAPERS, PRIMER), so a single cone-search against its
  master emission-line table (`dja_msaexp_emission_lines_v4.4`, **80,367
  extractions** with RA/Dec, `z_best`, DJA `grade`, Balmer-line fluxes and UV
  slope `beta`; S3, position-searchable) subsumes the individual releases the
  roadmap listed. No per-archive queries needed.
- **The machinery is validated the same way Q3b was.** Cross-matching the 216
  (of 260) known **Kokorev LRDs** against DJA-Spec recovers **90/260 with a
  public spectrum within 1″, at median separation 0.01″**: a pinpoint match.
  Spectra are found wherever they exist, so a candidate *non*-match is a real
  absence, not a coordinate/units bug. (It also quantifies the landscape: ~35%
  of already-known photometric LRDs on these fields have since been observed
  spectroscopically.)
- **0/10 candidates have a public NIRSpec spectrum.** Nearest public spectrum to
  any candidate is **3.4″** (atl008513), a different, resolved source; the
  result is unchanged at a generous 2″ radius. Even the two Akins-CW
  rediscoveries (atl006594/atl006181), though *photometrically* published, have
  no NIRSpec (nearest 55″/26″), consistent with COSMOS-Web LRDs being
  photometrically selected.
- **What this means.** The literal "only a spectrum confirms" cannot be
  discharged from the archive today: all 10 survivors are genuinely
  *unobserved*, not previously-confirmed objects we missed. That is the
  expected and honest outcome: it leaves the candidate list intact as a
  **prioritised spectroscopic-follow-up target list** (Step 3's scientific
  unit) and points the remaining evidence-gathering at **Step 2** (parent-
  catalog selection replay, to convert "absent" into a defensible novelty
  claim) and Step 3 (SED/slope/MIRI). It is also the M5 on-ramp: when these
  fields' next NIRSpec release lands, this exact search re-run is the
  confirmation test.

**Step 2: selection-function replay (fixes the novelty claim).** Cross-match
the candidates against the *parent photometric catalogues* (DJA / COSMOS-Web /
JADES source lists: every detected object, not the LRD sub-selection), pull
catalogue-grade fluxes, and apply each paper's *published cuts* (Akins:
F277W−F444W>1.5 + compactness + BD rejection; Kocevski: β_UV/β_opt slopes +
r_eff<1.5× stellar PSF; Kokorev: its cuts). Verdict per candidate: *passes their
cuts but absent* = genuinely new; *fails their cuts* = absence explained. Also
confirms each is a real detected source (our aperture photometry is only
plausibility-grade).

**Status / results.** ✅ Done: `python -m experiments.candidate_selection_replay`
(auto-fetches the parent catalogues → `research/candidates_selection_replay.csv`).
**The honest, sobering result: 0/8 novel candidates survive catalogue-grade
photometry.** This materially revises Q3b (see the revision note there).

- **Parent catalogues.** The van der Wel+25 DJA "morpho-phot" catalogues
  (`gds`, `primer-uds-north/south`, `primer-cosmos-west/east`; ~354k sources)
  carry catalogue-grade multi-band fluxes *and* single-Sérsic sizes in one file.
  Their row count equals the plain `_phot.fits` detection lists, so they *are*
  the parent detection catalogues, so a non-match is a genuine absence, not a
  morphology-subset artefact. COSMOS-Web sources outside PRIMER are recovered
  from the Akins+24 COSMOS-Web table (which carries their COSMOS-Web photometry).
- **The replay is validated both ways.** Known Kokorev LRDs recover at 110/111
  (UDS) and 26/27 (GOODS-S), come out **red** (median F277W−F444W ≈ 1.0) and
  **compact** (~100% by r_h < 1.5 r_h,stars). And the two Akins rediscoveries,
  fed their COSMOS-Web photometry, **pass the Akins cut** (F277W−F444W = 1.56 /
  1.92, C444 = 0.55 / 0.57, SNR_F444W = 137 / 158, brown-dwarf χ² rejected). So
  the machinery passes true LRDs and is trustworthy when it fails a candidate.
- **Verdict for all 10:**

  | verdict | ids | reading |
  |---|---|---|
  | published LRD (2) | atl006594, atl006181 | the Akins COSMOS-Web rediscoveries; pass Akins (already "known", not novel) |
  | below cuts (5) | atl041542, atl041002, atl039298, atl015287, atl007048 | real JWST galaxies, but **not red** in catalogue photometry (F277W−F444W ≈ −0.1…+0.2) and mostly extended → fail *every* published cut |
  | no JWST counterpart (3) | atl008513, atl026058, atl039645 | no catalogued JWST source within 1–8″ → spurious atlas detection / off-footprint |

- **Why Q3b over-called it.** The atlas (hence candidate) positions come from
  **Galaxy Zoo CANDELS** (relatively bright, human-classified galaxies) and
  the Q3b redness (`r44_20` = F444/F200 = 3–23) came from *plausibility-grade*
  aperture photometry on service cutouts, exactly the caveat the M3b/Q3b notes
  flagged. Catalogue-grade photometry deflates it: at the same positions,
  F277W−F444W ≈ 0, not the ≈1 of real LRDs. The five bright matches (SNR 8–294)
  are ordinary galaxies whose *cutout appearance* embedded near the LRD locus;
  three "candidates" aren't catalogued JWST sources at all.
- **Net for the novelty question.** "Absent from the LRD selections" is now
  **explained, not novelty**: the measurable candidates fail the cuts because
  they are not red/compact, and three are not real detections. The only genuine
  LRDs the anomaly→retrieval chain surfaced are the two Akins rediscoveries,
  which remain a clean, real validation of the *method*, just not a new
  discovery. Caveats: compactness is unified as Kocevski's r_h < 1.5 r_h,stars
  (a disclosed stand-in for the papers' various aperture-ratio compactness
  measures), but the "below cuts" verdicts are driven by **colour** (the
  dominant LRD discriminator), so they are robust to the compactness definition;
  Akins' brown-dwarf SED veto is only fully replayable for sources in the Akins
  table.

**Step 3: SED/slope/MIRI strengthening → a ranked target list.** Full-SED
template fit (LRD vs brown-dwarf vs dusty-galaxy), measure β_UV/β_opt (doubles
as the Kocevski replay), MIRI F770W/F1800W where COSMOS-Web has it. Present the
result as a **prioritised spectroscopic-follow-up target list**, the honest and
correct scientific unit. The caveat is intrinsic to every photometric LRD
sample; the response is to maximise archival evidence and frame the residual as
targets, not to pretend it is gone.

**Status.** Steps 1–2 ✅ done. Step 1: clean null (all 10 spectroscopically
unobserved). Step 2: catalogue-grade replay **collapses the novel candidate
list to zero**: 5 are real-but-not-red galaxies, 3 are non-detections and the
only survivors are the 2 already-known Akins LRDs. **Step 3 (SED/MIRI) is now
moot for a discovery claim**, as there is no novel candidate left to strengthen.
The standing conclusion for Q3c: the anomaly→retrieval→veto→cross-match chain is
a validated *method* (it independently rediscovered 2 published LRDs), but on
these fields, with catalogue-grade photometry, it yields **no new LRD**. The
faint-LRD frontier is a fundamentally different (fainter, JWST-catalogue-native)
population than our Galaxy-Zoo-seeded atlas can reach, which points to M5 and a
JWST-detected parent sample rather than more work on this candidate list.

## Q4: Does the encoder generalise across fields?

**Question.** Our own probe says a classifier can guess the survey field 63% of
the time from the embedding (depth/PSF fingerprints). Does the science transfer
anyway? Train the encoder on two fields, test on the third: does
morphology-kNN hold up, do the held-out field's LRDs still light up?

**Method.** Retrain SimCLR on GOODS-S+UDS cutouts only (~hours on MPS);
re-embed COSMOS + its LRDs with the frozen encoder; recompute kNN-morphology
and LRD enrichment on the held-out field. Compare to the all-field encoder.

**What counts as a result.** Enrichment surviving (CI overlapping the
all-field number) on a field the encoder never saw = the anomaly signal is
about galaxies, not surveys. A big drop = the current result is partly a depth
artefact, which would *change the M3 claim* and matter before any public
write-up.

**Status / results.** ✅ **Done, and it passes on all three fields.**
`experiments/train_atlas.py --exclude-field` trains three leave-one-field-out
M3b encoders (~2 h each on MPS; NT-Xent 1.81–1.84, vs 1.78 for the full
atlas); `experiments/q4_generalise.py` then scores each held-out field's
galaxies + LRDs with **both** the held-out encoder (never saw the field) and
the all-field M3b encoder, on *identical* galaxies, so only the encoder
differs and field-size effects cancel. AUC + top-10% enrichment (bootstrap CIs)
`models/q4_generalise.json`:

| held-out field | all-field enc AUC / enr | held-out enc AUC / enr | survives? |
|---|---|---|---|
| UDS (2,895 gal, 111 LRD) | 0.828 / 4.41× | 0.800 / 4.14× (CI 3.24–5.05) | ✅ |
| GOODS-S (2,855 gal, 27 LRD) | 0.974 / 10.0× | 0.959 / 9.26× (CI 8.15–10.0) | ✅ |
| COSMOS (3,923 gal, 78 LRD) | 0.939 / 8.59× | 0.931 / 8.46× (CI 7.69–9.23) | ✅ |

- **The LRD-anomaly signal is about galaxies, not surveys.** Dropping a whole
  field from training costs almost nothing on that field: AUC falls by only
  0.01–0.03 and every held-out enrichment CI overlaps its all-field
  counterpart. kNN-morphology is likewise flat (e.g. UDS 63.8%→63.3%). The
  depth/PSF fingerprint the field-probe sees does *not* drive the anomaly
  result.
- **Why the per-field AUCs differ** (0.83 UDS vs 0.97 GOODS-S): the reference
  cloud here is a *single* field (~3k galaxies) rather than the full 9,673, and
  the per-field LRD samples differ in brightness/count, so these aren't
  directly comparable to the headline all-field AUC 0.91. The Q4 comparison is
  strictly held-out-vs-all-field *within each field*, which is what isolates
  generalisation.
- **Verdict:** the pre-publish gate is cleared. The M3b LRD selector is not a
  survey-depth artefact: it transfers to fields the encoder never trained on.

## Q5: M5, image + spectra contrastive (the big bet)

**Question.** AstroCLIP lineage: align cutout-image embeddings with NIRSpec
spectrum embeddings via a contrastive loss. Retrieval both ways (image→spectrum,
spectrum→image), and image embeddings that inherit spectroscopic structure
(redshift, emission-line strength, AGN-ness) without labels.

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
