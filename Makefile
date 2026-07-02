# Lynceus — the reproduction DAG, one milestone per target.
#
# Everything runs offline on a laptop (Apple MPS / CUDA / CPU auto-selected);
# only the *-data targets touch the network. Rough costs on the machine of
# record (M4 Mac mini, 10-core GPU, 24 GB):
#
#   m1-data  network   ~300 cutouts via the DJA service — RATE-LIMITED, be gentle
#   m1       ~minutes  ViT fine-tune, 5-fold CV
#   m2-data  network   ~10k cutouts via the DJA service — the long, gentle fetch
#   m2       ~hours    SimCLR, 80 epochs on MPS (the one big training run)
#   m3-data  network   216 LRD cutouts
#   m3       ~minutes  re-embed + UMAP + anomaly + web assets
#   m4-data  network   one 375 MB catalogue download, then offline
#   m4       ~minutes  MLP train + ONNX export + web assets
#   stats    ~seconds  error bars + calibration — pure post-processing, no refetch
#
# The stats targets are safe to re-run any time; they only need the cached
# embeddings/model produced by the milestones above.

PY := .venv/bin/python

.PHONY: help venv test serve m1 m1-data m2 m2-data m3 m3-data m3b-data m3b m4 m4-data stats all

help:            ## show this help
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

venv:            ## create .venv (needs python3.13; 3.14 has no torch wheels)
	python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

test:            ## run the unit tests
	$(PY) -m pytest -q

serve:           ## serve the demos at http://localhost:8137
	cd web && python3 -m http.server 8137

# --- M1: morphology (ViT on Galaxy Zoo labels + JWST cutouts) ---------------
data/gz_candels.fits:
	curl -L -o $@ https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-candels/gz_candels_table_2_main_release.fits

m1-data: data/gz_candels.fits  ## fetch GZ labels + ~900 JWST cutouts (network)
	$(PY) -m experiments.fetch_gz --per-class 300

m1:              ## train ViT (5-fold CV) -> ONNX -> web assets
	$(PY) -m experiments.train_vit --folds 5 --epochs 10
	$(PY) -m experiments.export_onnx
	$(PY) -m experiments.build_web_assets --per-class 7

# --- M2: the self-supervised atlas -------------------------------------------
m2-data:         ## fetch ~10k unlabelled JWST cutouts (network, slow)
	$(PY) -m experiments.fetch_atlas --n 10000

m2:              ## train SimCLR (the big run) -> embed -> UMAP -> atlas assets
	$(PY) -m experiments.train_atlas --epochs 80
	$(PY) -m experiments.build_atlas

# --- M3: the anomaly hunt (needs the M2 encoder) ------------------------------
data/lrd_kokorev.fits:
	curl -L -o $@ https://raw.githubusercontent.com/VasilyKokorev/lrd_phot/master/lrd_table_v1.1.fits

m3-data: data/lrd_kokorev.fits  ## fetch + embed the 216 known Little Red Dots (network)
	$(PY) -m experiments.fetch_lrd

m3:              ## rebuild atlas assets with anomaly scores + LRD overlay
	$(PY) -m experiments.build_atlas

# --- M3b: point-source-aware representation (RESEARCH.md Q3 -> M3b) ----------
m3b-data:        ## fetch + cache the RAW cubes (network; makes all stretches local forever)
	$(PY) -m experiments.fetch_raw

m3b:             ## noise-aware cutouts -> retrain SimCLR (the other big run) -> rerun Q1+Q3
	$(PY) -m experiments.build_m3b_cutouts
	$(PY) -m experiments.train_atlas --cutouts-dir data/atlas/cutouts_m3b --out ssl_encoder_m3b.pt --epochs 80
	$(PY) -m experiments.embed_m3b
	$(PY) -m experiments.lrd_selection --m3b
	$(PY) -m experiments.candidates --m3b

# --- M4: the dropout hunter (photo-z) -----------------------------------------
data/ceers_morpho.fits.gz:
	curl -L -o $@ https://s3.amazonaws.com/aurelien-sepp/ceers-full-grizli-v7.2/catalog/ceers-full-grizli-v7.2_morpho-phot.fits.gz

m4-data: data/ceers_morpho.fits.gz  ## fetch the CEERS catalogue (375 MB) + build photoz.npz
	$(PY) -m experiments.build_photoz_dataset

m4:              ## train photo-z MLP -> ONNX -> dropout web assets
	$(PY) -m experiments.train_photoz --epochs 60
	$(PY) -m experiments.export_photoz_onnx
	$(PY) -m experiments.build_dropout_assets

# --- rigor: error bars + calibration (run AFTER the milestones they audit) ---
stats:           ## bootstrap CIs + permutation nulls + PDF calibration -> web JSON
	$(PY) -m experiments.significance
	$(PY) -m experiments.calibrate_photoz

all: m1-data m1 m2-data m2 m3-data m3 m4-data m4 stats  ## full pipeline, in dependency order
