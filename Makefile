.PHONY: ingest universe fetch backfill-weather init-db build-db features refresh train predict clean-raw clean-cache clean-db clean

# Full data acquisition: universe -> fetch raw sessions -> build the DB.
ingest:
	uv run python -m racecast.ingest.pipeline

# Just the universe generator's own __main__ (current season only).
universe:
	uv run python -m racecast.ingest.universe

# Step 2: download raw session results for units not yet on disk.
fetch:
	uv run python -m racecast.ingest.sessions

# One-off: add the weather key to ok files fetched before weather support (2018+).
backfill-weather:
	uv run python -m racecast.ingest.sessions --weather

# Create data/racecast.sqlite with the schema applied (safe to re-run).
init-db:
	uv run python -m racecast.db.connect

# Step 3: build data/racecast.sqlite from raw/ (idempotent — only new/changed files).
build-db:
	uv run python -m racecast.db.build

# Step 4: DB -> data/datasets/feature_base.parquet + feature_matrix.parquet
features:
	uv run python -m racecast.features.build

# Step 5: fetch + build-db + features in one go. Add YEAR/ROUND to also print
# that race's rows once refreshed (e.g. right after its qualifying finishes).
#   make refresh
#   make refresh YEAR=2026 ROUND=17
refresh:
	uv run python -m racecast.refresh $(YEAR) $(ROUND)

# Step 6: train model_0 on SELECTED_FEATURES, save to data/models/model_0.joblib.
train:
	uv run python -m racecast.models.model_0.train

# Step 7: fetch/refresh one race's features, load the saved model, predict.
#   make predict YEAR=2026 ROUND=14
predict:
	uv run python -m racecast.models.model_0.predict $(YEAR) $(ROUND)

# Wipe raw/ but keep dotfiles (.gitkeep).
clean-raw:
	find raw -mindepth 1 -maxdepth 1 ! -name '.*' -exec rm -rf {} +

# Wipe the FastF1 HTTP cache.
clean-cache:
	find fastf1_cache -mindepth 1 -maxdepth 1 ! -name '.*' -exec rm -rf {} +

# Drop the derived DB (rebuild it from raw/ with build_db.py).
clean-db:
	rm -f data/racecast.sqlite

clean: clean-raw clean-cache clean-db
