.PHONY: pipeline universe fetch backfill-weather init-db build-db clean-raw clean-cache clean-db clean

# Build the universe (1950 -> current season) and cache season schedules.
pipeline:
	uv run python -m racecast.data_pipeline

# Just the universe generator's own __main__ (current season only).
universe:
	uv run python -m racecast.universe

# Step 2: download raw session results for units not yet on disk.
fetch:
	uv run python -m racecast.session_loader

# One-off: add the weather key to ok files fetched before weather support (2018+).
backfill-weather:
	uv run python -m racecast.session_loader --weather

# Create data/racecast.sqlite with the schema applied (safe to re-run).
init-db:
	uv run python -m racecast.db

# Step 3: build data/racecast.sqlite from raw/ (idempotent — only new/changed files).
build-db:
	uv run python -m racecast.build_db

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
