.PHONY: pipeline universe fetch backfill-weather clean-raw clean-cache clean

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

# Wipe raw/ but keep dotfiles (.gitkeep).
clean-raw:
	find raw -mindepth 1 -maxdepth 1 ! -name '.*' -exec rm -rf {} +

# Wipe the FastF1 HTTP cache.
clean-cache:
	find fastf1_cache -mindepth 1 -maxdepth 1 ! -name '.*' -exec rm -rf {} +

clean: clean-raw clean-cache
