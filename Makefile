.PHONY: pipeline universe clean-raw clean-cache clean

# Build the universe (1950 -> current year) and cache season schedules.
pipeline:
	uv run python -m racecast.data_pipeline

# Just the universe generator's own __main__ (current season only).
universe:
	uv run python -m racecast.universe

# Wipe raw/ but keep dotfiles (.gitkeep).
clean-raw:
	find raw -mindepth 1 -maxdepth 1 ! -name '.*' -exec rm -rf {} +

# Wipe the FastF1 HTTP cache.
clean-cache:
	find fastf1_cache -mindepth 1 -maxdepth 1 ! -name '.*' -exec rm -rf {} +

clean: clean-raw clean-cache
