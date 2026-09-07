from pathlib import Path

# Repo root: this file is src/racecast/config.py -> parents[2] is the repo dir.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "raw"
FASTF1_CACHE = REPO_ROOT / "fastf1_cache"

# F1 world championship began in 1950 — the earliest season with any data.
FIRST_SEASON = 1950

# Seconds to wait between uncached schedule fetches — keeps us under the Ergast
# API burst limit when backfilling many seasons at once. Bump if you still 429.
SCHEDULE_FETCH_DELAY = 0.6

# FastF1 schedule session-name  ->  our slug. Add sprint types later.
SESSION_NAME_TO_TYPE = {
    "Qualifying": "qualifying",
    "Race": "race",
}
