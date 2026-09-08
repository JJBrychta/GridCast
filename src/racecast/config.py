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
SESSION_FETCH_DELAY = 5.0

# Schedule columns holding datetimes. Normalised on load: the cached schedule.json
# stores them as epoch-millis, a fresh fetch gives datetime64 — this unifies both.
SCHEDULE_DATE_COLUMNS = (
    "EventDate",
    "Session1DateUtc",
    "Session2DateUtc",
    "Session3DateUtc",
    "Session4DateUtc",
    "Session5DateUtc",
)

# FastF1 schedule session-name  ->  our slug. Add sprint types later.
SESSION_NAME_TO_TYPE = {
    "Qualifying": "qualifying",
    "Race": "race",
}
# our slug -> FastF1 name, for get_session()
SESSION_TYPE_TO_NAME = {slug: name for name, slug in SESSION_NAME_TO_TYPE.items()}

# An empty session result is only marked "no_data" (never retried) once the
# session is older than this. Inside the window it's treated as "official
# results not posted yet" -> retry next run.
RESULTS_LAG_DAYS = 3
