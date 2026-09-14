from datetime import datetime

from racecast.db.build import build
from racecast.ingest.cache import enable_cache
from racecast.config import FIRST_SEASON
from racecast.ingest.net import RateLimited
from racecast.ingest.sessions import fetch
from racecast.ingest.universe import UniverseGenerator

def run(first_season: int = FIRST_SEASON, last_season: int | None = None) -> None:
    """Universe -> fetch raw sessions -> build the DB. Both fetch and build are
    incremental (see their own docstrings), so re-running this after a single
    new session (e.g. a quali that just finished) is cheap."""
    try:
        enable_cache()
        universe = UniverseGenerator(
            first_season=first_season, last_season=last_season or datetime.now().year
        ).generate()
        print(f"{len(universe)} units in universe")
        fetch(universe)
        build()
    except RateLimited as exc:
        print(f"{exc}\nCached schedules are kept — re-run later to continue.")


if __name__ == "__main__":
    run()
