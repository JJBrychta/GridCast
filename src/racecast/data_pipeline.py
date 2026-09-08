from racecast.config import FIRST_SEASON
from racecast.net import RateLimited
from racecast.universe import UniverseGenerator
from racecast.session_loader import fetch
from datetime import datetime
from racecast.cache import enable_cache

if __name__ == "__main__":
    try:
        enable_cache()
        universe = UniverseGenerator(first_season=FIRST_SEASON, last_season=datetime.now().year).generate()  # FIRST_SEASON -> current season
        # universe = UniverseGenerator(first_season=2026, last_season=2026).generate()  # FIRST_SEASON -> current season
        print(f"{len(universe)} units in universe")
        fetch(universe)
    except RateLimited as exc:
        print(f"{exc}\nCached schedules are kept — re-run later to continue.")
