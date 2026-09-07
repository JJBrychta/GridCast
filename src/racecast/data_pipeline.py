from racecast.net import RateLimited
from racecast.universe import UniverseGenerator


if __name__ == "__main__":
    try:
        universe = UniverseGenerator().generate()  # FIRST_SEASON -> current year
        print(f"{len(universe)} units in universe")
    except RateLimited as exc:
        print(f"{exc}\nCached schedules are kept — re-run later to continue.")
