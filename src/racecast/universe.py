import fastf1
import pandas as pd

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from racecast.cache import enable_cache
from racecast.config import (
    FIRST_SEASON,
    RAW_DIR,
    SCHEDULE_FETCH_DELAY,
    SESSION_NAME_TO_TYPE,
)
from racecast.net import with_retries


@dataclass(frozen=True)
class Unit:
    season: int
    round: int
    session_type: str

    def __str__(self) -> str:
        return f"{self.season}-R{self.round:02d}-{self.session_type}"


def load_schedule(
    season: int, *, current_season: int, fetch_delay: float = 0.0
) -> pd.DataFrame:
    """Season event schedule. Cached on disk for past seasons, always re-fetched
    for the current season (new GPs get added mid-season).

    ``fetch_delay`` throttles the API call (skipped entirely for cached seasons).
    """

    path = RAW_DIR / str(season) / "schedule.json"

    if path.exists() and season < current_season:
        return pd.read_json(path, orient="records")

    schedule = with_retries(
        lambda: fastf1.get_event_schedule(season),
        what=f"event schedule {season}",
        pre_delay=fetch_delay,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if season == current_season:
        print(f"Current season: {season}")
    else:
        print(f"Universe generator found new season: {season}")
    schedule.to_json(path, orient="records", indent=2)
    return schedule


def iter_units(schedule: pd.DataFrame, season: int) -> Iterator[Unit]:
    """One Unit per (round, wanted session type).

    Round 0 is always pre-season testing (possibly several test weekends, all
    numbered 0) -> skipped. Every real race weekend has exactly one Qualifying
    and one Race, so for the current session scope we don't need to look at the
    per-event session names at all.
    """
    for round_number in schedule["RoundNumber"]:
        if round_number < 1:
            continue
        for session_type in SESSION_NAME_TO_TYPE.values():
            yield Unit(season, int(round_number), session_type)


class UniverseGenerator:
    def __init__(
        self,
        first_season: int = FIRST_SEASON,
        last_season: int | None = None,
        fetch_delay: float = SCHEDULE_FETCH_DELAY,
    ):
        current_season = datetime.now(timezone.utc).year
        self.first_season = first_season
        self.last_season = last_season or current_season
        self.fetch_delay = fetch_delay

    def generate(self) -> list[Unit]:
        enable_cache()
        current_season = datetime.now(timezone.utc).year
        units: list[Unit] = []
        for season in range(self.first_season, self.last_season + 1):
            schedule = load_schedule(
                season,
                current_season=current_season,
                fetch_delay=self.fetch_delay,
            )
            units.extend(iter_units(schedule, season))
        return units


if __name__ == "__main__":
    universe = UniverseGenerator().generate()
    print(f"{len(universe)} units")
    for unit in universe:
        print(unit)
