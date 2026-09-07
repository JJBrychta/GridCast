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
    year: int, *, current_year: int, fetch_delay: float = 0.0
) -> pd.DataFrame:
    """Season event schedule. Cached on disk for past years, always re-fetched
    for the current year (new GPs get added mid-season).

    ``fetch_delay`` throttles the API call (skipped entirely for cached years).
    """

    path = RAW_DIR / str(year) / "schedule.json"

    if path.exists() and year < current_year:
        return pd.read_json(path, orient="records")

    schedule = with_retries(
        lambda: fastf1.get_event_schedule(year),
        what=f"event schedule {year}",
        pre_delay=fetch_delay,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if year == current_year:
        print(f"Current season: {current_year}")
    else:
        print(f"Universe generator found new season: {year}")
    schedule.to_json(path, orient="records", indent=2)
    return schedule


def iter_units(schedule: pd.DataFrame, year: int) -> Iterator[Unit]:
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
            yield Unit(year, int(round_number), session_type)


class UniverseGenerator:
    def __init__(
        self,
        start_year: int = FIRST_SEASON,
        end_year: int | None = None,
        fetch_delay: float = SCHEDULE_FETCH_DELAY,
    ):
        now_year = datetime.now(timezone.utc).year
        self.start_year = start_year
        self.end_year = end_year or now_year
        self.fetch_delay = fetch_delay

    def generate(self) -> list[Unit]:
        enable_cache()
        current_year = datetime.now(timezone.utc).year
        units: list[Unit] = []
        for year in range(self.start_year, self.end_year + 1):
            schedule = load_schedule(
                year, current_year=current_year, fetch_delay=self.fetch_delay
            )
            units.extend(iter_units(schedule, year))
        return units


if __name__ == "__main__":
    universe = UniverseGenerator().generate()
    print(f"{len(universe)} units")
    # for unit in universe:
    #     print(unit)
