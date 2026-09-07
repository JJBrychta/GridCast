import fastf1
import pandas as pd

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from racecast.cache import enable_cache
from racecast.config import (
    FIRST_SEASON,
    RAW_DIR,
    SCHEDULE_DATE_COLUMNS,
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


def _normalize_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    """Force the schedule's date columns to datetime, whatever the source.

    A fresh FastF1 fetch gives datetime64; the cached schedule.json gives
    epoch-millis. ``pd.to_datetime(..., unit="ms")`` is a no-op on the former
    and the right conversion for the latter.
    """
    for col in SCHEDULE_DATE_COLUMNS:
        if col in schedule.columns:
            schedule[col] = pd.to_datetime(schedule[col], unit="ms", errors="coerce")
    return schedule


def load_schedule(
    season: int, *, current_season: int, fetch_delay: float = 0.0
) -> pd.DataFrame:
    """Season event schedule. Cached on disk for past seasons, always re-fetched
    for the current season (new GPs get added mid-season).

    ``fetch_delay`` throttles the API call (skipped entirely for cached seasons).
    """

    path = RAW_DIR / str(season) / "schedule.json"

    if path.exists() and season < current_season:
        return _normalize_dates(pd.read_json(path, orient="records"))

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
    return _normalize_dates(schedule)


def iter_units(
    schedule: pd.DataFrame, season: int, *, now: datetime | None = None
) -> Iterator[Unit]:
    """One Unit per wanted session that has already started.

    Round 0 (pre-season testing, possibly several weekends, all numbered 0) is
    skipped. For each event we scan its session slots Session1..Session5: a slot
    whose name is in SESSION_NAME_TO_TYPE is emitted, but only once its own
    SessionNDateUtc is in the past -- so an upcoming weekend's qualifying enters
    the universe on its Saturday and the race the next day. Qualifying isn't
    always the same slot (sprint weekends move it), hence matching by name.
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = pd.Timestamp(now)

    for _, row in schedule.iterrows():
        round_number = int(row["RoundNumber"])
        if round_number < 1:
            continue
        for slot in range(1, 6):
            session_type = SESSION_NAME_TO_TYPE.get(row.get(f"Session{slot}"))
            # Did not match wanted session type
            if session_type is None:
                continue
            started = row.get(f"Session{slot}DateUtc")
            if pd.isna(started):
                started = row.get("EventDate")
            if pd.isna(started):
                if season < cutoff.year:  # ancient & undated -> it happened
                    yield Unit(season, round_number, session_type)
                continue
            if pd.Timestamp(started) <= cutoff:
                yield Unit(season, round_number, session_type)


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
    universe = UniverseGenerator(first_season=2026, last_season=2026).generate()
    print(f"{len(universe)} units")
    for unit in universe:
        print(unit)
