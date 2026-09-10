import fastf1
import pandas as pd
from fastf1.ergast import Ergast

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scipy.constants import year

from racecast.ingest.cache import enable_cache
from racecast.config import (
    FIRST_SEASON,
    RAW_DIR,
    SCHEDULE_DATE_COLUMNS,
    FETCH_DELAY,
    SESSION_NAME_TO_TYPE,
)
from racecast.ingest.net import with_retries


@dataclass(frozen=True)
class Unit:
    season: int
    round: int
    session_type: str

    def __str__(self) -> str:
        return f"{self.season}-R{self.round:02d}-{self.session_type}"


def raw_path(unit: Unit) -> Path:
    """raw/<season>/round-NN/<session_type>.json — the unit's file on disk.

    The fetch step's whole resume mechanism: a unit is done iff this file exists.
    """
    return (
        RAW_DIR
        / str(unit.season)
        / f"round-{unit.round:02d}"
        / f"{unit.session_type}.json"
    )


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

    # A hard fetch failure (rate limit / every backend down) has already been
    # raised as RateLimited by with_retries and never reaches here. What can
    # still slip through is an empty frame — don't cache that, or a past season
    # would trust the empty file forever. Return it unwritten; iter_units yields
    # nothing for it and the next run retries the fetch.
    if schedule is None or schedule.empty:
        print(f"No schedule data for season {season}; will retry next run")
        return schedule if schedule is not None else pd.DataFrame()

    path.parent.mkdir(parents=True, exist_ok=True)
    if season == current_season:
        print(f"Current season: {season}")
    else:
        print(f"Universe generator found new season: {season}")
    schedule.to_json(path, orient="records", indent=2)
    return _normalize_dates(schedule)


def load_circuits(
    season: int, *, current_season: int, fetch_delay: float = 0.0
) -> pd.DataFrame:
    """Ergast per-round circuit data for a season: round -> circuitId, circuitName,
    locality, country, coords.

    ``fastf1.get_event_schedule`` (used by ``load_schedule``) drops the circuit
    identity — only ``Country`` / ``Location`` (a town) survive. Ergast keeps the
    proper ``circuitId`` slug and full name. Cached on disk like the schedule;
    consumed only by the build step, not by ``iter_units``.
    """

    path = RAW_DIR / str(season) / "circuits.json"

    if path.exists() and season < current_season:
        return pd.read_json(path, orient="records")

    circuits = with_retries(
        lambda: Ergast(result_type="pandas").get_race_schedule(season=season),
        what=f"circuits {season}",
        pre_delay=fetch_delay,
    )

    if circuits is None or circuits.empty:
        print(f"No circuit data for season {season}; will retry next run")
        return circuits if circuits is not None else pd.DataFrame()

    path.parent.mkdir(parents=True, exist_ok=True)
    # drop the ErgastSimpleResponse subclass before serialising
    print(f"Found circuit info for season {season}")
    pd.DataFrame(circuits).to_json(path, orient="records", indent=2)
    return pd.DataFrame(circuits)


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
    # FastF1's SessionNDateUtc values are naive UTC, so compare against a naive
    # UTC "now". `now` is injectable so tests can pin a moment.
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = pd.Timestamp(now)

    for _, row in schedule.iterrows():
        # Round 0 == pre-season testing (there can be several, all numbered 0).
        # Real championship rounds start at 1.
        round_number = int(row["RoundNumber"])
        if round_number < 1:
            continue

        # A weekend has up to 5 session slots. We don't know which slot holds
        # qualifying vs the race (sprint weekends reorder them), so walk all 5
        # and match on the session *name*.
        for slot in range(1, 6):
            # e.g. "Qualifying" -> "qualifying", "Race" -> "race";
            # "Practice 2" / "Sprint" / "None" -> not in the map -> None.
            session_type = SESSION_NAME_TO_TYPE.get(row.get(f"Session{slot}"))
            if session_type is None:
                continue  # not a session type we collect

            # When did this session start? Prefer its own timestamp; fall back
            # to the weekend date if the session one is missing (quali/race are
            # within ~a day of it, close enough to decide "has it happened").
            started = row.get(f"Session{slot}DateUtc")
            if pd.isna(started):
                started = row.get("EventDate")

            if pd.isna(started):
                # No date anywhere. Only safe call: if the whole season is in
                # the past it certainly ran; otherwise leave it out.
                if season < cutoff.year:
                    yield Unit(season, round_number, session_type)
                continue

            # The gate: emit only once the session has actually started, so a
            # future race never enters the universe but its qualifying (a day
            # earlier) can — which is what the prediction workflow needs.
            if pd.Timestamp(started) <= cutoff:
                yield Unit(season, round_number, session_type)


class UniverseGenerator:
    def __init__(
        self,
        first_season: int = FIRST_SEASON,
        last_season: int | None = None,
        fetch_delay: float = FETCH_DELAY,
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
            # raw only — populates raw/<season>/circuits.json for the build step
            load_circuits(
                season,
                current_season=current_season,
                fetch_delay=self.fetch_delay,
            )
            units.extend(iter_units(schedule, season))
        return units


if __name__ == "__main__":
    universe = UniverseGenerator(first_season=FIRST_SEASON, last_season=datetime.now().year).generate()
    print(f"{len(universe)} units")
    for unit in universe:
        print(unit)
