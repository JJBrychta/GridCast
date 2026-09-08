"""Step 2 — fetch each universe session that has no raw file yet.

Resume model: a unit is "done" iff ``raw_path(unit)`` exists, so the work list
is ``universe - files-on-disk``. Writes are atomic (temp file + rename), so a
raw file is either fully present or absent — never half-written. On a rate limit
the run stops cleanly; re-running resumes from the first missing file.

Outcome per unit:
  ok       results present             -> write {status: "ok", results: [...]}
  no_data  empty + session is old      -> write {status: "no_data", results: []}
  retry    empty + future / too recent -> write nothing, picked up next run
"""

import enum
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.req import Cache

from racecast.cache import enable_cache
from racecast.config import (
    RESULTS_LAG_DAYS,
    SESSION_FETCH_DELAY,
    SESSION_TYPE_TO_NAME, FIRST_SEASON,
)
from racecast.net import RateLimited, with_retries
from racecast.universe import Unit, UniverseGenerator, raw_path

class Outcome(enum.Enum):
    OK = "ok"
    NO_DATA = "no_data"
    RETRY = "retry"


def _load_session(unit: Unit):
    """get_session + results-only load. Returns the loaded Session."""
    name = SESSION_TYPE_TO_NAME[unit.session_type]
    session = fastf1.get_session(unit.season, unit.round, name)
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session


def _session_start(unit: Unit, session) -> pd.Timestamp | None:
    """Best available start time for this session as a (naive UTC) Timestamp:
    its own SessionNDateUtc, else the weekend's EventDate."""
    event = session.event
    target = SESSION_TYPE_TO_NAME[unit.session_type]
    for i in range(1, 6):
        if event.get(f"Session{i}") == target:
            ts = event.get(f"Session{i}DateUtc")
            if not pd.isna(ts):
                return pd.Timestamp(ts)
    ts = event.get("EventDate")
    return pd.Timestamp(ts) if not pd.isna(ts) else None


def _results_records(results: pd.DataFrame) -> list[dict]:
    """Every column of session.results, verbatim — one record per driver.

    No column selection here: that's the build step's job. The raw layer exists
    precisely so deciding we want a field later never means re-scraping.
    to_json handles NaN/NaT -> null; timedeltas (Q1/Q2/Q3, Time) -> milliseconds.
    """
    return json.loads(results.reset_index(drop=True).to_json(orient="records"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)  # atomic on the same filesystem
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _decide(unit: Unit, session, now: datetime) -> tuple[Outcome, dict | None]:
    """Classify a loaded session -> (outcome, payload to write, or None)."""
    meta = {
        "season": unit.season,
        "round": unit.round,
        "session_type": unit.session_type,
        "event_name": str(session.event.get("EventName", "")),
        "fetched_at": now.isoformat(timespec="seconds"),
    }

    results = session.results
    if results is not None and len(results) > 0:
        return Outcome.OK, {
            "meta": meta,
            "status": "ok",
            "results": _results_records(results),
        }

    # Empty results. If the session is old enough that data would exist by now,
    # write a permanent no_data marker so we stop re-fetching it. Otherwise
    # (future, or still inside the results-posting lag) -> retry, write nothing.
    started = _session_start(unit, session)
    lag_cutoff = pd.Timestamp(now) - timedelta(days=RESULTS_LAG_DAYS)
    if started is not None and started < lag_cutoff:
        return Outcome.NO_DATA, {"meta": meta, "status": "no_data", "results": []}

    return Outcome.RETRY, None


def fetch(universe: list[Unit]) -> dict[Outcome, int]:
    enable_cache()

    # The resume mechanism: skip any unit whose raw file already exists (ok or
    # no_data both wrote one; only retry left nothing behind).
    todo = [unit for unit in universe if not raw_path(unit).exists()]
    print(f"{len(universe)} units in universe, {len(todo)} to fetch")

    counts = {outcome: 0 for outcome in Outcome}
    started = time.monotonic()
    loaded = 0  # units for which we actually called the API this run
    reqs0 = Cache._request_counter  # FastF1's running GET/POST tally

    for unit in todo:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            session = with_retries(
                lambda: _load_session(unit),
                what=f"session {unit}",
                pre_delay=SESSION_FETCH_DELAY,
            )
        except RateLimited as exc:
            # API said stop — bail cleanly, nothing half-written; next run resumes.
            loaded += 1  # the throttle sleep ran before the 429
            print(f"\n{exc}\nStopping — re-run to resume from the first missing file.")
            break
        except Exception as exc:  # FastF1 raises many exception types
            # Unexpected but not fatal: leave no file so it's retried next run.
            loaded += 1
            print(f"  {unit}  ->  error: {exc!r} (retry next run)")
            counts[Outcome.RETRY] += 1
            continue

        loaded += 1
        # payload is None only for retry -> write nothing, unit stays in todo.
        outcome, payload = _decide(unit, session, now)
        if payload is not None:
            _atomic_write_json(raw_path(unit), payload)
        counts[outcome] += 1

    # Statistics
        run = time.monotonic() - started
        reqs = Cache._request_counter - reqs0
        rpm = reqs / (run / 60) if run else 0.0
        print(
            f"  {unit}  ->  {outcome.value}"
            f"    [{loaded} loaded · {reqs} reqs · {run:.0f}s · {rpm:.0f} req/min]"
        )

    elapsed = time.monotonic() - started
    throttle = loaded * SESSION_FETCH_DELAY
    working = max(elapsed - throttle, 0.0)
    total_reqs = Cache._request_counter - reqs0
    print("done: " + ", ".join(f"{o.value}={counts[o]}" for o in Outcome))
    if loaded:
        rpm = total_reqs / (elapsed / 60) if elapsed else 0.0
        print(
            f"loaded {loaded} units, {total_reqs} API requests, in {elapsed:.0f}s "
            f"= {working:.0f}s API + {throttle:.0f}s throttle "
            f"({working / loaded:.1f}s/unit API, {rpm:.0f} req/min)"
        )
    return counts


if __name__ == "__main__":
    universe = UniverseGenerator(first_season=FIRST_SEASON, last_season=datetime.now().year).generate()
    fetch(universe)