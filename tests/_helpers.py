"""Non-fixture test helpers: builders, fakes, and fixed timestamps.

Kept out of conftest.py so tests can import the names explicitly.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

# Fixed reference times. Tests pin `now` explicitly so nothing depends on the
# wall clock.
NOW = datetime(2024, 6, 1, 12, 0, 0)
OLD = datetime(2015, 1, 1)          # unambiguously in the past
FUTURE = datetime(2099, 1, 1)       # unambiguously in the future

# The subset of FastF1 schedule columns our code touches.
SCHEDULE_COLUMNS = ["RoundNumber", "EventName", "EventDate"]
for _i in range(1, 6):
    SCHEDULE_COLUMNS += [f"Session{_i}", f"Session{_i}DateUtc"]


def schedule_event(
    round_number: int,
    *slots: tuple[str, datetime | None],
    event_date: datetime | None = None,
    event_name: str = "Test Grand Prix",
) -> dict:
    """One schedule row. ``slots`` are ``(session_name, start_dt)`` for
    Session1, Session2, ... Unfilled slots default to ``(None, None)``, the way
    FastF1 leaves them.
    """
    row: dict = {
        "RoundNumber": round_number,
        "EventName": event_name,
        "EventDate": event_date,
    }
    for i in range(1, 6):
        name, when = slots[i - 1] if i <= len(slots) else (None, None)
        row[f"Session{i}"] = name
        row[f"Session{i}DateUtc"] = when
    return row


def schedule_df(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(events, columns=SCHEDULE_COLUMNS)


_FASTF1_SESSION_NAME = {"race": "Race", "qualifying": "Qualifying"}


def event_for(session_type: str, started: datetime, *, name: str = "Test GP") -> dict:
    """A minimal ``session.event`` mapping carrying one session's start time."""
    return {
        "EventName": name,
        "Session4": _FASTF1_SESSION_NAME[session_type],
        "Session4DateUtc": started,
        "EventDate": started,
    }


class FakeSession:
    """Stand-in for ``fastf1.core.Session`` — only ``.results`` and ``.event``."""

    def __init__(self, results: pd.DataFrame | None, event: dict | None = None):
        self.results = results
        self.event = event if event is not None else {"EventName": "Test GP"}


def raises(exc: BaseException):
    """Raise ``exc`` — lets a lambda raise: ``lambda: raises(ValueError())``."""
    raise exc
