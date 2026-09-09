"""Step 3 — build ``data/racecast.sqlite`` from ``raw/``.

The DB is a faithful, disposable mirror of the raw JSON archive. The only
computed columns are ``results.dnf`` and the ``session_weather`` rollup;
everything else is copied verbatim. Feature engineering / training-window
filtering happens later, on top of this.

Two passes (the FK chain forces the order — results reference sessions ->
events -> circuits + seasons):

  1. ``raw/<season>/schedule.json`` + ``circuits.json``
        -> seasons, circuits, events, sessions
  2. ``raw/<season>/round-NN/{race,qualifying}.json``
        -> drivers, constructors, results, session_weather

Idempotent and resumable:
  * an ``ingested_files`` hash ledger skips any raw file that hasn't changed
  * one transaction per file — a crash leaves the DB as if that file was untouched
  * every write is an UPSERT, so a full re-run produces an identical DB
  * ``rm data/racecast.sqlite`` (``make clean-db``) drops the ledger -> full rebuild
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from racecast._console import paint
from racecast.config import RAW_DIR, SESSION_NAME_TO_TYPE
from racecast.db import connect

# --------------------------------------------------------------------------- #
#  small value parsers — raw JSON is loosely typed, the DB is STRICT
# --------------------------------------------------------------------------- #


def _int(value) -> int | None:
    """Coerce to int. Handles the shapes seen in raw: ``1.0`` (float),
    ``"1"`` (str, e.g. DriverNumber), ``None``, ``""``. Bad input -> None."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    """Coerce to float (points, lat/long). Bad input -> None."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(millis) -> str | None:
    """Epoch-milliseconds -> ISO-8601 string.

    The cached ``schedule.json`` stores every date as epoch-millis (that's how
    ``DataFrame.to_json`` serialises datetimes). SQLite has no date type, so the
    DB stores ISO strings — comparable and sortable as text.
    """
    if millis is None or millis == "":
        return None
    return pd.Timestamp(int(millis), unit="ms").isoformat()


def _dnf(classified_position) -> int | None:
    """Race only. ``ClassifiedPosition`` is ``"1".."20"`` for a classified
    finish, or a letter code (``R`` retired, ``D`` disqualified, ``W`` withdrawn,
    ``N`` not classified). A non-numeric code means the driver did not finish.
    Returns None for qualifying (no such field) / missing data.
    """
    if classified_position in (None, ""):
        return None
    return 0 if str(classified_position).isdigit() else 1


def _rel(path: Path) -> str:
    """Path relative to RAW_DIR — what the ingested_files ledger stores, so the
    ledger is portable if the repo moves."""
    return str(path.relative_to(RAW_DIR))


def _sha256(path: Path) -> str:
    """Content hash of a raw file — the ledger's change-detection key."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
#  ingested_files ledger — "have I already loaded this exact file?"
# --------------------------------------------------------------------------- #


def _hash_matches(conn: sqlite3.Connection, path: Path, content_hash: str) -> bool:
    """True if this file is in the ledger with the same hash -> nothing to do."""
    row = conn.execute(
        "SELECT content_hash FROM ingested_files WHERE raw_path = ?", (_rel(path),)
    ).fetchone()
    return row is not None and row["content_hash"] == content_hash


def _mark_ingested(conn: sqlite3.Connection, path: Path, content_hash: str) -> None:
    """Record (or update) the ledger row. Called INSIDE the file's transaction,
    so the ledger and the data it produced commit or roll back together."""
    conn.execute(
        """INSERT INTO ingested_files (raw_path, content_hash, ingested_at)
           VALUES (?, ?, ?)
           ON CONFLICT(raw_path) DO UPDATE SET
             content_hash = excluded.content_hash,
             ingested_at  = excluded.ingested_at""",
        (_rel(path), content_hash, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


# --------------------------------------------------------------------------- #
#  dimension cache — {natural key -> surrogate id} for the small "entity" tables
# --------------------------------------------------------------------------- #


class DimCache:
    """Avoids a DB round-trip per driver/team/circuit lookup during the build.

    On construction it loads every existing (ref -> id, mutable-fields) pair.
    Each accessor then:
      * returns the cached id on a hit (and UPDATEs the name only if it changed
        — "last write wins", but no write when nothing changed, so re-runs of an
        unchanged archive touch these tables zero times);
      * INSERTs + caches on a miss.

    Because we are the only writer and the cache is seeded from the DB, "not in
    cache" reliably means "not in the DB" — a plain INSERT is safe.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        # year -> season_id  (seasons have no mutable fields)
        self._season: dict[int, int] = {
            r["year"]: r["season_id"]
            for r in conn.execute("SELECT year, season_id FROM seasons")
        }
        # circuit_ref -> (circuit_id, (name, locality, country, lat, long))
        self._circuit: dict[str, tuple[int, tuple]] = {
            r["circuit_ref"]: (
                r["circuit_id"],
                (r["name"], r["locality"], r["country"], r["lat"], r["long"]),
            )
            for r in conn.execute("SELECT * FROM circuits")
        }
        # driver_ref -> (driver_id, (first_name, last_name, abbreviation, country_code))
        self._driver: dict[str, tuple[int, tuple]] = {
            r["driver_ref"]: (
                r["driver_id"],
                (r["first_name"], r["last_name"], r["abbreviation"], r["country_code"]),
            )
            for r in conn.execute("SELECT * FROM drivers")
        }
        # team_ref -> (constructor_id, name)
        self._constructor: dict[str, tuple[int, str | None]] = {
            r["team_ref"]: (r["constructor_id"], r["name"])
            for r in conn.execute("SELECT team_ref, constructor_id, name FROM constructors")
        }

    def season(self, year: int) -> int:
        if year not in self._season:
            row = self.conn.execute(
                "INSERT INTO seasons (year) VALUES (?) RETURNING season_id", (year,)
            ).fetchone()
            self._season[year] = row["season_id"]
        return self._season[year]

    def circuit(self, rec: dict) -> int:
        """`rec` is one row of circuits.json (Ergast get_race_schedule)."""
        ref = rec["circuitId"]
        fields = (
            rec.get("circuitName"),
            rec.get("locality"),
            rec.get("country"),
            _float(rec.get("lat")),
            _float(rec.get("long")),
        )
        cached = self._circuit.get(ref)
        if cached is not None:
            circuit_id, cached_fields = cached
            if fields != cached_fields:  # metadata changed upstream -> refresh
                self.conn.execute(
                    "UPDATE circuits SET name=?, locality=?, country=?, lat=?, long=? "
                    "WHERE circuit_id=?",
                    (*fields, circuit_id),
                )
                self._circuit[ref] = (circuit_id, fields)
            return circuit_id
        row = self.conn.execute(
            "INSERT INTO circuits (circuit_ref, name, locality, country, lat, long) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING circuit_id",
            (ref, *fields),
        ).fetchone()
        self._circuit[ref] = (row["circuit_id"], fields)
        return row["circuit_id"]

    def driver(self, rec: dict) -> int:
        """`rec` is one entry of a session file's ``results`` array."""
        ref = rec.get("DriverId")
        if not _valid_ref(ref):
            recovered = self._recover_driver(rec)
            if recovered is not None:
                return recovered
            raise ValueError(f"unusable DriverId {ref!r} for {rec.get('LastName')!r}")
        fields = (
            rec.get("FirstName") or None,
            rec.get("LastName") or None,
            rec.get("Abbreviation") or None,  # "" for pre-~1980 drivers -> NULL
            rec.get("CountryCode") or None,   # "" for old drivers       -> NULL
        )
        cached = self._driver.get(ref)
        if cached is not None:
            driver_id, cached_fields = cached
            if fields != cached_fields:
                self.conn.execute(
                    "UPDATE drivers SET first_name=?, last_name=?, abbreviation=?, "
                    "country_code=? WHERE driver_id=?",
                    (*fields, driver_id),
                )
                self._driver[ref] = (driver_id, fields)
            return driver_id
        row = self.conn.execute(
            "INSERT INTO drivers (driver_ref, first_name, last_name, abbreviation, "
            "country_code) VALUES (?, ?, ?, ?, ?) RETURNING driver_id",
            (ref, *fields),
        ).fetchone()
        self._driver[ref] = (row["driver_id"], fields)
        return row["driver_id"]

    def _recover_driver(self, rec: dict) -> int | None:
        """A result row with no ``DriverId`` slug — Ergast occasionally ships one
        for a single driver at a single (usually recent) session. The row still
        carries the abbreviation and name, so match it to a driver already in the
        DB by ``(abbreviation, last name)`` — queried live, so it finds drivers
        loaded earlier this run *or* by a previous build. ``"VER"`` alone is
        shared by Verstappen and Vergne, hence the last name too. Returns None
        when there's no row or more than one; the caller then drops the row.
        """
        abbr = (rec.get("Abbreviation") or "").strip()
        last = (rec.get("LastName") or "").strip()
        first = (rec.get("FirstName") or "").strip()
        if not abbr or not last:
            return None
        hits = self.conn.execute(
            "SELECT driver_id, first_name FROM drivers "
            "WHERE abbreviation = ? COLLATE NOCASE AND last_name = ? COLLATE NOCASE",
            (abbr, last),
        ).fetchall()
        if len(hits) == 1:
            return hits[0]["driver_id"]
        # abbreviation + last name isn't unique (Michael vs Mick Schumacher, both
        # "MSC") — the first name breaks the tie.
        if len(hits) > 1 and first:
            named = [h for h in hits if (h["first_name"] or "").lower() == first.lower()]
            if len(named) == 1:
                return named[0]["driver_id"]
        return None

    def constructor(self, rec: dict) -> int:
        ref = rec.get("TeamId")
        if not _valid_ref(ref):
            recovered = self._recover_constructor(rec)
            if recovered is not None:
                return recovered
            raise ValueError(f"unusable TeamId {ref!r} for {rec.get('TeamName')!r}")
        name = rec.get("TeamName") or None
        cached = self._constructor.get(ref)
        if cached is not None:
            constructor_id, cached_name = cached
            if name != cached_name:
                self.conn.execute(
                    "UPDATE constructors SET name=? WHERE constructor_id=?",
                    (name, constructor_id),
                )
                self._constructor[ref] = (constructor_id, name)
            return constructor_id
        row = self.conn.execute(
            "INSERT INTO constructors (team_ref, name) VALUES (?, ?) RETURNING constructor_id",
            (ref, name),
        ).fetchone()
        self._constructor[ref] = (row["constructor_id"], name)
        return row["constructor_id"]

    def _recover_constructor(self, rec: dict) -> int | None:
        """As _recover_driver, for a row with no ``TeamId`` slug: match the
        verbatim ``TeamName`` against a constructor already in the DB. None when
        there's no row or more than one (e.g. "Racing Point" currently names two
        constructor rows — Force India and the later team).
        """
        name = (rec.get("TeamName") or "").strip()
        if not name:
            return None
        hits = self.conn.execute(
            "SELECT constructor_id FROM constructors WHERE name = ? COLLATE NOCASE",
            (name,),
        ).fetchall()
        return hits[0]["constructor_id"] if len(hits) == 1 else None


# --------------------------------------------------------------------------- #
#  pass 1 — schedules + circuits  ->  seasons, circuits, events, sessions
# --------------------------------------------------------------------------- #

# Column order for the events upsert. season_id + round_number are the conflict
# key (see the UNIQUE constraint); everything else is refreshed on conflict.
_EVENT_COLS = (
    "season_id", "round_number", "circuit_id", "event_name", "official_name",
    "event_format", "event_date", "country", "location",
)


def _upsert_event(conn: sqlite3.Connection, values: dict) -> int:
    """INSERT the event, or UPDATE it in place if (season, round) already exists.
    Returns the event_id either way (RETURNING works with DO UPDATE)."""
    updates = ", ".join(
        f"{c} = excluded.{c}"
        for c in _EVENT_COLS
        if c not in ("season_id", "round_number")
    )
    row = conn.execute(
        f"INSERT INTO events ({', '.join(_EVENT_COLS)}) "
        f"VALUES ({', '.join(':' + c for c in _EVENT_COLS)}) "
        f"ON CONFLICT(season_id, round_number) DO UPDATE SET {updates} "
        f"RETURNING event_id",
        values,
    ).fetchone()
    return row["event_id"]


def _upsert_session(
    conn: sqlite3.Connection, event_id: int, session_type: str, when: str | None
) -> None:
    """One row per (event, session_type). Only the date can change."""
    conn.execute(
        """INSERT INTO sessions (event_id, session_type, session_date)
           VALUES (?, ?, ?)
           ON CONFLICT(event_id, session_type)
             DO UPDATE SET session_date = excluded.session_date""",
        (event_id, session_type, when),
    )


def pass1_reference(conn: sqlite3.Connection, cache: DimCache) -> tuple[int, int]:
    """Build the reference tables from every ``raw/<season>/schedule.json``
    (paired with its ``circuits.json``). Returns (processed, skipped)."""
    processed = skipped = 0

    for schedule_path in sorted(RAW_DIR.glob("*/schedule.json")):
        season = int(schedule_path.parts[-2])
        circuits_path = schedule_path.with_name("circuits.json")

        # An event needs BOTH files, so reprocess the season if EITHER changed.
        schedule_hash = _sha256(schedule_path)
        circuits_hash = _sha256(circuits_path) if circuits_path.exists() else None
        if _hash_matches(conn, schedule_path, schedule_hash) and (
            circuits_hash is None or _hash_matches(conn, circuits_path, circuits_hash)
        ):
            skipped += 1
            continue

        schedule = json.loads(schedule_path.read_text())
        # round number -> its circuit record (may be empty if circuits.json is missing)
        circuits_by_round: dict[int, dict] = {}
        if circuits_path.exists():
            circuits_by_round = {
                int(rec["round"]): rec
                for rec in json.loads(circuits_path.read_text())
            }

        try:  # one transaction for the whole season's reference data
            season_id = cache.season(season)

            for row in schedule:
                round_number = int(row["RoundNumber"])
                if round_number < 1:
                    continue  # round 0 == pre-season testing

                circuit_rec = circuits_by_round.get(round_number)
                circuit_id = cache.circuit(circuit_rec) if circuit_rec else None  # nullable

                event_id = _upsert_event(conn, {
                    "season_id": season_id,
                    "round_number": round_number,
                    "circuit_id": circuit_id,
                    "event_name": row.get("EventName"),
                    "official_name": row.get("OfficialEventName"),
                    "event_format": row.get("EventFormat"),
                    "event_date": _iso(row.get("EventDate")),
                    "country": row.get("Country"),
                    "location": row.get("Location"),   # host town, not the circuit
                })

                # Walk the five session slots; keep only the names we map
                # (Qualifying, Race). Slot order varies on sprint weekends, so
                # match by name, not position.
                for slot in range(1, 6):
                    session_type = SESSION_NAME_TO_TYPE.get(row.get(f"Session{slot}"))
                    if session_type is None:
                        continue
                    _upsert_session(
                        conn, event_id, session_type,
                        _iso(row.get(f"Session{slot}DateUtc")),
                    )

            _mark_ingested(conn, schedule_path, schedule_hash)
            if circuits_hash is not None:
                _mark_ingested(conn, circuits_path, circuits_hash)
            conn.commit()
            processed += 1
        except Exception as exc:  # noqa: BLE001 — one bad season must not abort the build
            conn.rollback()
            print(paint(f"  [error] {_rel(schedule_path)}: {exc!r}", "red"))

    return processed, skipped


# --------------------------------------------------------------------------- #
#  pass 2 — session result files  ->  drivers, constructors, results, weather
# --------------------------------------------------------------------------- #

# Column order for the results upsert. (session_id, driver_id) is the conflict
# key; every other column is refreshed on conflict (provisional -> official).
_RESULT_COLS = (
    "session_id", "session_type", "driver_id", "constructor_id", "driver_number",
    "position", "classified_position", "grid_position", "status", "dnf",
    "points", "laps", "time_ms", "q1_ms", "q2_ms", "q3_ms",
)


def _result_values(
    session_id: int, session_type: str, driver_id: int, constructor_id: int, r: dict
) -> dict:
    """One ``results`` row from one entry of a session file's ``results`` array.

    ``results`` is a single wide table: race rows fill the race columns and NULL
    the qualifying ones, and vice-versa. Sprint / sprint_qualifying will reuse
    these same two column groups.
    """
    common = {
        "session_id": session_id,
        "session_type": session_type,      # denormalised from sessions for join-free filtering
        "driver_id": driver_id,
        "constructor_id": constructor_id,
        "driver_number": _int(r.get("DriverNumber")),
        "position": _int(r.get("Position")),   # finish order (race) / classification (quali)
    }
    if session_type == "race":
        return common | {
            "classified_position": r.get("ClassifiedPosition") or None,
            "grid_position": _int(r.get("GridPosition")),
            "status": r.get("Status") or None,
            "dnf": _dnf(r.get("ClassifiedPosition")),
            "points": _float(r.get("Points")),
            "laps": _int(r.get("Laps")),
            "time_ms": _int(r.get("Time")),         # ms; gap for non-winners, total for the winner
            "q1_ms": None, "q2_ms": None, "q3_ms": None,
        }
    # qualifying
    return common | {
        "classified_position": None, "grid_position": None, "status": None,
        "dnf": None, "points": None, "laps": None, "time_ms": None,
        "q1_ms": _int(r.get("Q1")),
        "q2_ms": _int(r.get("Q2")),   # null if knocked out in Q1
        "q3_ms": _int(r.get("Q3")),   # null if knocked out in Q2
    }


def _upsert_result(conn: sqlite3.Connection, values: dict) -> None:
    updates = ", ".join(
        f"{c} = excluded.{c}"
        for c in _RESULT_COLS
        if c not in ("session_id", "driver_id")
    )
    conn.execute(
        f"INSERT INTO results ({', '.join(_RESULT_COLS)}) "
        f"VALUES ({', '.join(':' + c for c in _RESULT_COLS)}) "
        f"ON CONFLICT(session_id, driver_id) DO UPDATE SET {updates}",
        values,
    )


def _aggregate_weather(rows: list[dict]) -> dict | None:
    """Per-minute weather timeseries (the raw file's ``weather`` array) ->
    one summary row. Returns None when there's no weather (pre-2018 / [])."""
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    def stat(col: str, how: str) -> float | None:
        if col not in df.columns:
            return None
        series = df[col].dropna()
        if series.empty:
            return None
        return round(float(getattr(series, how)()), 2)

    rain = df["Rainfall"] if "Rainfall" in df.columns else pd.Series(dtype="boolean")
    return {
        "air_temp_min": stat("AirTemp", "min"),
        "air_temp_avg": stat("AirTemp", "mean"),
        "air_temp_max": stat("AirTemp", "max"),
        "track_temp_min": stat("TrackTemp", "min"),
        "track_temp_avg": stat("TrackTemp", "mean"),
        "track_temp_max": stat("TrackTemp", "max"),
        "humidity_avg": stat("Humidity", "mean"),
        "pressure_avg": stat("Pressure", "mean"),
        "wind_speed_avg": stat("WindSpeed", "mean"),
        "wind_speed_max": stat("WindSpeed", "max"),
        # did it rain at all, and for what fraction of the session
        "rainfall_any": int(bool(rain.any())) if len(rain) else None,
        "rainfall_pct": round(float(rain.mean()) * 100, 1) if len(rain) else None,
    }


_WEATHER_COLS = (
    "session_id", "air_temp_min", "air_temp_avg", "air_temp_max",
    "track_temp_min", "track_temp_avg", "track_temp_max",
    "humidity_avg", "pressure_avg", "wind_speed_avg", "wind_speed_max",
    "rainfall_any", "rainfall_pct",
)


def _upsert_weather(conn: sqlite3.Connection, session_id: int, agg: dict) -> None:
    values = {"session_id": session_id, **agg}
    updates = ", ".join(f"{c} = excluded.{c}" for c in _WEATHER_COLS if c != "session_id")
    conn.execute(
        f"INSERT INTO session_weather ({', '.join(_WEATHER_COLS)}) "
        f"VALUES ({', '.join(':' + c for c in _WEATHER_COLS)}) "
        f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
        values,
    )


_BAD_REFS = {"", "nan", "none", "null"}


def _valid_ref(value) -> bool:
    """A usable natural key: a non-empty string that isn't a stringified null.

    FastF1's results DataFrame carries missing DriverId/TeamId as NaN, which
    ``to_json`` can serialise as the literal string ``"nan"`` — truthy, so it
    would sail into the DB as ``driver_ref='nan'`` without this check.
    """
    return isinstance(value, str) and value.strip().lower() not in _BAD_REFS


def _has_usable_row(results: list[dict]) -> bool:
    """At least one result row we can actually load. If none, the whole session
    isn't in Ergast yet -> skip and retry rather than mark it done with 0 rows."""
    return any(_valid_ref(r.get("DriverId")) and _valid_ref(r.get("TeamId")) for r in results)


def _lookup_session(
    conn: sqlite3.Connection, season: int, rnd: int, session_type: str
) -> int | None:
    """Resolve a raw file's (season, round, session_type) to its session_id.

    Reads from the DB — populated by pass 1, this run or a previous one — so
    pass 2 works even when pass 1 skipped everything as unchanged.
    """
    row = conn.execute(
        """SELECT s.session_id
           FROM sessions s
           JOIN events e   ON s.event_id  = e.event_id
           JOIN seasons se ON e.season_id = se.season_id
           WHERE se.year = ? AND e.round_number = ? AND s.session_type = ?""",
        (season, rnd, session_type),
    ).fetchone()
    return row["session_id"] if row else None


def pass2_results(conn: sqlite3.Connection, cache: DimCache) -> tuple[int, int, int]:
    """Load every ``raw/<season>/round-NN/*.json``. Returns
    (processed, skipped-unchanged, skipped-not-ready)."""
    processed = skipped = not_ready = 0

    for path in sorted(RAW_DIR.glob("*/round-*/*.json")):
        content_hash = _sha256(path)
        if _hash_matches(conn, path, content_hash):
            skipped += 1
            continue

        doc = json.loads(path.read_text())
        meta = doc["meta"]
        season, rnd, session_type = meta["season"], meta["round"], meta["session_type"]

        session_id = _lookup_session(conn, season, rnd, session_type)
        if session_id is None:
            # Its schedule hasn't been built yet (shouldn't happen — the universe
            # writes schedules before emitting units). NOT marked ingested, so
            # it's retried on the next run once pass 1 catches up.
            not_ready += 1
            print(paint(f"  [skip] no session row for {_rel(path)}", "yellow"))
            continue

        rows = doc.get("results", []) if doc.get("status") == "ok" else []
        if doc.get("status") == "ok" and rows and not _has_usable_row(rows):
            # Every row lacks a slug -> Ergast hasn't populated this session yet.
            # NOT marked ingested, so it's retried; re-fetching the file also fixes it.
            not_ready += 1
            print(paint(f"  [skip] no usable rows (missing DriverId/TeamId) in {_rel(path)}", "yellow"))
            continue

        try:  # one transaction per file
            if doc.get("status") == "ok":
                for r in rows:
                    # A few rows can still be missing slugs (an Ergast gap for a
                    # single driver at one session). DimCache tries to recover
                    # the row by matching name/abbreviation against a driver /
                    # constructor already loaded; if it can't, it raises and we
                    # drop just that row and keep the rest.
                    had_slug = _valid_ref(r.get("DriverId")) and _valid_ref(r.get("TeamId"))
                    try:
                        driver_id = cache.driver(r)
                        constructor_id = cache.constructor(r)
                    except ValueError:
                        print(paint(
                            f"  [drop row] {_rel(path)}: {r.get('FirstName')} "
                            f"{r.get('LastName')} (no slug, no match)", "yellow",
                        ))
                        continue
                    if not had_slug:
                        print(paint(
                            f"  [recovered] {_rel(path)}: {r.get('FirstName')} "
                            f"{r.get('LastName')} matched by name", "cyan",
                        ))
                    _upsert_result(conn, _result_values(
                        session_id, session_type, driver_id, constructor_id, r
                    ))
                agg = _aggregate_weather(doc.get("weather", []))
                if agg is not None:
                    _upsert_weather(conn, session_id, agg)
            # status == "no_data" -> the session/event exist (from pass 1) but
            # there are no result rows; nothing to insert, just mark it done.

            _mark_ingested(conn, path, content_hash)
            conn.commit()
            processed += 1
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            print(paint(f"  [error] {_rel(path)}: {exc!r}", "red"))

    return processed, skipped, not_ready


# --------------------------------------------------------------------------- #
#  entry point
# --------------------------------------------------------------------------- #


def _row_counts(conn: sqlite3.Connection) -> str:
    tables = ("seasons", "circuits", "events", "sessions", "drivers",
              "constructors", "results", "session_weather")
    return "  ".join(
        f"{t}={conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]}" for t in tables
    )


def build() -> None:
    conn = connect()               # opens/creates data/racecast.sqlite, applies schema.sql
    cache = DimCache(conn)          # load the {ref -> id} maps once

    p1_done, p1_skip = pass1_reference(conn, cache)
    print(paint(f"pass 1 (schedules): {p1_done} processed, {p1_skip} unchanged", "cyan"))

    p2_done, p2_skip, p2_not_ready = pass2_results(conn, cache)
    line = f"pass 2 (results): {p2_done} processed, {p2_skip} unchanged"
    if p2_not_ready:
        line += f", {p2_not_ready} not ready (retried next run)"
    print(paint(line, "cyan"))

    print(paint("db: " + _row_counts(conn), "bold"))
    conn.close()


if __name__ == "__main__":
    build()
