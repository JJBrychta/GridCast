"""Tests for racecast.build_db — parsers, DimCache, and the two build passes."""

from __future__ import annotations

import json

import pytest

from racecast.build_db import (
    DimCache,
    _aggregate_weather,
    _dnf,
    _int,
    _iso,
    pass1_reference,
    pass2_results,
)
from racecast.db import connect


# --------------------------------------------------------------------------- #
#  pure parsers
# --------------------------------------------------------------------------- #

class TestParsers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1.0, 1), ("1", 1), (5504742, 5504742), (None, None), ("", None), ("x", None)],
    )
    def test_int(self, value, expected):
        assert _int(value) == expected

    def test_iso_from_epoch_millis(self):
        assert _iso(1709337600000) == "2024-03-02T00:00:00"

    def test_iso_none(self):
        assert _iso(None) is None

    @pytest.mark.parametrize(
        ("classified", "expected"),
        [("1", 0), ("20", 0), ("R", 1), ("D", 1), ("W", 1), ("", None), (None, None)],
    )
    def test_dnf(self, classified, expected):
        assert _dnf(classified) == expected


class TestAggregateWeather:
    def test_rolls_up_a_dry_session(self):
        rows = [
            {"AirTemp": 18.0, "TrackTemp": 22.0, "Humidity": 40, "Pressure": 1010,
             "WindSpeed": 1.0, "Rainfall": False},
            {"AirTemp": 20.0, "TrackTemp": 26.0, "Humidity": 50, "Pressure": 1012,
             "WindSpeed": 3.0, "Rainfall": False},
        ]
        agg = _aggregate_weather(rows)
        assert agg["air_temp_min"] == 18.0
        assert agg["air_temp_avg"] == 19.0
        assert agg["air_temp_max"] == 20.0
        assert agg["track_temp_max"] == 26.0
        assert agg["wind_speed_max"] == 3.0
        assert agg["rainfall_any"] == 0
        assert agg["rainfall_pct"] == 0.0

    def test_partial_rain(self):
        rows = [{"Rainfall": False}, {"Rainfall": True}, {"Rainfall": True}, {"Rainfall": False}]
        agg = _aggregate_weather(rows)
        assert agg["rainfall_any"] == 1
        assert agg["rainfall_pct"] == 50.0

    def test_empty_returns_none(self):
        assert _aggregate_weather([]) is None


# --------------------------------------------------------------------------- #
#  fixtures — a tmp DB + a tmp raw/ tree
# --------------------------------------------------------------------------- #

@pytest.fixture
def raw(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr("racecast.build_db.RAW_DIR", raw_dir)
    return raw_dir


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    yield c
    c.close()


def _schedule_event(round_number, *, quali_ms=1_700_000_000_000, race_ms=1_700_086_400_000):
    return {
        "RoundNumber": round_number,
        "Country": "Bahrain",
        "Location": "Sakhir",
        "OfficialEventName": "FORMULA 1 TEST GP",
        "EventName": "Test Grand Prix",
        "EventDate": race_ms,
        "EventFormat": "conventional",
        "Session4": "Qualifying", "Session4DateUtc": quali_ms,
        "Session5": "Race", "Session5DateUtc": race_ms,
    }


def _circuit_row(round_number):
    return {"round": round_number, "circuitId": "bahrain",
            "circuitName": "Bahrain International Circuit",
            "locality": "Sakhir", "country": "Bahrain", "lat": 26.03, "long": 50.51}


def _race_result(driver_ref, position, *, classified=None, status="Finished", team="rb"):
    return {
        "DriverId": driver_ref, "FirstName": driver_ref.title(), "LastName": "X",
        "Abbreviation": driver_ref[:3].upper(), "CountryCode": "NED",
        "TeamId": team, "TeamName": team.upper(),
        "DriverNumber": "1", "Position": float(position),
        "ClassifiedPosition": classified if classified is not None else str(position),
        "GridPosition": float(position), "Status": status,
        "Points": 25.0 if position == 1 else 0.0, "Laps": 57.0, "Time": 5_000_000,
    }


def _quali_result(driver_ref, position, *, team="rb"):
    return {
        "DriverId": driver_ref, "FirstName": driver_ref.title(), "LastName": "X",
        "Abbreviation": driver_ref[:3].upper(), "CountryCode": "NED",
        "TeamId": team, "TeamName": team.upper(),
        "DriverNumber": "1", "Position": float(position), "ClassifiedPosition": "",
        "Q1": 90000, "Q2": 89000 if position <= 2 else None, "Q3": 88000 if position == 1 else None,
    }


def _write(raw, season, name, doc, round_dir=None):
    d = raw / str(season) / round_dir if round_dir else raw / str(season)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(doc))


# --------------------------------------------------------------------------- #
#  pass 1
# --------------------------------------------------------------------------- #

class TestPass1:
    def test_builds_seasons_circuits_events_sessions(self, raw, conn):
        _write(raw, 2024, "schedule.json", [_schedule_event(0), _schedule_event(1)])
        _write(raw, 2024, "circuits.json", [_circuit_row(1)])

        processed, skipped = pass1_reference(conn, DimCache(conn))

        assert (processed, skipped) == (1, 0)
        assert conn.execute("SELECT year FROM seasons").fetchone()["year"] == 2024
        assert conn.execute("SELECT name FROM circuits").fetchone()["name"] == \
            "Bahrain International Circuit"
        ev = conn.execute("SELECT * FROM events").fetchone()
        assert ev["round_number"] == 1  # round 0 (testing) skipped
        assert ev["event_date"] == _iso(1_700_086_400_000)  # from _schedule_event's race_ms
        types = {r["session_type"] for r in conn.execute("SELECT session_type FROM sessions")}
        assert types == {"qualifying", "race"}

    def test_missing_circuits_file_leaves_circuit_id_null(self, raw, conn):
        _write(raw, 2024, "schedule.json", [_schedule_event(1)])  # no circuits.json

        pass1_reference(conn, DimCache(conn))

        assert conn.execute("SELECT circuit_id FROM events").fetchone()["circuit_id"] is None
        assert conn.execute("SELECT count(*) FROM circuits").fetchone()[0] == 0

    def test_unchanged_schedule_is_skipped_on_rerun(self, raw, conn):
        _write(raw, 2024, "schedule.json", [_schedule_event(1)])
        _write(raw, 2024, "circuits.json", [_circuit_row(1)])

        assert pass1_reference(conn, DimCache(conn)) == (1, 0)
        assert pass1_reference(conn, DimCache(conn)) == (0, 1)

    def test_changed_schedule_is_reprocessed(self, raw, conn):
        _write(raw, 2024, "schedule.json", [_schedule_event(1)])
        pass1_reference(conn, DimCache(conn))

        _write(raw, 2024, "schedule.json", [_schedule_event(1), _schedule_event(2)])
        assert pass1_reference(conn, DimCache(conn)) == (1, 0)
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 2


# --------------------------------------------------------------------------- #
#  pass 2
# --------------------------------------------------------------------------- #

class TestPass2:
    @pytest.fixture
    def built_schedule(self, raw, conn):
        _write(raw, 2024, "schedule.json", [_schedule_event(1)])
        _write(raw, 2024, "circuits.json", [_circuit_row(1)])
        pass1_reference(conn, DimCache(conn))
        return DimCache(conn)

    def test_race_results_and_weather(self, raw, conn, built_schedule):
        _write(raw, 2024, "race.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "race"},
            "status": "ok",
            "results": [_race_result("max_verstappen", 1),
                        _race_result("hulkenberg", 20, classified="R", status="Accident")],
            "weather": [{"AirTemp": 20.0, "Rainfall": False}, {"AirTemp": 22.0, "Rainfall": True}],
        }, round_dir="round-01")

        processed, skipped, missing = pass2_results(conn, built_schedule)

        assert (processed, skipped, missing) == (1, 0, 0)
        rows = conn.execute(
            "SELECT d.driver_ref, r.position, r.dnf, r.status, r.q1_ms "
            "FROM results r JOIN drivers d ON r.driver_id = d.driver_id ORDER BY r.position"
        ).fetchall()
        assert [dict(r) for r in rows] == [
            {"driver_ref": "max_verstappen", "position": 1, "dnf": 0, "status": "Finished", "q1_ms": None},
            {"driver_ref": "hulkenberg", "position": 20, "dnf": 1, "status": "Accident", "q1_ms": None},
        ]
        w = conn.execute("SELECT rainfall_any, rainfall_pct FROM session_weather").fetchone()
        assert (w["rainfall_any"], w["rainfall_pct"]) == (1, 50.0)

    def test_qualifying_results_fill_only_quali_columns(self, raw, conn, built_schedule):
        _write(raw, 2024, "qualifying.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "qualifying"},
            "status": "ok",
            "results": [_quali_result("max_verstappen", 1), _quali_result("gasly", 20)],
        }, round_dir="round-01")

        pass2_results(conn, built_schedule)

        ver = conn.execute(
            "SELECT r.* FROM results r JOIN drivers d ON r.driver_id=d.driver_id "
            "WHERE d.driver_ref='max_verstappen'"
        ).fetchone()
        assert (ver["q1_ms"], ver["q2_ms"], ver["q3_ms"]) == (90000, 89000, 88000)
        assert ver["grid_position"] is None and ver["status"] is None and ver["dnf"] is None

    def test_no_data_file_creates_no_rows_but_is_marked(self, raw, conn, built_schedule):
        _write(raw, 2024, "qualifying.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "qualifying"},
            "status": "no_data", "results": [],
        }, round_dir="round-01")

        processed, _, _ = pass2_results(conn, built_schedule)

        assert processed == 1
        assert conn.execute("SELECT count(*) FROM results").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM ingested_files").fetchone()[0] == 3  # sched+circ+quali

    def test_all_rows_missing_slugs_skips_the_file_without_marking(
        self, raw, conn, built_schedule
    ):
        # Ergast lag: every row has names but empty DriverId/TeamId
        bad = _race_result("ver", 1)
        bad["DriverId"] = bad["TeamId"] = ""
        _write(raw, 2024, "race.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "race"},
            "status": "ok", "results": [bad],
        }, round_dir="round-01")

        processed, _, not_ready = pass2_results(conn, built_schedule)

        assert (processed, not_ready) == (0, 1)
        assert conn.execute("SELECT count(*) FROM results").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM drivers").fetchone()[0] == 0
        assert conn.execute(  # not marked -> retried next run
            "SELECT count(*) FROM ingested_files WHERE raw_path LIKE '%race.json'"
        ).fetchone()[0] == 0

    def test_a_few_rows_missing_slugs_are_dropped_the_rest_loaded(
        self, raw, conn, built_schedule
    ):
        good = _quali_result("max_verstappen", 1)
        ghost = _quali_result("_", 20)          # "nan" DriverId, like an old Ergast gap
        ghost["DriverId"] = ghost["TeamId"] = "nan"
        _write(raw, 2024, "qualifying.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "qualifying"},
            "status": "ok", "results": [good, ghost],
        }, round_dir="round-01")

        processed, _, not_ready = pass2_results(conn, built_schedule)

        assert (processed, not_ready) == (1, 0)   # file processed + marked
        refs = [r["driver_ref"] for r in conn.execute("SELECT driver_ref FROM drivers")]
        assert refs == ["max_verstappen"]         # the "nan" row was dropped
        assert conn.execute("SELECT count(*) FROM results").fetchone()[0] == 1

    def test_slugless_row_is_recovered_by_name_against_a_known_driver(
        self, raw, conn, built_schedule
    ):
        # round 1: Verstappen races with a proper slug -> he's now in the DB
        _write(raw, 2024, "race.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "race"},
            "status": "ok", "results": [_race_result("max_verstappen", 1, team="rb")],
        }, round_dir="round-01")
        pass2_results(conn, built_schedule)

        # round 2 quali: Ergast ships his row with no DriverId/TeamId, only names
        _write(raw, 2024, "schedule.json", [_schedule_event(1), _schedule_event(2)])
        _write(raw, 2024, "circuits.json", [_circuit_row(1), _circuit_row(2)])
        pass1_reference(conn, DimCache(conn))
        ghost = _quali_result("max_verstappen", 1, team="rb")
        ghost["DriverId"] = ghost["TeamId"] = "nan"
        _write(raw, 2024, "qualifying.json", {
            "meta": {"season": 2024, "round": 2, "session_type": "qualifying"},
            "status": "ok",
            "results": [_quali_result("gasly", 2, team="rb"), ghost],  # one good row -> file isn't "not ready"
        }, round_dir="round-02")

        processed, _, not_ready = pass2_results(conn, DimCache(conn))

        assert (processed, not_ready) == (1, 0)
        # verstappen's slugless row reused his existing driver row (gasly is the
        # only new one); no phantom driver_ref='nan'
        assert {r["driver_ref"] for r in conn.execute("SELECT driver_ref FROM drivers")} == \
            {"max_verstappen", "gasly"}
        assert [r["team_ref"] for r in conn.execute("SELECT team_ref FROM constructors")] == ["rb"]
        got = conn.execute(
            "SELECT d.driver_ref, c.team_ref, r.position, r.q1_ms FROM results r "
            "JOIN drivers d ON r.driver_id = d.driver_id "
            "JOIN constructors c ON r.constructor_id = c.constructor_id "
            "JOIN sessions s ON r.session_id = s.session_id "
            "WHERE s.session_type = 'qualifying' AND d.driver_ref = 'max_verstappen'"
        ).fetchone()
        assert (got["team_ref"], got["position"], got["q1_ms"]) == ("rb", 1, 90000)

    def test_first_name_breaks_an_ambiguous_abbreviation_and_last_name(self, conn):
        cache = DimCache(conn)
        michael = cache.driver({"DriverId": "michael_schumacher", "FirstName": "Michael",
                                "LastName": "Schumacher", "Abbreviation": "MSC"})
        mick = cache.driver({"DriverId": "mick_schumacher", "FirstName": "Mick",
                             "LastName": "Schumacher", "Abbreviation": "MSC"})

        assert cache._recover_driver({"Abbreviation": "MSC", "FirstName": "Mick",
                                      "LastName": "Schumacher", "DriverId": "nan"}) == mick
        assert cache._recover_driver({"Abbreviation": "MSC", "FirstName": "Michael",
                                      "LastName": "Schumacher", "DriverId": "nan"}) == michael
        # no first name to disambiguate -> refuse
        assert cache._recover_driver({"Abbreviation": "MSC", "LastName": "Schumacher",
                                      "DriverId": "nan"}) is None

    def test_slugless_row_with_no_match_is_still_dropped(self, raw, conn, built_schedule):
        ghost = _quali_result("nobody", 20, team="rb")
        ghost["DriverId"] = ghost["TeamId"] = "nan"
        ghost["Abbreviation"], ghost["LastName"] = "ZZZ", "Nobody"  # unknown to the DB
        _write(raw, 2024, "qualifying.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "qualifying"},
            "status": "ok",
            "results": [_quali_result("max_verstappen", 1), ghost],
        }, round_dir="round-01")

        processed, _, not_ready = pass2_results(conn, built_schedule)

        assert (processed, not_ready) == (1, 0)
        refs = [r["driver_ref"] for r in conn.execute("SELECT driver_ref FROM drivers")]
        assert refs == ["max_verstappen"]

    def test_ambiguous_abbreviation_needs_the_last_name_to_recover(self, conn):
        # two drivers share "VER"; only (abbr, last name) disambiguates
        cache = DimCache(conn)
        cache.driver({"DriverId": "max_verstappen", "FirstName": "Max",
                      "LastName": "Verstappen", "Abbreviation": "VER"})
        vergne = cache.driver({"DriverId": "vergne", "FirstName": "Jean-Éric",
                               "LastName": "Vergne", "Abbreviation": "VER"})

        assert cache._recover_driver(
            {"Abbreviation": "VER", "LastName": "Verstappen", "DriverId": "nan"}
        ) == cache._driver["max_verstappen"][0]
        # "VER" + "Vergne" is just as specific
        assert cache._recover_driver(
            {"Abbreviation": "VER", "LastName": "Vergne", "DriverId": "nan"}
        ) == vergne

    def test_session_file_without_a_schedule_is_skipped_not_marked(self, raw, conn):
        _write(raw, 2024, "race.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "race"},
            "status": "ok", "results": [_race_result("ver", 1)],
        }, round_dir="round-01")

        processed, skipped, missing = pass2_results(conn, DimCache(conn))

        assert (processed, missing) == (0, 1)
        assert conn.execute("SELECT count(*) FROM ingested_files").fetchone()[0] == 0

    def test_provisional_then_official_updates_rows(self, raw, conn, built_schedule):
        path_args = ("round-01",)
        _write(raw, 2024, "race.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "race"},
            "status": "ok", "results": [_race_result("ver", 2), _race_result("ham", 1, team="mer")],
        }, *path_args)
        pass2_results(conn, built_schedule)

        # official: positions swapped
        _write(raw, 2024, "race.json", {
            "meta": {"season": 2024, "round": 1, "session_type": "race"},
            "status": "ok", "results": [_race_result("ver", 1), _race_result("ham", 2, team="mer")],
        }, *path_args)
        pass2_results(conn, DimCache(conn))

        pos = dict(conn.execute(
            "SELECT d.driver_ref, r.position FROM results r "
            "JOIN drivers d ON r.driver_id=d.driver_id"
        ).fetchall())
        assert pos == {"ver": 1, "ham": 2}
        assert conn.execute("SELECT count(*) FROM results").fetchone()[0] == 2  # updated, not duplicated


# --------------------------------------------------------------------------- #
#  DimCache
# --------------------------------------------------------------------------- #

class TestDimCache:
    def test_driver_name_is_last_write_wins(self, conn):
        cache = DimCache(conn)
        first = cache.driver({"DriverId": "x", "FirstName": "Bob", "LastName": "One"})
        again = cache.driver({"DriverId": "x", "FirstName": "Robert", "LastName": "One"})
        assert first == again
        row = conn.execute("SELECT first_name FROM drivers WHERE driver_id = ?", (first,)).fetchone()
        assert row["first_name"] == "Robert"

    def test_same_ref_returns_same_id(self, conn):
        cache = DimCache(conn)
        a = cache.constructor({"TeamId": "rb", "TeamName": "Red Bull"})
        b = cache.constructor({"TeamId": "rb", "TeamName": "Red Bull"})
        assert a == b
        assert conn.execute("SELECT count(*) FROM constructors").fetchone()[0] == 1
