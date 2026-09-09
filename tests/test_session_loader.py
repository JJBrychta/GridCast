"""Tests for racecast.session_loader — helpers, _decide, and fetch()."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from _helpers import FUTURE, NOW, OLD, FakeSession, event_for, raises
from racecast.net import RateLimited
from racecast.session_loader import (
    Outcome,
    _atomic_write_json,
    _decide,
    _load_session,
    _records,
    _session_start,
    _weather_records,
    fetch,
)
from racecast.universe import Unit


class TestSessionStart:
    def test_uses_the_matching_session_slot(self):
        event = {
            "Session4": "Qualifying",
            "Session4DateUtc": pd.Timestamp("2024-06-01 14:00"),
            "EventDate": pd.Timestamp("2024-06-02"),
        }
        got = _session_start(Unit(2024, 1, "qualifying"), FakeSession(None, event))
        assert got == pd.Timestamp("2024-06-01 14:00")

    def test_falls_back_to_event_date(self):
        event = {
            "Session4": "Qualifying",
            "Session4DateUtc": None,
            "EventDate": pd.Timestamp("2024-06-02"),
        }
        got = _session_start(Unit(2024, 1, "qualifying"), FakeSession(None, event))
        assert got == pd.Timestamp("2024-06-02")

    def test_returns_none_when_nothing_is_dated(self):
        event = {"Session4": "Qualifying", "Session4DateUtc": None, "EventDate": None}
        assert _session_start(Unit(2024, 1, "qualifying"), FakeSession(None, event)) is None


class TestRecords:
    def test_one_dict_per_row_with_index_dropped(self):
        df = pd.DataFrame({"Position": [1, 2]}, index=[44, 1])
        assert _records(df) == [{"Position": 1}, {"Position": 2}]

    def test_nan_becomes_null(self):
        df = pd.DataFrame({"Points": [25.0, np.nan]})
        assert _records(df) == [{"Points": 25.0}, {"Points": None}]

    def test_timedelta_becomes_milliseconds(self):
        df = pd.DataFrame({"Q3": pd.to_timedelta(["0 days 00:01:30"])})
        assert _records(df) == [{"Q3": 90_000}]


class TestLoadSession:
    @staticmethod
    def _capture_load_kwargs(monkeypatch) -> dict:
        captured: dict = {}

        class _Session:
            def load(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "racecast.session_loader.fastf1.get_session",
            lambda *a, **k: _Session(),
        )
        return captured

    def test_weather_requested_for_2018_and_later(self, monkeypatch):
        captured = self._capture_load_kwargs(monkeypatch)
        _load_session(Unit(2024, 1, "race"))
        assert captured["weather"] is True

    def test_weather_not_requested_before_2018(self, monkeypatch):
        captured = self._capture_load_kwargs(monkeypatch)
        _load_session(Unit(2017, 1, "race"))
        assert captured["weather"] is False

    def test_laps_telemetry_messages_always_off(self, monkeypatch):
        captured = self._capture_load_kwargs(monkeypatch)
        _load_session(Unit(2024, 1, "race"))
        assert captured["laps"] is False
        assert captured["telemetry"] is False
        assert captured["messages"] is False


class TestWeatherRecords:
    def test_returns_records_when_weather_present(self):
        df = pd.DataFrame({"AirTemp": [18.9, 19.1], "Rainfall": [False, True]})
        session = FakeSession(pd.DataFrame(), weather=df)
        assert _weather_records(session) == [
            {"AirTemp": 18.9, "Rainfall": False},
            {"AirTemp": 19.1, "Rainfall": True},
        ]

    def test_empty_when_weather_not_loaded(self):
        # FakeSession(weather=None) -> .weather_data raises, like DataNotLoadedError
        assert _weather_records(FakeSession(pd.DataFrame())) == []

    def test_empty_when_weather_frame_is_empty(self):
        assert _weather_records(FakeSession(pd.DataFrame(), weather=pd.DataFrame())) == []


class TestAtomicWriteJson:
    def test_writes_valid_json_and_creates_parents(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.json"
        _atomic_write_json(path, {"x": 1, "y": [2, 3]})
        assert json.loads(path.read_text()) == {"x": 1, "y": [2, 3]}

    def test_leaves_no_temp_file_on_success(self, tmp_path):
        path = tmp_path / "c.json"
        _atomic_write_json(path, {"x": 1})
        assert list(tmp_path.iterdir()) == [path]

    def test_leaves_no_partial_file_on_write_failure(self, tmp_path, monkeypatch):
        path = tmp_path / "c.json"
        monkeypatch.setattr(
            "racecast.session_loader.json.dump",
            lambda *a, **k: raises(RuntimeError("disk full")),
        )
        with pytest.raises(RuntimeError):
            _atomic_write_json(path, {"x": 1})
        assert list(tmp_path.iterdir()) == []


class TestDecide:
    def test_results_present_is_ok(self):
        session = FakeSession(
            pd.DataFrame({"DriverId": ["a", "b"]}),
            {"EventName": "Bahrain GP"},
            weather=pd.DataFrame({"AirTemp": [18.9]}),
        )
        outcome, payload = _decide(Unit(2024, 1, "qualifying"), session, NOW)

        assert outcome is Outcome.OK
        assert payload is not None
        assert payload["status"] == "ok"
        assert payload["results"] == [{"DriverId": "a"}, {"DriverId": "b"}]
        assert payload["weather"] == [{"AirTemp": 18.9}]
        assert payload["meta"] == {
            "season": 2024,
            "round": 1,
            "session_type": "qualifying",
            "event_name": "Bahrain GP",
            "fetched_at": "2024-06-01T12:00:00",
        }

    def test_ok_payload_weather_empty_when_not_loaded(self):
        session = FakeSession(pd.DataFrame({"DriverId": ["a"]}))  # no weather
        _, payload = _decide(Unit(1975, 1, "race"), session, NOW)
        assert payload["weather"] == []

    def test_empty_and_old_is_no_data_marker(self):
        session = FakeSession(pd.DataFrame(), event_for("qualifying", NOW - timedelta(days=30)))
        outcome, payload = _decide(Unit(1975, 1, "qualifying"), session, NOW)

        assert outcome is Outcome.NO_DATA
        assert payload == {
            "meta": {
                "season": 1975,
                "round": 1,
                "session_type": "qualifying",
                "event_name": "Test GP",
                "fetched_at": "2024-06-01T12:00:00",
            },
            "status": "no_data",
            "results": [],
        }

    def test_empty_and_recent_is_retry(self):
        session = FakeSession(pd.DataFrame(), event_for("qualifying", NOW - timedelta(hours=1)))
        outcome, payload = _decide(Unit(2024, 12, "qualifying"), session, NOW)
        assert outcome is Outcome.RETRY
        assert payload is None

    def test_empty_and_future_is_retry(self):
        session = FakeSession(pd.DataFrame(), event_for("race", NOW + timedelta(days=7)))
        outcome, payload = _decide(Unit(2024, 20, "race"), session, NOW)
        assert outcome is Outcome.RETRY
        assert payload is None

    def test_results_none_is_treated_as_empty(self):
        session = FakeSession(None, event_for("qualifying", NOW - timedelta(days=30)))
        outcome, _ = _decide(Unit(1975, 1, "qualifying"), session, NOW)
        assert outcome is Outcome.NO_DATA


@pytest.fixture
def fetch_env(monkeypatch, tmp_path):
    """Isolate fetch(): raw/ under tmp_path, no cache init, no real sleeps."""
    monkeypatch.setattr("racecast.universe.RAW_DIR", tmp_path)  # used by raw_path()
    monkeypatch.setattr("racecast.session_loader.enable_cache", lambda: None)
    monkeypatch.setattr("racecast.net.time.sleep", lambda *_: None)
    return tmp_path


def _stub_loader(monkeypatch, mapping: dict[Unit, FakeSession | Exception]):
    seen: list[Unit] = []

    def _load(unit: Unit) -> FakeSession:
        seen.append(unit)
        result = mapping[unit]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("racecast.session_loader._load_session", _load)
    return seen


class TestFetch:
    def test_writes_ok_and_no_data_but_not_retry(self, fetch_env, monkeypatch):
        raw = fetch_env
        u_ok = Unit(2024, 1, "race")
        u_nodata = Unit(1975, 1, "qualifying")
        u_retry = Unit(2024, 20, "race")
        _stub_loader(
            monkeypatch,
            {
                u_ok: FakeSession(pd.DataFrame({"DriverId": ["x"]}), event_for("race", OLD)),
                u_nodata: FakeSession(pd.DataFrame(), event_for("qualifying", OLD)),
                u_retry: FakeSession(pd.DataFrame(), event_for("race", FUTURE)),
            },
        )

        counts = fetch([u_ok, u_nodata, u_retry])

        assert counts == {Outcome.OK: 1, Outcome.NO_DATA: 1, Outcome.RETRY: 1}
        assert (raw / "2024" / "round-01" / "race.json").exists()
        assert (raw / "1975" / "round-01" / "qualifying.json").exists()
        assert not (raw / "2024" / "round-20" / "race.json").exists()

    def test_ok_file_has_meta_and_records(self, fetch_env, monkeypatch):
        _stub_loader(
            monkeypatch,
            {
                Unit(2024, 1, "race"): FakeSession(
                    pd.DataFrame({"DriverId": ["a"], "Position": [1]}),
                    event_for("race", OLD, name="Bahrain GP"),
                )
            },
        )

        fetch([Unit(2024, 1, "race")])

        doc = json.loads((fetch_env / "2024" / "round-01" / "race.json").read_text())
        assert doc["status"] == "ok"
        assert doc["results"] == [{"DriverId": "a", "Position": 1}]
        assert doc["meta"]["event_name"] == "Bahrain GP"

    def test_units_already_on_disk_are_skipped(self, fetch_env, monkeypatch):
        done = Unit(2024, 1, "race")
        target = fetch_env / "2024" / "round-01" / "race.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}")

        seen = _stub_loader(monkeypatch, {done: FakeSession(pd.DataFrame(), event_for("race", OLD))})

        counts = fetch([done])

        assert seen == []
        assert sum(counts.values()) == 0

    def test_rate_limit_stops_the_run_leaving_later_units_untouched(
        self, fetch_env, monkeypatch
    ):
        u1, u2, u3 = (Unit(2024, n, "race") for n in (1, 2, 3))
        ok = FakeSession(pd.DataFrame({"DriverId": ["a"]}), event_for("race", OLD))
        seen = _stub_loader(monkeypatch, {u1: ok, u2: RateLimited("429"), u3: ok})

        counts = fetch([u1, u2, u3])

        assert seen == [u1, u2]  # stopped, u3 never attempted
        assert counts[Outcome.OK] == 1
        assert (fetch_env / "2024" / "round-01" / "race.json").exists()
        assert not (fetch_env / "2024" / "round-03" / "race.json").exists()

    def test_resume_picks_up_where_a_rate_limited_run_stopped(
        self, fetch_env, monkeypatch
    ):
        u1, u2 = Unit(2024, 1, "race"), Unit(2024, 2, "race")
        ok = FakeSession(pd.DataFrame({"DriverId": ["a"]}), event_for("race", OLD))
        attempts: list[Unit] = []

        def _load(unit: Unit) -> FakeSession:
            attempts.append(unit)
            if unit == u2 and attempts.count(u2) == 1:
                raise RateLimited("429")  # fail u2 only on the first pass
            return ok

        monkeypatch.setattr("racecast.session_loader._load_session", _load)

        fetch([u1, u2])  # writes u1, stops at u2
        fetch([u1, u2])  # u1 skipped (on disk), u2 retried and succeeds

        assert attempts == [u1, u2, u2]
        assert (fetch_env / "2024" / "round-02" / "race.json").exists()

    def test_unexpected_loader_error_is_counted_as_retry_and_does_not_stop(
        self, fetch_env, monkeypatch
    ):
        u1, u2 = Unit(2024, 1, "race"), Unit(2024, 2, "race")
        ok = FakeSession(pd.DataFrame({"DriverId": ["a"]}), event_for("race", OLD))
        seen = _stub_loader(monkeypatch, {u1: KeyError("weird"), u2: ok})

        counts = fetch([u1, u2])

        assert seen == [u1, u2]  # did not stop
        assert counts[Outcome.RETRY] == 1
        assert counts[Outcome.OK] == 1
        assert not (fetch_env / "2024" / "round-01" / "race.json").exists()


class TestBackfillWeather:
    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        monkeypatch.setattr("racecast.session_loader.RAW_DIR", tmp_path)
        monkeypatch.setattr("racecast.session_loader.enable_cache", lambda: None)
        monkeypatch.setattr("racecast.net.time.sleep", lambda *_: None)
        return tmp_path

    @staticmethod
    def _write(raw, season, rnd, session_type, doc):
        p = raw / str(season) / f"round-{rnd:02d}" / f"{session_type}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc))
        return p

    def test_only_touches_2018plus_ok_files_without_weather(self, env, monkeypatch):
        from racecast.session_loader import backfill_weather

        ok = {"meta": {"season": 0, "round": 1, "session_type": "race"}, "status": "ok", "results": [{}]}
        need = self._write(env, 2019, 1, "race", {**ok, "meta": {"season": 2019, "round": 1, "session_type": "race"}})
        old = self._write(env, 2015, 1, "race", {**ok, "meta": {"season": 2015, "round": 1, "session_type": "race"}})
        has = self._write(env, 2020, 1, "race",
                          {**ok, "meta": {"season": 2020, "round": 1, "session_type": "race"}, "weather": [1]})
        nod = self._write(env, 2021, 1, "qualifying", {"meta": {"season": 2021, "round": 1, "session_type": "qualifying"},
                                                       "status": "no_data", "results": []})

        seen: list[Unit] = []

        def _load(unit: Unit) -> FakeSession:
            seen.append(unit)
            return FakeSession(pd.DataFrame(), weather=pd.DataFrame({"AirTemp": [20.0]}))

        monkeypatch.setattr("racecast.session_loader._load_session", _load)

        updated = backfill_weather()

        assert updated == 1
        assert seen == [Unit(2019, 1, "race")]
        assert json.loads(need.read_text())["weather"] == [{"AirTemp": 20.0}]
        assert "weather" not in json.loads(old.read_text())      # pre-2018 untouched
        assert json.loads(has.read_text())["weather"] == [1]     # already had it
        assert "weather" not in json.loads(nod.read_text())      # no_data untouched

    def test_rate_limit_stops_and_is_resumable(self, env, monkeypatch):
        from racecast.session_loader import backfill_weather

        for rnd in (1, 2):
            self._write(env, 2019, rnd, "race",
                        {"meta": {"season": 2019, "round": rnd, "session_type": "race"},
                         "status": "ok", "results": [{}]})

        calls: list[Unit] = []

        def _load(unit: Unit) -> FakeSession:
            calls.append(unit)
            if unit.round == 2 and calls.count(unit) == 1:
                raise RateLimited("429")
            return FakeSession(pd.DataFrame(), weather=pd.DataFrame({"AirTemp": [20.0]}))

        monkeypatch.setattr("racecast.session_loader._load_session", _load)

        assert backfill_weather() == 1          # round 1 done, stops at round 2
        assert backfill_weather() == 1          # resumes: round 1 skipped, round 2 done
        assert calls == [Unit(2019, 1, "race"), Unit(2019, 2, "race"), Unit(2019, 2, "race")]
