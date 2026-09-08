"""Tests for racecast.universe — Unit, raw_path, _normalize_dates, iter_units,
load_schedule."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from _helpers import NOW, raises, schedule_df, schedule_event
from racecast.universe import (
    Unit,
    UniverseGenerator,
    _normalize_dates,
    iter_units,
    load_circuits,
    load_schedule,
    raw_path,
)


class TestUnit:
    def test_str_is_zero_padded(self):
        assert str(Unit(2024, 5, "race")) == "2024-R05-race"

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Unit(2024, 5, "race").season = 2025  # type: ignore[misc]

    def test_hashable_and_value_equal(self):
        assert len({Unit(2024, 1, "race"), Unit(2024, 1, "race")}) == 1


class TestRawPath:
    @pytest.fixture(autouse=True)
    def _raw_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("racecast.universe.RAW_DIR", tmp_path)
        self.raw = tmp_path

    def test_full_layout(self):
        assert raw_path(Unit(2024, 5, "race")) == (
            self.raw / "2024" / "round-05" / "race.json"
        )

    def test_round_is_zero_padded_to_two_digits(self):
        assert raw_path(Unit(2024, 5, "race")).parent.name == "round-05"
        assert raw_path(Unit(2024, 20, "race")).parent.name == "round-20"

    def test_session_type_distinguishes_the_file(self):
        assert raw_path(Unit(2024, 1, "race")) != raw_path(Unit(2024, 1, "qualifying"))

    def test_deterministic(self):
        assert raw_path(Unit(2024, 1, "race")) == raw_path(Unit(2024, 1, "race"))


class TestNormalizeDates:
    def test_epoch_millis_become_datetime(self):
        df = pd.DataFrame({"EventDate": [1_700_000_000_000]})
        out = _normalize_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(out["EventDate"])
        assert out["EventDate"].iloc[0] == pd.Timestamp("2023-11-14 22:13:20")

    def test_already_datetime_is_unchanged(self):
        ts = pd.Timestamp("2024-03-02 15:00")
        out = _normalize_dates(pd.DataFrame({"Session5DateUtc": [ts]}))
        assert pd.api.types.is_datetime64_any_dtype(out["Session5DateUtc"])
        assert out["Session5DateUtc"].iloc[0] == ts

    def test_missing_column_is_tolerated(self):
        _normalize_dates(pd.DataFrame({"RoundNumber": [1]}))  # no KeyError

    def test_missing_values_become_nat(self):
        out = _normalize_dates(
            pd.DataFrame({"Session1DateUtc": [1_700_000_000_000, None]})
        )
        assert out["Session1DateUtc"].isna().iloc[1]
        assert not out["Session1DateUtc"].isna().iloc[0]


class TestIterUnits:
    def test_round_zero_is_skipped(self):
        sched = schedule_df(
            [schedule_event(0, ("Race", NOW - timedelta(days=1)))]
        )
        assert list(iter_units(sched, 2024, now=NOW)) == []

    def test_past_session_emitted_future_session_not(self):
        sched = schedule_df(
            [
                schedule_event(
                    1,
                    ("Qualifying", NOW - timedelta(days=1)),
                    ("Race", NOW + timedelta(days=1)),
                    event_date=NOW,
                )
            ]
        )
        assert list(iter_units(sched, 2024, now=NOW)) == [Unit(2024, 1, "qualifying")]

    def test_qualifying_enters_before_the_race_same_weekend(self):
        """Standing on Saturday afternoon: quali is done, the race is tomorrow."""
        now = datetime(2024, 6, 1, 15, 0)
        sched = schedule_df(
            [
                schedule_event(
                    14,
                    ("Qualifying", datetime(2024, 6, 1, 14, 0)),
                    ("Race", datetime(2024, 6, 2, 13, 0)),
                    event_date=datetime(2024, 6, 2),
                )
            ]
        )
        assert list(iter_units(sched, 2024, now=now)) == [Unit(2024, 14, "qualifying")]

    def test_matches_session_by_name_not_by_slot(self):
        """2023 sprint layout puts Qualifying in Session2, not Session4."""
        past = NOW - timedelta(days=2)
        sched = schedule_df(
            [
                schedule_event(
                    1,
                    ("Practice 1", past),
                    ("Qualifying", past),
                    ("Sprint Shootout", past),
                    ("Sprint", past),
                    ("Race", past),
                    event_date=past,
                )
            ]
        )
        assert set(iter_units(sched, 2024, now=NOW)) == {
            Unit(2024, 1, "qualifying"),
            Unit(2024, 1, "race"),
        }

    def test_unwanted_session_types_are_ignored(self):
        past = NOW - timedelta(days=2)
        sched = schedule_df(
            [schedule_event(1, ("Practice 1", past), ("Practice 2", past))]
        )
        assert list(iter_units(sched, 2024, now=NOW)) == []

    def test_falls_back_to_event_date_when_session_date_missing(self):
        sched = schedule_df(
            [schedule_event(1, ("Race", None), event_date=NOW - timedelta(days=1))]
        )
        assert list(iter_units(sched, 2024, now=NOW)) == [Unit(2024, 1, "race")]

    def test_undated_session_emitted_only_for_a_past_season(self):
        sched = schedule_df([schedule_event(1, ("Race", None), event_date=None)])
        assert list(iter_units(sched, 1975, now=NOW)) == [Unit(1975, 1, "race")]
        assert list(iter_units(sched, NOW.year, now=NOW)) == []

    def test_now_defaults_to_the_clock(self):
        sched = schedule_df(
            [
                schedule_event(
                    1,
                    ("Race", datetime(2000, 1, 1)),
                    event_date=datetime(2000, 1, 1),
                )
            ]
        )
        assert list(iter_units(sched, 2000)) == [Unit(2000, 1, "race")]


class TestLoadSchedule:
    @pytest.fixture(autouse=True)
    def _raw_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("racecast.universe.RAW_DIR", tmp_path)
        self.raw = tmp_path

    def _write_cache(self, season: int, df: pd.DataFrame) -> None:
        path = self.raw / str(season) / "schedule.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient="records", indent=2)

    def test_past_season_reads_cache_without_touching_the_api(self, monkeypatch):
        self._write_cache(
            2023,
            pd.DataFrame(
                [
                    {
                        "RoundNumber": 1,
                        "EventName": "X",
                        "EventDate": pd.Timestamp("2023-03-05"),
                        "Session5": "Race",
                        "Session5DateUtc": pd.Timestamp("2023-03-05 15:00"),
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            "racecast.universe.fastf1.get_event_schedule",
            lambda season: raises(AssertionError("API hit for a cached past season")),
        )

        out = load_schedule(2023, current_season=2024)

        assert list(out["RoundNumber"]) == [1]
        assert pd.api.types.is_datetime64_any_dtype(out["EventDate"])

    def test_past_season_cache_miss_fetches_and_writes(self, monkeypatch):
        fresh = pd.DataFrame(
            [
                {
                    "RoundNumber": 1,
                    "EventName": "Monaco",
                    "EventDate": pd.Timestamp("1975-05-11"),
                    "Session5": "Race",
                    "Session5DateUtc": pd.Timestamp("1975-05-11 14:00"),
                }
            ]
        )
        monkeypatch.setattr(
            "racecast.universe.fastf1.get_event_schedule", lambda season: fresh
        )

        out = load_schedule(1975, current_season=2024)

        assert out["EventName"].iloc[0] == "Monaco"
        assert (self.raw / "1975" / "schedule.json").exists()
        # written form round-trips back to datetimes
        reloaded = load_schedule(1975, current_season=2024)
        assert pd.api.types.is_datetime64_any_dtype(reloaded["Session5DateUtc"])

    def test_current_season_always_refetched_even_with_a_cache_file(self, monkeypatch):
        self._write_cache(2024, pd.DataFrame([{"RoundNumber": 1, "EventName": "stale"}]))
        calls: list[int] = []

        def _fetch(season: int) -> pd.DataFrame:
            calls.append(season)
            return pd.DataFrame(
                [{"RoundNumber": 1, "EventName": "fresh", "EventDate": pd.Timestamp("2024-03-02")}]
            )

        monkeypatch.setattr("racecast.universe.fastf1.get_event_schedule", _fetch)

        out = load_schedule(2024, current_season=2024)

        assert calls == [2024]
        assert out["EventName"].iloc[0] == "fresh"
        assert (self.raw / "2024" / "schedule.json").exists()

    def test_empty_frame_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(
            "racecast.universe.fastf1.get_event_schedule",
            lambda season: pd.DataFrame(),
        )

        out = load_schedule(1999, current_season=2024)

        assert out.empty
        assert not (self.raw / "1999" / "schedule.json").exists()


class TestLoadCircuits:
    @pytest.fixture(autouse=True)
    def _raw_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("racecast.universe.RAW_DIR", tmp_path)
        self.raw = tmp_path

    def _stub_ergast(self, monkeypatch, frame: pd.DataFrame) -> list[int]:
        seen: list[int] = []

        class _FakeErgast:
            def __init__(self, **_):
                pass

            def get_race_schedule(self, *, season):
                seen.append(season)
                return frame

        monkeypatch.setattr("racecast.universe.Ergast", _FakeErgast)
        return seen

    def test_past_season_cache_miss_fetches_and_writes(self, monkeypatch):
        frame = pd.DataFrame(
            [{"round": 1, "circuitId": "monaco", "circuitName": "Circuit de Monaco"}]
        )
        seen = self._stub_ergast(monkeypatch, frame)

        out = load_circuits(1975, current_season=2024)

        assert seen == [1975]
        assert out["circuitId"].iloc[0] == "monaco"
        assert (self.raw / "1975" / "circuits.json").exists()

    def test_past_season_reads_cache_without_calling_ergast(self, monkeypatch):
        (self.raw / "1975").mkdir(parents=True)
        pd.DataFrame([{"round": 1, "circuitId": "monaco"}]).to_json(
            self.raw / "1975" / "circuits.json", orient="records"
        )
        seen = self._stub_ergast(monkeypatch, pd.DataFrame())

        out = load_circuits(1975, current_season=2024)

        assert seen == []
        assert out["circuitId"].iloc[0] == "monaco"

    def test_empty_response_is_not_cached(self, monkeypatch):
        self._stub_ergast(monkeypatch, pd.DataFrame())

        out = load_circuits(1999, current_season=2024)

        assert out.empty
        assert not (self.raw / "1999" / "circuits.json").exists()


class TestUniverseGenerator:
    def test_last_season_defaults_to_the_current_year(self):
        gen = UniverseGenerator(first_season=2020)
        assert gen.last_season == datetime.now(timezone.utc).year

    def test_generate_walks_every_season_and_concatenates_units(self, monkeypatch):
        monkeypatch.setattr("racecast.universe.enable_cache", lambda: None)
        schedules = {
            2020: schedule_df(
                [schedule_event(1, ("Race", datetime(2020, 3, 1)), event_date=datetime(2020, 3, 1))]
            ),
            2021: schedule_df(
                [schedule_event(1, ("Race", datetime(2021, 3, 1)), event_date=datetime(2021, 3, 1))]
            ),
        }
        monkeypatch.setattr(
            "racecast.universe.load_schedule",
            lambda season, *, current_season, fetch_delay: schedules[season],
        )
        monkeypatch.setattr(
            "racecast.universe.load_circuits",
            lambda season, *, current_season, fetch_delay: pd.DataFrame(),
        )

        units = UniverseGenerator(first_season=2020, last_season=2021).generate()

        assert units == [Unit(2020, 1, "race"), Unit(2021, 1, "race")]
