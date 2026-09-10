"""Tests for racecast.db — connection + schema bootstrap."""

from __future__ import annotations

import sqlite3

import pytest

from racecast.db.connect import connect, table_names

EXPECTED_TABLES = {
    "seasons",
    "circuits",
    "events",
    "sessions",
    "constructors",
    "drivers",
    "results",
    "session_weather",
    "ingested_files",
}


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    yield conn
    conn.close()


class TestConnect:
    def test_creates_the_file_and_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "racecast.sqlite"
        connect(path).close()
        assert path.exists()

    def test_applies_the_full_schema(self, db):
        assert set(table_names(db)) == EXPECTED_TABLES

    def test_is_idempotent(self, tmp_path):
        p = tmp_path / "x.sqlite"
        connect(p).close()
        connect(p).close()  # second run must not raise
        assert set(table_names(connect(p))) == EXPECTED_TABLES

    def test_foreign_keys_are_enforced(self, db):
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO events (season_id, round_number, event_name) "
                "VALUES (999, 1, 'orphan')"
            )
            db.commit()

    def test_strict_typing_is_enforced(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO seasons (year) VALUES ('not an int')")
            db.commit()

    def test_rows_come_back_as_mappings(self, db):
        db.execute("INSERT INTO seasons (year) VALUES (2024)")
        row = db.execute("SELECT * FROM seasons").fetchone()
        assert row["year"] == 2024


class TestResultsTable:
    def test_session_type_check_rejects_unknown(self, db):
        db.execute("INSERT INTO seasons (year) VALUES (2024)")
        db.execute(
            "INSERT INTO events (season_id, round_number, event_name) "
            "VALUES (1, 1, 'Test GP')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO sessions (event_id, session_type) VALUES (1, 'brunch')"
            )
            db.commit()

    def test_one_row_per_driver_per_session(self, db):
        db.executescript(
            """
            INSERT INTO seasons (year) VALUES (2024);
            INSERT INTO events (season_id, round_number, event_name)
                VALUES (1, 1, 'Test GP');
            INSERT INTO sessions (event_id, session_type) VALUES (1, 'race');
            INSERT INTO drivers (driver_ref) VALUES ('ver');
            INSERT INTO constructors (team_ref, name) VALUES ('rb', 'Red Bull');
            INSERT INTO results (session_id, session_type, driver_id, constructor_id, position)
                VALUES (1, 'race', 1, 1, 1);
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO results (session_id, session_type, driver_id, constructor_id, position) "
                "VALUES (1, 'race', 1, 1, 2)"
            )
            db.commit()

    def test_dnf_check_rejects_non_boolean(self, db):
        db.executescript(
            """
            INSERT INTO seasons (year) VALUES (2024);
            INSERT INTO events (season_id, round_number, event_name) VALUES (1, 1, 'GP');
            INSERT INTO sessions (event_id, session_type) VALUES (1, 'race');
            INSERT INTO drivers (driver_ref) VALUES ('ver');
            INSERT INTO constructors (team_ref, name) VALUES ('rb', 'Red Bull');
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO results (session_id, session_type, driver_id, constructor_id, dnf) "
                "VALUES (1, 'race', 1, 1, 2)"
            )
            db.commit()
