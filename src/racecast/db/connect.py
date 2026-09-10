"""SQLite connection + schema bootstrap for the derived DB (data/racecast.sqlite).

The DB is disposable — a faithful mirror of raw/ built by build_db.py. Deleting
it and rebuilding is the supported "migration".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from racecast.config import DB_PATH

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open (creating if missing) the RaceCast DB.

    Applies schema.sql (idempotent — all CREATE ... IF NOT EXISTS), enforces
    foreign keys, and returns rows as ``sqlite3.Row`` (dict-style access).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL.read_text())
    conn.commit()
    return conn


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


if __name__ == "__main__":
    conn = connect()
    print(f"{DB_PATH}  ({', '.join(table_names(conn))})")
    conn.close()
