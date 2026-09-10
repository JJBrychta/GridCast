"""Pull data out of the derived DB (``data/racecast.sqlite``) as DataFrames.

The one interface both notebooks and feature code use, so a query is written
once and reused everywhere — no DataSpell cell metadata, no ``%%sql`` magic.

    from racecast.features.query import query, save

    drivers = query("SELECT * FROM drivers")
    save(drivers, "drivers")            # -> data/queries/drivers.parquet

``build_db.py`` builds the DB from ``raw/``; this reads it back. Next step on
top of this: feature engineering (sliding-window aggregates per driver /
constructor), which imports ``query`` just like a notebook does.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from racecast.config import DB_PATH, REPO_ROOT

# Scratch space for ad-hoc query results — safe to wipe.
QUERIES_DIR = REPO_ROOT / "data" / "queries"
# Keeper datasets — the feature matrices you train models on / publish.
DATASETS_DIR = REPO_ROOT / "data" / "datasets"


def query(
    sql: str,
    params: dict | tuple | None = None,
    *,
    db: Path | str = DB_PATH,
) -> pd.DataFrame:
    """Run a read-only SQL query and return the result as a DataFrame.

    Opens and closes its own **read-only** connection — convenient for one-off
    interactive use. In a loop, open one ``sqlite3`` connection yourself and
    pass it to ``pd.read_sql_query`` directly.

    ``params`` feeds SQLite placeholders — ``:name`` with a dict, ``?`` with a
    tuple — so values are never string-formatted into the SQL:

        query("SELECT * FROM drivers WHERE last_name = :ln", {"ln": "Hamilton"})
    """
    db = Path(db)
    if not db.exists():
        raise FileNotFoundError(f"{db} not found — run `make build-db` first")

    # mode=ro: a query can never mutate the DB, and it won't create an empty
    # file if the path is wrong.
    uri = f"file:{db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def save(df: pd.DataFrame, name: str, *, dir: Path | str = QUERIES_DIR) -> Path:
    """Persist a result to ``<dir>/<name>``.

    Defaults to Parquet, which round-trips dtypes exactly (nullable ints, dates,
    booleans) and is ~10x smaller — the right choice for something you'll load
    back for feature engineering. Pass a name ending in ``.csv`` for a
    human-readable / universally-openable copy instead.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / (name if "." in name else f"{name}.parquet")

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path

if __name__ == "__main__":
    # Showcase: dump the drivers dimension and checkpoint it as Parquet.
    drivers = query("SELECT * FROM drivers ORDER BY last_name, first_name")
    print(drivers.head(10).to_string(index=False))
    print(f"\n{len(drivers)} rows")

    path = save(drivers, "drivers")
    print(f"saved -> {path.relative_to(REPO_ROOT)}")
