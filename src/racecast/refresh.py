"""Run after a session you care about (quali, race) has happened: fetch it
from the API, fold it into the derived DB, recompute the feature matrix, and
hand back that race's rows.

    uv run python -m racecast.refresh 2026 17

Every step is either incremental (fetch, DB build — see their own modules)
or cheap to redo in full (the feature matrix, seconds over ~10k rows), so
re-running the whole chain for one new session is fine.
"""

from __future__ import annotations

import pandas as pd

from racecast.features.build import (
    BASIC_FEATURES,
    DRIVER_FEATURES,
    HISTORY_FEATURES,
    TEAM_FEATURES,
    build_matrix,
    load_base,
    relativize,
)
from racecast.features.query import DATASETS_DIR, save
from racecast.ingest.pipeline import run as ingest_and_build_db


def refresh() -> pd.DataFrame:
    """Fetch -> build DB -> rebuild feature_matrix.parquet. Returns the matrix."""
    ingest_and_build_db()

    base = load_base()
    save(base, "feature_base", dir=DATASETS_DIR)

    matrix = build_matrix(base, BASIC_FEATURES + HISTORY_FEATURES, target=["podium", "finish_position", "relevance"])
    matrix = relativize(matrix, driver_cols=DRIVER_FEATURES, team_cols=TEAM_FEATURES)
    save(matrix, "feature_matrix", dir=DATASETS_DIR)
    return matrix


def refresh_and_get_race(year: int, round_number: int) -> pd.DataFrame:
    """refresh(), then filter down to one race's rows."""
    matrix = refresh()
    return matrix[(matrix["year"] == year) & (matrix["round_number"] == round_number)]


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        year, round_number = int(sys.argv[1]), int(sys.argv[2])
        race = refresh_and_get_race(year, round_number)
        print(f"\n{len(race)} rows for {year} round {round_number}")
        print(race[["driver_ref", "team_ref"]].to_string(index=False))
    else:
        refresh()
