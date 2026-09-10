"""Live integration tests — hit the real FastF1 / Ergast API.

Skipped unless ``pytest --run-network``. They exist to catch upstream schema or
behaviour drift that the mocked unit tests cannot.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network


def test_load_real_race_results_has_expected_shape():
    from racecast.ingest.cache import enable_cache
    from racecast.ingest.sessions import _load_session
    from racecast.ingest.universe import Unit

    enable_cache()
    session = _load_session(Unit(2024, 1, "race"))

    results = session.results
    assert len(results) == 20
    assert {"DriverId", "Position", "GridPosition", "Status", "Points"} <= set(
        results.columns
    )


def test_universe_generation_for_a_completed_season():
    from racecast.ingest.universe import UniverseGenerator

    units = UniverseGenerator(first_season=2023, last_season=2023, fetch_delay=0.0).generate()

    assert len(units) == 44  # 22 rounds x {qualifying, race}
    assert {u.session_type for u in units} == {"qualifying", "race"}
    assert all(u.season == 2023 for u in units)
