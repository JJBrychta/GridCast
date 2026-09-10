"""Shared pytest configuration.

Unit tests never touch the network or the real cache. The one integration test
that does is marked ``@pytest.mark.network`` and skipped unless ``--run-network``
is passed.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that hit the real FastF1 / Ergast API",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``racecast.ingest.net.time.sleep`` with a recorder. Returns the list of
    durations it was asked to sleep for."""
    slept: list[float] = []
    monkeypatch.setattr("racecast.ingest.net.time.sleep", slept.append)
    return slept
