"""FastF1's on-disk HTTP cache. Call enable_cache() once, early, per process."""

import fastf1

from racecast.config import FASTF1_CACHE

_enabled = False


def enable_cache() -> None:
    global _enabled
    if _enabled:
        return
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))
    _enabled = True
