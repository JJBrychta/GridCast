"""FastF1's on-disk HTTP cache. Call enable_cache() once, early, per process."""

import fastf1

from racecast.config import FASTF1_CACHE

_enabled = False


def enable_cache() -> None:
    """Enable the FastF1 HTTP cache and quiet its console logging.

    ``set_log_level`` lowers FastF1's *handler* level only — records still
    propagate to the root logger, so net.py's 429 log-probe keeps working.
    A notebook that wants the progress logs back can call
    ``fastf1.set_log_level("INFO")`` afterwards.
    """
    global _enabled
    if _enabled:
        return
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))
    fastf1.set_log_level("ERROR")
    _enabled = True
