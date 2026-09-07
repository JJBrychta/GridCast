"""Shared network-call plumbing: retry transient failures, bail on rate limits."""

import time
import warnings
from collections.abc import Callable
from typing import TypeVar

import requests
from fastf1.exceptions import RateLimitExceededError

T = TypeVar("T")


class RateLimited(Exception):
    """The F1/Ergast API rate-limited us.

    The run should stop cleanly and be restarted later rather than sitting in a
    long backoff hammering a closed door. Past schedules are cached on disk, so a
    re-run resumes where this one stopped.
    """


def _is_rate_limit(exc: Exception, warning_text: str) -> bool:
    if isinstance(exc, RateLimitExceededError):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        if exc.response.status_code == 429:
            return True
    # FastF1's get_event_schedule swallows the HTTP 429 (only warns) and then
    # raises a bare ValueError once every backend has failed. Recognise both.
    if "429" in warning_text or "rate limit" in warning_text.lower():
        return True
    if isinstance(exc, ValueError) and "failed to load any schedule data" in str(exc).lower():
        return True
    return False


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return False


def with_retries(
    fn: Callable[[], T],
    *,
    what: str = "request",
    max_retries: int = 5,
    base_delay: float = 2.0,
    pre_delay: float = 0.0,
) -> T:
    """Call ``fn()`` and return its result.

    - rate limit (429, incl. FastF1's swallow-then-ValueError case) -> RateLimited
    - connection error / timeout / HTTP 5xx  -> retry with exponential backoff
    - any other exception                    -> propagate unchanged

    ``pre_delay`` sleeps before the first attempt — a crude throttle for loops
    that call this repeatedly against a rate-limited API.
    """
    if pre_delay:
        time.sleep(pre_delay)

    for attempt in range(1, max_retries + 1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                return fn()
            except Exception as exc:
                warning_text = " ".join(str(w.message) for w in caught)
                if _is_rate_limit(exc, warning_text):
                    raise RateLimited(f"rate limited while fetching {what}") from exc
                if not _is_transient(exc) or attempt == max_retries:
                    raise
                delay = base_delay * 2 ** (attempt - 1)
                print(
                    f"[retry] {what}: {exc!r} — attempt {attempt}/{max_retries}, "
                    f"sleeping {delay:.0f}s"
                )
                time.sleep(delay)

    raise AssertionError("unreachable")
