"""Shared network-call plumbing: retry transient failures, bail on rate limits.

``with_retries`` is the single choke point for every F1 API call (schedule
fetches and session loads). A 429 reaches it two ways:

  * as an exception — ``get_event_schedule`` raises ``ValueError`` once all
    backends fail; some paths raise ``RateLimitExceededError`` / ``HTTPError``.
  * swallowed — ``session.load`` catches the 429 internally, logs it, and
    returns an empty result with no exception at all.

The log watcher below turns the swallowed case back into a ``RateLimited`` so
both behave identically: the run stops cleanly, nothing half-written, next run
resumes.
"""

import logging
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


class _RateLimitLogProbe(logging.Handler):
    """Trips when a log record — its message or its attached exception —
    mentions a 429. FastF1 / requests_cache log rate-limit errors instead of
    raising them for session loads, so this is the only way ``with_retries``
    can see them.

    NOTE: relies on the ``fastf1`` / ``requests_cache`` loggers staying at
    WARNING or below. Muting them above WARNING blinds this probe.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tripped = False

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage().lower()
        if record.exc_info and record.exc_info[1] is not None:
            text += " " + str(record.exc_info[1]).lower()
        if "429" in text or "too many requests" in text:
            self.tripped = True


@contextmanager
def _watch_logs_for_rate_limit() -> Iterator[_RateLimitLogProbe]:
    probe = _RateLimitLogProbe()
    root = logging.getLogger()
    root.addHandler(probe)
    try:
        yield probe
    finally:
        root.removeHandler(probe)


def with_retries(
    fn: Callable[[], T],
    *,
    what: str = "request",
    max_retries: int = 5,
    base_delay: float = 2.0,
    pre_delay: float = 0.0,
) -> T:
    """Call ``fn()`` and return its result.

    - rate limit (429 — raised, or swallowed-then-logged) -> RateLimited
    - connection error / timeout / HTTP 5xx  -> retry with exponential backoff
    - any other exception                    -> propagate unchanged

    ``pre_delay`` sleeps before the first attempt — a crude throttle for loops
    that call this repeatedly against a rate-limited API.
    """
    if pre_delay:
        time.sleep(pre_delay)

    for attempt in range(1, max_retries + 1):
        with (
            warnings.catch_warnings(record=True) as caught,
            _watch_logs_for_rate_limit() as probe,
        ):
            warnings.simplefilter("always")
            try:
                result = fn()
            except RateLimited:
                raise
            except Exception as exc:
                warning_text = " ".join(str(w.message) for w in caught)
                if _is_rate_limit(exc, warning_text) or probe.tripped:
                    raise RateLimited(f"rate limited while fetching {what}") from exc
                if not _is_transient(exc) or attempt == max_retries:
                    raise
                delay = base_delay * 2 ** (attempt - 1)
                print(
                    f"[retry] {what}: {exc!r} — attempt {attempt}/{max_retries}, "
                    f"sleeping {delay:.0f}s"
                )
                time.sleep(delay)
            else:
                # fn() returned normally, but FastF1 may have swallowed a 429 and
                # handed back an empty result — the probe saw it in the logs.
                if probe.tripped:
                    raise RateLimited(
                        f"rate limited while fetching {what} "
                        f"(429 swallowed by FastF1, seen in logs)"
                    )
                return result

    raise AssertionError("unreachable")
