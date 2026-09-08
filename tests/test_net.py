"""Tests for racecast.net — retry / rate-limit classification and with_retries."""

from __future__ import annotations

import logging

import pytest
import requests
from fastf1.exceptions import RateLimitExceededError

from _helpers import raises
from racecast.net import (
    RateLimited,
    _is_rate_limit,
    _is_transient,
    _RateLimitLogProbe,
    with_retries,
)


def http_error(status: int) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"{status} error", response=response)


class _Flaky:
    """Callable that raises ``exc`` for its first ``fails`` calls, then returns."""

    def __init__(self, *, fails: int, exc: BaseException, value: str = "ok"):
        self.fails = fails
        self.exc = exc
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return self.value


# --------------------------------------------------------------------------- #
#  _is_rate_limit
# --------------------------------------------------------------------------- #

class TestIsRateLimit:
    @pytest.mark.parametrize(
        ("exc", "warning_text", "expected"),
        [
            (RateLimitExceededError("x"), "", True),
            (http_error(429), "", True),
            (http_error(500), "", False),
            (ValueError("Failed to load any schedule data."), "", True),
            (ValueError("unrelated"), "", False),
            (RuntimeError("x"), "Request returned: 429", True),
            (RuntimeError("x"), "hit the RATE LIMIT", True),
            (RuntimeError("x"), "all good", False),
        ],
    )
    def test_classification(self, exc, warning_text, expected):
        assert _is_rate_limit(exc, warning_text) is expected


class TestIsTransient:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (requests.exceptions.ConnectionError(), True),
            (requests.exceptions.Timeout(), True),
            (http_error(503), True),
            (http_error(404), False),
            (ValueError(), False),
        ],
    )
    def test_classification(self, exc, expected):
        assert _is_transient(exc) is expected


# --------------------------------------------------------------------------- #
#  _RateLimitLogProbe
# --------------------------------------------------------------------------- #

class TestRateLimitLogProbe:
    def test_trips_on_429_in_message(self):
        probe = _RateLimitLogProbe()
        probe.emit(logging.makeLogRecord({"msg": "Request returned: 429"}))
        assert probe.tripped

    def test_trips_on_too_many_requests_phrase(self):
        probe = _RateLimitLogProbe()
        probe.emit(logging.makeLogRecord({"msg": "server said Too Many Requests"}))
        assert probe.tripped

    def test_trips_on_429_only_in_attached_exception(self):
        probe = _RateLimitLogProbe()
        try:
            raise http_error(429)
        except Exception as exc:
            record = logging.makeLogRecord(
                {"msg": "load failed", "exc_info": (type(exc), exc, exc.__traceback__)}
            )
        probe.emit(record)
        assert probe.tripped

    def test_ignores_benign_failure_records(self):
        probe = _RateLimitLogProbe()
        probe.emit(
            logging.makeLogRecord(
                {"msg": "Failed to load driver list and session results!"}
            )
        )
        assert not probe.tripped


# --------------------------------------------------------------------------- #
#  with_retries
# --------------------------------------------------------------------------- #

@pytest.mark.usefixtures("no_sleep")
class TestWithRetries:
    def test_returns_value_on_success(self):
        assert with_retries(lambda: 42) == 42

    def test_calls_fn_exactly_once_on_success(self):
        fn = _Flaky(fails=0, exc=RuntimeError())
        with_retries(fn)
        assert fn.calls == 1

    def test_pre_delay_sleeps_before_first_attempt(self, no_sleep):
        with_retries(lambda: 1, pre_delay=5.0)
        assert no_sleep == [5.0]

    def test_no_pre_delay_when_zero(self, no_sleep):
        with_retries(lambda: 1, pre_delay=0.0)
        assert no_sleep == []

    def test_retries_transient_then_succeeds(self):
        fn = _Flaky(fails=2, exc=requests.exceptions.ConnectionError())
        assert with_retries(fn, max_retries=5) == "ok"
        assert fn.calls == 3

    def test_gives_up_after_max_retries(self):
        fn = _Flaky(fails=99, exc=requests.exceptions.ConnectionError())
        with pytest.raises(requests.exceptions.ConnectionError):
            with_retries(fn, max_retries=3)
        assert fn.calls == 3

    def test_429_raises_rate_limited_immediately(self, no_sleep):
        fn = _Flaky(fails=99, exc=http_error(429))
        with pytest.raises(RateLimited):
            with_retries(fn)
        assert fn.calls == 1          # not retried
        assert no_sleep == []         # no backoff

    def test_schedule_value_error_raises_rate_limited(self):
        with pytest.raises(RateLimited):
            with_retries(lambda: raises(ValueError("Failed to load any schedule data.")))

    def test_swallowed_429_in_logs_raises_even_when_fn_returns(self):
        def fn() -> str:
            logging.getLogger("racecast.test").warning("Request returned: 429")
            return "empty-frame"      # fn returns normally — the 429 was swallowed

        with pytest.raises(RateLimited, match="swallowed"):
            with_retries(fn)

    def test_clean_empty_load_returns_normally(self):
        def fn() -> str:
            logging.getLogger("racecast.test").warning(
                "No result data for this session available on Ergast!"
            )
            return "empty-frame"

        assert with_retries(fn) == "empty-frame"

    def test_preexisting_rate_limited_passes_through_unchanged(self):
        with pytest.raises(RateLimited, match="already classified"):
            with_retries(lambda: raises(RateLimited("already classified")))

    def test_unexpected_exception_propagates(self):
        with pytest.raises(KeyError):
            with_retries(lambda: raises(KeyError("boom")))
