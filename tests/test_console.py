"""Tests for racecast._console.paint."""

from __future__ import annotations

import pytest

from racecast._console import clear_status, paint, status


@pytest.fixture
def color_on(monkeypatch):
    monkeypatch.setattr("racecast._console._ENABLED", True)


@pytest.fixture
def color_off(monkeypatch):
    monkeypatch.setattr("racecast._console._ENABLED", False)


class TestPaint:
    def test_plain_when_disabled(self, color_off):
        assert paint("hi", "green") == "hi"

    def test_wraps_with_ansi_when_enabled(self, color_on):
        assert paint("hi", "green") == "\033[32mhi\033[0m"

    def test_combines_multiple_styles(self, color_on):
        assert paint("hi", "red", "bold") == "\033[31;1mhi\033[0m"

    def test_no_styles_is_left_plain(self, color_on):
        assert paint("hi") == "hi"


class TestStatus:
    def test_status_is_noop_when_disabled(self, color_off, capsys):
        status("working…")
        clear_status()
        assert capsys.readouterr().out == ""

    def test_status_writes_a_transient_line_when_enabled(self, color_on, capsys):
        status("working…")
        out = capsys.readouterr().out
        assert "working…" in out
        assert out.startswith("\r\033[K")  # returns to col 0 and clears the line
        assert "\n" not in out             # transient — no newline

    def test_clear_status_erases_the_line(self, color_on, capsys):
        clear_status()
        assert capsys.readouterr().out == "\r\033[K"
