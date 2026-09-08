"""Tiny ANSI-colour helper for the pipeline scripts.

`paint` is a no-op when stdout isn't a TTY or `NO_COLOR` is set, so piping /
redirecting output stays clean.
"""

from __future__ import annotations

import os
import sys

_ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ

_CODES = {
    "green": "32",
    "red": "31",
    "yellow": "33",
    "cyan": "36",
    "dim": "2",
    "bold": "1",
}


def paint(text: str, *styles: str) -> str:
    if not _ENABLED or not styles:
        return text
    seq = ";".join(_CODES[s] for s in styles)
    return f"\033[{seq}m{text}\033[0m"


def status(text: str) -> None:
    """Write a transient one-line status that the next ``print`` overwrites.

    No-op when not a TTY, so piped/redirected output never contains it.
    """
    if _ENABLED:
        sys.stdout.write("\r\033[K" + text)
        sys.stdout.flush()


def clear_status() -> None:
    """Erase a line written by :func:`status`. Call before the real ``print``."""
    if _ENABLED:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
