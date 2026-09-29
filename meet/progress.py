"""A one-line live gauge for long setup steps.

pip and the model download can each run for minutes with nothing on screen,
which reads as "hung". This redraws a single line in place, several times a
second, from whatever the step can say about itself (a count, a size, a phase)
plus the elapsed time, so the screen is never still while work is happening.

When output is not a terminal (piped, logged), nothing is drawn: the finished
step still prints its normal tick line, and a log file gets no control codes.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import TextIO

WIDTH = 20


def _glyphs(stream: TextIO) -> tuple[str, str]:
    try:
        "█░".encode(getattr(stream, "encoding", None) or "ascii")
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def bar(done: int, total: int, glyphs: tuple[str, str] = ("█", "░")) -> str:
    total = max(total, 1)
    filled = min(WIDTH, int(WIDTH * min(done, total) / total))
    return glyphs[0] * filled + glyphs[1] * (WIDTH - filled)


class Gauge:
    """Redraw `render()` on one line until stopped. Use as a context manager."""

    def __init__(self, render: Callable[[str], str], *, stream: TextIO | None = None,
                 interval: float = 0.25) -> None:
        self.stream = stream or sys.stdout
        self.render = render
        self.interval = interval
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.glyphs = _glyphs(self.stream)
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = 0

    @property
    def elapsed(self) -> str:
        return clock(time.monotonic() - self.started)

    def draw(self) -> None:
        if not self.live:
            return
        try:
            text = "    " + self.render(self.elapsed)
        except Exception:  # a gauge must never take down the step it describes
            return
        pad = max(0, self._width - len(text))
        self._width = len(text)
        self.stream.write("\r" + text + " " * pad)
        self.stream.flush()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.draw()

    def __enter__(self) -> "Gauge":
        if self.live:
            self.draw()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="gauge")
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.live:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()


def folder_mb(path) -> float:
    """Size of everything under `path`, for download progress. Never raises."""
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total / 1e6
