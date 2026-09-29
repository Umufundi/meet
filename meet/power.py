"""Keep the computer awake while a meeting is being recorded.

A laptop on a meeting table sees no keyboard or mouse for an hour, and Windows
and macOS will happily sleep it mid-sentence, which stops the recording. This
asks the OS not to sleep for as long as the meeting runs, and nothing more:
the screen may still turn off, and closing a laptop lid still sleeps it.

    Windows   SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    macOS     `caffeinate -i -w <pid>` (ends by itself if Meet dies)
    Linux     `systemd-inhibit` where available

Every path is best effort. Failing to hold the machine awake is reported, never
fatal: a meeting that might be interrupted by sleep beats no meeting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake(platform: str | None = None) -> Iterator[bool]:
    """Hold off system sleep inside the block. Yields whether it worked."""
    platform = platform or sys.platform
    if platform == "win32":
        yield from _windows()
    elif platform == "darwin":
        yield from _helper(["caffeinate", "-i", "-w", str(os.getpid())])
    elif shutil.which("systemd-inhibit"):
        yield from _helper(["systemd-inhibit", "--what=sleep:idle", "--who=meet",
                            "--why=recording a meeting", "sleep", "infinity"])
    else:
        yield False


def _windows() -> Iterator[bool]:
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        held = bool(kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        yield False
        return
    try:
        yield held
    finally:
        # The request belongs to this thread; clearing it restores normal sleep.
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _helper(argv: list[str]) -> Iterator[bool]:
    if not shutil.which(argv[0]):
        yield False
        return
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
