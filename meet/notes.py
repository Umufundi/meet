"""Hand the finished recording to the after-meeting notes pass, if installed.

The notes pass (transcript clean-up, task tagging, summary and minutes) is slow
and heavy, so it runs in its own window after the meeting has been written,
never inside the live loop. It is optional: without it the meeting record is
exactly what Meet always wrote.

MEET_NOTES_CMD overrides where the command lives; the default is the
meet-notes.cmd that the local post-processing setup installs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def command() -> Path | None:
    configured = os.environ.get("MEET_NOTES_CMD")
    path = Path(configured) if configured else Path.home() / ".meet-post" / "meet-notes.cmd"
    return path if path.is_file() else None


def launch(directory: Path, names: list[str]) -> Path | None:
    """Start the notes pass in a new window. Returns the command used, or None.

    Names go through names.txt next to the audio, never the command line: a
    batch file re-parses its arguments, and a name with & or % in it would be
    run or expanded by cmd.
    """
    cmd = command()
    if cmd is None or not (directory / "audio.wav").is_file():
        return None
    (directory / "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    try:
        if sys.platform == "win32":
            subprocess.Popen([str(cmd), str(directory)], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen([str(cmd), str(directory)], start_new_session=True)
    except OSError:
        return None
    return cmd
