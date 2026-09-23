"""Line-oriented frontend: works over SSH, in a pipe, and inside a log.

The Textual app is the one a human sits in front of. This one exists because a
meeting tool that only renders into a full-screen TUI cannot be driven by a
script, replayed over a recording, or read back from a terminal buffer later,
and all three of those are how the identity policy gets tuned.

Input is read on its own thread. Nothing here can stall the event pump, so a
question printed to the screen can sit unanswered while the transcript keeps
scrolling past it.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from .. import events
from ..session.meeting import Meeting, Question
from ..sidecar import Listener
from . import commands

# A queue read that times out and an EOF on stdin both need to be told apart
# from a real command, and from each other.
_EOF = object()

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _render_question(question: Question) -> str:
    lines = [
        "",
        f"{BOLD}  IDENTITY QUESTION [{question.id}]{RESET}  {DIM}{question.reason}{RESET}",
        f'   "{question.text}"',
    ]
    for index, (_slug, name, similarity) in enumerate(question.options, start=1):
        lines.append(f"   [{index}] {name:<18} {similarity * 100:.0f}%")
    lines.append(f"   answer: {question.id} <name>   (a new name enrols that person)")
    lines.append("")
    return "\n".join(lines)


def _stdin_thread(inbox: "Queue[str | object]") -> None:
    """Read commands until the human stops typing.

    EOF means "no more commands", not "end the meeting". Piping a short script of
    corrections into a replay must not truncate the audio the moment the script
    runs out, and a live session with no attached terminal should keep recording
    until the audio stops or someone interrupts it.
    """
    for line in sys.stdin:
        inbox.put(line.rstrip("\n"))
    inbox.put(_EOF)


def run(
    meeting: Meeting,
    listener: Listener,
    *,
    output_dir: Path,
    echo: bool = True,
    idle_timeout: float | None = None,
) -> Path:
    """Pump listener events into the meeting until the human or the audio ends.

    `idle_timeout` ends a replay once the listener has exited and the queue has
    drained; live capture passes None and ends on `:end`.
    """
    inbox: Queue[str | object] = Queue()
    stdin_open = True
    threading.Thread(target=_stdin_thread, args=(inbox,), daemon=True, name="stdin").start()

    announced: set[int] = set()
    wav_path: str | None = None
    sample_rate = 16_000
    finished = False
    last_event = time.monotonic()

    def say(text: str) -> None:
        if echo and text:
            print(text, flush=True)

    say(f"{BOLD}{meeting.title}{RESET}  {DIM}meeting {meeting.id}{RESET}")
    if meeting.roster:
        say(f"{DIM}expected: {', '.join(meeting.roster.values())}{RESET}")

    while not finished:
        drained = False
        for event in listener.drain():
            drained = True
            last_event = time.monotonic()
            if isinstance(event, events.Ready):
                sample_rate = event.sample_rate
                wav_path = event.wav_path
                say(f"{DIM}listening · {event.device} · {event.asr_model} · {event.embed_model}{RESET}")
            elif isinstance(event, events.Utterance):
                line = meeting.on_utterance(event)
                if line is not None:
                    speaker = meeting.label_for(line.cluster_key)
                    mark = "~" if line.provisional else " "
                    say(f"{_clock(line.start_ms)} {speaker:>10}{mark} {line.text}")
            elif isinstance(event, events.SidecarError):
                say(f"{DIM}listener: {event.message}{RESET}")
                if event.fatal:
                    finished = True
            elif isinstance(event, events.Stopped):
                wav_path = event.wav_path or wav_path

        for question in meeting.questions.values():
            if question.id not in announced:
                announced.add(question.id)
                say(_render_question(question))

        try:
            command = inbox.get(timeout=0.1 if stdin_open else 0.05)
        except Empty:
            command = None
        if command is _EOF:
            stdin_open = False
        elif isinstance(command, str):
            result = commands.apply(meeting, command)
            say(result.message)
            if result.changed:
                # A correction rewrites history, so redraw what is now true.
                say(f"{DIM}--- transcript after correction ---{RESET}")
                for speaker, text, start_ms in meeting.transcript():
                    say(f"{_clock(start_ms)} {speaker:>10}  {text}")
            finished = finished or result.finished

        if not drained and not listener.alive and idle_timeout is not None:
            if time.monotonic() - last_event > idle_timeout:
                finished = True

    listener.stop()
    # The listener drains its own backlog on shutdown; collect what it emitted.
    for event in listener.drain():
        if isinstance(event, events.Utterance):
            line = meeting.on_utterance(event)
            if line is not None:
                say(f"{_clock(line.start_ms)} {meeting.label_for(line.cluster_key):>10}  {line.text}")
        elif isinstance(event, events.Stopped):
            wav_path = event.wav_path or wav_path

    meeting.finish(wav_path, sample_rate)
    return output_dir
