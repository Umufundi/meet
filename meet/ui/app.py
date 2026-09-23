"""The full-screen terminal the human actually sits in front of.

Two rules shape the layout.

The transcript must be able to rewrite itself. Naming a speaker forty seconds
late has to change every earlier line from that voice, visibly, so each line is
its own widget keyed by utterance id and its speaker span is patched in place.
Redrawing the whole log would lose scroll position at exactly the moment the
human is reading back to check the correction landed.

The question must never look like a modal. It sits in a panel below the
transcript that keeps scrolling behind it, because the meeting does not stop for
the question and the interface should not pretend otherwise.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Input, Static

from .. import events
from ..session.meeting import Meeting
from ..sidecar import Listener
from . import commands

# Ink and amber. No accent is ever carried by an edge: a line's state is shown
# by the weight and colour of its own text, and the question panel by a full
# border, so nothing depends on a stripe the reader has to learn to interpret.
CSS = """
Screen { background: #14120f; color: #e8e3d9; }

#bar {
    height: 1; dock: top; background: #1f1c17; color: #c9c2b4; padding: 0 1;
}
#bar > Static { width: 1fr; content-align: left middle; }
#bar > .right { content-align: right middle; }

/* New lines appear next to the question and the prompt, where the eye already
   is, instead of at the top of an empty screen. */
#transcript { height: 1fr; padding: 0 1; scrollbar-size: 1 1; align-vertical: bottom; }

.line { height: auto; width: 1fr; }
/* A fixed gutter so a wrapped line stays under the words it belongs to rather
   than sliding back to column zero and breaking the speaker column. */
.gutter { width: 19; height: auto; content-align: right top; }
.body { width: 1fr; height: auto; }

#question {
    display: none; margin: 1 1 0 1; padding: 0 1;
    border: round #d7a13b; background: #1c1812;
}
#question.open { display: block; }

#foot { dock: bottom; height: 4; }
#prompt { height: 3; border: round #3a352c; background: #1a1712; }
#status { height: 1; padding: 0 1; color: #8d8577; background: #14120f; }
"""


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class Line(Horizontal):
    """One transcript row: a fixed gutter, then the words.

    Split into two widgets rather than one string so that a long utterance wraps
    under its own text. A single Static would wrap back to column zero and the
    speaker column, the thing the reader is scanning, would stop being a column.
    """

    def __init__(self, utterance_id: int, cluster_key: str, start_ms: int, text: str) -> None:
        super().__init__(classes="line")
        self.utterance_id = utterance_id
        self.cluster_key = cluster_key
        self.start_ms = start_ms
        self.body = text
        self.speaker = "?"
        self.provisional = False
        self._gutter = Static(classes="gutter")
        self._body = Static(classes="body")

    def compose(self) -> ComposeResult:
        yield self._gutter
        yield self._body

    def on_mount(self) -> None:
        self.paint()

    def paint(self) -> None:
        gutter = Text()
        gutter.append(f"{_clock(self.start_ms)}  ", style="#6f6a5e")
        if self.speaker == "?":
            gutter.append(f"{'?':>11}", style="bold #d7a13b")
        elif self.provisional:
            gutter.append(f"{self.speaker[:11]:>11}", style="italic #a89c82")
        else:
            gutter.append(f"{self.speaker[:11]:>11}", style="bold #e8e3d9")
        self._gutter.update(gutter)
        self._body.update(
            Text("  " + self.body, style="#cfc9bc" if self.provisional else "#e8e3d9")
        )


class QuestionPanel(Static):
    def show(self, meeting: Meeting) -> None:
        pending = sorted(meeting.questions.values(), key=lambda q: q.id)
        if not pending:
            self.remove_class("open")
            return
        question = pending[0]
        body = Text()
        body.append("IDENTITY QUESTION", style="bold #d7a13b")
        body.append(f"   {question.reason}\n", style="#8d8577")
        body.append(f'"{question.text}"\n\n', style="#e8e3d9")
        if question.options:
            for index, (_slug, name, similarity) in enumerate(question.options, start=1):
                body.append(f"  [{index}] {name:<18}", style="#e8e3d9")
                body.append(f"{similarity * 100:.0f}%\n", style="#8d8577")
        body.append(f"  answer:  {question.id} <name>", style="#d7a13b")
        if len(pending) > 1:
            body.append(f"   ({len(pending) - 1} more waiting)", style="#8d8577")
        self.update(body)
        self.add_class("open")


class MeetApp(App):
    CSS = CSS
    BINDINGS = [("ctrl+c", "wrap_up", "end meeting")]

    elapsed = reactive(0)

    def __init__(self, meeting: Meeting, listener: Listener, output_dir: Path) -> None:
        super().__init__()
        self.meeting = meeting
        self.listener = listener
        self.output_dir = output_dir
        self.wav_path: str | None = None
        self.sample_rate = 16_000
        self.level = 0.0
        self._started = time.monotonic()
        self._lines: dict[int, Line] = {}
        self.finished = False
        # Model load takes several seconds before the first frame of audio is
        # even read. Without a visible state the tool looks deaf at exactly the
        # moment a user is deciding whether it works.
        self.ready = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="bar"):
            yield Static(self.meeting.title.upper(), id="title")
            yield Static("", id="meter", classes="right")
        with Vertical(id="foot"):
            yield Static("starting listener: loading speech and voice models", id="status")
            yield Input(placeholder="answer a question, or :help", id="prompt")
        with Vertical():
            yield VerticalScroll(id="transcript")
            yield QuestionPanel(id="question")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.08, self.pump)
        self.set_interval(0.5, self.tick)

    # ── event pump ─────────────────────────────────────────────────────────

    def pump(self) -> None:
        log = self.query_one("#transcript", VerticalScroll)
        appended = False
        for event in self.listener.drain():
            if isinstance(event, events.Ready):
                self.sample_rate = event.sample_rate
                self.wav_path = event.wav_path
                self.ready = True
                self.status(f"listening · {event.asr_model} · {event.embed_model} · :help for commands")
            elif isinstance(event, events.Level):
                self.level = event.peak
            elif isinstance(event, events.Utterance):
                line = self.meeting.on_utterance(event)
                if line is None:
                    continue
                widget = Line(line.utterance_id, line.cluster_key, line.start_ms, line.text)
                widget.speaker = self.meeting.label_for(line.cluster_key)
                widget.provisional = line.provisional
                self._lines[line.utterance_id] = widget
                log.mount(widget)
                appended = True
            elif isinstance(event, events.SidecarError):
                self.status(f"listener: {event.message}")
                if event.fatal:
                    self.action_wrap_up()
            elif isinstance(event, events.Stopped):
                self.wav_path = event.wav_path or self.wav_path
        if appended:
            log.scroll_end(animate=False)
        self.query_one("#question", QuestionPanel).show(self.meeting)

    def tick(self) -> None:
        self.elapsed = int(time.monotonic() - self._started)
        bars = min(12, int(self.level * 40))
        meter = "▌" * bars + "·" * (12 - bars)
        if not self.ready:
            meter = "loading" + "." * (self.elapsed % 4) + " " * (5 - self.elapsed % 4)
        live = "" if self.listener.alive else "  (audio ended)"
        self.query_one("#meter", Static).update(
            f"{meter}   {self.elapsed // 60:02d}:{self.elapsed % 60:02d} elapsed{live}"
        )

    # ── human input ────────────────────────────────────────────────────────

    def on_input_submitted(self, message: Input.Submitted) -> None:
        field = self.query_one("#prompt", Input)
        result = commands.apply(self.meeting, message.value)
        field.value = ""
        if result.message:
            self.status(result.message.splitlines()[0] if "\n" not in result.message else result.message)
        if result.changed:
            self.relabel()
        if result.finished:
            self.action_wrap_up()

    def relabel(self) -> None:
        """Repaint every line whose cluster now resolves to a different name.

        This is the visible half of "one correction, three effects": the human
        sees the whole past of that voice change at once, which is what makes the
        correction feel like it landed rather than like a note filed somewhere.
        """
        for line in self.meeting.lines:
            widget = self._lines.get(line.utterance_id)
            if widget is None:
                continue
            widget.cluster_key = line.cluster_key
            speaker = self.meeting.label_for(line.cluster_key)
            provisional = line.provisional and speaker != "?"
            if widget.speaker != speaker or widget.provisional != provisional:
                widget.speaker = speaker
                widget.provisional = provisional
                widget.paint()
        self.query_one("#question", QuestionPanel).show(self.meeting)

    def status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def action_wrap_up(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.listener.stop()
        for event in self.listener.drain():
            if isinstance(event, events.Utterance):
                self.meeting.on_utterance(event)
            elif isinstance(event, events.Stopped):
                self.wav_path = event.wav_path or self.wav_path
        self.meeting.finish(self.wav_path, self.sample_rate)
        self.exit()
