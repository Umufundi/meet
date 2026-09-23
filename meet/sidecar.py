"""Supervises the listener process and turns its stdout into a queue of events.

The reader runs on its own thread and never touches meeting state. That is the
whole point of the design: a question can sit unanswered on screen for forty
seconds while audio keeps arriving, because nothing in the capture path ever
waits on the human. If the core stalls, the queue grows; if the queue grows past
a bound, the oldest *levels* are dropped and utterances are kept, because a
dropped meter tick is invisible and a dropped utterance is a hole in the record.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from . import events
from .config import repo_root

LISTENER_PYTHON = repo_root() / "listener" / ".venv" / "bin" / "python"


class ListenerMissing(RuntimeError):
    """The listener venv has not been built yet. `meet setup` creates it."""


class Listener:
    def __init__(
        self,
        wav_path: Path,
        *,
        device: int | None = None,
        asr_model: str = "small.en",
        max_queue: int = 4096,
    ) -> None:
        if not LISTENER_PYTHON.exists():
            raise ListenerMissing(
                f"listener interpreter not found at {LISTENER_PYTHON}; run `meet setup` first"
            )
        self.wav_path = wav_path
        self.device = device
        self.asr_model = asr_model
        self._source: Path | None = None
        self._realtime = False
        self.events: queue.Queue[events.Event | events.Unknown] = queue.Queue(maxsize=max_queue)
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr: threading.Thread | None = None
        self.stderr_tail: list[str] = []

    def replay(self, source: Path, *, realtime: bool = False) -> None:
        """Read a recording instead of the microphone, through the same stages."""
        self._source = source
        self._realtime = realtime

    def start(self) -> None:
        argv = [
            str(LISTENER_PYTHON),
            "-m",
            "meet_listen",
            "--wav",
            str(self.wav_path),
            "--asr-model",
            self.asr_model,
        ]
        if self._source is not None:
            argv += ["--file", str(self._source)]
            if self._realtime:
                argv.append("--realtime")
        elif self.device is not None:
            argv += ["--device", str(self.device)]
        self._process = subprocess.Popen(
            argv,
            cwd=str(repo_root() / "listener"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True, name="listener-stdout")
        self._reader.start()
        self._stderr = threading.Thread(target=self._pump_stderr, daemon=True, name="listener-stderr")
        self._stderr.start()

    def _pump(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            event = events.parse_line(line)
            if event is None:
                continue
            self._offer(event)
        self._offer(events.Stopped(wav_path=str(self.wav_path), duration_ms=0))

    def _pump_stderr(self) -> None:
        """Model downloads and torch warnings land here. Kept for the error panel,
        never printed to stdout, where it would corrupt the NDJSON stream."""
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            line = line.rstrip()
            if not line:
                continue
            self.stderr_tail.append(line)
            del self.stderr_tail[:-40]

    def _offer(self, event: events.Event | events.Unknown) -> None:
        try:
            self.events.put_nowait(event)
            return
        except queue.Full:
            pass
        # Backpressure: shed meter ticks, never speech.
        drained = []
        try:
            while True:
                item = self.events.get_nowait()
                if not isinstance(item, events.Level):
                    drained.append(item)
        except queue.Empty:
            pass
        for item in drained:
            try:
                self.events.put_nowait(item)
            except queue.Full:
                break
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def drain(self) -> Iterator[events.Event | events.Unknown]:
        """Yield everything queued right now, then return. Never blocks."""
        while True:
            try:
                yield self.events.get_nowait()
            except queue.Empty:
                return

    def stop(self, timeout: float = 20.0) -> None:
        """Ask the listener to finish cleanly so the WAV header is written."""
        if self._process is None:
            return
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.write(events.encode({"t": "stop"}))
                self._process.stdin.flush()
                self._process.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def diagnostics(self) -> str:
        code = self._process.poll() if self._process else None
        tail = "\n".join(self.stderr_tail[-12:])
        return f"listener exit={code}\n{tail}" if tail else f"listener exit={code}"


def check_listener() -> tuple[bool, str]:
    """Report whether the listener venv exists and can import its stack."""
    if not LISTENER_PYTHON.exists():
        return False, f"missing interpreter: {LISTENER_PYTHON}"
    probe = subprocess.run(
        [str(LISTENER_PYTHON), "-c", "import sounddevice, faster_whisper, speechbrain; print('ok')"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if probe.returncode != 0:
        return False, (probe.stderr or probe.stdout).strip().splitlines()[-1:][0] if (
            probe.stderr or probe.stdout
        ) else "import failed"
    return True, "ok"


def list_devices() -> str:
    """Input devices, as the listener's audio backend sees them."""
    if not LISTENER_PYTHON.exists():
        raise ListenerMissing(f"listener interpreter not found at {LISTENER_PYTHON}")
    probe = subprocess.run(
        [str(LISTENER_PYTHON), "-m", "meet_listen", "--devices"],
        capture_output=True,
        text=True,
        cwd=str(repo_root() / "listener"),
        timeout=60,
    )
    return probe.stdout or probe.stderr or ""


if __name__ == "__main__":  # manual probe: python -m meet.sidecar
    ok, detail = check_listener()
    print(detail, file=sys.stderr if not ok else sys.stdout)
