"""Supervises the listener process and turns its stdout into a queue of events.

The reader runs on its own thread and never touches meeting state. That is the
whole point of the design: a question can sit unanswered on screen for forty
seconds while audio keeps arriving, because nothing in the capture path ever
waits on the human. If the core stalls, the queue grows; if the queue grows past
a bound, the oldest *levels* are dropped and utterances are kept, because a
dropped meter tick is invisible and a dropped utterance is a hole in the record.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from . import events, runtime


class ListenerMissing(RuntimeError):
    """The listener venv has not been built yet. `meet setup` creates it."""


def _interpreter() -> Path:
    python = runtime.listener_python()
    if python is None or not python.exists():
        where = python or runtime.venv_python(runtime.listener_venv())
        raise ListenerMissing(f"listener runtime not found at {where}; run `meet setup` first")
    return python


def listener_command() -> tuple[list[str], Path]:
    """argv prefix and working directory for launching the listener."""
    python = _interpreter()
    return [str(python), "-m", "meet_listen"], runtime.listener_cwd(python)


def _no_window() -> int:
    # A console window flashing up for the sidecar on Windows is noise at best
    # and, when launched from a GUI shell, a stray window the user may close.
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if runtime.IS_WINDOWS else 0


class Listener:
    def __init__(
        self,
        wav_path: Path,
        *,
        device: int | None = None,
        asr_model: str = "small.en",
        max_queue: int = 4096,
        command: list[str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        # `command` replaces `python -m meet_listen`; tests use it to put a
        # scripted listener behind the real process boundary.
        if command is None:
            command, default_cwd = listener_command()
            cwd = cwd or default_cwd
        self.command = command
        self.cwd = cwd or runtime.home()
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
            *self.command,
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
            cwd=str(self.cwd),
            env=runtime.listener_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=_no_window(),
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

    def stop(self, timeout: float | None = None) -> None:
        """Ask the listener to finish, and wait for it.

        No timeout by default: after a long meeting on a slow machine the
        listener may still be transcribing the last minutes, and cutting it off
        loses exactly those lines. The recording itself is already closed and
        safe by then. Ctrl+C while waiting stops it at once.
        """
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
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
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


def _run_listener(
    args: list[str], timeout: float, offline: bool | None = None
) -> subprocess.CompletedProcess:
    argv, cwd = listener_command()
    return subprocess.run(
        [*argv, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
        env=runtime.listener_env(offline=offline),
        timeout=timeout,
        creationflags=_no_window(),
    )


def last_json(text: str) -> dict | None:
    """The last JSON object in a block of output, if any."""
    for line in reversed((text or "").strip().splitlines()):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _json_result(proc: subprocess.CompletedProcess) -> dict:
    """The last JSON object the listener printed, or an error record."""
    data = last_json(proc.stdout)
    if data is not None:
        return data
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {"ok": False, "error": tail[-1] if tail else f"exit {proc.returncode}"}


def probe(timeout: float = 180.0) -> dict:
    """Ask the listener what it has: package versions, embedding model id,
    input devices. Imports the stack but loads no model weights."""
    try:
        return _json_result(_run_listener(["--probe"], timeout))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"listener probe timed out after {timeout:.0f}s"}


def prime(asr_model: str, timeout: float = 3600.0, offline: bool = False) -> dict:
    """Load every model once, downloading if needed, and report what was loaded."""
    try:
        return _json_result(
            _run_listener(["--prime", "--asr-model", asr_model], timeout, offline=offline)
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"model load timed out after {timeout:.0f}s"}


def mic_test(seconds: float = 3.0, device: int | None = None) -> dict:
    """Record a few seconds and report level, clipping, and noise."""
    args = ["--mic-test", str(seconds)]
    if device is not None:
        args += ["--device", str(device)]
    try:
        return _json_result(_run_listener(args, timeout=seconds + 60.0))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "microphone test timed out"}


def check_listener() -> tuple[bool, str]:
    """Report whether the listener runtime exists and can import its stack."""
    try:
        info = probe()
    except ListenerMissing as exc:
        return False, str(exc)
    if not info.get("ok"):
        return False, str(info.get("error") or "import failed")
    return True, "ok"


def list_devices() -> str:
    """Input devices, as the listener's audio backend sees them."""
    return _run_listener(["--devices"], timeout=60).stdout or ""


if __name__ == "__main__":  # manual probe: python -m meet.sidecar
    ok, detail = check_listener()
    print(detail, file=sys.stderr if not ok else sys.stdout)
