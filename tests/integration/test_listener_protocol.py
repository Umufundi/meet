"""The core against a scripted listener, across a real process boundary.

The fake listener (tests/fixtures/fake_listener) speaks the same NDJSON over
stdout as the real one, so these tests run the actual supervisor thread, event
pump, backpressure, shutdown, and CLI, just without torch or a microphone.
"""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meet import events, runtime, sidecar
from meet.cli import app
from meet.config import POLICY
from meet.memory import db
from meet.session import output
from meet.session.meeting import Meeting
from meet.ui import plain

FAKE = Path(__file__).resolve().parents[1] / "fixtures" / "fake_listener"


def utterance(seq, vector, text, model="ecapa-voxceleb@sb1"):
    return {
        "t": "utterance", "seq": seq, "start_ms": seq * 4000, "end_ms": seq * 4000 + 3000,
        "text": text, "no_speech": 0.01, "embedding": [float(x) for x in vector], "dim": len(vector),
        "model_id": model, "quality": {"speech_s": 3.0, "snr_db": 20.0, "clipping": 0.0, "rms": 0.05},
    }


@pytest.fixture
def scenario(tmp_path, monkeypatch, voices):
    """Write a scenario the fake listener will play, and return the lines."""

    def make(extra_noise=True):
        lines = [
            {"t": "ready", "sample_rate": 16000, "device": "fake", "asr_model": "tiny", "embed_model": "e",
             "wav_path": "x"},
            {"t": "level", "rms": 0.01, "peak": 0.2},
            utterance(1, voices.sample("a"), "Let's start with the budget."),
            utterance(2, voices.sample("b"), "I sent the numbers yesterday."),
            utterance(3, voices.sample("a"), "Thanks, those look right."),
        ]
        if extra_noise:
            lines[2:2] = ["not json at all", "[1,2]", {"t": "hologram"}, {"t": "partial", "text": "Let"}]
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(lines))
        monkeypatch.setenv("FAKE_LISTENER_SCENARIO", str(path))
        return lines

    return make


def fake_listener(tmp_path):
    return sidecar.Listener(tmp_path / "audio.wav", command=[sys.executable, "-m", "meet_listen"],
                            cwd=FAKE)


def test_replay_end_to_end_through_the_plain_frontend(tmp_path, conn, scenario):
    scenario()
    listener = fake_listener(tmp_path)
    listener.replay(tmp_path / "in.wav")
    listener.start()
    meeting = Meeting(conn, title="Budget", use_jev=False, policy=POLICY.first_meeting())
    plain.run(meeting, listener, output_dir=tmp_path, echo=False, idle_timeout=0.3)

    assert [t for _, t, _ in meeting.transcript()] == [
        "Let's start with the budget.", "I sent the numbers yesterday.", "Thanks, those look right."]
    assert meeting.lines[0].cluster_key == meeting.lines[2].cluster_key != meeting.lines[1].cluster_key
    assert len(meeting.questions) == 2
    row = conn.execute("SELECT audio_path, sample_rate, ended_at FROM meeting WHERE id=?",
                       (meeting.id,)).fetchone()
    assert row["audio_path"] and row["sample_rate"] == 16000 and row["ended_at"]
    assert (tmp_path / "audio.wav").exists()
    written = output.write(meeting, tmp_path / "out")
    assert len(written) == 7


def test_live_mode_stops_cleanly_on_request(tmp_path, scenario):
    scenario(extra_noise=False)
    listener = fake_listener(tmp_path)
    listener.start()
    seen = []
    import time

    deadline = time.monotonic() + 10
    while len([e for e in seen if isinstance(e, events.Utterance)]) < 3 and time.monotonic() < deadline:
        seen.extend(listener.drain())
        time.sleep(0.02)
    assert listener.alive  # still listening: live capture never ends on its own
    listener.stop(timeout=5)
    assert not listener.alive
    seen.extend(listener.drain())
    assert isinstance(seen[-1], events.Stopped)
    assert "pretend models" in "\n".join(listener.stderr_tail)


def test_backpressure_sheds_meter_ticks_never_speech(tmp_path):
    listener = sidecar.Listener(tmp_path / "a.wav", command=["unused"], max_queue=8)
    for i in range(6):
        listener._offer(events.Level(rms=0.1, peak=0.1))
    for i in range(10):
        listener._offer(events.Partial(text=str(i), start_ms=0))
    kept = list(listener.drain())
    assert not any(isinstance(e, events.Level) for e in kept)
    assert len(kept) == 8


def test_missing_runtime_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "listener_python", lambda: None)
    with pytest.raises(sidecar.ListenerMissing, match="meet setup"):
        sidecar.Listener(tmp_path / "a.wav")


def test_fatal_listener_error_ends_the_meeting(tmp_path, conn, monkeypatch):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"t": "error", "message": "model load failed: no weights", "fatal": True}]))
    monkeypatch.setenv("FAKE_LISTENER_SCENARIO", str(path))
    listener = fake_listener(tmp_path)
    listener.start()  # live mode: only the fatal error can end this run
    meeting = Meeting(conn, title="t", use_jev=False)
    plain.run(meeting, listener, output_dir=tmp_path, echo=False)
    assert meeting.lines == []
    assert not listener.alive


def test_cli_replay_offline_without_jev(tmp_path, monkeypatch, scenario, voices):
    """`meet replay` through the real CLI and the real `python -m meet_listen`
    launch path, with no network, no API key, and no models."""
    scenario()
    monkeypatch.setenv("MEET_LISTENER_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(FAKE))
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF")
    result = CliRunner().invoke(app, ["replay", str(wav), "--title", "Budget Review", "--no-extract"])
    assert result.exit_code == 0, result.output
    assert "Let's start with the budget." in result.output
    meetings = list((runtime.home() / "meetings").iterdir())
    assert len(meetings) == 1 and meetings[0].name.endswith("budget-review")
    written = {p.name for p in meetings[0].iterdir()}
    assert written >= {"audio.wav", "transcript.md", "minutes.md", "state.json"}
    state = json.loads((meetings[0] / "state.json").read_text())
    assert state["stats"]["utterances"] == 3
    assert state["stats"]["jev_calls"] == 0
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM utterance").fetchone()[0] == 3
