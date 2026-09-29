import json

from meet import events


def test_blank_and_junk_lines_are_ignored():
    assert events.parse_line("") is None
    assert events.parse_line("   \n") is None
    assert events.parse_line("loading whisper...") is None
    assert events.parse_line("[1, 2, 3]") is None


def test_unknown_kind_is_kept_not_raised():
    event = events.parse_line('{"t": "hologram", "x": 1}')
    assert isinstance(event, events.Unknown)
    assert event.raw["x"] == 1


def test_utterance_round_trip():
    raw = {
        "t": "utterance", "seq": 3, "start_ms": 100, "end_ms": 900, "text": "hello",
        "no_speech": 0.1, "embedding": [0.6, 0.8], "dim": 2, "model_id": "m",
        "quality": {"speech_s": 0.8, "snr_db": 12.5, "clipping": 0.0, "rms": 0.02},
    }
    event = events.parse_line(json.dumps(raw))
    assert isinstance(event, events.Utterance)
    assert (event.seq, event.start_ms, event.end_ms, event.text) == (3, 100, 900, "hello")
    assert event.embedding == [0.6, 0.8]
    assert event.quality.snr_db == 12.5


def test_missing_quality_fails_closed():
    """Absent quality evidence must never look like clean audio."""
    event = events.parse_line('{"t": "utterance", "text": "hi", "embedding": [1.0]}')
    assert event.quality.clipping == 1.0
    assert event.quality.speech_s == 0.0
    assert event.quality.snr_db == 0.0


def test_other_kinds():
    assert isinstance(events.parse_line('{"t":"ready","sample_rate":16000}'), events.Ready)
    assert events.parse_line('{"t":"level","rms":0.1,"peak":0.5}').peak == 0.5
    assert events.parse_line('{"t":"partial","text":"so"}').text == "so"
    err = events.parse_line('{"t":"error","message":"x","fatal":true}')
    assert isinstance(err, events.SidecarError) and err.fatal
    assert events.parse_line('{"t":"stopped","wav_path":"a.wav"}').wav_path == "a.wav"


def test_encode_is_one_line():
    line = events.encode({"t": "stop"})
    assert line.endswith("\n") and line.count("\n") == 1
    assert json.loads(line) == {"t": "stop"}
