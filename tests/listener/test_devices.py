"""Device discovery against a stand-in for sounddevice, shaped like a real
Windows laptop: the same microphones listed once per host API, and
`sd.default.device` left at sounddevice's "no override" value of -1."""

import sys
import types

from meet_listen import diagnose
from meet_listen.audio import default_input


def fake_sounddevice(default_index=1, broken=False):
    devices = [
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000},
        {"name": "Microphone Array (Intel Smart Sound)", "max_input_channels": 2,
         "default_samplerate": 48000},
        {"name": "Stereo Mix", "max_input_channels": 2, "default_samplerate": 48000},
    ]
    sd = types.SimpleNamespace(default=types.SimpleNamespace(device=[-1, -1]))

    def query_devices(device=None, kind=None):
        if kind == "input":
            if broken or default_index is None:
                raise ValueError("No input device matching")
            return {**devices[default_index], "index": default_index}
        return devices

    sd.query_devices = query_devices
    return sd


def test_default_input_comes_from_portaudio_not_the_override_setting():
    # The bug a real Windows 11 install hit: -1 was read as "no default".
    assert default_input(fake_sounddevice(default_index=1)) == 1
    assert default_input(fake_sounddevice(default_index=None)) is None


def test_probe_reports_the_windows_default_microphone(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice(default_index=1))
    devices, default, error = diagnose._devices()
    assert error is None
    assert default == 1
    assert [d["index"] for d in devices] == [1, 2]
    assert next(d["name"] for d in devices if d["index"] == default).startswith("Microphone Array")
