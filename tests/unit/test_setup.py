from pathlib import Path

from meet import install, runtime

LOCK = Path(__file__).resolve().parents[2] / "listener" / "requirements.lock"


def test_committed_lock_parses_and_pins_everything():
    reqs = install.parse_lock(LOCK.read_text())
    names = {r.name for r in reqs}
    assert {"torch", "torchaudio", "speechbrain", "faster-whisper", "silero-vad", "numpy"} <= names
    for r in reqs:
        assert r.version, r.name
        assert "--hash=sha256:" in r.block, r.name


def test_torch_and_torchaudio_are_a_matched_pair():
    pins = {r.name: r.version for r in install.parse_lock(LOCK.read_text())}
    assert pins["torch"].split(".")[:2] == pins["torchaudio"].split(".")[:2]


def test_cpu_split_moves_torch_out_and_drops_cuda():
    text = LOCK.read_text()
    hashed, torch = install.split_lock(text, cpu_torch=True)
    assert set(torch) == {"torch", "torchaudio"}
    kept = {r.name for r in install.parse_lock(hashed)}
    assert not kept & {"torch", "torchaudio"}
    assert not [n for n in kept if n.startswith(install.CUDA_PREFIXES)]
    everything, none = install.split_lock(text, cpu_torch=False)
    assert none == {} and len(install.parse_lock(everything)) == len(install.parse_lock(text))


def test_parse_keeps_markers_and_continuations():
    text = (
        "# header\n"
        "a==1.0 \\\n    --hash=sha256:aa\n    # via b\n"
        "nvidia-x==2 ; sys_platform == 'linux' \\\n    --hash=sha256:bb\n"
    )
    reqs = install.parse_lock(text)
    assert [(r.name, r.version) for r in reqs] == [("a", "1.0"), ("nvidia-x", "2")]
    assert reqs[0].block.count("\n") == 2


def test_setup_refuses_an_unsupported_computer_before_installing(monkeypatch):
    monkeypatch.setattr(runtime, "unsupported_reason", lambda: ("macOS 11.7 is too old", "Needs macOS 12"))
    monkeypatch.setattr(install.Setup, "make_venv", lambda self: (_ for _ in ()).throw(AssertionError))
    out = []
    assert not install.Setup(echo=out.append, mic_test=False).run()
    assert any("macOS 11.7 is too old" in line for line in out)


def test_listener_lock_installs_on_macos_12_and_windows_x64():
    """The pins that hold the macOS floor must survive a re-lock."""
    pins = {r.name: r.version for r in install.parse_lock(LOCK.read_text())}
    onnx = tuple(int(x) for x in pins["onnxruntime"].split(".")[:2])
    av = tuple(int(x) for x in pins["av"].split(".")[:2])
    assert onnx < (1, 20) and av <= (14, 2)


def test_setup_stops_with_a_fix_when_the_source_is_missing(monkeypatch):
    monkeypatch.setattr(runtime, "listener_source", lambda: None)
    out = []
    assert not install.Setup(echo=out.append, mic_test=False).run()
    assert any("listener source" in line for line in out)
    assert any("Reinstall" in line for line in out)
