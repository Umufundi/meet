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


def _ver(v):
    return tuple(int(x) for x in v.split(".")[:2])


def test_platform_pins_survive_a_relock():
    """macOS 12 needs onnxruntime < 1.20 on Python 3.12; Python 3.13 has no
    wheel below 1.20 at all (a real Windows install failed on exactly that).
    Full wheel coverage is checked by scripts/check_wheels.py."""
    reqs = install.parse_lock(LOCK.read_text())
    onnx = {("3.13" if ">= '3.13'" in r.block.splitlines()[0] else "3.12"): _ver(r.version)
            for r in reqs if r.name == "onnxruntime"}
    assert onnx["3.12"] < (1, 20)
    assert (1, 20) <= onnx["3.13"] < (1, 24)
    av = [r for r in reqs if r.name == "av"]
    assert len(av) == 1 and _ver(av[0].version) <= (14, 2)


def test_pip_gauge_counts_packages_and_logs_everything(tmp_path, monkeypatch):
    import io
    import sys

    script = tmp_path / "fake_pip.py"
    script.write_text(
        "print('Collecting numpy==2.5.3 (from -r req.txt (line 3))')\n"
        "print('Ignoring nvidia-x: markers do not match')\n"
        "print('Collecting torch==2.11.0 (from -r req.txt (line 9))')\n"
        "print('Installing collected packages: numpy, torch')\n"
    )
    frames = []

    class Screen(io.StringIO):
        encoding = "utf-8"

        def isatty(self):
            return True

        def write(self, text):
            frames.append(text)
            return len(text)

    monkeypatch.setattr(sys, "stdout", Screen())
    setup = install.Setup(echo=lambda _: None, mic_test=False)
    setup._pip([sys.executable, str(script)], "installing", total=3)
    drawn = "".join(frames)
    assert "0/3" in drawn  # the gauge is up before pip says anything
    assert "Collecting torch" in setup.log_path.read_text()


def test_failed_step_shows_the_last_lines_not_a_traceback(tmp_path):
    import sys

    setup = install.Setup(echo=lambda _: None, mic_test=False)
    code = "print('ERROR: No matching distribution found for onnxruntime==1.19.2'); raise SystemExit(1)"
    try:
        setup._run([sys.executable, "-c", code], "installing locked dependencies")
    except install.SetupFailed as exc:
        assert "No matching distribution" in str(exc)
        assert str(setup.log_path) in exc.fix
    else:
        raise AssertionError("a failing step must raise")


def test_setup_stops_with_a_fix_when_the_source_is_missing(monkeypatch):
    monkeypatch.setattr(runtime, "listener_source", lambda: None)
    out = []
    assert not install.Setup(echo=out.append, mic_test=False).run()
    assert any("listener source" in line for line in out)
    assert any("Reinstall" in line for line in out)


def _fake_steps(monkeypatch, installs):
    monkeypatch.setattr(install.Setup, "make_venv", lambda self: Path("python"))
    monkeypatch.setattr(install.Setup, "install", lambda self, py, root, lock: installs.append(root))
    monkeypatch.setattr(install.Setup, "verify",
                        lambda self: {"ok": True, "packages": {}, "embed_model": "e", "python": "3.13"})
    monkeypatch.setattr(install.Setup, "microphones", lambda self, info: None)
    monkeypatch.setattr(install.doctor, "run", lambda **kw: install.doctor.Report())


def test_setup_reinstalls_the_listener_whenever_its_code_changes(monkeypatch):
    """Any change to listener code, not only to the lock, must rebuild it."""
    installs = []
    _fake_steps(monkeypatch, installs)
    runtime.write_manifest({"listener_digest": "0000stale"})
    assert install.Setup(echo=lambda _: None, mic_test=False, skip_models=True).run()
    assert len(installs) == 1
    assert runtime.read_manifest()["listener_digest"] == runtime.source_digest()

    # Same code again: nothing to rebuild.
    assert install.Setup(echo=lambda _: None, mic_test=False, skip_models=True).run()
    assert len(installs) == 1

    # A listener source file changes (not the lock): rebuilt.
    monkeypatch.setattr(runtime, "source_digest", lambda root=None: "1111changed")
    assert install.Setup(echo=lambda _: None, mic_test=False, skip_models=True).run()
    assert len(installs) == 2
