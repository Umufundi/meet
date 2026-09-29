"""OS abstraction: interpreter layout, runtime resolution, credential stores."""

import json
from pathlib import Path

import pytest

from meet import runtime
from meet.identity import credentials
from tests.conftest import REAL_LOOKUP


def test_venv_layout_per_os(tmp_path):
    assert runtime.venv_python(tmp_path, windows=True) == tmp_path / "Scripts" / "python.exe"
    assert runtime.venv_python(tmp_path, windows=False) == tmp_path / "bin" / "python"


def test_everything_lives_under_meet_home(tmp_path):
    home = tmp_path / "home"
    assert runtime.listener_venv() == home / "runtime" / "listener"
    assert runtime.models_dir() == home / "models"
    assert runtime.manifest_path() == home / "runtime" / "manifest.json"


def test_listener_resolution_order(tmp_path, monkeypatch):
    dev = tmp_path / "dev" / "python"
    monkeypatch.setattr(runtime, "dev_listener_python", lambda: dev)
    assert runtime.listener_python() is None

    dev.parent.mkdir(parents=True)
    dev.touch()
    assert runtime.listener_python() == dev

    built = runtime.venv_python(runtime.listener_venv())
    built.parent.mkdir(parents=True)
    built.touch()
    assert runtime.listener_python() == built

    monkeypatch.setenv("MEET_LISTENER_PYTHON", "/opt/custom/python")
    assert runtime.listener_python() == Path("/opt/custom/python")


def test_installed_runtime_does_not_run_from_the_checkout(tmp_path):
    built = runtime.venv_python(runtime.listener_venv())
    assert runtime.listener_cwd(built) == runtime.home()


def test_listener_env_is_offline_once_models_are_primed():
    env = runtime.listener_env()
    assert env["HF_HOME"].startswith(str(runtime.home()))
    assert "HF_HUB_OFFLINE" not in env
    runtime.write_manifest({"models_primed": True})
    env = runtime.listener_env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["MEET_MODELS_OFFLINE"] == "1"
    assert runtime.listener_env(offline=False).get("HF_HUB_OFFLINE") is None


def test_manifest_round_trip_and_corruption():
    assert runtime.read_manifest() == {}
    runtime.write_manifest({"asr_model": "small.en"})
    assert runtime.read_manifest()["asr_model"] == "small.en"
    assert runtime.read_manifest()["manifest_version"] == runtime.MANIFEST_VERSION
    runtime.manifest_path().write_text("{not json")
    assert runtime.read_manifest() == {}


def test_source_digest_tracks_listener_code(tmp_path):
    src = tmp_path / "listener"
    (src / "meet_listen").mkdir(parents=True)
    (src / "meet_listen" / "a.py").write_text("x = 1\n")
    (src / "requirements.lock").write_text("numpy==1\n")
    first = runtime.source_digest(src)
    (src / "meet_listen" / "a.py").write_text("x = 1\r\n")
    assert runtime.source_digest(src) == first  # a Windows checkout is the same code
    (src / "requirements.lock").write_text("numpy==2\n")
    assert runtime.source_digest(src) != first


def test_listener_source_is_the_checkout_in_development():
    assert runtime.listener_source() == Path(__file__).resolve().parents[2] / "listener"


@pytest.mark.parametrize("system, machine, mac, blocked", [
    ("win32", "AMD64", None, False),          # Windows 10/11 x64
    ("win32", "ARM64", None, True),           # native ARM Python: no torch wheel
    ("darwin", "arm64", (12, 0), False),      # Monterey, the locked floor
    ("darwin", "arm64", (14, 5), False),
    ("darwin", "arm64", (11, 7), True),       # Big Sur: scipy/PyAV have no wheel
    ("darwin", "x86_64", (14, 5), True),      # Intel, or Python under Rosetta
    ("linux", "x86_64", None, False),
])
def test_unsupported_machines_are_named_before_setup(system, machine, mac, blocked):
    reason = runtime.unsupported_reason(system, machine, mac)
    assert (reason is not None) == blocked
    if reason:
        problem, fix = reason
        assert problem and fix


@pytest.mark.parametrize("version, ok", [
    ((3, 11), False), ((3, 12), True), ((3, 13), True), ((3, 14), False),
])
def test_python_range_matches_what_the_lock_is_verified_for(version, ok):
    assert (runtime.python_problem(version) is None) == ok


def test_macos_12_needs_python_312():
    assert runtime.unsupported_reason("darwin", "arm64", (12, 7), (3, 13)) is not None
    assert runtime.unsupported_reason("darwin", "arm64", (12, 7), (3, 12)) is None
    assert runtime.unsupported_reason("darwin", "arm64", (13, 0), (3, 13)) is None


# ── credentials ────────────────────────────────────────────────────────


def test_credential_blob_decoding():
    assert credentials.decode_blob("sk-abc".encode("utf-16-le")) == "sk-abc"
    assert credentials.decode_blob(b"sk-abcd") == "sk-abcd"
    assert credentials.decode_blob(b"sk-abc") == "sk-abc"  # even-length UTF-8
    assert credentials.decode_blob(b"") is None


def test_lookup_dispatches_per_os(monkeypatch):
    calls = []
    monkeypatch.setattr(credentials, "_run", lambda argv: calls.append(argv[0]) or "secret")
    monkeypatch.setattr(credentials, "_windows", lambda service: "win-secret")
    monkeypatch.setattr(credentials.shutil, "which", lambda name: "/usr/bin/" + name)
    assert REAL_LOOKUP("S", platform="win32") == "win-secret"
    assert REAL_LOOKUP("S", platform="darwin") == "secret"
    assert REAL_LOOKUP("S", platform="linux") == "secret"
    assert calls == ["security", "secret-tool"]
    monkeypatch.setattr(credentials.shutil, "which", lambda name: None)
    assert REAL_LOOKUP("S", platform="linux") is None

    def broken(service):
        raise OSError("no advapi32")

    monkeypatch.setattr(credentials, "_windows", broken)
    assert REAL_LOOKUP("S", platform="win32") is None
    assert "cmdkey" in credentials.where("win32")


def test_manifest_is_written_atomically_as_json():
    path = runtime.write_manifest({"a": 1})
    assert json.loads(path.read_text())["a"] == 1
    assert not path.with_suffix(".tmp").exists()
