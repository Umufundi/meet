import sys

from meet import power


def test_unsupported_platform_reports_false_and_never_raises(monkeypatch):
    monkeypatch.setattr(power.shutil, "which", lambda name: None)
    with power.keep_awake("linux") as awake:
        assert awake is False
    with power.keep_awake("darwin") as awake:
        assert awake is False


def test_helper_process_lives_exactly_as_long_as_the_meeting(monkeypatch):
    started = []
    real_popen = power.subprocess.Popen

    def popen(argv, **kw):
        started.append(argv)
        return real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kw)

    monkeypatch.setattr(power.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(power.subprocess, "Popen", popen)
    with power.keep_awake("darwin") as awake:
        assert awake is True
    assert started and started[0][0] == "caffeinate"


def test_windows_request_is_set_and_cleared(monkeypatch):
    import ctypes
    import types

    calls = []
    kernel32 = types.SimpleNamespace(SetThreadExecutionState=lambda flags: calls.append(flags) or 1)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)
    with power.keep_awake("win32") as awake:
        assert awake is True
        assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED]
    assert calls[-1] == power.ES_CONTINUOUS
