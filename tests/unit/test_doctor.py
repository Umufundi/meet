import pytest

from meet import doctor, runtime

GOOD_PROBE = {
    "ok": True, "errors": [], "python": "3.12.7", "platform": "Windows 11",
    "packages": {"torch": "2.11.0", "speechbrain": "1.1.1"}, "embed_model": "ecapa-voxceleb@sb1",
    "devices": [{"index": 1, "name": "Intel Smart Sound", "channels": 2, "rate": 48000}],
    "default_device": 1, "audio_error": None,
}
GOOD_MIC = {"ok": True, "device": "Intel Smart Sound", "peak": 0.4, "clipping": 0.0, "all_zero": False,
            "noise_dbfs": -60.0}


def install_listener(monkeypatch, tmp_path):
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(runtime, "listener_python", lambda: python)


def primed(**extra):
    runtime.write_manifest({
        "models_primed": True, "asr_model": "small.en", "embed_model": "ecapa-voxceleb@sb1",
        "packages": {"torch": "2.11.0", "speechbrain": "1.1.1"},
        "listener_digest": runtime.source_digest(), **extra,
    })


def labels(report, status):
    return [c.label for c in report.checks if c.status == status]


def test_fresh_machine_is_not_ready_and_says_run_setup(monkeypatch):
    monkeypatch.setattr(runtime, "listener_python", lambda: None)
    report = doctor.run(probe_fn=lambda: {"ok": False})
    assert not report.ready
    assert all("meet setup" in c.fix for c in report.failures)
    text = doctor.render(report)
    assert "NOT READY" in text and "meet setup" in text


def test_healthy_machine_is_ready(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed()
    report = doctor.run(audio=True, probe_fn=lambda: GOOD_PROBE, mic_fn=lambda: GOOD_MIC)
    assert report.ready, doctor.render(report)
    assert "Microphone: Intel Smart Sound" in labels(report, doctor.OK)
    assert doctor.render(report).endswith("READY FOR MEETING")


def test_jev_is_never_a_blocker(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed()
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    jev = [c for c in report.checks if c.section == "Jev"]
    assert jev and jev[0].status == doctor.INFO
    assert report.ready


@pytest.mark.parametrize("build, path", [
    (22631, "Settings > Privacy & security > Microphone"),   # Windows 11
    (19045, "Settings > Privacy > Microphone"),              # Windows 10 22H2
])
def test_windows_privacy_block_names_the_right_settings_page(monkeypatch, tmp_path, build, path):
    install_listener(monkeypatch, tmp_path)
    primed()
    monkeypatch.setattr(runtime, "IS_WINDOWS", True)
    monkeypatch.setattr(runtime, "windows_build", lambda: build)
    silent = {**GOOD_MIC, "all_zero": True, "peak": 0.0}
    report = doctor.run(audio=True, probe_fn=lambda: GOOD_PROBE, mic_fn=lambda: silent)
    assert not report.ready
    assert path in report.failures[0].fix


@pytest.mark.parametrize("version, path", [
    ((14, 5), "System Settings > Privacy & Security > Microphone"),
    ((12, 7), "System Preferences > Security & Privacy > Privacy > Microphone"),
])
def test_macos_microphone_fix_matches_the_os_version(monkeypatch, version, path):
    monkeypatch.setattr(runtime, "IS_WINDOWS", False)
    monkeypatch.setattr(runtime, "IS_MACOS", True)
    monkeypatch.setattr(runtime, "macos_version", lambda: version)
    assert path in doctor.mic_fix()


def test_unsupported_computer_is_a_failure_with_a_fix(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed()
    monkeypatch.setattr(runtime, "unsupported_reason", lambda: ("Intel Macs are not supported", "Use M1+"))
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    assert [c.label for c in report.failures] == ["this computer"]


def test_no_microphone(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed()
    report = doctor.run(probe_fn=lambda: {**GOOD_PROBE, "devices": [], "default_device": None})
    assert [c.label for c in report.failures] == ["microphone"]


def test_broken_listener_and_drift(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed(packages={"torch": "2.10.0"})
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    assert "dependencies" in labels(report, doctor.WARN)
    report = doctor.run(probe_fn=lambda: {"ok": False, "errors": ["import torch: DLL load failed"]})
    assert "listener dependencies" in labels(report, doctor.FAIL)
    assert "--force" in report.failures[0].fix


def test_stale_listener_code_is_flagged(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed(listener_digest="0000")
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    assert "listener code" in labels(report, doctor.WARN)


def test_models_not_primed_blocks(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    assert "speech and voice models" in labels(report, doctor.FAIL)


def test_offline_model_load_is_checked_on_request(monkeypatch, tmp_path):
    install_listener(monkeypatch, tmp_path)
    primed()
    report = doctor.run(models=True, probe_fn=lambda: GOOD_PROBE,
                        prime_fn=lambda m: {"ok": False, "error": "LocalEntryNotFoundError"})
    assert "offline load" in labels(report, doctor.FAIL)


def test_samples_from_another_model_are_reported(monkeypatch, tmp_path, conn, voices):
    from meet.memory import db, people

    install_listener(monkeypatch, tmp_path)
    primed()
    live = db.connect()
    slug = db.upsert_person(live, "Marcus")
    people.learn(live, slug, voices.sample("a"), "old-model@0", trust="human", speech_s=3, snr_db=20,
                 clipping=0, rms=0.05)
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    assert "samples from another speaker model" in labels(report, doctor.WARN)


def test_database_from_a_newer_meet_is_a_failure_not_a_crash(monkeypatch, tmp_path):
    import sqlite3

    from meet.config import db_path
    from meet.memory import db

    db_path().parent.mkdir(parents=True, exist_ok=True)
    newer = sqlite3.connect(db_path())
    newer.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    newer.close()
    install_listener(monkeypatch, tmp_path)
    primed()
    report = doctor.run(probe_fn=lambda: GOOD_PROBE)
    assert [c.label for c in report.failures] == ["database"]


def test_ascii_fallback_marks(monkeypatch):
    class Latin1:
        encoding = "cp1252"

    monkeypatch.setattr("sys.stdout", Latin1())
    assert doctor._marks()[doctor.OK] == "+"
