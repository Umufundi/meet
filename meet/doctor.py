"""`meet doctor`: one screen that says whether a meeting will work, and if not, why.

Every check is a small function returning a `Check`. A failed check always
carries the one thing the user should do next, in plain words; stack traces
belong in `--verbose`. Warnings never block a meeting. Jev in particular is
optional by design and can only ever produce an informational line.

The heavy probes (listener imports, model load, microphone) run in the
listener's own interpreter, because the core deliberately cannot import torch.
They are injected so the logic here is testable without any of them.
"""

from __future__ import annotations

import platform
import shutil
import sqlite3
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata

from . import runtime, sidecar
from .config import db_path, home
from .identity import credentials, jev
from .memory import db

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"

MIN_FREE_GB = 1.0
LOW_FREE_GB = 5.0


@dataclass(frozen=True, slots=True)
class Check:
    section: str
    label: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass(slots=True)
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, section: str, label: str, status: str, detail: str = "", fix: str = "") -> None:
        self.checks.append(Check(section, label, status, detail, fix))

    @property
    def ready(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]


def meet_version() -> str:
    try:
        return metadata.version("meet")
    except metadata.PackageNotFoundError:
        return "dev"


def os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return f"{system} {platform.release()}".strip()


def mic_fix() -> str:
    if runtime.IS_WINDOWS:
        # Windows 11 renamed the page; Windows 10 users cannot find the new name.
        where = (
            "Settings > Privacy & security > Microphone"
            if runtime.windows_build() >= 22000
            else "Settings > Privacy > Microphone"
        )
        return (
            f"Windows may be blocking the microphone. Open {where}, turn on "
            "\"Allow desktop apps to access your microphone\", then run `meet doctor --audio`."
        )
    if runtime.IS_MACOS:
        # macOS 13 replaced System Preferences with System Settings.
        where = (
            "System Settings > Privacy & Security > Microphone"
            if runtime.macos_version() >= (13,)
            else "System Preferences > Security & Privacy > Privacy > Microphone"
        )
        return f"Allow your terminal in {where}, then run `meet doctor --audio`."
    return "Check the input is not muted (e.g. `pavucontrol`), then run `meet doctor --audio`."


# ── individual sections ─────────────────────────────────────────────────


def _runtime(report: Report, probe: dict | None, manifest: dict) -> None:
    s = "Runtime"
    wrong = runtime.python_problem()
    if wrong:
        report.add(s, f"Python {platform.python_version()}", FAIL, *wrong)
    else:
        report.add(s, f"Python {platform.python_version()}", OK)
    blocked = runtime.unsupported_reason()
    if blocked:
        report.add(s, "this computer", FAIL, blocked[0], blocked[1])

    python = runtime.listener_python()
    if python is None or not python.exists():
        report.add(s, "listener runtime", FAIL, "not installed", "Run `meet setup`.")
        return
    report.add(s, "listener runtime", OK, str(python))

    if probe is None:
        return
    if not probe.get("ok"):
        errors = probe.get("errors") or [probe.get("error") or "unknown error"]
        report.add(s, "listener dependencies", FAIL, "; ".join(map(str, errors))[:300],
                   "Run `meet setup --force` to rebuild the listener runtime.")
        return
    installed = probe.get("packages") or {}
    expected = manifest.get("packages") or {}
    drift = sorted(
        f"{name} {expected[name]} -> {installed.get(name)}"
        for name in expected
        if expected.get(name) and installed.get(name) != expected[name]
    )
    if drift:
        report.add(s, "dependencies", WARN, "differ from setup: " + ", ".join(drift),
                   "Run `meet setup --force` to restore the locked versions.")
    else:
        torch = installed.get("torch")
        report.add(s, "dependencies", OK, f"torch {torch}" if torch else "")

    recorded = manifest.get("listener_digest")
    current = runtime.source_digest()
    if recorded and current and recorded != current:
        report.add(s, "listener code", WARN, "changed since `meet setup`",
                   "Run `meet setup` so the runtime matches this version of Meet.")


def _database(report: Report) -> None:
    s = "Runtime"
    try:
        conn = db.connect()
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    except db.DatabaseTooNew as exc:
        report.add(s, "database", FAIL, str(exc), "Install the newer Meet that wrote this database.")
        return
    except (sqlite3.Error, OSError) as exc:
        report.add(s, "database", FAIL, str(exc), f"Check that {db_path()} is readable and not locked.")
        return
    if ok != "ok":
        report.add(s, "database", FAIL, f"integrity check: {ok}",
                   f"Back up {db_path()} and contact support; do not delete it.")
        return
    version = db.schema_version(conn)
    report.add(s, f"database (schema {version})" if version else "database", OK, str(db_path()))


def _models(report: Report, manifest: dict, loaded: dict | None) -> None:
    s = "Models"
    if not manifest.get("models_primed"):
        report.add(s, "speech and voice models", FAIL, "not downloaded yet",
                   "Run `meet setup` (needs internet once; meetings then run offline).")
        return
    report.add(s, f"Whisper {manifest.get('asr_model', '?')}", OK)
    report.add(s, f"speaker model {manifest.get('embed_model', '?')}", OK)
    report.add(s, "Silero VAD", OK)
    if loaded is None:
        return
    if loaded.get("ok"):
        t = loaded.get("timings") or {}
        report.add(s, "offline load", OK, ", ".join(f"{k} {v}s" for k, v in t.items()))
    else:
        report.add(s, "offline load", FAIL, str(loaded.get("error")),
                   "Run `meet setup --force` to download the models again.")


def _audio(report: Report, probe: dict | None, mic: dict | None) -> None:
    s = "Audio"
    if probe is None or not probe.get("ok"):
        return
    if probe.get("audio_error"):
        report.add(s, "audio backend", FAIL, str(probe["audio_error"]),
                   "Linux: install PortAudio (`sudo apt install libportaudio2`). "
                   "Windows/macOS: run `meet setup --force`.")
        return
    devices = probe.get("devices") or []
    if not devices:
        report.add(s, "microphone", FAIL, "no input device found",
                   "Connect or enable a microphone. " + mic_fix())
        return
    default = probe.get("default_device")
    named = next((d["name"] for d in devices if d["index"] == default), None)
    if named:
        report.add(s, f"Microphone: {named}", OK, f"{len(devices)} input(s) available")
    else:
        report.add(s, "microphone", WARN, f"{len(devices)} input(s), no system default",
                   "Pick one with `meet devices`, then `meet start --mic <index>`.")

    if mic is None:
        return
    if not mic.get("ok"):
        report.add(s, "capture", FAIL, str(mic.get("error")), mic_fix())
        return
    if mic.get("all_zero"):
        report.add(s, "signal", FAIL, "the microphone delivered pure silence", mic_fix())
        return
    peak = float(mic.get("peak", 0.0))
    if peak < 0.003:
        report.add(s, "signal", WARN, f"very quiet (peak {peak:.4f})",
                   "Speak during the test, raise the input level, or check a mute switch.")
    else:
        report.add(s, "signal detected", OK, f"peak {peak:.2f}")
    clipping = float(mic.get("clipping", 0.0))
    if clipping > 0.01:
        report.add(s, "clipping", WARN, f"{clipping * 100:.1f}% of samples clipped",
                   "Lower the input level in the OS sound settings.")
    else:
        report.add(s, "no clipping", OK)
    noise = float(mic.get("noise_dbfs", -99.0))
    if noise > -40:
        report.add(s, "background noise", WARN, f"{noise:.0f} dBFS floor",
                   "A noisy room lowers accuracy; move the laptop away from fans or HVAC.")
    else:
        report.add(s, "background noise", OK, f"{noise:.0f} dBFS floor")


def _storage(report: Report) -> None:
    s = "Storage"
    root = home()
    free_gb = shutil.disk_usage(root).free / 1e9
    detail = f"{free_gb:.0f} GB available (a meeting records about 0.12 GB/hour)"
    if free_gb < MIN_FREE_GB:
        report.add(s, "disk space", FAIL, detail, "Free some disk space before recording.")
    elif free_gb < LOW_FREE_GB:
        report.add(s, "disk space", WARN, detail, "Disk is getting full.")
    else:
        report.add(s, f"{free_gb:.0f} GB available", OK)
    probe = root / f".write-test-{uuid.uuid4().hex[:6]}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
        report.add(s, f"{root} writable", OK)
    except OSError as exc:
        report.add(s, f"{root} writable", FAIL, str(exc), "Check folder permissions or set MEET_HOME.")


def _identity(report: Report, embed_model: str | None) -> None:
    s = "Identity"
    try:
        conn = db.connect()
    except (sqlite3.Error, OSError, db.DatabaseTooNew):
        return
    people = conn.execute("SELECT COUNT(*) FROM person").fetchone()[0]
    rows = conn.execute(
        "SELECT model_id, COUNT(*) n FROM voice_sample WHERE revoked_at IS NULL GROUP BY model_id"
    ).fetchall()
    by_model = {r["model_id"]: r["n"] for r in rows}
    usable = by_model.get(embed_model, 0) if embed_model else sum(by_model.values())
    report.add(s, f"{people} people, {usable} usable voice samples", OK if people or not by_model else INFO)
    stale = {m: n for m, n in by_model.items() if embed_model and m != embed_model}
    if stale:
        detail = ", ".join(f"{n} from {m}" for m, n in stale.items())
        report.add(s, "samples from another speaker model", WARN, detail + " are ignored",
                   "Voices recorded under a different model must be re-learned in a meeting.")


def _jev(report: Report) -> None:
    if jev.api_key():
        report.add("Jev", "API key found", OK, "used only to break ties between voices")
    else:
        report.add("Jev", "not configured (optional)", INFO,
                   "ties between voices are asked to you instead",
                   f"To enable: {credentials.where()}")


# ── orchestration ───────────────────────────────────────────────────────


def run(
    *,
    audio: bool = False,
    models: bool = False,
    probe_fn: Callable[[], dict] = sidecar.probe,
    mic_fn: Callable[[], dict] = sidecar.mic_test,
    prime_fn: Callable[[str], dict] | None = None,
) -> Report:
    report = Report()
    manifest = runtime.read_manifest()
    python = runtime.listener_python()
    probe: dict | None = None
    if python is not None and python.exists():
        try:
            probe = probe_fn()
        except sidecar.ListenerMissing:
            probe = None

    _runtime(report, probe, manifest)
    _database(report)

    loaded = None
    if models and manifest.get("models_primed") and probe and probe.get("ok"):
        prime = prime_fn or (lambda m: sidecar.prime(m, offline=True))
        loaded = prime(manifest.get("asr_model", "small.en"))
    _models(report, manifest, loaded)

    mic = mic_fn() if audio and probe and probe.get("ok") and not probe.get("audio_error") else None
    _audio(report, probe, mic)
    _storage(report)
    _identity(report, (probe or {}).get("embed_model") or manifest.get("embed_model"))
    _jev(report)
    return report


def _marks() -> dict[str, str]:
    """Check marks where the console can print them, ASCII where it cannot
    (a redirected Windows console is often still cp1252)."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✓✗!·→".encode(encoding)
        return {OK: "✓", FAIL: "✗", WARN: "!", INFO: "·"}
    except (UnicodeEncodeError, LookupError):
        return {OK: "+", FAIL: "x", WARN: "!", INFO: "-"}


def render(report: Report, *, verbose: bool = False) -> str:
    marks = _marks()
    out = [f"MEET {meet_version()}", os_name(), ""]
    section = None
    for check in report.checks:
        if check.section != section:
            if section is not None:
                out.append("")
            out.append(check.section)
            section = check.section
        line = f"  {marks[check.status]} {check.label}"
        if check.detail and (verbose or check.status in (FAIL, WARN, INFO)):
            line += f"  ({check.detail})" if check.status != FAIL else f"\n      {check.detail}"
        out.append(line)
        if check.fix and (check.status != OK or verbose):
            out.append(f"      {'→' if marks[OK] == '✓' else '->'} {check.fix}")
    out.append("")
    out.append("READY FOR MEETING" if report.ready else f"NOT READY ({len(report.failures)} to fix)")
    return "\n".join(out)
