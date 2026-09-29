"""`meet setup`: build the listener runtime from the lock, fetch models, verify.

The output is a short list of ticks, not pip's scroll. Every subprocess writes
to a log file under ~/.meet/logs, and a failure prints the last few lines of it
plus the one thing to do next. A user should never have to read a Python
traceback to find out their microphone is blocked.

Dependencies come from `listener/requirements.lock`, installed with
`--require-hashes --no-deps`: exactly the versions that were tested, with no
resolver run on the user's machine. On Linux, PyPI's torch wheels bundle CUDA
(several GB), so torch and torchaudio come from PyTorch's CPU index at the
locked version and the CUDA-only packages are left out. On Windows and macOS the
PyPI wheels are already CPU builds.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import doctor, runtime, sidecar
from .memory import db

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = {"torch", "torchaudio"}
# Present in the universal lock only for Linux CUDA builds of torch.
CUDA_PREFIXES = ("nvidia-", "triton", "cuda-")


class SetupFailed(RuntimeError):
    def __init__(self, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True, slots=True)
class Requirement:
    name: str
    version: str
    block: str


def parse_lock(text: str) -> list[Requirement]:
    """Split a `uv pip compile --generate-hashes` lock into requirement blocks.

    A block is a `name==version ...` line plus its indented continuation lines
    (hashes, `# via` comments). Kept as text so a filtered lock is still a
    valid, hash-checked requirements file.
    """
    out: list[Requirement] = []
    current: list[str] = []

    def close() -> None:
        if current:
            head = current[0].split(";")[0].split("\\")[0].strip()
            name, _, version = head.partition("==")
            out.append(Requirement(name.strip().lower(), version.strip(), "\n".join(current)))
            current.clear()

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") and not line.startswith(" "):
            close()
            continue
        if line.startswith((" ", "\t")):
            if current:
                current.append(line)
            continue
        close()
        current.append(line)
    close()
    return out


def split_lock(text: str, *, cpu_torch: bool) -> tuple[str, dict[str, str]]:
    """The lock to install with hashes, and the torch pins to fetch separately."""
    keep: list[str] = []
    torch: dict[str, str] = {}
    for req in parse_lock(text):
        if cpu_torch and req.name in TORCH_PACKAGES:
            torch[req.name] = req.version
            continue
        if cpu_torch and req.name.startswith(CUDA_PREFIXES):
            continue
        keep.append(req.block)
    return "\n".join(keep) + "\n", torch


class Setup:
    def __init__(
        self,
        *,
        asr_model: str = "small.en",
        force: bool = False,
        skip_models: bool = False,
        mic_test: bool = True,
        torch_index: str = "auto",
        echo: Callable[[str], None] = print,
    ) -> None:
        self.asr_model = asr_model
        self.force = force
        self.skip_models = skip_models
        self.mic_test = mic_test
        self.cpu_torch = torch_index == "cpu" or (torch_index == "auto" and sys.platform.startswith("linux"))
        self.echo = echo
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = runtime.logs_dir() / f"setup-{stamp}.log"
        self.marks = doctor._marks()

    # ── output ─────────────────────────────────────────────────────────

    def ok(self, text: str) -> None:
        self.echo(f"  {self.marks['ok']} {text}")

    def warn(self, text: str) -> None:
        self.echo(f"  {self.marks['warn']} {text}")

    def _run(self, argv: list[str], what: str, timeout: float = 3600) -> None:
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(argv)}\n")
            log.flush()
            try:
                proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SetupFailed(f"{what} failed: {exc}", f"See {self.log_path}") from None
        if proc.returncode != 0:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
            raise SetupFailed(
                f"{what} failed\n      " + "\n      ".join(tail),
                f"Check your internet connection and re-run `meet setup`. Full log: {self.log_path}",
            )

    # ── steps ──────────────────────────────────────────────────────────

    def check_python(self) -> None:
        if sys.version_info < (3, 12):
            raise SetupFailed(
                f"Python {sys.version.split()[0]} is too old",
                "Install Python 3.12 (Windows: `winget install Python.Python.3.12`) and reinstall Meet.",
            )
        self.ok(f"Python {sys.version.split()[0]}")

    def make_dirs(self) -> None:
        for path in (runtime.runtime_dir(), runtime.models_dir(), runtime.logs_dir()):
            path.mkdir(parents=True, exist_ok=True)
        from .config import meetings_dir

        meetings_dir()
        self.ok(f"Meet home {runtime.home()}")

    def source(self) -> tuple[Path, str]:
        root = runtime.listener_source()
        if root is None or not (root / "requirements.lock").exists():
            raise SetupFailed("listener source and lock not found", "Reinstall Meet from a full checkout.")
        return root, (root / "requirements.lock").read_text(encoding="utf-8")

    def make_venv(self) -> Path:
        venv = runtime.listener_venv()
        python = runtime.venv_python(venv)
        if self.force and venv.exists():
            shutil.rmtree(venv)
        if not python.exists():
            self._run([sys.executable, "-m", "venv", str(venv)], "creating the listener environment", 600)
        self.ok("listener environment")
        return python

    def install(self, python: Path, root: Path, lock: str) -> None:
        pip = [str(python), "-m", "pip", "--disable-pip-version-check", "install", "--no-input"]
        hashed, torch = split_lock(lock, cpu_torch=self.cpu_torch)
        if torch:
            pins = [f"{name}=={version}" for name, version in sorted(torch.items())]
            self._run([*pip, "--no-deps", "--index-url", TORCH_CPU_INDEX, *pins], "installing CPU PyTorch")
            self.ok(f"CPU PyTorch {torch.get('torch', '')}")
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text(hashed, encoding="utf-8")
            self._run(
                [*pip, "--require-hashes", "--no-deps", "-r", str(req)], "installing locked dependencies"
            )
        self.ok("locked dependencies")
        self._run([*pip, "--no-deps", "--force-reinstall", str(root)], "installing the listener")
        self.ok("listener")

    def verify(self) -> dict:
        info = sidecar.probe()
        if not info.get("ok"):
            errors = info.get("errors") or [info.get("error")]
            raise SetupFailed(
                "the listener cannot import its stack: " + "; ".join(map(str, errors)),
                f"Run `meet setup --force`. Full log: {self.log_path}",
            )
        self.ok(f"listener imports (torch {info['packages'].get('torch')})")
        return info

    def database(self) -> None:
        conn = db.connect()
        self.ok(f"database (schema {db.schema_version(conn)})")

    def models(self) -> dict:
        self.echo(f"    downloading models ({self.asr_model} + speaker + VAD), this can take a few minutes")
        result = sidecar.prime(self.asr_model, offline=False)
        if not result.get("ok"):
            raise SetupFailed(
                f"model download failed: {result.get('error')}",
                "Check your internet connection (models come from huggingface.co) and re-run `meet setup`.",
            )
        self.ok(f"Whisper {self.asr_model}")
        self.ok(f"speaker model {result.get('embed_model')}")
        self.ok("Silero VAD")
        return result

    def microphones(self, info: dict) -> None:
        devices = info.get("devices") or []
        if info.get("audio_error"):
            self.warn(str(info["audio_error"]))
            return
        if not devices:
            self.warn("no microphone found — " + doctor.mic_fix())
            return
        default = next((d["name"] for d in devices if d["index"] == info.get("default_device")), None)
        self.ok(f"{len(devices)} microphone(s)" + (f", default: {default}" if default else ""))
        if not self.mic_test:
            return
        self.echo("    say a few words — testing the microphone for 3 seconds")
        mic = sidecar.mic_test(3.0)
        if not mic.get("ok"):
            self.warn(f"microphone test failed: {mic.get('error')}")
        elif mic.get("all_zero"):
            self.warn("the microphone delivered pure silence — " + doctor.mic_fix())
        elif float(mic.get("peak", 0.0)) < 0.003:
            self.warn("the microphone is very quiet; check the input level")
        else:
            self.ok(f"signal detected (peak {float(mic['peak']):.2f})")

    # ── orchestration ──────────────────────────────────────────────────

    def run(self) -> bool:
        self.echo("MEET SETUP")
        try:
            self.check_python()
            self.make_dirs()
            root, lock = self.source()
            python = self.make_venv()
            manifest = runtime.read_manifest()
            reinstalled = self.force or manifest.get("listener_digest") != runtime.source_digest(root)
            if reinstalled:
                # A fresh manifest: nothing the previous runtime claimed survives
                # a reinstall until it is proven again below.
                runtime.write_manifest({})
                self.install(python, root, lock)
            else:
                self.ok("listener already up to date")
            info = self.verify()
            record = {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "meet_version": doctor.meet_version(),
                "python": info.get("python"),
                "platform": info.get("platform"),
                "listener_digest": runtime.source_digest(root),
                "lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
                "torch_index": TORCH_CPU_INDEX if self.cpu_torch else "pypi",
                "packages": info.get("packages"),
                "embed_model": info.get("embed_model"),
            }
            if not reinstalled and self.skip_models:
                # Models already fetched for this runtime stay valid.
                for key in ("asr_model", "snapshots", "models_primed"):
                    if key in manifest:
                        record[key] = manifest[key]
            runtime.write_manifest(record)
            self.database()
            if not self.skip_models:
                primed = self.models()
                record.update(
                    asr_model=self.asr_model,
                    embed_model=primed.get("embed_model"),
                    snapshots=primed.get("snapshots"),
                    models_primed=True,
                )
                runtime.write_manifest(record)
            self.microphones(info)
        except SetupFailed as exc:
            self.echo(f"  {self.marks['fail']} {exc}")
            if exc.fix:
                self.echo(f"      {exc.fix}")
            return False
        self.echo("")
        report = doctor.run()
        self.echo(doctor.render(report))
        return report.ready
