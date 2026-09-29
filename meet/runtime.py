"""Where things live on disk, per operating system, and what was installed there.

The listener runtime used to be `listener/.venv/bin/python` inside the git
checkout, which assumed Unix and tied a working install to wherever the source
happened to be cloned. Everything durable now lives under one root:

    ~/.meet/                       (%USERPROFILE%\\.meet on Windows)
    ├── runtime/listener/          listener virtualenv, built by `meet setup`
    ├── runtime/manifest.json      exactly what was installed, and from what
    ├── models/                    model weights (Hugging Face cache)
    ├── meetings/                  one directory per meeting
    ├── logs/
    └── voices.db

The manifest is the reproducibility record: "this meeting was heard by these
package versions and these model snapshots" must be answerable months later,
because voice embeddings from a different model build are not comparable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from .config import home, repo_root

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

MANIFEST_VERSION = 1


def venv_python(venv: Path, *, windows: bool | None = None) -> Path:
    """The interpreter inside a virtualenv. The layout differs by OS, not by taste."""
    if IS_WINDOWS if windows is None else windows:
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def runtime_dir() -> Path:
    return home() / "runtime"


def listener_venv() -> Path:
    return runtime_dir() / "listener"


def models_dir() -> Path:
    path = home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path() -> Path:
    return runtime_dir() / "manifest.json"


def listener_source() -> Path | None:
    """The listener package source: the checkout in development, the copy
    bundled into the wheel otherwise. None if neither exists."""
    for candidate in (repo_root() / "listener", Path(__file__).resolve().parent / "_listener"):
        if (candidate / "meet_listen" / "__main__.py").exists():
            return candidate
    return None


def dev_listener_python() -> Path:
    """The pre-`meet setup` layout: a venv inside the checkout's listener/."""
    return venv_python(repo_root() / "listener" / ".venv")


def listener_python() -> Path | None:
    """Resolve the listener interpreter, most explicit first.

    1. MEET_LISTENER_PYTHON, for tests and unusual installs;
    2. the runtime `meet setup` builds under ~/.meet;
    3. a developer venv at listener/.venv in the checkout.
    """
    override = os.environ.get("MEET_LISTENER_PYTHON")
    if override:
        return Path(override)
    for candidate in (venv_python(listener_venv()), dev_listener_python()):
        if candidate.exists():
            return candidate
    return None


def listener_cwd(python: Path) -> Path:
    """Run the dev venv from the checkout so edits take effect without a
    reinstall; run the installed runtime from ~/.meet so the checkout's copy
    cannot shadow the version the manifest describes."""
    if python == dev_listener_python() and listener_source() is not None:
        return repo_root() / "listener"
    return home()


def listener_env(*, offline: bool | None = None) -> dict[str, str]:
    """Environment for the listener process.

    Models are cached under ~/.meet/models rather than the user's global
    Hugging Face cache, so `meet forget`-style cleanup and backups have one
    place to look. Once `meet setup` has primed the models, the listener runs
    with the hub offline: a meeting must never depend on the network, and a
    model must never be silently updated between two meetings.
    """
    env = dict(os.environ)
    models = models_dir()
    env.setdefault("HF_HOME", str(models / "hf"))
    env.setdefault("MEET_MODEL_CACHE", str(models / "ecapa"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if offline is None:
        offline = bool(read_manifest().get("models_primed"))
    if offline:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("MEET_MODELS_OFFLINE", "1")
    return env


def source_digest(root: Path | None = None) -> str:
    """A short hash of the listener's code and lock.

    Stored in the manifest at setup time; `meet doctor` compares it to the
    current source so an updated checkout with a stale runtime is caught before
    a meeting rather than halfway through one.
    """
    root = root or listener_source()
    if root is None:
        return ""
    h = hashlib.sha256()
    code = sorted((root / "meet_listen").glob("*.py"))
    files = [*code, root / "pyproject.toml", root / "requirements.lock"]
    for path in files:
        if path.exists():
            h.update(path.name.encode())
            h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def read_manifest() -> dict:
    try:
        return json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_manifest(data: dict) -> Path:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest_version": MANIFEST_VERSION, **data}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path
