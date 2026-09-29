"""Check that every locked package has a wheel for every supported computer.

    python scripts/check_wheels.py            # both locks, needs network (PyPI)

A lock that resolves is not a lock that installs: `uv pip compile --universal`
picks versions by metadata, and a pinned version can still lack a wheel for one
Python version or OS. This found onnxruntime 1.19.2 missing Python 3.13 wheels,
after a real Windows install failed on it. Run it after every re-lock.

For each target (Python 3.12/3.13 x Windows x64, macOS arm64, Linux x64) it
evaluates each requirement's environment marker, then looks for a matching wheel
on PyPI. For macOS it also reports the oldest macOS version the set installs on.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from functools import cache
from pathlib import Path

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parent.parent
LOCKS = [ROOT / "requirements.lock", ROOT / "listener" / "requirements.lock"]
PYTHONS = ["3.12", "3.13"]
PLATFORMS = {
    "windows-x64": {"sys_platform": "win32", "platform_system": "Windows", "platform_machine": "AMD64",
                    "tag": "win_amd64"},
    "macos-arm64": {"sys_platform": "darwin", "platform_system": "Darwin", "platform_machine": "arm64",
                    "tag": "macosx"},
    "linux-x64": {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64",
                  "tag": "manylinux"},
}
# Linux torch comes from the PyTorch CPU index at setup, not PyPI; its PyPI
# CUDA companions are dropped there. Nothing else is exempt.
LINUX_CPU_INDEX = ("torch", "torchaudio")


def requirements(lock: Path) -> list[tuple[str, str, Marker | None]]:
    out = []
    for line in lock.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;\\]+)\s*(?:;\s*([^\\]+?))?\s*\\?$", line)
        if m:
            out.append((m[1].lower(), m[2], Marker(m[3]) if m[3] else None))
    return out


@cache
def wheels(name: str, version: str) -> tuple[str, ...]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as r:
        data = json.load(r)
    return tuple(u["filename"] for u in data["urls"] if u["filename"].endswith(".whl"))


def matches(filename: str, python: str, tag: str) -> tuple[bool, tuple[int, int] | None]:
    """Does this wheel install on (python, platform)? And its macOS floor."""
    _, _, rest = filename[:-4].partition("-")
    parts = rest.split("-")
    py_tag, abi, plat = parts[-3], parts[-2], parts[-1]
    cp = "cp" + python.replace(".", "")
    py_ok = (
        py_tag in (cp, "py3", "py2.py3")
        or ".py3" in py_tag or "py3." in py_tag
        or (abi == "abi3" and py_tag.startswith("cp3") and int(py_tag[3:]) <= int(cp[3:]))
    )
    if not py_ok:
        return False, None
    if plat == "any":
        return True, None
    if tag == "macosx":
        mac = [p for p in plat.split(".") if p.startswith("macosx") and ("arm64" in p or "universal2" in p)]
        if not mac:
            return False, None
        floors = [tuple(int(x) for x in re.match(r"macosx_(\d+)_(\d+)", p).groups()) for p in mac]
        return True, min(floors)
    if tag == "manylinux":
        return ("x86_64" in plat and "linux" in plat), None
    return tag in plat.split("."), None


def main() -> int:
    problems = []
    for lock in LOCKS:
        reqs = requirements(lock)
        print(f"{lock.relative_to(ROOT)}: {len(reqs)} pins")
        for python in PYTHONS:
            for label, target in PLATFORMS.items():
                env = {k: v for k, v in target.items() if k != "tag"}
                env |= {"python_version": python, "python_full_version": python + ".0",
                        "implementation_name": "cpython", "platform_python_implementation": "CPython"}
                floor = (0, 0)
                count = 0
                for name, version, marker in reqs:
                    if marker is not None and not marker.evaluate(env):
                        continue
                    if label == "linux-x64" and name in LINUX_CPU_INDEX:
                        continue
                    count += 1
                    hits = [matches(f, python, target["tag"]) for f in wheels(name, version)]
                    ok = [fl for good, fl in hits if good]
                    if not ok:
                        problems.append(f"{name}=={version}: no wheel for Python {python} on {label}")
                        continue
                    specific = [fl for fl in ok if fl is not None]
                    if specific and None not in ok:
                        floor = max(floor, min(specific))
                note = f", macOS {floor[0]}.{floor[1]}+" if label == "macos-arm64" else ""
                print(f"  Python {python} {label:<12} {count} packages{note}")
    if problems:
        print("\nMISSING WHEELS:")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print("\nevery pin installs on every target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
