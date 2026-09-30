"""`meet update` against real git repositories on disk."""

import subprocess
import sys

import pytest

from meet import update


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def upstream(tmp_path):
    """A repository standing in for GitHub, with one commit on main."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "user.email", "t@example.com", cwd=repo)
    git("config", "user.name", "t", cwd=repo)
    (repo / "README.md").write_text("v1\n")
    git("add", ".", cwd=repo)
    git("commit", "-q", "-m", "first version", cwd=repo)
    return repo


def commit(repo, message):
    (repo / "README.md").write_text(message + "\n")
    git("commit", "-q", "-am", message, cwd=repo)


def run(upstream, **kw):
    out, installs, setups = [], [], []
    code = update.run(repo=str(upstream), branch="main", echo=out.append,
                      install=lambda src: installs.append((src / "README.md").read_text()),
                      setup=lambda: setups.append(1) or 0, **kw)
    return code, out, installs, setups


def test_first_update_clones_installs_and_runs_setup(upstream):
    code, out, installs, setups = run(upstream)
    assert code == 0
    assert installs == ["v1\n"] and setups == [1]
    assert (update.source_dir() / ".git").is_dir()
    assert update.installed_commit()


def test_nothing_new_means_nothing_reinstalled(upstream):
    run(upstream)
    code, out, installs, setups = run(upstream)
    assert code == 0 and installs == [] and setups == []
    assert "up to date" in out[-1]


def test_new_commits_are_listed_then_installed(upstream):
    run(upstream)
    commit(upstream, "fix the microphone default")
    commit(upstream, "add the slash menu")
    code, out, installs, setups = run(upstream)
    text = "\n".join(out)
    assert "fix the microphone default" in text and "add the slash menu" in text
    assert installs == ["add the slash menu\n"] and setups == [1]


def test_check_reports_without_installing(upstream):
    run(upstream)
    commit(upstream, "something new")
    code, out, installs, setups = run(upstream, check=True)
    assert installs == [] and "update is waiting" in out[-1]
    # ...and the real update afterwards still sees it as new.
    assert run(upstream)[2] == ["something new\n"]


def test_missing_git_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(update.shutil, "which", lambda name: None)
    with pytest.raises(update.UpdateFailed) as err:
        update.run()
    assert "winget install Git.Git" in err.value.fix


def test_unreachable_repo_is_a_clear_failure(tmp_path):
    with pytest.raises(update.UpdateFailed) as err:
        run(tmp_path / "does-not-exist")
    assert "git clone failed" in str(err.value)


@pytest.mark.parametrize("stderr, expect", [
    ("remote: Repository not found.\nfatal: repository 'https://github.com/Umufundi/meet.git/' not found",
     "credential-manager github login"),
    ("fatal: Authentication failed for 'https://github.com/Umufundi/meet.git/'", "credential-manager"),
    ("fatal: unable to access '...': Could not resolve host: github.com", "internet connection"),
    ("fatal: something else entirely", "run `meet update` again"),
])
def test_failures_say_what_to_do(stderr, expect):
    """The first real Windows `meet update` hit "Repository not found": GitHub's
    answer to a private repo when Git sends no login (or the wrong account)."""
    assert expect.lower() in update.failure_fix(stderr).lower()


def test_update_never_imports_the_rest_of_meet():
    """On Windows a loaded DLL cannot be replaced; the updater must load none."""
    probe = (
        "import runpy, sys\n"
        "sys.argv = ['meet', 'update', '--help']\n"
        "try:\n    runpy.run_module('meet', run_name='__main__')\n"
        "except SystemExit:\n    pass\n"
        "print(sorted(m for m in ('numpy', 'meet.cli', 'textual', 'httpx') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"


# ── the launcher ──────────────────────────────────────────────────────


OLD_CMD = b'@"C:\\Users\\3A HealthCare\\.meet\\core\\Scripts\\python.exe" -m meet %*\r\n'


def test_old_windows_launcher_gains_dash_p_safely(tmp_path):
    (tmp_path / "meet.cmd").write_bytes(OLD_CMD)
    assert update.upgrade_launcher(tmp_path)
    new = (tmp_path / "meet.cmd").read_bytes()
    assert new == OLD_CMD.replace(b" -m meet ", b" -Pm meet ")
    # cmd.exe resumes a running batch file at the old length: that must be
    # the final newline, i.e. an empty line, never leftover command text.
    assert len(new) == len(OLD_CMD) + 1
    assert new[len(OLD_CMD):] == b"\n"
    assert not update.upgrade_launcher(tmp_path)  # already upgraded: untouched


def test_old_posix_launcher_gains_dash_p(tmp_path):
    old = b'#!/bin/sh\nexec "/home/flo/.meet/core/bin/python" -m meet "$@"\n'
    (tmp_path / "meet").write_bytes(old)
    assert update.upgrade_launcher(tmp_path)
    assert (tmp_path / "meet").read_bytes() == old.replace(b" -m meet ", b" -Pm meet ")


@pytest.mark.parametrize("content", [
    b"@echo off\r\n\"C:\\x\\meet.exe\" %*\r\n",          # the meet.exe era: not ours to touch
    OLD_CMD + b"\r\n",                                  # anything but the exact old file
    b"echo something the user wrote\r\n",
])
def test_unknown_launchers_are_left_alone(tmp_path, content):
    (tmp_path / "meet.cmd").write_bytes(content)
    assert not update.upgrade_launcher(tmp_path)
    assert (tmp_path / "meet.cmd").read_bytes() == content


def test_dash_p_stops_a_checkout_in_the_current_folder_shadowing_meet(tmp_path):
    """Found on a real Windows PC: `meet update` run from inside an unzipped
    download ran that download's old code, leaving the runtime stale."""
    (tmp_path / "meet").mkdir()
    (tmp_path / "meet" / "__main__.py").write_text("print('SHADOW')\n")
    run = lambda *flags: subprocess.run(  # noqa: E731
        [sys.executable, *flags, "-m", "meet", "--help"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert "SHADOW" in run()          # the trap, as it was
    assert "SHADOW" not in run("-P")  # what the launcher and updater now do
    assert "Usage" in run("-P")
