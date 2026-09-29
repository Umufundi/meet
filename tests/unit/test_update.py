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
