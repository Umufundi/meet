# Meet

Put a laptop on the table, `meet start`, and talk. Meet transcribes the room
live, separates speakers, names the ones it knows, marks the ones it doesn't as
`?`, and occasionally asks you *who said this?* One answer fixes that speaker's
whole history and teaches Meet their voice for next time. Everything runs
locally; the internet is needed once, at setup.

## Install

Supported computers:

| | Supported | Not supported |
|---|---|---|
| Windows | 10 and 11, x64 (Intel/AMD) | native ARM64 Python (Snapdragon laptops: install x64 Python instead) |
| macOS | Apple Silicon (M1 and later), macOS 13+ (macOS 12 with Python 3.12) | Intel Macs, macOS 11 and older |
| Linux | x64 | |

Python 3.12 or 3.13. `meet setup` and `meet doctor` check all of this first
and say so plainly.

**Windows** (PowerShell, no admin rights needed):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

**macOS / Linux**:

```sh
sh scripts/install.sh
```

Both need Python 3.12 or 3.13 (`winget install Python.Python.3.12` on Windows). They
install the app into `~/.meet/core`, put `meet` on your PATH, and run
`meet setup`, which:

- builds the listener runtime in `~/.meet/runtime` from a hash-locked
  dependency set (`listener/requirements.lock`);
- downloads Whisper, the ECAPA speaker model, and Silero VAD into
  `~/.meet/models`, then switches the listener to offline mode;
- initializes `~/.meet/voices.db`, tests the microphone, and runs `meet doctor`.

Everything lives under `~/.meet` (`%USERPROFILE%\.meet`); set `MEET_HOME` to
move it.

## Use

```sh
meet doctor            # everything green?  (--audio, --models, --verbose)
meet start -t "Monday staff" -p "Flo, Marcus, Sarah"
meet replay recording.wav
meet people            # who Meet can recognise
meet forget Marcus     # erase one person's voiceprints
```

During a meeting: `3 Marcus` answers question 3, `:wrong Sarah` fixes the last
line, `:merge 2 4`, `:undo`, `:decision`, `:action`, `:end`. `:help` lists them.

### Jev (optional)

With a TypeSafe API key, Meet asks Jev to break ties between voices that are
genuinely close, and to extract decisions and actions at the end. Without one,
those questions come to you and minutes are still written. Store the key in the
OS credential store, not a file:

```powershell
cmdkey /generic:"TypeSafe API Key" /user:typesafe /pass:<key>   # Windows
```
```sh
security add-generic-password -s "TypeSafe API Key" -a typesafe -w <key>   # macOS
```

or set `TYPESAFE_API_KEY`.

## Develop

```sh
uv venv -p 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest          # unit + integration; no models, network, or mic needed
.venv/bin/python -m ruff check .
```

Tests run locally (or on your own VM), not on hosted CI. Integration tests put
a scripted listener (`tests/fixtures/fake_listener`) behind the real process
boundary, so the supervisor, CLI, and output path run for real without torch.

Regenerate a lock after changing dependencies:

```sh
uv pip compile listener/pyproject.toml --universal --python-version 3.12 --generate-hashes -o listener/requirements.lock
uv pip compile pyproject.toml --universal --python-version 3.12 --generate-hashes -o requirements.lock
.venv/bin/python scripts/check_wheels.py   # every pin has a wheel for every supported computer
```

A lock that resolves is not a lock that installs: a pinned version can lack a
wheel for one Python or OS. `check_wheels.py` checks Python 3.12 and 3.13 on
Windows x64, macOS arm64 and Linux x64 against PyPI.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the V1 contract, quality gates, and
the order work is done in.
