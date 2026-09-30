# Roadmap to 9.5/10

Meet is run as a closed-loop production program, not "implement features until
it feels done". Order of work:

```
Windows installable → reliable recording → good transcript → good speaker
boundaries → correct identity → low-friction corrections → trusted transcript
→ trusted minutes → 9.5/10
```

Always fix the weakest measured link. Do not redesign UI, add integrations, or
add features while a core metric is below target.

## V1 contract

```
meet setup  (once)  →  meet doctor (all green)  →  meet start  →  laptop on table
  ├─ transcript appears live            ├─ unknown people become ?
  ├─ speakers are separated             ├─ Meet occasionally asks "who said this?"
  ├─ known people get names             ├─ a correction fixes history and teaches the voice
  └─ the meeting NEVER stops listening
:end → audio.wav, transcript.md/json, speakers.json, minutes.md,
       decisions.json, actions.json, state.json
```

**Not in V1:** Teams integration, mobile app, cloud accounts, video, dashboards,
calendar integrations, meeting bots, multi-device recording, enterprise admin.
(Parked with a plan under "Later" below: Teams live bot, phone access.)

**Invariants:** fail closed on identity (uncertain → `?`), fail open on
recording (anything else may fail; audio capture continues). Jev is optional;
an API outage never stops a meeting.

## Scoring

100 points; 9.5/10 means ≥ 95 overall **and no critical area below 9/10**.

| Area | Weight |
|---|---|
| Installation & first run | 10 |
| Capture reliability | 15 |
| Transcription | 15 |
| Speaker segmentation | 15 |
| Speaker identity | 20 |
| Human correction UX | 10 |
| Output / minutes | 5 |
| Recovery / data integrity | 5 |
| Privacy | 5 |

## Hard production gates

| Gate | Target |
|---|---|
| Fresh Windows installs succeeding | ≥ 95%, then 99%+ |
| Lost audio under supported conditions | 0 |
| Recoverable raw recording after crash | ~100% |
| Confidently wrong speaker | < 1%, pushing to < 0.25% |
| Identity questions/hour, established participants | < 2, ideally < 1 |
| Settled utterance latency | median < 2.5 s, p95 < 5 s |
| One correction → corresponding lines corrected | 100% |
| Questions over repeat meetings | trend down |
| WER | < 10% quiet office, < 15% meeting room |

## Queue

Status: ✅ done · 🟡 partial · ⬜ not started

### P0: installable and testable

| # | Item | Status |
|---|---|---|
| 001 | Test harness + core invariants | ✅ `tests/`: unit, integration (fake listener over the real process boundary), listener endpointing |
| 002 | Windows OS/path abstraction | ✅ `meet/runtime.py`; everything under `~/.meet`, runtime detached from the checkout |
| 003 | `meet setup` | ✅ `meet/install.py`; venv, locked install, models, DB, mic test, doctor |
| 004 | Lock listener dependencies / model versions | 🟡 hash-locked packages (torch/torchaudio 2.11 matched pair); model snapshots **recorded** in the manifest at setup, not yet **pinned** (set `WHISPER_REVISION` / `EMBED_REVISION` in `listener/meet_listen/models.py` once hashes are chosen) |
| 005 | Upgrade `meet doctor` | ✅ `meet/doctor.py`; `--audio`, `--models`, `--verbose`; every failure carries a fix |
| 006 | Windows `install.ps1` bootstrap | ✅ `scripts/install.ps1` (+ `install.sh`) |
| 007 | Cross-platform Jev credentials | ✅ env → Credential Manager / Keychain / Secret Service |
| 008 | Fresh Windows installation test | ✅ real Windows 11 laptop (Python 3.13): install.ps1 → setup → doctor READY → `meet update`. Found and fixed: PS 5.1 `$PSScriptRoot`, Python 3.13 lock gap, default-mic misreport, private-repo update sign-in. A first real meeting (`meet start`) is still to run |

**— FIRST REAL GATE —**

| # | Item | Status |
|---|---|---|
| 009 | Live microphone smoke test | 🟡 `meet doctor --audio` / setup mic test exist; needs real-device runs |
| 010 | Capture integrity counters (overflows, queue depth, backlog, write errors, worker latency) | ⬜ |
| 011 | Crash recovery (CREATED → RECORDING → PROCESSING → FINALIZED, recover from WAV) | ⬜ |
| 012 | Fixture/replay benchmark runner | ⬜ |

**— RELIABILITY GATE —**

| # | Item | Status |
|---|---|---|
| 013 | ASR benchmark corpus | ⬜ |
| 014 | ASR latency/accuracy tuning | ⬜ |
| 015 | Speaker-change detection within long spans | ⬜ |
| 016 | Overlap/cross-talk backend experiment (optional backend; core stays backend-independent) | ⬜ |
| 017 | Diarization ground-truth scoring (DER, over-merge vs over-split) | ⬜ |

**— AUDIO INTELLIGENCE GATE —**

| # | Item | Status |
|---|---|---|
| 018 | Speaker identification benchmark (false accept/reject, confidently-wrong rate) | ⬜ |
| 019 | Threshold calibration on real voices | ⬜ |
| 020 | Longitudinal learning tests | ⬜ |
| 021 | Optional `meet learn <name>` enrollment | ⬜ |
| 022 | Identity-question UX (`> 2` picks option 2, shortcuts) | ⬜ |
| 023 | Post-meeting `meet review` | ⬜ |

**— IDENTITY GATE —**

| # | Item | Status |
|---|---|---|
| 024 | Structured action/decision validation | ⬜ |
| 025 | Evidence-linked minutes (speaker, timestamp, utterance, source) | ⬜ |
| 026 | DB migrations | 🟡 `PRAGMA user_version` stamped (schema 1); a newer DB is refused; no migration steps yet |
| 027 | Windows data protection (DPAPI / ACLs) | ⬜ |
| 028 | Retention / delete / export semantics | ⬜ |

**— PRODUCT GATE —**

| # | Item | Status |
|---|---|---|
| 029 | 10-room real-world gauntlet | ⬜ |
| 030 | Adversarial suite | ⬜ |
| 031 | Regression corpus (every bug becomes a fixture) | ⬜ |
| 032 | `meet benchmark` | ⬜ |
| 033 | Weakest-link loop until ≥ 9.5 | ⬜ |

## Known issues found so far

- **Blip segments (→ 013–015).** `Endpointer.min_segment_ms` is compared with the
  padded segment (pre-roll + speech + 700 ms hangover), so a single 32 ms VAD
  blip (door, cough) becomes a ~0.9 s segment sent to Whisper, which can
  hallucinate text on noise. Pinned by a strict `xfail` in
  `tests/listener/test_endpointer.py`. Fix against the ASR corpus so short real
  answers ("yes") are not dropped blind.
- **Two people with the same name (→ 022).** Names map to one slug
  case-insensitively, so two different Michaels become one person. Needs a
  disambiguation step when naming.
- **Linux setup needs the PyTorch CPU index** (download.pytorch.org). Where it
  is blocked, `meet setup --torch-index pypi` works but pulls the CUDA build
  (~5 GB). Windows and macOS always use PyPI's CPU wheels.
- **Platform floor.** Windows 10/11 x64; Apple Silicon on macOS 13+ (12 with
  Python 3.12); Linux x64; Python 3.12 or 3.13 only.
  PyTorch ships no Intel-Mac or Windows-ARM64 build of the locked version.
  `onnxruntime < 1.20` and `av < 14.3` are pinned only to keep macOS 12 and
  13 installable (newer releases need macOS 13/14); scipy (via speechbrain)
  sets the macOS 12 floor. onnxruntime has no Python 3.13 wheel below 1.20, so
  the lock splits by Python version (found by the first real Windows install,
  which picked 3.13). `scripts/check_wheels.py` verifies every pin on every
  target; run it after each re-lock. `tests/unit/test_setup.py` guards the pins.
- **torchaudio stopped at 2.11.** The listener is capped at torch < 2.12 until
  speechbrain no longer needs torchaudio or an alternative is chosen.

## Later (after V1): parked on purpose

Decided, not scheduled. None of this starts while a V1 gate is open.

### Small follow-ups

- [ ] `meet replay` accepts phone recordings (`.m4a`, `.mp3`, any sample rate)
      by decoding through PyAV, already in the listener lock. Record on a phone,
      process on any computer later.
- [ ] `meet update` / setup: say "checking models" instead of "downloading
      models" when they are already cached, and skip the 3-second microphone
      test when nothing about audio changed (keep it for first install and
      `meet doctor --audio`).

### T — Microsoft Teams live bot (chosen: option C)

Why Teams at all: in an online meeting where everyone has their own device,
Teams already names every speaker. It cannot split a shared room microphone
("Conference Room 3" = four people). That is exactly what Meet does, so the
bot's job is hybrid meetings.

```
  Teams meeting ──► Meet Bot (joins as participant, shown as "recording")
                        │  unmixed audio per dominant speaker + who it is
                        ▼
                    Meet core ── named streams: labelled by Teams identity
                        │     └─ shared room streams: voice identification
                        ▼
                    live notes + "who said this?" in chat, minutes at the end
```

Blockers only the owner can clear:

- [ ] Microsoft 365 tenant with global-admin approval for the bot's
      application permissions (join calls, access call media).
- [ ] Azure subscription. Application-hosted media bots must run on Windows
      in Azure (C#/.NET media SDK). Expect $70-150/month for a VM that can
      transcribe live.
- [ ] A domain for the bot's TLS certificate.
- [ ] Recording and consent policy: the bot must announce recording to all
      participants; check the stricter jurisdictions where meetings happen.

Phases:

- [ ] T1. Meet core accepts named streams alongside shared ones: named
      streams are pinned to their person, shared streams go through voice ID.
      Plus a simulator that replays a fake Teams meeting. Buildable and
      testable locally. (1-2 weeks)
- [ ] T2. C# bot: join by invite or schedule, receive unmixed audio, forward
      it to Meet core. Deployed and tested in the owner's tenant. (3-5 weeks)
- [ ] T3. Chat surface: live notes, identity questions answered in chat
      ("1 Marcus / 2 James"), minutes posted when the meeting ends. (2-3 weeks)
- [ ] T4. Packaging: Teams app manifest, admin install guide, optional Teams
      Store listing (publisher verification). (2+ weeks)

Cheaper steps considered, kept as fallbacks: import a Teams recording and
`.vtt` transcript afterwards (`meet import`, ~1-2 weeks, no admin), or pull
them automatically through Microsoft Graph (~3-5 weeks, admin consent).

### Phone

- [ ] Browser companion: `meet serve` on a PC left on at home or the office;
      the phone opens a secure link (e.g. over Tailscale, since phone browsers
      only allow the microphone on HTTPS), records, and shows the live
      transcript and questions. Same models, audio stays on the owner's
      machines.
- [ ] Rejected for now: fully in-browser Meet (smaller, less accurate models,
      a JavaScript rewrite) and a rented cloud server (monthly cost, meeting
      audio leaves the owner's devices).
