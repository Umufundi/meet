"""Paths, policy thresholds, and the one place a number is allowed to be a guess.

Every threshold here is a starting value with its provenance written next to it.
They are expected to move once real meetings have been run; `THRESHOLD_VERSION`
is stamped onto every decision so a later retune can tell which policy produced
which label instead of silently reinterpreting old evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# Bump whenever any number below changes. Stored on every identity decision and
# every learned voice sample.
THRESHOLD_VERSION = "2026-09-23.1"


def home() -> Path:
    """Root for durable state. Override with MEET_HOME for a scratch profile."""
    root = Path(os.environ.get("MEET_HOME") or (Path.home() / ".meet"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return home() / "voices.db"


def meetings_dir() -> Path:
    path = home() / "meetings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Policy:
    """Two different units live here and must never share a threshold.

    Cosine similarity between L2-normalized ECAPA embeddings is NOT a
    probability. Real same-speaker pairs land around 0.55-0.85 depending on
    microphone, distance, and health; cross-speaker pairs reach 0.45 on a bad
    day. A threshold of 0.93 on that scale is not "very confident", it is
    "effectively never", and a policy that mixes the two silently stops
    auto-labelling anything. Jev returns a real probability and gets its own
    numbers. Fields are prefixed by the scale they belong to.
    """

    # ── Voice evidence: cosine on L2-normalized embeddings ───────────────
    # Below this, no stored person is proposed at all.
    # silverstein/minutes ships 0.65 for CAM++; ECAPA behaves comparably.
    match_min_similarity: float = 0.65
    # Label it, but softly, and look again at reconciliation.
    provisional_similarity: float = 0.72
    # Label it with no human in the loop. Deliberately short of the ceiling a
    # real voice reaches, because demanding near-identity means asking forever.
    auto_similarity: float = 0.82
    # Confidence alone never decides. Sarah .84 against James .82 is a coin flip
    # wearing a high number, so a thin lead always goes to the human.
    min_margin: float = 0.08
    # Joining a live cluster is a looser question than naming a person: the cost
    # of an over-split is one extra question, of an over-merge a wrong name.
    cluster_join_similarity: float = 0.55

    # ── Jev arbitration: genuine probabilities in [0, 1] ─────────────────
    jev_auto_confidence: float = 0.93
    jev_provisional_confidence: float = 0.75
    jev_min_margin: float = 0.15

    # ── Quality gates for learning ───────────────────────────────────────
    # A segment must clear all of these before it can become a durable voice
    # sample. Values follow the solo-enrollment gates in minutes/voice.rs.
    learn_min_speech_s: float = 2.0
    enroll_min_speech_s: float = 5.0
    min_snr_db: float = 8.0
    max_clipping: float = 0.05
    # ~-48 dBFS. Below this the microphone is effectively muted, and a noise
    # floor normalized up to "signal" is the classic way to poison a profile.
    min_rms: float = 0.004
    # Lowest similarity between a new sample and the profile it claims to
    # extend. A sample that disagrees with itself is how profiles rot.
    min_window_consistency: float = 0.70

    # ── Whisper hygiene ──────────────────────────────────────────────────
    max_no_speech: float = 0.6

    # ── Profile shape ────────────────────────────────────────────────────
    # A single centroid cannot hold one voice across a headset, a far room
    # microphone, and a cold. Keep the best N samples and score against the
    # strongest of them as well as the centroid.
    max_samples_per_person: int = 64
    top_k_samples_scored: int = 3

    def first_meeting(self) -> "Policy":
        """Looser gates when nobody is enrolled yet and asking constantly is
        worse than a provisional label the human can correct in one keystroke."""
        return replace(self, auto_similarity=0.78, provisional_similarity=0.68, min_margin=0.06)


POLICY = Policy()

# Jev. The key is read from the environment, falling back to the OS credential
# store (Credential Manager, Keychain, Secret Service) so it never sits in a dotfile.
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
JEV_KEYCHAIN_SERVICE = "TypeSafe API Key"
JEV_TIMEOUT_S = 8.0
