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
    # Calibrated 2026-09-30 on a real 50-minute meeting through one laptop
    # microphone across a table (6 speakers, 426 utterances, human-labelled).
    # Scored as `people.match` scores (max of centroid, top-3 sample mean):
    #   same person      median 0.67, 10th percentile 0.28
    #   different person median 0.14, 90th percentile 0.23
    # The first values here (0.65 / 0.72 / 0.82, borrowed from a headset-mic
    # tool) sat above the typical *same*-person score, so a voice the human
    # had already named was asked about again and again.
    # Below this, no stored person is proposed at all.
    match_min_similarity: float = 0.32
    # Label it, but softly, and look again at reconciliation.
    provisional_similarity: float = 0.40
    # Label it with no human in the loop. Two-thirds of real same-person
    # utterances clear it; the 90th percentile of other people is 0.23.
    auto_similarity: float = 0.50
    # Confidence alone never decides. A thin lead always goes to Jev, then the
    # human.
    min_margin: float = 0.10
    # Joining a live cluster is a looser question than naming a person: the cost
    # of an over-split is one extra question, of an over-merge a wrong name.
    # One utterance against a centroid: same-person median 0.58 at >= 4 s.
    cluster_join_similarity: float = 0.40
    # A voice already named in this meeting pulls harder than an anonymous
    # cluster: re-asking about a person the human just named is the failure
    # users notice first.
    named_join_similarity: float = 0.34
    # Below this much speech a voice vector is mostly noise (same-person median
    # 0.21 under 2 s). Such a line is still transcribed and labelled when the
    # evidence is clear, but it never starts a cluster and never asks.
    ask_min_speech_s: float = 2.0
    # An automatic label may teach the profile only when it is this sure. Kept
    # well above auto_similarity so a borderline label cannot feed the profile
    # that produced it.
    auto_learn_similarity: float = 0.60
    # How long a skipped voice stays quiet, in meeting time: the rest of the
    # meeting. Re-asking a voice the human could not place tends to get it
    # named from one ambiguous line; a stray Enter is reversed with /undo.
    skip_for_ms: int = 24 * 60 * 60 * 1000

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
    # Similarity between a new automatic sample and the person's centroid.
    # Below it the sample may be a different voice misfiled by the clusterer,
    # which is how profiles rot. Measured against the centroid, not the worst
    # stored sample: a minimum over samples falls as a profile grows, so the
    # more a person was confirmed, the less the tool could learn about them.
    min_window_consistency: float = 0.45

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
        return replace(self, auto_similarity=0.46, provisional_similarity=0.36, min_margin=0.08)


POLICY = Policy()

# Jev. The key is read from the environment, falling back to the OS credential
# store (Credential Manager, Keychain, Secret Service) so it never sits in a dotfile.
# TYPESAFE_BASE_URL points Meet at any server speaking the same wire format,
# e.g. a local Laya (ollaya serve) at http://127.0.0.1:11435.
JEV_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"
JEV_MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
JEV_KEYCHAIN_SERVICE = "TypeSafe API Key"
JEV_TIMEOUT_S = 8.0
