"""Shared fixtures.

Every test gets its own MEET_HOME, no Jev key, and no OS credential store, so
the suite never touches ~/.meet, the network, or a real keychain. Voices are
synthetic: each speaker is a random unit vector, and an utterance is that vector
plus noise, tuned so same-speaker cosine sits near 0.9 and different speakers
near 0. That is cleaner than real ECAPA, deliberately: these tests pin the
*logic* (who gets asked, what gets relabelled, what gets learned), and the
benchmark corpus, not the unit suite, is where real-voice thresholds are tuned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from meet import events
from meet.identity import credentials, jev
from meet.memory import db

# The real OS-store lookup, kept before the autouse fixture stubs it out.
REAL_LOOKUP = credentials.lookup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "listener"))

DIM = 192
MODEL = "ecapa-voxceleb@sb1"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MEET_HOME", str(tmp_path / "home"))
    for var in ("TYPESAFE_API_KEY", "JEV_API_KEY", "MEET_LISTENER_PYTHON", "HF_HUB_OFFLINE",
                "MEET_MODELS_OFFLINE", "HF_HOME", "MEET_MODEL_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("meet.identity.credentials.lookup", lambda *a, **k: None)
    jev._stored_key.cache_clear()
    yield
    jev._stored_key.cache_clear()


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "voices.db")
    yield c
    c.close()


class Voices:
    """Deterministic synthetic speakers."""

    def __init__(self, seed: int = 7, noise: float = 0.035) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.bases: dict[str, np.ndarray] = {}

    def base(self, who: str) -> np.ndarray:
        if who not in self.bases:
            v = self.rng.standard_normal(DIM).astype(np.float32)
            self.bases[who] = v / np.linalg.norm(v)
        return self.bases[who]

    def sample(self, who: str, noise: float | None = None) -> np.ndarray:
        v = self.base(who) + self.rng.standard_normal(DIM).astype(np.float32) * (
            self.noise if noise is None else noise
        )
        return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def voices() -> Voices:
    return Voices()


GOOD = events.Quality(speech_s=3.0, snr_db=20.0, clipping=0.0, rms=0.05)


class Script:
    """Builds a meeting's worth of utterance events with increasing times."""

    def __init__(self, voices: Voices, model_id: str = MODEL) -> None:
        self.voices = voices
        self.model_id = model_id
        self.seq = 0
        self.clock = 0

    def say(self, who: str, text: str | None = None, *, quality: events.Quality = GOOD,
            no_speech: float = 0.0, embedding: list[float] | None = None) -> events.Utterance:
        self.seq += 1
        start = self.clock
        self.clock += 4000
        vector = self.voices.sample(who) if embedding is None else embedding
        return events.Utterance(
            seq=self.seq,
            start_ms=start,
            end_ms=start + 3500,
            text=text or f"{who} says something number {self.seq}",
            no_speech=no_speech,
            embedding=[float(x) for x in vector],
            dim=len(vector),
            model_id=self.model_id,
            quality=quality,
        )


@pytest.fixture
def script(voices) -> Script:
    return Script(voices)
