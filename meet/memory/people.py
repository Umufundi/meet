"""Voice memory: what a person sounds like, and whether this vector is them.

One centroid per person is not enough. The same voice through a headset, across
a conference table, and through a cold lands in different places in embedding
space, and averaging them produces a centroid that matches none of the three
well. So a person is stored as a centroid *plus* the individual samples that
built it, and a probe is scored against both:

    score = max(similarity to centroid, mean of the 3 closest live samples)

The centroid carries the person's typical voice; the top-3 mean carries the
specific conditions they have been heard in before, without letting a single
lucky sample decide a match on its own.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from ..config import POLICY, THRESHOLD_VERSION, Policy
from .db import now, pack, unpack


@dataclass(frozen=True, slots=True)
class Candidate:
    slug: str
    name: str
    similarity: float
    centroid_similarity: float
    sample_count: int
    confirmed_samples: int


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Ranked people for one probe, plus the margin that decides whether to ask."""

    candidates: list[Candidate]
    model_id: str

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def margin(self) -> float:
        """Lead of the top candidate over the runner-up.

        A thin margin is the failure the confidence number hides: two people who
        both score .93 are not a .93 answer, they are a question.
        """
        if len(self.candidates) < 2:
            return self.candidates[0].similarity if self.candidates else 0.0
        return self.candidates[0].similarity - self.candidates[1].similarity


def _vector(row: sqlite3.Row) -> np.ndarray | None:
    """Decode a stored embedding, or None if the row is damaged.

    One corrupt blob must cost one sample, not the meeting: matching runs on
    every utterance, so an exception here would take identity down for the
    rest of the session.
    """
    try:
        v = unpack(row["embedding"], row["dim"])
    except (ValueError, TypeError):
        return None
    return v if np.all(np.isfinite(v)) and float(np.linalg.norm(v)) > 1e-12 else None


def _live_samples(conn: sqlite3.Connection, model_id: str) -> dict[str, list[tuple[np.ndarray, str]]]:
    rows = conn.execute(
        "SELECT person_slug, embedding, dim, trust FROM voice_sample "
        "WHERE model_id=? AND revoked_at IS NULL",
        (model_id,),
    ).fetchall()
    out: dict[str, list[tuple[np.ndarray, str]]] = {}
    for r in rows:
        v = _vector(r)
        if v is not None:
            out.setdefault(r["person_slug"], []).append((v, r["trust"]))
    return out


def rebuild_profile(conn: sqlite3.Connection, slug: str, model_id: str) -> int:
    """Recompute one person's centroid from their live samples.

    Cheap and idempotent, so it runs after every learned sample and after every
    revoke. The centroid is derived state; the samples are the truth.
    """
    rows = conn.execute(
        "SELECT embedding, dim FROM voice_sample "
        "WHERE person_slug=? AND model_id=? AND revoked_at IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (slug, model_id, POLICY.max_samples_per_person),
    ).fetchall()
    vectors = [v for v in (_vector(r) for r in rows) if v is not None]
    if not vectors or len({v.size for v in vectors}) != 1:
        conn.execute("DELETE FROM voice_profile WHERE person_slug=? AND model_id=?", (slug, model_id))
        return 0
    stack = np.stack(vectors)
    centroid = stack.mean(axis=0)
    conn.execute(
        "INSERT INTO voice_profile(person_slug, model_id, embedding, dim, sample_count, updated_at) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(person_slug, model_id) DO UPDATE SET "
        "embedding=excluded.embedding, dim=excluded.dim, "
        "sample_count=excluded.sample_count, updated_at=excluded.updated_at",
        (slug, model_id, pack(centroid), int(stack.shape[1]), len(rows), now()),
    )
    return len(vectors)


def match(
    conn: sqlite3.Connection,
    probe: np.ndarray,
    model_id: str,
    policy: Policy = POLICY,
    restrict_to: set[str] | None = None,
) -> MatchResult:
    """Rank known people against one voice vector.

    `restrict_to` is the attendee list when the meeting was started with one. It
    narrows the field without ever inventing a match: a caller who was not on the
    list simply cannot be proposed, which is the correct failure (a question)
    rather than the dangerous one (a confident wrong name).
    """
    probe = np.asarray(probe, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(probe))
    if norm <= 1e-12:
        return MatchResult([], model_id)
    probe = probe / norm

    names = {r["slug"]: r["display_name"] for r in conn.execute("SELECT slug, display_name FROM person")}
    centroids = {
        r["person_slug"]: v
        for r in conn.execute(
            "SELECT person_slug, embedding, dim FROM voice_profile WHERE model_id=?", (model_id,)
        )
        if (v := _vector(r)) is not None
    }
    samples = _live_samples(conn, model_id)

    candidates: list[Candidate] = []
    for slug, vectors in samples.items():
        if restrict_to is not None and slug not in restrict_to:
            continue
        vectors = [(v, trust) for v, trust in vectors if v.size == probe.size]
        if not vectors:
            continue
        sims = np.array([float(np.dot(probe, v / (np.linalg.norm(v) or 1.0))) for v, _ in vectors])
        top = np.sort(sims)[::-1][: policy.top_k_samples_scored]
        centroid = centroids.get(slug)
        centroid_sim = 0.0
        if centroid is not None and centroid.size == probe.size:
            cn = float(np.linalg.norm(centroid))
            if cn > 1e-12:
                centroid_sim = float(np.dot(probe, centroid / cn))
        score = max(centroid_sim, float(top.mean()))
        candidates.append(
            Candidate(
                slug=slug,
                name=names.get(slug, slug),
                similarity=score,
                centroid_similarity=centroid_sim,
                sample_count=len(vectors),
                confirmed_samples=sum(1 for _, trust in vectors if trust == "human"),
            )
        )

    candidates.sort(key=lambda c: c.similarity, reverse=True)
    # Anything below the floor is noise dressed as a candidate; drop it so the
    # margin is computed against a real rival rather than a stranger.
    candidates = [c for c in candidates if c.similarity >= policy.match_min_similarity][:5]
    return MatchResult(candidates, model_id)


def consistency_with_profile(
    conn: sqlite3.Connection, slug: str, model_id: str, probe: np.ndarray
) -> float | None:
    """Similarity between this probe and the centroid of the person's samples.

    Returns None for a person with no samples yet. A low value on an `auto`
    sample means the profile is about to be stretched by something that may not
    be the same voice, which is exactly how profiles rot.

    The centroid, not the minimum over samples: the minimum only falls as a
    profile grows, so a person the human confirmed often became the person the
    tool could no longer learn about.
    """
    rows = conn.execute(
        "SELECT embedding, dim FROM voice_sample WHERE person_slug=? AND model_id=? AND revoked_at IS NULL",
        (slug, model_id),
    ).fetchall()
    if not rows:
        return None
    probe = np.asarray(probe, dtype=np.float32).ravel()
    probe = probe / (float(np.linalg.norm(probe)) or 1.0)
    vectors = [v for v in (_vector(r) for r in rows) if v is not None and v.size == probe.size]
    if not vectors:
        return None
    centroid = np.stack([v / (float(np.linalg.norm(v)) or 1.0) for v in vectors]).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return float(np.dot(probe, centroid / norm)) if norm > 1e-12 else None


@dataclass(frozen=True, slots=True)
class LearnOutcome:
    stored: bool
    reason: str
    consistency: float | None = None


def learn(
    conn: sqlite3.Connection,
    slug: str,
    embedding: np.ndarray,
    model_id: str,
    *,
    trust: str,
    speech_s: float,
    snr_db: float,
    clipping: float,
    rms: float,
    meeting_id: str | None = None,
    cluster_key: str | None = None,
    utterance_id: int | None = None,
    similarity: float | None = None,
    margin: float | None = None,
    policy: Policy = POLICY,
) -> LearnOutcome:
    """Add one voice sample, or refuse and say why.

    Refusing is the common case and the right default. Learning from every
    utterance is how a profile ends up describing the room rather than a person:
    laughter, two-word interjections, and cross-talk all produce embeddings that
    are confidently wrong. The gates below are the whole defence.

    A human-confirmed sample still has to be audible (a person can only tell you
    *who* it was, not that the audio was clean), but it is allowed to disagree
    with the existing profile: that disagreement is usually the profile being
    wrong, and the human is the authority.
    """
    if speech_s < policy.learn_min_speech_s:
        return LearnOutcome(False, f"only {speech_s:.1f}s of speech (need {policy.learn_min_speech_s:.1f}s)")
    if rms < policy.min_rms:
        return LearnOutcome(False, "input level is at the noise floor")
    if snr_db < policy.min_snr_db:
        return LearnOutcome(False, f"signal-to-noise {snr_db:.1f} dB (need {policy.min_snr_db:.0f} dB)")
    if clipping > policy.max_clipping:
        return LearnOutcome(False, f"{clipping * 100:.1f}% of the segment is clipped")

    consistency = consistency_with_profile(conn, slug, model_id, embedding)
    if trust == "auto" and consistency is not None and consistency < policy.min_window_consistency:
        return LearnOutcome(False, f"disagrees with the stored profile ({consistency:.2f})", consistency)

    conn.execute(
        "INSERT INTO voice_sample(person_slug, embedding, dim, model_id, trust, meeting_id, "
        "cluster_key, utterance_id, speech_s, snr_db, clipping, consistency, similarity, margin, "
        "policy, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            slug,
            pack(embedding),
            int(np.asarray(embedding).size),
            model_id,
            trust,
            meeting_id,
            cluster_key,
            utterance_id,
            speech_s,
            snr_db,
            clipping,
            consistency,
            similarity,
            margin,
            THRESHOLD_VERSION,
            now(),
        ),
    )
    rebuild_profile(conn, slug, model_id)
    return LearnOutcome(True, "stored", consistency)


def record_decision(
    conn: sqlite3.Connection,
    meeting_id: str,
    cluster_key: str,
    *,
    source: str,
    chosen_slug: str | None,
    candidates: list[Candidate],
    utterance_id: int | None = None,
    top_p: float | None = None,
    confidence: float | None = None,
    margin: float | None = None,
) -> None:
    """Append the reasoning behind a label. Never updates an earlier row."""
    payload = json.dumps(
        [
            {
                "slug": c.slug,
                "name": c.name,
                "similarity": round(c.similarity, 4),
                "centroid": round(c.centroid_similarity, 4),
                "samples": c.sample_count,
                "confirmed": c.confirmed_samples,
            }
            for c in candidates
        ]
    )
    conn.execute(
        "INSERT INTO identity_decision(meeting_id, cluster_key, utterance_id, source, chosen_slug, "
        "candidates, top_p, margin, confidence, policy, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            meeting_id,
            cluster_key,
            utterance_id,
            source,
            chosen_slug,
            payload,
            top_p,
            margin,
            confidence,
            THRESHOLD_VERSION,
            now(),
        ),
    )
