"""Durable state: people, voice samples, meetings, and every identity decision.

Two invariants make the product's central promise cheap to keep.

1. An utterance stores a *cluster*, never a person. Naming a cluster therefore
   relabels its whole history in one write, with no possibility of a partial
   update leaving half a meeting attributed to the wrong human. `:merge` is a
   cluster remap for the same reason.

2. Voice samples are append-only. SQLite triggers refuse DELETE and refuse any
   UPDATE other than setting `revoked_at` exactly once. Biometric evidence that
   can be quietly rewritten is not evidence, and a corrupted profile is close to
   impossible to debug after the fact. Forgetting a person is a revoke plus a
   centroid rebuild, which leaves an audit trail; `delete_person` exists for the
   privacy path and is the only statement allowed to bypass the trigger.

Embeddings are stored as raw float32 little-endian blobs, L2-normalized at write
time, always alongside the `model_id` that produced them. Vectors from different
models are never compared.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import db_path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS person (
    slug         TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_sample (
    id           INTEGER PRIMARY KEY,
    person_slug  TEXT NOT NULL REFERENCES person(slug),
    embedding    BLOB NOT NULL,
    dim          INTEGER NOT NULL,
    model_id     TEXT NOT NULL,
    -- 'human'  : a person said this is who it was
    -- 'auto'   : accepted above auto_similarity with margin, never human-confirmed
    trust        TEXT NOT NULL CHECK (trust IN ('human','auto')),
    meeting_id   TEXT,
    cluster_key  TEXT,
    utterance_id INTEGER,
    speech_s     REAL NOT NULL,
    snr_db       REAL NOT NULL,
    clipping     REAL NOT NULL,
    consistency  REAL,
    similarity   REAL,
    margin       REAL,
    policy       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    revoked_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_sample_person_model ON voice_sample(person_slug, model_id);

CREATE TRIGGER IF NOT EXISTS voice_sample_no_delete
BEFORE DELETE ON voice_sample
WHEN (SELECT value FROM meta WHERE key = 'erasing') IS NULL
BEGIN SELECT RAISE(ABORT, 'voice samples are append-only; revoke instead'); END;

CREATE TRIGGER IF NOT EXISTS voice_sample_no_edit
BEFORE UPDATE OF person_slug, embedding, dim, model_id, trust, meeting_id,
    cluster_key, utterance_id, speech_s, snr_db, clipping, consistency,
    similarity, margin, policy, created_at ON voice_sample
BEGIN SELECT RAISE(ABORT, 'voice samples are immutable'); END;

CREATE TRIGGER IF NOT EXISTS voice_sample_revoke_once
BEFORE UPDATE OF revoked_at ON voice_sample
WHEN NEW.revoked_at IS NULL OR OLD.revoked_at IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'revocation is one-way'); END;

-- Derived: the centroid of a person's live samples under one embedding model.
-- Always rebuildable from voice_sample, so this table may be rewritten freely.
CREATE TABLE IF NOT EXISTS voice_profile (
    person_slug  TEXT NOT NULL REFERENCES person(slug),
    model_id     TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    dim          INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (person_slug, model_id)
);

CREATE TABLE IF NOT EXISTS meeting (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    audio_path  TEXT,
    sample_rate INTEGER
);

-- One row per voice the live pass thinks it heard. `person_slug` is the single
-- place a human correction lands.
CREATE TABLE IF NOT EXISTS cluster (
    meeting_id  TEXT NOT NULL REFERENCES meeting(id),
    key         TEXT NOT NULL,
    person_slug TEXT REFERENCES person(slug),
    decided_by  TEXT,          -- 'auto' | 'jev' | 'human' | NULL while unknown
    decided_at  TEXT,
    merged_into TEXT,          -- set when :merge folds this cluster into another
    PRIMARY KEY (meeting_id, key)
);

CREATE TABLE IF NOT EXISTS utterance (
    id          INTEGER PRIMARY KEY,
    meeting_id  TEXT NOT NULL REFERENCES meeting(id),
    cluster_key TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB,
    dim         INTEGER,
    model_id    TEXT,
    speech_s    REAL,
    snr_db      REAL,
    clipping    REAL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_utterance_meeting ON utterance(meeting_id, seq);

-- Why a cluster carries the name it carries. Never overwritten: a correction
-- appends a new row, so the sequence auto -> jev -> human stays readable.
CREATE TABLE IF NOT EXISTS identity_decision (
    id           INTEGER PRIMARY KEY,
    meeting_id   TEXT NOT NULL,
    cluster_key  TEXT NOT NULL,
    utterance_id INTEGER,
    source       TEXT NOT NULL CHECK (source IN ('auto','jev','human','merge')),
    chosen_slug  TEXT,
    candidates   TEXT NOT NULL,   -- JSON: [{slug,name,similarity,probability}]
    top_p        REAL,
    margin       REAL,
    confidence   REAL,
    policy       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decision_cluster ON identity_decision(meeting_id, cluster_key);

"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pack(vector: np.ndarray) -> bytes:
    """float32 little-endian, L2-normalized. Normalizing at the boundary means
    cosine similarity downstream is a dot product and cannot drift."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("embedding is empty or non-finite")
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        raise ValueError("embedding has zero length")
    return (v / norm).astype("<f4").tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    if len(blob) != dim * 4:
        raise ValueError(f"embedding blob is {len(blob)} bytes, expected {dim * 4}")
    return np.frombuffer(blob, dtype="<f4", count=dim).astype(np.float32)


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    first = not target.exists()
    conn = sqlite3.connect(target, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if first:
        # Voiceprints are biometric data. Nobody else on the machine reads them.
        target.chmod(0o600)
    return conn


@dataclass(frozen=True, slots=True)
class Person:
    slug: str
    display_name: str
    sample_count: int = 0


def slugify(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in name.strip())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "unnamed"


def upsert_person(conn: sqlite3.Connection, name: str) -> str:
    slug = slugify(name)
    conn.execute(
        "INSERT INTO person(slug, display_name, created_at) VALUES (?,?,?) "
        "ON CONFLICT(slug) DO UPDATE SET display_name=excluded.display_name",
        (slug, name.strip(), now()),
    )
    return slug


def list_people(conn: sqlite3.Connection, model_id: str | None = None) -> list[Person]:
    rows = conn.execute(
        """
        SELECT p.slug, p.display_name,
               (SELECT COUNT(*) FROM voice_sample s
                 WHERE s.person_slug = p.slug AND s.revoked_at IS NULL
                   AND (? IS NULL OR s.model_id = ?)) AS n
          FROM person p ORDER BY p.display_name
        """,
        (model_id, model_id),
    ).fetchall()
    return [Person(r["slug"], r["display_name"], r["n"]) for r in rows]


def delete_person(conn: sqlite3.Connection, slug: str) -> int:
    """The privacy path: erase one person's biometric data for real.

    This is the only caller allowed past the append-only trigger, and it says so
    out loud by flipping a flag the trigger reads. Transcript text is left alone;
    only the voiceprints go.
    """
    with conn:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('erasing','1')")
        try:
            n = conn.execute("DELETE FROM voice_sample WHERE person_slug=?", (slug,)).rowcount
            conn.execute("DELETE FROM voice_profile WHERE person_slug=?", (slug,))
            conn.execute("UPDATE cluster SET person_slug=NULL WHERE person_slug=?", (slug,))
            conn.execute("DELETE FROM person WHERE slug=?", (slug,))
        finally:
            conn.execute("DELETE FROM meta WHERE key='erasing'")
    return n

