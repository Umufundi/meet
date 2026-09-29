import os
import sqlite3

import numpy as np
import pytest

from meet.memory import db, people
from tests.conftest import MODEL


def _learn(conn, slug, vector, trust="human"):
    return people.learn(conn, slug, vector, MODEL, trust=trust, speech_s=3.0, snr_db=20.0,
                        clipping=0.0, rms=0.05)


def test_pack_normalizes_and_rejects_bad_vectors():
    v = db.unpack(db.pack(np.array([3.0, 4.0])), 2)
    assert np.allclose(v, [0.6, 0.8])
    for bad in (np.array([]), np.array([np.nan, 1.0]), np.zeros(4)):
        with pytest.raises(ValueError):
            db.pack(bad)
    with pytest.raises(ValueError):
        db.unpack(b"\x00" * 7, 2)


def test_voice_samples_are_append_only(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    assert _learn(conn, slug, voices.sample("a")).stored
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM voice_sample")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE voice_sample SET trust='auto'")
    conn.execute("UPDATE voice_sample SET revoked_at='now'")
    with pytest.raises(sqlite3.IntegrityError, match="one-way"):
        conn.execute("UPDATE voice_sample SET revoked_at='later'")


def test_delete_person_is_the_only_erase_path(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    _learn(conn, slug, voices.sample("a"))
    _learn(conn, slug, voices.sample("a"))
    assert db.delete_person(conn, slug) == 2
    assert conn.execute("SELECT COUNT(*) FROM voice_sample").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM voice_profile").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM person").fetchone()[0] == 0
    # The bypass flag must not outlive the erase.
    assert conn.execute("SELECT COUNT(*) FROM meta WHERE key='erasing'").fetchone()[0] == 0
    _learn(conn, db.upsert_person(conn, "Sarah"), voices.sample("b"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM voice_sample")


def test_duplicate_names_share_one_person(conn):
    a = db.upsert_person(conn, "Marcus Lee")
    b = db.upsert_person(conn, "marcus  lee")
    assert a == b == "marcus-lee"
    assert len(db.list_people(conn)) == 1


def test_reopening_an_existing_database_keeps_everything(tmp_path, voices):
    path = tmp_path / "v.db"
    first = db.connect(path)
    _learn(first, db.upsert_person(first, "Marcus"), voices.sample("a"))
    first.close()
    again = db.connect(path)
    assert [p.sample_count for p in db.list_people(again)] == [1]
    assert db.schema_version(again) == db.SCHEMA_VERSION


def test_unversioned_database_is_stamped(tmp_path):
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(db.SCHEMA)
    legacy.execute("INSERT INTO person VALUES ('x','X','t')")
    legacy.commit()
    legacy.close()
    conn = db.connect(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT display_name FROM person").fetchone()[0] == "X"


def test_database_from_a_newer_meet_is_refused(tmp_path):
    path = tmp_path / "new.db"
    newer = sqlite3.connect(path)
    newer.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    newer.close()
    with pytest.raises(db.DatabaseTooNew):
        db.connect(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_new_database_is_private(tmp_path):
    path = tmp_path / "p.db"
    db.connect(path).close()
    assert (path.stat().st_mode & 0o777) == 0o600
