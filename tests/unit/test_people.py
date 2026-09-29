import numpy as np

from meet.config import POLICY
from meet.memory import db, people
from tests.conftest import MODEL


def learn(conn, slug, vector, *, trust="human", model=MODEL, **quality):
    q = dict(speech_s=3.0, snr_db=20.0, clipping=0.0, rms=0.05)
    q.update(quality)
    return people.learn(conn, slug, vector, model, trust=trust, **q)


def test_quality_gates_refuse_bad_audio(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    v = voices.sample("a")
    assert not learn(conn, slug, v, speech_s=1.0).stored
    assert not learn(conn, slug, v, rms=0.0001).stored
    assert not learn(conn, slug, v, snr_db=3.0).stored
    assert not learn(conn, slug, v, clipping=0.2).stored
    assert learn(conn, slug, v).stored


def test_auto_sample_must_agree_with_profile_but_human_may_not(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    assert learn(conn, slug, voices.sample("a")).stored
    stranger = voices.sample("b")
    refused = learn(conn, slug, stranger, trust="auto")
    assert not refused.stored and "disagrees" in refused.reason
    assert learn(conn, slug, stranger, trust="human").stored


def test_match_ranks_and_applies_floor(conn, voices):
    for name, who in (("Marcus", "a"), ("Sarah", "b")):
        slug = db.upsert_person(conn, name)
        for _ in range(3):
            learn(conn, slug, voices.sample(who))
    result = people.match(conn, voices.sample("a"), MODEL)
    assert result.best.slug == "marcus"
    # Sarah is a stranger to this voice and must not appear as a rival.
    assert [c.slug for c in result.candidates] == ["marcus"]
    assert result.best.similarity > POLICY.auto_similarity
    assert people.match(conn, voices.sample("c"), MODEL).candidates == []


def test_match_respects_roster(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    learn(conn, slug, voices.sample("a"))
    assert people.match(conn, voices.sample("a"), MODEL, restrict_to={"sarah"}).candidates == []


def test_embeddings_from_another_model_are_never_compared(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    learn(conn, slug, voices.sample("a"), model="other-model@1")
    assert people.match(conn, voices.sample("a"), MODEL).candidates == []
    assert people.match(conn, voices.sample("a"), "other-model@1").best.slug == "marcus"


def test_corrupt_embedding_costs_one_sample_not_the_meeting(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    learn(conn, slug, voices.sample("a"))
    conn.execute(
        "INSERT INTO voice_sample(person_slug, embedding, dim, model_id, trust, speech_s, snr_db, "
        "clipping, policy, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (slug, b"\x00\x01\x02", 192, MODEL, "human", 3.0, 20.0, 0.0, "t", "t"),
    )
    nan = np.full(192, np.nan, dtype="<f4").tobytes()
    conn.execute(
        "INSERT INTO voice_sample(person_slug, embedding, dim, model_id, trust, speech_s, snr_db, "
        "clipping, policy, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (slug, nan, 192, MODEL, "human", 3.0, 20.0, 0.0, "t", "t"),
    )
    assert people.rebuild_profile(conn, slug, MODEL) == 1
    assert people.match(conn, voices.sample("a"), MODEL).best.slug == "marcus"
    assert learn(conn, slug, voices.sample("a"), trust="auto").stored


def test_revoke_then_rebuild_drops_profile(conn, voices):
    slug = db.upsert_person(conn, "Marcus")
    learn(conn, slug, voices.sample("a"))
    conn.execute("UPDATE voice_sample SET revoked_at='now'")
    assert people.rebuild_profile(conn, slug, MODEL) == 0
    assert conn.execute("SELECT COUNT(*) FROM voice_profile").fetchone()[0] == 0
    assert people.match(conn, voices.sample("a"), MODEL).candidates == []


def test_margin_semantics():
    c = lambda s, sim: people.Candidate(s, s, sim, sim, 1, 1)  # noqa: E731
    assert people.MatchResult([], MODEL).margin == 0.0
    assert people.MatchResult([c("a", 0.9)], MODEL).margin == 0.9
    assert abs(people.MatchResult([c("a", 0.9), c("b", 0.85)], MODEL).margin - 0.05) < 1e-9
