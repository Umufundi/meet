"""The product's central promise, pinned down.

    Human says cluster 4 = Sarah
      -> ALL old lines from cluster 4 become Sarah
      -> profile learned, decision audit written
      -> future lines keep the identity
      -> next meeting, Sarah is recognised without asking
"""

import pytest

from meet.config import POLICY
from meet.identity import jev as jev_client
from meet.session.meeting import UNKNOWN_LABEL, Meeting
from tests.conftest import MODEL


def meeting(conn, **kw):
    kw.setdefault("title", "Staff")
    kw.setdefault("use_jev", False)
    kw.setdefault("policy", POLICY.first_meeting())
    return Meeting(conn, **kw)


def speakers(m):
    return [speaker for speaker, _, _ in m.transcript()]


def test_unknown_voices_render_as_question_mark_and_ask_once_per_cluster(conn, script):
    m = meeting(conn)
    for who in ("a", "a", "b", "a", "a", "a"):
        m.on_utterance(script.say(who))
    assert speakers(m) == [UNKNOWN_LABEL] * 6
    # Six unknown lines from two voices: two questions, not six.
    assert len(m.questions) == 2
    assert m.stats.questions_asked == 2
    assert {q.cluster_key for q in m.questions.values()} == {"1", "2"}


def test_one_answer_relabels_the_whole_cluster_history(conn, script):
    m = meeting(conn)
    for who in ("a", "b", "a", "a"):
        m.on_utterance(script.say(who))
    question = next(q for q in m.questions.values() if q.cluster_key == "1")
    assert m.answer(question.id, "Marcus") == "Marcus"
    assert speakers(m) == ["Marcus", UNKNOWN_LABEL, "Marcus", "Marcus"]
    # The answer closed the question for that voice; the other voice is still open.
    assert [q.cluster_key for q in m.questions.values()] == ["2"]
    # Future lines from the same voice arrive already named, without asking.
    line = m.on_utterance(script.say("a"))
    assert m.label_for(line.cluster_key) == "Marcus"
    assert m.stats.questions_asked == 2

    row = conn.execute("SELECT person_slug, decided_by FROM cluster WHERE meeting_id=? AND key='1'",
                       (m.id,)).fetchone()
    assert (row["person_slug"], row["decided_by"]) == ("marcus", "human")
    audit = [r["source"] for r in conn.execute(
        "SELECT source FROM identity_decision WHERE meeting_id=? AND cluster_key='1' ORDER BY id", (m.id,))]
    assert audit[0] == "auto" and "human" in audit


def test_answer_teaches_one_human_sample_and_gated_auto_samples(conn, script):
    m = meeting(conn)
    for _ in range(3):
        m.on_utterance(script.say("a"))
    q = next(iter(m.questions.values()))
    m.answer(q.id, "Marcus")
    trusts = [r["trust"] for r in conn.execute(
        "SELECT trust FROM voice_sample WHERE person_slug='marcus' AND revoked_at IS NULL")]
    # Exactly one line is direct human evidence: the one the question showed.
    assert trusts.count("human") == 1
    assert trusts.count("auto") == 2
    anchored = conn.execute(
        "SELECT utterance_id FROM voice_sample WHERE trust='human'").fetchone()["utterance_id"]
    assert anchored == q.utterance_id


def test_learned_voice_is_recognised_next_meeting_without_asking(conn, script):
    first = meeting(conn)
    for _ in range(4):
        first.on_utterance(script.say("a"))
    first.answer(next(iter(first.questions)), "Marcus")

    second = meeting(conn, policy=POLICY)
    for _ in range(3):
        second.on_utterance(script.say("a"))
    assert speakers(second) == ["Marcus"] * 3
    assert second.questions == {}
    assert second.stats.auto_labelled == 3


def test_wrong_retargets_the_last_speaker(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.on_utterance(script.say("b"))
    assert m.correct_last("Sarah") == "Sarah"
    assert speakers(m) == [UNKNOWN_LABEL, "Sarah"]
    assert m.stats.corrections == 1


def test_undo_reverts_label_and_revokes_what_was_learned(conn, script):
    m = meeting(conn)
    for _ in range(3):
        m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Sarah")
    assert conn.execute("SELECT COUNT(*) FROM voice_sample WHERE revoked_at IS NULL").fetchone()[0] == 3
    assert m.undo() == UNKNOWN_LABEL
    assert speakers(m) == [UNKNOWN_LABEL] * 3
    assert conn.execute("SELECT COUNT(*) FROM voice_sample WHERE revoked_at IS NULL").fetchone()[0] == 0
    # Revoked, not deleted: the audit trail survives.
    assert conn.execute("SELECT COUNT(*) FROM voice_sample").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM voice_profile").fetchone()[0] == 0
    with pytest.raises(IndexError):
        m.undo()


def test_undo_restores_the_previous_name(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.name_cluster("1", "Marcus")
    m.name_cluster("1", "James")
    assert m.undo() == "Marcus"
    assert speakers(m) == ["Marcus"]


def test_merge_moves_lines_and_inherits_the_name(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.on_utterance(script.say("b"))
    m.on_utterance(script.say("a"))
    m.name_cluster("1", "Marcus")
    assert m.merge("2", "1")
    assert speakers(m) == ["Marcus"] * 3
    assert {ln.cluster_key for ln in m.lines} == {"1"}
    stored = {r[0] for r in conn.execute("SELECT cluster_key FROM utterance WHERE meeting_id=?", (m.id,))}
    assert stored == {"1"}
    merged = conn.execute("SELECT merged_into FROM cluster WHERE meeting_id=? AND key='2'",
                          (m.id,)).fetchone()
    assert merged["merged_into"] == "1"
    assert not m.merge("1", "1")


def test_merge_into_an_unnamed_cluster_carries_the_name_to_the_database(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.on_utterance(script.say("b"))
    m.name_cluster("2", "Sarah")
    assert m.merge("2", "1")
    assert speakers(m) == ["Sarah", "Sarah"]
    row = conn.execute("SELECT person_slug FROM cluster WHERE meeting_id=? AND key='1'", (m.id,)).fetchone()
    assert row["person_slug"] == "sarah"


def test_names_are_case_insensitive(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.on_utterance(script.say("b"))
    m.name_cluster("1", "Marcus")
    m.name_cluster("2", "marcus")
    assert conn.execute("SELECT COUNT(*) FROM person").fetchone()[0] == 1
    assert speakers(m) == ["Marcus", "Marcus"]


def test_noise_is_not_a_line(conn, script, voices):
    m = meeting(conn)
    assert m.on_utterance(script.say("a", text="   ")) is None
    assert m.on_utterance(script.say("a", no_speech=0.95)) is None
    assert m.on_utterance(script.say("a", embedding=[])) is None
    assert m.lines == [] and m.questions == {}


def test_voice_from_another_model_is_a_question_not_a_guess(conn, script, voices):
    first = meeting(conn)
    first.on_utterance(script.say("a"))
    first.name_cluster("1", "Marcus")
    script.model_id = "different-model@2"
    second = meeting(conn, policy=POLICY)
    second.on_utterance(script.say("a"))
    assert speakers(second) == [UNKNOWN_LABEL]
    assert len(second.questions) == 1


def test_attendee_roster_prevents_proposing_absent_people(conn, script):
    first = meeting(conn)
    for _ in range(3):
        first.on_utterance(script.say("a"))
    first.name_cluster("1", "Marcus")
    second = meeting(conn, policy=POLICY, attendees=["Sarah", "James"])
    second.on_utterance(script.say("a"))
    assert speakers(second) == [UNKNOWN_LABEL]


# ── Jev is optional and can never stop the meeting ─────────────────────


def _tie(conn, script, voices):
    """Two enrolled people whose voices the probe sits exactly between."""
    import numpy as np

    from meet.memory import db, people

    for name, who in (("Marcus", "a"), ("Sarah", "b")):
        slug = db.upsert_person(conn, name)
        people.learn(conn, slug, voices.base(who), MODEL, trust="human", speech_s=3, snr_db=20,
                     clipping=0, rms=0.05)
    mid = voices.base("a") + voices.base("b")
    return list(mid / np.linalg.norm(mid))


def test_jev_failure_falls_back_to_asking(conn, script, voices, monkeypatch):
    between = _tie(conn, script, voices)

    def down(**_):
        raise jev_client.JevUnavailable("offline")

    monkeypatch.setattr(jev_client, "ask", down)
    m = Meeting(conn, title="t", use_jev=True, policy=POLICY.first_meeting().__class__(
        match_min_similarity=0.5, provisional_similarity=0.6, auto_similarity=0.65))
    line = m.on_utterance(script.say("x", embedding=between))
    assert line is not None and m.label_for(line.cluster_key) == UNKNOWN_LABEL
    assert len(m.questions) == 1
    assert m.stats.jev_failures == 1


def test_jev_can_break_a_tie(conn, script, voices, monkeypatch):
    between = _tie(conn, script, voices)
    monkeypatch.setattr(jev_client, "ask", lambda **_: jev_client.JevAnswer(
        slug="sarah", confidence=0.97, probabilities={"sarah": 0.97, "marcus": 0.02, "UNKNOWN": 0.01},
        latency_ms=5))
    m = Meeting(conn, title="t", use_jev=True, policy=POLICY.first_meeting().__class__(
        match_min_similarity=0.5, provisional_similarity=0.6, auto_similarity=0.65))
    line = m.on_utterance(script.say("x", embedding=between))
    assert m.label_for(line.cluster_key) == "Sarah"
    assert m.questions == {}
    source = conn.execute("SELECT source FROM identity_decision WHERE chosen_slug='sarah'").fetchone()[0]
    assert source == "jev"
