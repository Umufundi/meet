"""The tool meets the human halfway.

Measured on a real meeting (2026-09-30): the human named the same people over
and over, was asked who said "ok", could not skip, and answering meant typing a
question number before the name. These tests pin the fixes.
"""

import numpy as np

from meet import events
from meet.config import POLICY
from meet.memory import people
from meet.session.clustering import UNPLACED, OnlineClusterer
from meet.session.meeting import UNKNOWN_LABEL, Meeting
from meet.ui import commands
from tests.conftest import MODEL

SHORT = events.Quality(speech_s=0.8, snr_db=20.0, clipping=0.0, rms=0.05)


def meeting(conn, **kw):
    kw.setdefault("title", "Staff")
    kw.setdefault("use_jev", False)
    kw.setdefault("policy", POLICY.first_meeting())
    return Meeting(conn, **kw)


def near(centroid: np.ndarray, similarity: float, seed: int = 3) -> list[float]:
    """A unit vector at exactly `similarity` to `centroid`, unrelated to anything else."""
    rng = np.random.default_rng(seed)
    c = centroid / np.linalg.norm(centroid)
    u = rng.standard_normal(c.size).astype(np.float32)
    u -= float(np.dot(u, c)) * c
    u /= np.linalg.norm(u)
    v = similarity * c + np.sqrt(1 - similarity**2) * u
    return [float(x) for x in v]


def test_a_short_line_never_asks_and_never_starts_a_voice(conn, script):
    m = meeting(conn)
    line = m.on_utterance(script.say("a", "ok", quality=SHORT))
    assert m.label_for(line.cluster_key) == UNKNOWN_LABEL
    assert line.cluster_key == UNPLACED
    assert not m.questions and m.stats.questions_asked == 0
    assert not [c for c in m.clusterer.clusters.values() if c.count]


def test_a_short_line_from_a_named_voice_keeps_the_name(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Marcus")
    line = m.on_utterance(script.say("a", "ok", quality=SHORT))
    assert m.label_for(line.cluster_key) == "Marcus"


def test_skip_closes_the_question_and_that_voice_is_not_asked_again(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    assert m.skip() is True
    for _ in range(4):
        m.on_utterance(script.say("a"))
    assert not m.questions and m.stats.questions_asked == 1
    # Still nameable later, which relabels the whole history.
    m.name_cluster("1", "Marcus")
    assert {s for s, _, _ in m.transcript()} == {"Marcus"}


def test_enter_on_an_empty_line_skips(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    result = commands.apply(m, "")
    assert "skipped" in result.message and not m.questions


def test_a_double_enter_after_a_name_does_not_skip_the_next_question(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.on_utterance(script.say("b"))
    commands.apply(m, "Marcus")
    commands.apply(m, "")
    assert len(m.questions) == 1


def test_the_bare_word_end_does_not_stop_the_meeting(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    assert not commands.apply(m, "end").finished
    assert commands.apply(m, "/end").finished


def test_a_name_alone_answers_the_question_on_screen(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    result = commands.apply(m, "Marcus")
    assert result.changed and "Marcus" in result.message
    assert [s for s, _, _ in m.transcript()] == ["Marcus"]


def test_a_name_with_no_question_waiting_is_not_invented_as_a_person(conn, script):
    m = meeting(conn)
    result = commands.apply(m, "Marcus")
    assert "no question" in result.message
    assert conn.execute("SELECT COUNT(*) FROM person WHERE slug='marcus'").fetchone()[0] == 0


def test_skip_command(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    assert "skipped" in commands.apply(m, "/skip").message
    assert "no question" in commands.apply(m, "/skip").message


def test_reusing_a_name_folds_the_voices_and_one_undo_splits_them(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Marcus")
    # The same person, heard through a different position: a separate cluster.
    far = near(m.clusterer.clusters["1"].centroid, 0.10)
    m.on_utterance(script.say("a", embedding=far))
    question = next(iter(m.questions.values()))
    assert question.cluster_key == "2"
    m.answer(question.id, "Marcus")
    assert [s for s, _, _ in m.transcript()] == ["Marcus", "Marcus"]
    assert [k for k, c in m.clusterer.clusters.items() if c.count] == ["1"]
    assert m.stats.corrections == 0
    # A slip on the second answer is fully reversible: one undo splits the
    # voices again and un-names the second one.
    m.undo()
    assert [s for s, _, _ in m.transcript()] == ["Marcus", UNKNOWN_LABEL]
    assert m.clusterer.clusters["1"].pinned and m.clusterer.clusters["1"].count == 1
    assert m.clusterer.clusters["2"].count == 1 and not m.clusterer.clusters["2"].pinned
    stored = conn.execute("SELECT cluster_key FROM utterance WHERE meeting_id=? ORDER BY seq", (m.id,))
    assert [r["cluster_key"] for r in stored] == ["1", "2"]


def test_merging_the_same_person_keeps_the_human_pin(voices):
    clusterer = OnlineClusterer()
    clusterer.assign(voices.sample("a"))
    clusterer.assign(voices.sample("b"))
    clusterer.clusters["1"].person_slug = "marcus"          # automatic label
    clusterer.pin("2", "marcus")                              # human label
    clusterer.merge("2", "1")
    assert clusterer.clusters["1"].pinned


def test_a_named_voice_pulls_harder_than_a_stranger(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Marcus")
    between = (POLICY.named_join_similarity + POLICY.cluster_join_similarity) / 2
    line = m.on_utterance(script.say("a", embedding=near(m.clusterer.clusters["1"].centroid, between)))
    assert m.label_for(line.cluster_key) == "Marcus"
    assert m.stats.questions_asked == 1


def test_an_unfamiliar_voice_offers_the_people_already_named(conn, script):
    m = meeting(conn)
    for who, name in (("a", "Marcus"), ("b", "Sarah")):
        m.on_utterance(script.say(who))
        m.answer(max(m.questions), name)
    m.on_utterance(script.say("c"))
    question = next(iter(m.questions.values()))
    assert {name for _, name, _ in question.options} >= {"Marcus", "Sarah"}


def test_tab_completes_people_in_the_room_first(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Nicole")
    assert commands.complete_name(m, "ni") == "Nicole"
    assert commands.complete_name(m, "/ni") is None
    assert commands.complete_name(m, "zz") is None


def test_the_cluster_cap_leaves_a_question_mark_not_a_wrong_name(voices):
    clusterer = OnlineClusterer(max_clusters=2)
    clusterer.assign(voices.sample("a"))
    clusterer.assign(voices.sample("b"))
    assert clusterer.assign(voices.sample("c")).key == UNPLACED


def test_one_answer_teaches_the_rest_of_that_voice(conn, script, voices):
    """Consistency is judged against the centroid, so a growing profile keeps
    learning instead of rejecting every sample that is not its worst match."""
    m = meeting(conn)
    for _ in range(6):
        m.on_utterance(script.say("a", embedding=[float(x) for x in voices.sample("a", noise=0.06)]))
    m.answer(next(iter(m.questions)), "Marcus")
    stored = conn.execute(
        "SELECT COUNT(*) FROM voice_sample WHERE person_slug='marcus' AND revoked_at IS NULL"
    ).fetchone()[0]
    assert stored == 6
    probe = voices.sample("a", noise=0.06)
    assert people.consistency_with_profile(conn, "marcus", MODEL, probe) > POLICY.min_window_consistency


def test_command_words_typed_without_a_slash_are_commands_not_people(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    for word in ("skip", "help"):
        commands.apply(m, word)
    assert not m.questions
    assert "restored" in commands.apply(m, "Undo").message
    assert conn.execute("SELECT COUNT(*) FROM person").fetchone()[0] == 0


def test_wrong_after_an_unplaced_line_never_renames_an_earlier_voice(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Sarah")
    m.on_utterance(script.say("b", "ok", quality=SHORT))
    result = commands.apply(m, "/wrong Marcus")
    assert not result.changed and "too short" in result.message
    assert m.label_for("1") == "Sarah" and m.stats.corrections == 0


def test_wrong_renames_the_last_line(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    assert commands.apply(m, "/wrong Marcus").changed
    assert m.label_for("1") == "Marcus" and m.stats.corrections == 1


def test_wrong_with_no_placed_line_is_a_message_not_a_crash(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a", "ok", quality=SHORT))
    result = commands.apply(m, "/wrong Marcus")
    assert not result.changed and "too short" in result.message
    assert m.stats.corrections == 0


def test_a_short_line_cannot_rename_an_unnamed_voice(conn, script, voices):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.skip()
    before = m.clusterer.clusters["1"].person_slug
    m.on_utterance(script.say("a", "ok", quality=SHORT))
    assert m.clusterer.clusters["1"].person_slug == before


def test_undo_brings_back_a_skipped_question(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    commands.apply(m, "")
    assert not m.questions
    assert "restored" in commands.apply(m, "/undo").message
    assert m.questions
    m.on_utterance(script.say("a"))
    assert m.stats.questions_asked == 1 and len(m.questions) == 1


def test_notes_names_travel_in_a_file_not_the_command_line(tmp_path, monkeypatch):
    from meet import notes

    cmd = tmp_path / "meet-notes.cmd"
    cmd.write_text("@echo off\n")
    folder = tmp_path / "2026 meeting"
    folder.mkdir()
    (folder / "audio.wav").write_bytes(b"RIFF")
    calls = []
    monkeypatch.setenv("MEET_NOTES_CMD", str(cmd))
    monkeypatch.setattr(notes.subprocess, "Popen", lambda args, **kw: calls.append(args))
    assert notes.launch(folder, ["AT&T Smith", "Ms. 100%"]) == cmd
    assert calls == [[str(cmd), str(folder)]]
    assert (folder / "names.txt").read_text(encoding="utf-8").splitlines() == ["AT&T Smith", "Ms. 100%"]


def test_notes_are_optional(tmp_path, monkeypatch):
    from meet import notes

    monkeypatch.setenv("MEET_NOTES_CMD", str(tmp_path / "missing.cmd"))
    assert notes.launch(tmp_path, ["Flo"]) is None


def test_a_local_laya_needs_no_key(monkeypatch):
    from meet.identity import jev

    monkeypatch.setattr(jev, "api_key", lambda: None)
    monkeypatch.setattr(jev, "JEV_URL", "http://127.0.0.1:11435/v1/systemone")
    assert jev.request_key() == jev.LOCAL_KEY
    monkeypatch.setattr(jev, "JEV_URL", "https://api.typesafe.ai/v1/systemone")
    assert jev.request_key() is None


def test_undo_of_a_fold_keeps_lines_heard_after_it(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    m.answer(next(iter(m.questions)), "Marcus")
    m.on_utterance(script.say("a", embedding=near(m.clusterer.clusters["1"].centroid, 0.10)))
    m.answer(next(iter(m.questions)), "Marcus")
    for _ in range(3):
        m.on_utterance(script.say("a"))
    m.undo()
    one = m.clusterer.clusters["1"]
    on_one = [ln.utterance_id for ln in m.lines if ln.cluster_key == "1"]
    assert one.count == len(on_one) == 4 and sorted(one.utterance_ids) == sorted(on_one)


def test_notes_launch_failure_is_quiet(tmp_path, monkeypatch):
    from meet import notes

    cmd = tmp_path / "meet-notes.cmd"
    cmd.write_text("x")
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("MEET_NOTES_CMD", str(cmd))

    def boom(*a, **k):
        raise OSError("no")

    monkeypatch.setattr(notes.subprocess, "Popen", boom)
    assert notes.launch(tmp_path, ["Flo"]) is None


def test_undoing_a_skip_after_the_voice_was_named_does_not_ask_again(conn, script):
    m = meeting(conn)
    m.on_utterance(script.say("a"))
    commands.apply(m, "/skip")
    m.name_cluster("1", "Marcus")
    m._undo.pop()  # leave only the skip on the stack
    assert "named Marcus" in commands.apply(m, "/undo").message
    assert not m.questions
