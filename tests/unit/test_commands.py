from meet.config import POLICY
from meet.session.meeting import Meeting
from meet.ui import commands


def make(conn, script, *who):
    m = Meeting(conn, title="t", use_jev=False, policy=POLICY.first_meeting())
    for w in who:
        m.on_utterance(script.say(w))
    return m


def test_bare_answer(conn, script):
    m = make(conn, script, "a")
    qid = next(iter(m.questions))
    r = commands.apply(m, f"{qid} Marcus")
    assert r.changed and "Marcus" in r.message
    assert m.questions == {}


def test_user_errors_never_raise(conn, script):
    m = make(conn, script)
    for raw in ("", "hello", "7", "7 Marcus", ":name", ":name 9 X", ":merge 1", ":merge 1 2",
                ":wrong", ":wrong Bob", ":undo", ":bogus", ":", ":people"):
        r = commands.apply(m, raw)
        assert not r.finished
        assert not r.changed, raw


def test_command_grammar(conn, script):
    m = make(conn, script, "a", "b")
    assert commands.apply(m, ":name 1 Marcus").changed
    assert commands.apply(m, ":merge 2 1").changed
    assert "Marcus" in commands.apply(m, ":people").message
    assert commands.apply(m, ":wrong Sarah").changed
    assert commands.apply(m, ":undo").changed
    assert commands.apply(m, ":decision ship it").message == "DECISION: ship it"
    assert commands.apply(m, ":action").message.startswith("ACTION: ")
    assert "/merge" in commands.apply(m, ":help").message
    assert commands.apply(m, ":end").finished


def test_slash_is_the_command_prefix(conn, script):
    m = make(conn, script, "a", "b")
    assert commands.apply(m, "/name 1 Marcus").changed
    assert commands.apply(m, "/merge 2 1").changed
    assert "Marcus" in commands.apply(m, "/people").message
    assert "/merge" in commands.apply(m, "/").message  # a bare slash is help
    assert commands.apply(m, "/end").finished
    assert "did you mean /merge" in commands.apply(m, "/mer").message


def test_menu_narrows_as_you_type():
    names = lambda text: [c.name for c in commands.suggest(text)]  # noqa: E731
    assert names("/") == [c.name for c in commands.COMMANDS]
    assert names("/me") == ["merge"]
    assert names(":un") == ["undo"]
    assert names("/merge 2") == ["merge"]   # usage stays visible while typing arguments
    assert names("/zzz") == []
    assert names("Marcus") == [] and names("3 Marcus") == []


def test_every_command_in_the_menu_is_understood(conn, script):
    m = make(conn, script, "a")
    for command in commands.COMMANDS:
        assert "unknown command" not in commands.apply(m, f"/{command.name}").message, command.name


def test_bare_number_picks_an_option(conn, script, voices):
    from meet.memory import db, people

    for name, who in (("Marcus", "a"), ("Sarah", "b")):
        slug = db.upsert_person(conn, name)
        people.learn(conn, slug, voices.base(who), "ecapa-voxceleb@sb1", trust="human", speech_s=3,
                     snr_db=20, clipping=0, rms=0.05)
    import numpy as np

    between = voices.base("a") + voices.base("b")
    from meet.config import POLICY

    m = Meeting(conn, title="t", use_jev=False, policy=POLICY.first_meeting().__class__(
        match_min_similarity=0.5, provisional_similarity=0.6, auto_similarity=0.65))
    m.on_utterance(script.say("x", embedding=list(between / np.linalg.norm(between))))
    question = next(iter(m.questions.values()))
    assert len(question.options) == 2
    assert "pick 1-2" in commands.apply(m, "5").message
    second = question.options[1][1]
    result = commands.apply(m, "2")
    assert result.changed and second in result.message
    assert m.questions == {}
    assert "no question is waiting" in commands.apply(m, "1").message
