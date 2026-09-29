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
    assert ":merge" in commands.apply(m, ":help").message
    assert commands.apply(m, ":end").finished
