import json

from meet.config import POLICY
from meet.session import output
from meet.session.meeting import Meeting

FILES = {"transcript.md", "transcript.json", "speakers.json", "decisions.json", "actions.json",
         "state.json", "minutes.md"}


def test_record_is_complete_and_uses_corrected_names(conn, script, tmp_path):
    m = Meeting(conn, title="Monday Staff", use_jev=False, policy=POLICY.first_meeting())
    m.on_utterance(script.say("a", "We should send the paperwork Monday."))
    m.on_utterance(script.say("b", "Agreed, I will draft it."))
    m.name_cluster("1", "Marcus")
    m.note("DECISION", "send the paperwork Monday")
    extracts = [output.Extract("Marcus", "We should send the paperwork Monday.", 0, "DECISION", 0.9),
                output.Extract("?", "Agreed, I will draft it.", 4000, "ACTION", 0.8)]
    written = output.write(m, tmp_path / "out", extracts)

    assert {p.name for p in written} == FILES
    out = tmp_path / "out"
    transcript = json.loads((out / "transcript.json").read_text())
    assert [t["speaker"] for t in transcript] == ["Marcus", "?"]
    speakers = json.loads((out / "speakers.json").read_text())
    assert speakers["1"] == {"person": "Marcus", "slug": "marcus", "named_by_human": True, "utterances": 1}
    assert len(json.loads((out / "decisions.json").read_text())) == 1
    assert len(json.loads((out / "actions.json").read_text())) == 1
    state = json.loads((out / "state.json").read_text())
    assert state["stats"]["utterances"] == 2 and state["meeting_id"] == m.id

    minutes = (out / "minutes.md").read_text()
    assert "# Monday Staff" in minutes
    assert "- Marcus" in minutes and "- ?" not in minutes
    assert "## Decisions" in minutes and "## Marked during the meeting" in minutes


def test_empty_meeting_still_writes_a_record(conn, tmp_path):
    m = Meeting(conn, title="Nothing", use_jev=False)
    written = output.write(m, tmp_path / "empty")
    assert {p.name for p in written} == FILES
    assert json.loads((tmp_path / "empty" / "transcript.json").read_text()) == []
