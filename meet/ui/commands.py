"""The live command language, shared by every frontend.

Kept separate from any renderer so the terminal UI, the plain stream, and
anything later (a Teams cockpit, a phone) all speak the same grammar and inherit
corrections identically. Every command returns a one-line result; none of them
can block capture, because none of them touch the listener.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..session.meeting import Meeting

HELP = """\
  <n> <name>        answer question n: name the speaker (an unknown name enrols them)
  :people           who is in the room, and how well each voice is known
  :name <c> <name>  name speaker cluster c directly
  :merge <a> <b>    clusters a and b are the same person
  :wrong <name>     the last line was actually this person
  :undo             reverse the last naming
  :decision [text]  mark the last line (or this text) as a decision
  :action [text]    mark it as an action item
  :help             this list
  :end              finish the meeting and write the record"""


@dataclass(frozen=True, slots=True)
class Result:
    message: str
    finished: bool = False
    changed: bool = False


def apply(meeting: Meeting, raw: str) -> Result:
    """Interpret one line of human input. Never raises for user error."""
    text = raw.strip()
    if not text:
        return Result("")

    if not text.startswith(":"):
        # `3 Marcus` — answer question 3. The bare form is the common case and
        # deserves the shortest possible keystroke count during a live meeting.
        head, _, rest = text.partition(" ")
        if head.isdigit() and rest.strip():
            try:
                name = meeting.answer(int(head), rest.strip())
            except KeyError:
                return Result(f"no open question {head}")
            return Result(f"speaker identified as {name}", changed=True)
        if head.isdigit():
            return Result("give a name: e.g. `3 Marcus`")
        return Result("commands start with ':' — try :help")

    parts = text[1:].split()
    command = parts[0].lower() if parts else ""
    args = parts[1:]

    if command in {"help", "?"}:
        return Result(HELP)

    if command == "people":
        rows = []
        for key, cluster in sorted(meeting.clusterer.clusters.items(), key=lambda kv: int(kv[0])):
            if cluster.count == 0 and cluster.person_slug is None:
                continue
            mark = "confirmed" if cluster.pinned else "provisional" if cluster.person_slug else "unknown"
            spoken = len([ln for ln in meeting.lines if ln.cluster_key == key])
            rows.append(f"  [{key}] {meeting.label_for(key):<16} {mark:<12} {spoken} lines")
        return Result("\n".join(rows) if rows else "  nobody has spoken yet")

    if command == "name":
        if len(args) < 2:
            return Result("usage: :name <cluster> <name>")
        try:
            named = meeting.name_cluster(args[0], " ".join(args[1:]))
        except KeyError:
            return Result(f"no cluster {args[0]}")
        return Result(f"cluster {args[0]} is {named}", changed=True)

    if command == "merge":
        if len(args) != 2:
            return Result("usage: :merge <a> <b>")
        if not meeting.merge(args[0], args[1]):
            return Result(f"cannot merge {args[0]} into {args[1]}")
        return Result(f"cluster {args[0]} folded into {args[1]}", changed=True)

    if command == "wrong":
        if not args:
            return Result("usage: :wrong <name>")
        try:
            named = meeting.correct_last(" ".join(args))
        except IndexError:
            return Result("nothing has been said yet")
        return Result(f"corrected to {named}", changed=True)

    if command == "undo":
        try:
            restored = meeting.undo()
        except (IndexError, KeyError):
            return Result("nothing to undo")
        return Result(f"reverted to {restored}", changed=True)

    if command in {"decision", "action"}:
        return Result(meeting.note(command.upper(), " ".join(args) or None))

    if command == "end":
        return Result("finishing", finished=True)

    return Result(f"unknown command :{command} — try :help")
