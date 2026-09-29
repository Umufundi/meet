"""The live command language, shared by every frontend.

Kept separate from any renderer so the terminal UI, the plain stream, and
anything later (a Teams cockpit, a phone) all speak the same grammar and inherit
corrections identically. Every command returns a one-line result; none of them
can block capture, because none of them touch the listener.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..session.meeting import Meeting


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    usage: str
    summary: str

    @property
    def form(self) -> str:
        return f"/{self.name} {self.usage}".rstrip()


# The one list every frontend reads: the `/` menu, completion, and help all
# come from here, so they cannot drift apart. `:name` works too, for habit.
COMMANDS = (
    Command("people", "", "who is speaking, and how sure Meet is about each"),
    Command("name", "<speaker#> <name>", "name a speaker directly"),
    Command("wrong", "<name>", "the last line was actually this person"),
    Command("merge", "<a> <b>", "speakers a and b are the same person"),
    Command("undo", "", "reverse the last naming"),
    Command("decision", "[text]", "mark the last line (or this text) as a decision"),
    Command("action", "[text]", "mark the last line (or this text) as an action item"),
    Command("help", "", "list everything you can type"),
    Command("end", "", "finish the meeting and write the record"),
)
PREFIXES = ("/", ":")


def suggest(text: str) -> list[Command]:
    """Commands matching what has been typed so far, for the live menu.

    `/` alone lists everything; `/me` narrows to /merge; once a space is typed
    only the exact command stays, so its usage remains visible while typing
    the arguments. Anything not starting with a prefix gets no menu.
    """
    if not text.startswith(PREFIXES):
        return []
    word, space, _ = text[1:].partition(" ")
    word = word.lower()
    if space:
        return [c for c in COMMANDS if c.name == word]
    return [c for c in COMMANDS if c.name.startswith(word)]


HELP = "\n".join(
    ["  1-4               pick that option for the question on screen",
     "  <q#> <name>       answer question q# with a name (a new name enrols them)"]
    + [f"  {c.form:<18}{c.summary}" for c in COMMANDS]
)


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

    if not text.startswith(PREFIXES):
        # `3 Marcus` answers question 3; a bare `2` picks option 2 of the
        # question on screen. These are the common case during a live meeting
        # and deserve the fewest keystrokes.
        head, _, rest = text.partition(" ")
        if head.isdigit() and rest.strip():
            try:
                name = meeting.answer(int(head), rest.strip())
            except KeyError:
                return Result(f"no open question {head}")
            return Result(f"speaker identified as {name}", changed=True)
        if head.isdigit():
            return _pick(meeting, int(head))
        return Result("commands start with / (try /help), or type a number to answer")

    parts = text[1:].split()
    command = parts[0].lower() if parts else ""
    args = parts[1:]

    if command in {"help", "?", ""}:
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

    near = suggest("/" + command[:2])
    hint = f" (did you mean {near[0].form}?)" if near else ""
    return Result(f"unknown command /{command}{hint} — try /help")


def _pick(meeting: Meeting, choice: int) -> Result:
    """A bare number: option `choice` of the question on screen (the oldest)."""
    if not meeting.questions:
        return Result("no question is waiting; to name a speaker use /name <speaker#> <name>")
    question = min(meeting.questions.values(), key=lambda q: q.id)
    if not 1 <= choice <= len(question.options):
        if not question.options:
            return Result(f"no suggestions for this voice; type {question.id} <name>")
        return Result(f"pick 1-{len(question.options)}, or type {question.id} <name>")
    slug, _name, _similarity = question.options[choice - 1]
    name = meeting.answer(question.id, slug)
    return Result(f"speaker identified as {name}", changed=True)
