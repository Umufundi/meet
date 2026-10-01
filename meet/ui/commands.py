"""The live command language, shared by every frontend.

Kept separate from any renderer so the terminal UI, the plain stream, and
anything later (a Teams cockpit, a phone) all speak the same grammar and inherit
corrections identically. Every command returns a one-line result; none of them
can block capture, because none of them touch the listener.
"""

from __future__ import annotations

import time
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
    Command("skip", "", "leave the question on screen unanswered for now (Enter does the same)"),
    Command("decision", "[text]", "mark the last line (or this text) as a decision"),
    Command("action", "[text]", "mark the last line (or this text) as an action item"),
    Command("help", "", "list everything you can type"),
    Command("end", "", "finish the meeting and write the record"),
)
PREFIXES = ("/", ":")
# Commands that also work typed without the slash, so "skip" is never a person.
# /end and /merge are left out: a bare word must never stop the recording or
# fold two voices.
BARE_COMMANDS = {"skip", "undo", "wrong", "name", "people", "help", "decision", "action"}
DOUBLE_ENTER_S = 1.5


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
    ["  <name>            answer the question on screen (Tab completes known names)",
     "  1-4               pick that option for the question on screen",
     "  Enter             skip the question on screen (a quick second Enter is ignored; /skip always works)",
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
    now = time.monotonic()
    recent = now - meeting.last_input_at < DOUBLE_ENTER_S
    meeting.last_input_at = now
    if not text:
        # Enter on an empty line: "I don't know who that was". Skipping must be
        # cheaper than answering, or people type junk names to clear the panel.
        # A second Enter right after typing a name is a double-tap, not a skip.
        if recent or not meeting.questions:
            return Result("")
        return _skip(meeting)

    if not text.startswith(PREFIXES) and text.split()[0].lower() in BARE_COMMANDS:
        # "skip", "undo", "wrong Marcus" typed without the slash are commands,
        # never a new person called Skip.
        text = "/" + text

    if not text.startswith(PREFIXES):
        # A name alone answers the question on screen; `3 Marcus` answers
        # question 3; a bare `2` picks option 2. These are the common case
        # during a live meeting and deserve the fewest keystrokes.
        head, _, rest = text.partition(" ")
        if head.isdigit() and rest.strip():
            try:
                name = meeting.answer(int(head), rest.strip())
            except KeyError:
                return Result(f"no open question {head}")
            return Result(f"speaker identified as {name}", changed=True)
        if head.isdigit():
            return _pick(meeting, int(head))
        if not meeting.questions:
            return Result("no question is waiting; to name a speaker use /name <speaker#> <name>")
        if len(text) < 2:
            return Result("type the person's name, a number to pick, or Enter to skip")
        question = min(meeting.questions.values(), key=lambda q: q.id)
        name = meeting.answer(question.id, text)
        return Result(f"speaker identified as {name}", changed=True)

    parts = text[1:].split()
    command = parts[0].lower() if parts else ""
    args = parts[1:]

    if command in {"help", "?", ""}:
        return Result(HELP)

    if command == "skip":
        return _skip(meeting) if meeting.questions else Result("no question is waiting")

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
        except (IndexError, KeyError) as exc:
            return Result(str(exc).strip("'\""))
        return Result(f"corrected to {named}", changed=True)

    if command == "undo":
        try:
            restored = meeting.undo()
        except (IndexError, KeyError):
            return Result("nothing to undo")
        if restored.startswith(("question restored", "that voice")):
            # Undoing a skip: a message about the question, nothing relabelled.
            return Result(restored)
        return Result(f"reverted to {restored}", changed=True)

    if command in {"decision", "action"}:
        return Result(meeting.note(command.upper(), " ".join(args) or None))

    if command == "end":
        return Result("finishing", finished=True)

    near = suggest("/" + command[:2])
    hint = f" (did you mean {near[0].form}?)" if near else ""
    return Result(f"unknown command /{command}{hint} — try /help")


def _skip(meeting: Meeting) -> Result:
    meeting.skip()
    return Result("skipped; that voice will not be asked about again (/undo brings it back)")


def complete_name(meeting: Meeting, typed: str) -> str | None:
    """Tab on a partly typed name: the first known person it could be.

    People already named in this meeting come first, then everyone Meet has a
    voice for, so the likely answer is the one Tab lands on.
    """
    prefix = typed.strip().lower()
    if not prefix or typed.startswith(PREFIXES) or prefix[0].isdigit():
        return None
    present = [meeting.label_for(key) for key in meeting.clusterer.named()]
    stored = [
        row["display_name"]
        for row in meeting.conn.execute(
            "SELECT p.display_name FROM person p WHERE EXISTS (SELECT 1 FROM voice_sample s "
            "WHERE s.person_slug = p.slug AND s.revoked_at IS NULL) ORDER BY p.display_name"
        )
    ]
    for name in [*present, *stored]:
        if name.lower().startswith(prefix) and name != "?":
            return name
    return None


def _pick(meeting: Meeting, choice: int) -> Result:
    """A bare number: option `choice` of the question on screen (the oldest)."""
    if not meeting.questions:
        return Result("no question is waiting; to name a speaker use /name <speaker#> <name>")
    question = min(meeting.questions.values(), key=lambda q: q.id)
    if not 1 <= choice <= len(question.options):
        if not question.options:
            return Result("no suggestions for this voice; type the person's name")
        return Result(f"pick 1-{len(question.options)}, or type the person's name")
    slug, _name, _similarity = question.options[choice - 1]
    name = meeting.answer(question.id, slug)
    return Result(f"speaker identified as {name}", changed=True)
