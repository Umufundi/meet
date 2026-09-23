"""Terminal entry points."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer

from . import sidecar
from .config import POLICY, meetings_dir
from .identity import jev
from .memory import db, people
from .session import output
from .session.meeting import Meeting
from .ui import plain

app = typer.Typer(add_completion=False, help="Who is speaking, resolved live, taught by a human.")


def _slot(title: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    safe = db.slugify(title) or "meeting"
    path = meetings_dir() / f"{stamp}-{safe}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finish(meeting: Meeting, directory: Path, extract: bool) -> None:
    extracts: list[output.Extract] = []
    if extract:
        try:
            extracts = output.extract(meeting)
        except jev.JevUnavailable as exc:
            typer.secho(f"semantic extraction skipped: {exc}", fg=typer.colors.YELLOW, err=True)
    written = output.write(meeting, directory, extracts)
    stats = meeting.stats
    typer.echo("")
    typer.secho(f"{directory}", bold=True)
    for path in written:
        typer.echo(f"  {path.name}")
    typer.echo("")
    typer.echo(
        f"{stats.utterances} utterances · {stats.questions_asked} questions "
        f"({meeting.questions_per_hour():.1f}/hour) · {stats.auto_labelled} auto · "
        f"{stats.corrections} corrected · {stats.samples_learned} voice samples learned"
    )


@app.command()
def devices() -> None:
    """List microphones the listener can open."""
    try:
        typer.echo(sidecar.list_devices())
    except sidecar.ListenerMissing as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@app.command()
def doctor() -> None:
    """Check that the listener, the database, and Jev are reachable."""
    ok, detail = sidecar.check_listener()
    typer.secho(f"listener   {'ok' if ok else detail}", fg=typer.colors.GREEN if ok else typer.colors.RED)
    conn = db.connect()
    known = db.list_people(conn)
    typer.echo(f"memory     {len(known)} people, {db.db_path() if hasattr(db, 'db_path') else ''}")
    typer.secho(
        f"jev        {'key found' if jev.api_key() else 'no API key (questions go to you instead)'}",
        fg=typer.colors.GREEN if jev.api_key() else typer.colors.YELLOW,
    )
    try:
        typer.echo("inputs\n" + sidecar.list_devices())
    except sidecar.ListenerMissing:
        pass


@app.command(name="people")
def people_command() -> None:
    """Who the system can recognise, and how well."""
    conn = db.connect()
    known = db.list_people(conn)
    if not known:
        typer.echo("nobody enrolled yet — name a speaker during a meeting and they are learned")
        return
    for person in known:
        row = conn.execute(
            "SELECT COUNT(*) n FROM voice_sample WHERE person_slug=? AND trust='human' "
            "AND revoked_at IS NULL",
            (person.slug,),
        ).fetchone()
        typer.echo(f"  {person.display_name:<20} {person.sample_count:>3} samples ({row['n']} confirmed)")


@app.command()
def forget(name: str) -> None:
    """Erase one person's voiceprints. Transcripts keep their text."""
    conn = db.connect()
    slug = db.slugify(name)
    row = conn.execute("SELECT display_name FROM person WHERE slug=?", (slug,)).fetchone()
    if row is None:
        typer.secho(f"no such person: {name}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not typer.confirm(f"permanently erase every voiceprint for {row['display_name']}?"):
        raise typer.Exit(0)
    removed = db.delete_person(conn, slug)
    typer.echo(f"erased {removed} voice samples for {row['display_name']}")


@app.command()
def start(
    title: str = typer.Option("Meeting", "--title", "-t"),
    attendees: str = typer.Option("", "--people", "-p", help="comma-separated expected attendees"),
    mic: int | None = typer.Option(None, "--mic", help="input device index from `meet devices`"),
    model: str = typer.Option("small.en", "--model", help="faster-whisper model size"),
    no_jev: bool = typer.Option(False, "--no-jev", help="never consult Jev; ask the human instead"),
    no_extract: bool = typer.Option(False, "--no-extract", help="skip decision/action extraction"),
) -> None:
    """Start listening to the room."""
    _run(
        title=title,
        attendees=attendees,
        source=None,
        mic=mic,
        model=model,
        use_jev=not no_jev,
        extract=not no_extract,
        realtime=False,
    )


@app.command()
def replay(
    wav: Path = typer.Argument(..., exists=True, dir_okay=False),
    title: str = typer.Option("", "--title", "-t"),
    attendees: str = typer.Option("", "--people", "-p"),
    model: str = typer.Option("small.en", "--model"),
    realtime: bool = typer.Option(False, "--realtime", help="pace playback to wall clock"),
    no_jev: bool = typer.Option(False, "--no-jev"),
    no_extract: bool = typer.Option(False, "--no-extract"),
) -> None:
    """Run a 16 kHz mono recording through the same live path.

    Identical code to `start` from the VAD onwards, so a recording is not a
    second-class input: it is the same meeting, heard later.
    """
    _run(
        title=title or wav.stem,
        attendees=attendees,
        source=wav,
        mic=None,
        model=model,
        use_jev=not no_jev,
        extract=not no_extract,
        realtime=realtime,
    )


def _run(
    *,
    title: str,
    attendees: str,
    source: Path | None,
    mic: int | None,
    model: str,
    use_jev: bool,
    extract: bool,
    realtime: bool,
) -> None:
    roster = [name.strip() for name in attendees.split(",") if name.strip()]
    conn = db.connect()
    known = db.list_people(conn)
    policy = POLICY if known else POLICY.first_meeting()

    directory = _slot(title)
    meeting = Meeting(conn, title=title, attendees=roster, policy=policy, use_jev=use_jev)

    try:
        listener = sidecar.Listener(
            wav_path=directory / "audio.wav", device=mic, asr_model=model
        )
    except sidecar.ListenerMissing as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    if source is not None:
        listener.replay(source, realtime=realtime)
    listener.start()

    try:
        plain.run(
            meeting,
            listener,
            output_dir=directory,
            idle_timeout=2.0 if source is not None else None,
        )
    except KeyboardInterrupt:
        typer.echo("\nstopping", err=True)
        listener.stop()
    finally:
        if not listener.alive and listener.stderr_tail and not meeting.lines:
            typer.secho(listener.diagnostics(), fg=typer.colors.RED, err=True)

    _finish(meeting, directory, extract)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
