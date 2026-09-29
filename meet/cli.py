"""Terminal entry points."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer

from . import doctor as doctor_mod
from . import install, runtime, sidecar
from .config import POLICY, meetings_dir
from .identity import jev
from .memory import db
from .session import output
from .session.meeting import Meeting
from .ui import plain
from .ui.app import MeetApp

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
def setup(
    model: str = typer.Option("small.en", "--model", help="faster-whisper model to download"),
    force: bool = typer.Option(False, "--force", help="rebuild the listener runtime from scratch"),
    skip_models: bool = typer.Option(False, "--skip-models", help="do not download models now"),
    no_mic_test: bool = typer.Option(False, "--no-mic-test", help="skip the 3-second microphone test"),
    torch_index: str = typer.Option(
        "auto", "--torch-index", help="auto | cpu | pypi  (auto = CPU wheels on Linux, PyPI elsewhere)"
    ),
) -> None:
    """Install the listener runtime and models, then check everything. Run once."""
    ok = install.Setup(
        asr_model=model,
        force=force,
        skip_models=skip_models,
        mic_test=not no_mic_test and sys.stdin.isatty(),
        torch_index=torch_index,
        echo=typer.echo,
    ).run()
    raise typer.Exit(0 if ok else 1)


@app.command()
def doctor(
    audio: bool = typer.Option(False, "--audio", help="also record 3 seconds from the microphone"),
    models: bool = typer.Option(False, "--models", help="also load every model, offline"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="show details for passing checks too"),
) -> None:
    """Check that a meeting will work, and say what to fix if it will not."""
    report = doctor_mod.run(audio=audio, models=models)
    typer.echo(doctor_mod.render(report, verbose=verbose))
    raise typer.Exit(0 if report.ready else 1)


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
    model: str | None = typer.Option(None, "--model", help="faster-whisper model (default: from setup)"),
    no_jev: bool = typer.Option(False, "--no-jev", help="never consult Jev; ask the human instead"),
    no_extract: bool = typer.Option(False, "--no-extract", help="skip decision/action extraction"),
    plain_ui: bool = typer.Option(False, "--plain", help="line output instead of the full screen"),
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
        plain_ui=plain_ui,
    )


@app.command()
def replay(
    wav: Path = typer.Argument(..., exists=True, dir_okay=False),
    title: str = typer.Option("", "--title", "-t"),
    attendees: str = typer.Option("", "--people", "-p"),
    model: str | None = typer.Option(None, "--model"),
    realtime: bool = typer.Option(False, "--realtime", help="pace playback to wall clock"),
    no_jev: bool = typer.Option(False, "--no-jev"),
    no_extract: bool = typer.Option(False, "--no-extract"),
    plain_ui: bool = typer.Option(True, "--plain/--screen", help="line output, or the full screen"),
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
        plain_ui=plain_ui,
    )


def _run(
    *,
    title: str,
    attendees: str,
    source: Path | None,
    mic: int | None,
    model: str | None,
    use_jev: bool,
    extract: bool,
    realtime: bool,
    plain_ui: bool = False,
) -> None:
    roster = [name.strip() for name in attendees.split(",") if name.strip()]
    # The model `meet setup` downloaded; meetings run offline, so any other
    # model would fail to load.
    model = model or runtime.read_manifest().get("asr_model") or "small.en"
    try:
        conn = db.connect()
    except db.DatabaseTooNew as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    # Only a stored voice counts as knowing someone. Attendee names alone create
    # person rows with no samples, and must not switch off the first-meeting
    # policy.
    known = [p for p in db.list_people(conn) if p.sample_count]
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

    # The full screen needs a real terminal. Piped or redirected output falls
    # back to line mode rather than rendering control codes into a file.
    interactive = sys.stdout.isatty() and sys.stdin.isatty()
    try:
        if plain_ui or not interactive:
            plain.run(
                meeting,
                listener,
                output_dir=directory,
                idle_timeout=2.0 if source is not None else None,
            )
        else:
            MeetApp(meeting, listener, directory).run()
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
