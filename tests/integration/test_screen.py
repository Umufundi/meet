"""The full-screen app, driven by Textual's test pilot with a silent listener."""

import asyncio

from textual.widgets import Input, Static

from meet.config import POLICY
from meet.session.meeting import Meeting
from meet.ui import app as screen


class QuietListener:
    alive = True

    def __init__(self):
        self.stopped = False

    def drain(self):
        return iter(())

    def stop(self, timeout=None):
        self.stopped = True
        self.alive = False


def run(coro):
    return asyncio.run(coro)


def test_slash_opens_the_menu_and_tab_completes(conn, tmp_path):
    async def go():
        meeting = Meeting(conn, title="t", use_jev=False, policy=POLICY.first_meeting())
        app = screen.MeetApp(meeting, QuietListener(), tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("/")
            menu = app.query_one("#menu", Static)
            assert menu.has_class("open")
            await pilot.press("m", "e")
            assert "/merge" in str(menu.render())
            await pilot.press("tab")
            field = app.query_one("#prompt", Input)
            assert field.value == "/merge "
            field.value = ""
            await pilot.pause()
            assert not menu.has_class("open")

    run(go())


def test_arrows_move_the_menu_and_tab_takes_the_highlight(conn, tmp_path):
    async def go():
        meeting = Meeting(conn, title="t", use_jev=False, policy=POLICY.first_meeting())
        app = screen.MeetApp(meeting, QuietListener(), tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.press("down", "down")          # people -> name -> wrong
            assert "› /wrong" in str(app.query_one("#menu", Static).render())
            await pilot.press("tab")
            assert app.query_one("#prompt", Input).value == "/wrong "
            app.query_one("#prompt", Input).value = ""
            await pilot.press("/", "up")                # wraps to the last: /end
            assert "› /end" in str(app.query_one("#menu", Static).render())

    run(go())


def test_enter_on_a_partial_command_picks_the_highlight(conn, tmp_path):
    async def go():
        meeting = Meeting(conn, title="t", use_jev=False, policy=POLICY.first_meeting())
        listener = QuietListener()
        app = screen.MeetApp(meeting, listener, tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("/", "m", "e", "enter")   # needs arguments: filled in
            assert app.query_one("#prompt", Input).value == "/merge "
            app.query_one("#prompt", Input).value = ""
            await pilot.press("/", "e", "enter")        # no arguments: runs
            await pilot.pause(0.3)
        assert listener.stopped and app.finished

    run(go())


def test_end_finishes_the_transcript_before_exiting(conn, tmp_path):
    async def go():
        meeting = Meeting(conn, title="t", use_jev=False)
        listener = QuietListener()
        app = screen.MeetApp(meeting, listener, tmp_path)
        async with app.run_test() as pilot:
            for key in "/end":
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause(0.3)
        assert listener.stopped and app.finished

    run(go())


def test_long_meetings_keep_a_bounded_screen(conn, tmp_path, script, monkeypatch):
    monkeypatch.setattr(screen, "MAX_VISIBLE_LINES", 5)

    class Talking(QuietListener):
        def __init__(self, utterances):
            super().__init__()
            self.pending = list(utterances)

        def drain(self):
            batch, self.pending = self.pending, []
            return iter(batch)

    async def go():
        meeting = Meeting(conn, title="t", use_jev=False)
        listener = Talking([script.say("a") for _ in range(12)])
        app = screen.MeetApp(meeting, listener, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            assert len(meeting.lines) == 12          # the meeting keeps everything
            assert len(app._lines) == 5              # the screen keeps the newest
            assert len(app.query(screen.Line)) == 5
            assert max(app._lines) == meeting.lines[-1].utterance_id

    run(go())
