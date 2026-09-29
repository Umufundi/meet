import io

from meet import progress


class Screen(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def test_bar_is_bounded():
    assert progress.bar(0, 10) == "░" * 20
    assert progress.bar(5, 10) == "█" * 10 + "░" * 10
    assert progress.bar(99, 10) == "█" * 20
    assert progress.bar(1, 0).count("█") == 20  # never divides by zero


def test_clock():
    assert progress.clock(0) == "0:00"
    assert progress.clock(125.9) == "2:05"


def test_gauge_redraws_one_line_and_clears_it():
    screen = Screen()
    with progress.Gauge(lambda t: f"working {t}", stream=screen, interval=0.01):
        pass
    out = screen.getvalue()
    assert "working 0:00" in out and "\n" not in out
    assert out.endswith("\r")


def test_gauge_is_silent_when_not_a_terminal():
    stream = io.StringIO()
    with progress.Gauge(lambda t: "x", stream=stream):
        pass
    assert stream.getvalue() == ""


def test_a_broken_render_never_breaks_the_step():
    screen = Screen()
    with progress.Gauge(lambda t: 1 / 0, stream=screen):
        pass


def test_ascii_fallback():
    class Latin1(Screen):
        encoding = "cp1252"

    assert progress._glyphs(Latin1()) == ("#", "-")


def test_folder_size(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 2_000_000)
    assert round(progress.folder_mb(tmp_path)) == 2
    assert progress.folder_mb(tmp_path / "missing") == 0.0
