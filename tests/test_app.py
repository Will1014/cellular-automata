import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PAGE2 = str(Path(APP).parent / "pages" / "2_Two_layers.py")


def generation(at) -> int:
    for m in at.markdown:
        found = re.search(r"Generation \*\*(\d+)\*\*", m.value)
        if found:
            return int(found.group(1))
    raise AssertionError("no generation counter on the page")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Wolfram class" in m.value for m in at.markdown)
    assert generation(at) == 0


def test_step_advances_and_runs_past_the_window_without_growing():
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(s for s in at.slider if s.label == "Generations per frame").set_value(50).run()
    for _ in range(8):  # 400 generations, more than the 300 shown
        button(at, "Step").click().run()
    assert not at.exception
    assert generation(at) == 400
    assert at.session_state.window.shape[0] == 300


def test_play_and_pause_toggle():
    at = AppTest.from_file(APP, default_timeout=60).run()
    button(at, "Play").click().run()
    assert at.session_state.running
    assert generation(at) >= 1  # the fragment advanced on the rerun
    button(at, "Pause").click().run()
    assert not at.session_state.running
    assert not at.exception


def test_reset_returns_to_generation_zero():
    at = AppTest.from_file(APP, default_timeout=60).run()
    button(at, "Step").click().run()
    button(at, "Reset").click().run()
    assert generation(at) == 0


def test_app_gradient_with_jenks_and_random_rule():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Random rule").run()
    colour_by = next(r for r in at.sidebar.radio if r.label == "Colour by")
    colour_by.set_value("Local surprisal").run()
    classes = next(s for s in at.sidebar.selectbox if s.label == "Classes")
    classes.set_value("Natural breaks (Jenks)").run()
    for _ in range(3):
        button(at, "Step").click().run()
    assert not at.exception
    assert any("Class breaks" in c.value for c in at.caption)


def test_two_layer_page_steps_on_hex_and_1d():
    at = AppTest.from_file(PAGE2, default_timeout=120).run()
    assert not at.exception
    for _ in range(5):
        button(at, "Step").click().run()
    assert generation(at) == 5
    assert any(m.label.startswith("Environment -> species") for m in at.metric)
    lattice = next(s for s in at.sidebar.selectbox if s.label == "Lattice")
    lattice.set_value("1D line").run()
    assert generation(at) == 0
    button(at, "Play").click().run()
    assert not at.exception
