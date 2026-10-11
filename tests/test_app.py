from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Wolfram class" in m.value for m in at.markdown)


def test_app_gradient_with_jenks_and_random_rule():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Random rule").run()
    colour_by = next(r for r in at.sidebar.radio if r.label == "Colour by")
    colour_by.set_value("Local surprisal").run()
    classes = next(s for s in at.sidebar.selectbox if s.label == "Classes")
    classes.set_value("Natural breaks (Jenks)").run()
    assert not at.exception
    assert any("Class breaks" in c.value for c in at.caption)
