"""Play a 1D cellular automaton live, with styling and metrics.

Run with:  .venv/bin/streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Streamlit Cloud runs from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellular_automata.engine import offsets_for_radius, random_state, single_cell, step
from cellular_automata.metrics import (
    compression_decay,
    compression_ratio,
    damage_spreading,
    langton_lambda,
    spatial_entropy,
    structured_complexity,
    transient_and_period,
)
from cellular_automata.rules import canonical_number, elementary, random_rule
from cellular_automata.styling import (
    BREAKS,
    DERIVED,
    GRADIENTS,
    PALETTES,
    categorical,
    classified,
    continuous,
    upscale,
)
from cellular_automata.wolfram_classes import CLASS_OF

st.set_page_config(page_title="Cellular automata", layout="wide")
st.title("Cellular automata")
ss = st.session_state

# --- Rule and world ---------------------------------------------------------
with st.sidebar:
    st.header("Rule")
    mode = st.radio("Rule source", ["Elementary (Wolfram number)", "Random rule"])
    if mode.startswith("Elementary"):
        k, radius = 2, 1
        number = st.number_input("Rule number", 0, 255, 110)
        table = elementary(int(number))
    else:
        k = st.slider("States", 2, 5, 2)
        radius = st.slider("Radius", 1, 3, 2)
        use_lambda = st.checkbox("Set Langton's lambda", True)
        lam = st.slider("Lambda", 0.0, 1.0, 0.5, 0.01) if use_lambda else None
        rule_seed = st.number_input("Rule seed", 0, 10**9, 0)
        table = random_rule(k, 2 * radius + 1, np.random.default_rng(int(rule_seed)), lam)
    offsets = offsets_for_radius(radius)

    st.header("World")
    width = st.slider("Width (cells)", 50, 600, 300, 10)
    rows_shown = st.slider("Generations shown", 50, 600, 300, 10)
    start = st.radio("Start", ["Random", "Single cell"], horizontal=True)
    seed = st.number_input("Start seed", 0, 10**9, 0)
    boundary = st.radio("Edges", ["periodic", "fixed"], horizontal=True)

    st.header("Style")
    colour_by = st.radio("Colour by", ["State"] + list(DERIVED))
    if colour_by == "State":
        palette = st.selectbox("Palette", list(PALETTES))
        colours = [st.color_picker(f"State {s}", PALETTES[palette][s % len(PALETTES[palette])],
                                   key=f"c{palette}{s}") for s in range(k)]
    else:
        gradient = st.selectbox("Gradient", list(GRADIENTS))
        classing = st.selectbox("Classes", ["Continuous"] + list(BREAKS))
        n_classes = st.slider("Number of classes", 2, 9, 5) if classing != "Continuous" else None
    scale = st.slider("Pixel size", 1, 6, 2)


# --- Live state ---------------------------------------------------------------
def reset() -> None:
    init = (random_state(width, k, np.random.default_rng(int(seed))) if start == "Random"
            else single_cell(width, k))
    ss.init = init
    ss.window = init[None, :].copy()
    ss.gen = 0


params = (table.tobytes(), k, offsets, width, start, int(seed), boundary)
if ss.get("params") != params:
    ss.params = params
    reset()
ss.setdefault("running", False)


def advance(n: int) -> None:
    state = ss.window[-1]
    new_rows = np.empty((n, width), dtype=np.uint8)
    for i in range(n):
        state = step(state, table, k, offsets, boundary)
        new_rows[i] = state
    ss.window = np.vstack([ss.window, new_rows])[-rows_shown:]
    ss.gen += n


def toggle() -> None:
    ss.running = not ss.running


left, right = st.columns([3, 1])
with left:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 2])
    c1.button("Pause" if ss.running else "Play", on_click=toggle, type="primary")
    c2.button("Step", on_click=lambda: advance(ss.get("per_frame", 1)), disabled=ss.running)
    c3.button("Reset", on_click=reset)
    c4.slider("Generations per frame", 1, 50, 1, key="per_frame")
    fps = c5.slider("Frames per second", 1, 30, 10)


def styled(window: np.ndarray):
    if colour_by == "State":
        return categorical(window, colours), None
    values = (DERIVED[colour_by](window, k) if colour_by == "Local surprisal"
              else DERIVED[colour_by](window))
    if classing == "Continuous":
        return continuous(values, GRADIENTS[gradient]), None
    return classified(values, GRADIENTS[gradient], classing, n_classes)


@st.fragment(run_every=(1 / fps) if ss.running else None)
def live() -> None:
    if ss.running:
        advance(ss.per_frame)
    image, breaks = styled(ss.window)
    canvas = np.full((rows_shown, width, 3), 255, dtype=np.uint8)
    canvas[: image.shape[0]] = image
    st.markdown(f"Generation **{ss.gen}**" + (" (running)" if ss.running else ""))
    st.image(upscale(canvas, scale), caption="Time runs down. The newest generation is the bottom row.")
    if breaks is not None:
        st.caption("Class breaks: " + ", ".join(f"{b:.3g}" for b in breaks))


with left:
    live()


# --- Metrics ----------------------------------------------------------------
@st.cache_data
def metric_panel(table_bytes, k, offsets, state, window, boundary):
    table = np.frombuffer(table_bytes, dtype=np.uint8)
    steps = max(window.shape[0] - 1, 1)
    late = window[window.shape[0] // 2 :]
    comp = compression_ratio(late, k)
    dmg = float(damage_spreading(table, state, k, offsets, steps, boundary)[-1])
    transient, period = transient_and_period(table, state, k, offsets, 2000, boundary)
    return {
        "Compression ratio": comp,
        "Damage spread": dmg,
        "Structured complexity": structured_complexity(dmg, comp),
        "Compression decay": compression_decay(window, k),
        "Spatial entropy": spatial_entropy(window),
        "Transient / period": f"{transient} / {period if period else '> 2000'}",
    }


with right:
    st.subheader("Metrics")
    if mode.startswith("Elementary"):
        rep = canonical_number(int(number))
        st.markdown(f"Wolfram class **{['I', 'II', 'III', 'IV'][CLASS_OF[rep] - 1]}**"
                    + (f" (equivalent to rule {rep})" if rep != number else ""))
    st.metric("Langton's lambda", f"{langton_lambda(table):.3f}")
    if ss.running:
        st.caption("Pause to measure the generations on screen.")
    elif ss.window.shape[0] < 20:
        st.caption("Run at least 20 generations to measure.")
    else:
        st.caption("Measured on the generations on screen; damage and period run on "
                   "from the newest generation.")
        for name, value in metric_panel(table.tobytes(), k, offsets, ss.window[-1],
                                        ss.window, boundary).items():
            st.metric(name, f"{value:.3f}" if isinstance(value, float) else value)
    with st.expander("Rule table"):
        st.code(" ".join(map(str, table[::-1])), language=None)
        st.caption("Outputs from the highest neighbourhood index down, Wolfram order.")
