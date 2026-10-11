"""Play back and style a 1D cellular automaton.

Run with:  .venv/bin/streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# Streamlit Cloud runs from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellular_automata.engine import offsets_for_radius, random_state, run, single_cell
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

# --- Rule -------------------------------------------------------------------
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
    steps = st.slider("Generations", 50, 1000, 300, 10)
    start = st.radio("Start", ["Random", "Single cell"], horizontal=True)
    seed = st.number_input("Start seed", 0, 10**9, 0)
    boundary = st.radio("Edges", ["periodic", "fixed"], horizontal=True)


@st.cache_data
def simulate(table_bytes, k, offsets, width, steps, start, seed, boundary):
    table = np.frombuffer(table_bytes, dtype=np.uint8)
    init = (random_state(width, k, np.random.default_rng(seed)) if start == "Random"
            else single_cell(width, k))
    return init, run(init, table, k, offsets, steps, boundary)


init, history = simulate(table.tobytes(), k, offsets, width, steps, start, int(seed), boundary)

# --- Styling ----------------------------------------------------------------
with st.sidebar:
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


@st.cache_data
def derived(history, name):
    return DERIVED[name](history) if name != "Local surprisal" else DERIVED[name](history, k)


breaks = None
if colour_by == "State":
    image = categorical(history, colours)
else:
    values = derived(history, colour_by)
    if classing == "Continuous":
        image = continuous(values, GRADIENTS[gradient])
    else:
        image, breaks = classified(values, GRADIENTS[gradient], classing, n_classes)

# --- Playback ---------------------------------------------------------------
left, right = st.columns([3, 1])
with left:
    if "gen" not in st.session_state or st.session_state.gen > steps:
        st.session_state.gen = steps
    c1, c2, c3 = st.columns([1, 1, 4])
    play = c1.button("Play")
    if c2.button("Reset"):
        st.session_state.gen = 0
    speed = c3.slider("Generations per frame", 1, 50, 5)
    gen = st.slider("Generation", 0, steps, key="gen")
    frame = st.empty()

    def show(t):
        canvas = np.full_like(image, 255)
        canvas[: t + 1] = image[: t + 1]
        frame.image(upscale(canvas, scale), caption=f"Generation {t} of {steps}")

    if play:
        for t in range(gen, steps + 1, speed):
            show(t)
            time.sleep(0.03)
        show(steps)
    else:
        show(gen)
    if breaks is not None:
        st.caption("Class breaks: " + ", ".join(f"{b:.3g}" for b in breaks))

# --- Metrics ----------------------------------------------------------------
@st.cache_data
def metric_panel(table_bytes, k, offsets, init, history, boundary):
    table = np.frombuffer(table_bytes, dtype=np.uint8)
    late = history[history.shape[0] // 2 :]
    comp = compression_ratio(late, k)
    dmg = float(damage_spreading(table, init, k, offsets, history.shape[0] - 1, boundary)[-1])
    transient, period = transient_and_period(table, init, k, offsets, 2000, boundary)
    return {
        "Compression ratio": comp,
        "Damage spread": dmg,
        "Structured complexity": structured_complexity(dmg, comp),
        "Compression decay": compression_decay(history, k),
        "Spatial entropy": spatial_entropy(history),
        "Langton's lambda": langton_lambda(table),
        "Transient / period": f"{transient} / {period if period else '> 2000'}",
    }


with right:
    st.subheader("Metrics")
    if mode.startswith("Elementary"):
        rep = canonical_number(int(number))
        cls = CLASS_OF[rep]
        st.markdown(f"Wolfram class **{['I', 'II', 'III', 'IV'][cls - 1]}**"
                    + (f" (equivalent to rule {rep})" if rep != number else ""))
    for name, value in metric_panel(table.tobytes(), k, offsets, init, history, boundary).items():
        st.metric(name, f"{value:.3f}" if isinstance(value, float) else value)
    with st.expander("Rule table"):
        st.code(" ".join(map(str, table[::-1])), language=None)
        st.caption("Outputs from the highest neighbourhood index down, Wolfram order.")
