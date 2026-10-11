"""Species and environment as two coupled layers on the same lattice, played live."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cellular_automata.coupled import (
    consumer_resource,
    layer_mutual_information,
    random_coupled_rule,
    step_coupled,
)
from cellular_automata.lattice import LATTICES
from cellular_automata.styling import GRADIENTS, categorical, ramp, render_hex, upscale

st.set_page_config(page_title="Two layers", layout="wide")
st.title("Species and environment")
st.caption("Two layers on one lattice. Each layer reads itself and the other. A second "
           "copy runs alongside with the cross-layer links cut (the other layer spatially "
           "shuffled), so any difference between the two is the interaction.")
ss = st.session_state
MI_WINDOW = 100  # generations of history used for the interaction measure
CHART_LENGTH = 1000  # generations kept for the occupancy chart

with st.sidebar:
    st.header("World")
    lattice_name = st.selectbox("Lattice", list(LATTICES), index=3)
    offsets = LATTICES[lattice_name]
    one_d = len(offsets[0]) == 1
    size = st.slider("Width" if one_d else "Grid side", 30, 600 if one_d else 150,
                     300 if one_d else 80)
    rows_shown = st.slider("Generations shown", 50, 600, 300) if one_d else None
    ke = st.slider("Environment levels", 2, 6, 3)
    env_period = st.slider("Environment updates every n steps", 1, 10, 1)
    start_occ = st.slider("Starting occupancy", 0.0, 1.0, 0.05, 0.01)
    seed = st.number_input("Seed", 0, 10**9, 0)

    st.header("Rule")
    kind = st.radio("Rule", ["Consumer-resource", "Random"])
    n = len(offsets)
    if kind == "Consumer-resource":
        colonise_at = st.slider("Colonise with at least this many occupied neighbours", 1, n,
                                1 if one_d else 2)
        regrow_at = st.slider("Regrow when neighbour ground sums to at least", 0, n * (ke - 1), 1)
        rule = consumer_resource(ke, n, colonise_at, regrow_at)
    else:
        rule_seed = st.number_input("Rule seed", 0, 10**9, 0)
        rule = random_coupled_rule(ke, n, np.random.default_rng(int(rule_seed)))

    st.header("Links")
    modes = ["on", "shuffled", "constant"]
    s_mode = st.selectbox("Species sees environment", modes)
    e_mode = st.selectbox("Environment sees species", modes)

    st.header("Style")
    species_colour = st.color_picker("Occupied", "#c0392b")
    gradient = st.selectbox("Environment gradient", list(GRADIENTS), index=3)
    cell_px = st.slider("Cell size", 1 if one_d else 3, 6 if one_d else 14, 2 if one_d else 7)


# --- Live state ---------------------------------------------------------------
def reset() -> None:
    shape = (size,) if one_d else (size, size)
    rng = np.random.default_rng(int(seed))
    s0 = (rng.random(shape) < start_occ).astype(np.uint8)
    e0 = rng.integers(0, ke, shape).astype(np.uint8)
    ss.live = {"s": s0, "e": e0, "rng": np.random.default_rng(int(seed))}
    ss.null = {"s": s0.copy(), "e": e0.copy(), "rng": np.random.default_rng(int(seed) + 1)}
    ss.gen = 0
    ss.recent = {"live": [(s0, e0)], "null": [(s0, e0)]}
    ss.occupancy = {"Coupled": [s0.mean()], "Links cut (null)": [s0.mean()]}


params = (lattice_name, size, ke, env_period, start_occ, int(seed),
          rule.species_table.tobytes(), rule.env_table.tobytes(), s_mode, e_mode)
if ss.get("params2") != params:
    ss.params2 = params
    reset()
ss.setdefault("running2", False)


def advance(steps: int) -> None:
    keep = max(MI_WINDOW, rows_shown or 0) + 1
    for _ in range(steps):
        update_env = (ss.gen + 1) % env_period == 0
        for name, links in (("live", (s_mode, e_mode)), ("null", ("shuffled", "shuffled"))):
            world = getattr(ss, name)
            world["s"], world["e"] = step_coupled(
                world["s"], world["e"], rule, offsets, update_env=update_env,
                species_sees_env=links[0], env_sees_species=links[1], rng=world["rng"])
            ss.recent[name].append((world["s"], world["e"]))
            del ss.recent[name][:-keep]
        ss.gen += 1
        ss.occupancy["Coupled"].append(ss.live["s"].mean())
        ss.occupancy["Links cut (null)"].append(ss.null["s"].mean())
    for series in ss.occupancy.values():
        del series[:-CHART_LENGTH]


def toggle() -> None:
    ss.running2 = not ss.running2


left, right = st.columns([3, 1])
with left:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 2])
    c1.button("Pause" if ss.running2 else "Play", on_click=toggle, type="primary", key="play2")
    c2.button("Step", on_click=lambda: advance(ss.get("per_frame2", 1)), disabled=ss.running2,
              key="step2")
    c3.button("Reset", on_click=reset, key="reset2")
    c4.slider("Generations per frame", 1, 50, 1, key="per_frame2")
    fps = c5.slider("Frames per second", 1, 30, 10, key="fps2")
    view = st.radio("Show", ["Side by side", "Overlay"], horizontal=True)

env_colours = ramp(GRADIENTS[gradient], ke)
occupied_rgb = categorical(np.zeros((1, 1), np.uint8), [species_colour])[0, 0]


def layer_images(s, e):
    env_rgb = env_colours[e]
    sp_rgb = np.where(s[..., None] == 1, occupied_rgb, np.uint8(245)).astype(np.uint8)
    overlay = np.where(s[..., None] == 1, occupied_rgb, env_rgb).astype(np.uint8)
    return sp_rgb, env_rgb, overlay


def render(rgb):
    if lattice_name == "Hexagonal":
        return render_hex(rgb, cell_px)
    return upscale(rgb, cell_px)


def interaction(name):
    pairs = ss.recent[name][-MI_WINDOW:]
    if len(pairs) < 3:
        return None, None
    s = np.stack([p[0] for p in pairs])
    e = np.stack([p[1] for p in pairs])
    return layer_mutual_information(e, s), layer_mutual_information(s, e)


@st.fragment(run_every=(1 / fps) if ss.running2 else None)
def live() -> None:
    if ss.running2:
        advance(ss.per_frame2)
    main, side = st.columns([3, 1])
    with main:
        st.markdown(f"Generation **{ss.gen}**" + (" (running)" if ss.running2 else ""))
        if one_d:
            pairs = ss.recent["live"][-rows_shown:]
            s = np.stack([p[0] for p in pairs])
            e = np.stack([p[1] for p in pairs])
            sp, env, over = layer_images(s, e)
            pad = rows_shown - s.shape[0]
            if pad:
                blank = np.full((pad, size, 3), 255, np.uint8)
                sp, env, over = (np.vstack([x, blank]) for x in (sp, env, over))
            caption = "Time runs down, newest generation at the bottom."
        else:
            sp, env, over = layer_images(ss.live["s"], ss.live["e"])
            caption = f"Generation {ss.gen}."
        if view == "Overlay":
            st.image(render(over), caption="Species over environment. " + caption)
        else:
            a, b = st.columns(2)
            a.image(render(sp), caption="Species. " + caption)
            b.image(render(env), caption="Environment. " + caption)
        st.line_chart(pd.DataFrame(ss.occupancy), height=180, x_label="Recent generations",
                      y_label="Occupancy")
    with side:
        st.subheader("Interaction")
        live_e2s, live_s2e = interaction("live")
        null_e2s, null_s2e = interaction("null")
        if live_e2s is None:
            st.caption("Run a few generations to measure.")
        else:
            st.metric("Environment -> species (bits)", f"{live_e2s:.3f}",
                      f"null {null_e2s:.3f}", delta_color="off")
            st.metric("Species -> environment (bits)", f"{live_s2e:.3f}",
                      f"null {null_s2e:.3f}", delta_color="off")
        st.metric("Occupancy", f"{ss.live['s'].mean():.3f}",
                  f"null {ss.null['s'].mean():.3f}", delta_color="off")
        st.caption(f"Bits: how much one layer at a cell tells you about the other layer at "
                   f"that cell one step later, over the last {MI_WINDOW} generations.")


with left:
    live()

with right, st.expander("Rule tables"):
    st.write("Species: rows own state, columns occupied neighbours, one block per "
             "environment level")
    for level in range(rule.ke):
        st.text(f"env {level}:\n{rule.species_table[:, level, :]}")
    st.write("Environment: rows own level, columns neighbour-level sum")
    for occ_state in (0, 1):
        st.text(f"{'occupied' if occ_state else 'empty'}:\n{rule.env_table[:, occ_state, :]}")
