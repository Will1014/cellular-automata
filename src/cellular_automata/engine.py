"""Run a 1D cellular automaton from a rule lookup table.

A rule is a lookup table indexed by neighbourhood. The neighbourhood is read
left to right as a base-k number, so the leftmost cell is the most
significant digit. For k=2, r=1 this is Wolfram's convention: neighbourhood
111 is index 7, and bit i of the rule number is the output for index i.

The lattice is a tuple of neighbour offsets, so a wider or asymmetric
neighbourhood is a different tuple, not different code.
"""

import numpy as np


def offsets_for_radius(r: int) -> tuple[int, ...]:
    return tuple(range(-r, r + 1))


def neighbourhood_index(
    state: np.ndarray, k: int, offsets: tuple[int, ...], boundary: str = "periodic"
) -> np.ndarray:
    """Lookup-table index of every cell's neighbourhood."""
    if boundary not in ("periodic", "fixed"):
        raise ValueError(f"unknown boundary {boundary!r}")
    pad = max(abs(o) for o in offsets)
    if boundary == "periodic":
        padded = np.concatenate([state[-pad:], state, state[:pad]]) if pad else state
    else:
        padded = np.pad(state, pad, constant_values=0)
    n = state.shape[0]
    idx = np.zeros(n, dtype=np.int64)
    for o in offsets:
        idx = idx * k + padded[pad + o : pad + o + n]
    return idx


def step(
    state: np.ndarray,
    table: np.ndarray,
    k: int,
    offsets: tuple[int, ...],
    boundary: str = "periodic",
) -> np.ndarray:
    return table[neighbourhood_index(state, k, offsets, boundary)]


def run(
    initial: np.ndarray,
    table: np.ndarray,
    k: int = 2,
    offsets: tuple[int, ...] = (-1, 0, 1),
    steps: int = 100,
    boundary: str = "periodic",
) -> np.ndarray:
    """Space-time history, shape (steps + 1, width). Row 0 is the initial state."""
    table = np.asarray(table, dtype=np.uint8)
    if table.shape[0] != k ** len(offsets):
        raise ValueError(f"table has {table.shape[0]} entries, expected {k ** len(offsets)}")
    history = np.empty((steps + 1, initial.shape[0]), dtype=np.uint8)
    history[0] = initial
    for t in range(steps):
        history[t + 1] = step(history[t], table, k, offsets, boundary)
    return history


def single_cell(width: int, k: int = 2) -> np.ndarray:
    state = np.zeros(width, dtype=np.uint8)
    state[width // 2] = 1 if k == 2 else k - 1
    return state


def random_state(width: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, k, size=width, dtype=np.uint8)


def final_states(
    initial: np.ndarray,
    table: np.ndarray,
    k: int = 2,
    offsets: tuple[int, ...] = (-1, 0, 1),
    steps: int = 100,
) -> np.ndarray:
    """Run many independent periodic lattices at once and return only their
    final states. `initial` has shape (runs, width); memory stays at one
    state per run, so thousands of runs fit where a full history would not."""
    table = np.asarray(table, dtype=np.uint8)
    if table.shape[0] != k ** len(offsets):
        raise ValueError(f"table has {table.shape[0]} entries, expected {k ** len(offsets)}")
    state = np.asarray(initial, dtype=np.uint8)
    for _ in range(steps):
        idx = np.zeros(state.shape, dtype=np.intp)
        for o in offsets:
            idx = idx * k + np.roll(state, -o, axis=1)
        state = table[idx]
    return state
