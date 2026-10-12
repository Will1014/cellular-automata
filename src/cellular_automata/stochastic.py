"""Stochastic 1D automata, run as many independent periodic lattices at once."""

import numpy as np


def directed_percolation(
    initial: np.ndarray, p: float, steps: int, rng: np.random.Generator
) -> np.ndarray:
    """Site directed percolation (Domany-Kinzel with p1 = p2 = p). A cell is on
    at the next step with probability p if it or its right-hand neighbour is
    on now, otherwise off. Using cells i and i+1 is the staggered lattice
    written on a straight one. `initial` has shape (runs, width); returns the
    final states. The critical point is p_c = 0.705485 (Jensen 1999, J. Phys. A 32:5233)."""
    state = np.asarray(initial, dtype=bool)
    for _ in range(steps):
        alive = state | np.roll(state, -1, axis=1)
        state = alive & (rng.random(state.shape) < p)
    return state.astype(np.uint8)
