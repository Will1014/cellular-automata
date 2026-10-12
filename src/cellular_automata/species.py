"""The species reader: which self-copying cores can grow from rare, and
which the run actually holds (plan section 3)."""

import numpy as np

from cellular_automata.patch import World, rate_of_change, run_patch


def background(world: World, members, x0, t_end: float = 1e4) -> np.ndarray:
    """The settled state of the world with `members` removed from the start.
    Raises if they come back, since then they were never absent."""
    x = np.asarray(x0, dtype=float).copy()
    x[-1] += x[list(members)].sum()
    x[list(members)] = 0.0
    _, xs = run_patch(world, x, t_end, n_out=2)
    if xs[-1][list(members)].max() > 1e-12:
        raise ValueError("members reappear from the background, so it is not a background")
    return xs[-1]


def invasion_rate(world: World, members, x_background) -> float:
    """Growth rate of `members` from rare against `x_background`: the leading
    eigenvalue of the equations' sensitivity, restricted to the members, at
    the background. Above 0 means the core grows from rare (it is viable).

    The sensitivity is taken by complex step, exact for mass action."""
    f = rate_of_change(world)
    members = list(members)
    J = np.empty((len(members), len(members)))
    for col, j in enumerate(members):
        x = np.asarray(x_background, dtype=complex).copy()
        x[j] += 1e-30j
        J[:, col] = f(x)[members].imag / 1e-30
    return float(np.linalg.eigvals(J).real.max())
