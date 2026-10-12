"""The species reader: which self-copying cores can grow from rare, and
which the run actually holds (plan section 3)."""

import numpy as np
from scipy.optimize import linprog

from cellular_automata.patch import World, rate_of_change, run_patch
from cellular_automata.screen import reactions


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


def is_organisation(world: World, members) -> bool:
    """Whether `members` form a chemical organisation (Dittrich and Speroni di
    Fenizio 2007): closed, so every reaction that can run on them alone makes
    nothing outside them, and self-maintaining, so those reactions can all run
    at once, each at a positive rate, without any member running down."""
    g = set(members)
    usable = [(a, b) for a, b in reactions(world) if set(a) <= g]
    if any(not set(b) <= g for _, b in usable):
        return False
    if not usable:
        return True
    rows = sorted(g)
    S = np.zeros((len(rows), len(usable)))
    for j, (a, b) in enumerate(usable):
        for i in a:
            S[rows.index(i), j] -= 1
        for i in b:
            S[rows.index(i), j] += 1
    # Rates of at least 1 (any positive rates rescale to this), S v >= 0.
    res = linprog(np.zeros(len(usable)), A_ub=-S, b_ub=np.zeros(len(rows)),
                  bounds=[(1, None)] * len(usable), method="highs")
    return res.status == 0
