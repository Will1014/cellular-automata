"""Minimal self-copying sets: the minimal autocatalytic subnetworks (MASs) of
Gagrani et al. 2024 (J. Phys. A 57:225601), found by repeatedly solving a
mixed-integer programme with SciPy's HiGHS solver.

Reactions here are one-way: a reversible recipe is two reactions."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from cellular_automata.network import Recipe


def directed(n: int, recipes: list[Recipe]) -> tuple[np.ndarray, np.ndarray]:
    """Input and output matrices (species by one-way reactions) with each
    recipe run forward, then each run back. Column j and j + len(recipes)
    are the two directions of recipe j."""
    r = len(recipes)
    s_in = np.zeros((n, 2 * r), dtype=np.int64)
    s_out = np.zeros((n, 2 * r), dtype=np.int64)
    for j, (left, right) in enumerate(recipes):
        for i in left:
            s_in[i, j] += 1
            s_out[i, j + r] += 1
        for i in right:
            s_out[i, j] += 1
            s_in[i, j + r] += 1
    return s_in, s_out


def minimal_autocatalytic_subnetworks(
    s_in: np.ndarray,
    s_out: np.ndarray,
    eps: float = 1e-3,
    limit: int | None = None,
    known: list[frozenset[int]] = (),
    on_found=None,
) -> list[tuple[frozenset[int], frozenset[int]]]:
    """Every MAS, as (reactions, core species). Each MILP finds the smallest
    reaction set holding an exclusively autocatalytic core (Theorem III.1):
    a flow on the chosen reactions, every chosen reaction carrying at least
    `eps`, under which every core species is made in net; every core species
    consumed by a chosen reaction; and every chosen reaction consuming and
    producing a core species. Each set found is then forbidden as a subset
    of later ones, so the sets come out in order of size and none contains
    another.

    `known` reaction sets, from an interrupted run, are forbidden from the
    start and not returned again. `on_found(reactions, core)` is called as
    each new set is found, so a long run can write as it goes."""
    n, R = s_in.shape
    S = (s_out - s_in).astype(float)
    eats = (s_in > 0).astype(float)
    makes = (s_out > 0).astype(float)
    # Variables: y (core species, n), z (chosen reactions, R), v (flows, R).
    rows, lo, hi = [], [], []

    def add(y=None, z=None, v=None, low=-np.inf, high=np.inf):
        row = np.zeros(n + 2 * R)
        if y is not None:
            row[:n] = y
        if z is not None:
            row[n:n + R] = z
        if v is not None:
            row[n + R:] = v
        rows.append(row)
        lo.append(low)
        hi.append(high)

    big = np.abs(S).sum(axis=1) + eps
    for i in range(n):
        # Made in net if in the core: S_i v - big_i y_i >= eps - big_i.
        add(y=-big[i] * np.eye(n)[i], v=S[i], low=eps - big[i])
        # Consumed by some chosen reaction if in the core.
        add(y=-np.eye(n)[i], z=eats[i], low=0.0)
    for r in range(R):
        unit = np.eye(R)[r]
        add(y=eats[:, r], z=-unit, low=0.0)   # eats a core species
        add(y=makes[:, r], z=-unit, low=0.0)  # makes a core species
        add(z=-unit, v=unit, high=0.0)        # v_r <= z_r
        add(z=-eps * unit, v=unit, low=0.0)   # v_r >= eps z_r
    add(y=np.ones(n), low=1.0)

    cost = np.concatenate([np.zeros(n), np.ones(R), np.zeros(R)])
    integrality = np.concatenate([np.ones(n + R), np.zeros(R)])
    bounds = Bounds(np.zeros(n + 2 * R), np.ones(n + 2 * R))
    for chosen in known:
        add(z=np.isin(np.arange(R), list(chosen)).astype(float), high=len(chosen) - 1.0)
    found = []
    while limit is None or len(found) < limit:
        res = milp(cost, constraints=LinearConstraint(np.array(rows), lo, hi),
                   integrality=integrality, bounds=bounds)
        if res.status == 2:  # infeasible: no set left that avoids those found
            break
        if res.status != 0:
            raise RuntimeError(f"MILP stopped without an answer: {res.message}")
        y = res.x[:n] > 0.5
        z = res.x[n:n + R] > 0.5
        chosen = frozenset(np.flatnonzero(z).tolist())
        core = frozenset(np.flatnonzero(y).tolist())
        found.append((chosen, core))
        if on_found is not None:
            on_found(chosen, core)
        add(z=z.astype(float), high=len(chosen) - 1.0)
    return found
