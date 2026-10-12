"""A small deterministic lattice of patches, built only to check the
structural screen's carryover from patch to lattice (phase 2). The real
lattice, stochastic with capacity K, is phase 4.

Every cell runs the world's recipes, drive and decay, the same in each
cell. Movement is hop into empty space: ingredient i moves from a cell to
each neighbour at rate move[i] times its amount here times the empty space
there, equally fast in every direction. The grid wraps around."""

import numpy as np
from scipy.integrate import solve_ivp

from cellular_automata.lattice import HEX_6
from cellular_automata.patch import World, rate_of_change


def lattice_rate_of_change(world: World, move, shape: tuple[int, int], offsets=HEX_6):
    """Equations of the lattice on flattened amounts, shape (n * cells,)."""
    f = rate_of_change(world)
    move = np.asarray(move, dtype=float)[:-1, None, None]
    n = world.n

    def rhs(_t, flat):
        x = flat.reshape((n,) + shape)
        dx = f(x)
        matter, empty = x[:-1], x[-1]
        for o in offsets:
            there = np.roll(empty, shift=(-o[0], -o[1]), axis=(0, 1))
            back = np.roll(matter, shift=(-o[0], -o[1]), axis=(1, 2))
            # Out of each cell to its neighbour at offset o, and in from it.
            hop = move * (back * empty - matter * there)
            dx[:-1] += hop
            dx[-1] -= hop.sum(axis=0)
        return dx.ravel()

    return rhs


def run_lattice(world: World, move, x0, t_end: float, offsets=HEX_6):
    """Integrate from `x0`, shape (n, rows, cols). Returns the final amounts."""
    x0 = np.asarray(x0, dtype=float)
    rhs = lattice_rate_of_change(world, move, x0.shape[1:], offsets)
    sol = solve_ivp(rhs, (0.0, t_end), x0.ravel(), method="LSODA", t_eval=[t_end],
                    rtol=1e-8, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(x0.shape)
