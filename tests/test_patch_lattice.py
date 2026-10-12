"""Small deterministic lattice of patches: conserves what it should, and a
uniform start matches the patch."""

import numpy as np

from cellular_automata.network import balanced_recipes, with_empty
from cellular_automata.patch import World, run_patch
from cellular_automata.patch_lattice import run_lattice

COMP = with_empty([[1, 0], [0, 1], [1, 1], [2, 0]])


def world_and_start(seed, shape=(4, 4), **kw):
    rng = np.random.default_rng(seed)
    recipes = balanced_recipes(COMP)
    world = World(recipes, np.append(rng.normal(0, 1, 4), 0.0), rng.uniform(0, 2, len(recipes)), **kw)
    x = rng.uniform(0.01, 0.1, (5,) + shape)
    x[-1] = 1.0 - x[:-1].sum(axis=0)
    return world, x, rng.uniform(0.1, 1.0, 5)


def test_closed_lattice_conserves_matter_and_each_cells_slots():
    world, x0, move = world_and_start(0)
    x = run_lattice(world, move, x0, t_end=50.0)
    assert np.abs(x.sum(axis=0) - 1.0).max() < 1e-9
    total = np.einsum("irc,ik->k", x, COMP)
    assert np.abs(total - np.einsum("irc,ik->k", x0, COMP)).max() < 1e-7


def test_uniform_start_matches_the_patch():
    world, x0, move = world_and_start(1, exchange={0: (1.0, 1.0), 3: (-1.0, 1.0)}, decay=0.01)
    x0[:] = x0[:, :1, :1]
    x = run_lattice(world, move, x0, t_end=50.0)
    _, xs = run_patch(world, x0[:, 0, 0], t_end=50.0, n_out=2)
    assert np.abs(x - xs[-1][:, None, None]).max() < 1e-6


def test_movement_alone_spreads_a_lump_evenly():
    # No recipes: one ingredient in one cell must spread to the same amount
    # everywhere, and never go negative on the way.
    world = World([], np.array([0.0, 0.0]), np.ones(0))
    x0 = np.zeros((2, 4, 4))
    x0[0, 0, 0] = 0.8
    x0[1] = 1.0 - x0[0]
    x = run_lattice(world, [1.0, 0.0], x0, t_end=200.0)
    assert np.abs(x[0] - 0.8 / 16).max() < 1e-6
    assert x.min() > -1e-12
