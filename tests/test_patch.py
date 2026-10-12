"""Deterministic patch: detailed balance when closed, drive when not
(docs/2026-10-12_phase1-tests.md in the Drive folder)."""

import numpy as np
import pytest

from cellular_automata.network import balanced_recipes, stoichiometric_matrix, with_empty
from cellular_automata.patch import World, driving_forces, entropy_production, run_patch


def random_world(seed, comp, **kw):
    rng = np.random.default_rng(seed)
    recipes = balanced_recipes(comp)
    energy = np.append(rng.normal(0, 1, len(comp) - 1), 0.0)
    return World(recipes, energy, rng.uniform(0, 2, len(recipes)), **kw), rng


def random_start(rng, n):
    x = rng.uniform(0.01, 0.1, n)
    x[-1] = 0.0
    x[-1] = 1.0 - x.sum()
    return x


COMP = with_empty([[1, 0], [0, 1], [1, 1], [2, 0], [2, 1], [1, 2]])


@pytest.mark.parametrize("seed", range(4))
def test_closed_world_relaxes_and_conserves(seed):
    world, rng = random_world(seed, COMP)
    for _ in range(2):
        x0 = random_start(rng, world.n)
        _, xs = run_patch(world, x0, t_end=2000.0)
        ep = np.array([entropy_production(world, x) for x in xs])
        assert np.all(np.diff(ep) <= 1e-12 * ep[0])
        assert ep[-1] < 1e-9 * ep[0]
        assert np.abs(xs @ COMP - x0 @ COMP).max() < 1e-9
        assert np.abs(xs.sum(axis=1) - 1.0).max() < 1e-9


def test_closed_world_settles_where_detailed_balance_says():
    world, rng = random_world(7, COMP)
    _, xs = run_patch(world, random_start(rng, world.n), t_end=2000.0)
    jf_over_jb = []
    kf, kb = world.rates()
    x = xs[-1]
    for (a, b), f, r in zip(world.recipes, kf, kb, strict=True):
        jf_over_jb.append(f * np.prod(x[list(a)]) / (r * np.prod(x[list(b)])))
    assert np.allclose(jf_over_jb, 1.0, atol=1e-6)


# Feed F and waste W are isomers (same composition), joined by every recipe
# that turns one into the other, catalysed or not.
ISOMERS = with_empty([[1], [1]])


def isomer_world(mu_feed, mu_waste):
    recipes = balanced_recipes(ISOMERS)
    return World(recipes, np.array([0.5, -0.5, 0.0]), np.ones(len(recipes)),
                 exchange={0: (mu_feed, 0.5), 1: (mu_waste, 0.5)})


def test_isomer_feed_and_waste_give_one_force():
    assert driving_forces(isomer_world(1.0, -1.0)) == 1


def test_one_kind_different_sizes_gives_one_force():
    comp = with_empty([[1], [2]])
    world = World(balanced_recipes(comp), np.array([0.0, 0.3, 0.0]), np.array([1.0]),
                  exchange={1: (1.0, 0.5), 0: (-1.0, 0.5)})
    assert driving_forces(world) == 1


def test_feed_and_waste_of_different_kinds_give_no_force():
    comp = with_empty([[1, 0], [0, 1], [1, 1]])
    world = World(balanced_recipes(comp), np.array([0.0, 0.0, -0.5, 0.0]), np.array([1.0]),
                  exchange={0: (1.0, 0.5), 1: (-1.0, 0.5)})
    assert driving_forces(world) == 0
    _, xs = run_patch(world, [0.1, 0.1, 0.1, 0.7], t_end=2000.0)
    assert entropy_production(world, xs[-1]) < 1e-9


def test_unequal_reservoirs_keep_producing_entropy():
    world = isomer_world(1.0, -1.0)
    _, xs = run_patch(world, [0.1, 0.1, 0.8], t_end=2000.0)
    late = [entropy_production(world, x) for x in xs[-20:]]
    assert min(late) > 1e-3
    assert np.ptp(late) < 1e-9


def test_equal_reservoirs_relax_to_zero():
    world = isomer_world(0.5, 0.5)
    _, xs = run_patch(world, [0.1, 0.1, 0.8], t_end=2000.0)
    assert entropy_production(world, xs[-1]) < 1e-9


def test_decay_removes_matter_at_its_rate():
    comp = with_empty([[1], [2]])
    world = World(balanced_recipes(comp), np.array([0.0, 0.3, 0.0]), np.array([1.0]), decay=0.01)
    x0 = np.array([0.2, 0.1, 0.7])
    t, xs = run_patch(world, x0, t_end=100.0)
    filled = xs[:, :-1].sum(axis=1)
    # The split 2 + empty <-> 1 + 1 changes the number filled, so test total
    # matter instead: every unit decays at the same rate.
    matter = xs @ comp[:, 0]
    assert np.allclose(matter, (x0 @ comp[:, 0]) * np.exp(-0.01 * t), rtol=1e-7)
    assert np.all(filled > 0)
    assert np.abs(xs.sum(axis=1) - 1.0).max() < 1e-9
    assert stoichiometric_matrix(3, world.recipes).sum() == 0
