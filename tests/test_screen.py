"""Structural screen against the hand-worked worlds in
docs/2026-10-12_phase2-tests.md (Drive folder)."""

from itertools import combinations

import numpy as np
import pytest

from cellular_automata.network import balanced_recipes, with_empty
from cellular_automata.patch import World, driving_forces
from cellular_automata.screen import structure, verdict

# A = (1,0), B = (0,1), C = (1,1), then empty space: A + B <-> C + empty.
COMP = with_empty([[1, 0], [0, 1], [1, 1]])
A, B, C, E = range(4)


def abc_world(**kw):
    recipes = balanced_recipes(COMP)
    return World(recipes, np.array([0.3, -0.2, 0.1, 0.0]), np.ones(len(recipes)), **kw)


def test_undriven_world_is_deficiency_zero_and_set_aside():
    s = structure(abc_world())
    assert (s["complexes"], s["linkage_classes"], s["rank"], s["deficiency"]) == (2, 1, 1, 0)
    assert s["weakly_reversible"]
    v = verdict(abc_world())
    assert v["verdict"] == "set aside"
    assert v["lattice_conditions"]


def test_feed_and_waste_breaking_both_laws_is_undriven():
    # Feeding A and taking C out breaks both matter laws, so there is no
    # driving force and the world sits at equilibrium whatever the potentials.
    for feed in (0.0, 3.0, 30.0):
        world = abc_world(exchange={A: (feed, 1.0), C: (-feed, 1.0)})
        s = structure(world)
        # A + B, C + empty; then A, empty, C joined by the two exchanges.
        assert (s["complexes"], s["linkage_classes"], s["rank"], s["deficiency"]) == (5, 2, 3, 0)
        v = verdict(world)
        assert (v["verdict"], v["theorem"]) == ("set aside", "detailed balance")


@pytest.mark.parametrize("comp", [
    with_empty([[1, 0], [0, 1], [1, 1]]),
    with_empty([[1, 0], [0, 1], [1, 1], [2, 0]]),
    with_empty([[1, 0], [0, 1], [1, 1], [1, 1]]),
])
def test_no_deficiency_zero_world_has_a_driving_force(comp):
    # Exchanges join single slots to empty space and recipes join pairs of
    # slots, so a loop through the drive is never a loop of recipes. With
    # deficiency zero every loop is a loop of recipes, so none is driven.
    recipes, n = balanced_recipes(comp), len(comp)
    for k in range(1, len(recipes) + 1):
        for mask in combinations(recipes, k):
            for h in range(n):
                for held in combinations(range(n - 1), h):
                    world = World(list(mask), np.zeros(n), np.ones(k),
                                  exchange={i: (0.0, 1.0) for i in held})
                    if structure(world)["deficiency"] == 0:
                        assert driving_forces(world) == 0


def test_decay_makes_it_run_it():
    world = abc_world(exchange={A: (3.0, 1.0), C: (-3.0, 1.0)}, decay=0.01)
    s = structure(world)
    # Nothing turns empty space back into B, so B -> empty is on no loop.
    assert not s["weakly_reversible"]
    assert verdict(world)["verdict"] == "run it"


def test_decay_on_reservoir_ingredients_only_keeps_weak_reversibility():
    world = World([], np.array([0.0, 0.0, 0.0, 0.0]), np.ones(0),
                  exchange={A: (1.0, 1.0), B: (0.0, 1.0), C: (0.0, 1.0)}, decay=0.01)
    assert structure(world)["weakly_reversible"]
