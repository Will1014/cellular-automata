"""Recipe enumeration against hand counts and Gagrani et al. 2024, Table I."""

import numpy as np
import pytest

from cellular_automata.network import (
    balanced_recipes,
    network_structure,
    stoichiometric_matrix,
    with_empty,
)


def sizes(L):
    return [[s] for s in range(1, L + 1)]


def test_two_kinds_three_ingredients_has_one_recipe():
    comp = with_empty([[1, 0], [0, 1], [1, 1]])
    assert balanced_recipes(comp) == [((0, 1), (2, 3))]


def test_adding_a_double_unit_gives_three_recipes():
    comp = with_empty([[1, 0], [0, 1], [1, 1], [2, 0]])
    assert sorted(balanced_recipes(comp)) == sorted(
        [((0, 1), (2, 4)), ((0, 0), (3, 4)), ((0, 2), (1, 3))]
    )


def test_every_recipe_conserves_matter_and_slots():
    rng = np.random.default_rng(0)
    comp = with_empty(rng.integers(0, 3, size=(6, 2)) + np.array([[1, 0]]))
    recipes = balanced_recipes(comp)
    assert recipes
    S = stoichiometric_matrix(len(comp), recipes)
    assert not (comp.T @ S).any()
    assert not S.sum(axis=0).any()


def test_last_row_must_be_empty():
    with pytest.raises(ValueError):
        balanced_recipes([[1], [2]])


# Gagrani et al. 2024, Table I: L, directed reactions, complexes, linkage
# classes, deficiency.
TABLE_I = [(3, 6, 6, 3, 1), (4, 14, 11, 5, 3), (5, 26, 17, 7, 6), (6, 44, 24, 9, 10),
           (7, 68, 32, 11, 15)]


@pytest.mark.parametrize("L, reactions, complexes, linkage, deficiency", TABLE_I)
def test_reference_form_matches_gagrani_table_i(L, reactions, complexes, linkage, deficiency):
    recipes = balanced_recipes(sizes(L), empty=False)
    assert 2 * len(recipes) == reactions
    s = network_structure(L, recipes)
    assert (s["complexes"], s["linkage_classes"], s["deficiency"]) == (
        complexes, linkage, deficiency)


@pytest.mark.parametrize("L", [3, 4, 5, 6, 7])
def test_empty_space_maps_one_to_one_onto_the_reference_form(L):
    assert len(balanced_recipes(with_empty(sizes(L)))) == len(
        balanced_recipes(sizes(L), empty=False))
