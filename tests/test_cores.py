"""Minimal self-copying sets against Gagrani et al. 2024, Table I. L = 5 to 7
run as jobs (scripts/count_cores.py); they take a minute to hours."""

from collections import Counter

import pytest

from cellular_automata.cores import directed, minimal_autocatalytic_subnetworks
from cellular_automata.network import balanced_recipes


def reference(L):
    return directed(L, balanced_recipes([[s] for s in range(1, L + 1)], empty=False))


@pytest.mark.parametrize("L, by_size", [(3, {2: 2}), (4, {2: 9, 3: 11})])
def test_counts_match_table_i(L, by_size):
    found = minimal_autocatalytic_subnetworks(*reference(L))
    assert dict(Counter(len(r) for r, _ in found)) == by_size


def test_l3_sets_are_the_two_printed_in_the_paper():
    # Recipes at L = 3: 0: 2 <-> 1+1, 1: 3 <-> 1+2, 2: 1+3 <-> 2+2; column j + 3
    # runs recipe j backwards. A1 = {1+2 -> 3, 1+3 -> 2+2}, A2 = {2 -> 1+1, 1+3 -> 2+2}.
    found = {r: c for r, c in minimal_autocatalytic_subnetworks(*reference(3))}
    assert found == {frozenset({4, 2}): frozenset({1, 2}), frozenset({0, 2}): frozenset({0, 1})}


def test_no_set_contains_another():
    sets = [r for r, _ in minimal_autocatalytic_subnetworks(*reference(4))]
    assert not any(a < b for a in sets for b in sets)


def test_resuming_from_part_of_a_run_finds_the_rest():
    full = [r for r, _ in minimal_autocatalytic_subnetworks(*reference(4))]
    rest = [r for r, _ in minimal_autocatalytic_subnetworks(*reference(4), known=full[:7])]
    assert set(full[:7]) | set(rest) == set(full)
    assert len(rest) == len(full) - 7
