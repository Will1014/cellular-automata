"""Reaction-network worlds: ingredients with compositions, and every recipe
that conserves matter (plan section 2).

Ingredients are rows of a composition matrix over kinds of matter. Empty
space, when present, is the last row and carries no matter. A recipe is a
pair of sides, each a sorted tuple of ingredient indices, listed once per
reversible pair."""

from collections import defaultdict
from itertools import combinations, combinations_with_replacement

import numpy as np

Side = tuple[int, ...]
Recipe = tuple[Side, Side]


def with_empty(compositions) -> np.ndarray:
    """Composition matrix with an empty-space row (all zeros) appended last."""
    comp = np.asarray(compositions, dtype=np.int64)
    return np.vstack([comp, np.zeros((1, comp.shape[1]), dtype=np.int64)])


def balanced_recipes(compositions, size: int = 2, empty: bool = True) -> list[Recipe]:
    """Every recipe of `size` slots in and `size` out that conserves each kind
    of matter, with distinct sides, once per reversible pair.

    With `empty`, the last row of `compositions` is empty space and every side
    has exactly `size` slots, so slots are conserved too. Without it (the
    reference form of Gagrani et al. 2024), sides have 1 to `size` ingredients."""
    comp = np.asarray(compositions, dtype=np.int64)
    n = comp.shape[0]
    if empty:
        if comp[-1].any():
            raise ValueError("with empty=True the last row must be empty space (all zeros)")
        sides = combinations_with_replacement(range(n), size)
    else:
        sides = (s for k in range(1, size + 1) for s in combinations_with_replacement(range(n), k))
    by_total: dict[tuple, list[Side]] = defaultdict(list)
    for side in sides:
        total = comp[list(side)].sum(axis=0)
        if total.any():
            by_total[tuple(total)].append(side)
    return [pair for group in by_total.values() for pair in combinations(group, 2)]


def stoichiometric_matrix(n: int, recipes: list[Recipe]) -> np.ndarray:
    """Net change in each of `n` ingredients (rows) when each recipe (columns)
    runs forward once."""
    S = np.zeros((n, len(recipes)), dtype=np.int64)
    for j, (left, right) in enumerate(recipes):
        for i in left:
            S[i, j] -= 1
        for i in right:
            S[i, j] += 1
    return S


def network_structure(n: int, recipes: list[Recipe]) -> dict:
    """Complexes, linkage classes and deficiency (Feinberg): deficiency is
    complexes minus linkage classes minus the rank of the stoichiometric matrix."""
    index: dict[Side, int] = {}
    for pair in recipes:
        for side in pair:
            index.setdefault(side, len(index))
    parent = list(range(len(index)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for left, right in recipes:
        parent[root(index[left])] = root(index[right])
    linkage = len({root(i) for i in range(len(index))})
    rank = int(np.linalg.matrix_rank(stoichiometric_matrix(n, recipes))) if recipes else 0
    return {
        "complexes": len(index),
        "linkage_classes": linkage,
        "rank": rank,
        "deficiency": len(index) - linkage - rank,
    }
