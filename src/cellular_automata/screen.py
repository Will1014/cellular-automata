"""The structural screen: what can be said about a world before running it
(plan section 3).

The screen is sound but incomplete. A world is set aside only where a
theorem allows it, with each of the theorem's conditions checked and
recorded; every other world gets "run it".

The world's full network includes the drive: each exchange is empty space
<-> the held ingredient, run both ways, and decay is ingredient -> empty
space, one way only (decision 4)."""

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cellular_automata.network import Side
from cellular_automata.patch import World, driving_forces

# What has to hold on the lattice for a patch label to carry over to it
# (docs/2026-10-12_phase2-tests.md). The patch can't check these; the
# lattice records them when it is built.
LATTICE_CONDITIONS = (
    "movement equally fast both ways between neighbours",
    "the same drive in every cell",
    "no recipe acts between neighbouring cells",
)


def reactions(world: World) -> list[tuple[Side, Side]]:
    """Every one-way reaction of the world: each recipe and exchange both
    ways, then decay of every non-empty ingredient when it is switched on."""
    empty = world.n - 1
    out = []
    for left, right in world.recipes:
        out += [(left, right), (right, left)]
    for i in world.exchange:
        out += [((empty,), (i,)), ((i,), (empty,))]
    if world.decay:
        out += [((i,), (empty,)) for i in range(empty)]
    return list(dict.fromkeys(out))


def structure(world: World) -> dict:
    """Complexes, linkage classes, rank, deficiency and weak reversibility of
    the world's full network. Weakly reversible means every reaction lies on
    a loop of reactions that leads back to where it started."""
    rxns = reactions(world)
    index: dict[Side, int] = {}
    for pair in rxns:
        for side in pair:
            index.setdefault(side, len(index))
    m = len(index)
    src = [index[a] for a, _ in rxns]
    dst = [index[b] for _, b in rxns]
    graph = coo_matrix((np.ones(len(rxns)), (src, dst)), shape=(m, m))
    linkage, _ = connected_components(graph, directed=True, connection="weak")
    _, strong = connected_components(graph, directed=True, connection="strong")
    S = np.zeros((world.n, len(rxns)))
    for j, (left, right) in enumerate(rxns):
        for i in left:
            S[i, j] -= 1
        for i in right:
            S[i, j] += 1
    rank = int(np.linalg.matrix_rank(S)) if rxns else 0
    return {
        "complexes": m,
        "linkage_classes": int(linkage),
        "rank": rank,
        "deficiency": m - int(linkage) - rank,
        "weakly_reversible": bool(all(strong[s] == strong[d] for s, d in zip(src, dst))),
    }


def verdict(world: World) -> dict:
    """"set aside" with the theorem and its checked conditions, or "run it".

    Two theorems apply, each giving one settled state per set of totals and
    nothing else, deterministically and (via the product-form distribution)
    stochastically:
    - detailed balance: no decay and no driving force, so the energies and
      reservoir potentials balance every recipe
    - deficiency zero: weakly reversible with deficiency 0, which makes the
      world complex balanced for any rates and any drive (Feinberg)."""
    s = structure(world)
    forces = driving_forces(world)
    checked = {
        "decay": world.decay,
        "driving_forces": forces,
        "weakly_reversible": s["weakly_reversible"],
        "deficiency": s["deficiency"],
    }
    if not world.decay and forces == 0:
        theorem = "detailed balance"
    elif s["weakly_reversible"] and s["deficiency"] == 0:
        theorem = "deficiency zero"
    else:
        return {"verdict": "run it", "theorem": None, "checked": checked, **s}
    return {"verdict": "set aside", "theorem": theorem, "checked": checked,
            "lattice_conditions": list(LATTICE_CONDITIONS), **s}


def siphons(world: World) -> list[tuple[int, ...]]:
    """Minimal siphons: groups of ingredients that, once all gone, nothing can
    make again, because every reaction making one of them uses one of them
    up. Found by checking every group, smallest first, which is fine at the
    sizes enumerated here (plan section 5)."""
    rxns = [(set(a), set(b)) for a, b in reactions(world)]
    found: list[tuple[int, ...]] = []
    for size in range(1, world.n + 1):
        for group in combinations(range(world.n), size):
            g = set(group)
            if any(g.issuperset(f) for f in found):
                continue
            if all(left & g for left, right in rxns if right & g):
                found.append(group)
    return found
