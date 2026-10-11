"""Two coupled layers on the same lattice: a species layer and an environment.

Species: 0 empty, 1 occupied.
Environment: levels 0 .. ke-1 (0 is poorest).

Each layer has its own rule, a lookup table over a small set of situations:

    species_table[own_species, own_env, occupied_neighbours]       -> 0/1
    env_table[own_env, own_species, sum_of_neighbour_env_levels]   -> 0..ke-1

That gives four channels: species-species, env-env, env->species and
species->env. Either cross-layer channel can be cut for a null comparison:

    "on"        the layer sees the other layer as it is
    "shuffled"  it sees a spatially shuffled copy (same amounts, no positions)
    "constant"  it sees a fixed value everywhere
"""

from dataclasses import dataclass

import numpy as np

from .lattice import neighbour_sum


@dataclass
class CoupledRule:
    species_table: np.ndarray  # shape (2, ke, n + 1)
    env_table: np.ndarray  # shape (ke, 2, n * (ke - 1) + 1)

    @property
    def ke(self) -> int:
        return self.env_table.shape[0]

    @property
    def n_neighbours(self) -> int:
        return self.species_table.shape[2] - 1

    def validate(self) -> None:
        ke, n = self.ke, self.n_neighbours
        if self.species_table.shape != (2, ke, n + 1):
            raise ValueError(f"species_table shape {self.species_table.shape}")
        if self.env_table.shape != (ke, 2, n * (ke - 1) + 1):
            raise ValueError(f"env_table shape {self.env_table.shape}")
        if self.species_table.max() > 1 or self.env_table.max() >= ke:
            raise ValueError("table output out of range")


def table_shapes(ke: int, n: int) -> tuple[tuple, tuple]:
    return (2, ke, n + 1), (ke, 2, n * (ke - 1) + 1)


def random_coupled_rule(ke: int, n: int, rng: np.random.Generator) -> CoupledRule:
    s_shape, e_shape = table_shapes(ke, n)
    return CoupledRule(rng.integers(0, 2, s_shape, dtype=np.uint8),
                       rng.integers(0, ke, e_shape, dtype=np.uint8))


def mutate_coupled(rule: CoupledRule, rng: np.random.Generator, rate: float) -> CoupledRule:
    s = rule.species_table.copy()
    hit = rng.random(s.shape) < rate
    s[hit] ^= 1
    e = rule.env_table.copy()
    hit = rng.random(e.shape) < rate
    e[hit] = (e[hit] + rng.integers(1, rule.ke, int(hit.sum()), dtype=np.uint8)) % rule.ke
    return CoupledRule(s, e)


def consumer_resource(ke: int, n: int, colonise_at: int = 1, regrow_at: int = 1) -> CoupledRule:
    """Hand-written example. Occupied cells use up their ground one level per
    environment update; empty ground regrows one level when its neighbours'
    levels sum to at least `regrow_at`; a species colonises good ground (top
    level) next to `colonise_at` occupied cells, and dies on the poorest ground."""
    s_shape, e_shape = table_shapes(ke, n)
    s = np.zeros(s_shape, dtype=np.uint8)
    for e in range(ke):
        for c in range(n + 1):
            s[1, e, c] = 1 if e > 0 else 0  # survive unless the ground is exhausted
            s[0, e, c] = 1 if (e == ke - 1 and c >= colonise_at) else 0
    env = np.zeros(e_shape, dtype=np.uint8)
    for e in range(ke):
        for total in range(e_shape[2]):
            env[e, 1, total] = max(e - 1, 0)  # consumed
            env[e, 0, total] = min(e + 1, ke - 1) if total >= regrow_at else e
    return CoupledRule(s, env)


def _cross(layer: np.ndarray, mode: str, rng: np.random.Generator, constant: int) -> np.ndarray:
    if mode == "on":
        return layer
    if mode == "shuffled":
        return rng.permutation(layer.ravel()).reshape(layer.shape)
    if mode == "constant":
        return np.full_like(layer, constant)
    raise ValueError(f"unknown coupling mode {mode!r}")


def step_coupled(species, env, rule: CoupledRule, offsets, update_env: bool = True,
                 species_sees_env: str = "on", env_sees_species: str = "on",
                 rng: np.random.Generator | None = None, constant_env: int = 0,
                 constant_species: int = 0):
    """One synchronous step of both layers. Both read the state before the step."""
    rng = rng if rng is not None else np.random.default_rng(0)
    occupied = neighbour_sum(species, offsets)
    env_seen = _cross(env, species_sees_env, rng, constant_env)
    new_species = rule.species_table[species, env_seen, occupied]
    if not update_env:
        return new_species, env
    env_total = neighbour_sum(env, offsets)
    species_seen = _cross(species, env_sees_species, rng, constant_species)
    new_env = rule.env_table[env, species_seen, env_total]
    return new_species, new_env


def run_coupled(species0, env0, rule: CoupledRule, offsets, steps: int, env_period: int = 1,
                species_sees_env: str = "on", env_sees_species: str = "on", seed: int = 0):
    """Histories of both layers, each shaped (steps + 1, *grid_shape).
    The environment updates on steps that are multiples of env_period."""
    rule.validate()
    if len(offsets) != rule.n_neighbours:
        raise ValueError(f"rule expects {rule.n_neighbours} neighbours, lattice has {len(offsets)}")
    rng = np.random.default_rng(seed)
    s_hist = np.empty((steps + 1, *species0.shape), dtype=np.uint8)
    e_hist = np.empty_like(s_hist)
    s_hist[0], e_hist[0] = species0, env0
    for t in range(steps):
        s_hist[t + 1], e_hist[t + 1] = step_coupled(
            s_hist[t], e_hist[t], rule, offsets, update_env=(t + 1) % env_period == 0,
            species_sees_env=species_sees_env, env_sees_species=env_sees_species, rng=rng)
    return s_hist, e_hist


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information (bits) between two equally shaped integer arrays,
    pairing values at the same position. 0 when independent."""
    a = x.ravel().astype(np.int64)
    b = y.ravel().astype(np.int64)
    kb = int(b.max()) + 1
    ka = int(a.max()) + 1
    joint = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb) / a.size
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())


def layer_mutual_information(source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
    """How much a cell's value in one layer tells you about the same cell in
    the other layer `lag` steps later. Histories are shaped (time, *grid).
    Lagged because each layer responds to the other's previous state; at
    lag 0 a feedback loop can read as no association at all."""
    if lag == 0:
        return mutual_information(source, target)
    return mutual_information(source[:-lag], target[lag:])
