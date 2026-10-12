"""The deterministic well-mixed patch: rates from energies, the drive, and
entropy production (plan section 2).

Amounts are fractions of the patch's capacity, empty space included, so they
sum to 1. Energies are in units of kT. A recipe's barrier sits `barrier`
above its higher side, so forward and back rates are at most 1 and their
ratio is fixed by the energy difference: every recipe runs both ways, and
an undriven world satisfies detailed balance."""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space

from cellular_automata.network import Recipe, stoichiometric_matrix


@dataclass(frozen=True)
class World:
    """A world on a patch. The last ingredient is empty space, energy 0.

    `exchange` maps an ingredient to (reservoir potential, barrier): the
    ingredient swaps with empty space against a reservoir, the only recipes
    that break conservation of matter. `decay` is one-way, every ingredient
    to empty space at this rate (decision 4), and is left out of entropy
    production."""

    recipes: list[Recipe]
    energy: np.ndarray
    barrier: np.ndarray
    exchange: dict[int, tuple[float, float]] = field(default_factory=dict)
    decay: float = 0.0

    @property
    def n(self) -> int:
        return len(self.energy)

    def rates(self) -> tuple[np.ndarray, np.ndarray]:
        """Forward and back rate constants for each recipe."""
        g = np.asarray(self.energy, dtype=float)
        left = np.array([g[list(a)].sum() for a, _ in self.recipes])
        right = np.array([g[list(b)].sum() for _, b in self.recipes])
        top = np.maximum(left, right) + self.barrier
        return np.exp(left - top), np.exp(right - top)


def _flux_function(world: World):
    """Forward and back flux of every recipe, then every exchange, under mass
    action, as a function of the amounts. Rates and indices are fixed once."""
    kf, kb = world.rates()
    held = np.array(list(world.exchange), dtype=np.intp)
    mu = np.array([m for m, _ in world.exchange.values()])
    g = np.asarray(world.energy, dtype=float)[held]
    top = np.maximum(mu, g) + np.array([b for _, b in world.exchange.values()])
    kf = np.concatenate([kf, np.exp(mu - top)])
    kb = np.concatenate([kb, np.exp(g - top)])
    empty = world.n - 1
    # Each recipe side as two slots (or one); a missing slot reads 1.
    def slots(sides):
        width = max((len(s) for s in sides), default=1)
        out = np.full((len(sides), width), world.n, dtype=np.intp)
        for j, s in enumerate(sides):
            out[j, : len(s)] = s
        return out
    left = slots([a for a, _ in world.recipes] + [(empty,)] * len(held))
    right = slots([b for _, b in world.recipes] + [(int(i),) for i in held])

    def fluxes(x):
        padded = np.append(x, 1.0)
        return kf * padded[left].prod(axis=1), kb * padded[right].prod(axis=1)

    return fluxes


def _full_matrix(world: World) -> np.ndarray:
    """Stoichiometric matrix with one extra column per exchange (empty -> ingredient)."""
    S = stoichiometric_matrix(world.n, world.recipes)
    X = np.zeros((world.n, len(world.exchange)), dtype=np.int64)
    for j, i in enumerate(world.exchange):
        X[i, j], X[-1, j] = 1, -1
    return np.hstack([S, X]).astype(float)


def entropy_production(world: World, x: np.ndarray) -> float:
    """Sum over recipes and exchanges of (J+ - J-) ln(J+ / J-), in kT per unit
    time. Decay is excluded, as declared in the plan."""
    jf, jb = _flux_function(world)(np.asarray(x, dtype=float))
    ok = (jf > 0) & (jb > 0)
    return float(np.sum((jf[ok] - jb[ok]) * np.log(jf[ok] / jb[ok])))


def run_patch(world: World, x0, t_end: float, n_out: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the patch from `x0` to `t_end`. Returns times and amounts,
    shape (n_out, n)."""
    S = _full_matrix(world)
    fluxes = _flux_function(world)

    def rhs(_t, x):
        jf, jb = fluxes(x)
        dx = S @ (jf - jb)
        if world.decay:
            lost = world.decay * x[:-1]
            dx[:-1] -= lost
            dx[-1] += lost.sum()
        return dx

    times = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(x0, dtype=float), method="LSODA",
                    t_eval=times, rtol=1e-10, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def driving_forces(world: World) -> int:
    """Number of independent driving forces: reservoirs minus the conservation
    laws of the undriven world that they break (Polettini and Esposito 2014)."""
    if not world.exchange:
        return 0
    S = stoichiometric_matrix(world.n, world.recipes).astype(float)
    laws = null_space(S.T).T
    held = list(world.exchange)
    # Exchange swaps an ingredient for empty space, so it breaks a law by the
    # law's weight on the ingredient minus its weight on empty space.
    broken = laws[:, held] - laws[:, [-1]]
    return len(held) - int(np.linalg.matrix_rank(broken, tol=1e-9))
