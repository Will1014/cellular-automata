"""Species reader: viable cores grow from rare exactly past the analytic
threshold (docs/2026-10-12_phase2-tests.md, Drive folder)."""

import numpy as np
import pytest
from scipy.optimize import brentq

from cellular_automata.patch import World, run_patch
from cellular_automata.species import background, invasion_rate

F, X, E = 0, 1, 2
DECAY = 0.05


def replicator(mu):
    # F + X <-> X + X, F from a reservoir at potential mu, everything decays.
    return World([((F, X), (X, X))], np.array([0.0, -0.5, 0.0]), np.ones(1),
                 exchange={F: (mu, 1.0)}, decay=DECAY)


def analytic_rate(mu):
    """kf F* - decay: the reservoir and decay set F*, and X grows by eating F."""
    kf = np.exp(-1.0)  # left -0.5, right -1.0, barrier 1 above the higher side
    top = max(mu, 0.0) + 1.0
    kin, kout = np.exp(mu - top), np.exp(-top)
    return kf * kin / (kin + kout + DECAY) - DECAY


def mu_at(ratio):
    """Reservoir potential at which kf F* is `ratio` times the decay rate."""
    return brentq(lambda m: analytic_rate(m) + DECAY - ratio * DECAY, -10.0, 10.0)


@pytest.mark.parametrize("ratio", [0.9, 1.1])
def test_replicator_grows_from_rare_exactly_past_the_threshold(ratio):
    world = replicator(mu_at(ratio))
    x_bg = background(world, [X], np.array([0.1, 0.0, 0.9]))
    rate = invasion_rate(world, [X], x_bg)
    assert rate == pytest.approx(analytic_rate(mu_at(ratio)), abs=1e-8)
    assert np.sign(rate) == np.sign(ratio - 1.0)
    # And by running it: a rare X shrinks below the threshold, grows above.
    x0 = x_bg.copy()
    x0[X], x0[E] = 1e-6, x0[E] - 1e-6
    _, xs = run_patch(world, x0, t_end=20.0, n_out=2)
    assert np.sign(xs[-1][X] - 1e-6) == np.sign(ratio - 1.0)


def test_background_refuses_members_that_come_back():
    world = replicator(1.0)
    with pytest.raises(ValueError):
        background(world, [F], np.array([0.1, 0.1, 0.8]))
