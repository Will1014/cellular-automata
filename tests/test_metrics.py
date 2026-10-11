import numpy as np

from cellular_automata.engine import random_state, run, single_cell
from cellular_automata.metrics import (
    compression_ratio,
    damage_spreading,
    langton_lambda,
    spatial_entropy,
    transient_and_period,
)
from cellular_automata.rules import elementary


def test_entropy_bounds():
    rng = np.random.default_rng(0)
    assert spatial_entropy(np.zeros((50, 200), dtype=np.uint8)) == 0.0
    noise = rng.integers(0, 2, (50, 2000), dtype=np.uint8)
    assert spatial_entropy(noise) > 0.99


def test_compression_orders_noise_above_regularity():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 2, (200, 200), dtype=np.uint8)
    flat = np.zeros((200, 200), dtype=np.uint8)
    assert compression_ratio(flat) < 0.05
    assert compression_ratio(noise) > 0.95


def test_langton_lambda():
    assert langton_lambda(elementary(0)) == 0.0
    assert langton_lambda(elementary(255)) == 1.0
    assert langton_lambda(elementary(30)) == 0.5


def test_damage_identity_rule_stays_one_cell():
    rng = np.random.default_rng(0)
    d = damage_spreading(elementary(204), random_state(101, 2, rng), steps=30)
    assert np.allclose(d, 1 / 101)


def test_damage_rule_0_heals_and_rule_90_spreads():
    rng = np.random.default_rng(0)
    init = random_state(201, 2, rng)
    assert damage_spreading(elementary(0), init, steps=10)[-1] == 0
    # Rule 90 is linear: damage is Pascal's triangle mod 2 from one cell.
    d90 = damage_spreading(elementary(90), init, steps=64)
    assert d90[64] == 2 / 201  # 2^6 rows: Sierpinski collapses to two cells


def test_identity_has_period_one():
    rng = np.random.default_rng(0)
    assert transient_and_period(elementary(204), random_state(30, 2, rng)) == (0, 1)


def test_shift_period_divides_width():
    rng = np.random.default_rng(0)
    transient, period = transient_and_period(elementary(170), random_state(12, 2, rng))
    assert transient == 0 and 12 % period == 0


def test_runs_are_deterministic_for_a_seed():
    a = run(random_state(50, 2, np.random.default_rng(7)), elementary(110), steps=40)
    b = run(random_state(50, 2, np.random.default_rng(7)), elementary(110), steps=40)
    assert (a == b).all()
    assert run(single_cell(9), elementary(30), steps=1).shape == (2, 9)


def test_structured_complexity_bounds():
    from cellular_automata.metrics import structured_complexity

    assert structured_complexity(0.0, 0.3) == 0.0  # perturbation dies: simple
    assert structured_complexity(0.5, 1.0) == 0.0  # incompressible: noise
    assert structured_complexity(0.5, 0.4) == 0.3


def test_compression_decay_sign():
    from cellular_automata.metrics import compression_decay

    rng = np.random.default_rng(0)
    noise_then_flat = np.vstack([rng.integers(0, 2, (500, 200), dtype=np.uint8),
                                 np.zeros((500, 200), dtype=np.uint8)])
    assert compression_decay(noise_then_flat) > 0.8
    steady = rng.integers(0, 2, (1000, 200), dtype=np.uint8)
    assert abs(compression_decay(steady)) < 0.05
