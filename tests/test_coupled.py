import numpy as np
import pytest

from cellular_automata.coupled import (
    CoupledRule,
    consumer_resource,
    layer_mutual_information,
    mutual_information,
    random_coupled_rule,
    run_coupled,
    step_coupled,
    table_shapes,
)
from cellular_automata.lattice import HEX_6, LINE_1


def identity_rule(ke, n):
    s_shape, e_shape = table_shapes(ke, n)
    s = np.zeros(s_shape, dtype=np.uint8)
    s[1] = 1
    e = np.zeros(e_shape, dtype=np.uint8)
    for level in range(ke):
        e[level] = level
    return CoupledRule(s, e)


def test_identity_rule_changes_nothing():
    rng = np.random.default_rng(0)
    s0 = rng.integers(0, 2, (8, 8)).astype(np.uint8)
    e0 = rng.integers(0, 3, (8, 8)).astype(np.uint8)
    s, e = run_coupled(s0, e0, identity_rule(3, 6), HEX_6, 10)
    assert (s == s0).all() and (e == e0).all()


def test_env_only_updates_on_its_period():
    rng = np.random.default_rng(1)
    rule = random_coupled_rule(3, 2, rng)
    s0 = rng.integers(0, 2, 50).astype(np.uint8)
    e0 = rng.integers(0, 3, 50).astype(np.uint8)
    _, e = run_coupled(s0, e0, rule, LINE_1, 12, env_period=4)
    for t in range(12):
        if (t + 1) % 4 != 0:
            assert (e[t + 1] == e[t]).all()


def test_shuffled_coupling_keeps_amounts():
    rng = np.random.default_rng(2)
    rule = random_coupled_rule(3, 2, rng)
    s0 = rng.integers(0, 2, 200).astype(np.uint8)
    e0 = rng.integers(0, 3, 200).astype(np.uint8)
    # Species reading a shuffled environment must equal reading that permutation.
    perm_rng = np.random.default_rng(9)
    s1, _ = step_coupled(s0, e0, rule, LINE_1, species_sees_env="shuffled", rng=perm_rng)
    shuffled = np.random.default_rng(9).permutation(e0)
    from cellular_automata.lattice import neighbour_sum
    expected = rule.species_table[s0, shuffled, neighbour_sum(s0, LINE_1)]
    assert (s1 == expected).all()
    assert sorted(shuffled) == sorted(e0)


def test_constant_coupling_equals_uniform_environment():
    rng = np.random.default_rng(3)
    rule = random_coupled_rule(2, 2, rng)
    s0 = rng.integers(0, 2, 60).astype(np.uint8)
    e0 = rng.integers(0, 2, 60).astype(np.uint8)
    a, _ = step_coupled(s0, e0, rule, LINE_1, species_sees_env="constant", update_env=False)
    b, _ = step_coupled(s0, np.zeros_like(e0), rule, LINE_1, update_env=False)
    assert (a == b).all()


def test_consumer_resource_semantics():
    rule = consumer_resource(3, 2)
    assert rule.species_table[1, 0, 0] == 0  # dies on exhausted ground
    assert rule.species_table[1, 1, 0] == 1  # survives on poor-but-not-exhausted ground
    assert rule.species_table[0, 2, 1] == 1  # colonises good ground next to an occupant
    assert rule.species_table[0, 2, 0] == 0  # but not without one
    assert rule.env_table[2, 1, 0] == 1  # grazed down a level
    assert rule.env_table[0, 0, 1] == 1  # regrows beside non-exhausted ground


def test_mutual_information_null_and_maximum():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 2, 100_000)
    b = rng.integers(0, 3, 100_000)
    assert mutual_information(a, b) < 0.001
    assert mutual_information(a, a) == pytest.approx(1.0, abs=0.01)


def test_lagged_mi_sees_one_step_dependence():
    rng = np.random.default_rng(5)
    x = rng.integers(0, 2, (1000, 20))
    y = np.vstack([rng.integers(0, 2, (1, 20)), x[:-1]])  # y copies x one step later
    assert layer_mutual_information(x, y, lag=1) > 0.95
    assert layer_mutual_information(x, y, lag=0) < 0.01


def test_lattice_mismatch_raises():
    with pytest.raises(ValueError):
        run_coupled(np.zeros(10, np.uint8), np.zeros(10, np.uint8),
                    consumer_resource(3, 6), LINE_1, 5)
