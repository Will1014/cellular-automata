import numpy as np
import pytest

from cellular_automata.engine import random_state, run
from cellular_automata.metrics import (
    block_entropy_curve,
    free_will_and_fate,
    transition_coefficient,
)
from cellular_automata.rules import complement, elementary, mirror


def periodic_rows(pattern, n_rows, width, rng):
    """Rows of a repeating pattern, each at a random phase."""
    p = len(pattern)
    tiled = np.tile(pattern, width // p + 2)
    return np.array([tiled[s : s + width] for s in rng.integers(0, p, n_rows)])


def test_fair_coin_has_full_free_will_and_no_fate():
    rows = np.random.default_rng(0).integers(0, 2, (200, 10000))
    H, trusted = block_entropy_curve(rows, max_block=12)
    h, E = free_will_and_fate(H, 12)
    assert h == pytest.approx(1, abs=0.01)
    assert E == pytest.approx(0, abs=0.05)
    assert trusted.all()


@pytest.mark.parametrize("period", [2, 3, 14])
def test_repeating_pattern_has_no_free_will_and_log_period_fate(period):
    rng = np.random.default_rng(period)
    # a pattern whose rotations are all distinct, so every phase is visible
    while True:
        pattern = rng.integers(0, 2, period)
        if len({tuple(np.roll(pattern, s)) for s in range(period)}) == period:
            break
    rows = periodic_rows(pattern, 200, period * 500, rng)
    H, _ = block_entropy_curve(rows, max_block=16)
    h, E = free_will_and_fate(H, 16)
    assert h == pytest.approx(0, abs=1e-3)
    assert E == pytest.approx(np.log2(period), abs=0.01)


def test_curve_beyond_dense_range_matches_direct_count():
    rng = np.random.default_rng(1)
    rows = periodic_rows(rng.integers(0, 2, 30), 50, 3000, rng)
    H, trusted = block_entropy_curve(rows, max_block=24, chunk=7)
    assert trusted[23]
    assert H[23] == pytest.approx(np.log2(30), abs=0.01)


def test_noisy_rows_go_untrusted_at_long_look_back():
    rows = np.random.default_rng(2).integers(0, 2, (100, 1000))
    _, trusted = block_entropy_curve(rows, max_block=24)
    assert trusted[5] and not trusted[23]


@pytest.mark.parametrize("number", [30, 54, 110, 106])
def test_mirror_and_complement_give_identical_free_will_and_fate(number):
    table = elementary(number)
    init = random_state(2000, 2, np.random.default_rng(number))
    base = block_entropy_curve(run(init, table, steps=300)[150:], max_block=16)[0]
    flipped = run(1 - init, complement(table, 2, 3), steps=300)[150:]
    reversed_ = run(init[::-1].copy(), mirror(table, 2, 3), steps=300)[150:]
    np.testing.assert_allclose(block_entropy_curve(flipped, max_block=16)[0], base)
    np.testing.assert_allclose(block_entropy_curve(reversed_, max_block=16)[0], base)


def test_rule_0_has_no_transition():
    C, seq = transition_coefficient(elementary(0), n=10, times=(50, 100, 150))
    assert abs(C) < 0.05
    assert len(seq) == 3
