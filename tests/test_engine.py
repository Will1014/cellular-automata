from math import comb

import numpy as np
import pytest

from cellular_automata.engine import offsets_for_radius, run, single_cell
from cellular_automata.rules import elementary, from_number


def naive_step(state, number):
    """Rule definition written out directly, independent of the engine."""
    n = len(state)
    out = []
    for i in range(n):
        left, centre, right = state[(i - 1) % n], state[i], state[(i + 1) % n]
        out.append((number >> (4 * left + 2 * centre + right)) & 1)
    return out


@pytest.mark.parametrize("number", range(256))
def test_engine_matches_definition_for_every_elementary_rule(number):
    rng = np.random.default_rng(number)
    state = rng.integers(0, 2, 41, dtype=np.uint8)
    history = run(state, elementary(number), steps=20)
    expected = list(state)
    for t in range(1, 21):
        expected = naive_step(expected, number)
        assert history[t].tolist() == expected


def test_rule_90_is_pascal_triangle_mod_2():
    width, steps = 129, 60
    history = run(single_cell(width), elementary(90), steps=steps)
    c = width // 2
    for t in range(steps + 1):
        for x in range(-t, t + 1):
            expected = comb(t, (t + x) // 2) % 2 if (t + x) % 2 == 0 else 0
            assert history[t, c + x] == expected


def test_rule_90_is_xor_of_neighbours():
    rng = np.random.default_rng(0)
    state = rng.integers(0, 2, 100, dtype=np.uint8)
    nxt = run(state, elementary(90), steps=1)[1]
    assert (nxt == np.roll(state, 1) ^ np.roll(state, -1)).all()


def test_rule_30_known_rows():
    rows = run(single_cell(21), elementary(30), steps=4)[:, 6:15]
    as_text = ["".join(map(str, r)) for r in rows]
    assert as_text == [
        "000010000",
        "000111000",
        "001100100",
        "011011110",
        "110010001",
    ]


def test_fixed_boundary_reads_zeros_outside():
    # Rule 2 copies the right neighbour leftwards when isolated (001 -> 1).
    state = np.array([0, 0, 0, 0, 1], dtype=np.uint8)
    nxt = run(state, elementary(2), steps=1, boundary="fixed")[1]
    assert nxt.tolist() == [0, 0, 0, 1, 0]


def test_wider_neighbourhood_and_more_states_run():
    k, offsets = 3, offsets_for_radius(2)
    table = from_number(12345, k, len(offsets))
    history = run(np.zeros(30, dtype=np.uint8), table, k, offsets, steps=5)
    assert history.shape == (6, 30)
    assert history.max() < k


def test_wrong_table_size_raises():
    with pytest.raises(ValueError):
        run(single_cell(10), np.zeros(4, dtype=np.uint8))


def test_final_states_match_run_for_each_lattice():
    rng = np.random.default_rng(3)
    initial = rng.integers(0, 2, (5, 37), dtype=np.uint8)
    from cellular_automata.engine import final_states

    out = final_states(initial, elementary(110), steps=25)
    for i in range(5):
        assert out[i].tolist() == run(initial[i], elementary(110), steps=25)[-1].tolist()
