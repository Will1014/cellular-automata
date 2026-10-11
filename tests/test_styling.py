import numpy as np
import pytest

from cellular_automata.styling import (
    activity,
    categorical,
    cell_age,
    classified,
    equal_interval_breaks,
    jenks_breaks,
    local_surprisal,
    quantile_breaks,
    ramp,
)


def test_categorical_maps_states_to_colours():
    img = categorical(np.array([[0, 1]], dtype=np.uint8), ["#000000", "#ff0000"])
    assert img.tolist() == [[[0, 0, 0], [255, 0, 0]]]
    with pytest.raises(ValueError):
        categorical(np.array([[2]], dtype=np.uint8), ["#000000", "#ffffff"])


def test_jenks_finds_obvious_clusters():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0, 0.1, 100), rng.normal(5, 0.1, 100), rng.normal(10, 0.1, 100)])
    b = jenks_breaks(v, 3)
    assert len(b) == 2
    assert 0.5 < b[0] < 4.5 and 5.5 < b[1] < 9.5


def test_jenks_matches_brute_force_on_small_input():
    from itertools import combinations

    v = np.sort(np.array([1, 2, 4, 5, 9, 10, 11, 20], dtype=float))

    def cost(splits):
        parts = np.split(v, splits)
        return sum(((p - p.mean()) ** 2).sum() for p in parts)

    best = min(combinations(range(1, len(v)), 2), key=cost)
    expected = [(v[i - 1] + v[i]) / 2 for i in best]
    assert np.allclose(jenks_breaks(v, 3), expected)


def test_quantile_gives_equal_counts():
    v = np.arange(1000, dtype=float)
    counts = np.bincount(np.digitize(v, quantile_breaks(v, 4)))
    assert counts.min() >= 249 and counts.max() <= 251


def test_equal_interval():
    assert np.allclose(equal_interval_breaks(np.array([0.0, 10.0]), 5), [2, 4, 6, 8])


def test_classified_uses_n_colours():
    v = np.linspace(0, 1, 100).reshape(10, 10)
    img, breaks = classified(v, ["#000000", "#ffffff"], "Equal interval", 4)
    assert len(breaks) == 3
    assert len({tuple(c) for c in img.reshape(-1, 3)}) == 4


def test_ramp_endpoints():
    r = ramp(["#000000", "#ffffff"], 3)
    assert r[0].tolist() == [0, 0, 0] and r[-1].tolist() == [255, 255, 255]


def test_cell_age_and_activity():
    h = np.array([[0], [0], [1], [1], [1]], dtype=np.uint8)
    assert cell_age(h)[:, 0].tolist() == [0, 1, 0, 1, 2]
    assert activity(h, window=2)[:, 0].tolist() == [0, 0, 1, 1, 0]


def test_surprisal_highlights_rare_pattern():
    h = np.zeros((20, 50), dtype=np.uint8)
    h[10, 25] = 1
    s = local_surprisal(h)
    assert s[10, 25] > s[0, 0]
