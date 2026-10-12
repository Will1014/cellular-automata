import numpy as np

from cellular_automata.stochastic import directed_percolation


def test_certain_spreading_keeps_a_full_lattice_full():
    out = directed_percolation(np.ones((3, 50)), 1.0, 20, np.random.default_rng(0))
    assert out.all()


def test_no_spreading_empties_the_lattice():
    out = directed_percolation(np.ones((3, 50)), 0.0, 1, np.random.default_rng(0))
    assert not out.any()


def test_activity_dies_below_and_persists_above_the_critical_point():
    start = np.ones((4, 2000))
    low = directed_percolation(start, 0.60, 2000, np.random.default_rng(1))
    high = directed_percolation(start, 0.80, 2000, np.random.default_rng(2))
    assert not low.any()
    assert high.mean() > 0.3


def test_same_seed_gives_same_run():
    start = np.ones((2, 100))
    a = directed_percolation(start, 0.7, 50, np.random.default_rng(5))
    b = directed_percolation(start, 0.7, 50, np.random.default_rng(5))
    assert (a == b).all()
