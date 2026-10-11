import numpy as np

from cellular_automata.lattice import HEX_6, LINE_1, SQUARE_8, hex_centres, line, neighbour_sum


def naive_sum(grid, offsets):
    out = np.zeros(grid.shape, dtype=int)
    for idx in np.ndindex(grid.shape):
        for off in offsets:
            j = tuple((i + o) % n for i, o, n in zip(idx, off, grid.shape))
            out[idx] += grid[j]
    return out


def test_neighbour_sum_matches_naive_1d_and_2d():
    rng = np.random.default_rng(0)
    g1 = rng.integers(0, 3, 17)
    assert (neighbour_sum(g1, LINE_1) == naive_sum(g1, LINE_1)).all()
    assert (neighbour_sum(g1, line(2)) == naive_sum(g1, line(2))).all()
    g2 = rng.integers(0, 3, (7, 9))
    for offs in (HEX_6, SQUARE_8):
        assert (neighbour_sum(g2, offs) == naive_sum(g2, offs)).all()


def test_hex_offsets_symmetric_and_distinct():
    assert len(set(HEX_6)) == 6
    assert {(-a, -b) for a, b in HEX_6} == set(HEX_6)


def test_hex_neighbours_are_equidistant_in_the_plane():
    # The axial shear is only right if all six neighbours sit at distance 1.
    x, y = hex_centres(10, 10)
    r, q = 5, 5
    dists = [np.hypot(x[r + dr, q + dq] - x[r, q], y[r + dr, q + dq] - y[r, q]) for dr, dq in HEX_6]
    assert np.allclose(dists, 1.0)
    # And the two dropped diagonals are not neighbours: they sit further away.
    for dr, dq in ((1, 1), (-1, -1)):
        assert np.hypot(x[r + dr, q + dq] - x[r, q], y[r + dr, q + dq] - y[r, q]) > 1.5
