"""Neighbourhoods as lists of index offsets, in any number of dimensions.

A regular lattice looks the same from every cell, so one list of offsets
describes every cell's neighbours. Grids are periodic (wrap at the edges).

Hexagonal grids use axial coordinates: the array is a sheared hex grid, and
each cell's six neighbours are six of the eight squares around it, every one
except the (+1, +1) and (-1, -1) diagonals.
"""

import numpy as np

LINE_1 = ((-1,), (1,))
SQUARE_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SQUARE_8 = SQUARE_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HEX_6 = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1))

LATTICES = {"1D line": LINE_1, "Square (4)": SQUARE_4, "Square (8)": SQUARE_8, "Hexagonal": HEX_6}


def line(radius: int) -> tuple[tuple[int, ...], ...]:
    return tuple((o,) for o in range(-radius, radius + 1) if o != 0)


def neighbour_sum(grid: np.ndarray, offsets) -> np.ndarray:
    """Sum of each cell's neighbours' values, periodic boundary."""
    total = np.zeros(grid.shape, dtype=np.int64)
    for off in offsets:
        # The neighbour at +off sits where rolling by -off brings it.
        total += np.roll(grid, tuple(-o for o in off), axis=tuple(range(grid.ndim)))
    return total


def hex_centres(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian centres of axial-coordinate cells, unit spacing between neighbours."""
    r, q = np.mgrid[0:rows, 0:cols]
    x = q + r / 2
    y = r * np.sqrt(3) / 2
    return x, y
