"""Rule tables: Wolfram numbering, random rules, mutation and symmetries."""

import numpy as np


def table_size(k: int, n_neighbours: int) -> int:
    return k**n_neighbours


def from_number(number: int, k: int = 2, n_neighbours: int = 3) -> np.ndarray:
    """Rule number to lookup table: digit i (base k) is the output for index i."""
    size = table_size(k, n_neighbours)
    if not 0 <= number < k**size:
        raise ValueError(f"rule {number} out of range for k={k}, {n_neighbours} neighbours")
    table = np.empty(size, dtype=np.uint8)
    for i in range(size):
        number, table[i] = divmod(number, k)
    return table


def to_number(table: np.ndarray, k: int = 2) -> int:
    return sum(int(v) * k**i for i, v in enumerate(table))


def elementary(number: int) -> np.ndarray:
    return from_number(number, 2, 3)


def _digits(index: int, k: int, n: int) -> list[int]:
    out = []
    for _ in range(n):
        index, d = divmod(index, k)
        out.append(d)
    return out[::-1]  # most significant (leftmost cell) first


def _index(digits: list[int], k: int) -> int:
    idx = 0
    for d in digits:
        idx = idx * k + d
    return idx


def mirror(table: np.ndarray, k: int, n_neighbours: int) -> np.ndarray:
    """Left-right reflection: reverse every neighbourhood."""
    out = np.empty_like(table)
    for i in range(table.shape[0]):
        out[i] = table[_index(_digits(i, k, n_neighbours)[::-1], k)]
    return out


def complement(table: np.ndarray, k: int, n_neighbours: int) -> np.ndarray:
    """State relabelling s -> k-1-s applied to inputs and output."""
    out = np.empty_like(table)
    for i in range(table.shape[0]):
        flipped = [k - 1 - d for d in _digits(i, k, n_neighbours)]
        out[i] = k - 1 - table[_index(flipped, k)]
    return out


def equivalents(table: np.ndarray, k: int, n_neighbours: int) -> list[np.ndarray]:
    m = mirror(table, k, n_neighbours)
    return [table, m, complement(table, k, n_neighbours), complement(m, k, n_neighbours)]


def canonical_number(number: int, k: int = 2, n_neighbours: int = 3) -> int:
    """Smallest rule number among a rule's mirror/complement equivalents."""
    table = from_number(number, k, n_neighbours)
    return min(to_number(t, k) for t in equivalents(table, k, n_neighbours))


def elementary_representatives() -> list[int]:
    return sorted({canonical_number(n) for n in range(256)})


def random_rule(
    k: int, n_neighbours: int, rng: np.random.Generator, lam: float | None = None
) -> np.ndarray:
    """Random table. If lam is given, each entry is non-quiescent (non-zero) with
    probability lam, which sets Langton's lambda in expectation."""
    size = table_size(k, n_neighbours)
    if lam is None:
        return rng.integers(0, k, size=size, dtype=np.uint8)
    table = np.zeros(size, dtype=np.uint8)
    active = rng.random(size) < lam
    table[active] = rng.integers(1, k, size=int(active.sum()), dtype=np.uint8)
    return table


def mutate(table: np.ndarray, k: int, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Each entry is replaced, with probability rate, by a different state."""
    out = table.copy()
    hit = rng.random(table.shape[0]) < rate
    shift = rng.integers(1, k, size=int(hit.sum()), dtype=np.uint8)
    out[hit] = (out[hit] + shift) % k
    return out
