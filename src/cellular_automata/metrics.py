"""Metrics on a space-time history (rows are time, columns are cells)."""

import zlib
from collections import Counter

import numpy as np

from .engine import run


def block_entropy(rows: np.ndarray, block: int = 3) -> float:
    """Shannon entropy (bits per cell) of length-`block` windows along each row,
    pooled over the given rows, periodic. 0 for uniform, 1 for fair coin."""
    counts: Counter = Counter()
    for row in rows:
        wrapped = np.concatenate([row, row[: block - 1]])
        windows = np.lib.stride_tricks.sliding_window_view(wrapped, block)
        counts.update(map(bytes, windows))
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum() / block)


def spatial_entropy(history: np.ndarray, block: int = 3, burn_in: float = 0.5) -> float:
    start = int(history.shape[0] * burn_in)
    return block_entropy(history[start:], block)


def temporal_entropy(history: np.ndarray, block: int = 3, burn_in: float = 0.5) -> float:
    start = int(history.shape[0] * burn_in)
    return block_entropy(history[start:].T, block)


def compression_ratio(history: np.ndarray, k: int = 2) -> float:
    """Compressed size over raw size. Low means regular, near 1 means noise.
    Uses bit-packing for k=2 so a random binary history scores close to 1."""
    raw = np.packbits(history).tobytes() if k == 2 else history.tobytes()
    return len(zlib.compress(raw, 9)) / len(raw)


def langton_lambda(table: np.ndarray, quiescent: int = 0) -> float:
    return float(np.mean(table != quiescent))


def damage_spreading(
    table: np.ndarray,
    initial: np.ndarray,
    k: int = 2,
    offsets: tuple[int, ...] = (-1, 0, 1),
    steps: int = 100,
    boundary: str = "periodic",
) -> np.ndarray:
    """Fraction of cells that differ over time after flipping the centre cell."""
    perturbed = initial.copy()
    c = initial.shape[0] // 2
    perturbed[c] = (perturbed[c] + 1) % k
    a = run(initial, table, k, offsets, steps, boundary)
    b = run(perturbed, table, k, offsets, steps, boundary)
    return (a != b).mean(axis=1)


def transient_and_period(
    table: np.ndarray,
    initial: np.ndarray,
    k: int = 2,
    offsets: tuple[int, ...] = (-1, 0, 1),
    max_steps: int = 2000,
    boundary: str = "periodic",
) -> tuple[int, int | None]:
    """Steps before the state first repeats, and the cycle length.
    Period is None if no repeat within max_steps."""
    from .engine import step

    seen = {initial.tobytes(): 0}
    state = initial
    for t in range(1, max_steps + 1):
        state = step(state, np.asarray(table, dtype=np.uint8), k, offsets, boundary)
        key = state.tobytes()
        if key in seen:
            return seen[key], t - seen[key]
        seen[key] = t
    return max_steps, None


def structured_complexity(damage: float, compression: float) -> float:
    """Edge-of-chaos score: high only when a perturbation spreads (not simple)
    and the history still compresses (not noise). Both inputs in [0, 1]."""
    return float(damage * max(0.0, 1.0 - compression))


def compression_decay(history: np.ndarray, k: int = 2, window: float = 0.2) -> float:
    """Compression ratio of an early window minus that of the final window.
    Positive when the history becomes more regular over time, as when moving
    structures collide and annihilate. The first 10% is skipped as transient."""
    t = history.shape[0]
    w = max(1, int(t * window))
    start = max(1, t // 10)
    early = history[start : start + w]
    late = history[t - w :]
    return compression_ratio(early, k) - compression_ratio(late, k)
