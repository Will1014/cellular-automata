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


def block_entropy_curve(
    rows: np.ndarray, max_block: int = 24, trust: float = 0.01, chunk: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Block entropy H(L) in bits (not per cell) of length-L windows along each
    row, periodic, pooled over rows, for L = 1..max_block. Binary rows only.

    Also returns, per L, whether the estimate is trusted: the number of
    distinct windows seen is at most `trust` times the number of windows.
    Untrusted entries are still filled where they were counted, else NaN.
    Rows are processed in chunks so memory stays small."""
    rows = np.asarray(rows, dtype=np.int64)
    n_rows, width = rows.shape
    total = n_rows * width
    dense = min(max_block, 20)  # bincount up to 2^20 patterns, unique() beyond
    counts = [np.zeros(2**L, dtype=np.int64) for L in range(1, dense + 1)]
    sparse: dict[int, list] = {L: [] for L in range(dense + 1, max_block + 1)}
    cap = trust * total
    for i in range(0, n_rows, chunk):
        block = rows[i : i + chunk]
        wrapped = np.concatenate([block, block[:, : max_block - 1]], axis=1)
        code = np.zeros(block.shape, dtype=np.int64)
        for L in range(1, max_block + 1):
            code = (code << 1) | wrapped[:, L - 1 : L - 1 + width]
            if L <= dense:
                counts[L - 1] += np.bincount(code.ravel(), minlength=2**L)
            elif sparse[L] is not None:
                vals, c = np.unique(code, return_counts=True)
                if len(vals) > cap:  # already untrusted, stop counting
                    sparse[L] = None
                else:
                    sparse[L].append((vals, c))
    H = np.full(max_block, np.nan)
    trusted = np.zeros(max_block, dtype=bool)
    for L in range(1, max_block + 1):
        if L <= dense:
            c = counts[L - 1]
            c = c[c > 0]
        else:
            if sparse[L] is None:
                continue
            vals = np.concatenate([v for v, _ in sparse[L]])
            cs = np.concatenate([c for _, c in sparse[L]])
            _, inv = np.unique(vals, return_inverse=True)
            c = np.bincount(inv, weights=cs)
        p = c / total
        H[L - 1] = float(-(p * np.log2(p)).sum())
        trusted[L - 1] = len(c) <= cap
    return H, trusted


def free_will_and_fate(H: np.ndarray, block: int) -> tuple[float, float]:
    """From a block entropy curve, free will (entropy rate) h(L) = H(L) - H(L-1)
    and fate (excess entropy) E(L) = H(L) - L h(L), at look-back L = block."""
    h_prev = H[block - 2] if block > 1 else 0.0
    h = H[block - 1] - h_prev
    return float(h), float(H[block - 1] - block * h)


def transition_coefficient(
    table: np.ndarray, n: int = 40, times: tuple[int, ...] = tuple(range(50, 601, 50)),
    block: int = 6,
) -> tuple[float, list[float]]:
    """Zenil's (2010) transition coefficient. Start j (1..n) is the Gray code of
    j as a `block`-cell pattern centred on a zero background wide enough that
    nothing wraps. c(t) is the mean absolute jump in zlib-compressed length of
    the t-step space-time diagram between starts j and j+1, divided by t.
    C is the least-squares slope of c(t) against t's index 1..len(times).
    Returns C and the c(t) sequence."""
    seq = []
    for t in times:
        width = block + 2 * t + 2
        lo = (width - block) // 2
        lengths = []
        for j in range(1, n + 1):
            g = j ^ (j >> 1)
            init = np.zeros(width, dtype=np.uint8)
            init[lo : lo + block] = [(g >> (block - 1 - i)) & 1 for i in range(block)]
            h = run(init, table, steps=t)
            lengths.append(len(zlib.compress(np.packbits(h).tobytes(), 9)))
        seq.append(float(np.abs(np.diff(lengths)).mean() / t))
    slope = np.polyfit(np.arange(1, len(times) + 1), seq, 1)[0]
    return float(slope), seq
