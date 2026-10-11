"""Turn a history into an RGB image: categorical colours for states, or a
classified gradient for a derived per-cell value."""

import numpy as np

PALETTES = {
    "Ink": ["#f7f4ec", "#1b1b1b", "#c0392b", "#2e86ab", "#f2a541"],
    "Gaia": ["#0f2a1d", "#9bd17a", "#f4d35e", "#ee964b", "#5c80bc"],
    "Night": ["#0b0c10", "#66fcf1", "#c3073f", "#f5f5f5", "#ffb400"],
}

GRADIENTS = {
    "Viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "Magma": ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
    "Greys": ["#ffffff", "#000000"],
    "Earth": ["#f6eddc", "#c9b37e", "#7a9a5a", "#2f5d50", "#10263b"],
}


def hex_to_rgb(colour: str) -> tuple[int, int, int]:
    c = colour.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def categorical(history: np.ndarray, colours: list[str]) -> np.ndarray:
    lut = np.array([hex_to_rgb(c) for c in colours], dtype=np.uint8)
    if history.max() >= len(lut):
        raise ValueError(f"{history.max() + 1} states but {len(lut)} colours")
    return lut[history]


def ramp(stops: list[str], n: int) -> np.ndarray:
    """n colours evenly interpolated along the gradient stops."""
    rgb = np.array([hex_to_rgb(c) for c in stops], dtype=float)
    pos = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    return np.stack([np.interp(t, pos, rgb[:, i]) for i in range(3)], axis=1).astype(np.uint8)


# --- Class breaks -----------------------------------------------------------

def equal_interval_breaks(values: np.ndarray, n: int) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return np.linspace(lo, hi, n + 1)[1:-1]


def quantile_breaks(values: np.ndarray, n: int) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0, 1, n + 1)[1:-1]))


def jenks_breaks(values: np.ndarray, n: int, max_sample: int = 600, seed: int = 0) -> np.ndarray:
    """Fisher-Jenks natural breaks: the n-class split of sorted values that
    minimises total within-class squared deviation. Exact on up to
    `max_sample` values; larger inputs are sampled first."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size > max_sample:
        v = np.random.default_rng(seed).choice(v, max_sample, replace=False)
    v = np.sort(v)
    m = v.size
    n = min(n, len(np.unique(v)))
    if n < 2:
        return np.array([])
    s1 = np.concatenate([[0.0], np.cumsum(v)])
    s2 = np.concatenate([[0.0], np.cumsum(v * v)])

    cost = np.full((n + 1, m + 1), np.inf)
    back = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for c in range(1, n + 1):
        for j in range(c, m + 1):
            i = np.arange(c - 1, j)
            tot = s1[j] - s1[i]
            ssd = (s2[j] - s2[i]) - tot * tot / (j - i)  # squared deviation of v[i:j]
            val = cost[c - 1, i] + ssd
            best = int(np.argmin(val))
            cost[c, j], back[c, j] = val[best], i[best]
    starts, j = [], m
    for c in range(n, 0, -1):
        i = back[c, j]
        starts.append(i)
        j = i
    starts = sorted(starts)[1:]
    # Break between the last value of one class and the first of the next.
    return np.array([(v[i - 1] + v[i]) / 2 for i in starts])


BREAKS = {"Natural breaks (Jenks)": jenks_breaks, "Quantile": quantile_breaks,
          "Equal interval": equal_interval_breaks}


def classified(values: np.ndarray, stops: list[str], method: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """RGB image and the break values used."""
    breaks = BREAKS[method](values, n)
    classes = np.digitize(values, breaks)
    colours = ramp(stops, len(breaks) + 1)
    return colours[classes], breaks


def continuous(values: np.ndarray, stops: list[str]) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    t = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values, dtype=float)
    colours = ramp(stops, 256)
    return colours[np.clip((t * 255).round().astype(int), 0, 255)]


# --- Derived per-cell values --------------------------------------------------

def cell_age(history: np.ndarray) -> np.ndarray:
    """Steps since each cell last changed state."""
    age = np.zeros(history.shape, dtype=float)
    for t in range(1, history.shape[0]):
        same = history[t] == history[t - 1]
        age[t] = np.where(same, age[t - 1] + 1, 0)
    return age


def activity(history: np.ndarray, window: int = 10) -> np.ndarray:
    """Number of state changes at each cell over the previous `window` steps."""
    changed = np.zeros(history.shape, dtype=float)
    changed[1:] = history[1:] != history[:-1]
    c = np.cumsum(changed, axis=0)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return out


def local_surprisal(history: np.ndarray, k: int = 2, width: int = 3) -> np.ndarray:
    """-log2 frequency of each cell's local pattern (itself and neighbours in
    the row), counted over the whole history. Rare patterns light up, which
    picks out moving structures against a regular background."""
    r = width // 2
    idx = np.zeros(history.shape, dtype=np.int64)
    for o in range(-r, r + 1):
        idx = idx * k + np.roll(history, -o, axis=1)
    counts = np.bincount(idx.ravel(), minlength=k**width)
    p = counts / counts.sum()
    return -np.log2(p[idx])


DERIVED = {"Cell age": cell_age, "Activity": activity, "Local surprisal": local_surprisal}


def upscale(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1) if factor > 1 else image


# --- Hexagonal rendering ------------------------------------------------------

def render_hex(rgb: np.ndarray, cell_px: int = 6, background: str = "#ffffff") -> np.ndarray:
    """Draw an axial-coordinate grid of colours as pointy-top hexagons.

    rgb has shape (rows, cols, 3). The grid is periodic, so the sheared
    parallelogram is wrapped into a rectangle: row r is drawn shifted by half
    a cell when r is odd, which is the same torus seen from a different cut.
    """
    from PIL import Image, ImageDraw

    rows, cols = rgb.shape[:2]
    w = cell_px  # centre-to-centre distance between horizontal neighbours
    radius = w / np.sqrt(3)
    dy = radius * 1.5
    width = int(np.ceil((cols + 0.5) * w))
    height = int(np.ceil((rows - 1) * dy + 2 * radius))
    img = Image.new("RGB", (width, height), hex_to_rgb(background))
    draw = ImageDraw.Draw(img)
    angles = np.deg2rad(90 + 60 * np.arange(6))
    ux, uy = radius * np.cos(angles), radius * np.sin(angles)
    for r in range(rows):
        # Axial q maps to display column (q + r // 2) % cols; invert to read rgb.
        shift = (r % 2) * w / 2
        cy = radius + r * dy
        for c in range(cols):
            q = (c - r // 2) % cols
            cx = w / 2 + c * w + shift
            draw.polygon(list(zip(cx + ux, cy - uy)), fill=tuple(int(v) for v in rgb[r, q]))
    return np.asarray(img)
