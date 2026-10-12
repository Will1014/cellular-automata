"""What every result records about how it was made."""

import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True,
                          text=True).stdout.strip()


def provenance() -> dict:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "rng": "numpy PCG64 via default_rng(seed)",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class Clocks:
    """Wall clock and monotonic clock. On macOS the monotonic clock stops while
    the machine sleeps, so a run that slept shows a gap between the two."""

    def __enter__(self):
        self._wall, self._mono = time.time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.time() - self._wall
        self.mono = time.perf_counter() - self._mono

    def as_dict(self) -> dict:
        return {"wall_s": round(self.wall, 2), "monotonic_s": round(self.mono, 2)}
