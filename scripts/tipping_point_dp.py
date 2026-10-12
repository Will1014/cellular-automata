"""Free will and fate across the tipping point of site directed percolation.

One spreading probability p per call; one JSON file out, skipped if it
exists. Every lattice starts full, runs `steps`, and contributes its final
row if anything is still alive (survivors only, as the plan requires).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cellular_automata.metrics import block_entropy_curve
from cellular_automata.provenance import Clocks, provenance
from cellular_automata.stochastic import directed_percolation

SETTINGS = {"runs": 200, "width": 10000, "steps": 1000, "start": "full", "max_block": 24,
            "trust": 0.01}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("p", type=float)
    a.add_argument("out", type=Path)
    a.add_argument("--seed", type=int, default=0)
    a = a.parse_args()
    path = a.out / f"p_{a.p:.4f}.json"
    if path.exists():
        return
    a.out.mkdir(parents=True, exist_ok=True)
    s = SETTINGS
    with Clocks() as clocks:
        rng = np.random.default_rng(a.seed)
        rows = directed_percolation(np.ones((s["runs"], s["width"])), a.p, s["steps"], rng)
        alive = rows.any(axis=1)
        if alive.any():
            H, trusted = block_entropy_curve(rows[alive], s["max_block"], s["trust"])
        else:
            H, trusted = np.zeros(s["max_block"]), np.ones(s["max_block"], dtype=bool)
    h = np.diff(np.concatenate([[0.0], H]))
    fate = H - np.arange(1, len(H) + 1) * h
    clean = lambda x: [None if np.isnan(v) else float(v) for v in x]
    result = {"p": a.p, "surviving_runs": int(alive.sum()), "density": float(rows.mean()),
              "H": clean(H), "free_will": clean(h), "fate": clean(fate),
              "trusted": trusted.tolist(), "seed": a.seed, "settings": s,
              "clocks": clocks.as_dict(), "provenance": provenance()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(path)


if __name__ == "__main__":
    main()
