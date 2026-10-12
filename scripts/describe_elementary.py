"""Describe one elementary rule by its free will and fate curves.

Each row comes from its own independent random start, so no row is a copy
of another and the trust check counts real samples. Writes one JSON file,
skipped if it already exists.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cellular_automata.engine import final_states
from cellular_automata.metrics import block_entropy_curve
from cellular_automata.provenance import Clocks, provenance
from cellular_automata.rules import elementary
from cellular_automata.wolfram_classes import CLASS_OF

SETTINGS = {"runs": 200, "width": 10000, "steps": 1000, "max_block": 24, "trust": 0.01}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("rule", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    path = a.out / f"rule_{a.rule:03d}.json"
    if path.exists():
        return
    a.out.mkdir(parents=True, exist_ok=True)
    s = SETTINGS
    with Clocks() as clocks:
        rng = np.random.default_rng(a.seed)
        initial = rng.integers(0, 2, (s["runs"], s["width"]), dtype=np.uint8)
        rows = final_states(initial, elementary(a.rule), steps=s["steps"])
        H, trusted = block_entropy_curve(rows, s["max_block"], s["trust"])
    h = np.diff(np.concatenate([[0.0], H]))
    fate = H - np.arange(1, len(H) + 1) * h
    clean = lambda x: [None if np.isnan(v) else float(v) for v in x]
    result = {"rule": a.rule, "wolfram_class": CLASS_OF.get(a.rule), "H": clean(H),
              "free_will": clean(h), "fate": clean(fate), "trusted": trusted.tolist(),
              "seed": a.seed, "settings": s, "clocks": clocks.as_dict(),
              "provenance": provenance()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(path)


if __name__ == "__main__":
    main()
