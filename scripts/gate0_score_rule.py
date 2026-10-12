"""Gate 0: score one elementary rule and write one JSON file.

Free will and fate at the fixed look-back, the full fate curve, Zenil's
transition coefficient, and the zlib and damage-spreading comparisons, on
the settings in docs/2026-10-12_gate0-keep-drop-rule.md. Skips the rule if
its file already exists, so an interrupted batch loses at most one rule.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cellular_automata.engine import random_state, run
from cellular_automata.metrics import (
    block_entropy_curve,
    compression_ratio,
    damage_spreading,
    free_will_and_fate,
    transition_coefficient,
)
from cellular_automata.provenance import Clocks, provenance
from cellular_automata.rules import elementary
from cellular_automata.wolfram_classes import CLASS_OF

SETTINGS = {"width": 10000, "steps": 2000, "burn_in": 1000, "look_back": 16, "max_block": 24,
            "trust": 0.01, "zenil_n": 40, "zenil_times": list(range(50, 601, 50)),
            "zenil_block": 6}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("rule", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--skip-transition", action="store_true")
    a = p.parse_args()
    path = a.out / f"rule_{a.rule:03d}.json"
    if path.exists():
        return
    a.out.mkdir(parents=True, exist_ok=True)
    s = SETTINGS
    table = elementary(a.rule)
    with Clocks() as clocks:
        per_seed = []
        for seed in a.seeds:
            init = random_state(s["width"], 2, np.random.default_rng(seed))
            kept = run(init, table, steps=s["steps"])[s["burn_in"] + 1 :]
            H, trusted = block_entropy_curve(kept, s["max_block"], s["trust"])
            fw, fate = free_will_and_fate(H, s["look_back"])
            per_seed.append({
                "seed": seed, "free_will": fw, "fate": fate,
                "H": [None if np.isnan(x) else x for x in H], "trusted": trusted.tolist(),
                "zlib": compression_ratio(kept),
                "damage": float(damage_spreading(table, init, steps=s["steps"])[-1]),
            })
        C, c_seq = (None, None) if a.skip_transition else transition_coefficient(
            table, s["zenil_n"], tuple(s["zenil_times"]), s["zenil_block"])
    mean = {k: float(np.mean([r[k] for r in per_seed]))
            for k in ("free_will", "fate", "zlib", "damage")}
    result = {"rule": a.rule, "wolfram_class": CLASS_OF.get(a.rule), **mean,
              "transition": C, "transition_seq": c_seq, "per_seed": per_seed,
              "seeds": a.seeds, "settings": s, "clocks": clocks.as_dict(),
              "provenance": provenance()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(path)


if __name__ == "__main__":
    main()
