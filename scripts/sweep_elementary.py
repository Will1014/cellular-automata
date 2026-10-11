"""Score the 88 distinct elementary rules and test whether the metrics
separate Wolfram's classes better than shuffled labels do.

Writes elementary_metrics.csv and class_separation.csv to the output dir.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from cellular_automata.engine import random_state, run
from cellular_automata.metrics import (
    compression_ratio,
    damage_spreading,
    langton_lambda,
    spatial_entropy,
    temporal_entropy,
    transient_and_period,
)
from cellular_automata.rules import elementary, elementary_representatives
from cellular_automata.wolfram_classes import CLASS_OF

METRICS = ["spatial_entropy", "temporal_entropy", "compression", "damage", "lambda", "period"]


def score(rule: int, width: int, steps: int, seeds: int) -> dict:
    table = elementary(rule)
    rows = []
    for s in range(seeds):
        init = random_state(width, 2, np.random.default_rng(s))
        h = run(init, table, steps=steps)
        late = h[steps // 2 :]
        _, period = transient_and_period(table, init, max_steps=4 * steps)
        rows.append([
            spatial_entropy(h),
            temporal_entropy(h),
            compression_ratio(late),
            damage_spreading(table, init, steps=steps)[-1],
            langton_lambda(table),
            np.log2(period) if period else np.log2(4 * steps),  # no repeat -> cap
        ])
    return dict(zip(METRICS, np.mean(rows, axis=0)))


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability a positive outranks a negative (Mann-Whitney), ties count half."""
    pos, neg = scores[positive], scores[~positive]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return greater / (len(pos) * len(neg))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--width", type=int, default=200)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--shuffles", type=int, default=1000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    reps = elementary_representatives()
    table = [{"rule": r, "wolfram_class": CLASS_OF[r], **score(r, a.width, a.steps, a.seeds)}
             for r in reps]
    with open(a.out / "elementary_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)

    classes = np.array([row["wolfram_class"] for row in table])
    rng = np.random.default_rng(0)
    contrasts = {"III+IV vs I+II": classes >= 3, "IV vs III": None}
    results = []
    for m in METRICS:
        x = np.array([row[m] for row in table])
        for name, positive in contrasts.items():
            if positive is None:  # class IV vs class III only
                keep = classes >= 3
                xs, pos = x[keep], classes[keep] == 4
            else:
                xs, pos = x, positive
            obs = auc(xs, pos)
            null = np.array([auc(xs, rng.permutation(pos)) for _ in range(a.shuffles)])
            p_two = (np.abs(null - 0.5) >= abs(obs - 0.5)).mean()
            results.append({"metric": m, "contrast": name, "n_pos": int(pos.sum()),
                            "auc": round(obs, 3), "null_mean": round(null.mean(), 3),
                            "null_95": round(np.quantile(np.abs(null - 0.5), 0.95) + 0.5, 3),
                            "p": round(p_two, 4)})
    with open(a.out / "class_separation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
