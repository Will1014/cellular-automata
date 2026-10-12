"""Gate 0: read the per-rule files, test each measure against the
shuffled-label null and apply docs/2026-10-12_gate0-keep-drop-rule.md.

Writes class_separation.csv, decisions.json, calibration.json,
fate_curves.csv and, if a symmetry directory is given, symmetry.csv.
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from cellular_automata.provenance import provenance
from cellular_automata.rules import canonical_number, elementary_representatives

# sign: +1 if class IV is predicted higher than class III, -1 if lower
NEW = {"free_will": -1, "fate": +1, "transition": +1}
OLD = {"free_will": "zlib", "fate": "zlib", "transition": "damage"}
SHUFFLES = 100_000


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based, ties share their mean rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def auc_from_ranks(r: np.ndarray, pos_idx: np.ndarray, n_pos: int, n_neg: int) -> np.ndarray:
    """AUC for each row of positive indices (Mann-Whitney, ties count half)."""
    return (r[pos_idx].sum(axis=-1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc(x: np.ndarray, pos: np.ndarray) -> float:
    r = ranks(x)
    return float(auc_from_ranks(r, np.flatnonzero(pos), pos.sum(), (~pos).sum()))


def exact_null(x: np.ndarray, n_pos: int) -> np.ndarray:
    r = ranks(x)
    idx = np.array(list(combinations(range(len(x)), n_pos)))
    return auc_from_ranks(r, idx, n_pos, len(x) - n_pos)


def shuffle_null(x: np.ndarray, n_pos: int, n: int, rng: np.random.Generator) -> np.ndarray:
    r = ranks(x)
    idx = rng.permuted(np.tile(np.arange(len(x)), (n, 1)), axis=1)[:, :n_pos]
    return auc_from_ranks(r, idx, n_pos, len(x) - n_pos)


def holm(p: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(p)
    passed = [False] * len(p)
    for k, i in enumerate(order):
        if p[i] > alpha / (len(p) - k):
            break
        passed[i] = True
    return passed


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def contrast_tests(values: dict, classes: np.ndarray, shuffles: int, rng) -> list[dict]:
    """The nine tests: IV vs III one-sided (exact), the other two two-sided."""
    out = []
    iii_iv = classes >= 3
    for m, sign in NEW.items():
        x = sign * values[m]  # oriented: predicted direction is high
        xs, pos = x[iii_iv], classes[iii_iv] == 4
        obs = auc(xs, pos)
        null = exact_null(xs, 4)
        out.append({"metric": m, "contrast": "IV vs III", "auc_oriented": obs,
                    "null_mean": float(null.mean()), "p": float((null >= obs - 1e-12).mean()),
                    "sided": "one"})
        for name, pos in (("IV vs rest", classes == 4), ("III+IV vs I+II", iii_iv)):
            obs = auc(x, pos)
            null = shuffle_null(x, pos.sum(), shuffles, rng)
            extreme = (np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12).sum()
            out.append({"metric": m, "contrast": name, "auc_oriented": obs,
                        "null_mean": float(null.mean()), "p": (1 + extreme) / (1 + shuffles),
                        "sided": "two"})
    for row, ok in zip(out, holm([r["p"] for r in out])):
        row["holm_pass"] = ok
    return out


def fate_curve(H: np.ndarray) -> np.ndarray:
    h = np.diff(np.concatenate([[0.0], H]))
    return H - np.arange(1, len(H) + 1) * h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("rules_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--symmetry-dir", type=Path)
    p.add_argument("--calibration-draws", type=int, default=1000)
    a = p.parse_args()

    reps = elementary_representatives()
    rows = [json.loads((a.rules_dir / f"rule_{r:03d}.json").read_text()) for r in reps]
    classes = np.array([r["wolfram_class"] for r in rows])
    values = {k: np.array([r[k] for r in rows], dtype=float)
              for k in ("free_will", "fate", "transition", "zlib", "damage")}
    commits = {r["provenance"]["commit"] for r in rows}
    dirty = any(r["provenance"]["dirty"] for r in rows)

    tests = contrast_tests(values, classes, SHUFFLES, np.random.default_rng(0))

    # criteria 2 and 3, and the decision
    iii = classes == 3
    core = np.isin([r["rule"] for r in rows], [54, 110])
    decisions = {}
    for m, sign in NEW.items():
        x, old = sign * values[m], values[OLD[m]]
        main_test = next(t for t in tests if t["metric"] == m and t["contrast"] == "IV vs III")
        keep = iii | core
        auc_core = auc(x[keep], core[keep])
        old_raw = auc(old[iii | (classes == 4)], classes[iii | (classes == 4)] == 4)
        old_best = max(old_raw, 1 - old_raw)
        rho = spearman(values[m], old)
        c1, c2, c3 = main_test["holm_pass"], auc_core >= 0.75, main_test["auc_oriented"] > old_best
        verdict = ("keep" if c1 and c2 and c3 else
                   "keep as a replacement" if c1 and c2 and abs(rho) >= 0.9 else "drop")
        decisions[m] = {"c1_separates_iv_from_iii": c1, "p_iv_vs_iii": main_test["p"],
                        "auc_iv_vs_iii": main_test["auc_oriented"],
                        "c2_auc_54_110_vs_iii": auc_core, "c2_pass": bool(c2),
                        "c3_old_measure": OLD[m], "c3_old_auc_best_direction": old_best,
                        "c3_pass": bool(c3), "spearman_with_old": rho, "verdict": verdict}

    # null against null: pure-noise measures through the same rule
    rng = np.random.default_rng(1)
    single, family = [], []
    for _ in range(a.calibration_draws):
        fake = {m: rng.normal(size=len(rows)) for m in NEW}
        t = contrast_tests(fake, classes, 2000, rng)
        single += [r["p"] <= 0.05 for r in t if r["contrast"] == "IV vs III"]
        family.append(any(r["holm_pass"] for r in t if r["contrast"] == "IV vs III"))
    calibration = {"draws": a.calibration_draws,
                   "iv_vs_iii_single_test_rate_at_0.05": float(np.mean(single)),
                   "any_measure_passes_criterion_1_rate": float(np.mean(family)),
                   "exact_null_means": [t["null_mean"] for t in tests
                                        if t["contrast"] == "IV vs III"]}

    # fate curves, seed-averaged over trusted look-backs
    curve_rows = []
    for r in rows:
        H = np.array([[np.nan if v is None else v for v in s["H"]] for s in r["per_seed"]])
        trusted = np.array([s["trusted"] for s in r["per_seed"]]).all(axis=0)
        E = fate_curve(np.nanmean(H, axis=0))
        last = int(np.flatnonzero(trusted).max()) + 1 if trusted.any() else 0
        rising = last >= 3 and E[last - 1] - E[last - 3] > 0.1
        curve_rows.append({"rule": r["rule"], "wolfram_class": r["wolfram_class"],
                           "last_trusted_look_back": last, "still_rising": bool(rising),
                           **{f"fate_L{L}": round(float(E[L - 1]), 4) if L <= last else ""
                              for L in range(1, 25)}})

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "class_separation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tests[0].keys())
        w.writeheader()
        w.writerows(tests)
    with open(a.out / "fate_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=curve_rows[0].keys())
        w.writeheader()
        w.writerows(curve_rows)
    with open(a.out / "rule_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "wolfram_class", *values])
        for i, r in enumerate(rows):
            w.writerow([r["rule"], r["wolfram_class"], *(values[k][i] for k in values)])

    if a.symmetry_dir:
        sym = []
        sd = {k: np.array([np.std([s[k] for s in r["per_seed"]]) for r in rows])
              for k in ("free_will", "fate")}
        by_rep = {r["rule"]: i for i, r in enumerate(rows)}
        for n in range(256):
            s = json.loads((a.symmetry_dir / f"rule_{n:03d}.json").read_text())
            i = by_rep[canonical_number(n)]
            entry = {"rule": n, "representative": canonical_number(n)}
            for k in ("free_will", "fate"):
                tol = 3 * max(sd[k][i], 1e-3)
                entry[f"{k}_diff"] = s[k] - values[k][i]
                entry[f"{k}_ok"] = abs(s[k] - values[k][i]) <= tol
            sym.append(entry)
        with open(a.out / "symmetry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sym[0].keys())
            w.writeheader()
            w.writerows(sym)
        calibration["symmetry_rules_outside_3sd"] = [e["rule"] for e in sym
                                                     if not (e["free_will_ok"] and e["fate_ok"])]

    (a.out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (a.out / "decisions.json").write_text(json.dumps(
        {"decisions": decisions, "rule_files_commits": sorted(commits),
         "rule_files_dirty": dirty, "aggregate_provenance": provenance()}, indent=2))
    print(json.dumps(calibration, indent=2))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
