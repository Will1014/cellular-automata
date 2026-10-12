"""Count the minimal self-copying sets of Gagrani's reference network (one
kind of matter, sizes 1 to L, no empty space) and compare with Table I.

One job per L. Each set is appended to `<out>/L<L>.jsonl` as it is found, so
an interrupted run resumes where it stopped. A summary with provenance and
both clocks is written to `<out>/L<L>_summary.json` when the run completes.

    python scripts/count_cores.py 6 <out_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path

from cellular_automata.cores import directed, minimal_autocatalytic_subnetworks
from cellular_automata.network import balanced_recipes
from cellular_automata.provenance import Clocks, provenance

# Gagrani et al. 2024, Table I: MAS counts by number of reactions.
TABLE_I = {3: {2: 2}, 4: {2: 9, 3: 11}, 5: {2: 24, 3: 98, 4: 48},
           6: {2: 48, 3: 461, 4: 768, 5: 331},
           7: {2: 85, 3: 1549, 4: 6028, 5: 5673, 6: 1709}}


def main(L: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"L{L}.jsonl"
    known = []
    if log.exists():
        known = [frozenset(json.loads(line)["reactions"]) for line in log.read_text().splitlines()]
    recipes = balanced_recipes([[s] for s in range(1, L + 1)], empty=False)
    s_in, s_out = directed(L, recipes)

    with log.open("a") as f, Clocks() as clocks:
        def write(reactions, core):
            f.write(json.dumps({"reactions": sorted(reactions), "core": sorted(core)}) + "\n")
            f.flush()

        new = minimal_autocatalytic_subnetworks(s_in, s_out, known=known, on_found=write)

    sizes = Counter(len(r) for r in known) + Counter(len(r) for r, _ in new)
    summary = {
        "L": L,
        "total": sum(sizes.values()),
        "by_size": dict(sorted(sizes.items())),
        "table_i": TABLE_I.get(L),
        "matches_table_i": dict(sorted(sizes.items())) == TABLE_I.get(L),
        "resumed_from": len(known),
        "clocks_this_session": clocks.as_dict(),
        "provenance": provenance(),
    }
    (out / f"L{L}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("L", "total", "by_size", "matches_table_i")}))


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
