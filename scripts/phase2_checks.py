"""Phase 2 checks against simulation (docs/2026-10-12_phase2-tests.md).

Four modes, each over one batch of randomly drawn worlds:
- organisations: driven worlds with decay, 5 starts each; every settled
  state's survivors must form an organisation
- patch: worlds the screen sets aside, 5 starts sharing their totals; all
  must settle to the same state and stay there
- lattice: the same worlds on an 8 x 8 hex lattice, each cell started
  differently; every run must end uniform, at the patch's settled state
- community: the organisation worlds again; which cores each settled
  state holds, and which of them are copying (descriptive)

One job per batch. Each world's record is appended to
`<out>/<mode>_b<batch>.jsonl` as it finishes, and the world's seed is
`batch * 1000 + index`, so a batch rerun reproduces it exactly. A summary
with provenance and both clocks is written when the batch completes.

    python scripts/phase2_checks.py organisations 0 <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

from cellular_automata.network import balanced_recipes, with_empty
from cellular_automata.patch import World, _full_matrix, rate_of_change, run_patch
from cellular_automata.patch_lattice import lattice_rate_of_change, run_lattice
from cellular_automata.provenance import Clocks, provenance
from cellular_automata.screen import verdict
from cellular_automata.species import community, is_organisation, world_cores

COMP = with_empty([[1, 0], [0, 1], [1, 1], [2, 0], [2, 1]])
RECIPES = balanced_recipes(COMP)
N = len(COMP)
PER_BATCH = 20
STARTS = 5
T_END = 1e4
T_LONGEST = 1e7
PRESENT = 1e-9   # an amount above this counts as present
SETTLED = 1e-14  # largest rate of change for a state to count as settled
SAME = 1e-6      # largest difference for two settled states to count as one


def draw_world(rng, driven: bool) -> World:
    k = int(rng.integers(2, 5))
    mask = [RECIPES[i] for i in sorted(rng.choice(len(RECIPES), size=k, replace=False))]
    energy = np.append(rng.normal(0, 1, N - 1), 0.0)
    barrier = rng.uniform(0, 2, k)
    if not driven:
        return World(mask, energy, barrier)
    feed, waste = rng.choice(N - 1, size=2, replace=False)
    mu = rng.uniform(0, 4)
    return World(mask, energy, barrier, exchange={int(feed): (mu, 1.0), int(waste): (-mu, 1.0)},
                 decay=0.01)


def draw_start(rng, shape=()) -> np.ndarray:
    x = rng.uniform(0.01, 0.1, (N,) + shape)
    x[-1] = 1.0 - x[:-1].sum(axis=0)
    return x


def same_totals(rng, world: World, x0) -> np.ndarray:
    """A start with the same conserved totals as `x0`: a random step along
    the world's reactions, shrunk until every amount stays positive."""
    step = _full_matrix(world) @ rng.normal(0, 1, _full_matrix(world).shape[1])
    scale = 0.05
    while (x0 + scale * step).min() <= 0:
        scale /= 2
    return x0 + scale * step


def settle(world, x0):
    """Run until settled, extending tenfold from T_END up to T_LONGEST. A run
    still moving at T_LONGEST is reported as unsettled, never as a failure."""
    x, t = np.asarray(x0, dtype=float), 0.0
    while True:
        span = T_END if t == 0 else 9 * t
        _, xs = run_patch(world, x, span, n_out=2)
        x, t = xs[-1], t + span
        speed = float(np.abs(rate_of_change(world)(x)).max())
        if speed <= SETTLED or t >= T_LONGEST:
            return x, speed


def check_organisations(rng):
    world = draw_world(rng, driven=True)
    starts = []
    for _ in range(STARTS):
        x, speed = settle(world, draw_start(rng))
        if speed > SETTLED:
            starts.append({"settled": False, "speed": speed})
            continue
        survivors = [int(i) for i in np.flatnonzero(x > PRESENT)]
        starts.append({"settled": True, "survivors": survivors,
                       "organisation": is_organisation(world, survivors)})
    failed = any(s["settled"] and not s["organisation"] for s in starts)
    return world, {"starts": starts, "failed": failed}


def check_community(rng):
    """Same worlds and starts as check_organisations (same seeds). Records
    which cores each settled state holds, and which of those are copying.
    Descriptive: a core is copying by definition when its reactions run on
    the survivors, so there is nothing here that can fail."""
    world = draw_world(rng, driven=True)
    cores = world_cores(world)
    starts = []
    for _ in range(STARTS):
        x, speed = settle(world, draw_start(rng))
        if speed > SETTLED:
            starts.append({"settled": False, "speed": speed})
            continue
        starts.append({"settled": True, "community": community(world, x, cores, PRESENT)})
    return world, {"cores": [sorted(c) for _, c in cores], "starts": starts, "failed": False}


def draw_set_aside(rng):
    while True:
        world = draw_world(rng, driven=False)
        v = verdict(world)
        if v["verdict"] == "set aside":
            return world, v["theorem"]


def check_patch(rng):
    world, theorem = draw_set_aside(rng)
    x0 = draw_start(rng)
    finals, speeds = [], []
    for j in range(STARTS):
        x, speed = settle(world, x0 if j == 0 else same_totals(rng, world, x0))
        finals.append(x)
        speeds.append(speed)
    spread = float(max(np.abs(f - finals[0]).max() for f in finals))
    settled = max(speeds) <= SETTLED
    return world, {"theorem": theorem, "spread": spread, "speed": max(speeds),
                   "settled": settled, "failed": settled and spread > SAME}


def check_lattice(rng):
    world, theorem = draw_set_aside(rng)
    move = rng.uniform(0.1, 1.0, N)
    x0 = draw_start(rng, (8, 8))
    rhs = lattice_rate_of_change(world, move, x0.shape[1:])
    x, t = x0, 0.0
    while True:
        span = T_END if t == 0 else 9 * t
        x, t = run_lattice(world, move, x, span), t + span
        speed = float(np.abs(rhs(0.0, x.ravel())).max())
        if speed <= SETTLED or t >= T_LONGEST:
            break
    patch, patch_speed = settle(world, x0.mean(axis=(1, 2)))
    unevenness = float(np.abs(x - x.mean(axis=(1, 2), keepdims=True)).max())
    off_patch = float(np.abs(x - patch[:, None, None]).max())
    settled = max(speed, patch_speed) <= SETTLED
    return world, {"theorem": theorem, "unevenness": unevenness, "off_patch": off_patch,
                   "speed": speed, "settled": settled,
                   "failed": settled and (unevenness > SAME or off_patch > SAME)}


CHECKS = {"organisations": check_organisations, "community": check_community, "patch": check_patch, "lattice": check_lattice}


def describe(world: World) -> dict:
    return {"recipes": [list(map(list, r)) for r in world.recipes],
            "energy": world.energy.tolist(), "barrier": world.barrier.tolist(),
            "exchange": {str(k): list(v) for k, v in world.exchange.items()},
            "decay": world.decay}


def main(mode: str, batch: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{mode}_b{batch}.jsonl"
    done = {json.loads(line)["seed"] for line in log.read_text().splitlines()} if log.exists() else set()
    with log.open("a") as f, Clocks() as clocks:
        for index in range(PER_BATCH):
            seed = batch * 1000 + index
            if seed in done:
                continue
            world, result = CHECKS[mode](np.random.default_rng(seed))
            f.write(json.dumps({"seed": seed, "world": describe(world), **result}) + "\n")
            f.flush()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    summary = {
        "mode": mode, "batch": batch, "worlds": len(records),
        "failed": [r["seed"] for r in records if r["failed"]],
        "unsettled": sum(not s["settled"] for r in records for s in r.get("starts", [r])),
        "composition": COMP.tolist(), "starts": STARTS, "t_end": T_END,
        "thresholds": {"present": PRESENT, "settled": SETTLED, "same": SAME},
        "clocks_this_session": clocks.as_dict(), "provenance": provenance(),
    }
    (out / f"{mode}_b{batch}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("mode", "batch", "worlds", "failed")}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
