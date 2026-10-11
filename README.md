# cellular-automata

Generate, play back, measure and evolve 1D cellular automaton rules.

Each cell in a row updates from itself and its neighbours by a fixed rule,
and the rows stacked over time form a space-time diagram. This project
explores which rules produce structured behaviour, using Wolfram's 256
elementary rules as a calibrated test bed, then evolves rules in larger
spaces with a genetic algorithm.

Status: scaffold only. Nothing below the module map exists yet.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Planned module map

| Module | Purpose |
| --- | --- |
| `engine` | Rule tables, stepping, edge handling. A lattice is a list of neighbour offsets. |
| `rules` | Wolfram rule numbers, random rules, mutation, symmetry classes |
| `metrics` | Entropy, compressibility, Langton's lambda, damage spreading, cycle length |
| `evolve` | Genetic algorithm over rule tables, with a random-rule baseline |
| `app` | Streamlit app for playback and styling |

## Known limitations

- 1D only. Hex and 3D lattices are planned in the design but not built.
