# cellular-automata

Generate, play back, measure and evolve 1D cellular automaton rules.

Each cell in a row updates from itself and its neighbours by a fixed rule,
and the rows stacked over time form a space-time diagram. This project
explores which rules produce structured behaviour, using Wolfram's 256
elementary rules as a calibrated test bed, then evolves rules in larger
spaces with a genetic algorithm.

Status: engine, metrics, styling, two-layer species/environment engine (1D, square, hex) and app built. Rule evolution not yet built.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Run the app

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## Module map

| Module | Purpose |
| --- | --- |
| `engine` | Rule tables, stepping, edge handling. A lattice is a list of neighbour offsets. |
| `rules` | Wolfram rule numbers, random rules, mutation, symmetry classes |
| `lattice` | Neighbourhood offsets: 1D, square, hexagonal (axial coordinates) |
| `coupled` | Two-layer species/environment engine with switchable cross-layer links |
| `metrics` | Entropy, compressibility, Langton's lambda, damage spreading, cycle length |
| `styling` | Palettes, gradients, Jenks/quantile/equal-interval breaks, derived values |
| `evolve` | Planned: genetic algorithm over rule tables, with a random-rule baseline |
| `app/` | Streamlit app for playback, styling and metrics |

## Known limitations

- The single-layer engine is 1D only. The two-layer engine runs 1D, square and hex grids.
- Same-time correlation between layers misses lagged feedback; use the lagged measure.
- The class IV metrics are validated in-sample on 4 class IV rules only.
- Local surprisal uses a fixed 3-cell window, too narrow to learn rule 110's
  14-cell background.
