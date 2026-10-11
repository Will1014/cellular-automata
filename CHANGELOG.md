# Changelog

All notable changes to this project are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning as defined in CONTRIBUTING.md.

## [Unreleased]

### Added
- Repository scaffold: package skeleton, licence, CI, citation file.
- `engine`: 1D runner over a rule lookup table, any state count and
  neighbour offsets, periodic or fixed boundary.
- `rules`: Wolfram rule numbering, mirror/complement symmetry, the 88
  elementary representatives, random rules with a target lambda, mutation.
- `metrics`: block entropy (spatial, temporal), compression ratio, Langton's
  lambda, damage spreading, transient and period.
- `metrics.structured_complexity` and `metrics.compression_decay`: class IV
  candidates. On the 88 elementary rules they separate class IV from the
  rest at AUC 0.85 and 0.96 against a shuffled-label null. In-sample only.
- `styling`: categorical palettes, gradients, class breaks (Fisher-Jenks
  natural breaks, quantile, equal interval), and derived per-cell values
  (cell age, activity, local surprisal).
- Streamlit app (`app/streamlit_app.py`): elementary or random rules,
  playback, styling and a metrics panel.
- `wolfram_classes`: Wolfram's classes for the 88 rules, from Martinez (2013).
- `scripts/sweep_elementary.py`: scores the 88 rules and tests class
  separation against a shuffled-label null.
