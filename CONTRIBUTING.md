# Contributing

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/pytest
```

## Versioning

Semantic versioning, `MAJOR.MINOR.PATCH`, read for scientific code:

- **MAJOR**: an API break, or any change that alters results. A metric fix
  that changes every score is major even if no signature changes.
- **MINOR**: new capability, results unchanged for existing code paths.
- **PATCH**: fixes with no effect on results, documentation, packaging.

## Releasing

1. Everything for the release is merged and CI is green.
2. Move `[Unreleased]` entries in `CHANGELOG.md` into a dated version section.
3. Bump the version in `pyproject.toml`, `src/cellular_automata/__init__.py`
   and `CITATION.cff`.
4. If results changed, say so at the top of the release notes, with what
   changed and by roughly how much.
5. Commit, tag `vX.Y.Z`, push the tag.
6. Create the GitHub release with the changelog section as its body.
