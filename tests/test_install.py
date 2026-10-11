from pathlib import Path

import cellular_automata


def test_import_resolves_to_this_repo():
    # Guards against an editable install pointing at another working copy.
    repo_root = Path(__file__).resolve().parents[1]
    assert Path(cellular_automata.__file__).resolve().is_relative_to(repo_root)
