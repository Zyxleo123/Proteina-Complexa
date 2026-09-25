"""Config loading for the LP->CP pre-build check suite.

Thin wrapper over `m15_config.load` that adds ONE thing: a top-level `inherit:` key naming
another YAML, merged under this one.

That exists for a single reason.  The Milestone 1.5 audit fixed the contact cutoff, the
pseudo-angle bound and the torsion-solver budget, and `README_M15_CEILING_AUDIT.md` is
explicit that a retention computed against a different cutoff is a different quantity that
must not be divided into one computed against this one.  Copying the `geometry` block into
a second file would make that divergence a one-character edit away and silent when it
happened.  Inheriting it makes the two configs the same numbers by construction.

Merge rule: dicts merge key by key, depth-first; any other type replaces wholesale.  So a
child can override `geometry.tau_max_deg` without restating the block, and overriding a
list replaces it rather than appending -- an append would silently keep a parent's entry
that the child meant to drop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from script_utils import m15_config

REPO = Path(__file__).resolve().parents[1]


def _merge(parent: Any, child: Any) -> Any:
    if isinstance(parent, dict) and isinstance(child, dict):
        out = dict(parent)
        for k, v in child.items():
            out[k] = _merge(parent.get(k), v) if k in parent else v
        return out
    return child


def load(path: str | Path, _seen: tuple[str, ...] = ()) -> dict:
    """Parse `path`, resolve `inherit:`, expand `$VAR`, and refuse anything left unexpanded."""
    path = Path(path)
    resolved = str(path.resolve())
    if resolved in _seen:
        raise SystemExit(f"{path}: inherit cycle: {' -> '.join(_seen + (resolved,))}")

    raw = yaml.safe_load(path.read_text()) or {}
    parent_ref = raw.pop("inherit", None)
    if parent_ref is None:
        # No parent: hand the file to the Milestone 1.5 loader so `$VAR` expansion and the
        # unexpanded-variable check stay in exactly one place.
        return m15_config.load(path)

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = REPO / parent_path
    merged = _merge(load(parent_path, _seen + (resolved,)), raw)

    # Re-run expansion and the check over the merged result: a child value may itself carry
    # a `$VAR`, and the parent's own check ran before this file's keys existed.
    merged = m15_config._expand(merged)
    m15_config._check(merged, path)
    return merged
