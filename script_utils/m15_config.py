"""Shared config loading for the Milestone 1.5 jobs.

Every constant lives in configs/pose_decoy/m15.yaml.  Paths in it use shell-style
`$VAR` so the same string means the same thing in the YAML, in the sbatch preamble and in
a per-submission env file -- there is no second interpolation syntax to keep in sync.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def _expand(node: Any) -> Any:
    if isinstance(node, str):
        return os.path.expandvars(node)
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


def load(path: str | Path) -> dict:
    """Parse the config and expand `$VAR` references from the environment.

    An unexpanded `$` left in a path is an unset variable, which would otherwise surface
    much later as a confusing "no such file" naming a literal dollar sign.
    """
    cfg = _expand(yaml.safe_load(Path(path).read_text()))
    _check(cfg, path)
    return cfg


def _check(node: Any, src: str | Path, trail: str = "") -> None:
    if isinstance(node, str) and "$" in node:
        raise SystemExit(
            f"{src}: unexpanded variable in '{trail}': {node!r}. The sbatch preamble must "
            f"export it before the job runs (see scripts/_m15_preamble.sh)."
        )
    if isinstance(node, dict):
        for k, v in node.items():
            _check(v, src, f"{trail}.{k}" if trail else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check(v, src, f"{trail}[{i}]")


def resolve(path_str: str) -> Path:
    """Repo-relative paths stay repo-relative regardless of the job's working directory."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO / p)
