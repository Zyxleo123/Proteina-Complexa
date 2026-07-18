"""PyRosetta interface energy scoring for binder evaluation.

Ported from PepGLAD ``evaluation/dG/energy.py`` (``pyrosetta_interface_energy``):
https://github.com/THUNLP-MT/PepGLAD/blob/bad015ca50c312a89482adb5220c3d907f13df5c/evaluation/dG/energy.py

Uses Rosetta ``InterfaceAnalyzerMover`` with ``pack_separated=True`` to compute
``dG_separated`` and related interface score terms. PyRosetta is optional; when
not installed, callers receive NaN-filled metric dicts.
"""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

# Score terms exported from InterfaceAnalyzerMover (pose.scores keys).
ROSETTA_INTERFACE_SCORE_KEYS = (
    "dG_separated",
    "dSASA_int",
    "dSASA_hphobic",
    "dSASA_polar",
    "delta_unsatHbonds",
    "sc_value",
    "interface_delta_E",
    "nres_int",
    "hbonds_int",
    "packstat",
    # Relax diagnostics (see `pyrosetta_interface_energy`). Not InterfaceAnalyzer
    # outputs; carried alongside so a relax that failed to resolve clashes is
    # visible instead of being laundered into a plausible-looking dG. A relaxed
    # `dG_separated` that is still large and positive means the clashes were not
    # relievable, which is itself the finding.
    "dG_separated_norelax",
    "total_score_norelax",
    "total_score_relaxed",
)

ROSETTA_METRIC_COLS = [f"binder_rosetta_{key}" for key in ROSETTA_INTERFACE_SCORE_KEYS]

_PYROSETTA_INIT_LOCK = threading.Lock()
_PYROSETTA_INITIALIZED = False


def _ensure_pyrosetta_init() -> None:
    """Initialize PyRosetta once per process (matches PepGLAD init flags)."""
    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return
    with _PYROSETTA_INIT_LOCK:
        if _PYROSETTA_INITIALIZED:
            return
        import pyrosetta

        pyrosetta.init(
            " ".join(
                [
                    "-mute",
                    "all",
                    "-use_input_sc",
                    "-ignore_unrecognized_res",
                    "-ignore_zero_occupancy",
                    "false",
                    "-load_PDB_components",
                    "false",
                    "-relax:default_repeats",
                    "2",
                    "-no_fconfig",
                    "-use_terminal_residues",
                    "true",
                    "-in:file:silent_struct_type",
                    "binary",
                ]
            )
        )
        _PYROSETTA_INITIALIZED = True


def pyrosetta_interface_energy(
    pdb_path: str,
    receptor_chains: list[str],
    ligand_chains: list[str],
    *,
    return_dict: bool = False,
    relax: bool = True,
) -> float | dict[str, float]:
    """Compute Rosetta interface energy via InterfaceAnalyzerMover.

    Args:
        pdb_path: Path to the complex PDB.
        receptor_chains: Target / receptor chain IDs (e.g. ``["A"]``).
        ligand_chains: Binder / peptide chain IDs (e.g. ``["B"]``).
        return_dict: If True, return all mapped interface score terms; else only
            ``dG_separated``.
        relax: Run a coordinate-constrained FastRelax on the complex before
            scoring. `InterfaceAnalyzerMover`'s `pack_separated` only repacks the
            *unbound* reference state; the bound pose is scored exactly as given.
            Directly generated (or refolded) backbones carry side-chain clashes, so
            `fa_rep` swamps the result and `dG_separated` comes out large and
            POSITIVE -- it then measures clash, not binding. Measured on a CPSea
            design: +773.6 raw vs -42.8 relaxed (dSASA_int 1040 A^2). Leave this on
            unless the input is already a relaxed/crystal structure.

    Returns:
        ``dG_separated`` float, or a dict of score-term -> value. When ``relax`` is
        set, the dict also carries ``dG_separated_norelax`` and the pre/post total
        scores, so a bad relax stays visible rather than silently replacing the raw
        number.
    """
    _ensure_pyrosetta_init()
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.relax import FastRelax

    def _analyze(p) -> dict[str, float]:
        interface = "".join(ligand_chains) + "_" + "".join(receptor_chains)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        mover.apply(p)
        return {key: float(p.scores[key]) for key in ROSETTA_INTERFACE_SCORE_KEYS if key in p.scores}

    pose = pyrosetta.pose_from_pdb(pdb_path)

    extra: dict[str, float] = {}
    if relax:
        sf = pyrosetta.create_score_function("ref2015")
        extra["total_score_norelax"] = float(sf(pose))
        extra["dG_separated_norelax"] = _analyze(pyrosetta.Pose(pose)).get("dG_separated", float("nan"))

        fast_relax = FastRelax(sf, 1)
        # Restrain to the input coordinates: we are scoring the geometry the model
        # produced, not searching for a better one nearby.
        fast_relax.constrain_relax_to_start_coords(True)
        fast_relax.apply(pose)
        extra["total_score_relaxed"] = float(sf(pose))

    scores = _analyze(pose)
    scores.update(extra)

    if return_dict:
        return scores
    if "dG_separated" not in scores:
        raise KeyError(f"dG_separated missing from InterfaceAnalyzerMover scores for {pdb_path}")
    return scores["dG_separated"]


def _nan_rosetta_metrics() -> dict[str, float]:
    import math

    return dict.fromkeys(ROSETTA_METRIC_COLS, math.nan)


def compute_rosetta_interface_metrics_single(
    pdb_path: str,
    binder_chain: str,
    target_chains: list[str] | str,
) -> dict[str, Any]:
    """Compute Rosetta interface metrics for one complex PDB.

    Binder is the ligand; target chains are the receptor, matching PepGLAD's
    ``ligand_chains`` / ``receptor_chains`` convention.

    Args:
        pdb_path: Complex PDB path.
        binder_chain: Binder chain ID (last chain by convention).
        target_chains: Target chain ID or comma-separated IDs.

    Returns:
        Dict mapping ``binder_rosetta_*`` column names to floats.
    """
    if isinstance(target_chains, str):
        receptor_chains = [c.strip() for c in target_chains.split(",") if c.strip()]
    else:
        receptor_chains = list(target_chains)

    try:
        raw_scores = pyrosetta_interface_energy(
            pdb_path,
            receptor_chains=receptor_chains,
            ligand_chains=[binder_chain],
            return_dict=True,
        )
        metrics: dict[str, Any] = {}
        for key in ROSETTA_INTERFACE_SCORE_KEYS:
            col = f"binder_rosetta_{key}"
            value = raw_scores.get(key)
            metrics[col] = round(float(value), 3) if value is not None else float("nan")
        return metrics
    except Exception as e:
        logger.error(f"Rosetta interface energy failed for {pdb_path}: {e}")
        return _nan_rosetta_metrics()


def is_pyrosetta_available() -> bool:
    """Return True if PyRosetta can be imported."""
    try:
        import pyrosetta  # noqa: F401

        return True
    except ImportError:
        return False
