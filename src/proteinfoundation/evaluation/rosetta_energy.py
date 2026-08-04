"""PyRosetta interface energy scoring for binder evaluation.

Ported from PepGLAD ``evaluation/dG/energy.py`` (``pyrosetta_interface_energy``):
https://github.com/THUNLP-MT/PepGLAD/blob/bad015ca50c312a89482adb5220c3d907f13df5c/evaluation/dG/energy.py

Uses Rosetta ``InterfaceAnalyzerMover`` with ``pack_separated=True`` to compute
``dG_separated`` and related interface score terms. PyRosetta is optional; when
not installed, callers receive NaN-filled metric dicts.

**Cyclic topology.** Generated complex PDBs carry no ``LINK``/``CONECT`` records
(``utils.pdb_utils.to_pdb`` never writes them), and Rosetta's default residue
typing treats every chain as an ordinary linear polymer with free N-/C-termini.
Left alone, a geometrically closed macrocycle is scored as if it were a linear
peptide that merely happens to sit in a ring-shaped conformation -- the same
"instrument problem" `evaluation.cyclic_offset` exists to fix for AF2, just for
Rosetta. `_declare_cyclic_bond` patches the loaded pose so the requested closing
bond is an explicit covalent connection before `FastRelax`/`InterfaceAnalyzerMover`
run, for **mainchain** and **disulfide** (both use existing, verified PyRosetta
APIs: `Conformation.declare_chemical_bond` after stripping terminus variants, and
`Conformation.add_disulfide_bond`, respectively). **Isopeptide is not yet
supported** -- declaring an arbitrary Lys-NZ <-> Asp/Glu-side-chain bond needs a
custom Rosetta residue-connection patch that plain amino acid types don't carry
(unlike a stripped terminus or a disulfide-patched CYS); this is a real, open gap
and is reported via `binder_rosetta_cyclic_topology_declared=False` rather than
silently scored as if it were closed.
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

ROSETTA_METRIC_COLS = [f"binder_rosetta_{key}" for key in ROSETTA_INTERFACE_SCORE_KEYS] + [
    # Not a score term -- whether the requested cyclic bond was actually declared
    # on the pose before scoring. NaN/False here means the reported dG (still
    # returned, not withheld) describes a topology missing its closing bond; see
    # module docstring. Always present (never NaN) so it can gate downstream
    # filtering the same way `n_valid_*` counts do elsewhere in this codebase.
    "binder_rosetta_cyclic_topology_declared"
]


def _declare_cyclic_bond(
    pose,
    binder_chain: str,
    cyclization_type: str,
    i_local: int,
    j_local: int,
) -> bool:
    """Declares the requested cyclic covalent bond on `pose`, in place.

    Args:
        pose: A loaded PyRosetta pose (post `pose_from_pdb`, pre relax/scoring).
        binder_chain: PDB chain ID of the binder.
        cyclization_type: `"mainchain"`, `"disulfide"`, or `"isopeptide"`.
        i_local, j_local: 0-based, binder-chain-LOCAL residue indices (same
            convention as CPSea's `cyclization_i`/`cyclization_j`), mapped to pose
            sequence positions via the pose's own `PDBInfo`. This assumes
            `utils.pdb_utils.to_pdb`'s per-chain-restarting numbering (binder-local
            index `k` <-> PDB resSeq `k + 1` on `binder_chain`), the convention
            every CPSea-generated PDB in this repo is written with.

    Returns:
        True if the bond was declared; False if the chemistry is not yet
        supported (isopeptide -- see module docstring), in which case scoring
        proceeds WITHOUT the bond and the caller should record that the result
        is missing it.

    Raises:
        ValueError: the endpoints could not be mapped to pose positions, or
            (for disulfide) `add_disulfide_bond` refused (e.g. not both CYS).
    """
    from pyrosetta.rosetta.core.chemical import VariantType
    from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue

    pdb_info = pose.pdb_info()
    seqpos_i = pdb_info.pdb2pose(binder_chain, i_local + 1)
    seqpos_j = pdb_info.pdb2pose(binder_chain, j_local + 1)
    if seqpos_i == 0 or seqpos_j == 0:
        raise ValueError(
            f"could not map cyclization endpoints (i_local={i_local}, j_local={j_local}) "
            f"to pose positions on chain {binder_chain!r} -- pdb2pose returned 0 (not found)"
        )

    if cyclization_type == "mainchain":
        # A closing amide needs a free N/C connection; the terminus variants
        # explicitly cap those with extra protons/oxygens and no open bond site, so
        # they must come off first -- mirrors what RosettaScripts' PeptideCyclizeMover
        # does internally.
        remove_variant_type_from_pose_residue(pose, VariantType.LOWER_TERMINUS_VARIANT, seqpos_i)
        remove_variant_type_from_pose_residue(pose, VariantType.UPPER_TERMINUS_VARIANT, seqpos_j)
        pose.conformation().declare_chemical_bond(seqpos_i, "N", seqpos_j, "C")
        return True

    if cyclization_type == "disulfide":
        ok = pose.conformation().add_disulfide_bond(seqpos_i, seqpos_j)
        if not ok:
            raise ValueError(
                f"add_disulfide_bond failed for pose residues {seqpos_i}/{seqpos_j} "
                f"(names {pose.residue(seqpos_i).name3()}/{pose.residue(seqpos_j).name3()}) "
                "-- are both really CYS?"
            )
        return True

    if cyclization_type == "isopeptide":
        logger.warning(
            "Rosetta isopeptide bond NOT declared (residues {}/{}): no custom residue-"
            "connection patch is wired up yet (see rosetta_energy._declare_cyclic_bond). "
            "Scoring proceeds on a topology missing its closing bond; "
            "binder_rosetta_cyclic_topology_declared=False marks this in the output.",
            seqpos_i,
            seqpos_j,
        )
        return False

    raise ValueError(f"unknown cyclization_type: {cyclization_type!r}")


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
    cyclization_type: str | None = None,
    cyclization_i: int | None = None,
    cyclization_j: int | None = None,
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
        cyclization_type: If given (``"mainchain"``, ``"disulfide"``, or
            ``"isopeptide"``), declares that closing bond on the pose before
            relax/scoring -- see module docstring and `_declare_cyclic_bond`.
            `None` (default) is a complete no-op, identical to today's behavior.
            Requires exactly one entry in `ligand_chains`.
        cyclization_i, cyclization_j: 0-based binder-LOCAL residue indices of the
            closing bond's two endpoints. Defaults to the binder's termini
            (``0``, ``binder_length - 1``) when `cyclization_type` is given and
            these are `None` -- CPSea binders empirically always cyclize between
            their termini for all three chemistries (520/520 sampled, see
            `cyclization.mask`'s `terminal_only` docstring).

    Returns:
        ``dG_separated`` float, or a dict of score-term -> value. When ``relax`` is
        set, the dict also carries ``dG_separated_norelax`` and the pre/post total
        scores, so a bad relax stays visible rather than silently replacing the raw
        number. When `cyclization_type` is given, the dict also carries
        `cyclic_topology_declared` (bool).
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
    if cyclization_type is not None:
        if len(ligand_chains) != 1:
            raise ValueError(
                f"cyclic bond declaration requires exactly one ligand chain, got {ligand_chains}"
            )
        binder_chain = ligand_chains[0]
        i_local, j_local = cyclization_i, cyclization_j
        if i_local is None or j_local is None:
            binder_length = sum(
                1
                for seqpos in range(1, pose.size() + 1)
                if pose.pdb_info().chain(seqpos) == binder_chain
            )
            i_local = 0 if i_local is None else i_local
            j_local = binder_length - 1 if j_local is None else j_local
        extra["cyclic_topology_declared"] = float(
            _declare_cyclic_bond(pose, binder_chain, cyclization_type, i_local, j_local)
        )

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
    cyclization_type: str | None = None,
    cyclization_i: int | None = None,
    cyclization_j: int | None = None,
) -> dict[str, Any]:
    """Compute Rosetta interface metrics for one complex PDB.

    Binder is the ligand; target chains are the receptor, matching PepGLAD's
    ``ligand_chains`` / ``receptor_chains`` convention.

    Args:
        pdb_path: Complex PDB path.
        binder_chain: Binder chain ID (last chain by convention).
        target_chains: Target chain ID or comma-separated IDs.
        cyclization_type, cyclization_i, cyclization_j: Forwarded to
            `pyrosetta_interface_energy` -- see its docstring and the module
            docstring. `cyclization_type=None` (default) is a complete no-op,
            identical to before this parameter existed.

    Returns:
        Dict mapping ``binder_rosetta_*`` column names to floats, plus
        ``binder_rosetta_cyclic_topology_declared`` (1.0 declared, 0.0 not
        declared/not requested -- never NaN, so a downstream filter doesn't need
        to special-case "no cyclization requested" separately from "failed").
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
            cyclization_type=cyclization_type,
            cyclization_i=cyclization_i,
            cyclization_j=cyclization_j,
        )
        metrics: dict[str, Any] = {}
        for key in ROSETTA_INTERFACE_SCORE_KEYS:
            col = f"binder_rosetta_{key}"
            value = raw_scores.get(key)
            metrics[col] = round(float(value), 3) if value is not None else float("nan")
        metrics["binder_rosetta_cyclic_topology_declared"] = float(
            raw_scores.get("cyclic_topology_declared", 0.0)
        )
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
