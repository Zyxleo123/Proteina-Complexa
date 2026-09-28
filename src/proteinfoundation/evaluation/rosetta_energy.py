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
run, for all three chemistries:

* **mainchain** -- `Conformation.declare_chemical_bond` after stripping the
  terminus variants that would otherwise cap the N and C connections.
* **disulfide** -- `Conformation.add_disulfide_bond`.
* **isopeptide** -- Rosetta's own `SidechainConjugation` patch (database
  `chemical/residue_type_sets/fa_standard/patches/SidechainConjugation.txt`) opens a
  real `CONNECT` at LYS:NZ / ASP:CG / GLU:CD, then `DeclareBond`. The patch is what
  makes the bond chemically real: without it NZ keeps its three ammonium protons and
  the carboxylate keeps both oxygens, so the declared bond would imply a five-bonded
  nitrogen.

Every chemistry therefore ends with `residue(i).is_bonded(j)` true, and an
undeclarable request raises rather than scoring an open topology --
`binder_rosetta_cyclic_topology_declared` has no False path left. The equivalent
logic, plus geometry measurement and ring constraints, lives in
`script_utils/cp_topology.py` for the GenKIC search arm; the two are kept to one
behaviour deliberately.
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


# Which atom is the amine and which the acyl carbon, per residue type, for an
# isopeptide bond: {name3: (bridging atom, its ring-interior neighbour)}.
_ISOPEPTIDE_AMINE = {"LYS": ("NZ", "CE")}
_ISOPEPTIDE_ACYL = {"ASP": ("CG", "CB"), "GLU": ("CD", "CG")}


def _binder_seqpos(pose, binder_chain: str, local_index: int) -> int:
    """Pose sequence position of binder-local residue `local_index`.

    Maps by chain membership and POSE ORDER, not by PDB residue number. The older
    `pdb2pose(binder_chain, local_index + 1)` lookup assumed the binder is numbered
    1..L, which holds for CPSea-generated PDBs but not for staged crystal complexes
    that preserve their source numbering -- chain B of `LNR_2qos_C_A` runs 158..168,
    where that lookup returns 0 and the caller raises.

    Raises:
        ValueError: the chain is absent, or `local_index` is out of range for it.
    """
    pdb_info = pose.pdb_info()
    positions = [
        seqpos
        for seqpos in range(1, pose.size() + 1)
        if pdb_info.chain(seqpos) == binder_chain
    ]
    if not positions:
        raise ValueError(
            f"no residues on binder chain {binder_chain!r}; chains present: "
            f"{sorted({pdb_info.chain(s) for s in range(1, pose.size() + 1)})}"
        )
    if not 0 <= local_index < len(positions):
        raise ValueError(
            f"binder-local index {local_index} out of range for chain {binder_chain!r} "
            f"(length {len(positions)})"
        )
    return positions[local_index]


def _isopeptide_atoms(aa_i: str, aa_j: str) -> dict[str, str]:
    """Resolve the isopeptide bridging atoms for anchors `aa_i`/`aa_j`.

    Either terminus may carry the lysine; the other must be Asp or Glu.

    Raises:
        ValueError: the pair is not one amine plus one acyl donor.
    """
    if aa_i in _ISOPEPTIDE_AMINE and aa_j in _ISOPEPTIDE_ACYL:
        (atom_i, next_i), (atom_j, prev_j) = _ISOPEPTIDE_AMINE[aa_i], _ISOPEPTIDE_ACYL[aa_j]
    elif aa_i in _ISOPEPTIDE_ACYL and aa_j in _ISOPEPTIDE_AMINE:
        (atom_i, next_i), (atom_j, prev_j) = _ISOPEPTIDE_ACYL[aa_i], _ISOPEPTIDE_AMINE[aa_j]
    else:
        raise ValueError(
            f"isopeptide needs one LYS and one ASP/GLU anchor, got aa_i={aa_i!r} aa_j={aa_j!r}"
        )
    return {"atom_i": atom_i, "atom_j": atom_j, "prev_j": prev_j, "next_i": next_i}


def _sidechain_connection_id(pose, seqpos: int, atom: str) -> int:
    """Residue-connection id at `seqpos` whose connect atom is `atom`.

    Looked up rather than assumed: the side-chain connection's index depends on how
    many polymer connections the residue still has, so a residue carrying a
    terminus variant numbers it differently from an internal one.

    Raises:
        ValueError: no connection on this residue uses `atom`.
    """
    residue = pose.residue(seqpos)
    target = residue.atom_index(atom)
    n_conn = residue.type().n_possible_residue_connections()
    for conn_id in range(1, n_conn + 1):
        if residue.residue_connect_atom_index(conn_id) == target:
            return conn_id
    used = [
        residue.atom_name(residue.residue_connect_atom_index(c)).strip()
        for c in range(1, n_conn + 1)
    ]
    raise ValueError(
        f"residue {seqpos} ({residue.name()}) has no residue connection at atom "
        f"{atom!r}; connections use atoms {used}"
    )


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
            sequence positions by POSE ORDER within `binder_chain` (see
            `_binder_seqpos`), which is correct for any residue numbering.

    Returns:
        True -- every supported chemistry ends with a real covalent connection on
        the pose. There is no longer a "declared=False" path: an undeclarable
        request raises instead of quietly scoring an open topology.

    Raises:
        ValueError: the endpoints are out of range for the binder chain, the
            anchors carry the wrong identities for the requested chemistry, or
            Rosetta refused the bond.
    """
    import pyrosetta.rosetta as rosetta
    from pyrosetta.rosetta.core.chemical import VariantType
    from pyrosetta.rosetta.core.pose import (
        add_variant_type_to_pose_residue,
        remove_variant_type_from_pose_residue,
    )

    seqpos_i = _binder_seqpos(pose, binder_chain, i_local)
    seqpos_j = _binder_seqpos(pose, binder_chain, j_local)

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
        # Rosetta's own SidechainConjugation patch opens a real CONNECT at LYS:NZ /
        # ASP:CG / GLU:CD, strips the extra protons (3 HZ -> 1 amide HZ) and the second
        # carboxyl oxygen (OD1+OD2 -> one amide OD), and retypes the atoms Nbb/CObb.
        # Declaring NZ--CG *without* it leaves NZ carrying three protons, i.e. a
        # five-bonded nitrogen, so the "bond" would be a chemical fiction.
        atoms = _isopeptide_atoms(
            pose.residue(seqpos_i).name3(), pose.residue(seqpos_j).name3()
        )
        for seqpos in (seqpos_i, seqpos_j):
            add_variant_type_to_pose_residue(pose, VariantType.SIDECHAIN_CONJUGATION, seqpos)
        declare = rosetta.protocols.simple_moves.DeclareBond()
        # add_termini=False, run_kic=False, kic_res1=0, kic_res2=0, rebuild_fold_tree=True
        declare.set(seqpos_j, atoms["atom_j"], seqpos_i, atoms["atom_i"], False, False, 0, 0, True)
        declare.apply(pose)
        # The patch's new atoms (the amide HZ, the single OD/OE, the virtual) have ICOORs
        # defined relative to the connection, so they sit at stale positions until rebuilt.
        # The connection id is looked up rather than assumed to be 3: a residue still
        # carrying a terminus variant has one fewer polymer connection, which shifts it.
        for seqpos, atom in ((seqpos_i, atoms["atom_i"]), (seqpos_j, atoms["atom_j"])):
            pose.conformation().rebuild_residue_connection_dependent_atoms(
                seqpos, _sidechain_connection_id(pose, seqpos, atom)
            )
        if not pose.residue(seqpos_i).is_bonded(seqpos_j):
            raise ValueError(
                f"isopeptide: declared the bond but pose residue {seqpos_i} still reports "
                f"not bonded to {seqpos_j} -- refusing to score an open topology as closed"
            )
        return True

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
