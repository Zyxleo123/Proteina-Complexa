"""Explicit cyclization chemistry, and what each predictor and scorer really supports.

Why this module exists
----------------------
A head-to-tail amide ring and a terminal disulfide are **different molecules**. Every
tool in this pipeline exposes some switch with "cyclic" in the name, and those switches
are not interchangeable:

  - Boltz's `cyclic: true` is a **head-to-tail polymer** flag on a chain. It is not a
    disulfide. A disulfide must be requested as an explicit bonded-atom pair constraint,
    on CYS SG atoms. Setting `cyclic` and calling the result a disulfide would be
    predicting the wrong molecule and labelling it with the right name.
  - AfCycDesign's contribution is a **positional encoding**: it wraps the pairwise
    sequence offset around the ring. That asserts residues 0 and L-1 are *backbone*
    neighbours, which is true exactly when the closing bond is the peptide bond. It says
    nothing about side-chain crosslinks and must not be advertised as supporting them.
    (This repo already implements and documents that restriction -- see
    `evaluation.cyclic_offset.CYCLIC_OFFSET_SUPPORTED_TYPES`.)
  - HighFold3 has two separate controls, `--head_to_tail` (a global bool) and
    `--disulfide_chain_res [[chain, i, j]]` (chain-scoped). They are different flags for
    the different chemistries, and only the disulfide one names a chain.
  - Rosetta, our scorer, declares `mainchain` and `disulfide` bonds on the pose but
    **refuses isopeptide** (`evaluation.rosetta_energy._declare_cyclic_bond`).

So "supported" is a per-(tool, chemistry) fact, and this module is the single table of
it. `CAPABILITIES` is consulted before a job is built; an unsupported combination is
recorded as `unsupported` and never predicted, rather than being run with the nearest
available switch and relabelled.

Index convention
----------------
`i` and `j` are **0-based, binder-chain-local** residue indices, which is the convention
CPSea's `cyclization_i`/`cyclization_j` use, the convention
`evaluation.rosetta_energy` maps through `PDBInfo` (`pdb2pose(chain, i + 1)`), and the
convention the whole repo writes PDBs with (binder-local index `k` <-> PDB resSeq
`k + 1`). Predictor adapters that need 1-based or chain-qualified indices convert at
their own boundary via `ChemistrySpec.as_one_based`, so exactly one place does the
arithmetic.

Terminal states
---------------
Recorded explicitly, because they are part of the molecule and part of what the scorer
has to be told. A head-to-tail ring has **no free termini**: both terminus variants come
off the pose before the bond is declared. A side-chain-closed ring keeps its free
N-terminus and C-terminus, and scoring it as if the backbone were closed would be a
different molecule again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# Chemistry names, matching `cyclization.constants.CYCLIZATION_TYPE_TO_NAME` exactly.
# Imported lazily in `assert_matches_project_constants` rather than at module import, so
# this module stays usable inside the isolated predictor venvs (which do not carry
# openfold, and `cyclization.constants` imports it).
MAINCHAIN = "mainchain"
DISULFIDE = "disulfide"
ISOPEPTIDE = "isopeptide"

TerminalState = Literal["bonded", "free"]


@dataclass(frozen=True)
class AtomBond:
    """One covalent bond, named at the atom level on both ends."""

    # Which endpoint of the (i, j) pair each atom belongs to.
    atom_i: str
    atom_j: str
    # Standard geometry for the bond itself. Sources are in
    # `cyclization.linkage_geometry.LINKAGE_GEOMETRY_REFERENCE`; repeated here as the
    # chemistry *specification*, with the tolerances frozen per tranche separately.
    length_A: float
    angle_i_deg: float
    angle_j_deg: float
    dihedral_deg: float
    # True where both torsion handednesses are chemically real, so a check must fold
    # +x and -x together. Disulfides occur at both +90 and -90 degrees.
    dihedral_symmetric: bool = False


@dataclass(frozen=True)
class ChemistrySpec:
    """The intended molecule: bond, residue identities, terminal states, index mapping."""

    name: str
    bond: AtomBond
    # Residue identities the chemistry *requires* at i and j. `None` means any residue
    # (head-to-tail closes through the backbone, so identity is unconstrained).
    required_residue_i: tuple[str, ...] | None
    required_residue_j: tuple[str, ...] | None
    # Terminal state of the binder chain's own N- and C-terminus in the closed molecule.
    n_terminus: TerminalState
    c_terminus: TerminalState
    # Human-readable statement of the ring, for manifests and the report.
    description: str

    @property
    def requires_anchor_substitution(self) -> bool:
        """Whether satisfying this chemistry can force sequence changes at i/j.

        True for disulfide (both endpoints must be CYS) and isopeptide (one endpoint
        must be LYS, the other an acid/amide). Those substitutions are real mutations and
        **count toward the mutation allowance** -- see `candidates.py`, which places them
        before sampling the free positions, so a chemistry-driven change can never push a
        candidate past its band without being counted.
        """
        return self.required_residue_i is not None or self.required_residue_j is not None

    def anchor_requirements(self, length: int) -> dict[int, tuple[str, ...]]:
        """{0-based binder-local index -> allowed residue 3-letter codes} for this ring.

        Terminal closure is assumed (i=0, j=length-1), which is what CPSea binders do for
        all three chemistries (520/520 sampled -- see `cyclization.mask`'s `terminal_only`
        docstring) and what `rosetta_energy` defaults to.
        """
        if length < 2:
            raise ValueError(f"need at least 2 residues to close a ring, got {length}")
        out: dict[int, tuple[str, ...]] = {}
        if self.required_residue_i is not None:
            out[0] = self.required_residue_i
        if self.required_residue_j is not None:
            out[length - 1] = self.required_residue_j
        return out

    @staticmethod
    def as_one_based(i: int, j: int) -> tuple[int, int]:
        """0-based binder-local -> 1-based, for tools that index residues from 1."""
        return i + 1, j + 1

    def to_dict(self) -> dict:
        return asdict(self)


# --- the chemistries ---------------------------------------------------------------
# Geometry values are the same ones `cyclization.linkage_geometry` supervises against, so
# the specification and the training-time loss cannot drift apart.

CHEMISTRIES: dict[str, ChemistrySpec] = {
    MAINCHAIN: ChemistrySpec(
        name=MAINCHAIN,
        bond=AtomBond(
            # C of the LAST residue to N of the FIRST -- the same orientation as every
            # in-chain peptide bond (residue k's C to residue k+1's N), wrapped around
            # the ring. Building N(i)->C(i) instead would bond two atoms that are each
            # already engaged in their own in-chain bond.
            atom_i="N",
            atom_j="C",
            length_A=1.33,
            angle_i_deg=121.7,  # C-N-CA at the N-terminal residue
            angle_j_deg=116.2,  # CA-C-N at the C-terminal residue
            dihedral_deg=180.0,  # trans amide
            dihedral_symmetric=False,
        ),
        required_residue_i=None,
        required_residue_j=None,
        n_terminus="bonded",
        c_terminus="bonded",
        description=(
            "Head-to-tail amide: C(last) - N(first). Backbone closure, so no residue "
            "identity is required and the chain has no free termini."
        ),
    ),
    DISULFIDE: ChemistrySpec(
        name=DISULFIDE,
        bond=AtomBond(
            atom_i="SG",
            atom_j="SG",
            length_A=2.05,
            angle_i_deg=103.8,  # CB-SG-SG
            angle_j_deg=103.8,
            dihedral_deg=90.0,  # CB-SG-SG-CB, both signs occur
            dihedral_symmetric=True,
        ),
        required_residue_i=("CYS",),
        required_residue_j=("CYS",),
        n_terminus="free",
        c_terminus="free",
        description=(
            "Terminal disulfide: SG - SG between two CYS. Side-chain closure, so the "
            "backbone termini stay FREE and both endpoints must be cysteine."
        ),
    ),
    ISOPEPTIDE: ChemistrySpec(
        name=ISOPEPTIDE,
        bond=AtomBond(
            # LYS NZ to the side-chain carbonyl carbon: CG for Asp/Asn, CD for Glu/Gln.
            # The partner atom depends on the acid residue, so `isopeptide_acid_atom`
            # resolves it rather than this field pretending there is one name.
            atom_i="NZ",
            atom_j="CG/CD",
            length_A=1.33,
            angle_i_deg=120.0,  # CE-NZ-C at the lysine
            angle_j_deg=117.0,  # NZ-C-CG at the acid
            dihedral_deg=180.0,
            dihedral_symmetric=False,
        ),
        required_residue_i=("LYS",),
        required_residue_j=("ASP", "ASN", "GLU", "GLN"),
        n_terminus="free",
        c_terminus="free",
        description=(
            "Side-chain isopeptide: LYS NZ - acid side-chain carbonyl C (CG for "
            "ASP/ASN, CD for GLU/GLN). Backbone termini stay FREE."
        ),
    ),
}


def isopeptide_acid_atom(resname: str) -> str:
    """Partner carbon for an isopeptide bond, by acid residue 3-letter code."""
    table = {"ASP": "CG", "ASN": "CG", "GLU": "CD", "GLN": "CD"}
    try:
        return table[resname.upper()]
    except KeyError as exc:
        raise ValueError(f"{resname!r} cannot carry an isopeptide bond") from exc


# --- capability table --------------------------------------------------------------

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
# "The tool runs, but the ring is invisible to it." Distinguished from `unsupported`
# because the prediction is still a real fixed-sequence complex prediction -- it just
# carries no cyclic conditioning, which is a conditioning difference to record, not a
# silent equivalence.
LINEAR_ONLY = "linear_only"


@dataclass(frozen=True)
class Capability:
    status: str
    # Exactly how the chemistry is requested from this tool. `None` when unsupported.
    control: str | None = None
    note: str = ""
    # True where the statement is still unverified by execution in THIS project, so the
    # report can separate "documented" from "measured".
    verified: bool = False


# Rows are (tool, chemistry). Every claim here is traceable either to code in this repo
# or to a named line in the pinned upstream clone; `note` says which.
CAPABILITIES: dict[tuple[str, str], Capability] = {
    # ---- Boltz-2 (upstream/boltz b1ebfc46) ----
    ("boltz2", MAINCHAIN): Capability(
        status=SUPPORTED,
        control="`cyclic: true` on the peptide chain in the YAML input",
        note=(
            "Boltz's own head-to-tail polymer flag. Documented in docs/prediction.md. "
            "Must NOT be used to stand in for a disulfide."
        ),
    ),
    ("boltz2", DISULFIDE): Capability(
        status=SUPPORTED,
        control="explicit `bond` constraint between the two CYS SG atoms (NOT `cyclic`)",
        note=(
            "Requested as a bonded-atom-pair constraint naming chain, residue index and "
            "atom on both ends. `cyclic: true` would request a different molecule."
        ),
    ),
    ("boltz2", ISOPEPTIDE): Capability(
        status=SUPPORTED,
        control="explicit `bond` constraint LYS NZ - acid CG/CD",
        note=(
            "Expressible as a bond constraint, but the shared Rosetta evaluator cannot "
            "declare it, so these predictions would be unscorable by our own scorer -- "
            "excluded from the tranche for that reason, not for lack of predictor support."
        ),
    ),
    # ---- AfCycDesign / cyclic ColabDesign ----
    ("afcycdesign", MAINCHAIN): Capability(
        status=SUPPORTED,
        control="cyclic sequence-offset wrap on the binder block (`cyclic_offset`)",
        note=(
            "Already implemented in this repo: evaluation/cyclic_offset.py, wired into "
            "rewards/alphafold2_reward.py with `cyclic_offset_applied` recorded per call."
        ),
        verified=True,
    ),
    ("afcycdesign", DISULFIDE): Capability(
        status=LINEAR_ONLY,
        control=None,
        note=(
            "The offset wrap is backbone-only and is restricted to mainchain by "
            "CYCLIC_OFFSET_SUPPORTED_TYPES. AF2 has no bonded-atom constraint, so a "
            "disulfide ring can be predicted only as an unconstrained linear peptide "
            "whose two CYS may or may not approach. Recorded as a conditioning "
            "difference; never labelled as cyclic support."
        ),
        verified=True,
    ),
    ("afcycdesign", ISOPEPTIDE): Capability(
        status=LINEAR_ONLY,
        control=None,
        note="Same as disulfide: side-chain closure is invisible to a backbone offset.",
        verified=True,
    ),
    # ---- HighFold3 (upstream/highfold3 2ff1ea31) ----
    ("highfold3", MAINCHAIN): Capability(
        status=SUPPORTED,
        control="`--head_to_tail 1`, with the PEPTIDE AS THE FIRST CHAIN",
        note=(
            "A global bool with no chain argument. Probed directly against the "
            "cpython-311 bytecode (the only form the module ships in): "
            "calc_cyclic_matrix_af3 sizes its matrix from where residue_index returns "
            "to 1, so the ring lands on the FIRST chain. Peptide first + receptor "
            "second gives a correct 5x5 ring on a 5-mer; receptor first gives an 8x8 "
            "matrix and rings the RECEPTOR, leaving the peptide linear. AF3's own "
            "examples list the protein first, so the natural ordering is the wrong one. "
            "The adapter writes the peptide first."
        ),
        verified=True,
    ),
    ("highfold3", DISULFIDE): Capability(
        status=SUPPORTED,
        control="`--disulfide_chain_res [[1, i, j]]` (1-based), peptide as the first chain",
        note=(
            "Documented as [[chain, Cys_i, Cys_j]], but the CHAIN FIELD IS IGNORED: "
            "[[1,1,6]] and [[2,1,6]] return bit-identical matrices on both the "
            "residue-wise and token-wise paths. The crosslink always applies to the "
            "first chain's block, so correctness depends on peptide-first ordering, not "
            "on the chain id. Residue indices are 1-based (probed: [[1,2,5]] puts the "
            "+/-1 offset between 0-based rows 1 and 4). Unlike the AF2 offset wrap this "
            "encodes a SIDE-CHAIN crosslink as a graph shortest path, which is why "
            "highfold3 can express a disulfide ring that afcycdesign cannot."
        ),
        verified=True,
    ),
    ("highfold3", ISOPEPTIDE): Capability(
        status=UNSUPPORTED,
        control=None,
        note="No documented control; only head-to-tail and disulfide are exposed.",
    ),
    # ---- the shared scorer ----
    ("rosetta", MAINCHAIN): Capability(
        status=SUPPORTED,
        control="strip terminus variants, then `declare_chemical_bond(i,'N',j,'C')`",
        note="evaluation/rosetta_energy.py::_declare_cyclic_bond.",
        verified=True,
    ),
    ("rosetta", DISULFIDE): Capability(
        status=SUPPORTED,
        control="`Conformation.add_disulfide_bond(i, j)`",
        note="Raises if both residues are not really CYS, rather than scoring silently.",
        verified=True,
    ),
    ("rosetta", ISOPEPTIDE): Capability(
        status=UNSUPPORTED,
        control=None,
        note=(
            "An arbitrary LYS-NZ <-> acid side-chain bond needs a residue-connection "
            "patch plain amino-acid types do not carry. Reported as "
            "binder_rosetta_cyclic_topology_declared=False, never scored as closed."
        ),
        verified=True,
    ),
}

PREDICTORS = ("boltz2", "afcycdesign", "highfold3")
SCORERS = ("rosetta",)


def capability(tool: str, chemistry: str) -> Capability:
    try:
        return CAPABILITIES[(tool, chemistry)]
    except KeyError as exc:
        raise KeyError(f"no capability recorded for ({tool!r}, {chemistry!r})") from exc


def common_chemistries(tools: tuple[str, ...] = PREDICTORS + SCORERS) -> list[str]:
    """Chemistries every listed tool genuinely supports.

    `LINEAR_ONLY` does not count as support: it means the ring is invisible to that tool.
    This is the gate that picks the chemistry for the cross-model comparison, so the
    comparison cannot quietly include an arm that was never told about the ring.
    """
    return [
        name
        for name in CHEMISTRIES
        if all(capability(tool, name).status == SUPPORTED for tool in tools)
    ]


def capability_table_rows() -> list[dict]:
    """Flat rows for the manifest / report, one per (tool, chemistry)."""
    rows = []
    for (tool, chem), cap in CAPABILITIES.items():
        rows.append(
            {
                "tool": tool,
                "chemistry": chem,
                "status": cap.status,
                "control": cap.control,
                "verified_by_execution": cap.verified,
                "note": cap.note,
            }
        )
    return sorted(rows, key=lambda r: (r["tool"], r["chemistry"]))


def assert_matches_project_constants() -> None:
    """Fail loudly if these chemistry names drift from `cyclization.constants`.

    Two vocabularies for the same three molecules is a bug that is nearly invisible in
    aggregate, so it is checked rather than assumed. Only callable where openfold is
    importable (the complexa venv), which is why it is not run at import time.
    """
    from proteinfoundation.cyclization.constants import CYCLIZATION_TYPE_TO_NAME

    project = set(CYCLIZATION_TYPE_TO_NAME.values())
    here = set(CHEMISTRIES)
    if project != here:
        raise AssertionError(
            f"chemistry vocabulary drift: cyclization.constants has {sorted(project)}, "
            f"cpseq.chemistry has {sorted(here)}"
        )
    from proteinfoundation.cyclization.constants import ISOPEPTIDE_ACID_CARBON_ATOM

    for resname, atom in ISOPEPTIDE_ACID_CARBON_ATOM.items():
        if isopeptide_acid_atom(resname) != atom:
            raise AssertionError(
                f"isopeptide partner atom for {resname} disagrees: "
                f"{isopeptide_acid_atom(resname)} here vs {atom} in cyclization.constants"
            )
