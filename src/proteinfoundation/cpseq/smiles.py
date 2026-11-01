"""Cyclic-peptide SMILES, for handing a macrocycle to AlphaFold 3 as a ligand.

Why a ligand at all
-------------------
AF3 has no way to be told that a POLYMER chain is cyclic: its relative position
encoding is a clipped linear offset, so for a head-to-tail 14-mer residues 0 and 13 read
as 13 apart when they are covalently bonded. (This repo documents the identical failure
for AF2 in `evaluation/cyclic_offset.py`, and for ProteinMPNN in
`README_CPSEQ_MPNN_FILL_SETTINGS.md` section 5.4.) HighFold3 patches that encoding; the
other route, taken here, is to stop describing the peptide as a polymer and describe it
as a molecule, where the closing bond is simply one of its bonds.

So the constraint is expressed in the chemistry rather than in a restraint, which is the
point: nothing has to hold the ring shut during prediction, because an open ring is not a
conformation of this molecule.

What it costs, and what to watch
--------------------------------
- **Tokens.** AF3 tokenises a ligand per ATOM, where a polymer chain is one token per
  residue. A 14-mer macrocycle is ~105-130 heavy atoms, so the peptide goes from 14
  tokens to ~120. `CyclicPeptide.n_heavy_atoms` is that count; it is reported so the
  cost is visible before a run rather than inferred from a wall clock afterwards.
- **Distribution.** AF3's ligands are overwhelmingly small molecules. A peptide-sized
  SMILES ligand is in-vocabulary but off-distribution, and no claim here assumes
  otherwise -- which is why the arm is calibrated on the six deposited crystal
  macrocycles before any design is folded.
- **Protonation.** `MolFromSequence` yields the NEUTRAL forms: Asp/Glu as COOH, Lys/Arg
  as free amines. That is kept rather than adjusted to pH 7, because the downstream
  Rosetta protocol applies its own protonation policy (recorded per row in `scorer`) and
  two different policies in one pipeline is worse than one explicit choice. Recorded in
  `protonation` so it travels with the row.

Stereochemistry is NOT hand-written here. It comes from RDKit's own residue templates
via `Chem.MolFromSequence`, so the L-configuration is correct by construction; writing
SMILES by string concatenation is how a D-amino acid gets into a design silently.
`verify()` re-checks it anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

# The 20 canonical residues. Deliberately not a superset: a letter this module does not
# know must raise, because `MolFromSequence` accepts things like 'X' and 'B' and would
# otherwise return a molecule that is not the peptide the caller asked for.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# One backbone N, one CA, one carbonyl C per residue, and the ring closes through all of
# them. A head-to-tail macrocycle of L residues therefore has a ring of exactly 3L atoms,
# which is a cheap and complete check that the closure landed where it was meant to.
BACKBONE_RING_ATOMS_PER_RESIDUE = 3

SUPPORTED_CHEMISTRIES = ("mainchain",)


class SmilesBuildError(ValueError):
    """The requested peptide could not be built as the molecule it claims to be."""


@dataclass(frozen=True)
class AtomRecord:
    """Where one heavy atom of the SMILES came from, in peptide terms."""

    index: int
    residue_number: int   # 1-based, N-terminus first
    residue_name: str     # three-letter, e.g. 'TYR'
    atom_name: str        # PDB atom name, e.g. 'CA'


@dataclass(frozen=True)
class CyclicPeptide:
    sequence: str
    chemistry: str
    smiles: str
    n_heavy_atoms: int
    macrocycle_ring_size: int
    atom_records: tuple[AtomRecord, ...]
    # Position k of the SMILES string is molecule atom `smiles_atom_order[k]`.
    #
    # This is the whole bridge back from a folder's output. A predictor handed a SMILES
    # returns coordinates in the order of THAT STRING, which is not the order of the
    # molecule RDKit built -- `MolFromSequence` emits N, CA, C, O per residue while
    # canonical SMILES output starts wherever canonicalisation chooses. Measured on
    # INPYLYP: SMILES position 0 is molecule atom 7 (ILE1 CD1), not atom 0 (ILE1 N).
    # Mapping by position without this permutation silently assigns every coordinate to
    # the wrong atom, and the result still looks like a peptide.
    smiles_atom_order: tuple[int, ...] = ()
    protonation: str = "neutral (COOH / free amine), as RDKit residue templates give it"
    mol: Chem.Mol | None = field(default=None, repr=False, compare=False)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def residue_atom_indices(self, residue_number: int) -> tuple[int, ...]:
        return tuple(r.index for r in self.atom_records
                     if r.residue_number == residue_number)

    def atom_index(self, residue_number: int, atom_name: str) -> int:
        """Index of one named atom, e.g. the closing bond's N or C.

        The closure geometry is measured between specific atoms, so the caller needs to
        name them rather than guess from order.
        """
        for r in self.atom_records:
            if r.residue_number == residue_number and r.atom_name == atom_name:
                return r.index
        raise SmilesBuildError(
            f"no atom {atom_name!r} in residue {residue_number} of {self.sequence}"
        )

    def closure_atom_indices(self) -> tuple[int, int]:
        """(C of the last residue, N of the first) -- the bond that makes it a ring."""
        return self.atom_index(self.length, "C"), self.atom_index(1, "N")

    def smiles_position(self, atom_index: int) -> int:
        """Where molecule atom `atom_index` sits in the SMILES string, and so in the
        predictor's output atom list."""
        try:
            return self.smiles_atom_order.index(atom_index)
        except ValueError:
            raise SmilesBuildError(
                f"atom {atom_index} is not in the SMILES output order"
            ) from None

    def predicted_atom_names(self) -> tuple[str, ...]:
        """The atom names a predictor is expected to give the ligand, in output order.

        AlphaFold 3 names a SMILES ligand's atoms `<element><running index per element>`
        in the order of the SMILES string: C1, C2, ..., N1, ... Verified atom-for-atom
        against AF3's own output for INPYLYP (62 of 62 names identical), which is what
        makes the mapping back to residues checkable rather than assumed.

        Returning the EXPECTED names lets the reader assert them against the file. If a
        predictor ever changes its naming, that assertion fails loudly instead of the
        coordinates being attached to the wrong residues.
        """
        if self.mol is None or not self.smiles_atom_order:
            raise SmilesBuildError("no molecule or SMILES order to name atoms from")
        counts: dict[str, int] = {}
        names = []
        for atom_index in self.smiles_atom_order:
            symbol = self.mol.GetAtomWithIdx(atom_index).GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
            names.append(f"{symbol}{counts[symbol]}")
        return tuple(names)

    def records_in_smiles_order(self) -> tuple[AtomRecord, ...]:
        """`atom_records` permuted into the predictor's output order."""
        return tuple(self.atom_records[i] for i in self.smiles_atom_order)


def _terminal_atoms(mol: Chem.Mol, length: int) -> tuple[int, int]:
    n_term = c_term = None
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise SmilesBuildError(
                "RDKit returned a peptide without PDB residue info; the atom-to-residue "
                "mapping this module depends on is unavailable"
            )
        name = info.GetName().strip()
        number = info.GetResidueNumber()
        if number == 1 and name == "N":
            n_term = atom.GetIdx()
        elif number == length and name == "C":
            c_term = atom.GetIdx()
    if n_term is None or c_term is None:
        raise SmilesBuildError(
            f"could not locate the termini of {length}-mer (N1={n_term}, C{length}={c_term})"
        )
    return n_term, c_term


def _carboxyl_hydroxyl(mol: Chem.Mol, c_term: int) -> int:
    """The -OH oxygen of the C-terminal carboxyl, the atom condensation removes.

    Picked by bond order and degree, not by index: the carbonyl O and the hydroxyl O are
    both oxygens on the same carbon, and taking the wrong one builds a ketone-bridged
    molecule that still sanitises.
    """
    for neighbour in mol.GetAtomWithIdx(c_term).GetNeighbors():
        if neighbour.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(c_term, neighbour.GetIdx())
        if bond.GetBondType() == Chem.BondType.SINGLE and neighbour.GetDegree() == 1:
            return neighbour.GetIdx()
    raise SmilesBuildError("no C-terminal hydroxyl oxygen to condense")


def cyclic_peptide(sequence: str, chemistry: str = "mainchain") -> CyclicPeptide:
    """Build a head-to-tail cyclic peptide and describe it atom by atom.

    `chemistry` is required and checked rather than defaulted silently, because a
    disulfide or isopeptide macrocycle is a different molecule and returning the
    head-to-tail one for it would be a wrong answer dressed as a right one.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        raise SmilesBuildError("empty sequence")
    if chemistry not in SUPPORTED_CHEMISTRIES:
        raise SmilesBuildError(
            f"chemistry {chemistry!r} is not supported here (have {SUPPORTED_CHEMISTRIES}). "
            "A disulfide or isopeptide ring is a different molecule; build it explicitly "
            "rather than letting this return the head-to-tail one."
        )
    bad = sorted(set(sequence) - set(CANONICAL_AA))
    if bad:
        raise SmilesBuildError(
            f"non-canonical residue(s) {bad} in {sequence!r}; "
            f"this module builds only {CANONICAL_AA}"
        )

    linear = Chem.MolFromSequence(sequence)
    if linear is None:
        raise SmilesBuildError(f"RDKit could not build a peptide from {sequence!r}")

    length = len(sequence)
    observed = max(a.GetPDBResidueInfo().GetResidueNumber() for a in linear.GetAtoms())
    if observed != length:
        raise SmilesBuildError(
            f"RDKit built {observed} residues from a {length}-mer; refusing to cyclise"
        )

    n_term, c_term = _terminal_atoms(linear, length)
    hydroxyl = _carboxyl_hydroxyl(linear, c_term)

    editable = Chem.RWMol(linear)
    editable.RemoveAtom(hydroxyl)
    # RemoveAtom renumbers everything above the removed index.
    shift = lambda i: i - 1 if i > hydroxyl else i  # noqa: E731
    editable.AddBond(shift(c_term), shift(n_term), Chem.BondType.SINGLE)

    mol = editable.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001
        raise SmilesBuildError(f"cyclised {sequence!r} does not sanitise: {exc}") from exc
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    records = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        records.append(
            AtomRecord(
                index=atom.GetIdx(),
                residue_number=info.GetResidueNumber(),
                residue_name=info.GetResidueName().strip(),
                atom_name=info.GetName().strip(),
            )
        )

    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    # MolToSmiles records the permutation it used as a private computed property, as a
    # bracketed string. Read immediately after the call that sets it, since any later
    # MolToSmiles on the same molecule overwrites it.
    smiles_text = Chem.MolToSmiles(mol)
    try:
        raw = mol.GetProp("_smilesAtomOutputOrder")
        atom_order = tuple(
            int(tok) for tok in raw.strip("[],").split(",") if tok.strip()
        )
    except KeyError:
        raise SmilesBuildError(
            "RDKit did not record _smilesAtomOutputOrder, so the SMILES cannot be "
            "mapped back to residues. Refusing to emit a SMILES whose output could not "
            "be read back."
        ) from None
    if sorted(atom_order) != list(range(mol.GetNumAtoms())):
        raise SmilesBuildError(
            f"SMILES output order is not a permutation of the {mol.GetNumAtoms()} atoms"
        )

    built = CyclicPeptide(
        sequence=sequence,
        chemistry=chemistry,
        smiles=smiles_text,
        n_heavy_atoms=mol.GetNumAtoms(),
        macrocycle_ring_size=max(ring_sizes) if ring_sizes else 0,
        atom_records=tuple(records),
        smiles_atom_order=atom_order,
        mol=mol,
    )
    verify(built)
    return built


def verify(peptide: CyclicPeptide) -> None:
    """Re-check the molecule is the one that was asked for. Raises on any failure.

    Called on every build. These are cheap, and each one corresponds to a way a
    plausible-looking SMILES is the wrong molecule: a peptide that did not close, one
    that closed through a sidechain, one that fragmented, or one carrying a D-residue.
    """
    mol = peptide.mol
    if mol is None:
        raise SmilesBuildError("nothing to verify")

    if len(Chem.GetMolFrags(mol)) != 1:
        raise SmilesBuildError(
            f"{peptide.sequence}: {len(Chem.GetMolFrags(mol))} fragments -- the ring did "
            "not close, so this is not one molecule"
        )

    expected_ring = BACKBONE_RING_ATOMS_PER_RESIDUE * peptide.length
    if peptide.macrocycle_ring_size != expected_ring:
        raise SmilesBuildError(
            f"{peptide.sequence}: largest ring is {peptide.macrocycle_ring_size} atoms, "
            f"expected {expected_ring} (3 per residue). A different size means the "
            "closure went through a sidechain rather than backbone N-CA-C."
        )

    # The macrocycle must contain the backbone N, CA and C of EVERY residue. Ring size
    # alone can be hit by the wrong atoms.
    macrocycle = max(mol.GetRingInfo().AtomRings(), key=len)
    in_ring = {peptide.atom_records[i] for i in macrocycle}
    for residue in range(1, peptide.length + 1):
        for name in ("N", "CA", "C"):
            if not any(r.residue_number == residue and r.atom_name == name
                       for r in in_ring):
                raise SmilesBuildError(
                    f"{peptide.sequence}: residue {residue} atom {name} is not in the "
                    "macrocycle; the closure is not head-to-tail through the backbone"
                )

    # The closing bond has to exist as a bond, not merely as two atoms in one ring.
    c_idx, n_idx = peptide.closure_atom_indices()
    if mol.GetBondBetweenAtoms(c_idx, n_idx) is None:
        raise SmilesBuildError(
            f"{peptide.sequence}: no bond between C{peptide.length} and N1"
        )

    # Stereo. Every alpha carbon of an L-residue is assigned; glycine has no centre, and
    # L-cysteine is R where the other nineteen are S (CIP, because sulfur outranks the
    # carboxyl). An UNASSIGNED centre is the real failure mode -- it means the SMILES
    # does not commit to a configuration and the folder may place either.
    unassigned = [i for i, tag in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False) if tag == "?"]
    if unassigned:
        offenders = sorted({peptide.atom_records[i].residue_number for i in unassigned})
        raise SmilesBuildError(
            f"{peptide.sequence}: unassigned stereocentres in residue(s) {offenders}; "
            "an unspecified configuration lets the folder choose a D-residue"
        )

    centres = dict(Chem.FindMolChiralCenters(
        mol, includeUnassigned=False, useLegacyImplementation=False))
    for record in peptide.atom_records:
        if record.atom_name != "CA":
            continue
        tag = centres.get(record.index)
        if record.residue_name == "GLY":
            if tag is not None:
                raise SmilesBuildError(
                    f"{peptide.sequence}: glycine {record.residue_number} has a "
                    "stereocentre at CA, which it cannot"
                )
            continue
        expected = "R" if record.residue_name == "CYS" else "S"
        if tag != expected:
            raise SmilesBuildError(
                f"{peptide.sequence}: CA of {record.residue_name}"
                f"{record.residue_number} is {tag}, expected {expected} -- that is a "
                "D-residue, not the L-peptide requested"
            )


def sequence_from_records(records: tuple[AtomRecord, ...]) -> str:
    """Read the sequence back out of the atom annotation.

    Used as a round-trip check and, more importantly, by the structure reader: once AF3
    returns coordinates for a bag of ligand atoms, this is what says which residue each
    one belongs to.
    """
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
        "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
        "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
        "TRP": "W", "TYR": "Y",
    }
    by_residue: dict[int, str] = {}
    for record in records:
        by_residue.setdefault(record.residue_number, record.residue_name)
    out = []
    for number in sorted(by_residue):
        name = by_residue[number]
        if name not in three_to_one:
            raise SmilesBuildError(f"residue {number} is {name!r}, not canonical")
        out.append(three_to_one[name])
    return "".join(out)
