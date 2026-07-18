"""Parses CPSea cyclization labels (binder-local endpoints + type).

Ground truth `type` is determined *exclusively* by the identity of the atoms
actually involved in the observed cyclization bond, read from `CONECT`
records in the preprocessed PDB:

    MAINCHAIN:  backbone terminal C <-> terminal N bond
    DISULFIDE:  CYS SG <-> CYS SG bond
    ISOPEPTIDE: LYS side-chain amine (NZ) <-> ASP/GLU/ASN/GLN side-chain
                carboxyl/carboxamide atom bond

Residue identity and terminal position are used only to *locate* / *index*
a bond that was already observed -- never to guess a type in the absence
of one. Concretely, we deliberately do NOT:
  - label MAINCHAIN just because a candidate pair happens to be the chain
    termini (we require an actual observed N<->C bond there too),
  - label DISULFIDE just because both residues happen to be CYS (we require
    the actual observed SG<->SG bond),
  - label ISOPEPTIDE just because the residue types match (we require the
    actual observed NZ<->carboxyl/carboxamide-atom bond).

The dataset metadata `cyclization_type` string column (written by
`script_utils/preprocess_cpsea.py`) is a coarse summary of the very same
CONECT records (it cannot distinguish DISULFIDE from ISOPEPTIDE at the
residue-index level we need, and lumps all non-head_tail/non-disulfide
bonds -- which in practice is *every* LYS:NZ<->ASP:CG isopeptide bond in
CPSea -- into "other"). It is therefore never used to assign `i`/`j`/`type`
directly; it is only cross-checked for logging (see `_labeled`).

If the actual bond can't be found and classified from CONECT records,
`has_cyclization` is False and the sample is excluded from the linkage
loss -- we never guess.
"""

from __future__ import annotations

import gzip
import os

from proteinfoundation.cyclization.constants import DISULFIDE, ISOPEPTIDE, MAINCHAIN, NO_CYCLIZATION_INDEX

# Side-chain atoms that can participate in an isopeptide bond as the
# carboxyl/carboxamide partner of a LYS NZ amine. Empirically, CPSea's CONECT
# records bond to the side-chain carbonyl carbon (CG for Asp/Asn, CD for
# Glu/Gln) rather than the oxygens, so both are accepted.
_CARBOXYL_OR_CARBOXAMIDE_ATOMS = {
    "ASP": {"OD1", "OD2", "CG"},
    "GLU": {"OE1", "OE2", "CD"},
    "ASN": {"OD1", "ND2", "CG"},
    "GLN": {"OE1", "NE2", "CD"},
}

# Coarse metadata hint expected for each type, used only for diagnostic
# cross-checking (never for assigning the label). ISOPEPTIDE has no
# dedicated hint value -- `preprocess_cpsea.py` lumps it into "other".
_EXPECTED_HINT_FOR_TYPE = {MAINCHAIN: "head_tail", DISULFIDE: "disulfide"}


def _no_label(reason: str) -> dict:
    return {
        "i": NO_CYCLIZATION_INDEX,
        "j": NO_CYCLIZATION_INDEX,
        "type": NO_CYCLIZATION_INDEX,
        "has_cyclization": False,
        "reason": reason,
    }


def _labeled(i: int, j: int, cyclization_type: int, reason: str, hint: str) -> dict:
    expected_hint = _EXPECTED_HINT_FOR_TYPE.get(cyclization_type)
    if hint and expected_hint is not None and hint != expected_hint:
        reason = f"{reason}(hint_mismatch:{hint})"
    return {"i": i, "j": j, "type": cyclization_type, "has_cyclization": True, "reason": reason}


def _open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def read_atom_conect_records(pdb_path: str) -> tuple[dict[int, dict], list[tuple[int, int]]]:
    """Parses ATOM/HETATM and CONECT records from a PDB file.

    Returns:
        (serial_to_atom, conect_pairs): `serial_to_atom` maps PDB atom serial
        -> {"atom_name", "resname", "chain_id", "resseq"}. `conect_pairs` is
        a de-duplicated list of (serial_a, serial_b) with serial_a < serial_b.
    """
    serial_to_atom: dict[int, dict] = {}
    pairs: set[tuple[int, int]] = set()
    with _open_text(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    serial = int(line[6:11])
                    atom_name = line[12:16].strip()
                    resname = line[17:20].strip()
                    chain_id = line[21]
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                serial_to_atom[serial] = {
                    "atom_name": atom_name,
                    "resname": resname,
                    "chain_id": chain_id,
                    "resseq": resseq,
                }
            elif line.startswith("CONECT"):
                content = line.rstrip("\n")
                serials = [int(content[i : i + 5]) for i in range(6, len(content), 5) if content[i : i + 5].strip()]
                if len(serials) < 2:
                    continue
                center = serials[0]
                for partner in serials[1:]:
                    if partner == center:
                        continue
                    pairs.add((min(center, partner), max(center, partner)))
    return serial_to_atom, sorted(pairs)


def _is_lys_nz(name: str, res: str) -> bool:
    return name == "NZ" and res == "LYS"


def _is_carboxyl_or_carboxamide(name: str, res: str) -> bool:
    return res in _CARBOXYL_OR_CARBOXAMIDE_ATOMS and name in _CARBOXYL_OR_CARBOXAMIDE_ATOMS[res]


def infer_cyclization_label(
    pdb_path: str | None,
    residue_pdb_idx,
    binder_length_hint: int | None = None,
    binder_chain_id: str = "B",
    cyclization_type_hint: str | None = None,
) -> dict:
    """Infers the binder-local cyclization label for one CPSea sample.

    Args:
        pdb_path: Path to the (preprocessed) PDB file with remapped CONECT
            records, or None if unavailable.
        residue_pdb_idx: Sequence of length L_b mapping binder-local index
            -> original PDB `resSeq` for that residue (post binder-only
            filtering, so index 0 is the first binder residue).
        binder_length_hint: Optional expected binder length (e.g. from
            metadata `peptide_length`) used only to detect whether the
            binder was cropped -- if cropped, we cannot trust that local
            index 0 / L_b - 1 are the true chain termini, so MAINCHAIN
            inference is skipped in that case (DISULFIDE / ISOPEPTIDE look
            up specific residues by resSeq, so cropping doesn't affect them).
        binder_chain_id: Chain id of the binder in `pdb_path` (default "B",
            matching `script_utils/preprocess_cpsea.py`'s chain rename).
        cyclization_type_hint: Optional coarse type string from dataset
            metadata (`"head_tail"`, `"disulfide"`, `"other"`, ...). Used
            *only* as a diagnostic cross-check recorded in `reason` on
            mismatch -- never to assign `i`/`j`/`type`.

    Returns:
        Dict with `i`, `j` (binder-local indices, i < j), `type` (int, or
        -1 if unknown), `has_cyclization` (bool), and `reason` (str, for
        debugging / sanity summaries).
    """
    residue_pdb_idx = [int(v.item()) if hasattr(v, "item") else int(v) for v in residue_pdb_idx]
    l_b = len(residue_pdb_idx)
    if l_b < 2:
        return _no_label("binder_too_short")

    resseq_to_local: dict[int, int] = {}
    for local_idx, resseq in enumerate(residue_pdb_idx):
        # First occurrence wins; duplicate resSeqs would indicate upstream
        # data issues we don't try to resolve here.
        resseq_to_local.setdefault(resseq, local_idx)

    uncropped = binder_length_hint is None or int(binder_length_hint) == l_b
    hint = (cyclization_type_hint or "").strip().lower()

    if not pdb_path or not os.path.exists(pdb_path):
        return _no_label(f"pdb_path_missing(hint={hint or 'none'})")

    try:
        serial_to_atom, conect_pairs = read_atom_conect_records(pdb_path)
    except Exception as e:  # noqa: BLE001 - dataset robustness, never crash on a bad file
        return _no_label(f"conect_parse_error:{type(e).__name__}")

    for a, b in conect_pairs:
        atom_a = serial_to_atom.get(a)
        atom_b = serial_to_atom.get(b)
        if atom_a is None or atom_b is None:
            continue
        if atom_a["chain_id"] != binder_chain_id or atom_b["chain_id"] != binder_chain_id:
            continue

        name_a, name_b = atom_a["atom_name"], atom_b["atom_name"]
        res_a, res_b = atom_a["resname"], atom_b["resname"]
        rs_a, rs_b = atom_a["resseq"], atom_b["resseq"]
        if rs_a not in resseq_to_local or rs_b not in resseq_to_local:
            continue
        i, j = sorted((resseq_to_local[rs_a], resseq_to_local[rs_b]))
        if i == j:
            continue

        # DISULFIDE: the actual bond is CYS:SG <-> CYS:SG. We never assign this
        # type just because two residues happen to both be CYS.
        if name_a == "SG" and name_b == "SG" and res_a == "CYS" and res_b == "CYS":
            return _labeled(i, j, DISULFIDE, "conect_disulfide", hint)

        # MAINCHAIN: the actual bond is a backbone N <-> C bond, and it must close
        # the chain (endpoints == the binder termini). We never assign this type
        # just because a candidate pair happens to sit at the termini.
        if uncropped and {name_a, name_b} == {"N", "C"} and {i, j} == {0, l_b - 1}:
            return _labeled(i, j, MAINCHAIN, "conect_mainchain", hint)

        # ISOPEPTIDE: the actual bond is LYS:NZ <-> an ASP/GLU/ASN/GLN side-chain
        # carboxyl/carboxamide atom. We never assign this type just because the
        # residue types match without an observed bond between those atoms.
        is_iso = (_is_lys_nz(name_a, res_a) and _is_carboxyl_or_carboxamide(name_b, res_b)) or (
            _is_lys_nz(name_b, res_b) and _is_carboxyl_or_carboxamide(name_a, res_a)
        )
        if is_iso:
            return _labeled(i, j, ISOPEPTIDE, "conect_isopeptide", hint)

    if not conect_pairs:
        return _no_label(f"no_conect_records(hint={hint or 'none'})")
    return _no_label(f"no_matching_conect_pair(hint={hint or 'none'})")
