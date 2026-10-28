"""What a predicted complex is doing at its interface, in a form a text model can read.

This module exists because of one measurement and one constraint.

The measurement (`README_CPSEQ_PEPBENCH_SCALEUP.md` §11): across 11,953 scored folds,
**peptide-receptor overlap was the sole cause of 3,457 rejections** -- removing it
entirely would raise the pass rate by 0.289, more than every other failure category
combined. A proposer that cannot see which position is overlapping cannot avoid
re-proposing a residue that overlaps. Interface geometry is therefore not decoration
here; it is the single most actionable thing about a scored candidate.

The constraint: `retention.py` drops complex PDBs at corpus scale (263x storage
overhead, almost none of it a result). So the summary has to be computed **while the
structure is still in hand** and persisted as the ~2 KB of text that survives. That is
why this is a standalone function returning a plain dict rather than something a report
runs later over a tree of PDBs -- at `deliverable` retention that tree is gone.

What it does NOT do
-------------------
It does not score, rank or gate. The frozen gate in `geometry.py` and `scorer.py` decides
what is valid, and nothing here can promote a candidate the gate rejected. This is a
*description* of a complex, produced for a prompt, and its numbers never enter a
selection rule. Keeping it out of the objective is deliberate: a measurement that feeds
both the proposer and the gate would let the proposer learn the gate's blind spots.

Positions are reported **1-based along the peptide sequence**, which is what a prompt can
talk about, and every position is listed whether or not it interacts -- "position 11 makes
no contact at all" is a proposal-relevant fact, and a summary that only listed contacts
would hide it.

Two honesty notes carried on the record itself:

- `plip.ok` is False when PLIP failed, and the summary then carries only the Biopython
  geometry. It is never silently an empty interaction list, because "no interactions
  found" and "the interaction detector crashed" would otherwise read identically to a
  proposer and to anyone reading the journal.
- The closing amide of a head-to-tail cyclic peptide is usually absent from the PDB's
  CONECT records, so PLIP perceives the ring as a linear chain. That affects its
  *intramolecular* perception only; every interaction reported here is peptide-to-receptor,
  which is unaffected. `ring_bond_perceived` records which case applied.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Heavy-atom separation below which two atoms are counted as a steric overlap. Matches
# the inter-chain clash definition used by `geometry.py` so that a position this module
# calls "overlapping" is the same position the gate rejected the candidate for.
CLASH_DIST_A = 2.2
# Heavy-atom separation counted as a contact. 4.5 A is the usual packing shell.
CONTACT_DIST_A = 4.5
# Receptor residues within this distance of the peptide are "lining" the pocket.
POCKET_DIST_A = 8.0

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Protonation variants PyRosetta's relax can leave behind.
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K", "ARN": "R", "TYM": "Y",
    "MSE": "M",
}

# PLIP attribute -> the short name used in the record and the prompt. Water bridges and
# halogen bonds are included although a canonical-alphabet peptide rarely makes them:
# when one does appear it is informative, and an absent category costs one empty list.
PLIP_CATEGORIES = (
    ("hydrophobic_contacts", "hydrophobic"),
    ("hbonds_ldon", "hbond"),
    ("hbonds_pdon", "hbond"),
    ("saltbridge_lneg", "saltbridge"),
    ("saltbridge_pneg", "saltbridge"),
    ("pistacking", "pistack"),
    ("pication_laro", "pication"),
    ("pication_paro", "pication"),
    ("water_bridges", "waterbridge"),
    ("halogen_bonds", "halogen"),
)


# --------------------------------------------------------------------------------------
# Secondary structure, from backbone dihedrals
# --------------------------------------------------------------------------------------
# There is no DSSP binary in this environment, and adding one for a coarse three-state
# label is not worth a new system dependency. Phi/psi quadrants give that label directly.
# It is labelled `ss_approx` everywhere so nobody mistakes it for a DSSP assignment.


def _ss_from_phi_psi(phi: float | None, psi: float | None) -> str:
    if phi is None or psi is None:
        return "?"
    p, s = np.degrees(phi), np.degrees(psi)
    if -160.0 < p < -20.0 and -120.0 < s < 50.0:
        return "H"
    if (-180.0 <= p < -45.0) and (90.0 < s <= 180.0 or -180.0 <= s < -150.0):
        return "E"
    if 20.0 < p < 125.0:
        return "L"  # left-handed / turn region, common at a cyclic peptide's corners
    return "-"


# --------------------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------------------


@dataclass
class PositionSummary:
    """One peptide position. `pos` is 1-based along the peptide sequence."""

    pos: int
    residue: str
    min_dist_A: float | None = None
    n_contacts: int = 0
    n_clashes: int = 0
    worst_clash_A: float | None = None
    buried_fraction: float | None = None
    ss_approx: str = "?"
    partners: list[str] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    clash_partners: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pos": self.pos,
            "residue": self.residue,
            "min_dist_A": None if self.min_dist_A is None else round(self.min_dist_A, 2),
            "n_contacts": self.n_contacts,
            "n_clashes": self.n_clashes,
            "worst_clash_A": (
                None if self.worst_clash_A is None else round(self.worst_clash_A, 2)
            ),
            "buried_fraction": (
                None if self.buried_fraction is None else round(self.buried_fraction, 3)
            ),
            "ss_approx": self.ss_approx,
            "partners": self.partners,
            "interactions": self.interactions,
            "clash_partners": self.clash_partners,
        }


def _res_label(chain: str, resname: str, resnum: int) -> str:
    return f"{chain}:{resname.title()}{resnum}"


def resolve_pdb(pdb_path: str | os.PathLike) -> tuple[str | None, str | None]:
    """A real, uncompressed path for `pdb_path`, decompressing a `.gz` if that is what
    survived.

    The ladder runs at `rescorable` retention by default, which gzips the raw complex
    (217 KB -> 47 KB). PLIP takes a filename and opens it itself, so a gzipped structure
    has to be materialised. Returns `(path, tempdir_to_clean)`; the caller removes the
    temporary directory when it is not None. Without this the summary would silently be
    unavailable for exactly the runs it was built for.
    """
    p = Path(pdb_path)
    if p.exists():
        return str(p), None
    gz = p.with_suffix(p.suffix + ".gz") if p.suffix != ".gz" else p
    if not gz.exists():
        return None, None
    import gzip
    import tempfile

    tmp = tempfile.mkdtemp(prefix="cpseq_iface_")
    dest = Path(tmp) / (gz.name[:-3] if gz.name.endswith(".gz") else gz.name)
    try:
        dest.write_bytes(gzip.decompress(gz.read_bytes()))
    except Exception:  # noqa: BLE001
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        return None, None
    return str(dest), tmp


def _pick_peptide_chain(model, expected_sequence: str | None) -> tuple[str | None, str]:
    """Which chain is the peptide.

    Preference order: an exact sequence match against `expected_sequence`, then a
    length match, then the shortest chain. The sequence match is tried first because a
    receptor can legitimately contain a short chain, and picking by length alone would
    then summarise the wrong molecule -- producing a plausible-looking interface record
    for a chain the search never touched.
    """
    chains = {}
    for ch in model:
        seq = "".join(
            THREE_TO_ONE.get(r.get_resname().strip().upper(), "X")
            for r in ch
            if r.id[0] == " "
        )
        if seq:
            chains[ch.id] = seq
    if not chains:
        return None, "no polymer chain found"
    if expected_sequence:
        exact = [c for c, s in chains.items() if s == expected_sequence]
        if len(exact) == 1:
            return exact[0], "exact sequence match"
        if len(exact) > 1:
            return sorted(exact)[0], f"sequence match, {len(exact)} chains tied"
        same_len = [c for c, s in chains.items() if len(s) == len(expected_sequence)]
        if len(same_len) == 1:
            return same_len[0], "length match (sequence differs from expected)"
    shortest = min(chains, key=lambda c: (len(chains[c]), c))
    return shortest, "shortest chain"


def _plip_interactions(
    pdb_path: str, peptide_chain: str
) -> tuple[dict[int, list[dict]], dict]:
    """PLIP's interaction inventory, keyed by peptide residue number.

    PLIP is given the peptide chain as the "ligand" through `config.PEPTIDES`, which is
    its supported path for peptide ligands. Its stdout is captured rather than printed:
    it writes progress lines that would interleave with the search driver's own per-fold
    log and make a run's output unreadable.
    """
    meta: dict = {"ok": False, "error": None, "n_interactions": 0}
    try:
        from plip.basic import config as plip_config
        from plip.structure.preparation import PDBComplex
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"plip import failed: {type(exc).__name__}: {exc}"
        return {}, meta

    by_pos: dict[int, list[dict]] = {}
    prev_peptides = list(getattr(plip_config, "PEPTIDES", []) or [])
    try:
        plip_config.PEPTIDES = [peptide_chain]
        buf = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(buf):
            warnings.simplefilter("ignore")
            cx = PDBComplex()
            cx.load_pdb(pdb_path)
            cx.analyze()
            sets = list(cx.interaction_sets.values())
        n = 0
        for iset in sets:
            for attr, short in PLIP_CATEGORIES:
                for it in getattr(iset, attr, []) or []:
                    # `*_l` is the ligand side, i.e. the peptide; the unsuffixed fields
                    # are the receptor side. Both orientations of a category (ldon/pdon,
                    # lneg/pneg) collapse to one short name, with `donor` recording which.
                    pos = getattr(it, "resnr_l", None)
                    rec_chain = getattr(it, "reschain", None)
                    rec_name = getattr(it, "restype", None)
                    rec_num = getattr(it, "resnr", None)
                    if pos is None or rec_name is None or rec_num is None:
                        continue
                    dist = getattr(it, "distance", None)
                    if dist is None:
                        dist = getattr(it, "distance_ah", None)
                    rec = {
                        "type": short,
                        "partner": _res_label(rec_chain or "?", str(rec_name), int(rec_num)),
                    }
                    if dist is not None:
                        rec["dist_A"] = round(float(dist), 2)
                    if attr in ("hbonds_ldon", "hbonds_pdon"):
                        rec["donor"] = "peptide" if attr == "hbonds_ldon" else "receptor"
                    by_pos.setdefault(int(pos), []).append(rec)
                    n += 1
        meta["ok"] = True
        meta["n_interactions"] = n
    except Exception as exc:  # noqa: BLE001
        # A PLIP failure must not take the evaluator down with it: the geometry gate and
        # the energy are the result, and this summary is an input to a proposer.
        meta["error"] = f"{type(exc).__name__}: {exc}"
        by_pos = {}
    finally:
        plip_config.PEPTIDES = prev_peptides
    return by_pos, meta


def summarize_complex(
    pdb_path: str | os.PathLike,
    *,
    peptide_sequence: str | None = None,
    peptide_chain: str | None = None,
    run_plip: bool = True,
    structure_stage: str = "raw",
) -> dict:
    """Describe one predicted complex's interface, per peptide position.

    `peptide_sequence` is the sequence the search believes it predicted. It is used to
    identify the chain and is then CHECKED against what the file contains: a mismatch is
    recorded in `sequence_matches` rather than corrected, because a summary silently
    attributed to the wrong sequence is worse than no summary.

    **Pass the RAW predicted complex, not the relaxed one.** `structure_stage` only
    records which was passed; it does not change the computation. The default says raw
    because that is the structure the gate judges: `evaluate.py` calls `validity_report`
    on `pdb_path` and uses the refined copy for energies alone. Relax is explicitly
    allowed to move atoms to relieve strain, so it repairs much of the very defect a
    proposer most needs to see. Measured on `lnr_6seo`/`VPLRARNLPPFFFKEP`: the raw
    complex overlaps His198 at 0.79 A at position 6 -- the reason the gate rejected it --
    and after relax the same position sits at 2.7 A with no overlap at all. A proposer
    shown the refined complex would be told that a rejected candidate's interface was
    clean, and would have no reason not to propose the same residue again.
    """
    import shutil

    requested = str(pdb_path)
    pdb_path, _tmpdir = resolve_pdb(requested)
    try:
        return _summarize_resolved(
            requested=requested,
            pdb_path=pdb_path,
            structure_stage=structure_stage,
            peptide_sequence=peptide_sequence,
            peptide_chain=peptide_chain,
            run_plip=run_plip,
        )
    finally:
        if _tmpdir:
            shutil.rmtree(_tmpdir, ignore_errors=True)


def _summarize_resolved(
    *,
    requested: str,
    pdb_path: str | None,
    structure_stage: str,
    peptide_sequence: str | None,
    peptide_chain: str | None,
    run_plip: bool,
) -> dict:
    from Bio.PDB import PDBParser

    out: dict = {
        "source_pdb": requested,
        "structure_stage": structure_stage,
        "ok": False,
        "error": None,
        "peptide_chain": peptide_chain,
        "chain_choice_reason": None,
        "peptide_sequence_expected": peptide_sequence,
        "peptide_sequence_observed": None,
        "sequence_matches": None,
        "plip": {"ok": False, "error": "not run", "n_interactions": 0},
        "ring_bond_perceived": False,
        "n_interface_contacts": 0,
        "n_interface_clashes": 0,
        "buried_sasa_A2": None,
        "positions": [],
        "receptor_pocket": [],
    }
    if pdb_path is None:
        out["error"] = "pdb not found (neither the path nor a .gz beside it)"
        return out
    if pdb_path != requested:
        out["decompressed_from_gz"] = True

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PDBParser(QUIET=True).get_structure("cx", pdb_path)[0]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"parse failed: {type(exc).__name__}: {exc}"
        return out

    if peptide_chain is None:
        peptide_chain, reason = _pick_peptide_chain(model, peptide_sequence)
        out["chain_choice_reason"] = reason
    else:
        out["chain_choice_reason"] = "caller-specified"
    if peptide_chain is None or peptide_chain not in model:
        out["error"] = f"peptide chain {peptide_chain!r} not in structure"
        return out
    out["peptide_chain"] = peptide_chain

    pep_residues = [r for r in model[peptide_chain] if r.id[0] == " "]
    if not pep_residues:
        out["error"] = f"chain {peptide_chain} has no standard residues"
        return out
    observed = "".join(
        THREE_TO_ONE.get(r.get_resname().strip().upper(), "X") for r in pep_residues
    )
    out["peptide_sequence_observed"] = observed
    if peptide_sequence is not None:
        out["sequence_matches"] = observed == peptide_sequence

    # --- receptor atoms (everything that is not the peptide chain) --------------------
    rec_atoms, rec_owner = [], []
    for ch in model:
        if ch.id == peptide_chain:
            continue
        for res in ch:
            if res.id[0] != " ":
                continue  # waters and hetero groups are not the receptor surface here
            label = _res_label(ch.id, res.get_resname().strip(), res.id[1])
            for atom in res:
                if atom.element == "H":
                    continue
                rec_atoms.append(atom.coord)
                rec_owner.append(label)
    if not rec_atoms:
        out["error"] = "no receptor atoms found outside the peptide chain"
        return out
    rec_xyz = np.asarray(rec_atoms, dtype=np.float32)
    rec_owner_arr = np.asarray(rec_owner)

    # --- buried SASA: the peptide in the complex against the peptide alone ------------
    # Computed rather than looked up from per-residue maxima, so the denominator is this
    # peptide in this conformation. A table of reference maxima would mix a conformational
    # effect into what is meant to be an interface measurement.
    per_pos_buried: dict[int, float] = {}
    try:
        from Bio.PDB.SASA import ShrakeRupley

        sr = ShrakeRupley()
        import copy

        pep_only = copy.deepcopy(model[peptide_chain])
        sr.compute(pep_only, level="R")
        alone = {r.id[1]: r.sasa for r in pep_only if r.id[0] == " "}
        sr.compute(model, level="R")
        in_cx = {r.id[1]: r.sasa for r in model[peptide_chain] if r.id[0] == " "}
        total_buried = 0.0
        for rid, a in alone.items():
            c = in_cx.get(rid)
            if c is None or a is None or a <= 0.0:
                continue
            buried = max(0.0, a - c)
            total_buried += buried
            per_pos_buried[rid] = min(1.0, buried / a)
        out["buried_sasa_A2"] = round(total_buried, 1)
    except Exception as exc:  # noqa: BLE001
        out["sasa_error"] = f"{type(exc).__name__}: {exc}"

    # --- approximate secondary structure --------------------------------------------
    ss_by_resid: dict[int, str] = {}
    try:
        from Bio.PDB.Polypeptide import PPBuilder

        for pp in PPBuilder().build_peptides(model[peptide_chain]):
            angles = pp.get_phi_psi_list()
            for res, (phi, psi) in zip(pp, angles):
                ss_by_resid[res.id[1]] = _ss_from_phi_psi(phi, psi)
    except Exception as exc:  # noqa: BLE001
        out["ss_error"] = f"{type(exc).__name__}: {exc}"

    # --- PLIP -------------------------------------------------------------------------
    plip_by_resnum: dict[int, list[dict]] = {}
    if run_plip:
        plip_by_resnum, out["plip"] = _plip_interactions(pdb_path, peptide_chain)

    # --- per-position geometry --------------------------------------------------------
    pocket_counts: dict[str, int] = {}
    pocket_positions: dict[str, set[int]] = {}
    positions: list[PositionSummary] = []
    total_contacts = total_clashes = 0

    for i, res in enumerate(pep_residues, start=1):
        resnum = res.id[1]
        heavy = np.asarray(
            [a.coord for a in res if a.element != "H"], dtype=np.float32
        )
        ps = PositionSummary(
            pos=i,
            residue=THREE_TO_ONE.get(res.get_resname().strip().upper(), "X"),
            buried_fraction=per_pos_buried.get(resnum),
            ss_approx=ss_by_resid.get(resnum, "?"),
            # PLIP keys by the residue number in the file, which is what it was given.
            interactions=plip_by_resnum.get(resnum, []),
        )
        if heavy.size:
            d = np.linalg.norm(rec_xyz[:, None, :] - heavy[None, :, :], axis=-1)
            per_rec = d.min(axis=1)
            ps.min_dist_A = float(per_rec.min())
            # The two counts are deliberately different units, because each has one job.
            #
            # `n_contacts` counts DISTINCT RECEPTOR ATOMS inside the packing shell: it
            # describes how enclosed the position is, and counting pairs would let one
            # receptor atom near a large side chain inflate it several-fold.
            #
            # `n_clashes` counts ATOM PAIRS, because that is what `geometry.clashes`
            # counts and what the gate rejects the candidate on. Summing distinct atoms
            # here would print a smaller number than the rejection reason the same
            # candidate carries in the journal -- measured on `lnr_6seo`, 3 atoms against
            # the gate's 4 pairs -- and a proposer told "3 overlaps" about a fold
            # rejected for 4 would be reading a quantity that exists nowhere else.
            contact_mask = per_rec < CONTACT_DIST_A
            ps.n_contacts = int(contact_mask.sum())
            ps.n_clashes = int((d < CLASH_DIST_A).sum())
            if ps.n_clashes:
                clash_rec_mask = per_rec < CLASH_DIST_A
                ps.worst_clash_A = float(per_rec[clash_rec_mask].min())
                ps.clash_partners = sorted(set(rec_owner_arr[clash_rec_mask].tolist()))
            ps.partners = sorted(set(rec_owner_arr[contact_mask].tolist()))
            total_contacts += ps.n_contacts
            total_clashes += ps.n_clashes
            for lab in set(rec_owner_arr[per_rec < POCKET_DIST_A].tolist()):
                pocket_counts[lab] = pocket_counts.get(lab, 0) + 1
                pocket_positions.setdefault(lab, set()).add(i)
        # Interaction partners that PLIP saw but the contact shell missed (a long salt
        # bridge, a water bridge) still belong in the partner list.
        for it in ps.interactions:
            if it.get("partner") and it["partner"] not in ps.partners:
                ps.partners.append(it["partner"])
        positions.append(ps)

    out["positions"] = [p.to_dict() for p in positions]
    out["n_interface_contacts"] = total_contacts
    out["n_interface_clashes"] = total_clashes
    out["receptor_pocket"] = [
        {
            "residue": lab,
            "n_peptide_positions": len(pocket_positions[lab]),
            "peptide_positions": sorted(pocket_positions[lab]),
        }
        for lab in sorted(pocket_counts, key=lambda k: (-pocket_counts[k], k))[:24]
    ]
    out["ok"] = True
    return out


# --------------------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------------------
# A JSON dump of the above would spend most of a context window on punctuation and
# repeated keys. These renderers produce a fixed-width table instead: one line per
# position, which is the unit a substitution proposal is about.


def _fmt(v: float | None, width: int, prec: int = 1) -> str:
    return "-".rjust(width) if v is None else f"{v:>{width}.{prec}f}"


def render_complex(summary: dict, *, title: str = "complex", max_partners: int = 3) -> str:
    """One complex as a compact per-position table."""
    if not summary.get("ok"):
        return f"[{title}] unavailable: {summary.get('error') or 'unknown error'}"

    lines: list[str] = []
    plip = summary.get("plip") or {}
    head = (
        f"[{title}] peptide={summary.get('peptide_sequence_observed')} "
        f"chain={summary.get('peptide_chain')} "
        f"buried_SASA={summary.get('buried_sasa_A2')}A2 "
        f"contacts={summary.get('n_interface_contacts')} "
        f"overlaps={summary.get('n_interface_clashes')}"
    )
    lines.append(head)
    if not plip.get("ok"):
        # Stated in the prompt itself, so the model is not left to infer that an empty
        # interaction column means "no interactions" when it means "detector failed".
        lines.append(
            f"  NOTE: interaction detection failed ({plip.get('error')}); the "
            f"contact/overlap/burial columns below are still valid."
        )
    if summary.get("sequence_matches") is False:
        lines.append(
            f"  NOTE: file holds {summary.get('peptide_sequence_observed')} but "
            f"{summary.get('peptide_sequence_expected')} was expected."
        )
    lines.append("  pos aa  ss  burial  dmin  ctc  ovl  interactions / nearby receptor")
    for p in summary.get("positions", []):
        inter = p.get("interactions") or []
        if inter:
            seen: list[str] = []
            for it in inter:
                tag = f"{it['type']}:{it['partner']}"
                if tag not in seen:
                    seen.append(tag)
            detail = ", ".join(seen[:max_partners])
            if len(seen) > max_partners:
                detail += f" (+{len(seen) - max_partners})"
        else:
            near = (p.get("partners") or [])[:max_partners]
            detail = ("near " + ", ".join(near)) if near else "no receptor contact"
        if p.get("n_clashes"):
            detail = (
                f"OVERLAP {p['worst_clash_A']}A with "
                f"{', '.join((p.get('clash_partners') or [])[:2])}; " + detail
            )
        lines.append(
            f"  {p['pos']:>3} {p['residue']}  {p['ss_approx']:>2} "
            f"{_fmt(p.get('buried_fraction'), 6, 2)} "
            f"{_fmt(p.get('min_dist_A'), 5, 1)} "
            f"{p.get('n_contacts', 0):>4} {p.get('n_clashes', 0):>4}  {detail}"
        )
    return "\n".join(lines)


def render_pocket(summary: dict, *, limit: int = 12) -> str:
    """The receptor residues lining the pocket, busiest first."""
    pocket = (summary.get("receptor_pocket") or [])[:limit]
    if not pocket:
        return ""
    items = [
        f"{e['residue']}(pos {','.join(str(x) for x in e['peptide_positions'][:4])})"
        for e in pocket
    ]
    return "Receptor pocket residues: " + "; ".join(items)
