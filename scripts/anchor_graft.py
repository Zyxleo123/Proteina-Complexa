"""Fix the terminal residue identities of the input peptide before an LP -> CP edit.

Why
---
The cyclization head abstains -- emits a null edge -- whenever the decoded sequence admits
no candidate anchor pair for the requested chemistry: two CYS for a disulfide, a LYS plus an
ASP/GLU/ASN/GLN for an isopeptide. Measured on the 3600 rows of
`evaluation_results/sdedit_pocket14_20260906_213400`, that is not an occasional failure, it
is the dominant one, and it is governed entirely by the SEQUENCE budget `t_lat`:

    t_lat            0.2    0.4    0.6    0.8    1.0
    mean subs        8.5    7.3    3.1    0.1    0.0
    disulfide abst  0.033  0.113  0.950  1.000  1.000
    isopeptide abst 0.046  0.113  0.796  0.983  0.983

The inputs are linear binders of median length 10 and **87% of them contain no cysteine at
all**, so a disulfide needs two specific mutations at two specific positions. At t_lat >= 0.8
the sampler is allowed ~0 substitutions, so it can never make them, and the whole
sequence-preserving half of the grid is unmeasurable.

This module removes that wall at the source rather than fighting it during sampling: it
GRAFTS the required residues onto the two endpoints of the INPUT peptide before it is
encoded. Because the high-t_lat corner reproduces the input sequence almost exactly (0.0-0.1
substitutions, identity 0.99-1.00), an input that carries the anchors yields an output that
carries them too, and abstention goes to ~0 by construction instead of by persuasion.

That also makes it the honest experiment. "Mutate the two termini to cysteine, then cyclise"
is what a chemist would actually do at the bench; asking a generative model to invent the
cysteines on its own was never the design question. And with the identities given, the
cyclization-bond-distance term in `sdedit_guidance.py` stops being structurally inert -- its
`atoms_valid` gate is what zeroes it on an abstaining sample -- so the DPS bond loss becomes
applicable in exactly this regime.

Which residue goes where
------------------------
Disulfide is symmetric: CYS at both ends, no choice to make.

Isopeptide is directional, and the orientation is not a guess. Of the 499 isopeptide rings
the model placed for itself in the pocket14 run, **it put LYS at the lower-index endpoint in
100% of them, paired with ASP in 98%** (ASN in the remaining 2%, and exactly one ring the
other way round). Grafting LYS at `i` and ASP at `j` therefore matches what the model already
does when left alone, rather than imposing an orientation it would have to fight.

What happens to the side chain
------------------------------
The autoencoder's encoder reads `x1_aatype`, `x1_a37coors_nm` and `x1_sidechain_angles`
TOGETHER, so relabelling a residue while leaving its old atoms in place would hand it an
object that does not exist: an aatype saying CYS over an alanine's atom set. The new side
chain is therefore BUILT, at ideal geometry and a standard rotamer -- see `rotamer.py`.

The first version of this module did the opposite: it kept backbone + CB and dropped the rest,
on the argument that inventing a rotamer would be dishonest. That failed completely, and the
reason is arithmetic. ALA's whole atom37 set is [N, CA, C, CB, O], which is exactly what the
truncation leaves behind, so "CYS with only a CB" and "ALA" are the same object to the
encoder -- same occupancy mask, same (absent) chi1 -- differing only in a 20-d one-hot the AE
has never seen contradicted. Every grafted endpoint came back decoded as alanine:

    graft  disulfide  A.VQGGAAGH.S    (wanted C ... C)
    graft  disulfide  A.GLTIYAQKQ.A   (wanted C ... C)

Truncating is not the conservative choice, it is a silent relabel to alanine. `sidechain=
"truncate"` keeps that behaviour reachable as the negative control it turned out to be.
"""

from __future__ import annotations

import torch
from openfold.np import residue_constants as rc

from rotamer import build_sidechain

from proteinfoundation.cyclization.constants import (
    AA_ASP,
    AA_CYS,
    AA_LYS,
    CYCLIZATION_TYPE_TO_NAME,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)

N_IDX, CA_IDX, C_IDX, O_IDX, CB_IDX = (rc.atom_order[a] for a in ("N", "CA", "C", "O", "CB"))
BACKBONE_IDX = (N_IDX, CA_IDX, C_IDX, O_IDX)

# (residue at the lower-index endpoint, residue at the higher-index endpoint).
# See the module docstring for where the isopeptide orientation comes from.
ANCHOR_SPEC: dict[int, tuple[int, int] | None] = {
    DISULFIDE: (AA_CYS, AA_CYS),
    ISOPEPTIDE: (AA_LYS, AA_ASP),
    MAINCHAIN: None,  # head-to-tail bonds the backbone; identity is irrelevant
}

# [21, 37] bool: which atom37 slots each residue type can carry at all.
_RESTYPE_ATOM_MASK = torch.tensor(rc.restype_atom37_mask, dtype=torch.bool)


def anchor_residues(cyc_type_idx: int) -> tuple[int, int] | None:
    """The (i, j) residue identities the requested chemistry needs, or None for mainchain."""
    if cyc_type_idx not in ANCHOR_SPEC:
        raise SystemExit(f"FATAL: unknown cyclization type index {cyc_type_idx}")
    return ANCHOR_SPEC[cyc_type_idx]


def graft_anchors(batch: dict, mask: torch.Tensor, cyc_type_idx: int,
                  keep_cb: bool = True, sidechain: str = "build") -> tuple[dict, dict]:
    """Return a batch whose peptide termini carry the requested chemistry's anchor residues.

    The input batch is not modified: the tensors that change are cloned, everything else is
    shared. MAINCHAIN is a no-op that still reports itself, so a mainchain arm run with the
    flag on is visibly untouched rather than silently different.

    Args:
        sidechain: "build" places a complete ideal side chain at a standard rotamer (the
            only mode that survives the AE round trip); "truncate" is the original
            backbone+CB behaviour, retained as the negative control that decodes to alanine.
        keep_cb: truncate mode only -- whether CB is kept. Ignored when building, which
            always places CB.

    Returns:
        (batch, info): `info` carries per-endpoint before/after residue names, how many
        side-chain atoms were built or dropped, and `n_grafted` -- 0 when the input already
        had the right residues, which is the case a null result must be read against.
    """
    if sidechain not in ("build", "truncate"):
        raise SystemExit(f"FATAL: graft sidechain mode must be build|truncate, got {sidechain!r}")
    spec = anchor_residues(cyc_type_idx)
    info: dict = {
        "graft_type": CYCLIZATION_TYPE_TO_NAME.get(cyc_type_idx, str(cyc_type_idx)),
        "graft_applied": int(spec is not None),
        "graft_sidechain": sidechain,
        "n_grafted": 0,
        "n_sidechain_atoms_built": 0,
        "n_sidechain_atoms_dropped": 0,
    }
    if spec is None:
        return batch, info

    if mask.shape[0] != 1:
        raise SystemExit(f"FATAL: graft_anchors assumes one peptide per call (B=1); "
                         f"got batch size {mask.shape[0]}.")
    idx = mask[0].bool().nonzero().flatten()
    if idx.numel() < 2:
        raise SystemExit("FATAL: cannot graft anchors onto a peptide of fewer than 2 residues.")
    ends = (int(idx[0]), int(idx[-1]))

    out = dict(batch)
    # The native sequence must survive the graft: `score_edit` measures the substitution
    # budget against `residue_type`, so without this the two grafted positions would be
    # subtracted from the input and the edit would look one or two mutations cheaper than
    # it was. Stashed rather than recomputed because the caller has no other copy.
    if "residue_type_native" not in out:
        out["residue_type_native"] = batch["residue_type"].clone()
    out["residue_type"] = batch["residue_type"].clone()
    out["coord_mask"] = batch["coord_mask"].clone()
    for key in ("coords", "coords_nm"):
        if key in batch:
            out[key] = batch[key].clone()

    device = out["residue_type"].device
    allowed = _RESTYPE_ATOM_MASK.to(device)
    todo: list[tuple[int, int]] = []
    for pos, want in zip(ends, spec):
        have = int(out["residue_type"][0, pos])
        info[f"graft_{pos_label(pos, ends)}_from"] = rc.restypes[have] if have < 20 else "?"
        info[f"graft_{pos_label(pos, ends)}_to"] = rc.restypes[want]
        if have == want:
            # Already the right residue: keep its real side chain, which is genuine
            # information about this binder, and do not count it as a graft.
            continue
        info["n_grafted"] += 1
        todo.append((pos, want))

    if not todo:
        return out, info

    if sidechain == "truncate":
        for pos, want in todo:
            have = int(batch["residue_type"][0, pos])
            out["residue_type"][0, pos] = want
            keep = torch.zeros(out["coord_mask"].shape[-1], dtype=torch.bool, device=device)
            keep[list(BACKBONE_IDX)] = True
            if keep_cb and bool(allowed[have, CB_IDX]) and bool(allowed[want, CB_IDX]):
                keep[CB_IDX] = True
            # Never claim an atom the OLD residue did not actually have.
            keep = keep & out["coord_mask"][0, pos]
            dropped = out["coord_mask"][0, pos] & ~keep
            info["n_sidechain_atoms_dropped"] += int(dropped.sum())
            out["coord_mask"][0, pos] = keep
            for key in ("coords", "coords_nm"):
                if key in out:
                    out[key][0, pos][dropped] = 0.0
        return out, info

    # --- build mode ---------------------------------------------------------------------
    # `coords` (Angstrom) and `coords_nm` are NOT the same tensor scaled: in this loader they
    # hold the same structure in DIFFERENT RIGID FRAMES -- coords_nm is centred and randomly
    # rotated by the dataset transform while coords keeps the crystal frame. Measured on the
    # LNR inputs they differ by up to 19 A yet fit each other to 1e-6 A RMSD with det(R)=+1.
    # Both are read by the encoder (coords_nm feeds x1_a37coors_nm; coords feeds the backbone
    # and side-chain ANGLE features), so the side chain is built once per tensor, in that
    # tensor's own frame. Converting between them would need the transform, which is not in
    # the batch -- and writing one frame's coordinates into the other tensor would put the
    # side chain somewhere arbitrary while every bond length still looked perfect.
    units = [(k, sc) for k, sc in (("coords", 1.0), ("coords_nm", 10.0)) if k in out]
    if not units:
        raise SystemExit("FATAL: graft needs coords or coords_nm in the batch to build a side chain.")

    pos_t = torch.tensor([p for p, _ in todo], device=device, dtype=torch.long)
    want_t = torch.tensor([w for _, w in todo], device=device, dtype=torch.long)
    src_mask = out["coord_mask"][0, pos_t].bool()
    builds: dict[str, tuple] = {}
    for key, to_A in units:
        builds[key] = build_sidechain(out[key][0, pos_t] * to_A, src_mask, want_t)

    if len(builds) == 2:
        (_, ma, pa), (_, mb, pb) = builds["coords"], builds["coords_nm"]
        if not (torch.equal(ma, mb) and torch.equal(pa, pb)):
            raise SystemExit("FATAL: the two coordinate tensors imply different occupancies; "
                             "coords and coords_nm are out of sync.")
        # Frame-invariant desync check, on the SOURCE structure rather than the built side
        # chain: the built side chain is ideal geometry in whatever frame it was given, so
        # every internal measurement of it agrees by construction even if one tensor's
        # backbone is wrong. Pairwise distances among the source atoms do not -- they are
        # invariant to the rigid transform between the two frames and to nothing else.
        for n, (pos, _) in enumerate(todo):
            occ = src_mask[n]
            pa_ = out["coords"][0, pos][occ]
            pb_ = out["coords_nm"][0, pos][occ] * 10.0
            delta = float((torch.cdist(pa_, pa_) - torch.cdist(pb_, pb_)).abs().max())
            if delta > 5e-2:
                raise SystemExit(
                    f"FATAL: residue {pos} has different internal geometry in coords and "
                    f"coords_nm (pairwise distances differ by {delta:.3g} A). The two tensors "
                    f"are not the same structure, so no single side-chain build is correct "
                    f"for both.")

    for n, (pos, want) in enumerate(todo):
        # .clone(): basic indexing gives a VIEW, and .bool() on a bool tensor returns self,
        # so without it `prev` aliases the row overwritten below and both counters come out
        # zero.
        prev = out["coord_mask"][0, pos].bool().clone()
        built_mask = builds[units[0][0]][1]
        out["residue_type"][0, pos] = want
        out["coord_mask"][0, pos] = built_mask[n].to(out["coord_mask"].dtype)
        # Write ONLY the slots the builder placed, and zero only the slots that went away.
        # N/CA/C/O fall in neither set, so they stay bit-identical in every tensor -- writing
        # whole rows would push the backbone through a unit or frame round trip and silently
        # perturb the thing the graft exists to preserve.
        drop = built_mask[n].logical_not()
        for key, to_A in units:
            built_A, _, placed = builds[key]
            row = out[key][0, pos]
            row[placed[n]] = (built_A[n] / to_A)[placed[n]].to(row.dtype)
            row[drop] = 0.0
        info["n_sidechain_atoms_built"] += int((built_mask[n] & ~prev).sum())
        info["n_sidechain_atoms_dropped"] += int((prev & ~built_mask[n]).sum())
    return out, info


def pos_label(pos: int, ends: tuple[int, int]) -> str:
    return "i" if pos == ends[0] else "j"


def add_cli_args(ap) -> None:
    g = ap.add_argument_group("terminal anchor grafting (off by default)")
    g.add_argument("--graft-anchors", action="store_true",
                   help="Before encoding, set the input peptide's two terminal residues to "
                        "the ones the requested chemistry needs (disulfide: CYS/CYS; "
                        "isopeptide: LYS at the first endpoint, ASP at the last; mainchain: "
                        "no-op). Side-chain atoms beyond CB are dropped from coord_mask at "
                        "those positions rather than reinterpreted. This removes head "
                        "abstention at the source, which is what makes the sequence-preserving "
                        "corner (t_lat >= 0.6) measurable at all.")
    g.add_argument("--graft-sidechain", choices=("build", "truncate"), default="build",
                   help="build (default): place a complete ideal side chain at a standard "
                        "rotamer, so the encoder sees a real residue. truncate: keep only "
                        "backbone+CB -- the original behaviour, which decodes back as ALANINE "
                        "because [N, CA, C, CB, O] IS alanine's complete atom set. Kept only "
                        "as that negative control.")
    g.add_argument("--graft-keep-cb", type=int, default=1,
                   help="truncate mode only: keep the source residue's CB when both residues "
                        "have one (default). Ignored when building, which always places CB.")


def graft_tag(args) -> str:
    """Filename/run_key fragment; "" when grafting is off.

    Without this a grafted arm resumes on top of an ungrafted one -- same run_key, so every
    row reads as "already done" -- and overwrites its PDBs.

    The build mode has its OWN tag rather than inheriting "graft". Rows written before the
    side chain was built are wrong (every grafted endpoint decoded as alanine) but are not
    marked as such on disk, and resume keys off this string: reusing "graft" would make a
    fixed run skip exactly the rows it exists to redo. "graft" and "graftnocb" stay bound to
    the truncating modes that produced them.
    """
    if not getattr(args, "graft_anchors", False):
        return ""
    if getattr(args, "graft_sidechain", "build") == "build":
        return "graftsc"
    return "graft" if int(getattr(args, "graft_keep_cb", 1)) else "graftnocb"
