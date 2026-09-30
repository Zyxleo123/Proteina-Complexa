"""Triplet export: write ``(generated LP, source CP, target)`` sets with their scores.

Each accepted sample produces two complex PDBs -- the generated LP with its receptor, and
the source CP with the same receptor -- plus one JSONL row carrying the decoded sequence,
the source id, the seed and every score component. Two files rather than one because
downstream Rosetta and AF2 tooling expects a single peptide chain per complex, and because
a triplet is only useful if the "before" and "after" are in the SAME receptor frame, which
writing them from one sampling call is what guarantees.

WHY BOTH STRUCTURES ARE WRITTEN AT SAMPLE TIME. The loader applies a per-process random
global rotation, so re-loading a peptide later cannot recover the frame its receptor was
in. A peptide saved without its receptor can never be reunited with it. Both members of
the pair are therefore written here, in one frame, in one call.

THE EXPORT GATE is stricter than the training losses:

* the decoded argmax sequence must equal the source CP's sequence exactly -- sequence
  identity is a premise of the task, not something to be traded off;
* the chain must be chemically linear -- the C(L-1)-N(0) distance outside any bond window
  AND the termini genuinely separated, so nothing downstream infers a cyclic connection;
* the backbone must be intact, since a chain broken in the middle is not a linear peptide,
  it is two of them.

Rejected samples are still recorded, with a reason. A silent drop makes the acceptance
rate unknowable, and the acceptance rate is one of the numbers that says whether this
approach works at all.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from loguru import logger
from openfold.np.residue_constants import restypes

from proteinfoundation.cp2lp.geometry import CA_IDX, C_IDX, O_IDX, terminal_gap_nm
from proteinfoundation.cp2lp.losses import (
    CLASH_THRESHOLD_NM,
    MIN_OPEN_TERMINAL_GAP_NM,
    clash_loss,
    contact_retention_loss,
    linear_terminus_valid,
    open_chain_geometry_loss,
    sequence_matches_source,
)
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

NM_TO_ANG = 10.0

#: atom37 index of OXT, the C-terminal carboxylate oxygen.
OXT_IDX = 36

#: C-OXT bond length in nm.
OXT_BOND_NM = 0.125


def seq_to_string(aatype: np.ndarray) -> str:
    """One-letter sequence from residue-type indices, 'X' for anything out of range."""
    return "".join(restypes[int(a)] if 0 <= int(a) < len(restypes) else "X" for a in aatype)


def build_oxt(atom37: torch.Tensor, atom_mask: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Places OXT on the C-terminal residue so the exported chain has a real free terminus.

    The decoder emits backbone O but no OXT -- it has never needed one, because a
    macrocycle has no free C-terminus. A linear peptide does, and a carboxylate written
    without it is a subtly wrong molecule: parsers will read the terminal residue as
    amidated or as mid-chain.

    Geometry is the standard sp2 construction: C is trigonal planar with CA, O and OXT, so
    the OXT direction is the negated bisector of the C->CA and C->O unit vectors.
    """
    out = atom37.clone()
    out_mask = atom_mask.clone()
    b = atom37.shape[0]
    idx = torch.arange(b, device=atom37.device)
    last = (lengths - 1).clamp(min=0)

    c = atom37[idx, last, C_IDX]
    ca = atom37[idx, last, CA_IDX]
    o = atom37[idx, last, O_IDX]

    d1 = ca - c
    d2 = o - c
    d1 = d1 / (torch.linalg.vector_norm(d1, dim=-1, keepdim=True) + 1e-8)
    d2 = d2 / (torch.linalg.vector_norm(d2, dim=-1, keepdim=True) + 1e-8)
    direction = -(d1 + d2)
    direction = direction / (torch.linalg.vector_norm(direction, dim=-1, keepdim=True) + 1e-8)

    # Only where the residue actually has the three atoms the construction needs.
    ok = atom_mask[idx, last, C_IDX] & atom_mask[idx, last, CA_IDX] & atom_mask[idx, last, O_IDX]
    oxt = c + OXT_BOND_NM * direction
    out[idx, last, OXT_IDX] = torch.where(ok[:, None], oxt, out[idx, last, OXT_IDX])
    out_mask[idx, last, OXT_IDX] = ok
    return out, out_mask


def _write_complex(
    path: str,
    pep_atom37_ang: np.ndarray,
    pep_aatype: np.ndarray,
    target_atom37_ang: np.ndarray,
    target_aatype: np.ndarray,
) -> None:
    """Receptor as chain A, peptide as chain B -- the layout the eval tooling expects."""
    pos = np.concatenate([target_atom37_ang, pep_atom37_ang], axis=0)
    aa = np.concatenate([target_aatype, pep_aatype], axis=0)
    chain_index = np.concatenate(
        [np.zeros(len(target_aatype)), np.ones(len(pep_aatype))]
    ).astype(np.int64)
    write_prot_to_pdb(
        prot_pos=pos,
        file_path=path,
        aatype=aa,
        chain_index=chain_index,
        overwrite=True,
        no_indexing=True,
    )


@torch.no_grad()
def export_triplets(
    out_dir: str,
    result: dict,
    source_ids: list[str],
    seed: int,
    samples_per_cp: int = 1,
    manifest_name: str = "triplets.jsonl",
    min_open_gap_nm: float = MIN_OPEN_TERMINAL_GAP_NM,
    clash_threshold_nm: float = CLASH_THRESHOLD_NM,
    write_rejected: bool = False,
) -> dict:
    """Writes every sample in one ``generate_lp_for_cp`` result, gated and scored.

    Args:
        out_dir: destination. ``structures/`` is created under it.
        result: the dict returned by ``CP2LPGenerator.generate_lp_for_cp``.
        source_ids: ``[n_cp]`` example ids, BEFORE the per-CP repeat. Expanded here so a
            caller cannot get the repeat factor wrong.
        seed: the seed this batch was drawn with; recorded per row so a triplet is
            reproducible.
        samples_per_cp: repeat factor used at generation time.
        write_rejected: also write PDBs for rejected samples. Off by default -- the JSONL
            row records the rejection either way, and the structures are rarely wanted.

    Returns:
        Summary counts, also written to ``<out_dir>/export_summary.json``.
    """
    struct_dir = os.path.join(out_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, manifest_name)

    gen_batch = result["gen_batch"]
    decoded = result["decoded"]
    src_decoded = result["src_decoded"]
    mask = result["mask"].bool()
    b = mask.shape[0]

    if len(source_ids) * samples_per_cp != b:
        raise ValueError(
            f"source_ids ({len(source_ids)}) x samples_per_cp ({samples_per_cp}) != batch ({b}); "
            "pass the ids BEFORE the repeat."
        )
    expanded_ids = [sid for sid in source_ids for _ in range(samples_per_cp)]
    sample_idx = [i for _ in source_ids for i in range(samples_per_cp)]

    lengths = mask.sum(dim=-1)
    src_aatype = gen_batch["residue_type"].long()

    # ---- gates
    seq_ok = sequence_matches_source(decoded["seq_logits"], src_aatype, mask)
    lin_ok = linear_terminus_valid(decoded["coors_nm"], mask, min_gap_nm=min_open_gap_nm)
    _, geom_metrics_all = open_chain_geometry_loss(decoded["coors_nm"], decoded["atom_mask_eff"], mask)

    # ---- per-example scores, computed one row at a time so a row's numbers are its own
    rows = []
    n_accept = 0
    tgap = terminal_gap_nm(decoded["coors_nm"], mask)
    sgap = terminal_gap_nm(src_decoded["coors_nm"], mask)
    pred_aatype = decoded["seq_logits"].argmax(dim=-1)

    with open(manifest_path, "a") as mf:
        for i in range(b):
            sel = mask[i]
            n_res = int(sel.sum())
            if n_res == 0:
                continue
            one = slice(i, i + 1)

            contact, cmetrics = contact_retention_loss(
                gen_atom37=decoded["coors_nm"][one],
                gen_atom_mask=decoded["atom_mask_eff"][one],
                src_atom37=src_decoded["coors_nm"][one],
                src_atom_mask=src_decoded["atom_mask_eff"][one],
                pep_mask=mask[one],
                target_atom37=gen_batch["x_target"][one],
                target_atom_mask=gen_batch["target_mask"][one],
                target_mask=gen_batch["seq_target_mask"][one].bool(),
            )
            _, clmetrics = clash_loss(
                atom37=decoded["coors_nm"][one],
                atom_mask=decoded["atom_mask_eff"][one],
                pep_mask=mask[one],
                target_atom37=gen_batch["x_target"][one],
                target_atom_mask=gen_batch["target_mask"][one],
                target_mask=gen_batch["seq_target_mask"][one].bool(),
                threshold_nm=clash_threshold_nm,
            )
            _, gmetrics = open_chain_geometry_loss(
                decoded["coors_nm"][one], decoded["atom_mask_eff"][one], mask[one]
            )

            chain_intact = float(gmetrics["geom_chain_intact_frac"]) == 1.0
            reasons = []
            if not bool(seq_ok[i]):
                reasons.append("sequence_mismatch")
            if not bool(lin_ok[i]):
                reasons.append("not_linear")
            if not chain_intact:
                reasons.append("backbone_broken")
            accepted = not reasons

            gen_seq = seq_to_string(pred_aatype[i][sel].cpu().numpy())
            src_seq = seq_to_string(src_aatype[i][sel].cpu().numpy())

            tsel = gen_batch["seq_target_mask"][i].bool()
            base = f"{expanded_ids[i]}__s{seed}__k{sample_idx[i]}"
            gen_pdb = os.path.join(struct_dir, f"{base}__gen_lp.pdb")
            src_pdb = os.path.join(struct_dir, f"{base}__src_cp.pdb")

            if accepted or write_rejected:
                pep_xyz, pep_mask37 = build_oxt(
                    decoded["coors_nm"][one], decoded["atom_mask_eff"][one], lengths[one]
                )
                pep_np = (pep_xyz[0][sel] * NM_TO_ANG).cpu().numpy()
                pep_np = pep_np * pep_mask37[0][sel][..., None].cpu().numpy()
                tgt_np = (gen_batch["x_target"][i][tsel] * NM_TO_ANG).cpu().numpy()
                tgt_aa = gen_batch["seq_target"][i][tsel].cpu().numpy()
                _write_complex(gen_pdb, pep_np, pred_aatype[i][sel].cpu().numpy(), tgt_np, tgt_aa)
                _write_complex(
                    src_pdb,
                    (src_decoded["coors_nm"][i][sel] * NM_TO_ANG).cpu().numpy(),
                    src_aatype[i][sel].cpu().numpy(),
                    tgt_np,
                    tgt_aa,
                )

            row = {
                "triplet_id": base,
                "source_id": expanded_ids[i],
                "seed": int(seed),
                "sample_idx": int(sample_idx[i]),
                "peptide_length": n_res,
                "sequence": gen_seq,
                "source_sequence": src_seq,
                "accepted": bool(accepted),
                "reject_reasons": reasons,
                "gen_lp_pdb": gen_pdb if (accepted or write_rejected) else None,
                "src_cp_pdb": src_pdb if (accepted or write_rejected) else None,
                "scores": {
                    "terminal_gap_nm": float(tgap[i]),
                    "src_terminal_gap_nm": float(sgap[i]),
                    "terminal_gap_delta_nm": float(tgap[i] - sgap[i]),
                    "contact_retention_frac": float(cmetrics["contact_retention_frac"]),
                    "contact_n_source": float(cmetrics["contact_n_source"]),
                    "contact_n_kept": float(cmetrics["contact_n_kept"]),
                    "contact_loss": float(contact),
                    "clash_inter_sum_nm": float(clmetrics.get("clash_inter_sum_nm", 0.0)),
                    "clash_intra_sum_nm": float(clmetrics.get("clash_intra_sum_nm", 0.0)),
                    "clash_min_inter_nm": float(clmetrics.get("clash_min_inter_nm", float("nan"))),
                    "peptide_bond_mae_nm": float(gmetrics["geom_peptide_bond_mae_nm"]),
                    "chain_intact": chain_intact,
                    "seq_identity_to_source": float((pred_aatype[i][sel] == src_aatype[i][sel]).float().mean()),
                },
            }
            mf.write(json.dumps(row) + "\n")
            rows.append(row)
            n_accept += int(accepted)

    summary = {
        "n_samples": len(rows),
        "n_accepted": n_accept,
        "acceptance_rate": (n_accept / len(rows)) if rows else 0.0,
        "reject_histogram": _reject_histogram(rows),
        "manifest": manifest_path,
        "structures": struct_dir,
        "seed": int(seed),
    }
    with open(os.path.join(out_dir, "export_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        f"CP2LP export: {n_accept}/{len(rows)} accepted "
        f"({100.0 * summary['acceptance_rate']:.1f}%) -> {manifest_path}"
    )
    return summary


def _reject_histogram(rows: list[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for r in rows:
        for reason in r["reject_reasons"]:
            hist[reason] = hist.get(reason, 0) + 1
    return hist


@torch.no_grad()
def measure_real_lps(
    out_dir: str,
    decoded: dict,
    batch: dict,
    mask: torch.Tensor,
    example_ids: list[str],
    manifest_name: str = "real_reference.jsonl",
    clash_threshold_nm: float = CLASH_THRESHOLD_NM,
) -> dict:
    """Scores REAL bound linear peptides with the same code that scores generated ones.

    The report's whole value is the real-vs-generated comparison, and a comparison run
    through two different measurement paths measures the paths as much as the peptides.
    This walks a real LP through the identical AE round trip and the identical geometry,
    clash and terminal-separation functions, writing the same score schema.

    Contact-retention fields are absent rather than zero: a real LP has no source CP, so
    the quantity is undefined, and a zero would silently average into the comparison.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, manifest_name)
    mask = mask.bool()
    b = mask.shape[0]
    tgap = terminal_gap_nm(decoded["coors_nm"], mask)
    aatype = batch["residue_type"].long()

    rows = 0
    with open(manifest_path, "a") as mf:
        for i in range(b):
            sel = mask[i]
            if int(sel.sum()) == 0:
                continue
            one = slice(i, i + 1)
            _, clmetrics = clash_loss(
                atom37=decoded["coors_nm"][one],
                atom_mask=decoded["atom_mask_eff"][one],
                pep_mask=mask[one],
                target_atom37=batch["x_target"][one],
                target_atom_mask=batch["target_mask"][one],
                target_mask=batch["seq_target_mask"][one].bool(),
                threshold_nm=clash_threshold_nm,
            )
            _, gmetrics = open_chain_geometry_loss(
                decoded["coors_nm"][one], decoded["atom_mask_eff"][one], mask[one]
            )
            mf.write(
                json.dumps(
                    {
                        "triplet_id": f"real__{example_ids[i]}",
                        "source_id": example_ids[i],
                        "kind": "real_lp",
                        "peptide_length": int(sel.sum()),
                        "sequence": seq_to_string(aatype[i][sel].cpu().numpy()),
                        "scores": {
                            "terminal_gap_nm": float(tgap[i]),
                            "clash_inter_sum_nm": float(clmetrics.get("clash_inter_sum_nm", 0.0)),
                            "clash_intra_sum_nm": float(clmetrics.get("clash_intra_sum_nm", 0.0)),
                            "clash_min_inter_nm": float(clmetrics.get("clash_min_inter_nm", float("nan"))),
                            "peptide_bond_mae_nm": float(gmetrics["geom_peptide_bond_mae_nm"]),
                            "chain_intact": float(gmetrics["geom_chain_intact_frac"]) == 1.0,
                        },
                    }
                )
                + "\n"
            )
            rows += 1
    logger.info(f"CP2LP reference: measured {rows} real LPs -> {manifest_path}")
    return {"n_real": rows, "manifest": manifest_path}
