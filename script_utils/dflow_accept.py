"""The LP -> CP acceptance check: hard gates, applied identically to every arm.

Why this is a module and not a few lines in the driver
-----------------------------------------------------
The whole point of the experiment is a comparison -- D-Flow against SDEdit against guided
SDEdit -- and a comparison is only worth reading if all three arms were judged by the same
instrument. That has gone wrong on this project before: the binder-evaluation harness scored
every one of ten runs at 0.0 because it applied miniprotein thresholds to macrocycles and
refolded a linear peptide, and the pass rate looked like a model result rather than an
instrument artifact.

So `accept()` takes arrays, not files and not a model, and both callers go through it:

    the GPU optimizer      builds arrays from the decoded sample tensors
    the baseline re-score  builds the same arrays by parsing existing runs' PDBs

`CONTRACT` lives in `scripts/dflow_losses.py` next to the differentiable terms, so a
threshold cannot be changed for the loss without changing it for acceptance.

What the gates are
------------------
Every gate is reported as its own boolean AND as the number behind it, because "joint
quality-qualified" is a conjunction and a single pass/fail hides which clause bound. The
substitution budget counts ANCHOR changes: a disulfide request spends its whole budget of 2
on the two cysteines, which is the honest accounting and is why the mutation-budget table
(0/1/2/3) is reported rather than only the contract's value.

`ring_declared` is separate from `ring_closed` on purpose. A ring that closes geometrically
but is written to a PDB with no CONECT record is, to every downstream consumer including
Rosetta, a linear peptide that happens to have its ends near each other -- and the earlier
eval on this project never measured cyclization at all. Geometry and declaration are two
different claims and both are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from dflow_losses import (  # noqa: E402
    CONTRACT,
    IDENTITY_TIERS,
    MUTATION_BUDGETS,
    NM_TO_ANG,
    _pair_dists,
)
from proteinfoundation.cyclization.constants import CYCLIZATION_TYPE_TO_NAME  # noqa: E402
from proteinfoundation.eval.cyclic_reconstruction_metrics import (  # noqa: E402
    CA_IDX,
    get_cb_position,
    per_sample_requested_bond_distance,
)

# Hard min, not softmin: acceptance reads the true closest approach. The softmin in
# `dflow_losses` exists only to keep a gradient alive.
_TAU_NM = 1e-3


def _contact_set(pep, pep_amask, tgt, tgt_amask, cutoff_nm):
    hard, _ = _pair_dists(pep, pep_amask, tgt, tgt_amask, _TAU_NM)
    hits = (hard < cutoff_nm).nonzero()
    return {(int(a), int(b)) for a, b in hits}, hard


def sequence_report(aatype: torch.Tensor, ref_aatype: torch.Tensor,
                    protect: list[int] | None = None) -> dict:
    """Substitutions, identity and protected-position violations vs the ORIGINAL LP.

    This is the explicit final sequence check the contract calls for. A latent-distance
    penalty does not guarantee sequence retention, and neither does freezing the latent
    track: `t_lat = 1` holds the LATENT fixed, but decoding also reads the CA trace, so a
    backbone that moved can decode to a different residue with the latent untouched. The
    only reliable statement is the one made on the decoded argmax, here.
    """
    sub = (aatype != ref_aatype)
    n_sub = int(sub.sum())
    L = int(aatype.shape[0])
    out = {
        "peptide_length": L,
        "n_substitutions": n_sub,
        # Integer division in Python, NOT a float32 tensor mean. `(~sub).float().mean()` on
        # 9 of 10 matches returns 0.8999999761581421, which is BELOW 0.9 -- so a candidate at
        # exactly the 90% tier, or exactly at the contract's 80% gate, would be rejected by
        # float32 rounding alone. The thresholds are round decimals and the counts are small
        # integers, so the ratio is computed exactly.
        "seq_identity": (L - n_sub) / L if L else float("nan"),
        "substituted_positions": sorted(int(p) for p in sub.nonzero().flatten()),
    }
    prot = sorted(set(protect or []))
    viol = [p for p in prot if p < L and bool(sub[p])]
    out["n_protected"] = len(prot)
    out["protected_violations"] = viol
    out["gate_protected"] = len(viol) == 0
    out["gate_identity"] = out["seq_identity"] >= CONTRACT["min_identity"]
    out["gate_substitutions"] = n_sub <= CONTRACT["max_substitutions"]
    # Reported tiers. Never a substitute for the gates above.
    for b in MUTATION_BUDGETS:
        out[f"budget_{b}"] = n_sub <= b
    for tier in IDENTITY_TIERS:
        out[f"identity_ge_{int(tier * 100)}"] = out["seq_identity"] >= tier
    return out


def accept(
    *,
    pep_coors_nm: torch.Tensor,
    pep_amask: torch.Tensor,
    pep_aatype: torch.Tensor,
    ref_coors_nm: torch.Tensor,
    ref_amask: torch.Tensor,
    ref_aatype: torch.Tensor,
    tgt_coors_nm: torch.Tensor,
    tgt_amask: torch.Tensor,
    cyc_type: int,
    cyc_i: int | None = None,
    cyc_j: int | None = None,
    protect: list[int] | None = None,
    ring_declared: bool | None = None,
) -> dict:
    """Every contract metric and gate for one candidate. All tensors are [L, ...] in nm.

    `pep_*` is the candidate, `ref_*` the ORIGINAL LP, `tgt_*` the receptor. No
    superposition anywhere: the receptor is shared and fixed, so a subtraction is the
    displacement inside the binding site, which is what the contract is about. Aligning
    first would report a small RMSD for a peptide that had slid out of the pocket.
    """
    if pep_coors_nm.shape[0] != ref_coors_nm.shape[0]:
        raise ValueError(
            f"candidate length {pep_coors_nm.shape[0]} != original {ref_coors_nm.shape[0]}; "
            "the primary comparison keeps peptide length unchanged."
        )
    L = int(pep_coors_nm.shape[0])
    i = 0 if cyc_i is None else int(cyc_i)
    j = L - 1 if cyc_j is None else int(cyc_j)
    dev = pep_coors_nm.device
    out: dict = {
        "cyc_type": int(cyc_type),
        "cyc_type_name": CYCLIZATION_TYPE_TO_NAME.get(int(cyc_type), str(cyc_type)),
        "cyc_i": i,
        "cyc_j": j,
    }

    # ---------------------------------------------------------------- ring
    bond = per_sample_requested_bond_distance(
        pred_atom37=pep_coors_nm[None],
        atom37_mask=pep_amask[None],
        seq_tokens=pep_aatype[None].long(),
        i=torch.tensor([i], dtype=torch.long, device=dev),
        j=torch.tensor([j], dtype=torch.long, device=dev),
        cyc_type=torch.tensor([int(cyc_type)], dtype=torch.long, device=dev),
    )
    lo, hi = float(bond["window_lo_A"][0]), float(bond["window_hi_A"][0])
    dist = float(bond["dist_A"][0])
    atoms_valid = bool(bond["atoms_valid"][0])
    out.update(
        bond_dist_A=dist if atoms_valid else None,
        bond_window_A=[lo, hi],
        bond_atoms_valid=atoms_valid,
        # An invalid anchor is a FAILED ring, never a missing measurement. Recording it as
        # None here and as False in the gate is what stops an abstention being read as an
        # unscored row and quietly dropped from the denominator.
        gate_ring_closed=bool(atoms_valid and lo <= dist <= hi),
    )
    cb, cb_real = get_cb_position(pep_coors_nm, pep_amask.bool())
    out["term_cb_dist_A"] = float(torch.norm(cb[i] - cb[j]) * NM_TO_ANG)
    out["term_cb_both_real"] = bool(cb_real[i]) and bool(cb_real[j])
    if ring_declared is not None:
        out["gate_ring_declared"] = bool(ring_declared)

    # ---------------------------------------------------------------- sequence
    out.update(sequence_report(pep_aatype, ref_aatype, protect))

    # ---------------------------------------------------------------- pose
    ca_out = pep_coors_nm[:, CA_IDX, :]
    ca_in = ref_coors_nm[:, CA_IDX, :]
    d2 = ((ca_out - ca_in) ** 2).sum(-1)
    out["ca_rmsd_A"] = float(torch.sqrt(d2.mean()) * NM_TO_ANG)
    out["ca_max_dev_A"] = float(torch.sqrt(d2).max() * NM_TO_ANG)
    out["gate_ca_rmsd"] = out["ca_rmsd_A"] <= CONTRACT["max_ca_rmsd_A"]

    # ---------------------------------------------------------------- contacts
    cutoff_nm = CONTRACT["contact_cutoff_A"] / NM_TO_ANG
    before, _ = _contact_set(ref_coors_nm, ref_amask.bool(), tgt_coors_nm, tgt_amask.bool(), cutoff_nm)
    after, hard = _contact_set(pep_coors_nm, pep_amask.bool(), tgt_coors_nm, tgt_amask.bool(), cutoff_nm)
    out["n_contacts_input"] = len(before)
    out["n_contacts_candidate"] = len(after)
    out["contact_retention"] = (len(before & after) / len(before)) if before else None
    out["gate_contacts"] = bool(
        before and out["contact_retention"] >= CONTRACT["min_contact_retention"]
    )

    # Interface RMSD is defined on the ORIGINAL interface residues, so it measures the same
    # residues before and after rather than following the candidate's own contacts around.
    iface = sorted({p for p, _ in before})
    if iface:
        idx = torch.tensor(iface, dtype=torch.long, device=dev)
        out["n_interface_residues"] = len(iface)
        out["iface_ca_rmsd_A"] = float(torch.sqrt(d2[idx].mean()) * NM_TO_ANG)
        out["gate_iface_ca_rmsd"] = out["iface_ca_rmsd_A"] <= CONTRACT["max_interface_ca_rmsd_A"]
    else:
        out["n_interface_residues"] = 0
        out["iface_ca_rmsd_A"] = None
        out["gate_iface_ca_rmsd"] = False

    # ---------------------------------------------------------------- clash
    clash_nm = CONTRACT["clash_min_A"] / NM_TO_ANG
    out["min_heavy_dist_A"] = float(hard.min() * NM_TO_ANG)
    out["n_clashes_receptor"] = int((hard < clash_nm).sum())
    ihard, _ = _pair_dists(pep_coors_nm, pep_amask.bool(), pep_coors_nm, pep_amask.bool(), _TAU_NM)
    ar = torch.arange(L, device=dev)
    far = (ar[:, None] - ar[None, :]).abs() >= 2
    far[i, j] = False
    far[j, i] = False
    iv = torch.where(far, ihard, torch.full_like(ihard, 1e4))
    out["min_intra_heavy_dist_A"] = float(iv.min() * NM_TO_ANG)
    out["n_clashes_intra"] = int((iv < clash_nm).sum() // 2)
    out["gate_clash"] = out["n_clashes_receptor"] == 0 and out["n_clashes_intra"] == 0

    # ---------------------------------------------------------------- joint
    # Structure-and-sequence gates only. The energy gate is applied by the Rosetta sidecar
    # and joined in the report, because it needs a relaxation this function cannot run --
    # so `gate_joint_structural` is explicitly NOT the headline number.
    gates = [
        "gate_ring_closed",
        "gate_identity",
        "gate_substitutions",
        "gate_protected",
        "gate_contacts",
        "gate_ca_rmsd",
        "gate_iface_ca_rmsd",
        "gate_clash",
    ]
    if "gate_ring_declared" in out:
        gates.append("gate_ring_declared")
    out["gates_checked"] = gates
    out["gate_joint_structural"] = all(bool(out[g]) for g in gates)
    out["gates_failed"] = [g for g in gates if not bool(out[g])]
    return out


def pareto_front(rows: list[dict], objectives: list[tuple[str, str]]) -> list[int]:
    """Indices of the nondominated rows. `objectives` is [(key, "min"|"max"), ...].

    Rows missing any objective key, or carrying None there, are dropped rather than
    defaulted: a candidate with no contact retention is not a candidate that retained
    nothing, and giving it a 0 would let it dominate nothing while looking scored.
    """
    usable = [
        k for k, r in enumerate(rows)
        if all(o in r and r[o] is not None for o, _ in objectives)
    ]

    def dominates(a, b):
        better = False
        for key, sense in objectives:
            va, vb = rows[a][key], rows[b][key]
            if sense == "min":
                if va > vb:
                    return False
                if va < vb:
                    better = True
            else:
                if va < vb:
                    return False
                if va > vb:
                    better = True
        return better

    return [a for a in usable if not any(dominates(b, a) for b in usable if b != a)]
