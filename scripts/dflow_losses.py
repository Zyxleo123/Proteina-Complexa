"""The LP -> CP objective, and the acceptance contract it is trying to satisfy.

One module holds both on purpose. The differentiable terms the optimizer descends and the
hard gates a candidate is finally judged by are two views of the same thresholds, and when
they live apart they drift: a loss that optimizes a 5 A contact cutoff while acceptance
measures 4.5 A produces candidates that look good right up to the point they are scored.
`CONTRACT` below is the single source of truth, read by both.

The acceptance contract
-----------------------
Fixed operational thresholds, not knobs to relax if the pass rate disappoints:

    ring        the requested terminal chemistry closes, measured with the repo's own
                `*_BOND_WINDOW_A` acceptance windows on the DECODED structure
    sequence    >= 80% identity to the ORIGINAL LP and <= 2 substitutions total, anchor
                changes included; explicitly protected positions may not change at all
    contacts    >= 80% of the original peptide-receptor residue-pair contacts retained,
                contact = any heavy-atom pair within 4.5 A
    pose        whole-peptide CA RMSD <= 2 A and original-interface-residue CA RMSD
                <= 1 A, both in the receptor frame with NO superposition
    clash       no severe clash: heavy-atom contacts below `clash_min_A` are counted and
                a candidate carrying any is rejected
    energy      paired Rosetta interface score, CP minus original LP, <= +5 REU, with the
                strict <= 0 tier reported alongside. REU are not kcal/mol. Scored by the
                CPU sidecar, not here.

Why "Cbeta proximity is insufficient" is load-bearing
-----------------------------------------------------
It was measured on this repo and the answer was unambiguous: over 242 scorable de novo
samples, `term_cb_dist_A` separated closed from open rings at AUC 0.462 -- chance. The ring
SHAPE forms whether or not a bond exists. So Cbeta distance is used here only as a
*reachability bracket* inside the loss, never as a closure criterion; acceptance reads the
anchor-atom bond distance and nothing else.

Why the closure loss cannot be allowed to vanish
------------------------------------------------
`per_sample_requested_bond_distance` multiplies its distance by `atoms_valid`, and the
decoder zeroes the coordinates of atoms the argmax residue does not have. For a disulfide
request on a sample whose endpoints decoded as anything but two cysteines there is no SG
atom, so the bond term is not merely small -- it is exactly zero, with exactly zero
gradient, on 100% of that population. Measured abstention rates on the earlier SDEdit grid
were 83-100% for disulfide/isopeptide at t_lat >= 0.6, so this is the common case, not the
edge case.

Three terms therefore carry the closure objective, and `ring_terms` refuses to build an
objective that has only the vanishing one:

    anchor_ce   cross-entropy on the DECODED SEQUENCE LOGITS at the two endpoints for the
                requested chemistry. Always defined -- logits exist whatever the argmax
                says -- and it addresses the failure the measurements actually found:
                isopeptide failure was ~50% anchor IDENTITY, with NZ-CG already at 1.3 A
                once the residues were right.
    anchor_cb   flat-bottom on the endpoint Cbeta-Cbeta distance, with a chemistry-specific
                reachability window. Always defined, and always differentiable through
                SUPERVISED coordinates: `get_cb_position` falls back to a virtual Cbeta
                built from N/CA/C whenever the real one is absent, so this term has a
                gradient even on a glycine endpoint.
    bond_fb     flat-bottom on the true anchor-atom distance in the repo's acceptance
                window. Live only once identity is right, which is when it matters: it
                sharpens geometry rather than finding it.

For MAINCHAIN the closing bond is C(j) -> N(i), both backbone atoms, so `bond_fb` is
always live and needs no help; `anchor_ce` is vacuous there (head-to-tail requires no
particular residue) and returns exactly zero.

Do NOT reach for the decoder's pre-mask coordinates to "fix" the vanishing bond term. The
37-slot coordinate head does emit a position for every slot, but slots for atoms the
residue does not have are never supervised during AE training, so an SG position read off a
proline is an arbitrary number. `decode_final` exposes `coors_nm_raw` for diagnostics; the
objective does not read it.
"""

from __future__ import annotations

import torch

from proteinfoundation.cyclization.bond_loss import flat_bottom_penalty
from proteinfoundation.cyclization.constants import (
    AA_ASN,
    AA_ASP,
    AA_CYS,
    AA_GLN,
    AA_GLU,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    CA_IDX,
    DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE_N_C_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
    get_cb_position,
    per_sample_requested_bond_distance,
)
from proteinfoundation.eval.sampled_binder_metrics import _as_residue_mask

NM_TO_ANG = 10.0

# --------------------------------------------------------------------------- the contract
CONTRACT = {
    # sequence
    "min_identity": 0.80,
    "max_substitutions": 2,
    # contacts: heavy-atom pair cutoff, and the fraction of the ORIGINAL contacts to keep
    "contact_cutoff_A": 4.5,
    "min_contact_retention": 0.80,
    # pose, receptor frame, no superposition
    "max_ca_rmsd_A": 2.0,
    "max_interface_ca_rmsd_A": 1.0,
    # clash: any heavy-atom pair closer than this, excluding bonded/neighbour pairs.
    #
    # CALIBRATED AGAINST NATIVES, not chosen to raise a pass rate. The contract's other
    # numbers are given as fixed operational thresholds; "no severe clash" is the one clause
    # stated qualitatively, so the threshold has to come from somewhere, and the only honest
    # source is the structures the instrument is supposed to accept.
    #
    # Measured by `script_utils/dflow_calibrate.py` on all 60 LNR natives scored against
    # themselves (evaluation_results/dflow_calibration_full/calibration.json, 2026-09-28):
    #
    #     native min heavy-atom distance to receptor: min 1.45 A, p05 2.13 A, median 2.64 A
    #     at 2.7 A  17/60 natives pass   <- the original value REJECTS 43 of 60 real crystals
    #     at 2.5 A  46/60
    #     at 2.3 A  54/60
    #     at 2.0 A  59/60
    #
    # 2.7 A is a hydrogen bond, not a clash, so the old value was measuring the wrong thing:
    # it would have failed every arm's `gate_clash` and driven joint success to 0 for reasons
    # that are entirely about the instrument. That exact failure -- a pass rate that looks
    # like a model result and is an artifact -- has already happened once on this project.
    # 2.0 A sits below every real contact in the native set but well above nothing: two
    # non-bonded heavy atoms inside 2.0 A is a genuine steric overlap.
    "clash_min_A": 2.0,
    # energy (applied by the Rosetta sidecar, stated here so there is one contract)
    "max_ddg_REU": 5.0,
    "strict_ddg_REU": 0.0,
}

# Reported alongside the primary criteria, never in place of them.
MUTATION_BUDGETS = (0, 1, 2, 3)
IDENTITY_TIERS = (0.90, 0.80, 0.70)

# Cbeta-Cbeta reachability brackets, per chemistry. These are "can this bond exist at all"
# windows, NOT closure criteria -- see the AUC 0.462 note in the module docstring.
#   disulfide  canonical CB-CB in a disulfide is ~3.8 A; the bracket is deliberately tight
#              because SG sits a fixed ~1.8 A off CB, so CB-CB really does pin the bond.
#   isopeptide the two side chains are long and of unequal length, so the bracket is wide
#              and is doing nothing more than keeping the endpoints within reach.
CB_WINDOW_A = {
    DISULFIDE: (3.2, 4.6),
    ISOPEPTIDE: (3.5, 8.5),
    MAINCHAIN: (3.0, 8.0),
}
BOND_WINDOW_A = {
    MAINCHAIN: MAINCHAIN_CN_BOND_WINDOW_A,
    DISULFIDE: DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE: ISOPEPTIDE_N_C_BOND_WINDOW_A,
}

ISOPEPTIDE_ACIDS = (AA_ASP, AA_GLU)
ISOPEPTIDE_AMIDES = (AA_ASN, AA_GLN)

LOSS_NAMES = (
    "anchor_ce",
    "anchor_cb",
    "bond_fb",
    "seq_ce",
    "ca_hinge",
    "iface_hinge",
    "contact_hinge",
    "clash",
    "prior",
)
# Terms whose gradient cannot vanish for lack of an anchor atom. An objective must contain
# at least one of these, or it can be identically zero on the whole abstaining population.
ALWAYS_LIVE = ("anchor_ce", "anchor_cb", "seq_ce", "ca_hinge", "iface_hinge", "contact_hinge")

DEFAULT_WEIGHTS = {
    "anchor_ce": 3.0,
    "anchor_cb": 1.0,
    "bond_fb": 1.0,
    "seq_ce": 2.0,
    "ca_hinge": 1.0,
    "iface_hinge": 2.0,
    "contact_hinge": 1.0,
    "clash": 1.0,
    "prior": 0.05,
}


def parse_weights(spec: str) -> dict[str, float]:
    """"anchor_ce:3,seq_ce" -> {"anchor_ce": 3.0, "seq_ce": 1.0}; "" -> DEFAULT_WEIGHTS."""
    if not spec or not spec.strip():
        return dict(DEFAULT_WEIGHTS)
    out: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(":")
        name = name.strip()
        if name not in LOSS_NAMES:
            raise SystemExit(f"FATAL: unknown loss term {name!r}; expected {list(LOSS_NAMES)}")
        out[name] = float(w) if w else 1.0
    if not out:
        raise SystemExit(f"FATAL: empty loss spec {spec!r}")
    active = [k for k, v in out.items() if v != 0.0]
    if not any(k in ALWAYS_LIVE for k in active):
        raise SystemExit(
            f"FATAL: loss spec {spec!r} contains only terms that can vanish when the decoded "
            f"endpoints lack the requested chemistry's anchor atoms (see `ALWAYS_LIVE`). Such "
            "an objective is identically zero with zero gradient on the abstaining population, "
            "which was 83-100% of disulfide/isopeptide edits at t_lat >= 0.6. Add at least one "
            f"of {list(ALWAYS_LIVE)}."
        )
    return out


def objective_tag(weights: dict[str, float]) -> str:
    """Filename- and run_key-safe identity of an objective setting."""
    return "-".join(
        f"{n}{'' if w == 1.0 else round(w, 4)}" for n, w in sorted(weights.items()) if w != 0.0
    )


# ------------------------------------------------------------------- reference extraction
def receptor_heavy_atoms(batch: dict, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Receptor atom37 coordinates (nm) and their presence mask, residues only.

    `target_mask` is ATOM-level [B, T, 37] in compact mode despite the name; reading it as
    a residue mask silently misaligns every interface number, so the codebase's own
    normaliser decides which residues are real and the raw tensor decides which atoms are.
    """
    x_target, target_mask = batch.get("x_target"), batch.get("target_mask")
    if x_target is None or target_mask is None:
        raise SystemExit(
            "FATAL: this objective needs a receptor in the batch (`x_target`/`target_mask`); "
            "this example has none, and a contact/clash term would be a silent no-op."
        )
    tm = target_mask.bool()
    res_keep = _as_residue_mask(tm)[0]  # [T]
    tgt = x_target[0]
    if tgt.dim() == 2:  # CA-only receptor
        coors = tgt[res_keep][:, None, :]  # [T', 1, 3]
        amask = torch.ones(coors.shape[:2], dtype=torch.bool, device=coors.device)
        return coors, amask
    coors = tgt[res_keep]  # [T', 37, 3]
    amask = tm[0][res_keep] if tm[0].dim() == 2 else torch.ones(
        coors.shape[:2], dtype=torch.bool, device=coors.device
    )
    if amask.shape != coors.shape[:2]:
        raise RuntimeError(f"receptor atom mask {tuple(amask.shape)} != coords {tuple(coors.shape[:2])}")
    return coors, amask


def heavy_pair_min_dist(
    a: torch.Tensor, a_mask: torch.Tensor, b: torch.Tensor, b_mask: torch.Tensor
) -> torch.Tensor:
    """Per residue-pair minimum heavy-atom distance, [La, Lb] in the input's units.

    Absent atoms are pushed to +inf rather than dropped, so the min is over real atoms only
    without needing a ragged representation.
    """
    d = torch.cdist(a.reshape(-1, 3), b.reshape(-1, 3))  # [La*37, Lb*37]
    valid = a_mask.reshape(-1)[:, None] & b_mask.reshape(-1)[None, :]
    d = torch.where(valid, d, torch.full_like(d, float("inf")))
    d = d.reshape(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
    return d.amin(dim=(1, 3))  # [La, Lb]


def original_contacts(batch: dict, mask: torch.Tensor, cutoff_A: float | None = None):
    """The ORIGINAL LP's peptide-receptor residue-pair contacts, by heavy-atom distance.

    Returns (pep_idx [P], tgt_idx [P], receptor coords/masks), all peptide-local indices.
    This is the denominator of `contact_retention`, so it is computed from the input
    structure once and never recomputed from a sample.
    """
    cutoff_nm = float(cutoff_A if cutoff_A is not None else CONTRACT["contact_cutoff_A"]) / NM_TO_ANG
    m = mask[0].bool()
    pep = batch["coords_nm"][0][m]  # [Lp, 37, 3]
    pep_amask = batch["coord_mask"][0][m].bool() if "coord_mask" in batch else torch.ones(
        pep.shape[:2], dtype=torch.bool, device=pep.device
    )
    tgt, tgt_amask = receptor_heavy_atoms(batch)
    dmin = heavy_pair_min_dist(pep, pep_amask, tgt, tgt_amask)  # [Lp, T']
    hits = (dmin < cutoff_nm).nonzero()
    if hits.numel() == 0:
        raise SystemExit(
            "FATAL: the input peptide has no heavy-atom contact with its receptor within "
            f"{cutoff_A or CONTRACT['contact_cutoff_A']} A -- contact retention would be "
            "undefined and every candidate would score NaN."
        )
    return hits[:, 0], hits[:, 1], tgt, tgt_amask


def interface_residues(pep_idx: torch.Tensor, n_pep: int, device) -> torch.Tensor:
    """Boolean [Lp]: peptide residues carrying at least one original contact."""
    out = torch.zeros(n_pep, dtype=torch.bool, device=device)
    out[pep_idx.unique()] = True
    return out


# ---------------------------------------------------------------------------- loss terms
def anchor_logprob(logp: torch.Tensor, cyc_type: int, i: int, j: int, allow_asn_gln: bool):
    """log p(requested chemistry is placeable at endpoints i, j), from decoded logits.

    `logp` is [Lp, 20] log-softmax. MAINCHAIN needs no particular identity -- the validity
    mask always accepts the terminal pair -- so its requirement is vacuous and this is
    exactly 0, which keeps a mixed-chemistry run from reading a structurally-inert term as
    a null result.

    ISOPEPTIDE is directional but the label does not fix WHICH endpoint carries the lysine,
    and either assignment satisfies the head. Scoring the smooth max over both orientations
    stops the gradient fighting a sample that has already committed to the other one.
    """
    if cyc_type == MAINCHAIN:
        return torch.zeros((), device=logp.device, dtype=logp.dtype)
    if cyc_type == DISULFIDE:
        return 0.5 * (logp[i, AA_CYS] + logp[j, AA_CYS])
    acids = list(ISOPEPTIDE_ACIDS) + (list(ISOPEPTIDE_AMIDES) if allow_asn_gln else [])
    acid_idx = torch.tensor(acids, dtype=torch.long, device=logp.device)
    acid_i = torch.logsumexp(logp[i, acid_idx], dim=0)
    acid_j = torch.logsumexp(logp[j, acid_idx], dim=0)
    both = torch.stack([logp[i, AA_LYS] + acid_j, logp[j, AA_LYS] + acid_i])
    return 0.5 * torch.logsumexp(both, dim=0)


def ring_terms(decoded: dict, mask: torch.Tensor, cyc_type: int, i: int, j: int,
               allow_asn_gln: bool, want: set[str]):
    """`anchor_ce`, `anchor_cb`, `bond_fb` plus diagnostics, on the decoded structure."""
    m = mask[0].bool()
    coors = decoded["coors_nm"][0][m][None]           # [1, Lp, 37, 3]
    amask = (decoded["atom_mask"].bool() & mask[..., None])[0][m][None]
    aatype = decoded["residue_type"][0][m][None].long()
    terms: dict[str, torch.Tensor] = {}
    diag: dict[str, float] = {}

    if "anchor_ce" in want:
        logp = torch.log_softmax(decoded["seq_logits"][0][m], dim=-1)
        lp = anchor_logprob(logp, cyc_type, i, j, allow_asn_gln)
        terms["anchor_ce"] = -lp
        diag["anchor_p"] = float(torch.exp(lp.detach()))

    if "anchor_cb" in want:
        # Virtual Cbeta from N/CA/C wherever the real one is absent, so this has a gradient
        # even on a glycine endpoint -- and it flows through supervised backbone atoms.
        cb, cb_real = get_cb_position(coors[0], amask[0])   # [Lp, 3], [Lp]
        d = torch.linalg.norm(cb[i] - cb[j]) * NM_TO_ANG
        lo, hi = CB_WINDOW_A[cyc_type]
        terms["anchor_cb"] = flat_bottom_penalty(d, torch.full_like(d, lo), torch.full_like(d, hi))
        diag["cb_dist_A"] = float(d.detach())
        diag["cb_both_real"] = float(bool(cb_real[i]) and bool(cb_real[j]))

    if "bond_fb" in want:
        dev = coors.device
        bond = per_sample_requested_bond_distance(
            pred_atom37=coors,
            atom37_mask=amask,
            seq_tokens=aatype,
            i=torch.tensor([i], dtype=torch.long, device=dev),
            j=torch.tensor([j], dtype=torch.long, device=dev),
            cyc_type=torch.tensor([cyc_type], dtype=torch.long, device=dev),
        )
        valid = bond["atoms_valid"].float()
        terms["bond_fb"] = (
            flat_bottom_penalty(bond["dist_A"], bond["window_lo_A"], bond["window_hi_A"]) * valid
        ).mean()
        diag["bond_dist_A"] = float(bond["dist_A"].detach()[0])
        diag["bond_atoms_valid"] = float(valid[0])
    return terms, diag


def sequence_term(decoded: dict, mask: torch.Tensor, ref_aatype: torch.Tensor,
                  free_positions: set[int], protect: torch.Tensor | None, protect_w: float):
    """Cross-entropy pulling the decoded sequence back to the ORIGINAL LP.

    `free_positions` are exempt: for a disulfide or isopeptide request the two endpoints
    MUST change identity, and billing that as damage would have this term fight `anchor_ce`
    for the same residues. Those two changes are still charged against the <= 2 substitution
    budget by the acceptance check -- exempt from the LOSS, never from the CONTRACT.

    `protect` positions get `protect_w` times the weight. A weight is not a guarantee, which
    is why acceptance also gates on them exactly; see `sequence_report`.
    """
    m = mask[0].bool()
    logits = decoded["seq_logits"][0][m]     # [Lp, 20]
    ref = ref_aatype[0][m].long()            # [Lp]
    w = torch.ones(logits.shape[0], device=logits.device)
    for p in free_positions:
        if 0 <= p < w.shape[0]:
            w[p] = 0.0
    if protect is not None and protect.numel():
        w[protect] = w[protect] * float(protect_w)
    ce = torch.nn.functional.cross_entropy(logits, ref, reduction="none")
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def pose_terms(decoded: dict, mask: torch.Tensor, ref_coords_nm: torch.Tensor,
               iface: torch.Tensor, want: set[str]):
    """Whole-peptide and interface CA hinges, in the receptor frame with NO superposition.

    The receptor is fixed conditioning in a target-centred frame, so the sample is already
    in the input's frame and a subtraction is the honest displacement. A Kabsch fit here
    would hide exactly the motion inside the binding site that the contract is about.

    Hinges, not plain MSE: the contract asks for RMSD below a threshold, and a squared-error
    term keeps pulling a pose that already satisfies it toward zero displacement, which
    spends the ring's freedom on a criterion that is already met.
    """
    m = mask[0].bool()
    ca_out = decoded["coors_nm"][0][m][:, CA_IDX, :]
    ca_in = ref_coords_nm[0][m][:, CA_IDX, :]
    d2 = ((ca_out - ca_in) ** 2).sum(-1)                       # [Lp] nm^2
    terms: dict[str, torch.Tensor] = {}
    diag: dict[str, float] = {}

    rmsd_A = torch.sqrt(d2.mean().clamp(min=0)) * NM_TO_ANG
    diag["ca_rmsd_A"] = float(rmsd_A.detach())
    if "ca_hinge" in want:
        terms["ca_hinge"] = torch.relu(rmsd_A - CONTRACT["max_ca_rmsd_A"]) ** 2

    if iface.any():
        irmsd_A = torch.sqrt(d2[iface].mean().clamp(min=0)) * NM_TO_ANG
        diag["iface_ca_rmsd_A"] = float(irmsd_A.detach())
        if "iface_hinge" in want:
            terms["iface_hinge"] = torch.relu(irmsd_A - CONTRACT["max_interface_ca_rmsd_A"]) ** 2
    elif "iface_hinge" in want:
        raise SystemExit(
            "FATAL: `iface_hinge` was requested but the input peptide has no interface "
            "residue -- the term would be a silent no-op."
        )
    return terms, diag


_BIG_NM = 1e4  # stand-in for "absent atom": far enough that no min or softmin ever picks it


def _pair_dists(a: torch.Tensor, a_mask: torch.Tensor, b: torch.Tensor, b_mask: torch.Tensor,
                tau_nm: float):
    """Per residue-pair heavy-atom distance, hard min and softmin, [La, Lb] each (nm).

    The softmin `-tau * logsumexp(-d/tau)` approaches the true min as tau -> 0 but spreads
    the gradient over every atom pair that is nearly closest. A hard min on 37x37 atom pairs
    routes the whole gradient through one argmin, which makes the term piecewise-flat and
    stalls the optimizer; the hard min is still returned, because that -- not the softmin --
    is what the acceptance check reads.
    """
    d = torch.cdist(a.reshape(-1, 3), b.reshape(-1, 3))
    valid = a_mask.reshape(-1)[:, None] & b_mask.reshape(-1)[None, :]
    d4 = torch.where(valid, d, torch.full_like(d, _BIG_NM)).reshape(
        a.shape[0], a.shape[1], b.shape[0], b.shape[1]
    )
    hard = d4.amin(dim=(1, 3))                                     # [La, Lb]
    flat = d4.permute(0, 2, 1, 3).reshape(a.shape[0], b.shape[0], -1)   # [La, Lb, 37*37]
    soft = -tau_nm * torch.logsumexp(-flat / tau_nm, dim=-1)       # [La, Lb]
    return hard, soft


def contact_clash_terms(decoded: dict, mask: torch.Tensor, pep_idx, tgt_idx, tgt, tgt_amask,
                        want: set[str], cyc_i: int, cyc_j: int, softmin_tau_A: float = 0.3):
    """`contact_hinge` (keep the original contacts) and `clash` (no severe overlap).

    Both read heavy-atom distances, the same definition acceptance uses. Clash is scored
    against the receptor AND within the peptide: closing a ring pulls the two ends together,
    which is exactly the move that can drive the chain into itself, so an inter-chain-only
    clash term would miss the defect this method is most likely to introduce.
    """
    m = mask[0].bool()
    pep = decoded["coors_nm"][0][m]
    pep_amask = (decoded["atom_mask"].bool() & mask[..., None])[0][m]
    tau_nm = float(softmin_tau_A) / NM_TO_ANG
    cutoff_nm = CONTRACT["contact_cutoff_A"] / NM_TO_ANG
    clash_nm = CONTRACT["clash_min_A"] / NM_TO_ANG
    terms: dict[str, torch.Tensor] = {}
    diag: dict[str, float] = {}

    hard, soft = _pair_dists(pep, pep_amask, tgt, tgt_amask, tau_nm)

    if "contact_hinge" in want:
        dsel = soft[pep_idx, tgt_idx]
        # One-sided: a contact that tightens is free; only stretching past the cutoff
        # acceptance measures with costs anything.
        terms["contact_hinge"] = (torch.relu(dsel - cutoff_nm) ** 2).mean() * (NM_TO_ANG ** 2)
        diag["n_contacts_input"] = int(pep_idx.shape[0])
        diag["contact_retention_soft"] = float((hard[pep_idx, tgt_idx].detach() < cutoff_nm).float().mean())

    if "clash" in want:
        pen = (torch.relu(clash_nm - soft) ** 2).sum()
        diag["min_heavy_dist_A"] = float(hard.detach().min() * NM_TO_ANG)
        diag["n_clashes_receptor"] = int((hard.detach() < clash_nm).sum())

        # Intra-peptide: exclude the sequence neighbours (i, i+-1), which are covalently
        # bonded and legitimately close, and the ring-closing pair, whose whole purpose is
        # to come within bonding distance.
        ihard, isoft = _pair_dists(pep, pep_amask, pep, pep_amask, tau_nm)
        Lp = pep.shape[0]
        ar = torch.arange(Lp, device=pep.device)
        far = (ar[:, None] - ar[None, :]).abs() >= 2
        far[cyc_i, cyc_j] = False
        far[cyc_j, cyc_i] = False
        pen = pen + (torch.relu(clash_nm - isoft) ** 2 * far.float()).sum()
        iv = torch.where(far, ihard.detach(), torch.full_like(ihard.detach(), _BIG_NM))
        diag["min_intra_heavy_dist_A"] = float(iv.min() * NM_TO_ANG)
        diag["n_clashes_intra"] = int((iv < clash_nm).sum() // 2)
        terms["clash"] = pen * (NM_TO_ANG ** 2)
    return terms, diag


def prior_term(z: dict[str, torch.Tensor], z_init: dict[str, torch.Tensor],
               mask: torch.Tensor) -> torch.Tensor:
    """Keep the optimized source state near the one it started from, per track.

    D-Flow's own argument is that the ODE map supplies most of the regularization, so this
    is deliberately weak by default. It is normalized per track by the number of valid
    residues and by the track's dimension, because CA coordinates (nm) and latents
    (unitless, 8-dim) are not comparable quantities and one raw sum would let whichever
    track happens to have the larger numbers set the whole penalty.
    """
    m = mask[..., None].float()
    total = None
    for dm, v in z.items():
        d2 = (((v - z_init[dm]) ** 2) * m).sum() / (m.sum().clamp(min=1.0) * v.shape[-1])
        total = d2 if total is None else total + d2
    return total


class Objective:
    """The full LP -> CP objective for one (example, chemistry). Call on a decoded sample.

    Everything that depends only on the INPUT -- original contacts, interface residues,
    reference coordinates and sequence -- is resolved once in the constructor, so an
    optimizer step costs one rollout plus one decode plus this.
    """

    def __init__(self, *, batch, mask, cyc_type: int, weights: dict[str, float],
                 allow_asn_gln: bool = True, protect: list[int] | None = None,
                 protect_w: float = 10.0, free_anchor_positions: bool | None = None):
        if mask.shape[0] != 1:
            raise ValueError(f"Objective is B=1 only; got {mask.shape[0]}")
        self.mask = mask
        self.cyc_type = int(cyc_type)
        self.weights = {k: v for k, v in weights.items() if v != 0.0}
        self.want = set(self.weights)
        self.allow_asn_gln = bool(allow_asn_gln)
        self.protect_w = float(protect_w)

        m = mask[0].bool()
        self.n_pep = int(m.sum())
        # Endpoints: first and last valid residue. Same convention as
        # `build_cyclization_validity_mask`'s terminal pair and as the design path's derived
        # endpoints, and the checkpoint is `terminal_only: true`, so these are the only
        # anchors the model can be asked for.
        self.i, self.j = 0, self.n_pep - 1
        if self.n_pep < 2:
            raise SystemExit("FATAL: a ring needs at least 2 residues.")

        self.ref_coords_nm = batch["coords_nm"].detach()
        self.ref_aatype = (
            batch.get("residue_type_native", batch["residue_type"]).detach()
        )
        # Anchor identity has to change for the side-chain chemistries, so those two
        # positions are exempt from `seq_ce` (never from the substitution budget).
        if free_anchor_positions is None:
            free_anchor_positions = self.cyc_type in (DISULFIDE, ISOPEPTIDE)
        self.free_positions = {self.i, self.j} if free_anchor_positions else set()

        prot = sorted(set(protect or []))
        bad = [p for p in prot if not (0 <= p < self.n_pep)]
        if bad:
            raise SystemExit(
                f"FATAL: protected position(s) {bad} outside the peptide (length {self.n_pep})."
            )
        overlap = sorted(set(prot) & self.free_positions)
        if overlap:
            raise SystemExit(
                f"FATAL: position(s) {overlap} are both protected and required to carry the "
                f"requested chemistry's anchors. A {self.cyc_type} ring at endpoints "
                f"({self.i}, {self.j}) cannot be built without changing them, so this request "
                "is unsatisfiable as stated rather than merely hard."
            )
        self.protect = torch.tensor(prot, dtype=torch.long, device=mask.device) if prot else None

        self.pep_idx = self.tgt_idx = self.tgt = self.tgt_amask = None
        self.iface = torch.zeros(self.n_pep, dtype=torch.bool, device=mask.device)
        if {"contact_hinge", "clash", "iface_hinge"} & self.want:
            self.pep_idx, self.tgt_idx, self.tgt, self.tgt_amask = original_contacts(batch, mask)
            self.iface = interface_residues(self.pep_idx, self.n_pep, mask.device)

    def __call__(self, decoded: dict, z=None, z_init=None):
        """Returns (total, per-term floats, diagnostics)."""
        terms: dict[str, torch.Tensor] = {}
        diag: dict[str, float] = {}

        t, d = ring_terms(decoded, self.mask, self.cyc_type, self.i, self.j,
                          self.allow_asn_gln, self.want)
        terms.update(t)
        diag.update(d)

        if "seq_ce" in self.want:
            terms["seq_ce"] = sequence_term(decoded, self.mask, self.ref_aatype,
                                            self.free_positions, self.protect, self.protect_w)

        t, d = pose_terms(decoded, self.mask, self.ref_coords_nm, self.iface, self.want)
        terms.update(t)
        diag.update(d)

        if {"contact_hinge", "clash"} & self.want:
            t, d = contact_clash_terms(decoded, self.mask, self.pep_idx, self.tgt_idx,
                                       self.tgt, self.tgt_amask, self.want, self.i, self.j)
            terms.update(t)
            diag.update(d)

        if "prior" in self.want:
            if z is None or z_init is None:
                raise ValueError("`prior` needs the source state and its initial value")
            terms["prior"] = prior_term(z, z_init, self.mask)

        total = sum(self.weights[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}, diag
