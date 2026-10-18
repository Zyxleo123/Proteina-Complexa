"""Non-adversarial objectives for the two backbone arms. Shared, identical, both arms.

WHAT IS DIFFERENT FROM `losses.py`, AND WHY EACH CHANGE WAS FORCED.

`losses.py` scores a DECODED all-atom peptide with a predicted sequence. These arms emit a
backbone and copy the source sequence, so three of its five terms have nothing to act on:
sequence retention (the sequence is fixed and exact by construction), contact retention
through sidechains (there are none until the packer runs, outside the gradient), and the
intra-residue geometry term in the form it takes there. What remains, re-derived:

preservation
    backbone displacement from the source in the FIXED RECEPTOR FRAME, weighted toward the
    residues that carry the source's interface. The weight set is derived ONCE from the
    source's all-atom contacts and then frozen -- it is a property of the input, not of the
    current output, and recomputing it per step would let the generator shrink its own
    objective by walking residues out of the interface.

sterics
    THE NORMALISATION IS THE WHOLE POINT HERE. `losses.clash_loss` divides the receptor
    overlap sum by the number of peptide-by-receptor PAIRS, batch-wide. That denominator
    scales with receptor size -- which is how the clash term came to be measured at 0.06% of
    the objective WHILE BEING VIOLATED, diluted roughly 1000x. This module normalises per
    PEPTIDE RESIDUE, separately per example, then averages across examples, so the term means
    "how much overlap does a residue of this peptide carry" and a receptor twice as large does
    not halve the penalty.

    MEASURED, one fixed clash, receptor padded with distant residues that add pairs but no
    overlap (`test_steric_loss_does_not_dilute_with_receptor_size`):

        receptor residues     this module     losses.clash_loss
                   20            0.514052              0.026932
                  100            0.514052              0.006586
                  400            0.514052              0.002771
                 1000            0.514052              0.002008

    Exactly constant against 13.4x of drift, and 19x to 256x larger in absolute terms. The
    old column is the measured dilution, not a hypothetical one.

    Nor is it averaged only over violating pairs: that denominator shrinks as the generator
    improves, so the per-pair mean can stay flat while the violation count collapses, and the
    loss reports no progress exactly when progress is happening.

terminal opening
    unchanged from `losses.terminal_opening_reward`, threshold included. Re-exported rather
    than reimplemented so the two paths cannot drift.

raw geometry safeguard (ARM A ONLY)
    peptide-bond and intra-residue bond geometry on the RAW output, flat-bottomed. Arm B
    satisfies these by construction and gets the term at weight zero -- the one place the two
    arms are allowed to differ, because for Arm B it would be a gradient-free constant and
    for Arm A it is the only thing standing between a free residual and a severed chain.

Nanometres, receptor frame. Every term returns ``(loss, metrics)`` with metrics detached.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from proteinfoundation.cp2lp.backbone import (
    BB_SLOT,
    TEMPLATE_C_N_NM,
    TEMPLATE_CA_C_NM,
    TEMPLATE_N_CA_NM,
    backbone_to_atom37,
)
from proteinfoundation.cp2lp.geometry import (
    PEPTIDE_BOND_TOL_NM,
    heavy_atom_distances,
    safe_norm,
    soft_contact,
)
from proteinfoundation.cp2lp.losses import (  # noqa: F401  (re-exported on purpose)
    CLASH_THRESHOLD_NM,
    MIN_OPEN_TERMINAL_GAP_NM,
    terminal_opening_reward,
)

#: Coefficient on the depth-growing half of the steric penalty. 0.0 is the historical linear
#: hinge and is the default, so no existing arm or control changes. See `_overlap_penalty` for
#: why a linear hinge sets an equilibrium rather than enforcing a floor, and why 2.0 is the
#: recommended first alternative. NOT weight-neutral: it scales the deep tail by up to 1 + this.
DEFAULT_STERIC_DEPTH_GAIN = 0.0

#: Residues per end exempted from the preservation weight. 0 is the historical behaviour --
#: every residue charged, termini included -- and is the default so no existing arm changes.
#: See the exemption block in `source_interface_weights` for why a non-zero value is what lets
#: a preservation ladder test "keep the interface AND stay open" rather than trace the
#: trade-off between them.
DEFAULT_PRESERVE_TERMINAL_EXEMPT_K = 0

#: Intra-residue bond tolerance for the Arm A safeguard, nm. Same magnitude as the peptide
#: bond's, because a 0.02 nm error in N-CA is no more acceptable than one in C-N.
INTRA_BOND_TOL_NM = 0.02

#: Interface weight given to a residue that carries no source contact. Nonzero, so that a
#: non-interface residue is still held near the source rather than free to wander into the
#: receptor -- it just costs less than an interface residue does.
NON_INTERFACE_WEIGHT = 0.25

#: Pseudo-Huber knee for the preservation term, nm. Below this the term is quadratic and its
#: gradient vanishes smoothly; above it the term is linear, which is the L1-like behaviour
#: intended (a displacement twice as large should cost twice as much, not four times).
#:
#: THIS EXISTS BECAUSE A PLAIN NORM HAS A KINK AT ITS MINIMUM, and that kink made the two arms
#: differ at initialisation for a reason that is not the representation. MEASURED at zero head
#: output, where both arms emit exactly the source and all three loss values are bit-identical:
#:
#:                    max|gen - src|      preservation gradient norm
#:     cartesian        1.000e-06 nm        0.0000       (exactly zero)
#:     torsion          1.733e-06 nm        0.3752       (nonzero)
#:
#: The Cartesian residual is EXACTLY zero in float, so `d/dx sqrt(x^2 + eps)` is exactly 0.
#: The NeRF rebuild leaves ~1.7e-06 nm of genuine float residue, which is above `sqrt(eps)`,
#: so the derivative is live -- and it is then multiplied by the torsion lever arm. In the
#: per-term gradient probe on a real batch that showed up as `preserve` carrying 0.0% of Arm
#: A's head gradient and 61.4% of Arm B's, from identical outputs: a gradient whose magnitude
#: was large and whose DIRECTION was numerical noise.
#:
#: 0.01 nm is 0.1 angstrom -- below the AE decode's own reproducibility and far below any edit
#: anyone would call an edit, so nothing meaningful is damped. At 1.7e-06 nm the gradient is
#: ~1.7e-04 of its asymptote; at 0.1 nm it is 0.995 of it.
PRESERVE_HUBER_NM = 0.01


# ------------------------------------------------------------------ the interface weights


def source_interface_weights(
    src_atom37: torch.Tensor,
    src_atom_mask: torch.Tensor,
    pep_mask: torch.Tensor,
    target_atom37: torch.Tensor,
    target_atom_mask: torch.Tensor,
    target_mask: torch.Tensor,
    cutoff_nm: float = 0.8,
    min_strength: float = 0.5,
    non_interface: float = NON_INTERFACE_WEIGHT,
    terminal_exempt_k: int = DEFAULT_PRESERVE_TERMINAL_EXEMPT_K,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``[b, n]`` per-residue preservation weight, from the SOURCE's all-atom contacts.

    DERIVED ONCE, FROM THE SOURCE, AND DETACHED. Three properties, each deliberate:

    * from the SOURCE's ALL-ATOM structure, including its sidechains -- that is where the
      binding mode actually lives, and the generated backbone has no sidechains to read it
      from. The plan's "derive that residue set once from source all-atom contacts".
    * DETACHED, so no gradient flows into the weights. A differentiable weight is an
      objective the generator can reduce by making the weights smaller instead of by
      preserving anything.
    * computed per residue, not per contact, because what the preservation term needs is
      "how much does moving THIS residue cost".

    Returns the weights and the diagnostics that say whether the weighting is doing anything
    at all -- a batch where every residue is an interface residue would make this term a
    uniform RMSD penalty wearing an interface-weighted name.
    """
    pm = pep_mask.bool()
    tm = target_mask.bool()
    with torch.no_grad():
        d = heavy_atom_distances(src_atom37, src_atom_mask, target_atom37, target_atom_mask)
        pair = (pm[:, :, None] & tm[:, None, :]).float()
        c = soft_contact(d, cutoff_nm) * pair
        strength = c.amax(dim=-1)  # [b, n] strongest contact this residue makes
        is_iface = (strength >= min_strength).float() * pm.float()
        w = (non_interface + (1.0 - non_interface) * strength.clamp(0.0, 1.0)) * pm.float()
        # TERMINAL EXEMPTION. Zero the weight on the first/last `k` residues, so the
        # preservation term stops charging for the one displacement the task REQUIRES.
        #
        # WHY. Opening a macrocycle from a 0.35 nm terminal gap to real LPs' 2.14 nm means the
        # two ring-bond residues must separate by ~1.8 nm, and this term bills every residue's
        # displacement from the source with no positional exemption -- a terminus that touches
        # the receptor gets weight up to 1.0. So raising `preserve_weight` suppresses the
        # OPENING, which is why zeroing it is what opened the gap in the first place. That
        # makes a plain weight ladder a trade-off curve rather than an attempt to break it.
        #
        # What the exemption buys: terminal travel becomes free while interior interface drift
        # stays charged -- and interface drift is what actually costs retention (partial
        # corr(edit, retention | gap) = -0.367, against corr(gap, retention | edit) = +0.005).
        #
        # WHAT IT COSTS, so the choice of k is made with eyes open: the termini carry 18.5% /
        # 42.3% / 66.9% of interface contacts at k = 1 / 2 / 3 per end, so exempting them gives
        # up charging for that share of the interface. k=1 is the minimal exemption; k=2 may be
        # needed because a torsion lever arm cannot move residue 0 without residue 1 following.
        #
        # MASK-AWARE BY CONSTRUCTION. "Last k" is per example and must come from the MASK, not
        # from the end of the padded tensor -- peptides here are 5-16 residues in a padded
        # batch, so indexing the tensor's tail would exempt PADDING and leave the real
        # C-terminus fully charged, which is the silent-no-op version of this feature. Counted
        # from both ends with a cumulative sum over the mask.
        k = int(terminal_exempt_k)
        if k > 0:
            mi = pm.long()
            fwd = torch.cumsum(mi, dim=1)                        # 1..L at valid positions
            rev = torch.cumsum(mi.flip(1), dim=1).flip(1)        # L..1 at valid positions
            exempt = pm & ((fwd <= k) | (rev <= k))
            w = w * (~exempt).to(w.dtype)

        nres = pm.float().sum(dim=1).clamp(min=1.0)
        metrics = {
            "iface_residue_frac": (is_iface.sum(dim=1) / nres).mean(),
            "iface_weight_mean": (w.sum(dim=1) / nres).mean(),
            # Non-zero only when the exemption is on, so a run's own log says whether the knob
            # took effect rather than leaving it to be inferred from the config.
            "preserve_exempt_frac": (
                (exempt.float().sum(dim=1) / nres).mean() if k > 0
                else torch.zeros((), device=w.device, dtype=w.dtype)
            ),
        }
    return w, metrics


# ----------------------------------------------------------------------- preservation


def backbone_preservation_loss(
    gen_bb: torch.Tensor,
    src_bb: torch.Tensor,
    pep_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    huber_nm: float = PRESERVE_HUBER_NM,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Interface-weighted backbone displacement from the source, in the receptor frame.

    NO ALIGNMENT. The peptide is never superposed onto the source before measuring, because
    the receptor frame IS the frame of interest: a peptide that has rotated out of the pocket
    and back in has not preserved its pose, however well it superposes onto itself. Aligning
    first would measure internal conformational change and call it pose preservation.

    A training PROXY, and only that: whether sidechain contacts actually survive is decided
    after packing, by the evaluation, and this term cannot see it.

    Args:
        gen_bb / src_bb: ``[b, n, 4, 3]`` nm.
        pep_mask: ``[b, n]``.
        weights: ``[b, n]`` from :func:`source_interface_weights`; uniform if omitted.
    """
    pm = pep_mask.bool()
    w = torch.ones_like(pm, dtype=gen_bb.dtype) if weights is None else weights.detach()
    w = w * pm.to(w.dtype)

    d = safe_norm(gen_bb - src_bb, dim=-1)  # [b, n, 4]
    # PSEUDO-HUBER, not the raw norm: `sqrt(d^2 + k^2) - k`. Linear in d far from zero, so
    # the term still bills a displacement twice as large at twice the cost, but quadratic
    # near zero so the gradient VANISHES at the identity instead of being a unit vector of
    # float noise. See `PRESERVE_HUBER_NM` for the measurement that forced this.
    k = float(huber_nm)
    cost = torch.sqrt(d * d + k * k) - k if k > 0 else d
    per_res = cost.mean(dim=-1)  # average over the four backbone atoms
    denom = w.sum(dim=1).clamp(min=1e-6)
    per_example = (w * per_res).sum(dim=1) / denom

    with torch.no_grad():
        nres = pm.float().sum(dim=1).clamp(min=1.0)
        ca = safe_norm(
            gen_bb[:, :, BB_SLOT["CA"]] - src_bb[:, :, BB_SLOT["CA"]], dim=-1
        ) * pm.float()
        metrics = {
            "preserve_bb_rmsd_nm": (
                ((d * d).mean(-1) * pm.float()).sum(dim=1) / nres
            ).sqrt().mean(),
            "preserve_ca_rmsd_nm": ((ca * ca).sum(dim=1) / nres).sqrt().mean(),
            "preserve_ca_max_nm": (ca).amax(dim=1).mean(),
        }
    return per_example.mean(), metrics


# ---------------------------------------------------------------------------- sterics


def _overlap_penalty(
    gap: torch.Tensor, threshold_nm: float, depth_gain: float
) -> torch.Tensor:
    """`relu(threshold - d)` plus an optional term that grows with overlap DEPTH. Nanometres.

    WHY A SHAPE KNOB AND NOT JUST A BIGGER WEIGHT. Measured on the release arms at their last
    step: `steric_weight` is 1.0, the largest of the active terms, and the term's gradient norm
    at the head is 0.78 against a total generator norm of 2.42 -- so steric is neither
    down-weighted nor drowned out. And the arms still ended at `steric_min_inter_nm` = 0.0705
    (cold) and 0.206 (warm) against real bound LPs' 0.262, i.e. INSIDE a covalent bond length.
    Three explanations are therefore already eliminated: the weight, the threshold (0.20 against
    real 0.262, only 0.06 nm apart) and "the term is drowned out".

    What is left is the functional form. `relu` is LINEAR in overlap depth, so its derivative is
    a constant 1 per violating pair no matter how deep the overlap goes; resistance per pair
    never rises, only the pair COUNT does. Against terms that keep pushing inward that balances
    at whatever depth makes count x constant-slope equal the opposing pull, and nothing in the
    form says 0.07 nm is catastrophically worse than 0.19 nm. **A linear hinge sets an
    equilibrium; it cannot enforce a floor.** Rosetta's packer does not produce this geometry
    precisely because `fa_rep` is steeply superlinear near contact -- which is also why the same
    packer repacks the zero-edit source backbone with 0 clashes out of 96 while the generated
    backbones clash on 74%.

    THE FORM. One knob, added to the historical term rather than replacing it:

        relu + depth_gain * relu**2 / threshold

    * `depth_gain = 0.0` returns `relu(threshold - d)` on the SAME tensor path -- the historical
      loss bit-for-bit, so every existing arm and control is unchanged. It is the default.
    * The penalty is MONOTONICALLY >= the historical one everywhere, for any `depth_gain >= 0`.
      That is the property a bare exponent does not have and the reason this is not
      `relu ** p`: `threshold * (relu/threshold) ** 2` is *weaker* than linear for every
      overlap shallower than half the threshold, and its restoring force decays to 0 at the
      boundary, so it would trade a deep-overlap fix for a shallow-overlap regression right
      where most pairs sit.
    * The restoring force is `1 + 2 * depth_gain * relu / threshold`: never below the linear 1,
      and rising with depth. At the cold arm's measured operating point (d = 0.0705, i.e.
      overlap 0.65 * threshold) `depth_gain = 2.0` gives 3.6x the linear force, while a bare
      exponent of 2 would give only 1.3x there.

    SCALE, stated because it does move: at full overlap the penalty is `threshold * (1 +
    depth_gain)`, so the term's magnitude grows by up to `1 + depth_gain`. That is intended --
    it is the deep tail being made expensive -- but it means `depth_gain` is not weight-neutral
    and an arm that changes both is not single-variable. Change this one first.

    Bounded and polynomial by construction, with no division by a variable: an inverse-power
    soft core would be stronger and would also reintroduce the NaN class this project has
    already paid for once.
    """
    overlap = F.relu(threshold_nm - gap)
    if depth_gain == 0.0:  # bit-identical to the historical term, not merely equivalent
        return overlap
    # `threshold_nm` is a positive constant, so this cannot divide by zero and the square keeps
    # the gradient finite everywhere including exactly at the boundary.
    return overlap + depth_gain * overlap * overlap / threshold_nm


def backbone_steric_loss(
    gen_bb: torch.Tensor,
    pep_mask: torch.Tensor,
    target_atom37: torch.Tensor | None = None,
    target_atom_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    threshold_nm: float = CLASH_THRESHOLD_NM,
    min_seq_sep: int = 2,
    depth_gain: float = DEFAULT_STERIC_DEPTH_GAIN,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backbone-vs-receptor and backbone-vs-backbone overlap, normalised PER PEPTIDE RESIDUE.

    `exponent` controls how the penalty grows with overlap DEPTH -- see `_overlap_penalty`. It
    defaults to the historical linear hinge. The `*_violating_*` metrics threshold the raw gap
    rather than the penalty, so incidence stays comparable across exponents even though the
    severity sums do not.

    The denominator is the thing to read here. For each example the overlap sum is divided by
    that example's peptide residue count, and only then averaged across examples. The two
    denominators this deliberately is NOT:

      * per peptide-by-receptor PAIR -- scales with receptor size, which is how the clash term
        in `losses.py` ended up diluted ~1000x and measured at 0.06% of the objective while
        being violated. A real clash must not get cheaper because the receptor is bigger.
      * per VIOLATING pair -- the denominator shrinks as the generator improves, so the
        reported value can sit flat while violations collapse, and the term is least
        informative exactly when it is working.

    Bonded exclusions: intra-peptide pairs with sequence separation < `min_seq_sep` are
    dropped. At separation 1 the residues share a peptide bond and C(i)-N(i+1) is SUPPOSED to
    be at bonding distance, so penalising it would fight the geometry term directly. Residues
    i and i+2 are kept -- they are genuinely non-bonded and can genuinely clash.

    The receptor keeps its full heavy-atom complement: a backbone that overlaps a receptor
    SIDECHAIN is clashing, and pretending the receptor is a backbone too would licence it.
    """
    pm = pep_mask.bool()
    b, n = pm.shape
    dtype = gen_bb.dtype
    atom37, atom_mask = backbone_to_atom37(gen_bb, pm)
    nres = pm.float().sum(dim=1).clamp(min=1.0)  # [b]

    d_self = heavy_atom_distances(atom37, atom_mask, atom37, atom_mask)  # [b, n, n]
    ar = torch.arange(n, device=gen_bb.device)
    sep = (ar[None, :] - ar[:, None]).abs()
    self_pair = (sep >= min_seq_sep)[None].expand(b, n, n) & pm[:, :, None] & pm[:, None, :]
    # /2 because the matrix counts each pair twice.
    pen_self = _overlap_penalty(d_self, threshold_nm, depth_gain)
    intra = (pen_self * self_pair.to(dtype)).sum(dim=(1, 2)) / 2.0
    per_example = intra / nres

    metrics: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        metrics["steric_intra_per_res_nm"] = (intra / nres).mean()
        n_self = self_pair.to(dtype).sum(dim=(1, 2)).clamp(min=1.0) / 2.0
        metrics["steric_intra_violating_frac"] = (
            ((F.relu(threshold_nm - d_self) > 0) & self_pair).to(dtype).sum(dim=(1, 2)) / 2.0
            / n_self
        ).mean()

    if target_atom37 is not None:
        tm = target_mask.bool()
        d_t = heavy_atom_distances(atom37, atom_mask, target_atom37, target_atom_mask)
        t_pair = pm[:, :, None] & tm[:, None, :]
        pen_t = _overlap_penalty(d_t, threshold_nm, depth_gain)
        inter = (pen_t * t_pair.to(dtype)).sum(dim=(1, 2))
        per_example = per_example + inter / nres
        with torch.no_grad():
            metrics["steric_inter_per_res_nm"] = (inter / nres).mean()
            far = torch.full_like(d_t, 1e4)
            dmin = torch.where(t_pair, d_t, far).amin(dim=(1, 2))
            metrics["steric_min_inter_nm"] = dmin.mean()
            # A COUNT as well as a sum: severity and incidence answer different questions,
            # and the evaluation reports both.
            metrics["steric_inter_violating_count"] = (
                ((F.relu(threshold_nm - d_t) > 0) & t_pair).to(dtype).sum(dim=(1, 2)) / nres
            ).mean()

    return per_example.mean(), metrics


# ------------------------------------------------------- raw geometry safeguard (Arm A)


def open_backbone_geometry_loss(
    gen_bb: torch.Tensor,
    pep_mask: torch.Tensor,
    peptide_tol_nm: float = PEPTIDE_BOND_TOL_NM,
    intra_tol_nm: float = INTRA_BOND_TOL_NM,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flat-bottomed bond-length geometry on the RAW backbone. ARM A's safeguard.

    THE TERMINAL BOND IS ABSENT, as everywhere on this path: bonded pairs are ``(i, i+1)`` for
    ``i < L-1`` and the ``(L-1, 0)`` closure is never scored. Including it would tell the
    generator that the ring it was asked to open is a defect to repair.

    Flat-bottomed, so once a bond is inside tolerance its gradient is exactly zero and the
    generator cannot buy adversarial slack by over-tightening a bond past the point where
    tightening means anything.

    Arm B runs this at weight zero: every bond it emits is the template within float
    precision, so the term is a constant and its gradient is noise around zero. It is still
    MEASURED for Arm B -- `backbone.backbone_validity` is the measurement, and the claim "by
    construction" needs a number behind it.
    """
    pm = pep_mask.bool()
    n = pm.shape[1]
    nat, caat, cat = (gen_bb[:, :, BB_SLOT[k]] for k in ("N", "CA", "C"))

    nres = pm.float().sum(dim=1).clamp(min=1.0)
    d_nca = safe_norm(caat - nat, dim=-1)
    d_cac = safe_norm(cat - caat, dim=-1)
    intra = (
        F.relu((d_nca - TEMPLATE_N_CA_NM).abs() - intra_tol_nm)
        + F.relu((d_cac - TEMPLATE_CA_C_NM).abs() - intra_tol_nm)
    ) * pm.float()
    per_example = intra.sum(dim=1) / nres

    metrics: dict[str, torch.Tensor] = {}
    if n > 1:
        pair = pm[:, :-1] & pm[:, 1:]
        npair = pair.float().sum(dim=1).clamp(min=1.0)
        d_cn = safe_norm(nat[:, 1:] - cat[:, :-1], dim=-1)
        dev = (d_cn - TEMPLATE_C_N_NM).abs()
        pep = F.relu(dev - peptide_tol_nm) * pair.float()
        per_example = per_example + pep.sum(dim=1) / npair
        with torch.no_grad():
            metrics["geom_peptide_bond_dev_nm"] = ((dev * pair).sum(1) / npair).mean()
            metrics["geom_peptide_bond_pass"] = (
                (((dev <= peptide_tol_nm) | ~pair) & pair).float().sum(1) / npair
            ).mean()
            # The ALL-BONDS test, which is the one that matters: a peptide with one break in
            # it is not a peptide, so reaching 0.95 of peptides needs a per-bond pass rate of
            # 0.9945 and a near-miss distribution is no comfort.
            metrics["geom_chain_intact_frac"] = (
                ((dev <= peptide_tol_nm) | ~pair).all(dim=1).float().mean()
            )
    with torch.no_grad():
        metrics["geom_intra_bond_dev_nm"] = (
            (((d_nca - TEMPLATE_N_CA_NM).abs() + (d_cac - TEMPLATE_CA_C_NM).abs()) * pm).sum(1)
            / nres
        ).mean()
    return per_example.mean(), metrics


# ------------------------------------------------------------------- real-LP anchor


def backbone_reconstruction_loss(
    pred_bb: torch.Tensor,
    true_bb: torch.Tensor,
    pep_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Plain Cartesian backbone reconstruction. The real-LP anchor's criterion, both arms.

    ONE CRITERION FOR BOTH ARMS, evaluated on the built output. Arm B is scored AFTER its
    builder rather than on its torsions, because a torsion-space criterion and a Cartesian one
    are different objectives and the arms would be anchored to different things. Scoring both
    in Cartesian nm means the anchor is the same requirement, reached two ways.

    The template reconstruction FLOOR applies: under `ANGLE_SOURCE_TEMPLATE` Arm B cannot get
    below ~0.118 nm here however well it predicts, so this value is only comparable across
    arms once the floor is subtracted -- which is why the floor is measured and recorded
    rather than discovered in a loss curve.
    """
    pm = pep_mask.bool()
    d = safe_norm(pred_bb - true_bb, dim=-1)  # [b, n, 4]
    m = pm[..., None].expand_as(d).to(d.dtype)
    denom = m.sum().clamp(min=1.0)
    loss = (d * m).sum() / denom
    with torch.no_grad():
        metrics = {
            "anchor_bb_rmsd_nm": ((d * d * m).sum() / denom).sqrt(),
            "anchor_bb_max_nm": (d * m).max(),
        }
    return loss, metrics
