"""Similarity-gradient guidance for the linear -> cyclic (LP -> CP) SDEdit.

What this adds to the sampler
-----------------------------
Plain SDEdit controls preservation with ONE knob per track: the start time. Lower `t_ca`
buys the backbone freedom to close a ring and pays for it with pose drift, and the two move
together along a single frontier. This module adds a second, independent handle: after every
Euler step the state is nudged down the gradient of an explicit similarity loss between the
peptide being generated and the linear input,

    x <- step(x_t, v)  -  lambda(t) * ||sampler step|| * normalize(dL/dx)

so the noise level says how much the model MAY move and the guidance says where it may not.
If that only slides along the same closure-vs-retention frontier, guidance is a
reparameterization of `t_ca` and buys nothing -- which is the point of measuring it.

Why post-step and not "add to v"
--------------------------------
Both tracks sample with `sampling_mode: sc`, which converts v -> score -> v with a
1/(1 - t) factor. A delta added to `v` would therefore be rescaled by a factor that diverges
as t -> 1, and `lambda` would not mean the same thing at either end of the schedule. Applying
the correction after the step keeps lambda interpretable: it is a multiple of the step the
sampler itself just took, which also makes one lambda meaningful for BOTH tracks even though
CA coordinates (nm) and latents (arbitrary units) are not comparable quantities.

The reference is free
---------------------
`add_clean_samples` already puts the encoded LINEAR peptide in `batch["x_1"]` for both tracks
-- CA trace and AE posterior mean -- in the same target-centred frame as the sample (the
SDEdit script deliberately never superposes). So the similarity target needs no alignment, no
re-encoding and no decoder pass.

Terminal residues are excluded by default
-----------------------------------------
Terminal travel IS the intended edit: closing a head-to-tail ring means moving the termini
together, and terminal identity changes are how anchors get placed for a disulfide or an
isopeptide. Billing that as damage is what inverted an earlier LNR conclusion, and guiding
against it would have the guidance fight the ring it is being asked for. `exclude_termini`
zeroes the first/last k valid residues in every loss term; k = 0 is kept as an ARM so the
cost of guiding the termini is measured rather than assumed.

Two gradient modes
------------------
identity  dL/dx_t = dL/dx1_hat, exact under the model's own parameterization
          x1_hat = x_t + (1 - t) v when v is held constant. One autograd call on a tensor
          that already exists: no extra network evaluation, no measurable cost.
dps       the full Jacobian: re-runs the denoiser on a differentiable copy of x_t and
          backprops through it (Diffusion Posterior Sampling). ~2.5x the wall-clock of an
          unguided step. Self-contained -- it builds and frees its own graph inside the
          sampler's `no_grad`, so the simulation loop keeps no cross-step activations.

Closure guidance, and why bond distance alone cannot fix an abstention
----------------------------------------------------------------------
The three terms above pull the sample TOWARD the linear input. The `bond_fb` / `bond_mse` /
`anchor_ce` / `anchor_cb` terms below push it toward a CLOSED RING instead, by decoding the
predicted clean sample and reading the same anchor-atom distance the closure metric is scored
with (`per_sample_requested_bond_distance`, so the loss and the metric cannot drift apart).

**A bond-distance loss is identically zero on an abstaining edit.** Measured on
`evaluation_results/sdedit_pocket_20260906_180646`, abstention (`requested_type_satisfied=0`)
is a pure function of the SEQUENCE budget `t_lat`, not of geometry:

    mainchain    0% at every one of the 20 grid cells  (termini are always valid endpoints)
    disulfide    2-12% at t_lat<=0.4  ->  95-100% at t_lat>=0.6
    isopeptide   2-13% at t_lat<=0.4  ->  83-98%  at t_lat>=0.6

The head abstains because the decoded sequence admits no CYS-CYS / LYS-acid pair at all
(`build_cyclization_validity_mask` is empty), and `per_sample_requested_bond_distance` gates
its disulfide/isopeptide distance on exactly that chemistry: with no cysteine at the endpoint
there is no SG atom, `atoms_valid` is False, and the decoder has already multiplied that
atom's coordinates by zero. The gradient of the bond term is therefore not small on the
abstaining population -- it is exactly zero, on 100% of it.

So the term that can actually move an abstention is `anchor_ce`: a cross-entropy on the
DECODED SEQUENCE LOGITS at the two endpoints for the requested chemistry, which reaches the
latent track and asks it to put the anchor residues there. That is also where the measurement
says the failure lives -- once the endpoint identities are right, the anchor atoms are already
essentially bonded (see the sidechain-closure result: NZ-CG at 1.3 A, 85% inside the strict
window). `bond_fb` then holds the geometry while the identity is being fixed, and `anchor_cb`
is the chemistry-free bridge: a flat-bottom on the endpoint CB-CB distance, which is defined
whatever the residues are, so it keeps pushing on the very samples `bond_fb` cannot see.

One consequence worth stating before a launch: `anchor_ce` needs the latent track to be
MOVING. At `t_lat=1.0` that track is frozen, takes no step, and the per-track `step_rms`
scaling makes the guidance a no-op by construction. The tractable abstaining band is
`t_lat` in {0.6, 0.8} for disulfide/isopeptide -- 83-100% abstention, and the latent track
still integrates there.
"""

from __future__ import annotations

import argparse
import math

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
    CB_IDX,
    _gather_atom,
    per_sample_requested_bond_distance,
)
from proteinfoundation.eval.sampled_binder_metrics import _as_residue_mask

CA_IDX = 1
NM_TO_ANG = 10.0
CONTACT_CUTOFF_NM = 1.0  # same CA-CA definition `score_edit` reports contact_retention with
# Chemistry-free endpoint bracket, identical to `cyc_cb_window_success`'s (3, 8) A. Wide on
# purpose: it is a "too far to ever bond" penalty, not a geometry target.
CB_WINDOW_A = (3.0, 8.0)

SIMILARITY_LOSS_NAMES = ("ca_mse", "contact_hinge", "lat_mse")
CLOSURE_LOSS_NAMES = ("bond_fb", "bond_mse", "anchor_ce", "anchor_cb")
LOSS_NAMES = SIMILARITY_LOSS_NAMES + CLOSURE_LOSS_NAMES
# Which track(s) each loss term reaches. A term only ever produces a gradient for its own
# tracks, so a spec of "ca_mse" alone leaves the latent track running exactly as it does
# unguided. Every closure term goes through the AE decoder, whose output is a function of
# BOTH the latents and the CA trace, so all of them list both.
LOSS_MODE = {
    "ca_mse": ("bb_ca",),
    "contact_hinge": ("bb_ca",),
    "lat_mse": ("local_latents",),
    "bond_fb": ("bb_ca", "local_latents"),
    "bond_mse": ("bb_ca", "local_latents"),
    "anchor_ce": ("bb_ca", "local_latents"),
    "anchor_cb": ("bb_ca", "local_latents"),
}
# Residues each chemistry needs at its two endpoints. Must agree with
# `build_cyclization_validity_mask`, which is what decides whether the head abstains.
ISOPEPTIDE_ACIDS = (AA_ASP, AA_GLU)
ISOPEPTIDE_AMIDES = (AA_ASN, AA_GLN)


def parse_loss_spec(spec: str) -> dict[str, float]:
    """"ca_mse,contact_hinge:2" -> {"ca_mse": 1.0, "contact_hinge": 2.0}."""
    losses: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(":")
        name = name.strip()
        if name not in LOSS_NAMES:
            raise SystemExit(f"FATAL: unknown guidance loss {name!r}; expected {list(LOSS_NAMES)}")
        losses[name] = float(w) if w else 1.0
    if not losses:
        raise SystemExit(f"FATAL: empty guidance loss spec {spec!r}")
    return losses


def add_cli_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("similarity guidance (off unless --guidance-w > 0)")
    g.add_argument("--guidance-w", type=float, default=0.0,
                   help="lambda_0. 0 disables guidance entirely and the sampler is called "
                        "with no hook at all, so the unguided arm is bit-identical to the "
                        "pre-guidance script at a fixed seed.")
    g.add_argument("--guidance-loss", default="ca_mse,contact_hinge",
                   help="Comma list, optional :weight. " + ", ".join(LOSS_NAMES))
    g.add_argument("--guidance-schedule", default="decay", choices=["decay", "const"],
                   help="decay: lambda fades as t -> 1 (normalized to 1 at the track's start "
                        "time), so guidance shapes the pose while it is still plastic and "
                        "lets the sampler settle local geometry. const: flat.")
    g.add_argument("--guidance-schedule-pow", type=float, default=1.0)
    g.add_argument("--guidance-exclude-termini", type=int, default=1,
                   help="Residues per end excluded from every loss term. Terminal travel is "
                        "the intended edit, not damage; k=0 measures the cost of guiding it. "
                        "Default 1, not 2, because these peptides are SHORT (lnr_test: median "
                        "10 residues, min 5, a quarter at 6 or fewer) -- k=2 would leave a "
                        "5-mer with a single guided residue and the arm would be inert by "
                        "construction rather than by result.")
    g.add_argument("--guidance-mode", default="identity", choices=["identity", "dps"],
                   help="identity: dL/dx1_hat (free). dps: backprop through the denoiser "
                        "(~2.5x wall-clock).")
    g.add_argument("--guidance-max-disp-A", type=float, default=0.25,
                   help="Cap on the per-residue CA displacement guidance may add in ONE step. "
                        "A blown-up gradient cannot teleport a residue, it just saturates.")
    g.add_argument("--guidance-stride", type=int, default=1,
                   help="Apply guidance every k-th Euler step. The closure terms decode the "
                        "predicted clean sample every step they run, so k>1 is the knob that "
                        "buys the wall-clock back; k=1 (default) guides every step.")
    g.add_argument("--guidance-cb-window-A", nargs=2, type=float, default=list(CB_WINDOW_A),
                   metavar=("LO", "HI"),
                   help="Flat-bottom window for `anchor_cb`, the chemistry-free endpoint "
                        "CB-CB bracket. Default matches `cyc_cb_window_success`.")


def guidance_tag(args) -> str:
    """Short, filename-safe identity of a guidance setting; "" when guidance is off.

    Stamped into `run_key` and the PDB name. Without it a second lambda arm resumes on top of
    the first (same run_key => "already done") and overwrites its PDBs.
    """
    if not getattr(args, "guidance_w", 0.0):
        return ""
    losses = parse_loss_spec(args.guidance_loss)
    body = "-".join(f"{n}{'' if w == 1.0 else round(w, 3)}" for n, w in sorted(losses.items()))
    sched = "c" if args.guidance_schedule == "const" else f"d{args.guidance_schedule_pow:g}"
    stride = int(getattr(args, "guidance_stride", 1) or 1)
    tag = (f"{body}_w{args.guidance_w:g}_{sched}_k{args.guidance_exclude_termini}"
           f"_{args.guidance_mode}")
    # Only widen the key when it is not the default, so existing rows/PDBs keep their names
    # and a resume of an earlier guided run still matches.
    return tag if stride == 1 else f"{tag}_s{stride}"


def terminal_weights(mask: torch.Tensor, k: int) -> torch.Tensor:
    """[B, n] float weights: 1 on valid residues, 0 on the first/last k valid ones."""
    w = mask.float().clone()
    if k <= 0:
        return w
    for b in range(mask.shape[0]):
        idx = mask[b].nonzero().flatten()
        if idx.numel() <= 2 * k:
            raise SystemExit(
                f"FATAL: peptide of {idx.numel()} residues cannot exclude {k} per end -- "
                f"every residue would be unweighted and the guidance would be a silent no-op."
            )
        w[b, idx[:k]] = 0.0
        w[b, idx[-k:]] = 0.0
    return w


class SimilarityGuidance:
    """Per-step similarity-gradient correction; instantiate one per edit, pass as `guidance_fn`."""

    def __init__(self, *, fm, predict_for_sampling, batch, mask, losses: dict[str, float],
                 weight: float, schedule: str, schedule_pow: float, exclude_termini: int,
                 mode: str, t_start: dict[str, float], max_disp_A: float,
                 contact_cutoff_nm: float = CONTACT_CUTOFF_NM,
                 autoencoder=None, cyc_type_idx: int | None = None,
                 allow_asn_gln: bool = True, cb_window_A=CB_WINDOW_A, stride: int = 1):
        if mask.shape[0] != 1:
            raise SystemExit("FATAL: SimilarityGuidance assumes one edit per call (B=1); "
                             f"got batch size {mask.shape[0]}.")
        self.fm = fm
        self.predict = predict_for_sampling
        self.losses = dict(losses)
        self.weight = float(weight)
        self.schedule = schedule
        self.schedule_pow = float(schedule_pow)
        self.mode = mode
        self.t_start = dict(t_start)
        self.max_disp_nm = float(max_disp_A) / NM_TO_ANG if max_disp_A > 0 else 0.0
        self.cutoff = float(contact_cutoff_nm)
        self.stride = max(1, int(stride))

        # The terminal exclusion only ever applies to the SIMILARITY terms. The closure terms
        # are defined ON the termini -- they are the anchors -- so a closure-only spec must not
        # inherit the exclusion, and in particular must not inherit its "peptide too short"
        # fatal, which would kill a 5-mer that has a perfectly well-defined ring.
        self._has_similarity = any(n in SIMILARITY_LOSS_NAMES for n in losses)
        self.wres = terminal_weights(mask.bool(),
                                     exclude_termini if self._has_similarity else 0)  # [B, n]
        self.wsum = self.wres.sum().clamp(min=1.0)
        self.ref = {dm: batch["x_1"][dm].detach().clone() for dm in fm.data_modes}
        # Only the tracks some loss term actually reads are ever corrected.
        self.active_modes = sorted({m for n in self.losses for m in LOSS_MODE[n]})
        missing = [m for m in self.active_modes if m not in fm.data_modes]
        if missing:
            raise SystemExit(f"FATAL: guidance loss needs data mode(s) {missing}, which this "
                             f"flow does not have ({fm.data_modes}).")

        self.pairs = None
        if "contact_hinge" in self.losses:
            self.tgt_ca, self.pairs = self._input_contacts(batch, mask)

        # ---- closure terms: decoder, endpoints, requested chemistry ----------------
        self.closure = sorted(n for n in self.losses if n in CLOSURE_LOSS_NAMES)
        self.ae = autoencoder
        self.mask = mask.bool()
        self.cb_lo_A, self.cb_hi_A = float(cb_window_A[0]), float(cb_window_A[1])
        self.cyc_type_idx = None if cyc_type_idx is None else int(cyc_type_idx)
        self.anchor_targets = None
        if self.closure:
            if autoencoder is None or cyc_type_idx is None:
                raise SystemExit(
                    f"FATAL: closure guidance {self.closure} needs `autoencoder` and "
                    "`cyc_type_idx` -- the bond it scores is defined by the decoded structure "
                    "and the REQUESTED chemistry, neither of which the sampler state carries.")
            # Endpoints: first and last valid residue. Same convention as the terminal pair in
            # `build_cyclization_validity_mask` and as the design path's derived endpoints.
            idx = self.mask[0].nonzero().flatten()
            if idx.numel() < 2:
                raise SystemExit("FATAL: closure guidance needs at least 2 valid residues.")
            self.cyc_i = idx[0].reshape(1).long()
            self.cyc_j = idx[-1].reshape(1).long()
            self.cyc_type = torch.full((1,), self.cyc_type_idx, dtype=torch.long,
                                       device=mask.device)
            acids = list(ISOPEPTIDE_ACIDS) + (list(ISOPEPTIDE_AMIDES) if allow_asn_gln else [])
            self.iso_acids = torch.tensor(acids, dtype=torch.long, device=mask.device)
            if "anchor_ce" in self.losses and self.cyc_type_idx == MAINCHAIN:
                # Not fatal: a mixed-chemistry run legitimately includes mainchain arms. But an
                # arm whose headline term is structurally zero must say so, not read as a null.
                print("WARNING: anchor_ce requested for a MAINCHAIN arm -- head-to-tail needs "
                      "no particular residue identity, so this term is exactly zero here and "
                      "the arm is guided by its other terms only.", flush=True)
            # The termini ARE the anchors, so the terminal exclusion must never reach the
            # closure terms; guarding here rather than trusting every future call site.
            self._closure_weightless = True

        self.n_guided = int(self.wres.sum())
        self.n_pairs = int(self.pairs.shape[0]) if self.pairs is not None else 0
        self.n_steps = 0          # steps the hook was called on
        self.n_applied = 0        # steps guidance actually ran on (stride)
        self.n_bond_valid = 0     # steps whose anchor atoms existed at all
        self.disp_nm_sq = 0.0     # accumulated squared CA displacement added by guidance
        self.loss_first: float | None = None
        self.loss_last: float | None = None
        self.terms_first: dict[str, float] = {}
        self.terms_last: dict[str, float] = {}
        self.diag_first: dict[str, float] = {}
        self.diag_last: dict[str, float] = {}

    # ------------------------------------------------------------------ setup
    def _input_contacts(self, batch, mask):
        """Receptor CA (nm) plus the (peptide, receptor) CA pairs in contact in the INPUT."""
        x_target, target_mask = batch.get("x_target"), batch.get("target_mask")
        if x_target is None or target_mask is None:
            raise SystemExit("FATAL: contact_hinge guidance needs a receptor in the batch "
                             "(`x_target`/`target_mask`), which this example has not got.")
        # `target_mask` is ATOM-level [B, T, 37] in compact mode despite the name; the
        # codebase's own normaliser is the only safe way to read it as residues.
        tmask = _as_residue_mask(target_mask.bool())[0]
        tgt = x_target[0]
        tgt_ca = tgt[:, CA_IDX, :] if tgt.dim() == 3 else tgt
        if tmask.shape[0] != tgt_ca.shape[0]:
            raise RuntimeError(f"target mask/coords mismatch: {tmask.shape[0]} vs {tgt_ca.shape[0]}")
        tgt_ca = tgt_ca[tmask].detach()                                    # [T, 3] nm
        pep_ca = batch["coords_nm"][0][:, CA_IDX, :].detach()              # [n, 3] nm, input LP
        d = torch.cdist(pep_ca, tgt_ca)                                    # [n, T]
        keep = (d < self.cutoff) & (self.wres[0] > 0)[:, None]
        pairs = keep.nonzero()                                             # [P, 2]
        if pairs.numel() == 0:
            raise SystemExit("FATAL: no input interface contacts survive the terminal "
                             "exclusion -- contact_hinge would be a silent no-op.")
        return tgt_ca, pairs

    # ------------------------------------------------------------------ closure loss
    def _anchor_logprob(self, logp: torch.Tensor) -> torch.Tensor:
        """log p(the requested chemistry is placeable at the two endpoints), per endpoint.

        `logp` is [n, 20] log-softmax over the decoded sequence logits. This is the term that
        addresses an ABSTENTION: the head emits a null edge when the decoded sequence admits
        no candidate pair, and that is a statement about residue identity at i/j, not about
        where the atoms are. Returns a scalar log-probability; the caller negates it.

        MAINCHAIN needs no particular identity (the validity mask always accepts the terminal
        pair), so its "requirement" is vacuous and this returns exactly 0.
        """
        i, j = int(self.cyc_i), int(self.cyc_j)
        if self.cyc_type_idx == MAINCHAIN:
            return torch.zeros((), device=logp.device, dtype=logp.dtype)
        if self.cyc_type_idx == DISULFIDE:
            return 0.5 * (logp[i, AA_CYS] + logp[j, AA_CYS])
        # ISOPEPTIDE is directional but the label does not fix which endpoint is the lysine,
        # and either assignment satisfies the head. Scoring the max (log-sum-exp, the smooth
        # one) rather than a fixed orientation keeps the gradient from fighting itself when
        # the sample has already committed to the other one.
        acid_i = torch.logsumexp(logp[i, self.iso_acids], dim=0)
        acid_j = torch.logsumexp(logp[j, self.iso_acids], dim=0)
        both = torch.stack([logp[i, AA_LYS] + acid_j, logp[j, AA_LYS] + acid_i])
        return 0.5 * torch.logsumexp(both, dim=0)

    def _closure_terms(self, x1: dict[str, torch.Tensor]):
        """Decode the predicted clean sample once and build every requested closure term."""
        decoded = self.ae.decode(z_latent=x1["local_latents"], ca_coors_nm=x1["bb_ca"],
                                 mask=self.mask)
        atom_mask = decoded["atom_mask"].bool() & self.mask[..., None]
        terms: dict[str, torch.Tensor] = {}
        diag: dict[str, float] = {}

        if "anchor_ce" in self.losses:
            logp = torch.log_softmax(decoded["seq_logits"][0], dim=-1)
            lp = self._anchor_logprob(logp)
            terms["anchor_ce"] = -lp
            diag["anchor_p"] = float(torch.exp(lp.detach()))

        if "anchor_cb" in self.losses:
            # Chemistry-free: CB exists whatever the endpoint residues are (bar glycine), so
            # unlike the bond terms this one still has a gradient on an abstaining sample.
            cb_i, vi = _gather_atom(decoded["coors_nm"], atom_mask, self.cyc_i, CB_IDX)
            cb_j, vj = _gather_atom(decoded["coors_nm"], atom_mask, self.cyc_j, CB_IDX)
            d = torch.linalg.norm(cb_i - cb_j, dim=-1) * NM_TO_ANG          # [1]
            lo = torch.full_like(d, self.cb_lo_A)
            hi = torch.full_like(d, self.cb_hi_A)
            pen = flat_bottom_penalty(d, lo, hi) * (vi & vj).float()
            terms["anchor_cb"] = pen.mean()
            diag["cb_dist_A"] = float(d.detach()[0])
            diag["cb_valid"] = float((vi & vj).float()[0])

        if "bond_fb" in self.losses or "bond_mse" in self.losses:
            bond = per_sample_requested_bond_distance(
                pred_atom37=decoded["coors_nm"], atom37_mask=atom_mask,
                seq_tokens=decoded["residue_type"].long(),
                i=self.cyc_i, j=self.cyc_j, cyc_type=self.cyc_type,
            )
            dist, lo, hi = bond["dist_A"], bond["window_lo_A"], bond["window_hi_A"]
            # `atoms_valid` is False on exactly the abstaining population: no cysteine at the
            # endpoint means no SG, and the decoder has already zeroed that coordinate. The
            # multiply keeps the term differentiable-but-zero there rather than letting a
            # meaningless distance drive the sample somewhere.
            valid = bond["atoms_valid"].float()
            if "bond_fb" in self.losses:
                terms["bond_fb"] = (flat_bottom_penalty(dist, lo, hi) * valid).mean()
            if "bond_mse" in self.losses:
                # The literal squared-error form, kept as an ABLATION against `bond_fb`. It is
                # one-sided in effect -- it pulls toward the window centre from both sides and
                # so keeps pulling on a bond that already closes, which is how sampled designs
                # ended up with anchor atoms fused at 0.76-0.87 A. Prefer `bond_fb`.
                centre = 0.5 * (lo + hi)
                terms["bond_mse"] = (((dist - centre) ** 2) * valid).mean()
            diag["bond_dist_A"] = float(dist.detach()[0])
            diag["bond_atoms_valid"] = float(valid[0])
        return terms, diag

    # ------------------------------------------------------------------ loss
    def _loss(self, x1: dict[str, torch.Tensor]):
        terms: dict[str, torch.Tensor] = {}
        diag: dict[str, float] = {}
        if self.closure:
            terms, diag = self._closure_terms(x1)
        if "ca_mse" in self.losses:
            d2 = ((x1["bb_ca"] - self.ref["bb_ca"]) ** 2).sum(-1)          # [B, n] nm^2
            terms["ca_mse"] = (d2 * self.wres).sum() / self.wsum
        if "contact_hinge" in self.losses:
            pi, ti = self.pairs[:, 0], self.pairs[:, 1]
            d = torch.linalg.vector_norm(x1["bb_ca"][0][pi] - self.tgt_ca[ti], dim=-1)
            # One-sided on purpose: a contact that tightens is free, only stretching past the
            # cutoff that `contact_retention` is measured with costs anything.
            terms["contact_hinge"] = (torch.relu(d - self.cutoff) ** 2).mean()
        if "lat_mse" in self.losses:
            d2 = ((x1["local_latents"] - self.ref["local_latents"]) ** 2).sum(-1)
            terms["lat_mse"] = (d2 * self.wres).sum() / self.wsum
        total = sum(self.losses[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}, diag

    def _lambda(self, dm: str, t_val: float) -> float:
        if self.schedule == "const":
            return self.weight
        span = max(1.0 - self.t_start[dm], 1e-6)
        frac = min(max((1.0 - t_val) / span, 0.0), 1.0)   # 1 at the track's start time, 0 at t=1
        return self.weight * (frac ** self.schedule_pow)

    # ------------------------------------------------------------------ gradients
    def _grads(self, batch, x_1_pred):
        if self.mode == "identity":
            with torch.enable_grad():
                leaves = {dm: x_1_pred[dm].detach().clone().requires_grad_(True)
                          for dm in self.fm.data_modes}
                total, terms, diag = self._loss(leaves)
                grads = torch.autograd.grad(
                    total, [leaves[dm] for dm in self.active_modes], allow_unused=True)
            return dict(zip(self.active_modes, grads)), float(total.detach()), terms, diag

        # dps: the real Jacobian. Own graph, built and freed here, so the sampler loop still
        # keeps no activations between steps.
        with torch.enable_grad():
            xt = {dm: batch["x_t"][dm].detach().clone().requires_grad_(True)
                  for dm in self.fm.data_modes}
            b = dict(batch)
            b["x_t"] = xt
            nn_out = self.predict(b, mode="full")
            nn_out = self.fm.nn_out_add_clean_sample_prediction(b, nn_out)
            x1 = {dm: nn_out[dm]["x_1"] for dm in self.fm.data_modes}
            total, terms, diag = self._loss(x1)
            grads = torch.autograd.grad(
                total, [xt[dm] for dm in self.active_modes], allow_unused=True)
        return dict(zip(self.active_modes, grads)), float(total.detach()), terms, diag

    # ------------------------------------------------------------------ the hook
    def __call__(self, x, *, batch, x_pre, x_1_pred, t, dt, mask, step):
        self.n_steps += 1
        if self.stride > 1 and (int(step) % self.stride) != 0:
            return x
        grads, total, terms, diag = self._grads(batch, x_1_pred)
        if self.loss_first is None:
            self.loss_first, self.terms_first, self.diag_first = total, terms, diag
        self.loss_last, self.terms_last, self.diag_last = total, terms, diag
        self.n_applied += 1
        self.n_bond_valid += int(diag.get("bond_atoms_valid", 0.0) > 0.5)

        out = dict(x)
        m = mask[..., None].float()
        for dm in self.active_modes:
            g = grads.get(dm)
            if g is None:
                continue
            g = g * m
            # Per-track scale-free normalization: unit RMS gradient, then a multiple of the
            # step the sampler itself just took. That is what lets ONE lambda mean the same
            # thing for nm-valued coordinates and for unitless latents.
            gn = torch.sqrt((g ** 2).sum() / self.wsum.clamp(min=1.0))
            if not torch.isfinite(gn) or float(gn) <= 0.0:
                continue
            step_rms = torch.sqrt((((x_pre[dm] - out[dm]) ** 2) * m).sum() / self.wsum)
            if float(step_rms) <= 0.0:
                continue  # a frozen track takes no step, so guidance takes none either
            lam = self._lambda(dm, float(t[dm].flatten()[0]))
            if lam == 0.0:
                continue
            delta = (lam * step_rms / gn) * g
            if dm == "bb_ca" and self.max_disp_nm > 0:
                norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                delta = delta * (self.max_disp_nm / norm.clamp(min=1e-12)).clamp(max=1.0)
            delta = delta * m
            out[dm] = out[dm] - delta
            if dm == "bb_ca":
                self.disp_nm_sq += float((delta ** 2).sum() / self.wsum)
        return out

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict[str, float]:
        """Diagnostics for the JSONL row: a guidance arm that did nothing must be visible."""
        out = {
            "guid_n_steps": self.n_steps,
            # Steps guidance actually ran on. With --guidance-stride k this is n_steps/k, and
            # a zero here is the signature of an arm that never fired.
            "guid_n_applied": self.n_applied,
            # Fraction of guided steps where the requested chemistry's anchor ATOMS existed in
            # the decoded structure. On an abstaining edit this is 0 and the bond term
            # contributed exactly nothing -- read it before reading a null bond-guidance result.
            "guid_bond_valid_frac": (self.n_bond_valid / self.n_applied) if self.n_applied else None,
            # How much peptide the guidance was actually allowed to touch. On a 5-mer with
            # k=2 this is 1, and an arm that looks like a null result is really an inert one.
            "guid_n_guided_residues": self.n_guided,
            "guid_n_contacts": self.n_pairs,
            # RMS CA displacement summed over steps, in Angstrom: the honest answer to
            # "did the guidance actually push?" independent of whether it helped.
            "guid_ca_disp_A": math.sqrt(self.disp_nm_sq) * NM_TO_ANG,
            "guid_loss_first": self.loss_first,
            "guid_loss_last": self.loss_last,
        }
        for k, v in (self.terms_first or {}).items():
            out[f"guid_{k}_first"] = v
        for k, v in (self.terms_last or {}).items():
            out[f"guid_{k}_last"] = v
        for k, v in (self.diag_first or {}).items():
            out[f"guid_{k}_first"] = v
        for k, v in (self.diag_last or {}).items():
            out[f"guid_{k}_last"] = v
        return out
