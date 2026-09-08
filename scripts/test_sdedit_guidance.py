"""CPU unit tests for the LP->CP similarity guidance. No GPU, no checkpoint, seconds to run.

Run:  .venv/bin/python scripts/test_sdedit_guidance.py

Covers the three things that would make a guided sweep look like a result while being wrong:
  * the analytic gradient disagreeing with a finite difference (guidance pushing sideways),
  * the terminal exclusion not actually excluding (guidance fighting the ring it asks for),
  * an arm that is silently inert (lambda applied to a zero-size sampler step, run_key
    colliding with the unguided arm so every row resumes as "already done").
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sdedit_guidance import (  # noqa: E402
    NM_TO_ANG,
    SimilarityGuidance,
    guidance_tag,
    parse_loss_spec,
    terminal_weights,
)
from proteinfoundation.cyclization.constants import (  # noqa: E402
    AA_ASP,
    AA_CYS,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)
from proteinfoundation.eval.cyclic_reconstruction_metrics import (  # noqa: E402
    CB_IDX,
    CG_IDX,
    C_IDX,
    NZ_IDX,
    N_IDX,
    SG_IDX,
)

N_PEP, N_TGT = 12, 20
MODES = ["bb_ca", "local_latents"]


def fake_fm():
    """Only `data_modes` is used by identity-mode guidance."""
    return types.SimpleNamespace(data_modes=MODES)


def fake_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(1, N_PEP, dtype=torch.bool)
    coords = torch.zeros(1, N_PEP, 37, 3)
    # A peptide laid along x, and a receptor sheet 0.6 nm away: everything is in contact,
    # so the hinge has work to do only once residues are pushed away.
    coords[0, :, 1, 0] = torch.arange(N_PEP, dtype=torch.float32) * 0.38
    x_target = torch.zeros(1, N_TGT, 37, 3)
    x_target[0, :, 1, 0] = torch.arange(N_TGT, dtype=torch.float32) * 0.38
    x_target[0, :, 1, 1] = 0.6
    batch = {
        "mask": mask,
        "coords_nm": coords,
        "x_target": x_target,
        "target_mask": torch.ones(1, N_TGT, 37, dtype=torch.bool),
        "x_1": {"bb_ca": coords[:, :, 1, :].clone(),
                "local_latents": torch.randn(1, N_PEP, 8, generator=g)},
    }
    return batch, mask


def make_guidance(losses, *, k=2, w=1.0, schedule="const", pow_=1.0, max_disp_A=0.25, seed=0):
    batch, mask = fake_batch(seed)
    guid = SimilarityGuidance(
        fm=fake_fm(), predict_for_sampling=None, batch=batch, mask=mask,
        losses=losses, weight=w, schedule=schedule, schedule_pow=pow_,
        exclude_termini=k, mode="identity", t_start={"bb_ca": 0.4, "local_latents": 0.6},
        max_disp_A=max_disp_A,
    )
    return guid, batch, mask


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"  ok  {name}")


# --------------------------------------------------------------------------- specs
def test_spec_and_tag():
    check("spec default weight", parse_loss_spec("ca_mse,contact_hinge")
          == {"ca_mse": 1.0, "contact_hinge": 1.0})
    check("spec explicit weight", parse_loss_spec("ca_mse:0.5") == {"ca_mse": 0.5})
    try:
        parse_loss_spec("ca_rmsd")
    except SystemExit:
        print("  ok  spec rejects unknown loss")
    else:
        raise AssertionError("FAIL: unknown loss name accepted")

    a = types.SimpleNamespace(guidance_w=0.0, guidance_loss="ca_mse", guidance_schedule="decay",
                              guidance_schedule_pow=1.0, guidance_exclude_termini=2,
                              guidance_mode="identity")
    check("tag empty when off", guidance_tag(a) == "")
    a.guidance_w = 2.0
    t2 = guidance_tag(a)
    a.guidance_w = 4.0
    check("tag separates lambda arms", t2 != guidance_tag(a), f"{t2} vs {guidance_tag(a)}")
    a.guidance_exclude_termini = 0
    check("tag separates k arms", guidance_tag(a) != t2)
    check("tag is filename safe", all(c.isalnum() or c in "._-" for c in guidance_tag(a)),
          guidance_tag(a))


# --------------------------------------------------------------------------- weights
def test_terminal_weights():
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, 2:9] = True                       # 7 valid residues at positions 2..8
    w = terminal_weights(mask, 2)
    kept = w[0].nonzero().flatten().tolist()
    check("k=2 drops 2 valid residues per end", kept == [4, 5, 6], kept)
    check("k=0 keeps all valid", terminal_weights(mask, 0)[0].sum() == 7)
    check("padding never weighted", float(terminal_weights(mask, 0)[0, 0]) == 0.0)
    try:
        terminal_weights(mask, 4)
    except SystemExit:
        print("  ok  refuses k that would zero the whole peptide")
    else:
        raise AssertionError("FAIL: k >= L/2 silently produced an inert guidance")


# --------------------------------------------------------------------------- gradients
def test_gradient_matches_finite_difference():
    for spec in ({"ca_mse": 1.0}, {"contact_hinge": 1.0}, {"lat_mse": 1.0},
                 {"ca_mse": 1.0, "contact_hinge": 2.0}):
        guid, batch, _ = make_guidance(spec)
        torch.manual_seed(0)
        x1 = {"bb_ca": batch["x_1"]["bb_ca"] + 0.35 * torch.randn(1, N_PEP, 3),
              "local_latents": batch["x_1"]["local_latents"] + 0.3 * torch.randn(1, N_PEP, 8)}
        x1 = {k: v.double() for k, v in x1.items()}
        guid.ref = {k: v.double() for k, v in guid.ref.items()}
        guid.wres = guid.wres.double()
        guid.tgt_ca = guid.tgt_ca.double() if guid.pairs is not None else None

        leaves = {k: v.clone().requires_grad_(True) for k, v in x1.items()}
        total, _, _ = guid._loss(leaves)
        analytic = torch.autograd.grad(total, list(leaves.values()), allow_unused=True)
        eps = 1e-6
        for (mode, g) in zip(leaves.keys(), analytic):
            if g is None:
                continue
            for _ in range(4):
                i = int(torch.randint(0, N_PEP, (1,)))
                j = int(torch.randint(0, x1[mode].shape[-1], (1,)))
                pert = {k: v.clone() for k, v in x1.items()}
                pert[mode][0, i, j] += eps
                lp, _, _ = guid._loss(pert)
                pert[mode][0, i, j] -= 2 * eps
                lm, _, _ = guid._loss(pert)
                fd = float((lp - lm) / (2 * eps))
                an = float(g[0, i, j])
                check(f"grad[{'+'.join(spec)}][{mode}][{i},{j}] == fd",
                      abs(fd - an) <= 1e-5 + 1e-3 * abs(fd), f"analytic {an:.3e} vs fd {fd:.3e}")


def test_terminal_residues_get_no_gradient():
    guid, batch, _ = make_guidance({"ca_mse": 1.0, "contact_hinge": 1.0}, k=2)
    x1 = {"bb_ca": batch["x_1"]["bb_ca"] + 0.5, "local_latents": batch["x_1"]["local_latents"]}
    leaves = {k: v.clone().requires_grad_(True) for k, v in x1.items()}
    total, _, _ = guid._loss(leaves)
    g = torch.autograd.grad(total, leaves["bb_ca"])[0][0]
    check("first 2 residues unguided", float(g[:2].abs().sum()) == 0.0, str(g[:2]))
    check("last 2 residues unguided", float(g[-2:].abs().sum()) == 0.0, str(g[-2:]))
    check("interior residues guided", float(g[2:-2].abs().sum()) > 0.0)


def test_contact_hinge_is_one_sided():
    guid, batch, _ = make_guidance({"contact_hinge": 1.0}, k=0)
    closer = {"bb_ca": batch["x_1"]["bb_ca"].clone(), "local_latents": batch["x_1"]["local_latents"]}
    closer["bb_ca"][0, :, 1] += 0.3                       # move INTO the receptor
    leaves = {k: v.clone().requires_grad_(True) for k, v in closer.items()}
    total, _, _ = guid._loss(leaves)
    check("tightening a contact is free", float(total) == 0.0, f"loss {float(total)}")
    farther = {"bb_ca": batch["x_1"]["bb_ca"].clone(), "local_latents": batch["x_1"]["local_latents"]}
    farther[ "bb_ca"][0, :, 1] -= 2.0                     # tear the interface apart
    total2, _, _ = guid._loss({k: v.clone() for k, v in farther.items()})
    check("breaking contacts costs", float(total2) > 0.0, f"loss {float(total2)}")


# --------------------------------------------------------------------------- the update
def test_update_descends_and_respects_cap():
    guid, batch, mask = make_guidance({"ca_mse": 1.0}, k=2, w=1.0, max_disp_A=0.25)
    torch.manual_seed(1)
    x_pre = {"bb_ca": batch["x_1"]["bb_ca"] + 0.4 * torch.randn(1, N_PEP, 3),
             "local_latents": batch["x_1"]["local_latents"].clone()}
    x_new = {k: v + 0.05 * torch.randn_like(v) for k, v in x_pre.items()}
    x1_pred = {k: v.clone() for k, v in x_new.items()}    # x1_hat = the current state, for the test
    t = {dm: torch.full((1,), 0.5) for dm in MODES}
    out = guid(x_new, batch=batch, x_pre=x_pre, x_1_pred=x1_pred, t=t,
               dt={dm: 0.01 for dm in MODES}, mask=mask, step=0)

    before = ((x_new["bb_ca"] - guid.ref["bb_ca"]) ** 2 * guid.wres[..., None]).sum()
    after = ((out["bb_ca"] - guid.ref["bb_ca"]) ** 2 * guid.wres[..., None]).sum()
    check("update descends the similarity loss", float(after) < float(before),
          f"{float(before):.5f} -> {float(after):.5f}")
    disp = (out["bb_ca"] - x_new["bb_ca"])
    check("excluded termini are not displaced", float(disp[0, :2].abs().sum()) == 0.0)
    check("latent track untouched by a bb-only loss",
          torch.equal(out["local_latents"], x_new["local_latents"]))
    max_disp_A = float(torch.linalg.vector_norm(disp, dim=-1).max()) * NM_TO_ANG
    check("per-step displacement cap respected", max_disp_A <= 0.25 + 1e-6, f"{max_disp_A:.4f} A")
    st = guid.stats()
    check("stats report a nonzero push", st["guid_ca_disp_A"] > 0 and st["guid_n_steps"] == 1, str(st))


def test_frozen_track_takes_no_guidance():
    """A track the sampler did not move must not be moved by guidance either."""
    guid, batch, mask = make_guidance({"ca_mse": 1.0}, k=2)
    x_pre = {"bb_ca": batch["x_1"]["bb_ca"] + 0.4, "local_latents": batch["x_1"]["local_latents"]}
    x_new = {k: v.clone() for k, v in x_pre.items()}       # zero-size step
    t = {dm: torch.full((1,), 0.5) for dm in MODES}
    out = guid(x_new, batch=batch, x_pre=x_pre, x_1_pred={k: v.clone() for k, v in x_new.items()},
               t=t, dt={dm: 0.0 for dm in MODES}, mask=mask, step=0)
    check("zero sampler step => zero guidance", torch.equal(out["bb_ca"], x_new["bb_ca"]))
    check("inert step is visible in stats", guid.stats()["guid_ca_disp_A"] == 0.0)


def test_schedule_fades_to_zero_at_t1():
    guid, _, _ = make_guidance({"ca_mse": 1.0}, w=3.0, schedule="decay", pow_=1.0)
    check("decay is 1 at the track start", abs(guid._lambda("bb_ca", 0.4) - 3.0) < 1e-9)
    check("decay halves at the midpoint", abs(guid._lambda("bb_ca", 0.7) - 1.5) < 1e-9)
    check("decay vanishes at t=1", abs(guid._lambda("bb_ca", 1.0)) < 1e-12)
    gc, _, _ = make_guidance({"ca_mse": 1.0}, w=3.0, schedule="const")
    check("const does not fade", gc._lambda("bb_ca", 0.99) == 3.0)


# --------------------------------------------------------------------------- the hook itself
def test_sampler_hook_defaults_off():
    import inspect

    from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher

    for fn in (ProductSpaceFlowMatcher.partial_simulation, ProductSpaceFlowMatcher._simulation_loop):
        sig = inspect.signature(fn)
        check(f"{fn.__name__} takes guidance_fn", "guidance_fn" in sig.parameters)
        check(f"{fn.__name__} defaults to None", sig.parameters["guidance_fn"].default is None)
    src = inspect.getsource(ProductSpaceFlowMatcher._simulation_loop)
    check("hook is applied after the Euler step",
          src.index("simulation_step(") < src.index("guidance_fn is not None"))




# --------------------------------------------------------------------- closure guidance
class FakeAE:
    """A differentiable stand-in for the partial autoencoder's `decode`.

    Real enough for the properties under test: the sequence is a function of the LATENTS
    (so `anchor_ce` has a gradient into that track), the atom coordinates are a function of
    the CA TRACE (so the bond terms have a gradient into that one), and the atom-presence
    mask is derived from the decoded residue types exactly as the real decoder's is -- which
    is what makes a missing SG on a non-cysteine endpoint a hard zero rather than a small
    number.
    """

    def __init__(self, latent_dim=8, n_aa=20, seed=0):
        from openfold.np import residue_constants
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(latent_dim, n_aa, generator=g)
        self.atom37_mask = torch.tensor(residue_constants.restype_atom37_mask[:n_aa],
                                        dtype=torch.bool)
        # Distinct offsets so no two anchor atoms coincide; nm.
        self.offsets = torch.zeros(37, 3)
        for k, atom in enumerate([N_IDX, C_IDX, CB_IDX, SG_IDX, NZ_IDX, CG_IDX]):
            self.offsets[atom, 0] = 0.05 * (k + 1)
            self.offsets[atom, 1] = 0.03 * (k + 1)

    def bias_toward(self, aa_idx, positions, strength=20.0):
        """Force the decoded residue at `positions` to be `aa_idx` by biasing the logits."""
        self.bias = getattr(self, "bias", None)
        return aa_idx, positions, strength

    def decode(self, *, z_latent, ca_coors_nm, mask):
        logits = z_latent @ self.W                                   # [B, n, 20]
        for aa_idx, positions, strength in getattr(self, "_forced", []):
            for pos in positions:
                logits = logits + torch.nn.functional.one_hot(
                    torch.tensor(aa_idx), logits.shape[-1]
                ).float() * strength * (
                    torch.arange(logits.shape[1]) == pos
                ).float()[None, :, None]
        aatype = logits.argmax(-1)
        coors = ca_coors_nm[..., None, :] + self.offsets[None, None]  # [B, n, 37, 3]
        return {
            "coors_nm": coors * self.atom37_mask[aatype][..., None],
            "seq_logits": logits,
            "residue_type": aatype,
            "residue_mask": mask,
            "atom_mask": self.atom37_mask[aatype],
        }

    def force(self, pairs):
        """pairs: list of (aa_idx, [positions]). Cleared by passing []."""
        self._forced = [(aa, pos, 40.0) for aa, pos in pairs]


def make_closure_guidance(losses, cyc_type, *, ae=None, k=1, w=1.0, seed=0, stride=1):
    batch, mask = fake_batch(seed)
    ae = ae or FakeAE()
    guid = SimilarityGuidance(
        fm=fake_fm(), predict_for_sampling=None, batch=batch, mask=mask,
        losses=losses, weight=w, schedule="const", schedule_pow=1.0,
        exclude_termini=k, mode="identity", t_start={"bb_ca": 0.4, "local_latents": 0.6},
        max_disp_A=0.25, autoencoder=ae, cyc_type_idx=cyc_type, stride=stride,
    )
    return guid, batch, mask, ae


def leaves_of(batch):
    return {dm: batch["x_1"][dm].detach().clone().requires_grad_(True) for dm in MODES}


def test_bond_term_is_exactly_zero_when_the_head_would_abstain():
    """The headline claim: a bond-distance loss cannot move an abstaining edit.

    With no cysteine at the endpoints there is no SG atom, `atoms_valid` is False, and the
    gradient is not merely small -- it is identically zero, on 100% of the abstaining
    population. An arm built only on bond distance would therefore be inert by construction
    exactly where it was aimed.
    """
    guid, batch, _, ae = make_closure_guidance({"bond_fb": 1.0}, DISULFIDE)
    ae.force([])                                  # endpoints are whatever the latents decode to
    leaves = leaves_of(batch)
    total, terms, diag = guid._loss(leaves)
    check("abstaining bond term is zero", float(total) == 0.0, f"loss={float(total)}")
    check("abstaining atoms_valid is 0", diag["bond_atoms_valid"] == 0.0)
    g = torch.autograd.grad(total, [leaves["bb_ca"], leaves["local_latents"]],
                            allow_unused=True, materialize_grads=True)
    check("abstaining bond gradient is zero",
          all(float(x.abs().max()) == 0.0 for x in g))

    # Same peptide, same coordinates, cysteine endpoints: now it bites.
    ae.force([(AA_CYS, [0, N_PEP - 1])])
    total2, _, diag2 = guid._loss(leaves_of(batch))
    check("with CYS endpoints atoms_valid is 1", diag2["bond_atoms_valid"] == 1.0)
    check("with CYS endpoints the bond term is live", float(total2) > 0.0,
          f"loss={float(total2)} dist={diag2['bond_dist_A']:.2f} A")


def test_anchor_ce_has_a_gradient_into_the_latent_track():
    """The term that CAN move an abstention: it reaches the sequence, not the coordinates."""
    guid, batch, _, _ = make_closure_guidance({"anchor_ce": 1.0}, DISULFIDE)
    leaves = leaves_of(batch)
    total, _, diag = guid._loss(leaves)
    g_lat, = torch.autograd.grad(total, [leaves["local_latents"]], retain_graph=True)
    check("anchor_ce reaches the latents", float(g_lat.abs().max()) > 0.0)
    check("anchor_ce gradient is on the endpoints only",
          float(g_lat[0, 1:N_PEP - 1].abs().max()) == 0.0,
          "an interior residue got a gradient from a terminal-anchor term")
    # One descent step must raise p(CYS at both endpoints), which is the whole objective.
    step = leaves["local_latents"] - 0.5 * g_lat / g_lat.abs().max()
    l2 = {"bb_ca": leaves["bb_ca"], "local_latents": step}
    _, _, diag2 = guid._loss(l2)
    check("anchor_ce descent raises p(anchor)", diag2["anchor_p"] > diag["anchor_p"],
          f"{diag['anchor_p']:.4f} -> {diag2['anchor_p']:.4f}")


def test_anchor_ce_is_exactly_zero_for_mainchain():
    """Head-to-tail needs no particular residue, so the term is vacuous -- and must be 0,
    not a small number that would silently steer a mainchain arm's sequence."""
    guid, batch, _, _ = make_closure_guidance({"anchor_ce": 1.0}, MAINCHAIN)
    total, _, _ = guid._loss(leaves_of(batch))
    check("mainchain anchor_ce is zero", float(total) == 0.0, f"loss={float(total)}")


def test_isopeptide_anchor_is_orientation_free():
    """The label does not say which endpoint is the lysine; both assignments must score the
    same, or the gradient fights a sample that has already committed to the other one."""
    ae_a, ae_b = FakeAE(), FakeAE()
    ae_a.force([(AA_LYS, [0]), (AA_ASP, [N_PEP - 1])])
    ae_b.force([(AA_ASP, [0]), (AA_LYS, [N_PEP - 1])])
    ga, batch, _, _ = make_closure_guidance({"anchor_ce": 1.0}, ISOPEPTIDE, ae=ae_a)
    gb, _, _, _ = make_closure_guidance({"anchor_ce": 1.0}, ISOPEPTIDE, ae=ae_b)
    la, _, _ = ga._loss(leaves_of(batch))
    lb, _, _ = gb._loss(leaves_of(batch))
    check("isopeptide orientation symmetry", abs(float(la) - float(lb)) < 1e-4,
          f"{float(la):.5f} vs {float(lb):.5f}")


def test_anchor_cb_survives_an_abstention():
    """The chemistry-free bridge: defined whatever the endpoint residues are, so it keeps
    pushing on the very samples the bond term cannot see."""
    guid, batch, _, ae = make_closure_guidance({"anchor_cb": 1.0}, DISULFIDE)
    ae.force([])
    leaves = leaves_of(batch)
    total, _, diag = guid._loss(leaves)
    check("cb term is live on an abstention", float(total) > 0.0,
          f"cb_dist={diag['cb_dist_A']:.2f} A (peptide is extended, so outside the window)")
    g, = torch.autograd.grad(total, [leaves["bb_ca"]])
    check("cb term reaches the CA track", float(g.abs().max()) > 0.0)
    # Descending must SHORTEN an over-long endpoint separation.
    step = leaves["bb_ca"] - 0.05 * g / g.abs().max()
    _, _, diag2 = guid._loss({"bb_ca": step, "local_latents": leaves["local_latents"]})
    check("cb descent pulls the endpoints together", diag2["cb_dist_A"] < diag["cb_dist_A"],
          f"{diag['cb_dist_A']:.2f} -> {diag2['cb_dist_A']:.2f} A")


def test_closure_only_spec_ignores_the_terminal_exclusion():
    """The termini ARE the anchors. A closure-only spec must not inherit the exclusion, nor
    its "peptide too short" fatal, which would kill a 5-mer with a well-defined ring."""
    guid, _, _, _ = make_closure_guidance({"bond_fb": 1.0}, DISULFIDE, k=2)
    check("closure-only keeps every residue weighted", int(guid.wres.sum()) == N_PEP,
          f"n_guided={int(guid.wres.sum())} of {N_PEP}")
    guid2, _, _, _ = make_closure_guidance({"bond_fb": 1.0, "ca_mse": 1.0}, DISULFIDE, k=2)
    check("mixed spec still excludes termini", int(guid2.wres.sum()) == N_PEP - 4)


def test_stride_skips_steps_and_is_in_the_tag():
    guid, batch, mask, _ = make_closure_guidance({"anchor_cb": 1.0}, DISULFIDE, stride=4)
    x = {dm: batch["x_1"][dm].clone() for dm in MODES}
    x_pre = {dm: v + 0.01 for dm, v in x.items()}
    t = {dm: torch.full((1,), 0.5) for dm in MODES}
    for step in range(8):
        guid(dict(x), batch=batch, x_pre=x_pre, x_1_pred=batch["x_1"], t=t, dt=None,
             mask=mask, step=step)
    check("stride applies every k-th step", (guid.n_steps, guid.n_applied) == (8, 2),
          f"steps={guid.n_steps} applied={guid.n_applied}")
    args = types.SimpleNamespace(guidance_w=1.0, guidance_loss="bond_fb",
                                 guidance_schedule="const", guidance_schedule_pow=1.0,
                                 guidance_exclude_termini=1, guidance_mode="dps",
                                 guidance_stride=4)
    check("stride is in the run_key tag", guidance_tag(args).endswith("_s4"), guidance_tag(args))
    args.guidance_stride = 1
    check("stride=1 leaves the tag unchanged", not guidance_tag(args).endswith("_s1"))


def main():
    for fn in (test_spec_and_tag, test_terminal_weights, test_gradient_matches_finite_difference,
               test_terminal_residues_get_no_gradient, test_contact_hinge_is_one_sided,
               test_update_descends_and_respects_cap, test_frozen_track_takes_no_guidance,
               test_schedule_fades_to_zero_at_t1, test_sampler_hook_defaults_off,
               test_bond_term_is_exactly_zero_when_the_head_would_abstain,
               test_anchor_ce_has_a_gradient_into_the_latent_track,
               test_anchor_ce_is_exactly_zero_for_mainchain,
               test_isopeptide_anchor_is_orientation_free,
               test_anchor_cb_survives_an_abstention,
               test_closure_only_spec_ignores_the_terminal_exclusion,
               test_stride_skips_steps_and_is_in_the_tag):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL GUIDANCE TESTS PASSED")


if __name__ == "__main__":
    main()
