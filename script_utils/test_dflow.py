"""Tests for the D-Flow LP -> CP path.

The one that matters most is `test_rollout_matches_partial_simulation`: this module runs its
own Euler loop so it can wrap each step in a gradient checkpoint, and a reimplemented
sampler that quietly differs from the production one would make every number in the
experiment incomparable to every number already on this repo. It is checked against
`partial_simulation` on a real `ProductSpaceFlowMatcher` with a deterministic stand-in
denoiser, for both integrators, bit for bit.

Everything here runs on CPU in seconds and needs no checkpoint. The things that genuinely
need the model -- a finite gradient through a real trajectory, and one end-to-end pass --
are the GPU smoke, not unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "script_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dflow_accept import accept, pareto_front, sequence_report  # noqa: E402
from dflow_losses import (  # noqa: E402
    CONTRACT,
    NM_TO_ANG,
    anchor_logprob,
    objective_tag,
    parse_weights,
)
from dflow_pdb import (  # noqa: E402
    RingUndeclarable,
    declare_ring,
    ring_atom_names,
    verify_ring,
)
from dflow_rollout import DiffRollout, build_schedule, resolve_sim_params  # noqa: E402
from proteinfoundation.cyclization.constants import AA_ASP, AA_CYS, AA_GLU, AA_LYS, DISULFIDE, ISOPEPTIDE, MAINCHAIN  # noqa: E402
from proteinfoundation.eval.cyclic_reconstruction_metrics import CA_IDX  # noqa: E402
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher  # noqa: E402

LAT_DIM = 8


def _fm(guidance_enabled=True):
    cfg = OmegaConf.create(
        {
            "product_flowmatcher": {
                "bb_ca": {"dim": 3, "guidance_enabled": guidance_enabled, "zero_com_noise": False},
                "local_latents": {"dim": LAT_DIM, "guidance_enabled": guidance_enabled,
                                  "zero_com_noise": False},
            }
        }
    )
    return ProductSpaceFlowMatcher(cfg)


def _sampling_args(mode="sc"):
    """A stand-in for `configs/pipeline/model_sampling.yaml`, same shape and same values."""
    blk = {
        "schedule": {"mode": "log", "p": 2.0},
        "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": mode, "sc_scale_noise": 0.1, "sc_scale_score": 1.0,
            "t_lim_ode": 0.98, "t_lim_ode_below": 0.02, "tsr_k": 1.0, "tsr_sigma": 1.0,
            "center_every_step": False,
        },
    }
    other = {k: (dict(v) if isinstance(v, dict) else v) for k, v in blk.items()}
    other["schedule"] = {"mode": "power", "p": 2.0}
    other["gt"] = {"mode": "tan", "p": 1.0, "clamp_val": None}
    return {"bb_ca": blk, "local_latents": other}


def _fake_denoiser(batch, mode="full"):
    """A deterministic, t- and x-dependent stand-in for the denoiser.

    Deliberately NOT a constant: a constant field would make almost any integration bug
    invisible, because every ordering of the same additions gives the same answer.
    """
    x = batch["x_t"]
    out = {}
    for dm, v in x.items():
        t = batch["t"][dm].reshape(-1, 1, 1)
        out[dm] = {"v": torch.tanh(v * 1.3 + t) * 0.7 - 0.2 * v}
    return out


def _setup(n=7, mode="sc", nsteps=6, seed=0):
    fm = _fm()
    mask = torch.ones(1, n, dtype=torch.bool)
    batch = {"mask": mask}
    torch.manual_seed(seed)
    x = {
        "bb_ca": torch.randn(1, n, 3),
        "local_latents": torch.randn(1, n, LAT_DIM),
    }
    return fm, mask, batch, x, _sampling_args(mode), nsteps


# ------------------------------------------------------------------ the equivalence check
@pytest.mark.parametrize("sampler_mode", ["vf", "sc"])
def test_rollout_matches_partial_simulation(sampler_mode):
    """DiffRollout's loop must reproduce `partial_simulation` exactly, for both integrators."""
    fm, mask, batch, x0, samp, nsteps = _setup(mode="sc")
    t_start = {"bb_ca": 0.0, "local_latents": 0.0}
    noise_seed = 1234

    roll = DiffRollout(
        fm=fm, predict_for_sampling=_fake_denoiser, batch=batch, mask=mask, nsteps=nsteps,
        t_start=t_start, sampling_model_args=samp, sampler_mode=sampler_mode,
        noise_seed=noise_seed, self_cond=True, grad_checkpoint=False,
    )
    got, _ = roll({k: v.clone() for k, v in x0.items()}, grad=False)

    ts, gt = build_schedule(fm, nsteps, samp, t_start)
    sim_params = resolve_sim_params(fm, samp, sampler_mode)
    # The rollout reseeds internally in `sc` mode; match that so the frozen noise agrees.
    if sampler_mode == "sc":
        torch.manual_seed(noise_seed)
    with torch.no_grad():
        want, _ = fm.partial_simulation(
            batch=dict(batch), x={k: v.clone() for k, v in x0.items()}, x_1_pred=None,
            mask=mask, predict_for_sampling=_fake_denoiser, start_step=0, end_step=nsteps,
            self_cond=True, ts=ts, gt=gt, simulation_step_params=sim_params,
            device=mask.device, guidance_w=1.0, ag_ratio=0.0, grad_enabled=False,
        )
    for dm in fm.data_modes:
        assert torch.equal(got[dm], want[dm]), f"{sampler_mode}/{dm} diverged from partial_simulation"


def test_sc_mode_is_deterministic_and_vf_needs_no_seed():
    """Frozen noise: two `sc` rollouts of the same source agree; and `vf` draws nothing."""
    fm, mask, batch, x0, samp, nsteps = _setup(mode="sc")
    t_start = {"bb_ca": 0.0, "local_latents": 0.0}
    for sampler_mode in ("sc", "vf"):
        roll = DiffRollout(
            fm=fm, predict_for_sampling=_fake_denoiser, batch=batch, mask=mask, nsteps=nsteps,
            t_start=t_start, sampling_model_args=samp, sampler_mode=sampler_mode,
            noise_seed=7, self_cond=True, grad_checkpoint=False,
        )
        a, _ = roll({k: v.clone() for k, v in x0.items()}, grad=False)
        torch.manual_seed(999)  # perturb the global RNG between the two calls
        b, _ = roll({k: v.clone() for k, v in x0.items()}, grad=False)
        for dm in fm.data_modes:
            assert torch.equal(a[dm], b[dm]), f"{sampler_mode} is not deterministic in {dm}"


def test_gradient_flows_to_source_and_checkpointing_agrees():
    """A finite gradient reaches the source state, and checkpointing does not change it."""
    fm, mask, batch, x0, samp, nsteps = _setup(mode="sc", nsteps=5)
    t_start = {"bb_ca": 0.0, "local_latents": 0.0}
    grads = {}
    for ckpt in (False, True):
        roll = DiffRollout(
            fm=fm, predict_for_sampling=_fake_denoiser, batch=batch, mask=mask, nsteps=nsteps,
            t_start=t_start, sampling_model_args=samp, sampler_mode="vf", noise_seed=0,
            self_cond=True, grad_checkpoint=ckpt,
        )
        z = {k: v.clone().requires_grad_(True) for k, v in x0.items()}
        out, _ = roll(z, grad=True)
        loss = sum((v ** 2).sum() for v in out.values())
        loss.backward()
        grads[ckpt] = {k: v.grad.clone() for k, v in z.items()}
        for dm, g in grads[ckpt].items():
            assert torch.isfinite(g).all(), f"non-finite gradient in {dm} (checkpoint={ckpt})"
            assert g.abs().max() > 0, f"identically zero gradient in {dm} (checkpoint={ckpt})"
    for dm in fm.data_modes:
        torch.testing.assert_close(grads[False][dm], grads[True][dm], rtol=1e-5, atol=1e-6)


def test_source_mode_ignores_x1_but_sdedit_requires_it():
    fm, mask, batch, x0, samp, nsteps = _setup()
    common = dict(fm=fm, predict_for_sampling=_fake_denoiser, batch=batch, mask=mask,
                  nsteps=nsteps, sampling_model_args=samp, sampler_mode="vf")
    r0 = DiffRollout(t_start={"bb_ca": 0.0, "local_latents": 0.0}, **common)
    z = r0.init_source(seed=3)
    assert set(z) == set(fm.data_modes)
    r1 = DiffRollout(t_start={"bb_ca": 0.6, "local_latents": 1.0}, **common)
    with pytest.raises(ValueError, match="needs the encoded input"):
        r1.init_source(seed=3)
    z1 = r1.init_source(x_1=x0, seed=3)
    # t_lat is clamped just inside 1, so the latent track starts essentially at the input.
    assert torch.allclose(z1["local_latents"], x0["local_latents"], atol=2e-2)


def test_schedule_matches_sdedit_builder():
    """`build_schedule` must agree with the SDEdit script's own, or the arms walk different
    time grids and stop being comparable."""
    from sdedit_cyclize import build_edit_schedule

    fm, mask, batch, x0, samp, nsteps = _setup(nsteps=9)
    for t_start in ({"bb_ca": 0.0, "local_latents": 0.0},
                    {"bb_ca": 0.4, "local_latents": 1.0}):
        a_ts, a_gt = build_schedule(fm, nsteps, samp, t_start)
        b_ts, b_gt = build_edit_schedule(fm, nsteps, samp, t_start)
        for dm in fm.data_modes:
            assert torch.equal(a_ts[dm], b_ts[dm]), f"ts mismatch for {dm} at {t_start}"
            assert torch.equal(a_gt[dm], b_gt[dm]), f"gt mismatch for {dm} at {t_start}"


# --------------------------------------------------------------------------- the objective
def test_parse_weights_refuses_a_vanishing_only_objective():
    """`bond_fb` alone is identically zero on the abstaining population; that must not be a
    configuration the driver will accept."""
    with pytest.raises(SystemExit, match="only terms that can vanish"):
        parse_weights("bond_fb")
    with pytest.raises(SystemExit, match="only terms that can vanish"):
        parse_weights("bond_fb:5,clash,prior")
    ok = parse_weights("bond_fb,anchor_ce:3")
    assert ok == {"bond_fb": 1.0, "anchor_ce": 3.0}
    assert parse_weights("")  # defaults
    with pytest.raises(SystemExit, match="unknown loss term"):
        parse_weights("nope")


def test_objective_tag_is_stable_and_drops_zeros():
    a = objective_tag({"anchor_ce": 3.0, "seq_ce": 1.0, "clash": 0.0})
    b = objective_tag({"seq_ce": 1.0, "anchor_ce": 3.0})
    assert a == b == "anchor_ce3.0-seq_ce"


def test_anchor_logprob_is_vacuous_for_mainchain_and_orientation_free_for_isopeptide():
    torch.manual_seed(0)
    logp = torch.log_softmax(torch.randn(6, 20), dim=-1)
    assert float(anchor_logprob(logp, MAINCHAIN, 0, 5, True)) == 0.0

    # A confident CYS/CYS pair should score far higher than a random one.
    strong = torch.full((6, 20), -20.0)
    strong[0, AA_CYS] = 0.0
    strong[5, AA_CYS] = 0.0
    strong = torch.log_softmax(strong, dim=-1)
    assert float(anchor_logprob(strong, DISULFIDE, 0, 5, True)) > float(
        anchor_logprob(logp, DISULFIDE, 0, 5, True)
    )

    # Isopeptide: LYS at either end must score the same, since either orientation satisfies
    # the head and the loss must not fight a sample that chose the other one.
    a = torch.full((6, 20), -20.0)
    a[0, AA_LYS] = 0.0
    a[5, AA_ASP] = 0.0
    b = torch.full((6, 20), -20.0)
    b[0, AA_ASP] = 0.0
    b[5, AA_LYS] = 0.0
    va = float(anchor_logprob(torch.log_softmax(a, dim=-1), ISOPEPTIDE, 0, 5, True))
    vb = float(anchor_logprob(torch.log_softmax(b, dim=-1), ISOPEPTIDE, 0, 5, True))
    assert abs(va - vb) < 1e-5
    # GLU is an acceptor too (CD rather than CG).
    c = torch.full((6, 20), -20.0)
    c[0, AA_LYS] = 0.0
    c[5, AA_GLU] = 0.0
    assert float(anchor_logprob(torch.log_softmax(c, dim=-1), ISOPEPTIDE, 0, 5, True)) > -1.0


# --------------------------------------------------------------------------- acceptance
def _ideal_peptide(L=8, seed=0):
    """A crude but valid peptide: atom37 with N/CA/C/O/CB filled, spaced so nothing clashes."""
    torch.manual_seed(seed)
    coors = torch.zeros(L, 37, 3)
    for k in range(L):
        base = torch.tensor([0.38 * k, 0.0, 0.0])
        coors[k, 0] = base + torch.tensor([-0.12, 0.05, 0.0])   # N
        coors[k, 1] = base                                       # CA
        coors[k, 2] = base + torch.tensor([0.12, 0.05, 0.0])     # C
        coors[k, 3] = base + torch.tensor([0.15, 0.17, 0.0])     # CB (slot 3 in atom37)
        coors[k, 4] = base + torch.tensor([0.20, -0.05, 0.0])    # O
    amask = torch.zeros(L, 37, dtype=torch.bool)
    amask[:, :5] = True
    aatype = torch.zeros(L, dtype=torch.long)
    return coors, amask, aatype


def _receptor(L=6):
    coors = torch.zeros(L, 37, 3)
    for k in range(L):
        coors[k, 1] = torch.tensor([0.38 * k, 0.40, 0.0])  # 4 A away: in contact, no clash
    amask = torch.zeros(L, 37, dtype=torch.bool)
    amask[:, 1] = True
    return coors, amask


def _accept_kwargs(pep, pep_amask, pep_aa, ref, ref_amask, ref_aa, cyc_type=MAINCHAIN):
    tgt, tgt_amask = _receptor()
    return dict(
        pep_coors_nm=pep, pep_amask=pep_amask, pep_aatype=pep_aa,
        ref_coors_nm=ref, ref_amask=ref_amask, ref_aatype=ref_aa,
        tgt_coors_nm=tgt, tgt_amask=tgt_amask, cyc_type=cyc_type,
    )


def test_accept_identical_copy_passes_everything_but_the_ring():
    coors, amask, aatype = _ideal_peptide()
    out = accept(**_accept_kwargs(coors, amask, aatype, coors.clone(), amask, aatype.clone()))
    assert out["seq_identity"] == 1.0
    assert out["n_substitutions"] == 0
    assert out["ca_rmsd_A"] == pytest.approx(0.0, abs=1e-5)
    assert out["contact_retention"] == 1.0
    assert out["gate_contacts"] and out["gate_ca_rmsd"] and out["gate_iface_ca_rmsd"]
    assert out["gate_clash"]
    # An extended chain is not a ring, so the joint gate must still fail -- and fail on the
    # ring specifically, which is what makes this a useful positive control.
    assert not out["gate_ring_closed"]
    assert out["gates_failed"] == ["gate_ring_closed"]
    assert not out["gate_joint_structural"]


def test_accept_catches_pose_drift_sequence_and_protection():
    coors, amask, aatype = _ideal_peptide()
    moved = coors.clone()
    moved[:, :, 0] += 0.30  # 3 A along x: past the 2 A whole-peptide gate
    out = accept(**_accept_kwargs(moved, amask, aatype, coors, amask, aatype.clone()))
    assert out["ca_rmsd_A"] == pytest.approx(3.0, abs=1e-3)
    assert not out["gate_ca_rmsd"]
    assert not out["gate_iface_ca_rmsd"]

    mutated = aatype.clone()
    mutated[2] = AA_CYS
    mutated[3] = AA_LYS
    mutated[4] = AA_ASP
    out = accept(**_accept_kwargs(coors, amask, mutated, coors, amask, aatype.clone()))
    assert out["n_substitutions"] == 3
    assert not out["gate_substitutions"]          # budget is 2
    assert out["seq_identity"] == pytest.approx(5 / 8)
    assert not out["gate_identity"]               # 62.5% < 80%
    assert out["budget_3"] and not out["budget_2"]
    assert out["identity_ge_70"] is False

    out = accept(**_accept_kwargs(coors, amask, mutated, coors, amask, aatype.clone()),
                 protect=[2, 7])
    assert out["protected_violations"] == [2]
    assert not out["gate_protected"]


def test_accept_rejects_a_clash_and_refuses_a_length_change():
    coors, amask, aatype = _ideal_peptide()
    clashing = coors.clone()
    clashing[0, :5] = coors[4, :5]  # fold residue 0 onto residue 4
    out = accept(**_accept_kwargs(clashing, amask, aatype, coors, amask, aatype.clone()))
    assert out["n_clashes_intra"] > 0
    assert not out["gate_clash"]

    with pytest.raises(ValueError, match="keeps peptide length unchanged"):
        accept(**_accept_kwargs(coors[:6], amask[:6], aatype[:6], coors, amask, aatype.clone()))


def test_accept_closed_disulfide_passes_the_ring_gate():
    coors, amask, aatype = _ideal_peptide()
    aa = aatype.clone()
    aa[0] = AA_CYS
    aa[-1] = AA_CYS
    am = amask.clone()
    sg = 5  # any free slot; SG_IDX is looked up from residue_constants below
    from proteinfoundation.eval.cyclic_reconstruction_metrics import SG_IDX

    c = coors.clone()
    am[0, SG_IDX] = True
    am[-1, SG_IDX] = True
    c[0, SG_IDX] = torch.tensor([0.0, 0.25, 0.0])
    c[-1, SG_IDX] = torch.tensor([0.0, 0.25, 0.205])  # 2.05 A apart: inside (1.8, 2.3)
    out = accept(**_accept_kwargs(c, am, aa, coors, amask, aatype.clone(), cyc_type=DISULFIDE))
    assert out["bond_atoms_valid"]
    assert out["bond_dist_A"] == pytest.approx(2.05, abs=1e-2)
    assert out["gate_ring_closed"]
    assert out["n_substitutions"] == 2 and out["gate_substitutions"]
    del sg


def test_abstention_is_a_failed_ring_not_a_missing_measurement():
    """No CYS at the endpoints means the disulfide FAILED, never that it was unscored."""
    coors, amask, aatype = _ideal_peptide()
    out = accept(**_accept_kwargs(coors, amask, aatype, coors, amask, aatype.clone(),
                                  cyc_type=DISULFIDE))
    assert out["bond_atoms_valid"] is False
    assert out["bond_dist_A"] is None
    assert out["gate_ring_closed"] is False
    assert "gate_ring_closed" in out["gates_failed"]


def test_sequence_report_tiers():
    ref = torch.zeros(10, dtype=torch.long)
    cand = ref.clone()
    cand[0] = 1
    r = sequence_report(cand, ref)
    assert r["seq_identity"] == 0.9
    assert r["identity_ge_90"] and r["identity_ge_80"] and r["identity_ge_70"]
    assert r["budget_1"] and not r["budget_0"]
    assert r["gate_identity"] and r["gate_substitutions"]


def test_identity_is_exact_at_the_threshold_boundaries():
    """A float32 mean puts 9/10 at 0.8999999 and 8/10 at 0.7999999, which fails both the
    90% tier and the contract's own 80% gate by rounding alone. The ratio must be exact."""
    for L, n_sub, want in ((10, 1, 0.9), (10, 2, 0.8), (5, 1, 0.8), (20, 2, 0.9)):
        ref = torch.zeros(L, dtype=torch.long)
        cand = ref.clone()
        cand[:n_sub] = 1
        r = sequence_report(cand, ref)
        assert r["seq_identity"] == want, f"L={L} n_sub={n_sub}"
        assert r["gate_identity"], f"L={L} n_sub={n_sub} must pass the 80% gate"
        assert r["identity_ge_80"]
        assert r["identity_ge_90"] == (want >= 0.90)


def test_pareto_front_drops_unscored_rows():
    rows = [
        {"a": 1.0, "b": 1.0},   # dominated by index 2
        {"a": 0.5, "b": 2.0},   # nondominated (best a)
        {"a": 0.9, "b": 0.5},   # nondominated (best b)
        {"a": None, "b": 0.1},  # unscored -> dropped, not defaulted to 0
        {"b": 0.0},             # missing key -> dropped
    ]
    front = pareto_front(rows, [("a", "min"), ("b", "min")])
    assert front == [1, 2]


# ---------------------------------------------------------------------------- ring declaration
def test_ring_atom_names_follow_the_metric():
    assert ring_atom_names(MAINCHAIN, 0, 0) == ("N", "C")
    assert ring_atom_names(DISULFIDE, AA_CYS, AA_CYS) == ("SG", "SG")
    assert ring_atom_names(ISOPEPTIDE, AA_LYS, AA_ASP) == ("NZ", "CG")
    assert ring_atom_names(ISOPEPTIDE, AA_LYS, AA_GLU) == ("NZ", "CD")
    assert ring_atom_names(ISOPEPTIDE, AA_ASP, AA_LYS) == ("CG", "NZ")
    with pytest.raises(RingUndeclarable, match="disulfide needs CYS"):
        ring_atom_names(DISULFIDE, AA_CYS, AA_LYS)
    with pytest.raises(RingUndeclarable, match="isopeptide needs LYS"):
        ring_atom_names(ISOPEPTIDE, AA_CYS, AA_CYS)


_PDB = """\
ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  CYS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   CYS A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  SG  CYS A   1       2.000   0.000   2.000  1.00  0.00           S
ATOM      5  N   CYS A   2       5.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  CYS A   2       6.458   0.000   0.000  1.00  0.00           C
ATOM      7  C   CYS A   2       7.009   1.420   0.000  1.00  0.00           C
ATOM      8  SG  CYS A   2       2.000   0.000   4.050  1.00  0.00           S
END
"""


def test_declare_and_verify_ring_roundtrip(tmp_path):
    p = tmp_path / "pep.pdb"
    p.write_text(_PDB)
    # Before declaring: geometry is in window but nothing is declared, and the gate must
    # report that rather than passing on geometry alone.
    pre = verify_ring(p, 1, 2, DISULFIDE)
    assert pre["dist_A"] == pytest.approx(2.05, abs=1e-3)
    assert pre["in_window"] and not pre["declared"]
    assert pre["reason"] == "no CONECT links the two anchor atoms"

    info = declare_ring(p, 1, 2, DISULFIDE, AA_CYS, AA_CYS)
    assert (info["serial_i"], info["serial_j"]) == (4, 8)
    post = verify_ring(p, 1, 2, DISULFIDE)
    assert post["declared"] and post["in_window"]
    assert post["atom_i"] == post["atom_j"] == "SG"
    assert p.read_text().rstrip().endswith("END")


def test_verify_ring_reports_an_out_of_window_declared_bond(tmp_path):
    p = tmp_path / "pep.pdb"
    p.write_text(_PDB.replace("   4.050", "   9.000"))
    declare_ring(p, 1, 2, DISULFIDE, AA_CYS, AA_CYS)
    v = verify_ring(p, 1, 2, DISULFIDE)
    # SG(1) is at z=2.000 and SG(2) now at z=9.000, so the declared bond spans 7 A -- well
    # outside (1.8, 2.3). A declared-but-open ring must not read as closed.
    assert v["declared"] and not v["in_window"]
    assert v["dist_A"] == pytest.approx(7.0, abs=1e-3)


def test_verify_ring_on_a_sequence_that_cannot_carry_the_request(tmp_path):
    p = tmp_path / "pep.pdb"
    p.write_text(_PDB)
    v = verify_ring(p, 1, 2, ISOPEPTIDE)
    assert not v["declared"]
    assert "isopeptide needs LYS" in v["reason"]


def test_contract_thresholds_are_the_documented_ones():
    """A silent threshold change would make the arms incomparable; pin them."""
    assert CONTRACT["min_identity"] == 0.80
    assert CONTRACT["max_substitutions"] == 2
    assert CONTRACT["contact_cutoff_A"] == 4.5
    assert CONTRACT["min_contact_retention"] == 0.80
    assert CONTRACT["max_ca_rmsd_A"] == 2.0
    assert CONTRACT["max_interface_ca_rmsd_A"] == 1.0
    assert CONTRACT["max_ddg_REU"] == 5.0
    assert CONTRACT["strict_ddg_REU"] == 0.0
    assert NM_TO_ANG == 10.0
    assert CA_IDX == 1
