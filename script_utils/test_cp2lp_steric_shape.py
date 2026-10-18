"""CPU tests for the steric penalty's SHAPE knob and the widened gradient probe.

The release arms ended at `steric_min_inter_nm` 0.0705 (cold) / 0.206 (warm) against real bound
LPs' 0.262 -- inside a covalent bond length -- with `steric_weight` already the largest active
weight and its gradient norm at 0.78 of a total 2.42. So the thing being changed here is the
penalty's shape, and these tests target the three ways that change could be wrong without
erroring:

* it silently alters the DEFAULT, which would retroactively change every existing arm and
  control and make the release-vs-torsion comparison a different experiment;
* the new form is weaker than the old one somewhere, trading a deep-overlap fix for a
  shallow-overlap regression right where most pairs sit;
* the `*_violating_*` metrics move with the knob, which would make incidence incomparable
  across a sweep and hide whether the fix worked.

Run: python script_utils/test_cp2lp_steric_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cp2lp.bb_losses import (  # noqa: E402
    DEFAULT_STERIC_DEPTH_GAIN,
    _overlap_penalty,
    backbone_steric_loss,
)

PASS: list[str] = []
FAIL: list[str] = []
THR = 0.20


def check(name: str, cond: bool, extra: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {extra}" if extra else ""))


def test_default_is_the_historical_hinge() -> None:
    print("the default is bit-identical to relu(threshold - d)")
    # THE WHOLE COMPARABILITY OF THE PROJECT RESTS ON THIS. Every bb_* arm, the torsion
    # control and both release arms were trained with the linear hinge; a default that is
    # merely *close* would make their loss curves silently incomparable with anything new.
    check("DEFAULT_STERIC_DEPTH_GAIN is 0.0", DEFAULT_STERIC_DEPTH_GAIN == 0.0,
          str(DEFAULT_STERIC_DEPTH_GAIN))
    d = torch.linspace(-0.05, 0.40, 4001)
    check(
        "gain=0 equals relu exactly (bitwise)",
        torch.equal(_overlap_penalty(d, THR, 0.0), F.relu(THR - d)),
    )
    check(
        "the default ARGUMENT is the historical hinge",
        torch.equal(_overlap_penalty(d, THR, DEFAULT_STERIC_DEPTH_GAIN), F.relu(THR - d)),
    )


def test_never_weaker_than_linear() -> None:
    print("the penalty is monotonically >= the historical one, everywhere")
    # This is the property a bare exponent does NOT have: threshold*(relu/threshold)**2 is
    # weaker than linear for every overlap below half the threshold. If this check ever fails,
    # the form has been swapped for one that regresses at shallow overlap.
    d = torch.linspace(-0.05, 0.40, 4001)
    lin = F.relu(THR - d)
    for gain in (0.0, 0.5, 1.0, 2.0, 5.0):
        v = _overlap_penalty(d, THR, gain)
        check(f"gain={gain} is >= linear everywhere",
              bool((v >= lin - 1e-12).all()), f"min delta {float((v - lin).min()):+.3e}")
    check("a positive gain is STRICTLY greater somewhere inside the threshold",
          float((_overlap_penalty(d, THR, 2.0) - lin).max()) > 1e-6)


def test_force_rises_with_depth_but_not_at_the_boundary() -> None:
    print("the restoring force grows with overlap depth")
    # The point of the knob. Numbers are the arms' MEASURED operating points, so a future
    # change that quietly flattens the profile fails here rather than in a GPU run.
    def force(d_val: float, gain: float) -> float:
        t = torch.tensor([d_val], requires_grad=True)
        _overlap_penalty(t, THR, gain).backward()
        return abs(float(t.grad))

    deep, half, edge = 0.0705, 0.10, 0.199  # cold arm, half overlap, just inside
    check("gain=0 force is depth-INDEPENDENT (the defect)",
          abs(force(deep, 0.0) - force(edge, 0.0)) < 1e-6,
          f"{force(deep, 0.0):.3f} vs {force(edge, 0.0):.3f}")
    check("gain=2 multiplies the force at the cold arm's depth by >3x",
          force(deep, 2.0) / force(deep, 0.0) > 3.0, f"{force(deep, 2.0):.3f}")
    check("gain=2 leaves the boundary force essentially unchanged",
          abs(force(edge, 2.0) / force(edge, 0.0) - 1.0) < 0.05, f"{force(edge, 2.0):.3f}")
    check("force is monotone in depth at gain=2",
          force(deep, 2.0) > force(half, 2.0) > force(edge, 2.0))
    check("no force outside the threshold, for any gain",
          all(force(0.25, g) == 0.0 for g in (0.0, 1.0, 2.0)))


def test_finite_values_and_gradients() -> None:
    print("values and gradients are finite, including exactly at the boundary")
    # An inverse-power soft core would be stronger here and would also reintroduce the NaN
    # class this project has already paid for once, so the polynomial form is pinned by a test.
    for gain in (0.0, 1.0, 2.0, 5.0):
        d = torch.linspace(-0.2, 0.6, 8001, requires_grad=True)
        v = _overlap_penalty(d, THR, gain)
        v.sum().backward()
        ok = bool(torch.isfinite(v).all() and torch.isfinite(d.grad).all())
        check(f"gain={gain}: all finite", ok)
    d = torch.tensor([THR], requires_grad=True)  # exactly on the threshold
    _overlap_penalty(d, THR, 2.0).backward()
    check("finite gradient exactly at d == threshold", bool(torch.isfinite(d.grad).all()))


def _pair_fixture(d_nm: float):
    """One peptide residue and one receptor atom a chosen distance apart."""
    bb = torch.zeros(1, 2, 4, 3)
    bb[0, :, :, 0] = torch.tensor([0.0, 1.0])[:, None]  # keep the two residues far apart
    mask = torch.ones(1, 2, dtype=torch.bool)
    tgt = torch.zeros(1, 1, 37, 3)
    tgt[0, 0, 1, 0] = d_nm  # receptor CA at d_nm from the first peptide atom
    tgt_am = torch.zeros(1, 1, 37, dtype=torch.bool)
    tgt_am[0, 0, 1] = True
    tmask = torch.ones(1, 1, dtype=torch.bool)
    return bb, mask, tgt, tgt_am, tmask


def test_violating_metrics_do_not_move_with_the_knob() -> None:
    print("incidence metrics threshold the raw gap, not the penalty")
    # Severity sums MUST move with the knob -- that is the knob. Incidence must NOT, or a
    # sweep cannot answer "did fewer pairs violate" because the yardstick moved too.
    bb, mask, tgt, tgt_am, tmask = _pair_fixture(0.10)  # a real violation at half overlap
    ref = None
    sev = []
    for gain in (0.0, 2.0, 5.0):
        loss, met = backbone_steric_loss(
            bb, mask, tgt, tgt_am, tmask, threshold_nm=THR, depth_gain=gain
        )
        sev.append(float(loss))
        cnt = float(met["steric_inter_violating_count"])
        dmin = float(met["steric_min_inter_nm"])
        if ref is None:
            ref = (cnt, dmin)
        check(f"gain={gain}: violating_count unchanged", abs(cnt - ref[0]) < 1e-9, f"{cnt}")
        check(f"gain={gain}: min_inter_nm unchanged", abs(dmin - ref[1]) < 1e-9, f"{dmin:.4f}")
    check("severity DOES rise with the gain", sev[0] < sev[1] < sev[2],
          " < ".join(f"{v:.5f}" for v in sev))


def test_no_violation_means_no_change_at_all() -> None:
    print("a non-overlapping configuration is untouched by the knob")
    # Real bound LPs sit at 0.262, outside the 0.20 threshold. The knob must be a no-op there,
    # otherwise it is not a clash fix, it is a pose change applied to everything.
    bb, mask, tgt, tgt_am, tmask = _pair_fixture(0.30)
    vals = [
        float(backbone_steric_loss(
            bb, mask, tgt, tgt_am, tmask, threshold_nm=THR, depth_gain=g
        )[0])
        for g in (0.0, 2.0, 5.0)
    ]
    check("every gain gives the same (zero) loss outside the threshold",
          all(abs(v - vals[0]) < 1e-12 for v in vals) and vals[0] == 0.0, str(vals))


def test_module_reads_the_knob_and_defaults_safely() -> None:
    print("the module plumbs the knob and defaults to the historical hinge")
    src = (
        Path(__file__).resolve().parents[1] / "src/proteinfoundation/cp2lp/module.py"
    ).read_text()
    check("config key is read", 'lw.get("steric_depth_gain", DEFAULT_STERIC_DEPTH_GAIN)' in src)
    # BOTH call sites, or the probe measures a different term than the objective uses -- which
    # is exactly the kind of mismatch that makes a gradient table quietly wrong.
    check("passed at both backbone_steric_loss call sites",
          src.count("depth_gain=self.bb_steric_depth_gain") == 2,
          str(src.count("depth_gain=self.bb_steric_depth_gain")))
    check("the probe and the objective use one attribute", "self.bb_steric_depth_gain" in src)


def test_grad_probe_covers_the_opposing_terms() -> None:
    print("the gradient probe reaches anchor and adversarial")
    src = (
        Path(__file__).resolve().parents[1] / "src/proteinfoundation/cp2lp/module.py"
    ).read_text()
    # steric's 0.78 only means something next to what opposes it. Until this change nothing
    # logged the opposing terms, so the tug-of-war was inferred rather than measured.
    check("anchor is stashed for the probe", 'self._probe_extra_terms["anchor"]' in src)
    check("adversarial is stashed for the probe",
          'self._probe_extra_terms["adversarial"]' in src)
    check("a shared helper measures all terms at the same params",
          "_grad_norms_at_head" in src and src.count("_grad_norms_at_head") >= 3,
          str(src.count("_grad_norms_at_head")))
    # A stale tensor from a previous step would be differentiated against a freed graph, which
    # crashes on the probe cadence -- hours in, not at smoke time.
    check("the stash is reset at the start of the step",
          "self._probe_extra_terms: dict[str, tuple[torch.Tensor, float]] = {}" in src)
    check("the stash is cleared after probing", src.count("self._probe_extra_terms = {}") >= 1)
    probe_block = src.split("# The opposing half of the gradient budget")[1].split(
        "if g_loss.requires_grad:"
    )[0]
    check("probing happens BEFORE the real backward", "manual_backward" not in probe_block)
    check("probing is gated on the same cadence", "self.bb_grad_probe_every" in probe_block)


def test_helper_weights_and_skips() -> None:
    print("the shared helper applies the weight and skips grad-less terms")
    import torch.nn as nn

    class Fake:
        _backbone_head_parameters = staticmethod(lambda: [Fake.p])
        p = nn.Parameter(torch.ones(3))
        _grad_norms_at_head = None

    from proteinfoundation.cp2lp.module import CP2LPGenerator

    f = Fake()
    f._grad_norms_at_head = CP2LPGenerator._grad_norms_at_head.__get__(f)
    term = (Fake.p * 2.0).sum()
    one = f._grad_norms_at_head({"t": (term, 1.0)})["gradnorm/t"]
    two = f._grad_norms_at_head({"t": (term, 3.0)})["gradnorm/t"]
    check("the weight scales the reported norm", abs(float(two) - 3.0 * float(one)) < 1e-5,
          f"{float(one):.4f} -> {float(two):.4f}")
    det = f._grad_norms_at_head({"t": (torch.tensor(1.0), 1.0)})["gradnorm/t"]
    check("a term without grad reports zero rather than raising", float(det) == 0.0)
    none = f._grad_norms_at_head({"t": (None, 1.0)})["gradnorm/t"]
    check("a None term reports zero rather than raising", float(none) == 0.0)


def test_terminal_exemption_is_mask_aware() -> None:
    print("the terminal exemption frees the REAL termini, never padding")
    from proteinfoundation.cp2lp.bb_losses import (  # noqa: E402
        DEFAULT_PRESERVE_TERMINAL_EXEMPT_K,
        source_interface_weights,
    )
    check("default exempt_k is 0 (historical)", DEFAULT_PRESERVE_TERMINAL_EXEMPT_K == 0)

    # A RAGGED batch is the whole point. Peptides here are 5-16 residues in a padded tensor, so
    # taking "last k" from the tensor's tail would exempt PADDING and leave the real C-terminus
    # fully charged -- the silent-no-op version of this feature, which would read as "the
    # exemption did nothing" rather than as a bug.
    L = [6, 10, 5]
    n, b = 12, 3
    pm = torch.zeros(b, n, dtype=torch.bool)
    for i, l in enumerate(L):
        pm[i, :l] = True
    src = torch.zeros(b, n, 37, 3)
    sam = torch.zeros(b, n, 37, dtype=torch.bool); sam[:, :, :5] = True
    tgt = torch.zeros(b, 4, 37, 3); tgt[..., 1] = 0.3   # receptor near everything
    tam = torch.zeros(b, 4, 37, dtype=torch.bool); tam[:, :, 1] = True
    tm = torch.ones(b, 4, dtype=torch.bool)

    w0, _ = source_interface_weights(src, sam, pm, tgt, tam, tm)
    wk0, m0 = source_interface_weights(src, sam, pm, tgt, tam, tm, terminal_exempt_k=0)
    check("k=0 is bit-identical to the historical default", torch.equal(w0, wk0))
    check("k=0 reports zero exempt fraction", float(m0["preserve_exempt_frac"]) == 0.0)

    for k in (1, 2):
        w, met = source_interface_weights(src, sam, pm, tgt, tam, tm, terminal_exempt_k=k)
        for i, l in enumerate(L):
            zero = [j for j, v in enumerate(w[i, :l].tolist()) if v == 0.0]
            want = sorted(set(list(range(min(k, l))) + list(range(max(0, l - k), l))))
            check(f"k={k} len={l}: exempts {want}", zero == want, f"got {zero}")
            check(f"k={k} len={l}: padding unweighted",
                  float(w[i, l:].abs().sum()) == 0.0)
        check(f"k={k} reports a non-zero exempt fraction",
              float(met["preserve_exempt_frac"]) > 0.0,
              f"{float(met['preserve_exempt_frac']):.3f}")


def test_exempt_rungs_are_single_variable() -> None:
    print("the exempt rungs differ from their control in the exemption alone")
    import os
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def load(n):
        with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
            return OmegaConf.to_container(compose(config_name=n), resolve=True)
    ctl = load("example/training_cp2lp_bb_retain_w025")["cp2lp"]
    for k in (1, 2):
        c = load(f"example/training_cp2lp_bb_retain_w025_x{k}")["cp2lp"]
        check(f"x{k} exempt_k is {k}", int(c["losses"]["preserve_terminal_exempt_k"]) == k)
        # The exemption only means anything at a MATCHED weight -- otherwise the rung differs in
        # two things and the comparison cannot attribute the effect to the exemption.
        check(f"x{k} shares the control's preserve_weight",
              c["losses"]["preserve_weight"] == ctl["losses"]["preserve_weight"],
              f"{c['losses']['preserve_weight']} vs {ctl['losses']['preserve_weight']}")
        check(f"x{k} shares the control's budget and milestone",
              c["max_g_updates"] == ctl["max_g_updates"]
              and c["eval_checkpoint_g_updates"] == ctl["eval_checkpoint_g_updates"])
        check(f"x{k} leaves the steric shape knob alone",
              float(c["losses"].get("steric_depth_gain", 0.0)) == 0.0)
    check("the control itself has NO exemption (k=0)",
          int(ctl["losses"].get("preserve_terminal_exempt_k", 0)) == 0)


def main() -> int:
    for fn in (
        test_default_is_the_historical_hinge,
        test_never_weaker_than_linear,
        test_force_rises_with_depth_but_not_at_the_boundary,
        test_finite_values_and_gradients,
        test_violating_metrics_do_not_move_with_the_knob,
        test_no_violation_means_no_change_at_all,
        test_module_reads_the_knob_and_defaults_safely,
        test_grad_probe_covers_the_opposing_terms,
        test_helper_weights_and_skips,
        test_terminal_exemption_is_mask_aware,
        test_exempt_rungs_are_single_variable,
    ):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
