"""Unit tests for script_utils/kinematic_ceiling.py.

Run:  .venv/bin/python -m pytest scripts/test_kinematic_ceiling.py -q

The analytic reach bound is the instrument the whole Milestone 1.5 ceiling rests on, so it
is tested against (a) closed-form values computable by hand, (b) a brute-force random
sampler over real backbone torsions, and (c) the NeRF rebuild, which must reproduce a
native structure exactly when no torsion is changed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from script_utils.kinematic_ceiling import (  # noqa: E402
    CA_CA_A, C_N_A, TAU_MAX_DEG, ClosureSpec, annulus_distance_range, backbone_torsions,
    ca_contact_set, dihedral, feasible_analytic, feasible_torsion, mainchain_spec,
    place_atom, reach_interval, rebuild_from_window, retention_ceiling, rise_per_link,
    scan_windows,
)


# ----------------------------------------------------------------------------- reach
def test_rise_per_link_matches_hand_value():
    # 3.80 * sin(75 deg) = 3.6706...  A 13-mer (12 links) then spans ~44 A, which is the
    # "fully extended 13-mer is about 45 A" figure the inventory quoted.
    assert rise_per_link(150.0) == pytest.approx(3.6706, abs=1e-3)
    assert 12 * rise_per_link(150.0) == pytest.approx(44.05, abs=0.05)


def test_reach_interval_small_n_is_exact():
    assert reach_interval(0) == (0.0, 0.0)
    assert reach_interval(1) == (CA_CA_A, CA_CA_A)
    lo, hi = reach_interval(2)
    assert lo == pytest.approx(2 * CA_CA_A * math.sin(math.radians(78.0) / 2), abs=1e-6)
    assert hi == pytest.approx(2 * CA_CA_A * math.sin(math.radians(150.0) / 2), abs=1e-6)


def test_reach_interval_is_monotone_and_permissive():
    prev = 0.0
    for n in range(1, 20):
        lo, hi = reach_interval(n)
        assert hi >= prev
        assert lo <= hi
        prev = hi
    assert reach_interval(5)[0] == 0.0  # permissive by design for n >= 3


def test_max_reach_never_exceeded_by_a_real_chain():
    """Brute force: no random CA chain with tau <= tau_max ever out-reaches the bound."""
    rng = np.random.default_rng(0)
    for n in (3, 6, 10):
        bound = reach_interval(n)[1]
        worst = 0.0
        for _ in range(3000):
            pts = [np.zeros(3), np.array([CA_CA_A, 0.0, 0.0])]
            prev_dir = pts[1] - pts[0]
            for _ in range(n - 1):
                tau = math.radians(rng.uniform(78.0, TAU_MAX_DEG))
                # Turn prev_dir by (180 - tau) about a random perpendicular axis.
                u = prev_dir / np.linalg.norm(prev_dir)
                rnd = rng.normal(size=3)
                perp = np.cross(u, rnd)
                perp /= np.linalg.norm(perp)
                theta = math.pi - tau
                new_dir = u * math.cos(theta) + np.cross(perp, u) * math.sin(theta)
                pts.append(pts[-1] + CA_CA_A * new_dir)
                prev_dir = pts[-1] - pts[-2]
            worst = max(worst, float(np.linalg.norm(pts[-1] - pts[0])))
        assert worst <= bound + 1e-6, f"n={n}: sampled {worst:.3f} > bound {bound:.3f}"


# -------------------------------------------------------------------------- annulus
def test_annulus_range_point_to_point():
    lo, hi = annulus_distance_range(10.0, (0.0, 0.0), (0.0, 0.0))
    assert (lo, hi) == (10.0, 10.0)


def test_annulus_range_balls():
    lo, hi = annulus_distance_range(10.0, (0.0, 3.0), (0.0, 4.0))
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(17.0)


def test_annulus_range_shell_forces_a_positive_minimum():
    # A rigid 5 A arm on a core of 1 A cannot bring its tip closer than 4 A to the other point.
    lo, hi = annulus_distance_range(1.0, (5.0, 5.0), (0.0, 0.0))
    assert lo == pytest.approx(4.0)
    assert hi == pytest.approx(6.0)


# ------------------------------------------------------------------------ feasibility
def _straight_chain(length: int) -> np.ndarray:
    return np.stack([np.array([i * rise_per_link(TAU_MAX_DEG), 0.0, 0.0]) for i in range(length)])


def test_holding_whole_extended_peptide_cannot_close():
    ca = _straight_chain(13)
    spec = mainchain_spec(13)
    ok, lo, hi = feasible_analytic(ca, spec, 0, 12)
    assert not ok, "a rigid 44 A extended chain must not be callable closed"
    assert lo > spec.hi


def test_releasing_enough_residues_restores_feasibility():
    ca = _straight_chain(13)
    spec = mainchain_spec(13)
    # Hold only the middle: both flanks are long enough to bring the termini together.
    ok, _, _ = feasible_analytic(ca, spec, 5, 7)
    assert ok


def test_feasible_window_shrinks_as_the_gap_grows():
    spec_len = 13
    best_lens = []
    for stretch in (0.4, 0.7, 1.0):
        ca = np.stack([np.array([i * rise_per_link(TAU_MAX_DEG) * stretch, 0.0, 0.0])
                       for i in range(spec_len)])
        spec = mainchain_spec(spec_len)
        res = scan_windows(ca, set(), np.zeros((0, 3)), spec)
        best_lens.append(res.max_feasible_window_len)
    assert best_lens[0] >= best_lens[1] >= best_lens[2], best_lens
    assert best_lens[0] > best_lens[2], "a more extended peptide must license a smaller window"


def test_native_atom_distance_overrides_the_ball_model():
    """Both anchors inside a rigid window: the answer is the measured distance, not a range."""
    ca = _straight_chain(13)
    spec = mainchain_spec(13)
    ok_ball, _, _ = feasible_analytic(ca, spec, 0, 12, native_atom_dist=None)
    ok_true, lo, hi = feasible_analytic(ca, spec, 0, 12, native_atom_dist=40.0)
    assert not ok_true and lo == hi == 40.0
    assert ok_ball is False  # the ball model happens to agree here, but it is not consulted


# -------------------------------------------------------------------------- retention
def test_retention_strict_counts_only_window_residues():
    ca = _straight_chain(10)
    rec = np.zeros((3, 3))
    contacts = {(0, 0), (1, 0), (5, 1), (9, 2)}
    strict, perm = retention_ceiling(contacts, rec, ca, 0, 1)
    assert strict == pytest.approx(0.5)
    assert perm >= strict


def test_retention_permissive_is_an_upper_bound():
    ca = _straight_chain(10)
    rng = np.random.default_rng(1)
    rec = rng.normal(scale=10.0, size=(12, 3))
    contacts = ca_contact_set(ca, rec)
    if not contacts:
        pytest.skip("no contacts in the random arrangement")
    for a, b in ((0, 2), (3, 6), (0, 9)):
        strict, perm = retention_ceiling(contacts, rec, ca, a, b)
        assert 0.0 <= strict <= perm <= 1.0


def test_contact_set_matches_the_10A_CA_definition():
    pep = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
    rec = np.array([[9.9, 0.0, 0.0], [10.1, 0.0, 0.0]])
    # 9.9 A is inside the 10 A cutoff, 10.1 A is outside, and the second peptide residue
    # is far from both -- so exactly one pair survives.
    assert ca_contact_set(pep, rec) == {(0, 0)}


# ------------------------------------------------------------------------------ NeRF
def _synthetic_backbone(length: int, seed: int = 0):
    """A real-geometry backbone built from random torsions, for round-trip testing."""
    from script_utils.kinematic_ceiling import (ANG_CA_C_N, ANG_C_N_CA, ANG_N_CA_C,
                                                CA_C_A, N_CA_A)
    rng = np.random.default_rng(seed)
    N = np.zeros((length, 3)); CA = np.zeros((length, 3)); C = np.zeros((length, 3))
    N[0] = np.array([0.0, 0.0, 0.0])
    CA[0] = np.array([N_CA_A, 0.0, 0.0])
    ang = math.radians(ANG_N_CA_C)
    C[0] = CA[0] + CA_C_A * np.array([math.cos(math.pi - ang), math.sin(math.pi - ang), 0.0])
    phi = rng.uniform(-170, -50, size=length)
    psi = rng.uniform(-60, 160, size=length)
    omega = np.full(length, 180.0)
    for k in range(length - 1):
        N[k + 1] = place_atom(N[k], CA[k], C[k], C_N_A, ANG_CA_C_N, psi[k])
        CA[k + 1] = place_atom(CA[k], C[k], N[k + 1], N_CA_A, ANG_C_N_CA, omega[k])
        C[k + 1] = place_atom(C[k], N[k + 1], CA[k + 1], CA_C_A, ANG_N_CA_C, phi[k + 1])
    return N, CA, C


def test_rebuild_is_identity_on_native_torsions():
    N, CA, C = _synthetic_backbone(11, seed=3)
    t = backbone_torsions(N, CA, C)
    phi = np.nan_to_num(t["phi"], nan=-120.0)
    psi = np.nan_to_num(t["psi"], nan=130.0)
    omega = np.nan_to_num(t["omega"], nan=180.0)
    for a, b in ((0, 10), (4, 6), (0, 3), (8, 10)):
        Nn, CAn, Cn = rebuild_from_window(N, CA, C, a, b, phi, psi, omega)
        assert np.allclose(CAn, CA, atol=1e-6), f"window ({a},{b}) moved the CA trace"
        assert np.allclose(Nn, N, atol=1e-6)
        assert np.allclose(Cn, C, atol=1e-6)


def test_rebuild_keeps_the_held_window_exactly_fixed():
    N, CA, C = _synthetic_backbone(12, seed=5)
    t = backbone_torsions(N, CA, C)
    phi = np.nan_to_num(t["phi"], nan=-120.0).copy()
    psi = np.nan_to_num(t["psi"], nan=130.0).copy()
    omega = np.nan_to_num(t["omega"], nan=180.0)
    a, b = 4, 8
    phi[0] += 55.0; psi[1] -= 70.0; phi[11] += 90.0   # only free residues touched
    Nn, CAn, Cn = rebuild_from_window(N, CA, C, a, b, phi, psi, omega)
    assert np.allclose(CAn[a:b + 1], CA[a:b + 1], atol=1e-9)
    assert not np.allclose(CAn[0], CA[0], atol=1e-3), "a changed free torsion must move something"


def test_dihedral_round_trips():
    N, CA, C = _synthetic_backbone(6, seed=7)
    assert dihedral(N[0], CA[0], C[0], N[1]) == pytest.approx(
        backbone_torsions(N, CA, C)["psi"][0], abs=1e-6)


# ------------------------------------------------------- analytic vs torsion agreement
def test_torsion_solver_agrees_with_the_analytic_verdict_on_clear_cases():
    """On cases far from the boundary the two instruments must not contradict each other."""
    N, CA, C = _synthetic_backbone(12, seed=11)
    spec = mainchain_spec(12)

    # Hold almost nothing -> both must say feasible.
    ok_a, _, _ = feasible_analytic(CA, spec, 5, 6)
    ok_t, resid = feasible_torsion(N, CA, C, spec, 5, 6, n_restarts=6, seed=0)
    assert ok_a, "analytic should license a 2-residue hold on a 12-mer"
    assert ok_t, f"torsion solve failed to close with 10 free residues (residual {resid:.2f} A)"

    # Hold everything on an extended chain -> both must say infeasible.
    d_native = float(np.linalg.norm(N[0] - C[11]))
    if d_native > 8.0:
        ok_a2, _, _ = feasible_analytic(CA, spec, 0, 11, native_atom_dist=d_native)
        ok_t2, _ = feasible_torsion(N, CA, C, spec, 0, 11, n_restarts=2, seed=0)
        assert not ok_a2 and not ok_t2


def test_analytic_never_calls_infeasible_what_the_solver_closes():
    """The analytic bound must be permissive: it may over-license, never under-license."""
    rng = np.random.default_rng(23)
    disagreements = []
    for seed in range(6):
        N, CA, C = _synthetic_backbone(10, seed=seed)
        spec = mainchain_spec(10)
        for _ in range(4):
            a = int(rng.integers(0, 8))
            b = int(rng.integers(a, 10))
            ok_a, _, _ = feasible_analytic(CA, spec, a, b,
                                           native_atom_dist=float(np.linalg.norm(N[0] - C[9]))
                                           if (a == 0 and b == 9) else None)
            ok_t, _ = feasible_torsion(N, CA, C, spec, a, b, n_restarts=4, seed=1)
            if ok_t and not ok_a:
                disagreements.append((seed, a, b))
    assert not disagreements, f"analytic under-licensed real closures: {disagreements}"


# -------------------------------------------------------- in-window atom pinning
def test_in_window_terminal_atom_is_pinned_not_free():
    """Holding a window holds its backbone, so an in-window N or C must not float.

    Modelling it as free within its 1.46 A bond adds ~3 A of slack across the pair, which
    was enough to license windows the torsion solver could not close.
    """
    from script_utils.kinematic_ceiling import feasible_analytic as fa

    N, CA, C = _synthetic_backbone(10, seed=17)
    spec = mainchain_spec(10)
    a, b = 0, 8                      # residue 0 in-window, residue 9 free
    ok_loose, lo_loose, _ = fa(CA, spec, a, b)                       # CA-trace-only reading
    ok_tight, lo_tight, _ = fa(CA, spec, a, b, atom_i_xyz=N[0], atom_j_xyz=C[9])
    assert lo_tight >= lo_loose - 1e-9, "pinning must not widen the achievable range"
    if ok_loose != ok_tight:
        assert ok_loose and not ok_tight, "pinning may only ever remove feasibility"


def test_pinning_makes_a_fully_held_ring_exact():
    N, CA, C = _synthetic_backbone(12, seed=19)
    spec = mainchain_spec(12)
    d = float(np.linalg.norm(N[0] - C[11]))
    from script_utils.kinematic_ceiling import feasible_analytic as fa
    ok, lo, hi = fa(CA, spec, 0, 11, atom_i_xyz=N[0], atom_j_xyz=C[11])
    assert lo == pytest.approx(d) and hi == pytest.approx(d)
    assert ok == (spec.lo <= d <= spec.hi)


# --------------------------------------------------------- torsion refinement
def test_refine_prunes_supersets_of_a_failure():
    """Holding MORE residues can never make closure easier, so a failure prunes its supersets."""
    from script_utils.kinematic_ceiling import (CeilingResult, refine_with_torsion)

    N, CA, C = _synthetic_backbone(12, seed=31)
    spec = mainchain_spec(12)
    # Measured on this backbone: [1,10] (2 free residues) cannot close, [2,9] (4 free) can.
    # [0,11] contains [1,10], so it must be skipped WITHOUT a solve once [1,10] fails.
    cands = [(0.90, 1.0, 1, 10), (0.89, 1.0, 0, 11), (0.88, 1.0, 2, 9)]
    res = CeilingResult("mainchain", True, 0.90, 1.0, 1, 10, 10, 10, 3, 78, False)
    out = refine_with_torsion(res, cands, N, CA, C, spec, max_solves=10, n_restarts=2,
                              tol_A=0.25, max_seconds=5.0)
    assert out.refine_solves == 2, (
        f"expected [1,10] to fail, [0,11] to be pruned as its superset, and [2,9] to "
        f"succeed -- 2 solves, got {out.refine_solves}")
    assert (out.refined_a, out.refined_b) == (2, 9)
    assert out.ceiling_refined == pytest.approx(0.88)


def test_refine_never_raises_the_analytic_ceiling():
    from script_utils.kinematic_ceiling import (CeilingResult, rank_feasible_windows,
                                                refine_with_torsion)

    N, CA, C = _synthetic_backbone(10, seed=37)
    spec = mainchain_spec(10)
    rec = np.random.default_rng(2).normal(scale=9.0, size=(14, 3))
    contacts = ca_contact_set(CA, rec)
    res = scan_windows(CA, contacts, rec, spec, atom_i_xyz=N[0], atom_j_xyz=C[9])
    cands = rank_feasible_windows(CA, contacts, rec, spec, atom_i_xyz=N[0], atom_j_xyz=C[9])
    out = refine_with_torsion(res, cands, N, CA, C, spec, max_solves=20, n_restarts=3,
                              tol_A=0.25, max_seconds=5.0)
    if math.isfinite(out.ceiling_refined):
        assert out.ceiling_refined <= res.ceiling_strict + 1e-9
