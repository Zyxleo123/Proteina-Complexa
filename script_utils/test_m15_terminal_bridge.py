"""The terminal-pinned bridged arm of the M1.5 ceiling audit.

The free bridged scan maximises over every (i, j) pair, which CPSea cannot reach: it
conditions on the chain termini, so (0, L-1) is the only anchor pair it can place a bridge
between.  These tests hold the pinned arm to that contract and to the ordering that makes
the pair interpretable -- the free scan is an upper bound on the pinned one, by
construction, because (0, L-1) is in the set it maximises over.

Run: .venv/bin/python -m pytest script_utils/test_m15_terminal_bridge.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from script_utils.m15_ceiling_audit import MIN_BRIDGE_SEP, audit_one  # noqa: E402

# The production windows, copied from the full run's bridge_windows.json rather than
# invented: a made-up window would test the plumbing against a bond length that no native
# bridge occupies.  `cb_*` is consumed by bridge_profile, `ca_*` by the ceiling scan.
WINDOWS = {
    "disulfide": {"ca_lo": 3.508995121593384, "ca_hi": 7.31631483482169,
                  "cb_lo": 2.7863587822595535, "cb_hi": 5.096188542208212},
    "isopeptide": {"ca_lo": 4.209265840812561, "ca_hi": 10.283220570911434,
                   "cb_lo": 3.789652565964836, "cb_hi": 7.879759369241926},
    "mainchain": {"ca_lo": 2.4828243305401188, "ca_hi": 4.4257062552020905,
                  "cb_lo": 2.246513620384004, "cb_hi": 6.50372393144645},
}

CFG = {"contact_cutoff_A": 10.0, "heavy_cutoff_A": 4.5, "tau_max_deg": 150.0,
       "mainchain_tol_A": 0.25, "refine": {"enabled": False}}


def synthetic(L: int = 12, seed: int = 0) -> dict:
    """An extended peptide lying alongside a receptor slab.

    Deliberately NOT a closed ring: an already-cyclic input would put every pair at bond
    distance and the ceilings would all be 1.0, which tests nothing.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(L, dtype=float)
    CA = np.stack([t * 3.5, 1.5 * np.sin(t * 0.9), 1.5 * np.cos(t * 0.9)], axis=1)
    N = CA + np.array([-1.2, 0.3, 0.0])
    C = CA + np.array([1.2, -0.3, 0.0])
    CB = CA + np.array([0.0, 1.5, 0.0])
    rec_ca = np.stack([t * 3.5, np.full(L, -7.0), np.zeros(L)], axis=1)
    pep_heavy = [np.stack([N[k], CA[k], C[k], CB[k]]) for k in range(L)]
    return {
        "N": N, "CA": CA, "C": C, "CB": CB,
        "resnames": ["ALA"] * L,
        "resseq": list(range(L)),
        "rec_ca": rec_ca,
        "rec_resseq": list(range(L)),
        "pep_heavy": pep_heavy,
        "rec_heavy": rec_ca + rng.normal(0, 0.01, rec_ca.shape),
    }


@pytest.mark.parametrize("typ", ["disulfide", "isopeptide"])
def test_pinned_anchors_are_the_termini(typ):
    row = audit_one(synthetic(), WINDOWS, CFG)
    assert row[f"{typ}_term_anchor_i"] == 0
    assert row[f"{typ}_term_anchor_j"] == 11


@pytest.mark.parametrize("typ", ["disulfide", "isopeptide"])
def test_free_scan_upper_bounds_the_pinned_scan(typ):
    """(0, L-1) is one of the pairs the free scan maximises over, so it cannot lose to it.

    This is the property that makes the two columns a meaningful pair: any gap between
    them is the cost of being restricted to the termini, not a difference in method.
    """
    row = audit_one(synthetic(), WINDOWS, CFG)
    if not row.get(f"{typ}_term_feasible_any"):
        pytest.skip("pinned arm infeasible on this synthetic case")
    assert row[f"{typ}_ceiling_strict"] >= row[f"{typ}_term_ceiling_strict"] - 1e-9


@pytest.mark.parametrize("typ", ["disulfide", "isopeptide"])
def test_free_and_pinned_columns_are_both_emitted(typ):
    row = audit_one(synthetic(), WINDOWS, CFG)
    for key in ("feasible_any", "ceiling_strict", "ceiling_permissive"):
        assert f"{typ}_{key}" in row, f"missing free column {typ}_{key}"
        assert f"{typ}_term_{key}" in row, f"missing pinned column {typ}_term_{key}"


def test_short_peptide_reports_infeasible_rather_than_crashing():
    """L-1 < MIN_BRIDGE_SEP leaves no legal terminal pair; the row must still be written.

    A crash here would lose the whole complex from the shard, including its head-to-tail
    result, which is measurable independently of whether a bridge fits.
    """
    st = synthetic(L=MIN_BRIDGE_SEP)      # j = L-1 = MIN_BRIDGE_SEP - 1, too close to i=0
    row = audit_one(st, WINDOWS, CFG)
    for typ in ("disulfide", "isopeptide"):
        assert row[f"{typ}_term_feasible_any"] is False
        assert np.isnan(row[f"{typ}_term_ceiling_strict"])
    assert "mainchain_ceiling_strict" in row


def test_pinned_arm_does_not_disturb_the_free_arm():
    """The free bridged columns must be byte-identical to what the pre-change code gave.

    Guarded by recomputing the free maximum here, independently of audit_one's loop.
    """
    from script_utils.kinematic_ceiling import bridge_spec, ca_contact_set, scan_windows

    st = synthetic()
    L = len(st["CA"])
    contacts = ca_contact_set(st["CA"], st["rec_ca"], CFG["contact_cutoff_A"])
    for typ in ("disulfide", "isopeptide"):
        w = WINDOWS[typ]
        best = None
        for i in range(L):
            for j in range(i + MIN_BRIDGE_SEP, L):
                spec = bridge_spec(typ, i, j, w["ca_lo"], w["ca_hi"])
                r = scan_windows(st["CA"], contacts, st["rec_ca"], spec,
                                 CFG["tau_max_deg"], CFG["contact_cutoff_A"],
                                 atom_i_xyz=st["CA"][i], atom_j_xyz=st["CA"][j])
                if r.feasible_any and (best is None or r.ceiling_strict > best):
                    best = r.ceiling_strict
        row = audit_one(st, WINDOWS, CFG)
        if best is None:
            assert row[f"{typ}_feasible_any"] is False
        else:
            assert row[f"{typ}_ceiling_strict"] == pytest.approx(best)
