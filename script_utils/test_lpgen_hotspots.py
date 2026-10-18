#!/usr/bin/env python3
"""Tests for the hotspot panel construction and the probe's dosage arithmetic.

These run in the PROJECT venv, where rfd3 is not installed, so they cover the parts that are
pure geometry and pure arithmetic. The parts that need the arm's environment -- whether the
annotation reaches ``data["feats"]``, what the realised live-hotspot rate is -- are measured
by `lpgen_hotspot_probe.py` on the real stack instead, because a mock of that pipeline would
only test the mock.

What is worth testing here is what would be invisible in a log line and wrong in the numbers:

the decoy patch must not touch the native site
    That separation IS the control. A decoy patch brushing the native interface is partly
    satisfiable by binding natively, so the model scores as obedient without having moved,
    and the arm quietly stops being a negative control. Checked as a distance, not by
    construction.

the request must be identical at every epoch
    A validation request that is re-drawn per epoch cannot be compared across checkpoints,
    which is the whole reason to validate periodically. So the subsample is checked for
    determinism across calls rather than assumed from "it uses a hash".

the ori must be pushed OUTWARD
    Getting the sign wrong puts the diffusion prior inside the protein, where no peptide can
    satisfy any hotspot. The failure then looks like a model that will not obey.

upstream's exclusive bound
    ``randint(0, ceil(n*frac))`` is uniform on ``[0, ceil-1]``, so a ceiling of 1 means
    ALWAYS ZERO hotspots while `add_ppi_hotspots` was true. The helper that detects this is
    tested on the exact boundary, because the whole dosage decision turns on it.

Run:  .venv/bin/python script_utils/test_lpgen_hotspots.py
      .venv/bin/python -m pytest -q script_utils/test_lpgen_hotspots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpgen_build_val_contigs import (  # noqa: E402
    decoy_hotspot_mask,
    deterministic_subsample,
    hotspot_selection,
    native_hotspot_mask,
    patch_ori,
)
from lpgen_hotspot_probe import (  # noqa: E402
    degenerate_draw_ceiling,
    expected_provided,
)


class FakeAtomArray:
    """Duck-typed stand-in exposing only what these functions read."""

    def __init__(self, chain_id, res_id, atom_name, coord):
        self.chain_id = np.asarray(chain_id, dtype=object)
        self.res_id = np.asarray(res_id)
        self.atom_name = np.asarray(atom_name, dtype=object)
        self.coord = np.asarray(coord, dtype=float)


def make_complex(sep=4.0, n_rec_res=40, r_sphere=12.0):
    """A spherical 'receptor' plus a short peptide sitting just off its +x pole.

    A sphere is used so that "farthest from the centroid" is a well-defined surface atom and
    the decoy seed lands on the opposite pole -- which is exactly the geometry the decoy
    construction assumes.
    """
    chain, res, name, xyz = [], [], [], []

    # Receptor: points on a Fibonacci sphere, one atom per residue.
    for i in range(n_rec_res):
        t = (i + 0.5) / n_rec_res
        z = 1.0 - 2.0 * t
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = np.pi * (1 + 5**0.5) * i
        chain.append("A"); res.append(i + 1); name.append("CA")
        xyz.append([r_sphere * r * np.cos(phi), r_sphere * r * np.sin(phi), r_sphere * z])

    # Peptide: 5 residues strung along y, offset in +x by the sphere radius plus `sep`.
    for j in range(5):
        chain.append("B"); res.append(j + 1); name.append("CA")
        xyz.append([r_sphere + sep, float(j) * 3.8 - 7.6, 0.0])

    return FakeAtomArray(chain, res, name, xyz)


# ------------------------------------------------------------------ native hotspots
def test_native_hotspots_are_receptor_only():
    aa = make_complex()
    m = native_hotspot_mask(aa, "B", cutoff=8.0)
    assert m.sum() > 0, "the peptide sits 4 A off the surface; something must be in contact"
    chain = np.asarray(aa.chain_id, dtype=str)
    assert not m[chain == "B"].any(), "a hotspot on the BINDER would leak the answer"


def test_native_hotspots_respect_the_cutoff_exactly():
    # One receptor atom at the origin, one peptide atom exactly 4.5 A away.
    aa = FakeAtomArray(
        ["A", "A", "B"], [1, 2, 1], ["CA", "CA", "CA"],
        [[0, 0, 0], [0, 0, 100.0], [4.5, 0, 0]],
    )
    assert native_hotspot_mask(aa, "B", cutoff=4.5).sum() == 1, "inclusive at the boundary"
    assert native_hotspot_mask(aa, "B", cutoff=4.4).sum() == 0
    # The distant receptor atom must never be included.
    assert not native_hotspot_mask(aa, "B", cutoff=4.5)[1]


def test_no_peptide_means_no_hotspots():
    aa = FakeAtomArray(["A", "A"], [1, 2], ["CA", "CA"], [[0, 0, 0], [1, 0, 0]])
    assert native_hotspot_mask(aa, "B", cutoff=8.0).sum() == 0


# ------------------------------------------------------------------ subsampling
def test_subsample_is_deterministic_and_a_subset():
    aa = make_complex()
    m = native_hotspot_mask(aa, "B", cutoff=10.0)
    n = max(1, int(m.sum()) // 3)
    a = deterministic_subsample(m, n, "example|native")
    b = deterministic_subsample(m, n, "example|native")
    assert np.array_equal(a, b), "the request must be identical at every epoch"
    assert a.sum() == n
    assert not (a & ~m).any(), "subsample must stay inside the ground-truth set"


def test_subsample_different_keys_differ():
    aa = make_complex()
    m = native_hotspot_mask(aa, "B", cutoff=10.0)
    n = max(2, int(m.sum()) // 3)
    if int(m.sum()) > n:  # only meaningful when there is room to choose differently
        a = deterministic_subsample(m, n, "x|native")
        b = deterministic_subsample(m, n, "x|decoy")
        assert not np.array_equal(a, b) or int(m.sum()) == n


def test_subsample_degenerate_inputs():
    m = np.array([True, False, True, True])
    assert deterministic_subsample(m, 0, "k").sum() == 0
    assert deterministic_subsample(m, -1, "k").sum() == 0
    # Asking for more than exist returns the whole set, not an error.
    assert np.array_equal(deterministic_subsample(m, 99, "k"), m)
    assert deterministic_subsample(np.zeros(4, dtype=bool), 2, "k").sum() == 0


# ------------------------------------------------------------------ decoy patch
def test_decoy_patch_is_far_from_the_native_site():
    aa = make_complex()
    nat = native_hotspot_mask(aa, "B", cutoff=10.0)
    dec, info = decoy_hotspot_mask(aa, "B", nat, radius=8.0, min_separation=15.0)
    assert dec is not None, f"a sphere must admit an opposite patch: {info}"
    assert dec.sum() > 0

    chain = np.asarray(aa.chain_id, dtype=str)
    assert not dec[chain == "B"].any(), "the decoy patch must be on the receptor"
    assert not (dec & nat).any(), "the decoy patch must not overlap the native hotspots"

    # The separation is the control, so measure it rather than trusting the construction.
    d = np.linalg.norm(
        aa.coord[dec][:, None, :] - aa.coord[nat][None, :, :], axis=-1
    ).min()
    assert d >= 15.0, f"decoy patch is only {d:.1f} A from the native site"


def test_decoy_refuses_when_separation_is_unsatisfiable():
    aa = make_complex()
    nat = native_hotspot_mask(aa, "B", cutoff=10.0)
    # A separation larger than the whole receptor cannot be met; the builder must get None
    # and omit the decoy arms rather than emit a patch that is secretly near the native site.
    dec, info = decoy_hotspot_mask(aa, "B", nat, radius=8.0, min_separation=1e4)
    assert dec is None and "separation" in info


def test_decoy_needs_native_hotspots():
    aa = make_complex()
    dec, info = decoy_hotspot_mask(aa, "B", np.zeros(len(aa.chain_id), dtype=bool),
                                   radius=8.0, min_separation=15.0)
    assert dec is None


# ------------------------------------------------------------------ ori
def test_patch_ori_is_pushed_outward():
    aa = make_complex()
    nat = native_hotspot_mask(aa, "B", cutoff=10.0)
    chain = np.asarray(aa.chain_id, dtype=str)
    core = aa.coord[chain == "A"].mean(0)
    patch_c = aa.coord[nat].mean(0)

    ori0, _ = patch_ori(aa, nat, "B", push=0.0)
    ori6, _ = patch_ori(aa, nat, "B", push=6.0)
    assert ori0 is not None and ori6 is not None

    d0 = np.linalg.norm(np.asarray(ori0) - core)
    d6 = np.linalg.norm(np.asarray(ori6) - core)
    # OUTWARD, not inward: a sign error here drops the prior inside the protein and the
    # failure reads as a disobedient model.
    assert d6 > d0, f"push moved the ori inward ({d6:.2f} <= {d0:.2f})"
    assert np.isclose(d6 - d0, 6.0, atol=1e-6)
    assert np.allclose(np.asarray(ori0), patch_c), "push 0 must be the patch centroid"


def test_patch_ori_refuses_a_degenerate_direction():
    # Patch centroid at the receptor core: no outward direction exists, and normalising
    # would emit NaNs that rfd3 would silently translate the structure by.
    aa = FakeAtomArray(
        ["A", "A", "B"], [1, 2, 1], ["CA", "CA", "CA"],
        [[-1, 0, 0], [1, 0, 0], [50, 0, 0]],
    )
    mask = np.array([True, True, False])
    ori, info = patch_ori(aa, mask, "B", push=6.0)
    assert ori is None and "core" in info


# ------------------------------------------------------------------ selection dict
def test_hotspot_selection_key_format_and_names():
    aa = FakeAtomArray(
        ["A", "A", "A", "B"], [7, 7, 9, 1], ["N", "CB", "CA", "CA"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    )
    sel = hotspot_selection(aa, np.array([True, True, False, False]))
    # `apply_selections` looks up exactly f"{chain_id}{res_id}" (input_parsing.py:508).
    assert set(sel) == {"A7"}
    assert sel["A7"] == ["N", "CB"], "atom-level, in array order"
    # Multiple residues become multiple keys.
    sel2 = hotspot_selection(aa, np.array([True, False, True, False]))
    assert set(sel2) == {"A7", "A9"} and sel2["A9"] == ["CA"]
    assert hotspot_selection(aa, np.zeros(4, dtype=bool)) == {}


# ------------------------------------------------------------------ dosage arithmetic
def test_ceiling_one_means_always_zero():
    # THE degenerate corner: randint(0, 1) is always 0, so these examples get no hotspots
    # while add_ppi_hotspots was true, and upstream's "No hotspots found" warning does NOT
    # fire (that branch needs the ceiling to be 0).
    for n_gt in (1, 2, 3, 4, 5):
        assert degenerate_draw_ceiling(n_gt, 0.2) == 1, n_gt
        assert expected_provided(n_gt, 0.2) == 0.0, n_gt
    # 6 atoms is the first n_gt that can provide anything at frac 0.2.
    assert degenerate_draw_ceiling(6, 0.2) == 2
    assert expected_provided(6, 0.2) == 0.5


def test_ceiling_zero_is_the_warned_case():
    assert degenerate_draw_ceiling(0, 0.2) == 0
    assert expected_provided(0, 0.2) == 0.0


def test_expected_provided_is_half_the_ceiling_not_the_fraction():
    # The config key reads `max_..._frac_to_provide`, which invites expecting n*frac/2 = 10
    # for n=100. The exclusive bound makes it (ceil(20)-1)/2 = 9.5, and the max reachable is
    # 19 rather than 20. Small, but it is why the realised dosage is not the configured one.
    assert degenerate_draw_ceiling(100, 0.2) == 20
    assert expected_provided(100, 0.2) == 9.5




# ------------------------------------------------------- hotspot retention metric
# These cover the correction described in rfd3_metrics.py: upstream's PPIMetrics counts
# contacts to `~is_motif_atom`, which is EMPTY on a sequence-given panel, so it reports a
# hard 0.0 on every spec. The replacement keys off generated coordinates instead.

def test_retention_is_one_when_every_hotspot_is_touched():
    from proteinfoundation.lpgen.rfd3_metrics import hotspot_retention
    hs = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    gen = np.array([[2.0, 0, 0], [3.0, 0, 0]])  # within 4.5 A of both
    r = hotspot_retention(hs, gen, distance_cutoff=4.5)
    assert r["fraction_hotspots_contacted"] == 1.0
    assert r["n_hotspots_requested"] == 2.0


def test_retention_is_partial_and_counts_per_hotspot():
    from proteinfoundation.lpgen.rfd3_metrics import hotspot_retention
    hs = np.array([[0.0, 0, 0], [100.0, 0, 0]])
    gen = np.array([[1.0, 0, 0]])
    r = hotspot_retention(hs, gen, distance_cutoff=4.5)
    assert r["fraction_hotspots_contacted"] == 0.5


def test_retention_cutoff_is_inclusive():
    from proteinfoundation.lpgen.rfd3_metrics import hotspot_retention
    hs = np.array([[0.0, 0, 0]])
    gen = np.array([[4.5, 0, 0]])
    assert hotspot_retention(hs, gen, 4.5)["fraction_hotspots_contacted"] == 1.0
    assert hotspot_retention(hs, gen, 4.4)["fraction_hotspots_contacted"] == 0.0


def test_no_request_reports_nothing_rather_than_zero():
    # The distinction the whole panel rests on: an ABSENT retention number means "nothing
    # was asked", and must never be a 0.0 that reads as total disobedience.
    from proteinfoundation.lpgen.rfd3_metrics import hotspot_retention
    r = hotspot_retention(np.zeros((0, 3)), np.array([[1.0, 0, 0]]))
    assert "fraction_hotspots_contacted" not in r
    assert r["n_hotspots_requested"] == 0.0
    # Same when nothing was generated -- e.g. a spec that released no atoms.
    r2 = hotspot_retention(np.array([[0.0, 0, 0]]), np.zeros((0, 3)))
    assert "fraction_hotspots_contacted" not in r2
    assert r2["n_generated_atoms"] == 0.0


def test_continuous_distances_separate_near_miss_from_ignoring():
    # THE reason the continuous terms are reported. Both of these score 0.0 on the binary
    # fraction, and the decoy arm's entire signal is the difference between them.
    from proteinfoundation.lpgen.rfd3_metrics import hotspot_retention
    hs = np.array([[0.0, 0, 0]])
    near = hotspot_retention(hs, np.array([[6.0, 0, 0]]), 4.5)
    far = hotspot_retention(hs, np.array([[60.0, 0, 0]]), 4.5)
    assert near["fraction_hotspots_contacted"] == far["fraction_hotspots_contacted"] == 0.0
    assert near["min_dist_to_hotspot"] < far["min_dist_to_hotspot"]
    assert near["com_to_hotspot_com"] < far["com_to_hotspot_com"]


def test_nonfinite_coordinates_are_dropped_not_propagated():
    from proteinfoundation.lpgen.rfd3_metrics import hotspot_retention
    hs = np.array([[0.0, 0, 0], [np.nan, 0, 0]])
    gen = np.array([[1.0, 0, 0], [np.inf, 0, 0]])
    r = hotspot_retention(hs, gen, 4.5)
    assert r["n_hotspots_requested"] == 1.0 and r["n_generated_atoms"] == 1.0
    assert r["fraction_hotspots_contacted"] == 1.0
    assert np.isfinite(r["min_dist_to_hotspot"])


def _main() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
