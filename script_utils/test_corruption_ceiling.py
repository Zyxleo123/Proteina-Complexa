"""Exercise `open_ladder` without OpenMM, a force field, or a real complex.

This function has now shipped two runtime bugs that `py_compile` could not see and that the
report-level synthetic tests could not reach, because they only ever read rows somebody else
wrote:

  * one bounded minimize() per rung, so the achieved gap tracked the iteration budget;
  * `len(rungs)` left behind after the loop variable was renamed to `plan`, which raised
    NameError on EVERY example of a 60-complex run while the per-example try/except kept the
    job exiting 0.

So the rung plan and the row schema get a test that runs anywhere. The physics is stubbed --
what is checked is the bookkeeping: which rungs exist, which are references, and that every
row a consumer reads is populated.

Run under either venv (pytest is only in .venv, openmm only in .venv_openmm):
    .venv/bin/python -m pytest script_utils/test_corruption_ceiling.py -q
    .venv_openmm/bin/python script_utils/test_corruption_ceiling.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_module_with_stubs():
    """Import corruption_ceiling with its OpenMM-side imports stubbed out.

    The stub modules answer ANY attribute, because soft_closure_project imports a handful of
    OpenMM names at module scope (CustomBondForce, Platform, app, ...) that this test never
    calls. `unit` is the exception: its members have to be real numbers, since coordinates are
    multiplied by them.
    """
    class _Permissive(types.ModuleType):
        def __getattr__(self, name):                      # anything not set explicitly
            return types.SimpleNamespace()

    if "openmm" not in sys.modules:
        unit = types.SimpleNamespace(nanometer=1.0, kilojoule_per_mole=1.0, picoseconds=1.0,
                                     angstrom=0.1)
        mod = _Permissive("openmm")
        mod.unit = unit
        mod.LocalEnergyMinimizer = types.SimpleNamespace(minimize=lambda *a, **k: None)
        mod.Platform = types.SimpleNamespace(getPlatformByName=lambda n: None)
        app = _Permissive("openmm.app")
        app.ForceField = lambda *a, **k: None
        mod.app = app
        sys.modules["openmm"] = mod
        sys.modules["openmm.unit"] = unit
        sys.modules["openmm.app"] = app
    if "pdbfixer" not in sys.modules:
        pf = _Permissive("pdbfixer")
        pf.PDBFixer = object
        sys.modules["pdbfixer"] = pf
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import script_utils.corruption_ceiling as cc  # noqa: E402
    return cc


class _FakeState:
    def __init__(self, xyz_nm):
        self._xyz = xyz_nm

    def getPositions(self):
        return types.SimpleNamespace(value_in_unit=lambda u: self._xyz)

    def getPotentialEnergy(self):
        return types.SimpleNamespace(value_in_unit=lambda u: -123.0)


class _FakeContext:
    """Moves the two pulled atoms straight to whatever r0 asks for, in one round."""

    def __init__(self, xyz_nm, ia, ib):
        self.xyz = xyz_nm.copy()
        self.ia, self.ib = ia, ib
        self.r0 = None

    def setPositions(self, xyz):
        self.xyz = np.asarray(xyz).copy()

    def setParameter(self, name, value):
        if name == "r0":
            self.r0 = value
            # place atom ib at distance r0 (nm) from ia along x
            self.xyz[self.ib] = self.xyz[self.ia] + np.array([value, 0.0, 0.0])

    def getState(self, **kw):
        return _FakeState(self.xyz)


def _fake_ec(cc, n_res=8):
    # A CLOSED loop, not a straight chain: consecutive CAs 0.38 nm apart and the termini
    # 0.38 nm apart too, so d0 ~ 3.8 A. That is what a CPSea head-to-tail macrocycle actually
    # measures (3.44 A), and it means the test ladder has rungs above d0 to walk.
    xyz_nm = np.zeros((n_res * 2, 3))
    radius = 0.38 / (2.0 * np.sin(np.pi / n_res))
    for i in range(n_res):
        theta = 2.0 * np.pi * i / n_res
        xyz_nm[i] = [radius * np.cos(theta), radius * np.sin(theta), 0.0]
    ia, ib = 0, n_res - 1
    ec = types.SimpleNamespace(
        base_xyz_nm=xyz_nm, ia=ia, ib=ib,
        pep_heavy_idx=np.arange(n_res), pep_ca_idx=np.arange(n_res),
        rec_heavy_idx=np.arange(n_res, n_res * 2),
        pep_res_atoms=[{"CA": i} for i in range(n_res)],
        contact_pairs=[], topology=types.SimpleNamespace(getNumAtoms=lambda: n_res * 2),
        ref={"xyz_A_ref": xyz_nm * 10.0, "rec_ca_A": xyz_nm[n_res:] * 10.0,
             "pep_ca_A": xyz_nm[:n_res] * 10.0, "contacts": set(), "terminal_cb_A": 1.0},
    )
    ec.context = _FakeContext(xyz_nm, ia, ib)
    return ec


def _run(tmpdir, ladder=(8.0, 10.0, 12.0)):
    cc = _load_module_with_stubs()
    cc.geometry_report = lambda *a, **k: {"contact_retention": 0.9, "ca_step_min_A": 3.7,
                                          "ca_step_max_A": 3.9, "min_pep_rec_heavy_A": 2.8,
                                          "n_contacts_input": 10, "n_contacts_projected": 9,
                                          "ca_rmsd_to_input_A": 0.2,
                                          "ca_max_dev_to_input_A": 0.5}
    cc.write_projected_pdb = lambda *a, **k: 0
    cc.terminal_cb_A = lambda *a, **k: 1.0
    args = types.SimpleNamespace(ladder_A=list(ladder), pull_rounds=4, tolerance=10.0,
                                 max_iterations=10, converge_tol_A=0.5, stall_eps_A=0.05,
                                 exclude_termini=2, keep_added_atoms=False, k_contact=0.0)
    ec = _fake_ec(cc)
    return cc, cc.open_ladder(ec, args, [], {}, "ex1", "mainchain", Path(tmpdir),
                              True, ["N-C"])


def test_plan_is_native_baseline_then_pulls(tmp_path):
    _, rows = _run(tmp_path)
    kinds = [r["rung_kind"] for r in rows]
    assert kinds[0] == "native"
    assert kinds[1] == "baseline", "the dG reference must be minimized at the native separation"
    assert set(kinds[2:]) == {"pull"}
    assert len(rows) == 2 + 3, "native + baseline + one row per ladder rung above d0"


def test_exactly_one_reference_of_each_kind(tmp_path):
    _, rows = _run(tmp_path)
    assert sum(r["is_native_rung"] for r in rows) == 1
    assert sum(r["is_baseline_rung"] for r in rows) == 1


def test_ladder_complete_marks_only_the_last_row(tmp_path):
    """This is the field resume keys on: if it never appears, every example is redone; if it
    appears early, a half-written ladder is treated as finished."""
    _, rows = _run(tmp_path)
    assert [r["ladder_complete"] for r in rows] == [False] * (len(rows) - 1) + [True]


def test_every_row_has_the_columns_its_consumers_read(tmp_path):
    """The Rosetta sidecar keys on pdb_tag/complex_pdb/cyc_type; the report reads rung_kind,
    limit_reason and ring_opened, and a guard that reads a missing column fails open."""
    _, rows = _run(tmp_path)
    for r in rows:
        for col in ("pdb_tag", "complex_pdb", "cyc_type", "example_id", "rung_kind",
                    "ring_opened", "terminal_ca_A", "ladder_complete", "contact_retention"):
            assert col in r, f"{col} missing from a {r['rung_kind']} row"
    native = next(r for r in rows if r["rung_kind"] == "native")
    assert native["limit_reason"] is None, "the unminimized rung has no limit to report"
    assert native["target_gap_A"] is None


def test_pull_rungs_only_go_upward(tmp_path):
    """A rung below the current separation would pull the termini together, which is the
    closure experiment, not this one."""
    _, rows = _run(tmp_path, ladder=(20.0, 30.0, 40.0))
    d0 = rows[0]["terminal_ca_A"]
    for r in rows:
        if r["rung_kind"] == "pull":
            assert r["target_gap_A"] > d0


def test_baseline_reports_a_limit_reason(tmp_path):
    """The baseline is minimized, so unlike native it must carry a limit_reason -- the report
    decides whether shortfalls are attributable by asking whether ANY row has one."""
    _, rows = _run(tmp_path)
    base = next(r for r in rows if r["rung_kind"] == "baseline")
    assert base["limit_reason"] in {"reached", "stalled", "rounds_exhausted"}


if __name__ == "__main__":
    import tempfile
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                print(f"  PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL {name}: {exc}")
    print("all passed" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
