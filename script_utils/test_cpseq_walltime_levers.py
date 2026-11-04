#!/usr/bin/env python3
"""Checks for the two scoring walltime levers: the relax skip, and the score pool.

Both levers cut measured work, so the only interesting failures are the ones that cut
MORE than intended and still look like a success. Each property below is the one whose
breakage would silently change a result rather than raise:

- **A skipped score record must not satisfy a full pass.** A record written under
  `--skip-invalid-relax` has no refined energy and no deformation. If the cache read
  accepted it, the fold would stay permanently unrelaxed while every later run reported
  a cache hit -- the same shape as the crashed-prediction cache bug that made 72 folds
  unrecoverable. So: refused for a full pass, accepted for another skipping pass.
- **The skip is gated on the protocol, not just the flag.** Under a postrelax-gated
  protocol (v4, v5) the pre-relax verdict is provisional, so skipping would decide a
  fold on a structure the protocol does not grade. The flag must be inert there.
- **`prediction_cached` must not call a missing record ready.** It is what keeps the
  score pool off the GPU; a false positive hands a worker a fold it will try to predict,
  and `GPU_LOCK` cannot serialise separate processes.

What is NOT tested here: that `score_structure(skip_relax=True)` keeps the raw analysis
and drops the refined one. That needs PyRosetta and a real complex, so it is measured on
real structures by `cpseq_score_cost_probe.py`, which asserts both and also that the
pre-relax verdict is identical with and without the skip.

No GPU, no PyRosetta, no network: the scorer and the predictor are both stubbed.

Usage:
    python script_utils/test_cpseq_walltime_levers.py
    pytest script_utils/test_cpseq_walltime_levers.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import evaluate as ev  # noqa: E402

# Schema-correct, and deliberately not the production numbers: `tolerances.load` reads
# the `tolerances` key and builds a `ChemistryTolerance`, so every required field has to
# be here or the fixture fails before any lever is exercised. The VALUES are irrelevant
# to these tests -- the verdict is stubbed -- but the SHAPE is not.
TOLERANCES = {
    "tolerances": {
        "mainchain": {
            "chemistry": "mainchain",
            "bond_length_min_A": 1.23,
            "bond_length_max_A": 1.43,
            "bond_length_source": "test fixture, not a calibrated gate",
            "angle_tol_deg": 7.54,
            "angle_tol_source": "test fixture",
            "torsion_tol_deg": 30.0,
            "torsion_tol_source": "test fixture",
            "n_controls": 4,
        }
    }
}


def _bed(tmp: Path) -> tuple[ev.TargetContext, ev.Protocol, ev.Store]:
    """A target, protocol and store wired to a temp dir. No real files are predicted."""
    tol_path = tmp / "geometry_tolerances.json"
    tol_path.write_text(json.dumps(TOLERANCES))
    target = ev.TargetContext(
        parent_id="test_parent",
        pdb_id="0xxx",
        reference_sequence="ACDEFGHIK",
        family_id="fam",
        split="test",
        receptor_sequences={"A": "MKVLAAGIVG"},
        receptor_chains=["A"],
        receptor_msa_path=None,
        pocket_contacts=[("A", 3)],
        target_template_cif=str(tmp / "tmpl.cif"),
        target_template_pdb=str(tmp / "tmpl.pdb"),
        chain_map={"A": "A"},
    )
    protocol = ev.Protocol(
        chemistry="mainchain", scoring_version="v3-cis-amide-muco-windows"
    )
    store = ev.Store(root=tmp / "run", tolerances_path=tol_path)
    return target, protocol, store


def _seed_prediction(store, target, protocol, seq: str, seed: int, status: str) -> str:
    """Write a prediction record the way the adapter would, and return its input hash."""
    cfg = protocol.boltz_config(seed)
    yaml_text = ev.build_input_yaml(
        peptide_sequence=seq,
        receptor_sequences=target.receptor_sequences,
        chemistry=protocol.chemistry,
        pocket_contacts=target.pocket_contacts,
        receptor_msa_path=target.receptor_msa_path,
        target_template_path=target.target_template_cif,
    )
    ihash = ev.input_hash(yaml_text, cfg)
    pdir = store.prediction_dir(target.parent_id, ihash)
    pdir.mkdir(parents=True, exist_ok=True)
    pdb = pdir / "model_0.pdb"
    pdb.write_text("ATOM      1  CA  ALA B   1       0.000   0.000   0.000\n")
    (pdir / "record.json").write_text(json.dumps({
        "input_hash": ihash,
        "predictor": "boltz2",
        "status": status,
        "samples": [{"sample_index": 0, "pdb_path": str(pdb)}],
    }))
    return ihash


# --------------------------------------------------------------------------------------
# prediction_cached: the gate that keeps the score pool off the GPU
# --------------------------------------------------------------------------------------


def test_prediction_cached_is_false_without_a_record() -> None:
    with tempfile.TemporaryDirectory() as td:
        target, protocol, store = _bed(Path(td))
        assert ev.prediction_cached(
            candidate_sequence="ACDEFGHIK", target=target, prediction_seed=1,
            protocol=protocol, store=store,
        ) is False


def test_prediction_cached_is_false_for_a_failed_record() -> None:
    """A cached failure is infrastructure and must be retried, never called ready.

    If this returned True the pool would hand the fold to a worker, which would read the
    failed record, decline it (`evaluate` only accepts status ok) and predict -- on the
    GPU, in a process GPU_LOCK cannot reach.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "timeout")
        assert ev.prediction_cached(
            candidate_sequence="ACDEFGHIK", target=target, prediction_seed=1,
            protocol=protocol, store=store,
        ) is False


def test_prediction_cached_is_true_for_an_ok_record() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "ok")
        assert ev.prediction_cached(
            candidate_sequence="ACDEFGHIK", target=target, prediction_seed=1,
            protocol=protocol, store=store,
        ) is True


def test_prediction_cached_is_seed_specific() -> None:
    """Seed is part of the boltz config and therefore of the hash.

    A gate that ignored it would call every seed of a sequence ready as soon as one was.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "ok")
        assert ev.prediction_cached(
            candidate_sequence="ACDEFGHIK", target=target, prediction_seed=2,
            protocol=protocol, store=store,
        ) is False


# --------------------------------------------------------------------------------------
# the skipped-record cache guard
# --------------------------------------------------------------------------------------


def _stub_scorer(monkeypatch, *, record_calls: list) -> None:
    """Replace the geometry and Rosetta calls with deterministic stubs.

    The structure is a one-atom file, so nothing real could be measured; what matters is
    which calls `evaluate` makes and what it writes to the score cache.
    """
    def fake_validity(pdb, **kw):
        return {"sequence_matches": True, "sequence_observed": kw["requested_sequence"],
                "closure": {"status": "ok", "closing_bond_A": 9.9},
                "clashes": {"n_inter": 0, "n_intra": 0}, "backbone": {"missing": 0}}

    def fake_verdict(validity, tol):
        # Always INVALID, which is the case the skip applies to.
        return {"valid": False, "reasons": ["closing bond 9.9 A outside window"],
                "checks": {"closure_in_window": False}}

    def fake_score(pdb, **kw):
        record_calls.append({"skip_relax": kw.get("skip_relax")})
        if kw.get("skip_relax"):
            return {"status": "ok_raw_only", "relax_skipped": True,
                    "raw": {"dG_separated": -3.0, "total_score": 10.0}}
        return {"status": "ok",
                "raw": {"dG_separated": -3.0, "total_score": 10.0},
                "refined": {"dG_separated": -7.0, "total_score": 5.0},
                "refinement_displacement": {"binder_ca_rmsd_A": 0.5},
                "deformation": {"status": "ok", "deformation_energy": 1.0}}

    monkeypatch.setattr(ev, "validity_report", fake_validity)
    monkeypatch.setattr(ev.tolmod, "verdict", fake_verdict)
    monkeypatch.setattr(ev, "score_structure", fake_score)
    monkeypatch.setattr(ev, "confidence_for_pdb", lambda p: {})
    monkeypatch.setattr(ev, "interface_pae_for_pdb", lambda p, **kw: {})
    monkeypatch.setattr(ev, "violation_magnitude", lambda v, t: 8.3)


def test_skipped_record_is_refused_by_a_full_pass(monkeypatch) -> None:
    """The property that keeps the lever from poisoning the cache.

    Pass 1 skips and writes an incomplete record. Pass 2 wants the relax: it must treat
    that record as a MISS and score properly, not read it as done.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "ok")
        calls: list = []
        _stub_scorer(monkeypatch, record_calls=calls)

        kw = dict(original_sequence="ACDEFGHIK", candidate_sequence="ACDEFGHIK",
                  target=target, prediction_seed=1, protocol=protocol, store=store)

        r1 = ev.evaluate(**kw, skip_invalid_relax=True)
        assert r1["status"] == "scored"
        assert r1["cache"]["score"] == "miss"
        assert calls[-1]["skip_relax"] is True, "pass 1 should have skipped the relax"

        # The record on disk marks itself.
        path = store.score_path(protocol.scoring_version, r1["input_hash"])
        assert json.loads(path.read_text())["relax_skipped"] is True

        # Pass 2, full: must NOT be a hit, and must call the scorer without the skip.
        r2 = ev.evaluate(**kw, skip_invalid_relax=False)
        assert r2["cache"]["score"] == "miss", "a skipped record must not satisfy a full pass"
        assert r2["cache"].get("score_recompute_over_skipped_record") is True
        assert calls[-1]["skip_relax"] is False, "pass 2 should have run the relax"
        assert json.loads(path.read_text())["relax_skipped"] is False

        # And now the full record IS reusable by both kinds of pass.
        r3 = ev.evaluate(**kw, skip_invalid_relax=True)
        assert r3["cache"]["score"] == "hit"


def test_skipped_record_satisfies_another_skipping_pass(monkeypatch) -> None:
    """Resume must not re-do work. Two skipping passes: the second is a hit."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "ok")
        calls: list = []
        _stub_scorer(monkeypatch, record_calls=calls)
        kw = dict(original_sequence="ACDEFGHIK", candidate_sequence="ACDEFGHIK",
                  target=target, prediction_seed=1, protocol=protocol, store=store)

        ev.evaluate(**kw, skip_invalid_relax=True)
        n = len(calls)
        r2 = ev.evaluate(**kw, skip_invalid_relax=True)
        assert r2["cache"]["score"] == "hit"
        assert len(calls) == n, "a skipping resume must not re-score"


def test_skip_is_inert_under_a_postrelax_protocol(monkeypatch) -> None:
    """v4/v5 grade the RELAXED complex, so the pre-relax verdict is provisional.

    The flag must be ignored there, or a fold would be decided on a structure the
    protocol does not grade. This is why the stage is checked and not only the flag --
    and it matters because `SCORING_PROTOCOL_VERSION` (the module default) IS postrelax.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        protocol = ev.Protocol(chemistry="mainchain",
                               scoring_version="v4-postrelax-validity")
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "ok")
        calls: list = []
        _stub_scorer(monkeypatch, record_calls=calls)

        ev.evaluate(
            original_sequence="ACDEFGHIK", candidate_sequence="ACDEFGHIK",
            target=target, prediction_seed=1, protocol=protocol, store=store,
            skip_invalid_relax=True,
        )
        assert calls[-1]["skip_relax"] is False, (
            "the skip must be inert under a postrelax-gated protocol even when asked for"
        )


def test_default_is_unchanged_behaviour(monkeypatch) -> None:
    """Not passing the flag must relax, exactly as every run so far has."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        target, protocol, store = _bed(tmp)
        _seed_prediction(store, target, protocol, "ACDEFGHIK", 1, "ok")
        calls: list = []
        _stub_scorer(monkeypatch, record_calls=calls)
        r = ev.evaluate(
            original_sequence="ACDEFGHIK", candidate_sequence="ACDEFGHIK",
            target=target, prediction_seed=1, protocol=protocol, store=store,
        )
        assert calls[-1]["skip_relax"] is False
        assert r["samples"][0]["relax_skipped"] is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
