"""Tests for the common LP manifest builder.

The parts worth testing are the ones that decide numbers nobody will re-derive by hand:

``assign_split``
    The whole point of the file is that the parent/fragment leak is *unrepresentable*.
    That rests on two properties: assignment is a pure function of the family key, and
    ``heldout`` wins over ``val`` so widening validation cannot eat the families reserved
    for the CP->LP comparison. Both are asserted directly, and then end-to-end on a frame
    whose inherited split deliberately leaks.

``read_chains`` / ``extract_one``
    A residue is ``(number, insertion code)``. Keying on the number alone silently merges
    two residues into one, which shortens the sequence by one letter and would then be
    caught -- or not -- as a length mismatch somewhere far away. Also: non-canonical
    residues, absent backbone atoms, and a metadata length that disagrees with the
    structure all have to come back as a *recorded status*, never as a dropped row.

``build_panel``
    Must span lengths and use distinct families, or "32 complexes spanning lengths and
    receptor families" is eight copies of one receptor.

sampling weights
    The realised mixture is the thing an arm draws, and it is NOT the row ratio -- the
    pool is 7.5% PepBench by rows and the configured mixture is 24%. A per-row weight
    that looks plausible can still integrate to the wrong share, so the test integrates it.

Run:  .venv/bin/python script_utils/test_lpgen_manifest.py
      .venv/bin/python -m pytest -q script_utils/test_lpgen_manifest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpgen_manifest import (  # noqa: E402
    HELDOUT,
    assign_split,
    build_panel,
    extract_one,
    main,
    read_chains,
)


# ------------------------------------------------------------------ synthetic structures
def atom_line(serial: int, name: str, resname: str, chain: str, resseq: int,
              icode: str = " ", xyz=(0.0, 0.0, 0.0)) -> str:
    """One ATOM record in the fixed columns the parser reads."""
    return (
        f"ATOM  {serial:>5d} {name:<4s}{resname:>4s} {chain}{resseq:>4d}{icode}"
        f"   {xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00  0.00\n"
    )


def write_complex(path: Path, peptide: list, receptor_len: int = 3,
                  backbone=("N", "CA", "C", "O")) -> None:
    """Write a two-chain complex: chain A receptor, chain B the given peptide.

    ``peptide`` is a list of ``(resname, resseq, icode)``.

    Residues are spaced ~3.8 A along x, like a real extended chain. Stacking them all at
    the origin instead would put N(first) and C(last) 0 A apart, and every synthetic
    peptide would then register as mainchain-cyclic -- the fixture would be testing the
    fixture's own coordinates rather than the code.
    """
    serial, lines = 1, []
    for i in range(receptor_len):
        for nm in backbone:
            lines.append(atom_line(serial, nm, "GLY", "A", i, " ", (0.0, 20.0, i * 3.8)))
            serial += 1
    for idx, (resname, resseq, icode) in enumerate(peptide):
        for nm in backbone:
            lines.append(atom_line(serial, nm, resname, "B", resseq, icode,
                                   (idx * 3.8, 0.0, 0.0)))
            serial += 1
    path.write_text("".join(lines))


# ----------------------------------------------------------------------------- the tests
def test_insertion_codes_are_distinct_residues():
    """Two residues numbered 5 and 5A are two residues, not one."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ins.pdb"
        write_complex(p, [("ALA", 5, " "), ("CYS", 5, "A"), ("TRP", 6, " ")])
        parsed = read_chains(str(p))
        assert len(parsed["chains"]["B"]) == 3, parsed["chains"]["B"]

        rec = extract_one(("ins", str(p), "B", 3))
        assert rec["status"] == "ok", rec
        assert rec["peptide_seq"] == "ACW", rec["peptide_seq"]


def test_sequence_order_follows_the_file():
    """The sequence is the order residues appear, not sorted by name or number."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ord.pdb"
        write_complex(p, [("TRP", 9, " "), ("ALA", 1, " "), ("GLY", 4, " ")])
        rec = extract_one(("ord", str(p), "B", 3))
        assert rec["peptide_seq"] == "WAG", rec["peptide_seq"]


def test_failures_are_recorded_not_dropped():
    """Every unusable row comes back with a status and a reason."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        noncanon = td / "x.pdb"
        write_complex(noncanon, [("ALA", 0, " "), ("SEP", 1, " ")])
        r = extract_one(("x", str(noncanon), "B", 2))
        assert r["status"] == "noncanonical_residue" and r["n_noncanonical"] == 1, r
        # 'X' keeps length == residue count, so this is NOT also a length mismatch.
        assert len(r["peptide_seq"]) == 2, r

        nobb = td / "nobb.pdb"
        write_complex(nobb, [("ALA", 0, " ")], backbone=("N", "O"))  # no CA, no C
        r = extract_one(("nobb", str(nobb), "B", 1))
        assert r["status"] == "missing_backbone" and r["n_missing_backbone"] == 1, r

        mism = td / "mism.pdb"
        write_complex(mism, [("ALA", 0, " "), ("GLY", 1, " ")])
        r = extract_one(("mism", str(mism), "B", 7))  # metadata says 7, structure has 2
        assert r["status"] == "length_mismatch", r

        nopep = td / "nopep.pdb"
        write_complex(nopep, [("ALA", 0, " ")])
        r = extract_one(("nopep", str(nopep), "Z", 1))  # ask for a chain that is not there
        assert r["status"] == "no_peptide_chain", r

        r = extract_one(("gone", str(td / "absent.pdb"), "B", 5))
        assert r["status"] == "unreadable", r

        # Receptor chain ids are discovered, not assumed to be 'A'.
        ok = td / "ok.pdb"
        write_complex(ok, [("ALA", 0, " ")])
        r = extract_one(("ok", str(ok), "B", 1))
        assert r["status"] == "ok" and r["receptor_chain_ids"] == "A", r


def test_measured_cyclic_topology_is_detected():
    """A disulfide-stapled or mainchain-closed peptide is cyclic however it is labelled.

    The inherited metadata says ``linear`` / ``conect_pairs_kept=0`` for the entire LP
    pool, because the sources ship no CONECT records -- so the only way to know is to
    measure. 77 real rows are like this.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def write_with_coords(path, peptide, extra):
            """peptide: [(resname,resseq)]; extra: {(resseq,atom): xyz} overrides."""
            serial, lines = 1, []
            for i in range(3):
                lines.append(atom_line(serial, "CA", "GLY", "A", i)); serial += 1
            for resname, resseq in peptide:
                for nm in ("N", "CA", "C", "O"):
                    xyz = extra.get((resseq, nm), (float(resseq) * 10.0, 0.0, 0.0))
                    lines.append(atom_line(serial, nm, resname, "B", resseq, " ", xyz))
                    serial += 1
                if resname == "CYS":
                    xyz = extra.get((resseq, "SG"), (float(resseq) * 10.0, 5.0, 0.0))
                    lines.append(atom_line(serial, "SG", resname, "B", resseq, " ", xyz))
                    serial += 1
            path.write_text("".join(lines))

        # Two cysteines with SG 2.03 A apart: a real disulfide, excluded by default.
        p = td / "ss.pdb"
        write_with_coords(p, [("CYS", 0), ("ALA", 1), ("CYS", 2)],
                          {(0, "SG"): (0.0, 0.0, 0.0), (2, "SG"): (2.03, 0.0, 0.0)})
        r = extract_one(("ss", str(p), "B", 3))
        assert r["status"] == "crosslinked_disulfide", r
        assert r["n_disulfide"] == 1, r
        assert "@2.03" in r["disulfide_pairs"], r["disulfide_pairs"]

        # Same two cysteines far apart: a perfectly good linear peptide.
        p = td / "nos.pdb"
        write_with_coords(p, [("CYS", 0), ("ALA", 1), ("CYS", 2)],
                          {(0, "SG"): (0.0, 0.0, 0.0), (2, "SG"): (9.0, 0.0, 0.0)})
        r = extract_one(("nos", str(p), "B", 3))
        assert r["status"] == "ok" and r["n_disulfide"] == 0, r

        # Shorter than any real S-S: anomalous, and NOT counted as a disulfide.
        p = td / "anom.pdb"
        write_with_coords(p, [("CYS", 0), ("CYS", 1)],
                          {(0, "SG"): (0.0, 0.0, 0.0), (1, "SG"): (1.2, 0.0, 0.0)})
        r = extract_one(("anom", str(p), "B", 2))
        assert r["status"] == "anomalous_sg_contact", r
        assert r["n_disulfide"] == 0, r

        # Mainchain closed head-to-tail: N(first) bonded to C(last).
        p = td / "cyc.pdb"
        write_with_coords(p, [("ALA", 0), ("ALA", 1), ("ALA", 2)],
                          {(0, "N"): (0.0, 0.0, 0.0), (2, "C"): (1.33, 0.0, 0.0)})
        r = extract_one(("cyc", str(p), "B", 3))
        assert r["status"] == "mainchain_cyclic", r
        assert abs(r["mainchain_closure_A"] - 1.33) < 1e-6, r


def test_keep_crosslinked_readmits_and_records():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tr, va = _tiny_dataset(td, with_disulfide=True)
        for flag, expect_rows in ((False, 0), (True, 1)):
            out = td / f"out_{flag}"
            argv = ["lpgen_manifest.py", "--lp-train", str(tr), "--lp-val", str(va),
                    "--out-dir", str(out), "--workers", "2"]
            if flag:
                argv.append("--keep-crosslinked")
            old, sys.argv = sys.argv, argv
            try:
                assert main() == 0
            finally:
                sys.argv = old
            man = pd.read_parquet(out / "lpgen_manifest.parquet")
            summary = json.loads((out / "lpgen_manifest.summary.json").read_text())
            got = int((man.example_id == "pepbench__ss").sum())
            assert got == expect_rows, (flag, got)
            assert summary["topology"]["kept_crosslinked"] is flag
            if not flag:
                assert summary["topology"]["n_excluded_cyclic"] == 1, summary["topology"]
                exc = pd.read_csv(out / "lpgen_excluded.csv")
                assert (exc[exc.example_id == "pepbench__ss"].stage == "topology").all()
            else:
                assert summary["topology"]["accepted_with_disulfide"] == 1


def test_residue_key_collision_is_detected():
    """Two different residues sharing one (chain, resseq, icode) key must be rejected.

    Measured on the real pool: 1,806 of 45,540 structures (3.97%), all ProtFrag. atomworks
    refuses such a file outright, and -- worse -- this parser and preprocess_lp.py both key
    on the same tuple, so the collided group counts as ONE residue and peptide_length and
    peptide_seq are understated *in agreement*, which is exactly why the length check cannot
    see it. The atom-name SET alone also cannot see it, which is why the reader counts atoms.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # A TYR and a PRO both numbered 7 in chain A -- the shape of the real defect.
        p = td / "collide.pdb"
        serial, lines = 1, []
        for nm in ("N", "CA", "C", "O", "CB", "CG"):
            lines.append(atom_line(serial, nm, "TYR", "A", 7, " ", (0.0, 20.0, 0.0)))
            serial += 1
        for nm in ("N", "CA", "C", "O", "CB", "CG"):
            lines.append(atom_line(serial, nm, "PRO", "A", 7, " ", (3.0, 20.0, 0.0)))
            serial += 1
        for i in range(3):
            for nm in ("N", "CA", "C", "O"):
                lines.append(atom_line(serial, nm, "ALA", "B", i, " ", (i * 3.8, 0.0, 0.0)))
                serial += 1
        p.write_text("".join(lines))

        r = extract_one(("collide", str(p), "B", 3))
        assert r["status"] == "residue_key_collision", r
        assert r["n_collided_res_receptor"] == 1, r
        assert r["n_collided_res_peptide"] == 0, r

        # Same defect inside the PEPTIDE chain: these are the 359 rows whose sequence is wrong.
        p2 = td / "collide_pep.pdb"
        serial, lines = 1, []
        for i in range(4):
            for nm in ("N", "CA", "C", "O"):
                lines.append(atom_line(serial, nm, "GLY", "A", i, " ", (0.0, 20.0, i * 3.8)))
                serial += 1
        for nm in ("N", "CA", "C", "O"):
            lines.append(atom_line(serial, nm, "ALA", "B", 2, " ", (0.0, 0.0, 0.0)))
            serial += 1
        for nm in ("N", "CA", "C", "O"):
            lines.append(atom_line(serial, nm, "SER", "B", 2, " ", (3.8, 0.0, 0.0)))
            serial += 1
        p2.write_text("".join(lines))
        r2 = extract_one(("collide_pep", str(p2), "B", 2))
        assert r2["status"] == "residue_key_collision", r2
        assert r2["n_collided_res_peptide"] == 1, r2

        # A clean structure reports zero collisions and passes.
        p3 = td / "clean.pdb"
        write_complex(p3, [("ALA", i, " ") for i in range(6)])
        r3 = extract_one(("clean", str(p3), "B", 6))
        assert r3["status"] == "ok", r3
        assert r3["n_collided_res_peptide"] == 0 and r3["n_collided_res_receptor"] == 0


def test_assign_split_is_pure_and_heldout_wins():
    kw = dict(reserve_frac=0.15, reserve_seed=0, val_frac=0.10, val_seed=7)
    fams = [f"fam{i:04d}" for i in range(4000)]
    first = [assign_split(f, **kw) for f in fams]

    # Pure: same answer every call, independent of order.
    assert first == [assign_split(f, **kw) for f in reversed(fams)][::-1]
    assert set(first) == {"train", "val", HELDOUT}

    # Reserved families stay reserved even when validation is asked for everything.
    greedy = dict(kw, val_frac=1.0)
    for f in (f for f, s in zip(fams, first) if s == HELDOUT):
        assert assign_split(f, **greedy) == HELDOUT, f

    # The reserved share tracks reserve_frac (hash-uniform, so approximately).
    share = first.count(HELDOUT) / len(first)
    assert 0.12 < share < 0.18, share


def test_panel_spans_lengths_and_distinct_families():
    rows = []
    for L in (6, 9, 12, 16):
        for k in range(10):
            rows.append({
                "example_id": f"pepbench__L{L}_{k}", "cluster_id": f"c{L}_{k}",
                "family_id": f"c{L}_{k}", "peptide_length": L,
                "dataset_source": "pepbench",
            })
    # Decoys that must never be picked: a different source, and a family already used.
    rows.append({"example_id": "protfrag__z", "cluster_id": "cz", "family_id": "cz",
                 "peptide_length": 9, "dataset_source": "protfrag"})
    rows.append({"example_id": "pepbench__dup", "cluster_id": "c9_0", "family_id": "c9_0",
                 "peptide_length": 9, "dataset_source": "pepbench"})
    df = pd.DataFrame(rows)

    panel = build_panel(df, size=8, sources=["pepbench"])
    assert len(panel) == 8, len(panel)
    assert panel.family_id.nunique() == 8, "families must be distinct"
    assert set(panel.peptide_length) == {6, 9, 12, 16}, sorted(set(panel.peptide_length))
    assert not panel.example_id.str.startswith("protfrag").any()
    # Deterministic.
    assert list(panel.example_id) == list(build_panel(df, 8, ["pepbench"]).example_id)

    # Asking for more than the pool holds returns the pool, not an exception.
    assert len(build_panel(df, 10_000, ["pepbench"])) == 40
    assert build_panel(df, 8, ["nosuchsource"]).empty


# ------------------------------------------------------------------------- end to end
def _write_stapled(path: Path, n_res: int = 8) -> None:
    """A peptide whose first and last residues are cysteines bridged at 2.03 A.

    The real pool has 77 of these, all labelled ``linear`` by the inherited metadata.
    """
    serial, lines = 1, []
    for i in range(3):
        lines.append(atom_line(serial, "CA", "GLY", "A", i, " ", (0.0, 20.0, i * 3.8)))
        serial += 1
    for i in range(n_res):
        resname = "CYS" if i in (0, n_res - 1) else "ALA"
        for nm in ("N", "CA", "C", "O"):
            lines.append(atom_line(serial, nm, resname, "B", i, " ", (i * 3.8, 0.0, 0.0)))
            serial += 1
        if resname == "CYS":
            # The two SG atoms sit 2.03 A apart regardless of how far the CAs are.
            sg = (0.0, 6.0, 0.0) if i == 0 else (2.03, 6.0, 0.0)
            lines.append(atom_line(serial, "SG", resname, "B", i, " ", sg))
            serial += 1
    path.write_text("".join(lines))


def _tiny_dataset(td: Path, n_pepbench: int = 40, n_protfrag: int = 160,
                  with_disulfide: bool = False) -> tuple:
    """A dataset shaped like the real one: ProtFrag dominant by rows, and an inherited
    split that leaks one family across train and val."""
    proc = td / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(example_id, family, source, L, split):
        pdb = proc / f"{example_id}.pdb"
        write_complex(pdb, [("ALA", i, " ") for i in range(L)])
        rows.append({
            "example_id": example_id, "path": str(pdb), "binder_chain_id": "B",
            "cluster_id": family, "split": split, "peptide_length": L,
            "receptor_length": 3, "cyclization_type": "linear",
            "source_path": f"/raw/{example_id}.pdb", "conect_pairs_kept": 0,
            "dataset_source": source,
        })

    for i in range(n_pepbench):
        add(f"pepbench__p{i}", f"fam{i:03d}", "pepbench", 5 + (i % 12), "train")
    for i in range(n_protfrag):
        add(f"protfrag__f{i}", f"fam{i % 60:03d}", "protfrag", 5 + (i % 12), "train")

    # The leak, reproduced: a val complex whose family is also in train as fragments.
    val_rows = []
    for i in range(12):
        fam = f"fam{i:03d}" if i < 3 else f"valfam{i:03d}"  # first 3 leak
        pdb = proc / f"pepbench__v{i}.pdb"
        L = 6 + i
        write_complex(pdb, [("ALA", k, " ") for k in range(L)])
        val_rows.append({
            "example_id": f"pepbench__v{i}", "path": str(pdb), "binder_chain_id": "B",
            "cluster_id": fam, "split": "val", "peptide_length": L,
            "receptor_length": 3, "cyclization_type": "linear",
            "source_path": f"/raw/v{i}.pdb", "conect_pairs_kept": 0,
            "dataset_source": "pepbench",
        })

    # One genuinely cyclic row, to prove the linear filter runs and records.
    pdb = proc / "pepbench__cyc.pdb"
    write_complex(pdb, [("ALA", k, " ") for k in range(8)])
    rows.append({
        "example_id": "pepbench__cyc", "path": str(pdb), "binder_chain_id": "B",
        "cluster_id": "cycfam", "split": "train", "peptide_length": 8,
        "receptor_length": 3, "cyclization_type": "head_tail",
        "source_path": "/raw/cyc.pdb", "conect_pairs_kept": 1,
        "dataset_source": "pepbench",
    })

    if with_disulfide:
        pdb = proc / "pepbench__ss.pdb"
        _write_stapled(pdb)
        rows.append({
            "example_id": "pepbench__ss", "path": str(pdb), "binder_chain_id": "B",
            "cluster_id": "ssfam", "split": "train", "peptide_length": 8,
            "receptor_length": 3, "cyclization_type": "linear",
            "source_path": "/raw/ss.pdb", "conect_pairs_kept": 0,
            "dataset_source": "pepbench",
        })

    tr_path, va_path = td / "lp_train.parquet", td / "lp_val.parquet"
    pd.DataFrame(rows).to_parquet(tr_path, index=False)
    pd.DataFrame(val_rows).to_parquet(va_path, index=False)
    return tr_path, va_path


def test_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tr, va = _tiny_dataset(td)
        out = td / "out"
        argv = [
            "lpgen_manifest.py", "--lp-train", str(tr), "--lp-val", str(va),
            "--out-dir", str(out), "--workers", "2", "--panel-size", "6",
            "--val-frac", "0.30",
        ]
        old, sys.argv = sys.argv, argv
        try:
            assert main() == 0
        finally:
            sys.argv = old

        man = pd.read_parquet(out / "lpgen_manifest.parquet")
        summary = json.loads((out / "lpgen_manifest.summary.json").read_text())
        excluded = pd.read_csv(out / "lpgen_excluded.csv")

        # The cyclic row is gone AND recorded.
        assert "pepbench__cyc" not in set(man.example_id)
        assert "pepbench__cyc" in set(excluded.example_id)
        assert summary["excluded_by_stage"]["linear_filter"] == 1

        # Sequences came back, one letter per residue.
        assert (man.peptide_seq.str.len() == man.peptide_length).all()
        assert (man.peptide_seq.str.fullmatch("A+")).all()
        assert (man.receptor_chain_ids == "A").all()
        assert "peptide_chain_id" in man.columns

        # AtomWorks pn_unit_iid columns: these, not bare chain ids, are what
        # GenericDFParser needs. Bare ids compose and load, then die in the spatial crop.
        assert (man.receptor_pn_unit_iid == "A_1").all(), man.receptor_pn_unit_iid.unique()
        assert (man.peptide_pn_unit_iid == "B_1").all(), man.peptide_pn_unit_iid.unique()

        # rfd3_example_id must parse under AtomWorks' own regex, which demands exactly four
        # brace-delimited fields and eval()s the first and last. RFD3's logging callbacks
        # raise on anything else, in on_train_batch_start -- i.e. only after the model and
        # the data path are already working.
        import re as _re
        assert man.rfd3_example_id.is_unique
        for rid, src, eid in zip(man.rfd3_example_id, man.dataset_source, man.example_id):
            m = _re.findall(r"\{(.*?)\}", rid)
            assert len(m) == 4, (rid, m)
            assert eval(m[0]) == ["lpgen", src], m[0]
            assert m[1] == eid, (m[1], eid)
            assert m[2] == "1"
            assert eval(m[3]) == ["A_1", "B_1"], m[3]

        # Families are disjoint across all three splits.
        fams = {s: set(g.family_id) for s, g in man.groupby("split")}
        for a in fams:
            for b in fams:
                if a < b:
                    assert not (fams[a] & fams[b]), (a, b, fams[a] & fams[b])

        # The inherited leak was real in the input and is repaired in the output.
        assert summary["inherited_split_leak_repaired"]["n_families"] == 3, summary[
            "inherited_split_leak_repaired"]
        for fam in summary["inherited_split_leak_repaired"]["families"]:
            assert man[man.family_id == fam].split.nunique() == 1

        # Weights: only train rows carry one, and they INTEGRATE to the requested mixture
        # (0.24/0.76), not to the row ratio (which is ~0.2/0.8 here and 0.075/0.925 real).
        assert (man[man.split != "train"].sample_weight == 0).all()
        realised = summary["source_fractions"]["realised_by_weight"]
        assert abs(realised["pepbench"] - 0.24) < 1e-6, realised
        assert abs(realised["protfrag"] - 0.76) < 1e-6, realised
        assert "lpmix" in summary["source_fractions"]["source"]

        # Panel: requested size, distinct families, from val, PepBench only.
        panel = pd.read_parquet(out / "lpgen_val_panel.parquet")
        assert len(panel) == 6, len(panel)
        assert panel.family_id.nunique() == 6
        assert (panel.dataset_source == "pepbench").all()
        assert set(panel.example_id) <= set(man[man.split == "val"].example_id)
        assert len(set(panel.peptide_length)) > 1, "panel must span lengths"


def test_smoke_limit_is_stratified_by_source():
    """--limit must keep both sources.

    example_ids sort with every ``pepbench__*`` before every ``protfrag__*``, so a plain
    alphabetical head would hand the smoke run zero ProtFrag rows -- and the smoke would
    then exercise neither the mixture weights nor the parent/fragment families where the
    inherited leak lives. This is the bug the first real smoke run actually hit.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tr, va = _tiny_dataset(td)
        out = td / "outlim"
        argv = [
            "lpgen_manifest.py", "--lp-train", str(tr), "--lp-val", str(va),
            "--out-dir", str(out), "--workers", "2", "--limit", "60",
        ]
        old, sys.argv = sys.argv, argv
        try:
            assert main() == 0
        finally:
            sys.argv = old

        man = pd.read_parquet(out / "lpgen_manifest.parquet")
        summary = json.loads((out / "lpgen_manifest.summary.json").read_text())
        assert summary["smoke_limit"] == 60
        assert len(man) <= 60, len(man)
        # Both sources present, and roughly in pool proportion rather than all-pepbench.
        got = set(man.dataset_source)
        assert got == {"pepbench", "protfrag"}, got
        assert (man.dataset_source == "protfrag").sum() > (man.dataset_source == "pepbench").sum()


def test_pepbench_frac_override_changes_the_realised_mixture():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tr, va = _tiny_dataset(td)
        out = td / "out80"
        argv = [
            "lpgen_manifest.py", "--lp-train", str(tr), "--lp-val", str(va),
            "--out-dir", str(out), "--workers", "2", "--pepbench-frac", "0.8",
        ]
        old, sys.argv = sys.argv, argv
        try:
            assert main() == 0
        finally:
            sys.argv = old
        summary = json.loads((out / "lpgen_manifest.summary.json").read_text())
        realised = summary["source_fractions"]["realised_by_weight"]
        assert abs(realised["pepbench"] - 0.8) < 1e-6, realised
        assert summary["source_fractions"]["source"] == "cli --pepbench-frac"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
