"""Tests for the native-layout exporter.

What is worth testing here is the handful of things that would be invisible in a log line
and wrong in training:

the 10 A whole-residue crop
    A residue is in the pocket if ANY of its heavy atoms is inside the cutoff, and then
    ALL of its atoms are written. Partial residues would still parse and would quietly
    train on broken sidechains. Hydrogens must not drag a residue in. The cutoff is
    inclusive at exactly 10.0 A, which is where PepMerge's own boundary sits.

the directory name
    POTFlow recovers the peptide chain with ``data['id'][-1]`` -- the last character of the
    directory name. If it does not end in the peptide chain letter, POTFlow reads the wrong
    chain and nothing errors.

verbatim atom lines
    Everything but the serial number is passed through. Coordinates and b-factors must come
    back bit-identical, because a reformatting writer is the usual way this goes wrong.

the listing/export agreement
    POTFlow's dataset IS ``os.listdir(structure_dir)``, so a stale directory left by an
    earlier larger run silently enlarges the training set. The exporter must refuse.

Run:  .venv/bin/python script_utils/test_lpgen_export_native.py
      .venv/bin/python -m pytest -q script_utils/test_lpgen_export_native.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpgen_export_native import export_one, main, read_atoms, write_pdb  # noqa: E402


def atom(serial, name, resname, chain, resseq, xyz, element=None, b=0.0):
    """One ATOM record on the real column spec.

    The element belongs in columns 77-78. Writing it one column late leaves the element
    field empty, and a parser then guesses it from the atom name -- which is a fixture bug
    that would otherwise look like an exporter bug.
    """
    el = element if element is not None else name[0]
    return (
        f"ATOM  {serial:>5d} {name:<4s}{resname:>4s} {chain}{resseq:>4d} "
        f"   {xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00{b:>6.2f}"
        f"          {el:>2s}\n"
    )


def write_case(path: Path, rec_offsets: list, pep_n: int = 3) -> None:
    """Receptor residues at the given x offsets; a 3-residue peptide at the origin."""
    serial, lines = 1, []
    for i, dx in enumerate(rec_offsets):
        for nm in ("N", "CA", "C", "O"):
            lines.append(atom(serial, nm, "GLY", "A", i, (dx, 0.0, 0.0))); serial += 1
    for i in range(pep_n):
        for nm in ("N", "CA", "C", "O"):
            lines.append(atom(serial, nm, "ALA", "B", i, (0.0, float(i) * 1.0, 0.0)))
            serial += 1
    path.write_text("".join(lines))


def _run(path, out, cutoff=10.0, aliases=False, chain="B"):
    return export_one(("ex1", str(path), chain, "train", str(out), cutoff, aliases))


def test_pocket_cutoff_is_inclusive_at_10A():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "c.pdb"
        # Straddle the boundary. Enough near residues to clear MIN_POCKET_RESIDUES, so
        # this test measures the cutoff and not the minimum-pocket gate.
        write_case(p, [1.0, 2.0, 3.0, 5.0, 9.99, 10.0, 10.01, 15.0])
        r = _run(p, td / "out")
        assert r["status"] == "ok", r
        # Inclusive at exactly 10.0; 10.01 and 15.0 are out.
        assert r["n_pocket_res"] == 6, r

        poc = read_atoms(str(td / "out" / "train" / "ex1_B" / "pocket.pdb"))
        xs = sorted(float(v[0][30:38]) for v in poc["res"].values())
        assert xs == [1.0, 2.0, 3.0, 5.0, 9.99, 10.0], xs


def test_whole_residues_and_hydrogens_do_not_pull_a_residue_in():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "c.pdb"
        lines = []
        serial = 1
        # Residue 0: one heavy atom just inside, the rest far away. Must be kept WHOLE.
        lines.append(atom(serial, "N", "GLY", "A", 0, (9.5, 0.0, 0.0))); serial += 1
        lines.append(atom(serial, "CA", "GLY", "A", 0, (40.0, 0.0, 0.0))); serial += 1
        lines.append(atom(serial, "C", "GLY", "A", 0, (41.0, 0.0, 0.0))); serial += 1
        # Residue 1: heavy atoms all far, but a HYDROGEN inside the cutoff. Must stay out.
        lines.append(atom(serial, "N", "GLY", "A", 1, (30.0, 0.0, 0.0))); serial += 1
        lines.append(atom(serial, "H", "GLY", "A", 1, (1.0, 0.0, 0.0), element="H"))
        serial += 1
        # Filler residues well inside, so the min-pocket gate does not mask the result.
        for j in (2, 3, 4):
            for nm in ("N", "CA"):
                lines.append(atom(serial, nm, "GLY", "A", j, (4.0, 0.0, 0.0))); serial += 1
        for i in range(3):
            for nm in ("N", "CA", "C", "O"):
                lines.append(atom(serial, nm, "ALA", "B", i, (0.0, float(i), 0.0)))
                serial += 1
        p.write_text("".join(lines))

        r = _run(p, td / "out")
        assert r["status"] == "ok", r
        # Residue 0 (one heavy atom inside) plus the 3 fillers; residue 1 is hydrogen-only
        # inside the cutoff and must NOT be pulled in.
        assert r["n_pocket_res"] == 4, r
        poc = read_atoms(str(td / "out" / "train" / "ex1_B" / "pocket.pdb"))
        resseqs = sorted(int(k[1][:4]) for k in poc["res"])
        assert resseqs == [0, 2, 3, 4], resseqs
        # The kept residue brought all three of its atoms, not just the near one.
        res0 = [v for k, v in poc["res"].items() if int(k[1][:4]) == 0][0]
        assert len(res0) == 3, len(res0)


def test_dir_name_ends_in_peptide_chain_id():
    """POTFlow reads the chain as data['id'][-1]."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "c.pdb"
        write_case(p, [5.0, 6.0, 7.0, 8.0])
        r = _run(p, td / "out", chain="B")
        assert r["dir_name"].endswith("B"), r["dir_name"]
        assert r["dir_name"][-1] == "B"
        assert (td / "out" / "train" / r["dir_name"] / "peptide.pdb").exists()


def test_atom_lines_are_verbatim_apart_from_serial():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "c.pdb"
        serial, lines = 1, []
        for i in range(5):
            for nm in ("N", "CA", "C", "O"):
                lines.append(atom(serial, nm, "GLY", "A", i, (5.0, 0.0, 0.0), b=37.42))
                serial += 1
        for i in range(3):
            for nm in ("N", "CA", "C", "O"):
                lines.append(atom(serial, nm, "ALA", "B", i, (0.0, float(i), 0.0), b=11.11))
                serial += 1
        src.write_text("".join(lines))

        _run(src, td / "out")
        d = td / "out" / "train" / "ex1_B"
        for fn in ("peptide.pdb", "pocket.pdb"):
            for line in open(d / fn):
                if not line.startswith("ATOM"):
                    continue
                # Everything from column 12 on is untouched, so b-factor and element survive.
                assert line[54:60] == "  1.00", repr(line[54:60])
                assert line[60:66].strip() in {"37.42", "11.11"}, repr(line[60:66])
        # Serials are renumbered from 1 and contiguous.
        serials = [int(l[6:11]) for l in open(d / "pocket.pdb") if l.startswith("ATOM")]
        assert serials == list(range(1, len(serials) + 1)), serials[:5]


def test_ter_and_end_records_are_spec_conformant():
    """A parser matches line[0:6] against "TER   ", so a bare "TER\\n" is unrecognized.

    This is not cosmetic: POTFlow runs mkdssp on peptide.pdb, DSSP is strict about chain
    termination, and upstream swallows any DSSP failure in a bare `except:` that silently
    relabels the whole peptide as helix.
    """
    import warnings as _w

    from Bio.PDB import PDBParser

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "c.pdb"
        write_case(p, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert _run(p, td / "out")["status"] == "ok"
        d = td / "out" / "train" / "ex1_B"

        for fn in ("peptide.pdb", "pocket.pdb"):
            lines = (d / fn).read_text().splitlines()
            ter = [l for l in lines if l.startswith("TER")]
            end = [l for l in lines if l.startswith("END")]
            assert len(ter) == 1 and len(end) == 1, (fn, lines[-3:])
            # Record names occupy exactly columns 1-6.
            assert ter[0][0:6] == "TER   ", repr(ter[0][0:6])
            assert end[0][0:6] == "END   ", repr(end[0][0:6])
            # TER carries the terminated chain, which is what DSSP reads.
            assert ter[0][21] in ("A", "B"), repr(ter[0][21])
            assert ter[0][17:20].strip() in ("GLY", "ALA"), repr(ter[0][17:20])

            # And no parser warnings at all now.
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                PDBParser(QUIET=False).get_structure("x", d / fn)
            assert not caught, [str(c.message) for c in caught]


def test_failures_are_recorded():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Receptor present but entirely beyond the cutoff -> not a bound complex.
        p = td / "far.pdb"
        write_case(p, [60.0, 61.0, 62.0, 63.0])
        r = _run(p, td / "out")
        assert r["status"] == "pocket_too_small", r
        assert not (td / "out" / "train" / "ex1_B").exists(), "must not write a bad complex"

        p2 = td / "nopep.pdb"
        write_case(p2, [5.0], pep_n=0)
        assert _run(p2, td / "out")["status"] == "no_peptide_chain"

        assert _run(td / "absent.pdb", td / "out")["status"] == "unreadable"


def _manifest(td: Path, n_train: int = 6, n_val: int = 3) -> Path:
    proc = td / "proc"; proc.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, n in (("train", n_train), ("val", n_val), ("heldout", 2)):
        for i in range(n):
            eid = f"{split}__e{i}"
            p = proc / f"{eid}.pdb"
            write_case(p, [5.0, 6.0, 7.0, 8.0, 30.0])
            rows.append({"example_id": eid, "path": str(p), "peptide_chain_id": "B",
                         "split": split, "dataset_source": "pepbench",
                         "family_id": f"f{i}", "peptide_length": 3})
    mp = td / "man.parquet"
    pd.DataFrame(rows).to_parquet(mp, index=False)
    return mp


def test_end_to_end_per_split_dirs_and_name_lists():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mp = _manifest(td)
        out = td / "native"
        argv = ["lpgen_export_native.py", "--manifest", str(mp), "--out-dir", str(out),
                "--workers", "2", "--ppflow-aliases"]
        old, sys.argv = sys.argv, argv
        try:
            assert main() == 0
        finally:
            sys.argv = old

        # heldout is reserved and must NOT be exported by default.
        assert not (out / "heldout").exists()
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["train", "val"]
        assert len(list((out / "train").iterdir())) == 6
        assert len(list((out / "val").iterdir())) == 3

        # One dir per complex, with both arms' filenames.
        d = out / "train" / "train__e0_B"
        for fn in ("peptide.pdb", "pocket.pdb",
                   "peptide_repaired.pdb", "receptor_repaired.pdb"):
            assert (d / fn).exists(), fn
        # The 30 A residue is excluded; 4 of 5 receptor residues are in the pocket.
        assert len(read_atoms(str(d / "pocket.pdb"))["res"]) == 4

        # PPFlow's native membership list is torch-loadable and matches the listing.
        names = torch.load(out / "train_names.pt")
        assert sorted(names) == sorted(p.name for p in (out / "train").iterdir())

        # POTFlow's blocklist exists and is empty by design.
        assert (out / "names.txt").read_text() == ""

        s = json.loads((out / "lpgen_export.summary.json").read_text())
        assert s["splits"]["train"]["exported"] == 6
        assert s["foldx_repair_applied"] is False
        assert s["pocket_cutoff_A"] == 10.0


def test_stale_directory_is_refused():
    """POTFlow's dataset is the directory listing, so leftovers must not pass silently."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mp = _manifest(td)
        out = td / "native"
        base = ["lpgen_export_native.py", "--manifest", str(mp), "--out-dir", str(out),
                "--workers", "2"]

        old, sys.argv = sys.argv, base
        try:
            assert main() == 0
        finally:
            sys.argv = old

        # A leftover from an imagined earlier, larger run.
        (out / "train" / "stale__e99_B").mkdir()
        old, sys.argv = sys.argv, base
        try:
            raised = False
            try:
                main()
            except SystemExit as exc:
                raised = "disagrees with the export" in str(exc)
            assert raised, "a stale directory must be refused"
        finally:
            sys.argv = old

        # --clean makes it pass again.
        old, sys.argv = sys.argv, base + ["--clean"]
        try:
            assert main() == 0
        finally:
            sys.argv = old
        assert not (out / "train" / "stale__e99_B").exists()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
