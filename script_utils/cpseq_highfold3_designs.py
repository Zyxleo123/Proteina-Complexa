#!/usr/bin/env python3
"""Fold DESIGNED peptide sequences with HighFold3 -- the second predictor's opinion.

The searches in this project select a sequence because Boltz-2's pose of it scored well
under Rosetta. That selection cannot, on its own, distinguish a good peptide from one that
suits Boltz-2. This folds the same sequences with an independent predictor so the question
becomes answerable; `cpseq_highfold3_controls.py` is the gate HighFold3 had to pass before
its opinion counted, and it passed at 0.64-3.39 A receptor-aligned peptide CA RMSD on six
crystal macrocycles (afcycdesign, for contrast, sat 34.67 A away and was discarded).

What this does NOT do is score. Folding is GPU work and Rosetta is CPU work, and this
project keeps them in separate jobs; this writes structures plus `folds.jsonl` and stops.

Three things it is careful about.

**Both sides of the pair are folded in one run.** A searched peptide and the native one it
is compared against go through the same predictor, the same alignment, the same recycles
and the same seeds. Folding them in separate runs would leave the comparison carrying
whatever differed between the runs, and on this cohort seed spread alone is larger than the
effect being measured.

**Replicates come from the predictor's seeds, not from repeated identical inputs.**
`--folds N` sets `modelSeeds` to N distinct values in ONE invocation, which also avoids
paying AF3's startup cost N times -- 44 s of every 70 s prediction, measured.

**mmCIF is converted, never reinterpreted.** AF3 writes mmCIF, and this repo's scoring path
reads PDB with a fixed-column parser that does not reject an mmCIF file: it reads fields at
PDB column offsets and yields nonsense. The conversion goes through biotite, and the written
PDB is read back and checked against the sequence that was requested before it is recorded.

Usage:
    python script_utils/cpseq_highfold3_designs.py \\
        --dataset-dir   /data/user_data/yixiz/cpseq/tranche1 \\
        --seq-source    /data/user_data/yixiz/cpseq/transfer/selected_vs_native.jsonl \\
        --out-dir       /data/user_data/yixiz/cpseq/transfer/hf3 \\
        --model-dir     /data/user_data/yixiz/models/af3 \\
        --folds 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import seq_inject  # noqa: E402
from proteinfoundation.cpseq.adapters import highfold3 as hf3  # noqa: E402
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402
from proteinfoundation.cpseq.geometry import read_chain, sequence_of  # noqa: E402

HF3_PY = "/data/user_data/yixiz/envs/highfold3/.venv/bin/python"
HF3_RUN = "/data/user_data/yixiz/upstream/highfold3/run_alphafold.py"


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> dict:
    """Convert one AF3 mmCIF model to PDB, preserving chain ids. First model only.

    Returns the per-chain sequences read back FROM THE PDB, so the caller can verify the
    conversion against what it asked the predictor to fold rather than assuming it.
    """
    from biotite.structure import array as struc_array  # noqa: F401  (import check)
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    cif = CIFFile.read(str(cif_path))
    atoms = get_structure(cif, model=1)
    out = PDBFile()
    out.set_structure(atoms)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(pdb_path))

    chains = sorted({str(c) for c in atoms.chain_id})
    return {c: sequence_of(read_chain(pdb_path, c)) for c in chains}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="Prepared dataset holding manifest/targets.csv (receptors, MSAs, chain maps).")
    ap.add_argument("--seq-source", type=Path, required=True,
                    help="Sequences to fold: .json/.jsonl/.csv rows of parent_id + sequence + label. "
                         "SEVERAL rows per parent is the point -- put the searched peptide and the "
                         "native one in the same file so both are folded under identical settings.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--folds", type=int, default=8,
                    help="Distinct predictor seeds per sequence (default 8). These are the "
                         "replicates the paired comparison averages over; one fold per sequence is "
                         "a filter, not a measurement.")
    ap.add_argument("--seed", type=int, default=42, help="First model seed; folds are seed..seed+N-1.")
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument("--parents", default="", help="Comma-separated subset of parent_ids.")
    ap.add_argument("--chemistry-from", default=None,
                    help="A search run dir whose state/*/outcomes.jsonl records each parent's "
                         "chemistry. Needed for corpora with no predictions/ dir of their own, "
                         "such as pepbench_valid.")
    ap.add_argument("--chemistry", default=None,
                    help="Assert one chemistry for every requested parent, overriding anything "
                         "recorded. A wrong value rings the wrong bond, so it is announced.")
    ap.add_argument("--journal-tag", default=None,
                    help="Suffix for this process's journal file (folds_<tag>.jsonl). Defaults to "
                         "the Slurm array task id, then the job id, then the pid. Array tasks must "
                         "not share one journal; resume always reads every folds*.jsonl present.")
    ap.add_argument("--timeout-s", type=int, default=14400)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-predict even when a structure is cached.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.dataset_dir / "manifest" / "targets.csv").set_index("parent_id")

    try:
        requests = seq_inject.load_requests_multi(args.seq_source, args.dataset_dir)
    except seq_inject.SequenceInjectionError as exc:
        print(f"[hf3-designs] sequence source rejected: {exc}")
        return 1

    # The chemistry each parent was prepared under, never invented. Three sources, tried in
    # order, because they are not all present for every corpus: `pepbench_valid` has no
    # `predictions/` dir at all, so the controls driver's single source would silently skip
    # all ten of its parents with "no chemistry recorded".
    chem: dict[str, str] = {}
    chem_source = "none"
    for rec in (args.dataset_dir / "predictions").rglob("record.json"):
        r = json.loads(rec.read_text())
        if r.get("parent_id") and r.get("chemistry"):
            chem[r["parent_id"]] = r["chemistry"]
    if chem:
        chem_source = f"{args.dataset_dir}/predictions/*/record.json"
    if args.chemistry_from:
        found = 0
        for f in sorted(Path(args.chemistry_from).glob("state/*/outcomes.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid_, ch = r.get("parent_id"), r.get("chemistry")
                if pid_ and ch and pid_ not in chem:
                    chem[pid_] = ch
                    found += 1
        if found:
            chem_source = (chem_source + " + " if chem != {} and chem_source != "none" else "") \
                + f"{args.chemistry_from}/state/*/outcomes.jsonl"
    if args.chemistry:
        # Explicit override, applied last so a recorded value cannot mask it, and announced
        # loudly: asserting a cyclization the dataset never recorded is a claim, and a wrong
        # one rings the wrong bond.
        disagree = {p: c for p, c in chem.items() if p in requests and c != args.chemistry}
        for p in requests:
            chem[p] = args.chemistry
        chem_source = f"--chemistry {args.chemistry} (explicit override)"
        if disagree:
            print(f"[hf3-designs] WARNING --chemistry {args.chemistry} overrides recorded "
                  f"values for {len(disagree)} parent(s): {disagree}")
    print(f"[hf3-designs] chemistry source: {chem_source}")
    missing = sorted(p for p in requests if p not in chem)
    if missing:
        print(f"[hf3-designs] {len(missing)} parent(s) have no chemistry and will be skipped: "
              f"{missing[:8]}. Pass --chemistry-from <search run dir> or --chemistry.")

    wanted = {p.strip() for p in args.parents.split(",") if p.strip()}
    cfg = hf3.HighFold3Config(num_seeds=args.folds, num_recycles=args.num_recycles, seed=args.seed)

    n_pairs = sum(len(v) for k, v in requests.items() if not wanted or k in wanted)
    labels = sorted({r.label for v in requests.values() for r in v})
    print(f"[hf3-designs] {len(requests)} parents, {n_pairs} sequences, labels={labels}, "
          f"{args.folds} seeds each -> {n_pairs * args.folds} folds")
    for pid, reqs in sorted(requests.items()):
        if len(reqs) < 2 and not wanted:
            print(f"[hf3-designs] NOTE {pid} has only one sequence ({reqs[0].label}); it "
                  "contributes nothing to a paired comparison on its own")

    rows: list[dict] = []
    # One journal per process, read as a set. Array tasks appending to a single file would
    # interleave their lines; resume therefore reads every journal and writes only its own.
    tag = (args.journal_tag
           or os.environ.get("SLURM_ARRAY_TASK_ID")
           or os.environ.get("SLURM_JOB_ID")
           or f"pid{os.getpid()}")
    folds_path = args.out_dir / f"folds_{tag}.jsonl"
    done: set[str] = set()
    for jp in sorted(args.out_dir.glob("folds*.jsonl")):
        for line in jp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["fold_id"])
            except Exception:  # noqa: BLE001, S112
                continue
    if done:
        print(f"[hf3-designs] {len(done)} sequence(s) already recorded across "
              f"{len(list(args.out_dir.glob('folds*.jsonl')))} journal(s); they will be skipped")

    with folds_path.open("a") as fh:
        for pid, reqs in sorted(requests.items()):
            if wanted and pid not in wanted:
                continue
            if pid not in targets.index:
                print(f"[hf3-designs] {pid}: not in the dataset manifest; skipped")
                continue
            t = targets.loc[pid]
            chemistry = chem.get(pid)
            if chemistry is None:
                print(f"[hf3-designs] {pid}: no chemistry recorded; skipped")
                continue

            cmap = json.loads(t["chain_map"])
            # `target_template_path`, not `reference_complex_path`: the latter exists only in
            # the controls dataset, which this driver's ancestor was written against. The
            # target template is receptor-only by construction -- the manifest's
            # `peptide_chain_excluded_from_template` records which chain was dropped -- which
            # is exactly what the receptor sequences should come from, since templating the
            # peptide would answer a different question confidently.
            parsed = parse_structure(Path(t["target_template_path"]))
            # Read by the MAPPED chain id, not the original. `chain_map` is
            # {original -> predictor id} and the template is written with the chains already
            # renamed, so looking up the original id finds nothing whenever the two differ.
            # It differs for 5 of the 20 parents here (lnr_4j86, lnr_5y59, pepbench_B_H_pdb3wa0,
            # pepbench_C_A_pdb2jz3, pepbench_C_D_pdb4wjq, whose receptors are chain B or C);
            # the rest only worked because their original id was already 'A'. The key stays the
            # ORIGINAL id, which is what the MSA map and the provenance record are keyed on.
            rec_seqs = {orig: chain_sequence(parsed, cmap[orig]) for orig in cmap}
            mp = json.loads(t["receptor_msa_paths"])
            msas = {}
            for orig in cmap:
                p = mp.get(cmap[orig]) or mp.get(orig)
                if p:
                    msas[orig] = p

            # An a3m's first record is its query; if it disagrees with the chain being folded,
            # AF3 is handed an alignment for a different protein and runs anyway.
            bad_msa = False
            for orig, a3m in msas.items():
                lines = Path(a3m).read_text().splitlines()
                query = next((ln.strip() for ln in lines[1:] if ln and not ln.startswith(">")), "")
                if query and query != rec_seqs[orig]:
                    print(f"[hf3-designs] {pid}: MSA query != receptor chain {orig} "
                          f"({len(query)} vs {len(rec_seqs[orig])} aa); refusing this parent")
                    bad_msa = True
            if bad_msa or any(not v for v in rec_seqs.values()):
                continue

            for req in reqs:
                fold_id = f"{pid}|{req.label}|{req.requested_sequence}"
                if fold_id in done and not args.force:
                    print(f"[hf3-designs] {fold_id}: recorded; skipped")
                    continue

                payload = hf3.build_input_json(
                    name=f"{pid}_{req.label}",
                    peptide_sequence=req.requested_sequence,
                    receptor_sequences=rec_seqs,
                    receptor_msas=msas,
                    chemistry=chemistry,
                    config=cfg,
                )
                h = hf3.input_hash(payload, cfg)
                wdir = args.out_dir / pid / req.label / h
                wdir.mkdir(parents=True, exist_ok=True)
                jp = wdir / "input.json"
                jp.write_text(json.dumps(
                    {k: v for k, v in payload.items() if not k.startswith("_cpseq_")}, indent=2))
                (wdir / "cpseq_meta.json").write_text(json.dumps(
                    {k: v for k, v in payload.items() if k.startswith("_cpseq_")}
                    | {"chemistry": chemistry, "input_hash": h, "request": req.to_dict()}, indent=2))

                cmd = hf3.build_command(
                    json_path=jp, out_dir=wdir / "hf3", model_dir=args.model_dir,
                    chemistry=chemistry, peptide_length=len(req.requested_sequence),
                    python_exec=HF3_PY, run_script=HF3_RUN,
                )
                cmd.append("--norun_data_pipeline")

                print(f"[hf3-designs] {pid} {req.label} {req.requested_sequence} "
                      f"(-{req.n_substitutions}, {chemistry}) -> {wdir}")
                if args.dry_run:
                    print("      " + " ".join(cmd))
                    continue

                existing = sorted((wdir / "hf3").rglob("*.cif"))
                if existing and not args.force:
                    print(f"      cached: {len(existing)} cif")
                    proc, rt = None, 0.0
                else:
                    t0 = time.time()
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=args.timeout_s)
                    rt = round(time.time() - t0, 1)

                row = {
                    "fold_id": fold_id,
                    "parent_id": pid,
                    "pdb_id": str(t["pdb_id"]),
                    "label": req.label,
                    "requested_sequence": req.requested_sequence,
                    "native_sequence": req.native_sequence,
                    "n_substitutions": req.n_substitutions,
                    "is_native": req.is_native,
                    "chemistry": chemistry,
                    "peptide_length": len(req.requested_sequence),
                    "predictor": "highfold3",
                    "input_hash": h,
                    "n_seeds_requested": args.folds,
                    "seed_base": args.seed,
                    "num_recycles": args.num_recycles,
                    "runtime_s": rt,
                    "returncode": 0 if proc is None else proc.returncode,
                    "from_cache": proc is None,
                }
                if proc is not None and proc.returncode != 0:
                    row["status"] = "predictor_error"
                    row["stderr_tail"] = proc.stderr[-2500:]
                    print(f"      FAILED rc={proc.returncode} in {rt}s")
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    continue

                cifs = sorted((wdir / "hf3").rglob("*.cif"))
                if not cifs:
                    row["status"] = "no_structure"
                    print(f"      no structure written ({rt}s)")
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    continue

                # Convert every model, and verify each against the requested peptide. A
                # conversion that silently dropped or reordered chains would otherwise be
                # discovered as a strange energy, days later.
                pep_chain = payload["_cpseq_peptide_chain"]
                converted, bad = [], []
                for cif in cifs:
                    pdb = wdir / "pdb" / (cif.stem + ".pdb")
                    try:
                        seqs = cif_to_pdb(cif, pdb)
                    except Exception as exc:  # noqa: BLE001
                        bad.append({"cif": str(cif), "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    got = seqs.get(pep_chain, "")
                    if got != req.requested_sequence:
                        bad.append({"cif": str(cif), "peptide_chain": pep_chain,
                                    "read_back": got, "requested": req.requested_sequence})
                        continue
                    converted.append({"cif": str(cif), "pdb": str(pdb),
                                      "binder_chain": pep_chain,
                                      "receptor_chains": sorted(c for c in seqs if c != pep_chain)})
                row["n_models"] = len(cifs)
                row["structures"] = converted
                row["conversion_failures"] = bad
                row["status"] = "ok" if converted and not bad else (
                    "partial_conversion" if converted else "conversion_failed")
                if bad:
                    print(f"      WARNING {len(bad)}/{len(cifs)} model(s) failed conversion or "
                          f"sequence read-back")
                print(f"      {len(converted)}/{len(cifs)} model(s) converted ({rt}s)")
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)

    if args.dry_run:
        print("[hf3-designs] dry-run: nothing predicted.")
        return 0

    ok = [r for r in rows if r.get("status") == "ok"]
    n_struct = sum(len(r.get("structures", [])) for r in rows)
    print(f"\n[hf3-designs] {len(ok)}/{len(rows)} sequences ok, {n_struct} structures converted")
    print(f"[hf3-designs] folds -> {folds_path}")
    if rows:
        per_label: dict[str, int] = {}
        for r in rows:
            if r.get("status") == "ok":
                per_label[r["label"]] = per_label.get(r["label"], 0) + len(r.get("structures", []))
        print(f"[hf3-designs] structures per label: {per_label}")
        unpaired = [
            pid for pid, reqs in requests.items()
            if len({r["label"] for r in rows if r["parent_id"] == pid and r.get("status") == "ok"}) < 2
        ]
        if unpaired:
            print(f"[hf3-designs] {len(unpaired)} parent(s) lack both sides and cannot enter a "
                  f"paired comparison: {sorted(unpaired)[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
