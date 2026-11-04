#!/usr/bin/env python3
"""Fold cyclic-peptide complexes with HighFold3 (AF3 + a cyclic position encoding).

Why this exists rather than reusing the AF3 driver: stock AF3 has no working mechanism
for a head-to-tail macrocycle, measured on 40 folds. As a SMILES ligand the ring closes
(1.372 A) but AF3 does not honour the stated stereochemistry -- 18/20 folds kept
D-residues even after relax. As a polymer chain with the closure in `bondedAtomPairs` the
chirality is right and the ring never closes -- 0/20, gaps 2.15-16.64 A, with `OXT` built
on every C-terminus, i.e. the declared bond never entered the chemical graph. HighFold3
patches the relative position encoding instead, which is the only approach of the three
that addresses the actual gap.

**Closure is measured on every fold, here, at fold time.** That is deliberate. Both AF3
representations were accepted at the input layer and then produced the wrong chemistry,
and in the polymer case the defect only surfaced downstream in the energies. Worse, under
`v5` scoring an open ring is FORCED shut by the relax restraint and then certified valid
(12/19 on folds that were 0/20 closed, with energies from 0.0 to +49,500 REU). So the
closure distance is recorded in the fold log, before any scoring can launder it.

Run as a JOB. Never in an interactive allocation: processes adopted into a node-hold
job's extern step get a memory ceiling below what a folder needs, and the failure is a
bare SIGKILL with no traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.adapters.af3 import read_model_cif  # noqa: E402
from proteinfoundation.cpseq.adapters.highfold3 import (  # noqa: E402
    HIGHFOLD3_REV,
    HighFold3Config,
    build_command,
    build_input_json,
)
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402
from proteinfoundation.cpseq.targets import fetch_receptor_msa  # noqa: E402

# The project's fold seeds. Used instead of the adapter's consecutive default so a
# HighFold3 fold is comparable seed-for-seed with the Boltz-2 arm on the same cohort.
LADDER_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 111)

# Keys the adapter carries for provenance that are NOT in AF3's schema. AF3's
# `_validate_keys` rejects the whole input if they are present -- verified against the
# pinned install: "Unexpected JSON keys in: _cpseq_peptide_chain". They are stripped here
# and kept in this driver's own record instead, so the provenance survives without
# breaking the parser. The adapter had never been run, so this was latent.
NON_SCHEMA_PREFIX = "_cpseq_"

AMIDE_WINDOW_A = (1.23, 1.43)


def closure_distance(cif: Path, peptide_chain: str) -> dict:
    """N of the first residue to C of the last, in the peptide chain. Measured, not assumed."""
    atoms = [a for a in read_model_cif(cif) if a.auth_asym_id == peptide_chain]
    if not atoms:
        return {"status": "no_peptide_chain",
                "chains": sorted({a.auth_asym_id for a in read_model_cif(cif)})}
    residues: dict[int, dict[str, object]] = {}
    for atom in atoms:
        try:
            residues.setdefault(int(atom.auth_seq_id), {})[atom.atom_name] = atom
        except ValueError:
            continue
    if not residues:
        return {"status": "no_numbered_residues"}
    lo, hi = min(residues), max(residues)
    first, last = residues[lo], residues[hi]
    if "N" not in first or "C" not in last:
        return {"status": "missing_backbone_atoms", "first": sorted(first),
                "last": sorted(last)}
    n, c = first["N"], last["C"]
    d = math.dist((n.x, n.y, n.z), (c.x, c.y, c.z))  # type: ignore[attr-defined]
    return {
        "status": "ok",
        "n_residues": len(residues),
        "closure_bond_A": round(d, 3),
        "in_amide_window": AMIDE_WINDOW_A[0] <= d <= AMIDE_WINDOW_A[1],
        # OXT on the last residue means the chain was built with a FREE carboxyl
        # terminus, i.e. the cyclisation was not expressed in the chemical graph. It is
        # the tell that caught the polymer-representation failure.
        "oxt_on_last_residue": "OXT" in last,
    }


def load_control_jobs(controls_path: Path, chemistry: str) -> list[dict]:
    jobs = []
    for control in json.loads(controls_path.read_text()):
        if control["chemistry"] != chemistry:
            print(f"[hf3] skipping {control['pdb_id']}: chemistry "
                  f"{control['chemistry']} != {chemistry}")
            continue
        parsed = parse_structure(control["structure_path"])
        jobs.append({
            "label": control["pdb_id"].upper(),
            "sequence": control["sequence"],
            "receptors": {cid: chain_sequence(parsed, cid)
                          for cid in control["receptor_chains"]},
            "native_path": control["structure_path"],
            "native_peptide_chain": control["peptide_chain"],
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--controls", type=Path,
                     help="controls.json -- the calibration gate")
    src.add_argument("--manifest", type=Path,
                     help="JSON list of {label, sequence, receptors:{chain:seq}}")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chemistry", default="mainchain")
    ap.add_argument("--seeds", default=",".join(str(s) for s in LADDER_SEEDS[:5]))
    ap.add_argument("--hf3-python", required=True)
    ap.add_argument("--hf3-run-script", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--boltz-python", required=True,
                    help="boltz's interpreter; its mmseqs2 client builds the MSA, so "
                         "HighFold3 sees the same alignment as the Boltz-2 arm")
    ap.add_argument("--msa-cache", type=Path, required=True)
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="repeatable. Appended verbatim to the HighFold3 command line. "
                         "Exists because compute-capability 7.x GPUs (V100/T4) need "
                         "--flash_attention_implementation=xla, which AF3 refuses to run "
                         "without and the adapter's build_command does not emit. Kept as "
                         "a passthrough rather than hardcoded: the right value depends on "
                         "the GPU a job lands on, and this pool is mixed.")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout-s", type=int, default=14400)
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    jobs = (load_control_jobs(args.controls, args.chemistry) if args.controls
            else json.loads(args.manifest.read_text()))
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        jobs = [j for j in jobs if j["label"].upper() in want]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== cpseq_hf3_fold ===")
    print(f"highfold3 rev  {HIGHFOLD3_REV}")
    print(f"seeds          {list(seeds)}")
    print(f"chemistry      {args.chemistry}")
    print(f"complexes      {len(jobs)}: {[j['label'] for j in jobs]}")

    rows = []
    for job in jobs:
        label = job["label"]
        t0 = time.time()
        row: dict = {"label": label, "sequence": job["sequence"],
                     "chemistry": args.chemistry, "highfold3_rev": HIGHFOLD3_REV,
                     "seeds": list(seeds), "predictor": "highfold3"}

        msa_paths, msa_info = {}, []
        for cid, seq in job["receptors"].items():
            got = fetch_receptor_msa(seq, args.msa_cache, args.boltz_python)
            if got.get("status") == "msa_failed" or not got.get("msa_path"):
                row |= {"status": "msa_failed", "msa": got}
                break
            msa_paths[cid] = got["msa_path"]
            msa_info.append({"chain": cid, "status": got["status"],
                             "n_sequences": got.get("n_sequences")})
        if row.get("status") == "msa_failed":
            print(f"[hf3] {label}: MSA failed")
            rows.append(row)
            continue
        row["msa"] = msa_info

        config = HighFold3Config(num_seeds=len(seeds))
        payload = build_input_json(
            name=label.lower(), peptide_sequence=job["sequence"],
            receptor_sequences=job["receptors"], receptor_msas=msa_paths,
            chemistry=args.chemistry, config=config,
        )
        # Keep the provenance, then strip it: AF3 refuses unknown top-level keys.
        row["chain_map"] = payload.get(f"{NON_SCHEMA_PREFIX}chain_map")
        peptide_chain = payload.get(f"{NON_SCHEMA_PREFIX}peptide_chain", "A")
        row["peptide_chain"] = peptide_chain
        payload = {k: v for k, v in payload.items()
                   if not k.startswith(NON_SCHEMA_PREFIX)}
        # The project's seeds, not the adapter's consecutive default, so folds line up
        # seed-for-seed with the Boltz-2 arm.
        payload["modelSeeds"] = list(seeds)

        job_dir = args.out_dir / label
        job_dir.mkdir(parents=True, exist_ok=True)
        input_json = job_dir / "hf3_input.json"
        input_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        run_dir = job_dir / "hf3_out"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(
            json_path=input_json, out_dir=run_dir, model_dir=args.model_dir,
            chemistry=args.chemistry, peptide_length=len(job["sequence"]),
            python_exec=args.hf3_python, run_script=args.hf3_run_script,
        )
        # The data pipeline must NOT run. AF3's genetic databases are ~1 TB and are
        # deliberately not installed here (docs/old_docs/RESOURCES_COLLECTED.md); the
        # alignments travel inline instead, from the same boltz mmseqs2 client the
        # Boltz-2 arm uses. Without this flag `run_alphafold.py` eagerly resolves every
        # database path at startup -- guarded by `if _RUN_DATA_PIPELINE.value:` -- and
        # dies with FileNotFoundError on bfd-first_non_consensus_sequences.fasta before
        # it ever looks at the input. The adapter's `build_command` does not emit it,
        # which is why this is added here; it is a project-wide invariant, not a
        # per-GPU choice like the flags that arrive via --extra-arg.
        cmd = [*cmd, "--norun_data_pipeline", *args.extra_arg]
        row["cmd"] = " ".join(str(c) for c in cmd)
        row["extra_args"] = list(args.extra_arg)
        print(f"\n[hf3] {label}: L={len(job['sequence'])}, "
              f"receptor {[len(s) for s in job['receptors'].values()]} res, "
              f"MSA depth {[m.get('n_sequences') for m in msa_info]}", flush=True)
        print(f"[hf3] {row['cmd']}", flush=True)

        try:
            proc = subprocess.run([str(c) for c in cmd], capture_output=True,
                                  text=True, timeout=args.timeout_s)
            status = "ok" if proc.returncode == 0 else "predictor_error"
            tail_out, tail_err = proc.stdout[-4000:], proc.stderr[-4000:]
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            status, rc, tail_out, tail_err = "timeout", None, "", f"timeout {args.timeout_s}s"

        # `*model.cif`, not `*_model.cif`: AF3 3.0.1 (the revision HighFold3 pins)
        # names each per-fold file `model.cif`, while 131ab9e7 names it
        # `<job>_seed-N_sample-K_model.cif`. The narrower glob matched only the
        # top-level copy of the top-ranked sample, whose parent is the job directory
        # rather than a `seed-*` one, so the filter below then discarded it and the run
        # reported 0 folds after folding 25 structures successfully.
        cifs = sorted(run_dir.rglob("*model.cif"))
        per_fold = []
        for cif in cifs:
            if "seed-" not in cif.parent.name:
                continue   # the top-level copy of the top-ranked sample; would double-count
            geo = closure_distance(cif, peptide_chain)
            geo["fold"] = cif.parent.name
            geo["cif"] = str(cif)
            per_fold.append(geo)

        closed = [g for g in per_fold if g.get("in_amide_window")]
        row |= {
            "status": status, "returncode": rc,
            "runtime_s": round(time.time() - t0, 2),
            "n_folds": len(per_fold),
            "n_closed": len(closed),
            "closure": per_fold,
            "run_dir": str(run_dir),
            "seconds_per_fold": (round((time.time() - t0) / len(per_fold), 1)
                                 if per_fold else None),
        }
        if status != "ok":
            row["stderr_tail"] = tail_err
            print(f"[hf3] {label}: {status} rc={rc}")
            print(tail_err[-1500:])
        else:
            gaps = [g["closure_bond_A"] for g in per_fold if "closure_bond_A" in g]
            oxt = sum(1 for g in per_fold if g.get("oxt_on_last_residue"))
            print(f"[hf3] {label}: ok in {row['runtime_s']}s "
                  f"({row['seconds_per_fold']}s/fold), {len(per_fold)} folds")
            if gaps:
                print(f"[hf3] {label}: CLOSURE {len(closed)}/{len(gaps)} in "
                      f"{AMIDE_WINDOW_A[0]}-{AMIDE_WINDOW_A[1]} A | "
                      f"gaps min {min(gaps):.2f} median "
                      f"{sorted(gaps)[len(gaps)//2]:.2f} max {max(gaps):.2f} A | "
                      f"OXT on {oxt}/{len(per_fold)}")
        (job_dir / "fold_record.json").write_text(json.dumps(row, indent=2))
        rows.append(row)

    out = args.out_dir / "folds.jsonl"
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    total = sum(r.get("n_folds", 0) for r in rows)
    closed = sum(r.get("n_closed", 0) for r in rows)
    print(f"\n{'complex':10s} {'status':14s} {'folds':>6s} {'closed':>7s} {'s/fold':>7s}")
    for row in rows:
        print(f"{row['label']:10s} {str(row.get('status')):14s} "
              f"{row.get('n_folds', 0):6d} {row.get('n_closed', 0):7d} "
              f"{(row.get('seconds_per_fold') or 0):7.1f}")
    print(f"\n[hf3] CLOSURE: {closed}/{total} folds inside the amide window")
    if total and closed == 0:
        print("[hf3] NO fold closed. HighFold3 is not expressing the macrocycle here, so")
        print("[hf3] the designs must NOT be folded with it -- and note that v5 scoring")
        print("[hf3] would certify these anyway by forcing the ring shut during relax.")
    print(f"[hf3] rows -> {out}")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
