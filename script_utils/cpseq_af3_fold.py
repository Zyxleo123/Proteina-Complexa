#!/usr/bin/env python3
"""Fold cyclic-peptide complexes with AF3, with the peptide as a SMILES ligand.

The GPU half of the AF3 arm. Scoring is a separate CPU job (`cpseq_af3_score.py`) for the
reason the rest of this project splits them: a protocol change then costs CPU and not
GPU-hours.

Alignments come from `targets.fetch_receptor_msa`, which shells out to **boltz's own**
mmseqs2 client. That is not a convenience -- it is what makes an AF3 number comparable to
a Boltz-2 one. Both predictors then see byte-identical alignments and differ as folders
rather than as pipelines. A cache hit does no network work, so re-running is free.

All seeds for one complex go into ONE AF3 invocation. AF3 runs every `modelSeeds` entry
against one loaded, compiled model, so N seeds in one call is far cheaper than N calls --
and the compilation cache makes the second complex cheaper still.

Controls mode (`--controls`) folds the deposited crystal macrocycles. That is the
calibration this arm needs before any design is folded: these are complexes whose answer
is known, so a pose that misses here is the instrument and not the design. They predate
the predictors' training data and may sit inside it, so recovery here establishes that the
pipeline is wired correctly and says nothing about prospective accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.adapters.af3 import (  # noqa: E402
    AF3_REV,
    DEFAULT_REPRESENTATION,
    REPRESENTATIONS,
    AF3Config,
    AF3InputError,
    ReceptorChain,
    build_input_json,
    describe_output,
    input_hash,
    predict,
)
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402
from proteinfoundation.cpseq.targets import fetch_receptor_msa  # noqa: E402


def load_control_jobs(controls_path: Path, chemistry: str) -> list[dict]:
    jobs = []
    for control in json.loads(controls_path.read_text()):
        if control["chemistry"] != chemistry:
            # Not a silent skip: a disulfide ring is a different molecule and this arm
            # cannot express it as a head-to-tail SMILES.
            print(f"[fold] skipping {control['pdb_id']}: chemistry "
                  f"{control['chemistry']} not expressible in this arm")
            continue
        parsed = parse_structure(control["structure_path"])
        jobs.append({
            "label": control["pdb_id"].upper(),
            "sequence": control["sequence"],
            "receptors": [
                {"chain_id": cid, "sequence": chain_sequence(parsed, cid)}
                for cid in control["receptor_chains"]
            ],
            "native_path": control["structure_path"],
            "native_peptide_chain": control["peptide_chain"],
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--controls", type=Path,
                     help="controls.json -- fold the deposited crystal macrocycles")
    src.add_argument("--manifest", type=Path,
                     help="JSON list of {label, sequence, receptors:[{chain_id,sequence}]}")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chemistry", default="mainchain")
    ap.add_argument("--seeds", default="101,202,303,404,505",
                    help="comma-separated model seeds, all folded in ONE AF3 call")
    ap.add_argument("--num-diffusion-samples", type=int, default=1)
    ap.add_argument("--af3-python", required=True)
    ap.add_argument("--af3-run-script", required=True)
    ap.add_argument("--af3-model-dir", required=True)
    ap.add_argument("--boltz-python", required=True,
                    help="boltz's interpreter -- its mmseqs2 client produces the MSA, so "
                         "both predictors see the same alignment")
    ap.add_argument("--msa-cache", type=Path, required=True)
    ap.add_argument("--representation", default=DEFAULT_REPRESENTATION,
                    choices=list(REPRESENTATIONS),
                    help="how the peptide is handed to AF3. 'polymer' = protein chain "
                         "with the closure in bondedAtomPairs (L by construction, 1 token "
                         "per residue); 'ligand' = macrocyclic SMILES (measured NOT to "
                         "preserve stereochemistry: 18/20 folds kept D-residues even "
                         "after relax)")
    ap.add_argument("--single-sequence", action="store_true",
                    help="skip the MSA entirely and predict single-sequence. A weaker "
                         "prediction, and recorded on every row as such, so a poor pose "
                         "is never silently blamed on the model.")
    ap.add_argument("--jax-cache-dir", default=None)
    ap.add_argument("--only", default="",
                    help="comma-separated labels, for one-complex preflights and shards")
    ap.add_argument("--timeout-s", type=int, default=14400)
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    config = AF3Config(
        model_seeds=seeds,
        num_diffusion_samples=args.num_diffusion_samples,
        jax_compilation_cache_dir=args.jax_cache_dir,
    )
    jobs = (load_control_jobs(args.controls, args.chemistry) if args.controls
            else json.loads(args.manifest.read_text()))
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        jobs = [j for j in jobs if j["label"].upper() in want]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=== cpseq_af3_fold ===")
    print(f"af3 revision   {AF3_REV}")
    print(f"seeds          {list(seeds)} x {args.num_diffusion_samples} sample(s) "
          f"= {config.folds_per_prediction} folds per complex")
    print(f"alignment      {'SINGLE-SEQUENCE (no MSA)' if args.single_sequence else 'boltz mmseqs2, cached'}")
    print(f"peptide as     {args.representation}"
          + ("  (closure in bondedAtomPairs; L by construction)" if args.representation == "polymer"
             else "  (closure in the SMILES ring bond)"))
    print(f"complexes      {len(jobs)}: {[j['label'] for j in jobs]}")
    print(f"out            {args.out_dir}")

    rows = []
    for job in jobs:
        label = job["label"]
        t0 = time.time()
        row: dict = {"label": label, "sequence": job["sequence"],
                     "chemistry": args.chemistry, "af3_rev": AF3_REV,
                     "seeds": list(seeds),
                     "num_diffusion_samples": args.num_diffusion_samples,
                     "single_sequence": bool(args.single_sequence),
                     "representation": args.representation}

        receptors = []
        msa_info = []
        for rec in job["receptors"]:
            if args.single_sequence:
                receptors.append(ReceptorChain(chain_id=rec["chain_id"],
                                               sequence=rec["sequence"],
                                               unpaired_msa=""))
                msa_info.append({"chain": rec["chain_id"], "status": "single_sequence"})
                continue
            got = fetch_receptor_msa(rec["sequence"], args.msa_cache, args.boltz_python)
            # `fetch_receptor_msa` returns exactly one of `cache_hit`, `fetched` or
            # `msa_failed`. Listing the SUCCESS statuses was wrong once already -- it
            # omitted `fetched`, so four alignments that had just been downloaded
            # successfully were thrown away and reported as failures. Gate on the
            # failure status instead, so a new success status cannot be read as an error.
            if got.get("status") == "msa_failed" or not got.get("msa_path"):
                print(f"[fold] {label}: MSA failed for chain {rec['chain_id']}: {got}")
                msa_info.append({"chain": rec["chain_id"], **got})
                receptors = []
                break
            a3m = Path(got["msa_path"]).read_text()
            receptors.append(ReceptorChain(chain_id=rec["chain_id"],
                                           sequence=rec["sequence"],
                                           unpaired_msa=a3m))
            msa_info.append({"chain": rec["chain_id"], "status": got["status"],
                             "n_sequences": got.get("n_sequences")})
        row["msa"] = msa_info
        if not receptors:
            row |= {"status": "msa_failed", "wall_time_s": round(time.time() - t0, 2)}
            rows.append(row)
            continue

        try:
            json_text, peptide, budget = build_input_json(
                name=label.lower(), receptor_chains=tuple(receptors),
                sequence=job["sequence"], chemistry=args.chemistry, config=config,
                representation=args.representation,
            )
        except AF3InputError as exc:
            print(f"[fold] {label}: input refused -- {exc}")
            row |= {"status": "input_refused", "error": str(exc),
                    "wall_time_s": round(time.time() - t0, 2)}
            rows.append(row)
            continue

        ihash = input_hash(json_text, config)
        job_dir = args.out_dir / label
        job_dir.mkdir(parents=True, exist_ok=True)
        input_json = job_dir / "af3_input.json"
        input_json.write_text(json_text)
        run_dir = job_dir / "af3_out"

        row |= {"input_hash": ihash, "token_budget": budget,
                "smiles": peptide.smiles,
                "macrocycle_ring_size": peptide.macrocycle_ring_size,
                "native_path": job.get("native_path"),
                "native_peptide_chain": job.get("native_peptide_chain"),
                "receptor_chains": [r.chain_id for r in receptors],
                "input_json": str(input_json)}

        print(f"\n[fold] {label}: L={peptide.length}, "
              f"{budget['total_tokens']} tokens "
              f"({budget['receptor_tokens']} receptor + {budget['ligand_tokens']} ligand, "
              f"{budget['ligand_tokenisation_multiplier']}x as a ligand), "
              f"MSA depth {[m.get('n_sequences') for m in msa_info]}", flush=True)

        result = predict(
            input_json=input_json, out_dir=run_dir, config=config,
            python_exec=args.af3_python, run_script=args.af3_run_script,
            model_dir=args.af3_model_dir, timeout_s=args.timeout_s,
        )
        n_cif = len(list(run_dir.rglob("*_model.cif"))) if run_dir.exists() else 0
        row |= {"status": result["status"], "returncode": result["returncode"],
                "af3_runtime_s": result["runtime_s"], "n_model_cif": n_cif,
                "seconds_per_fold": (round(result["runtime_s"] / config.folds_per_prediction, 1)
                                     if result["status"] == "ok" else None),
                "run_dir": str(run_dir),
                "wall_time_s": round(time.time() - t0, 2)}
        if result["status"] != "ok":
            row["stderr_tail"] = result["stderr_tail"][-2000:]
            print(f"[fold] {label}: {result['status']} rc={result['returncode']}")
            print(result["stderr_tail"][-1500:])
        else:
            print(f"[fold] {label}: ok in {result['runtime_s']}s "
                  f"({row['seconds_per_fold']}s per fold), {n_cif} model.cif")
        (job_dir / "fold_record.json").write_text(json.dumps(
            row | {"output_inventory": describe_output(run_dir)}, indent=2))
        rows.append(row)

    out = args.out_dir / "folds.jsonl"
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"\n{'label':10s} {'status':14s} {'tokens':>7s} {'folds':>6s} "
          f"{'runtime':>9s} {'s/fold':>7s}")
    for row in rows:
        tb = row.get("token_budget") or {}
        print(f"{row['label']:10s} {str(row.get('status')):14s} "
              f"{tb.get('total_tokens', 0):7d} {row.get('n_model_cif', 0):6d} "
              f"{(row.get('af3_runtime_s') or 0):9.1f} "
              f"{(row.get('seconds_per_fold') or 0):7.1f}")
    print(f"\n[fold] rows -> {out}")
    print("[fold] score them on CPU with scripts/cpseq_af3_score.sbatch")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
