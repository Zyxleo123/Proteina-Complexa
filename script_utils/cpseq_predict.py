"""Run structure prediction for candidate sequences, one predictor at a time.

Stage 1 of two. This script only *predicts*; scoring is `cpseq_score.py`, so rescoring
never costs a re-prediction. Every task writes its own `record.json` and its structures,
and a task whose record already exists is skipped -- which is what makes the job
resumable under `preempt`'s requeue, and what lets the tranche grow by re-running the
same command with a larger `--limit`.

Task identity
-------------
`task_id` = sha256 of (candidate_id, predictor, chemistry, predictor revision, resolved
config). It changes when any of those change and not otherwise, so a task is addressable
across shards and runs, and a settings change cannot silently reuse an old structure.

Capability gating
-----------------
A (predictor, chemistry) pair that the capability table does not mark `supported` is
**not run**. It is written out as a record with `status: chemistry_unsupported` and the
table's own reason, because an absent row and a row that says "this arm cannot express
this molecule" are different facts and the summary has to tell them apart.

Usage:
    python script_utils/cpseq_predict.py --predictor boltz2 \
        --dataset-dir /data/user_data/yixiz/cpseq/tranche1 \
        --shard-index 0 --shard-count 1 --limit 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import chemistry as chem  # noqa: E402


def task_id_for(candidate_id: str, predictor: str, chemistry: str, cfg_hash: str) -> str:
    key = f"{candidate_id}|{predictor}|{chemistry}|{cfg_hash}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def build_task_list(dataset_dir: Path, predictor: str, cfg_hash: str) -> list[dict]:
    man = dataset_dir / "manifest"
    cands = pd.read_csv(man / "candidates.csv")
    targets = pd.read_csv(man / "targets.csv").set_index("parent_id")

    tasks = []
    for _, c in cands.iterrows():
        if c["parent_id"] not in targets.index:
            continue
        t = targets.loc[c["parent_id"]]
        cap = chem.capability(predictor, c["chemistry"])
        tasks.append(
            {
                "task_id": task_id_for(c["candidate_id"], predictor, c["chemistry"], cfg_hash),
                "candidate_id": c["candidate_id"],
                "parent_id": c["parent_id"],
                "chemistry": c["chemistry"],
                "sequence": c["sequence"],
                "peptide_length": int(c["peptide_length"]),
                "n_mutations": int(c["n_mutations"]),
                "mutation_fraction": float(c["mutation_fraction"]),
                "is_reference": bool(c["is_reference"]),
                "family_id": c["family_id"],
                "split": c["split"],
                "predictor": predictor,
                "capability_status": cap.status,
                "capability_note": cap.note,
                "target_chain_spec": t["target_chain_spec"],
                "target_template_pdb": t["target_template_path"],
                "target_template_cif": t["target_template_cif"],
                "pocket_contacts": t["pocket_contacts"],
                "receptor_msa_paths": t["receptor_msa_paths"],
                "chain_map": t["chain_map"],
            }
        )
    # Stable order: parent, then chemistry, then candidate. Sharding slices this, so the
    # order must not depend on dict iteration or file order.
    tasks.sort(key=lambda d: (d["parent_id"], d["chemistry"], d["candidate_id"]))
    return tasks


def environment_record() -> dict:
    return {
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0],
    }


def run_boltz2(task: dict, out_dir: Path, args) -> dict:
    from proteinfoundation.cpseq.adapters.boltz2 import (
        BOLTZ_REV,
        Boltz2Config,
        build_input_yaml,
        input_hash,
        predict,
    )
    from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure

    cfg = Boltz2Config(diffusion_samples=args.samples, seed=args.seed)

    # Receptor sequences are read from the TARGET-ONLY TEMPLATE, not from a parent
    # context. The first version resolved them by parsing LNR and indexing by parent_id,
    # which meant this adapter could only ever run on LNR: the experimental controls
    # failed all six with `KeyError: 'ctrl_7k2m'`. The template is the receptor the
    # predictor is actually given, it exists for every dataset built by
    # cpseq_prepare_targets / cpseq_build_control_contexts, and reading it here keeps the
    # YAML sequence and the template coordinates guaranteed consistent.
    #
    # Cached per template path: a shard of 100 tasks over 12 parents parses 12 files.
    if not hasattr(run_boltz2, "_tmpl_cache"):
        run_boltz2._tmpl_cache = {}
    tmpl = task["target_template_pdb"]
    if tmpl not in run_boltz2._tmpl_cache:
        parsed = parse_structure(Path(tmpl))
        run_boltz2._tmpl_cache[tmpl] = {
            ch: chain_sequence(parsed, ch) for ch in sorted(parsed["atoms"])
        }
    # Keyed by the PREDICTOR's chain ids (what the template uses), so pocket contacts
    # must be remapped from original ids to match.
    receptor_sequences = run_boltz2._tmpl_cache[tmpl]

    msa_paths = json.loads(task["receptor_msa_paths"])
    chain_map = json.loads(task["chain_map"])
    first_msa = next(
        (msa_paths[c] for c in sorted(receptor_sequences) if msa_paths.get(c)), None
    )
    pocket = [
        (chain_map.get(ch, ch), idx) for ch, idx in json.loads(task["pocket_contacts"])
    ]

    yaml_text = build_input_yaml(
        peptide_sequence=task["sequence"],
        receptor_sequences=receptor_sequences,
        chemistry=task["chemistry"],
        pocket_contacts=pocket,
        receptor_msa_path=first_msa,
        target_template_path=task["target_template_cif"],
    )
    yaml_path = out_dir / "input.yaml"
    yaml_path.write_text(yaml_text)

    result = predict(
        yaml_path=yaml_path,
        out_dir=out_dir / "boltz",
        config=cfg,
        python_exec=args.boltz_python,
        boltz_cache=args.boltz_cache,
        timeout_s=args.timeout_s,
    )

    # Collect whatever structures landed, and verify each one's peptide sequence.
    from proteinfoundation.cpseq.adapters.afcycdesign import sequence_from_pdb_when_ready

    samples = []
    for n, pdb in enumerate(sorted((out_dir / "boltz").rglob("*.pdb"))):
        got = sequence_from_pdb_when_ready(pdb, "B")
        samples.append(
            {
                "sample_index": n,
                "status": "ok",
                "pdb_path": str(pdb),
                "sequence_requested": task["sequence"],
                "sequence_predicted": got,
                "sequence_matches": got == task["sequence"],
            }
        )
    return {
        "predictor": "boltz2",
        "predictor_revision": BOLTZ_REV,
        "config": cfg.to_dict(),
        "input_hash": input_hash(yaml_text, cfg),
        "pocket_conditioning": "boltz_pocket_constraint_force_false",
        "samples": samples,
        **{k: v for k, v in result.items() if k != "samples"},
    }


def run_afcycdesign(task: dict, out_dir: Path, args) -> dict:
    from proteinfoundation.cpseq.adapters.afcycdesign import (
        AfCycConfig,
        input_hash,
        predict,
    )

    # Clamp to what the arm can actually deliver. Requesting 4 here would ask for
    # parameter sets that do not exist on the templated path (IndexError), and seeds are
    # not an alternative -- see the adapter's module docstring.
    from proteinfoundation.cpseq.adapters.afcycdesign import SAMPLE_CEILING

    n = min(args.samples, SAMPLE_CEILING)
    cfg = AfCycConfig(seed=args.seed, model_nums=tuple(range(n)))
    result = predict(
        sequence=task["sequence"],
        template_path=Path(task["target_template_pdb"]),
        target_chain=task["target_chain_spec"],
        chemistry=task["chemistry"],
        out_dir=out_dir / "af2",
        config=cfg,
        af2_params_dir=args.af2_params_dir,
    )
    return {
        "predictor": "afcycdesign",
        "predictor_revision": "colabdesign@" + str(args.colabdesign_rev),
        "config": cfg.to_dict(),
        "input_hash": input_hash(
            sequence=task["sequence"],
            template_path=task["target_template_pdb"],
            target_chain=task["target_chain_spec"],
            chemistry=task["chemistry"],
            config=cfg,
        ),
        **result,
    }


RUNNERS = {"boltz2": run_boltz2, "afcycdesign": run_afcycdesign}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--predictor", required=True, choices=sorted(chem.PREDICTORS))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (preflight uses 1-2)")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chemistries", default="", help="comma separated; default all present")
    ap.add_argument("--lnr-root", default="/data/user_data/yixiz/LNR")
    ap.add_argument("--boltz-python", default="/data/user_data/yixiz/envs/boltz2/.venv/bin/python")
    ap.add_argument("--boltz-cache", default="/data/user_data/yixiz/boltz_cache")
    ap.add_argument("--af2-params-dir", default=os.environ.get("AF2_DIR", ""))
    ap.add_argument("--colabdesign-rev", default="zenodo-13309081")
    ap.add_argument("--timeout-s", type=int, default=7200)
    ap.add_argument("--force", action="store_true", help="re-run tasks that already have a record")
    args = ap.parse_args()

    cfg_hash = hashlib.sha256(
        json.dumps(
            {"samples": args.samples, "seed": args.seed, "predictor": args.predictor},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]

    tasks = build_task_list(args.dataset_dir, args.predictor, cfg_hash)
    print(f"[predict] config hash {cfg_hash} is part of every output path")
    if args.chemistries:
        keep = {c.strip() for c in args.chemistries.split(",")}
        tasks = [t for t in tasks if t["chemistry"] in keep]
    tasks = tasks[args.shard_index :: args.shard_count]
    if args.limit:
        tasks = tasks[: args.limit]

    pred_root = args.dataset_dir / "predictions" / args.predictor
    rows_path = args.dataset_dir / "predictions" / f"rows_{args.predictor}_{args.shard_index}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[predict] {args.predictor}: {len(tasks)} tasks "
          f"(shard {args.shard_index}/{args.shard_count}, cfg {cfg_hash})")
    print(f"[predict] env: {environment_record()}")

    n_done = n_skipped = n_unsupported = n_failed = 0
    for k, task in enumerate(tasks):
        # The config hash is part of the PATH, not just the task_id. Without it a run at
        # a different sample count or seed finds the old record.json and skips, so the
        # dataset silently mixes settings -- two candidates scored at 2 samples and the
        # rest at 4, with nothing on disk saying so.
        out_dir = pred_root / task["parent_id"] / f"{task['candidate_id']}__{cfg_hash}"
        record_path = out_dir / "record.json"
        if record_path.exists() and not args.force:
            n_skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        if task["capability_status"] != chem.SUPPORTED:
            record = {
                **task,
                "status": "chemistry_unsupported",
                "reason": task["capability_note"],
                "samples": [],
                "environment": environment_record(),
            }
            n_unsupported += 1
        else:
            t0 = time.time()
            try:
                out = RUNNERS[args.predictor](task, out_dir, args)
                record = {**task, **out, "environment": environment_record()}
            except Exception as exc:  # noqa: BLE001
                record = {
                    **task,
                    "status": "runner_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "samples": [],
                    "environment": environment_record(),
                }
            record["wall_time_s"] = round(time.time() - t0, 2)
            if record.get("status") == "ok":
                n_done += 1
            else:
                n_failed += 1

        record_path.write_text(json.dumps(record, indent=2))
        with open(rows_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        n_ok = sum(1 for s in record.get("samples", []) if s.get("status") == "ok")
        n_match = sum(1 for s in record.get("samples", []) if s.get("sequence_matches"))
        print(f"[predict] {k + 1}/{len(tasks)} {task['candidate_id']} "
              f"{task['chemistry']}: {record.get('status')} "
              f"samples_ok={n_ok} seq_match={n_match} "
              f"{record.get('wall_time_s', 0)}s")

    print(f"\n[predict] done={n_done} failed={n_failed} unsupported={n_unsupported} "
          f"skipped_existing={n_skipped}")
    print(f"[predict] rows -> {rows_path}")


if __name__ == "__main__":
    main()
