"""Stage 2: score predicted complexes with the shared chemistry-aware evaluator.

Separate from prediction on purpose. Rescoring -- a tolerance change, a new metric, a
fixed bug in the evaluator -- re-reads structures from disk and never re-runs a
predictor. Every sample of every prediction is scored, not each model's favourite, so
a per-sample spread is recoverable later.

Resumable per SAMPLE, not per task: a shard that dies mid-candidate resumes at the
sample it was on. Rows go to one jsonl per shard, so there is exactly one writer per
file and no concurrent-append corruption.

Usage:
    python script_utils/cpseq_score.py --dataset-dir /data/user_data/yixiz/cpseq/tranche1 \
        --predictor boltz2 --shard-index 0 --shard-count 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import tolerances as tolmod  # noqa: E402
from proteinfoundation.cpseq.geometry import validity_report  # noqa: E402
from proteinfoundation.cpseq.scorer import score_structure  # noqa: E402

BINDER_CHAIN = "B"


def iter_prediction_records(dataset_dir: Path, predictor: str):
    root = dataset_dir / "predictions" / predictor
    for rec_path in sorted(root.rglob("record.json")):
        try:
            yield json.loads(rec_path.read_text())
        except Exception:  # noqa: BLE001, S112
            continue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tolerances", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tol_path = args.tolerances or (args.dataset_dir / "manifest" / "geometry_tolerances.json")
    tol_set = tolmod.load(tol_path)
    print(f"[score] tolerances from {tol_path}")

    records = list(iter_prediction_records(args.dataset_dir, args.predictor))
    records = records[args.shard_index :: args.shard_count]
    if args.limit:
        records = records[: args.limit]
    print(f"[score] {args.predictor}: {len(records)} prediction records "
          f"(shard {args.shard_index}/{args.shard_count})")

    out_dir = args.dataset_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"rows_{args.predictor}_{args.shard_index}.jsonl"

    # Already-scored (task_id, sample_index) pairs, so a resumed shard skips them.
    done: set[tuple[str, int]] = set()
    if rows_path.exists() and not args.force:
        for line in rows_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["sample_index"]))
            except Exception:  # noqa: BLE001, S112
                continue
    print(f"[score] {len(done)} sample(s) already scored in this shard")

    # The free-relaxed reference is cached within the process; its key includes the
    # starting coordinates, so a hit only ever reuses an identical calculation.
    deformation_cache: dict = {}
    n_scored = n_skipped = n_failed = 0

    for rec in records:
        chemistry = rec["chemistry"]
        tol = tol_set.get(chemistry)
        samples = rec.get("samples", [])

        if not samples:
            # A prediction that produced nothing still gets a row. An all-invalid or
            # unrun candidate needs an explicit outcome, not a silently dropped line.
            row = {
                "task_id": rec["task_id"],
                "candidate_id": rec["candidate_id"],
                "parent_id": rec["parent_id"],
                "family_id": rec.get("family_id"),
                "split": rec.get("split"),
                "chemistry": chemistry,
                "predictor": rec["predictor"],
                "sequence_requested": rec["sequence"],
                "n_mutations": rec.get("n_mutations"),
                "mutation_fraction": rec.get("mutation_fraction"),
                "is_reference": rec.get("is_reference"),
                "sample_index": -1,
                "prediction_status": rec.get("status"),
                "prediction_reason": rec.get("reason"),
                "valid": False,
                "validity_reasons": [f"no samples: {rec.get('status')}"],
                "scoring_status": "no_structure",
            }
            if (row["task_id"], -1) not in done:
                with open(rows_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            continue

        for s in samples:
            key = (rec["task_id"], s.get("sample_index", -1))
            if key in done:
                n_skipped += 1
                continue
            pdb = s.get("pdb_path")
            base = {
                "task_id": rec["task_id"],
                "candidate_id": rec["candidate_id"],
                "parent_id": rec["parent_id"],
                "family_id": rec.get("family_id"),
                "split": rec.get("split"),
                "chemistry": chemistry,
                "predictor": rec["predictor"],
                "predictor_revision": rec.get("predictor_revision"),
                "input_hash": rec.get("input_hash"),
                "config": rec.get("config"),
                "pocket_conditioning": rec.get("pocket_conditioning"),
                "sequence_requested": rec["sequence"],
                "peptide_length": rec.get("peptide_length"),
                "n_mutations": rec.get("n_mutations"),
                "mutation_fraction": rec.get("mutation_fraction"),
                "is_reference": rec.get("is_reference"),
                "sample_index": s.get("sample_index", -1),
                "sample_identity": {
                    k: s.get(k) for k in ("af2_model_num", "cyclic_offset_applied") if k in s
                },
                "prediction_status": s.get("status"),
                "prediction_runtime_s": s.get("runtime_s"),
                "raw_pdb_path": pdb,
                # Predictor-specific confidences, kept under their own key so they are
                # never pooled across predictors -- each model's convention differs.
                "predictor_confidence": {
                    k: s.get(k) for k in ("plddt", "ptm", "i_ptm", "i_pae") if k in s
                },
            }

            if s.get("status") != "ok" or not pdb or not Path(pdb).exists():
                base |= {
                    "valid": False,
                    "validity_reasons": [f"prediction status {s.get('status')}"],
                    "scoring_status": "no_structure",
                }
                with open(rows_path, "a") as fh:
                    fh.write(json.dumps(base) + "\n")
                n_failed += 1
                continue

            t0 = time.time()
            try:
                receptor_chains = sorted(
                    set(json.loads(rec["chain_map"]).values())
                ) if rec.get("chain_map") else ["A"]
                val = validity_report(
                    pdb,
                    binder_chain=BINDER_CHAIN,
                    receptor_chains=receptor_chains,
                    chemistry=chemistry,
                    requested_sequence=rec["sequence"],
                )
                verd = tolmod.verdict(val, tol) if tol else {
                    "valid": False, "reasons": [f"no tolerance for {chemistry}"], "checks": {}
                }

                stem = Path(pdb).with_suffix("")
                sc = score_structure(
                    pdb,
                    binder_chain=BINDER_CHAIN,
                    receptor_chains=receptor_chains,
                    chemistry=chemistry,
                    requested_sequence=rec["sequence"],
                    refined_out_path=f"{stem}_refined.pdb",
                    isolated_out_path=f"{stem}_isolated_relaxed.pdb",
                    deformation_cache=deformation_cache,
                )
                base |= {
                    "valid": bool(verd["valid"]),
                    "validity_checks": verd["checks"],
                    "validity_reasons": verd["reasons"],
                    "geometry": {
                        "closure": val["closure"],
                        "chirality": {
                            k: v for k, v in val["chirality"].items() if k != "per_residue"
                        },
                        "backbone": val["backbone"],
                        "clashes": {
                            k: v for k, v in val["clashes"].items() if "examples" not in k
                        },
                        "atoms": {
                            k: v for k, v in val["atoms"].items() if k != "missing_detail"
                        },
                        "sequence_matches": val["sequence_matches"],
                        "sequence_observed": val["sequence_observed"],
                    },
                    "rosetta": {
                        k: sc.get(k)
                        # `restraints` belongs here: the task requires recording what
                        # was restrained during refinement, and the scorer produces it,
                        # but this fixed key list was silently dropping it.
                        for k in ("status", "raw", "refined", "refinement_displacement",
                                  "deformation", "ring_declared_complex", "refined_pdb_path",
                                  "scorer_chemistry_status", "reason", "restraints")
                    },
                    "protocol": sc.get("protocol"),
                    "scoring_status": sc.get("status"),
                    "scoring_runtime_s": round(time.time() - t0, 2),
                }
                n_scored += 1
            except Exception as exc:  # noqa: BLE001
                base |= {
                    "valid": False,
                    "validity_reasons": [f"{type(exc).__name__}: {exc}"],
                    "scoring_status": "scorer_error",
                    "scoring_runtime_s": round(time.time() - t0, 2),
                }
                n_failed += 1

            with open(rows_path, "a") as fh:
                fh.write(json.dumps(base) + "\n")

            print(f"[score] {rec['candidate_id']} s{base['sample_index']} "
                  f"{chemistry}: valid={base['valid']} "
                  f"status={base['scoring_status']} "
                  f"dG={((base.get('rosetta') or {}).get('refined') or {}).get('dG_separated')} "
                  f"{base.get('scoring_runtime_s')}s")

    print(f"\n[score] scored={n_scored} failed={n_failed} skipped={n_skipped}")
    print(f"[score] rows -> {rows_path}")


if __name__ == "__main__":
    main()
