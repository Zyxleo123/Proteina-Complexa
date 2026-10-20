"""Score the sequence-conditioned generator's output with the SAME evaluator as the predictors.

Why this exists separately from `cpseq_score.py`: the generator does not go through
`cpseq_predict.py`. It is driven by the repo's existing `eval_val_gen_testset.py`, which
dumps complexes as `step_*/manifest_rank*.jsonl` + PDBs. This reads that dump and runs the
identical geometry / tolerance / Rosetta path, writing rows in the identical schema, so
its `dG_separated` sits next to Boltz-2's on the same targets rather than next to a number
produced by a different protocol.

Two things it is careful about.

**Attribution is derived from the structure, not from batch order.** The dump rows carry
`bid` (index within a batch) and no `example_id`. Trusting that index would mean trusting
the val dataloader's ordering, and a wrong mapping silently attributes a sample to the
wrong target -- which would look like a plausible result. Instead each dumped complex is
matched to its parent by its **receptor sequence**, which is unique across the twelve
parents, so attribution is checkable from the file alone.

**Sequence compliance gates the energy.** Boltz and AfCycDesign are handed a sequence and
return it by construction. This model *decodes* sequence from latents, so it can quietly
design its own peptide; the config's own notes say `cond_recovery` is the metric that
separates "honoured the sequence" from "generated something else that closes nicely".
A sample whose decoded sequence differs from the requested one is recorded with its
identity and recovery fraction, and its dG is reported but flagged -- a binding energy for
a peptide you did not ask for is not an answer to the question.

Usage:
    python script_utils/cpseq_score_seqcond.py \
        --dump-dir /data/user_data/yixiz/cpseq/seqcond/gen/rosetta_dump \
        --dataset-dir /data/user_data/yixiz/cpseq/tranche1 \
        --predictor-name cpsea_seqcond_v2 \
        --ckpt-step 3000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import tolerances as tolmod  # noqa: E402
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402
from proteinfoundation.cpseq.geometry import read_chain, sequence_of, validity_report  # noqa: E402
from proteinfoundation.cpseq.scorer import score_structure  # noqa: E402


def parent_index(dataset_dir: Path) -> dict[str, dict]:
    """{receptor sequence -> parent info}, for structure-based attribution."""
    targets = pd.read_csv(dataset_dir / "manifest" / "targets.csv")
    cands = pd.read_csv(dataset_dir / "manifest" / "candidates.csv")
    refs = cands[(cands.chemistry == "mainchain") & (cands.is_reference)]
    out: dict[str, dict] = {}
    for _, t in targets.iterrows():
        parsed = parse_structure(Path(t["target_template_path"]))
        seq = "".join(chain_sequence(parsed, ch) for ch in sorted(parsed["atoms"]))
        row = refs[refs.parent_id == t["parent_id"]]
        out[seq] = {
            "parent_id": t["parent_id"],
            "pdb_id": t["pdb_id"],
            "family_id": t["family_id"],
            "split": t["split"],
            "reference_sequence": t["reference_sequence"],
            "candidate_id": (row.candidate_id.iloc[0] if len(row) else None),
            "receptor_len": len(seq),
        }
    return out


def _match_parent(rec_seq: str, index: dict[str, dict]) -> dict | None:
    """Attribute a dumped complex to its parent by receptor sequence.

    Exact match first, then containment: the CPSea loader crops the receptor
    (`crop_size: 256`), so for a target longer than the crop the dumped receptor is a
    contiguous *substring* of the template's chain, not the whole thing. Exact-only
    matching silently dropped 16 of 48 samples -- the four largest targets (527, 413, 300
    and 250 residues) -- which would have quietly turned a 12-parent comparison into an
    8-parent one weighted toward small receptors.

    Containment is still unambiguous here: a 100+ residue crop is unique across these
    twelve receptors. An ambiguous crop matching two parents returns None rather than
    picking one.
    """
    hit = index.get(rec_seq)
    if hit is not None:
        return hit
    if len(rec_seq) < 30:
        return None
    cands = [v for k, v in index.items() if rec_seq in k or k in rec_seq]
    if len(cands) == 1:
        return cands[0]

    # The crop is not a contiguous window. `CroppingTransform2` selects receptor residues
    # by proximity to the binder, so for a target longer than `crop_size: 256` the dumped
    # receptor is a DISCONTIGUOUS subset and neither equality nor substring matching finds
    # it -- which silently dropped the four largest targets (527, 413, 300, 250 residues),
    # 16 of 48 samples, biasing the comparison toward small receptors.
    #
    # 8-mer containment separates them unambiguously: measured 0.941 for the correct
    # parent against 0.000 for the runner-up. A match is accepted only when it clears an
    # absolute floor AND beats the runner-up decisively, so an ambiguous crop is still
    # reported unattributed rather than guessed.
    def kmers(seq: str, k: int = 8) -> set[str]:
        return {seq[i : i + k] for i in range(len(seq) - k + 1)}

    probe = kmers(rec_seq)
    if not probe:
        return None
    scored = sorted(
        ((len(probe & kmers(k)) / len(probe), v) for k, v in index.items()),
        key=lambda t: t[0],
        reverse=True,
    )
    best, runner = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
    if best[0] >= 0.5 and best[0] >= 5 * max(runner[0], 1e-9):
        return best[1]
    return None


def recovery(requested: str, got: str) -> float | None:
    if not requested or not got or len(requested) != len(got):
        return None
    return sum(1 for a, b in zip(requested, got) if a == b) / len(requested)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--predictor-name", default="cpsea_seqcond_v2")
    ap.add_argument("--ckpt-step", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chemistry", default="mainchain")
    args = ap.parse_args()

    tol_set = tolmod.load(args.dataset_dir / "manifest" / "geometry_tolerances.json")
    tol = tol_set[args.chemistry]
    index = parent_index(args.dataset_dir)
    print(f"[seqcond] {len(index)} parents indexed by receptor sequence")

    manifests = sorted(glob.glob(str(args.dump_dir / "step_*" / "manifest_rank*.jsonl")))
    if not manifests:
        print(f"[seqcond] no manifests under {args.dump_dir} -- nothing generated yet")
        return
    print(f"[seqcond] {len(manifests)} manifest file(s)")

    out_path = args.out or (args.dataset_dir / "scores" / f"rows_{args.predictor_name}_0.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:  # noqa: BLE001, S112
                continue

    deformation_cache: dict = {}
    n_ok = n_skip = n_unmatched = 0
    for man in manifests:
        step_dir = Path(os.path.dirname(man))
        for line in Path(man).read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            pdb = step_dir / row["pdb"]
            if not pdb.exists():
                continue
            binder_chain = row.get("binder_chain", "B")
            target_chains = [c for c in str(row.get("target_chains", "A")) if c.strip()]

            # Attribution from the receptor sequence, never from `bid`.
            parsed = parse_structure(pdb)
            rec_seq = "".join(chain_sequence(parsed, ch) for ch in target_chains)
            info = _match_parent(rec_seq, index)
            if info is None:
                n_unmatched += 1
                continue

            task_id = f"{args.predictor_name}|{info['parent_id']}|{row.get('step')}|{row.get('rank')}|{row.get('bid')}"
            if task_id in done:
                n_skip += 1
                continue

            requested = info["reference_sequence"]
            got = sequence_of(read_chain(pdb, binder_chain))
            rec = recovery(requested, got)

            t0 = time.time()
            val = validity_report(
                pdb,
                binder_chain=binder_chain,
                receptor_chains=target_chains,
                chemistry=args.chemistry,
                requested_sequence=requested,
            )
            verd = tolmod.verdict(val, tol)
            sc = score_structure(
                pdb,
                binder_chain=binder_chain,
                receptor_chains=target_chains,
                chemistry=args.chemistry,
                requested_sequence=requested,
                refined_out_path=str(pdb.with_suffix("")) + "_refined.pdb",
                isolated_out_path=str(pdb.with_suffix("")) + "_isolated_relaxed.pdb",
                deformation_cache=deformation_cache,
            )

            out_row = {
                "task_id": task_id,
                "candidate_id": info["candidate_id"],
                "parent_id": info["parent_id"],
                "family_id": info["family_id"],
                "split": info["split"],
                "chemistry": args.chemistry,
                "predictor": args.predictor_name,
                "predictor_revision": f"step={args.ckpt_step}" if args.ckpt_step else None,
                "input_hash": None,
                "config": {
                    "seq_cond": True,
                    "keep_frac": 1.0,
                    "nsteps": 400,
                    "n_repeat": 4,
                    "ckpt_step": args.ckpt_step,
                },
                # Not the same conditioning as the other arms, and said so rather than
                # implied: this model gets the receptor as x_target with the crop centred
                # on the binder site, not a target-only template plus a pocket constraint.
                "pocket_conditioning": "crop_centred_on_binder_site (x_target receptor)",
                "sequence_requested": requested,
                "peptide_length": len(requested),
                "n_mutations": 0,
                "mutation_fraction": 0.0,
                "is_reference": True,
                "sample_index": int(row.get("bid", -1)),
                "sample_identity": {"step": row.get("step"), "rank": row.get("rank"),
                                    "bid": row.get("bid")},
                "prediction_status": "ok",
                "prediction_runtime_s": None,
                "raw_pdb_path": str(pdb),
                "predictor_confidence": {},
                # The compliance question this arm uniquely has.
                "sequence_generated": got,
                "cond_recovery": rec,
                "sequence_matches": bool(got == requested),
                "valid": bool(verd["valid"]),
                "validity_checks": verd["checks"],
                "validity_reasons": verd["reasons"],
                "geometry": {
                    "closure": val["closure"],
                    "chirality": {k: v for k, v in val["chirality"].items() if k != "per_residue"},
                    "backbone": val["backbone"],
                    "clashes": {k: v for k, v in val["clashes"].items() if "examples" not in k},
                    "atoms": {k: v for k, v in val["atoms"].items() if k != "missing_detail"},
                    "sequence_matches": val["sequence_matches"],
                    "sequence_observed": val["sequence_observed"],
                },
                "rosetta": {
                    k: sc.get(k)
                    for k in ("status", "raw", "refined", "refinement_displacement",
                              "deformation", "ring_declared_complex", "refined_pdb_path",
                              "scorer_chemistry_status", "reason", "restraints")
                },
                "protocol": sc.get("protocol"),
                "scoring_status": sc.get("status"),
                "scoring_runtime_s": round(time.time() - t0, 2),
            }
            with open(out_path, "a") as fh:
                fh.write(json.dumps(out_row) + "\n")
            n_ok += 1
            dg = ((out_row["rosetta"] or {}).get("refined") or {}).get("dG_separated")
            print(f"[seqcond] {info['pdb_id']} bid={row.get('bid')} "
                  f"recovery={rec if rec is None else round(rec, 3)} "
                  f"seq_match={out_row['sequence_matches']} valid={out_row['valid']} "
                  f"dG={dg} {out_row['scoring_runtime_s']}s")

    print(f"\n[seqcond] scored={n_ok} skipped={n_skip} unmatched_receptor={n_unmatched}")
    print(f"[seqcond] rows -> {out_path}")
    if n_unmatched:
        print("[seqcond] NOTE: unmatched complexes had a receptor sequence not among the "
              "dataset's parents -- not attributed rather than guessed.")


if __name__ == "__main__":
    main()
