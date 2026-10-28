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


def load_requested_sidecar(path: Path) -> dict[str, dict]:
    """{native peptide sequence -> request record} written by `seq_inject.write_sidecar`.

    Without this, a run that was conditioned on a SEARCHED sequence is scored against the
    dataset's native peptide, so every sample fails the exact-sequence check and the whole
    run reads as non-compliant -- a wrong answer that looks like a finding. Empty dict when
    there is no sidecar, which is the native-sequence runs and is correct for them.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    schema = payload.get("schema")
    if schema != "cpseq-requested-sequences-1":
        raise SystemExit(f"{path}: unrecognised schema {schema!r}; refusing to guess its layout.")
    return payload.get("by_native_sequence", {})


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
    # Conditioning and the sampling settings were hardcoded into every row's `config`
    # when this script served exactly one run. An UNCONDITIONAL generator (no
    # `--seq-cond`, which is the default mode of `eval_val_gen_testset.py`) designs its
    # own peptide, so recording `seq_cond: true`, `n_mutations: 0` and
    # `is_reference: true` for it would assert that the model had been handed the
    # parent's LP sequence and returned it unchanged -- three false provenance claims on
    # a row whose energy is otherwise perfectly usable.
    ap.add_argument(
        "--conditioning",
        choices=("seq_cond", "partial_seq_cond", "unconditional"),
        default="seq_cond",
        help="what the generator was actually given; recorded per row",
    )
    ap.add_argument("--requested-sequences", type=Path, default=None,
                    help="The requested-sequence sidecar. Default: <dump-dir>/requested_sequences.json, "
                         "written by the generation run when it was given --seq-source. Without it a "
                         "run conditioned on searched peptides is scored against the native ones and "
                         "reads as wholly non-compliant.")
    ap.add_argument("--keep-frac", type=float, default=1.0)
    ap.add_argument("--n-repeat", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=400)
    args = ap.parse_args()

    tol_set = tolmod.load(args.dataset_dir / "manifest" / "geometry_tolerances.json")
    tol = tol_set[args.chemistry]
    index = parent_index(args.dataset_dir)
    print(f"[seqcond] {len(index)} parents indexed by receptor sequence")

    requested_sidecar = load_requested_sidecar(
        args.requested_sequences or (args.dump_dir / "requested_sequences.json")
    )
    if requested_sidecar:
        n_designed = sum(1 for r in requested_sidecar.values() if not r.get("is_native"))
        print(f"[seqcond] requested-sequence sidecar: {len(requested_sidecar)} parents, "
              f"{n_designed} carrying substitutions -- compliance is measured against these, "
              f"not the dataset's native peptides")
    else:
        print("[seqcond] no requested-sequence sidecar; compliance is vs each parent's native peptide")

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

            original = info["reference_sequence"]
            got = sequence_of(read_chain(pdb, binder_chain))

            if args.conditioning == "partial_seq_cond":
                # Only PART of the sequence was revealed, so the decoded sequence is
                # neither a request to comply with nor a free design. Scoring it against
                # the full LP (the seq_cond branch) would fail the exact-sequence gate on
                # every intentionally designed residue and report 0 % validity for a run
                # that was asked to design -- geometry would never be reached. So the
                # structure is scored as the molecule it is, and compliance is reported
                # separately as `cond_recovery` against the LP, which at reveal fraction
                # f should sit near f + (1-f)*chance if the model designed the rest.
                requested = None
                rec = recovery(original, got)
                requested_for_scoring = got
                seq_check_note = (
                    f"PARTIAL reveal (keep_frac={args.keep_frac}): the exact-sequence gate "
                    f"is scored against the DECODED sequence, so it is vacuous; compliance "
                    f"is cond_recovery vs the LP, and n_mutations is the distance from it"
                )
            elif args.conditioning == "seq_cond":
                # The model was handed this sequence; compliance is the question. When the run
                # injected a searched peptide, the sequence it was handed is NOT the dataset's
                # reference, and the sidecar is the only record of which one it was.
                asked = requested_sidecar.get(original)
                requested = asked["requested_sequence"] if asked else original
                rec = recovery(requested, got)
                # Scored against what was ASKED, so a model that designed its own
                # peptide fails the exact-sequence check rather than being scored as
                # though it had complied.
                requested_for_scoring = requested
                seq_check_note = (
                    "exact-sequence check is meaningful (sequence was requested"
                    + (
                        f"; INJECTED {asked['label']}, {asked['n_substitutions']} substitutions "
                        f"from the native {original})"
                        if asked
                        else ", the dataset's native peptide)"
                    )
                )
            else:
                # Nothing was requested, so there is nothing to comply with. The decoded
                # sequence IS the output, and it is scored as itself; the exact-sequence
                # gate inside `verdict` is therefore vacuous here and is labelled as
                # such, so a reader cannot mistake it for demonstrated compliance.
                requested = None
                rec = None
                requested_for_scoring = got
                seq_check_note = (
                    "exact-sequence check is VACUOUS (unconditional generation: the "
                    "decoded sequence is the output, not a request)"
                )

            # Distance from the parent's original LP sequence. For an unconditional
            # sample this is a measurement of what the model chose, not a mutation
            # instruction, and it is what decides whether the sample could serve as a
            # candidate under this project's floor(L/2) substitution ceiling.
            n_mut = (
                sum(1 for a, b in zip(original, got) if a != b)
                if len(original) == len(got)
                else None
            )
            budget = len(original) // 2

            t0 = time.time()
            val = validity_report(
                pdb,
                binder_chain=binder_chain,
                receptor_chains=target_chains,
                chemistry=args.chemistry,
                requested_sequence=requested_for_scoring,
            )
            verd = tolmod.verdict(val, tol)
            sc = score_structure(
                pdb,
                binder_chain=binder_chain,
                receptor_chains=target_chains,
                chemistry=args.chemistry,
                requested_sequence=requested_for_scoring,
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
                    "conditioning": args.conditioning,
                    "seq_cond": args.conditioning in ("seq_cond", "partial_seq_cond"),
                    "keep_frac": (
                        args.keep_frac
                        if args.conditioning in ("seq_cond", "partial_seq_cond") else None
                    ),
                    "nsteps": args.nsteps,
                    "n_repeat": args.n_repeat,
                    "ckpt_step": args.ckpt_step,
                },
                "sequence_check_note": seq_check_note,
                "original_lp_sequence": original,
                "within_substitution_budget": (
                    None if n_mut is None else bool(n_mut <= budget)
                ),
                "substitution_budget": budget,
                # Not the same conditioning as the other arms, and said so rather than
                # implied: this model gets the receptor as x_target with the crop centred
                # on the binder site, not a target-only template plus a pocket constraint.
                "pocket_conditioning": "crop_centred_on_binder_site (x_target receptor)",
                "sequence_requested": requested,
                "sequence_requested_source": (
                    "injected_sidecar" if (args.conditioning == "seq_cond" and requested != original)
                    else ("native_reference" if args.conditioning == "seq_cond" else None)
                ),
                "peptide_length": len(got),
                "n_mutations": n_mut,
                "mutation_fraction": (
                    None if n_mut is None else round(n_mut / len(original), 6)
                ),
                # A reference row is the UNMUTATED sequence returned under sequence
                # conditioning. An unconditional sample is never that, even when it
                # happens to decode zero substitutions away from the original.
                "is_reference": args.conditioning == "seq_cond" and n_mut == 0,
                "conditioning_mode": args.conditioning,
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
                # None, not False, when nothing was requested. `False` would assert that
                # the model was asked for a sequence and returned a different one, which
                # is a compliance failure; an unconditional sample has no request to
                # fail. The vacuity is also stated in `sequence_check_note`.
                "sequence_matches": (None if requested is None else bool(got == requested)),
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
