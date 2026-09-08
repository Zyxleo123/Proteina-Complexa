"""Rosetta interface dG BEFORE and AFTER the linear -> cyclic edit, on the same complex.

Why this exists
---------------
Every number the SDEdit sweep reports so far is about the RING: did the two atoms the
chemistry would bond end up at bonding distance, and how much of the input pose survived.
None of them says whether the edited peptide still *binds*. Closure bought by wrecking the
interface is not a result, and `contact_retention` (CA-CA within 10 A) cannot tell a
preserved interface from a sterically ruined one -- it counts neighbours, not energy.

So this scores the SAME complex twice with the same instrument:

    before   the structure the arm was actually given (the staged crystal linear peptide,
             or -- for the soft-closure `projected` arm -- the projected pose)
    after    the edited peptide in that same receptor

The pairing is the point. `dG_separated` is not comparable across targets (a 357-residue
receptor and a 294-residue one have different interface scales), but before/after on ONE
complex is, and the paired delta is what "did cyclization cost binding?" actually means.

Where the complexes come from
-----------------------------
`scripts/sdedit_cyclize.py --complex-pdb-dir` writes them DURING sampling, and that is the
only moment they can be written: the CPSea loader hands out a target-centred crop whose
frame is redrawn per process (two loads of one example measured 2.66 A apart), so a peptide
saved without its receptor can never be put back into it afterwards. If an arm was run
without that flag, this script says so and stops rather than inventing a placement -- the
arm has to be re-run, there is no offline fix. See scripts/complex_frame.py.

CPU only, no GPU, no model: it reads artefacts the GPU job already wrote, so the numbers
and the figure downstream re-render without re-sampling anything. Resumable, and shardable
by input peptide (each shard writes its own JSONL -- concurrent O_APPEND has corrupted rows
on this cluster).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proteinfoundation.evaluation.rosetta_energy import (  # noqa: E402
    compute_rosetta_interface_metrics_single,
    is_pyrosetta_available,
)

SUCCESS_COL = {"mainchain": "cyc/mainchain_cn_bond_success",
               "disulfide": "cyc/disulfide_bond_success",
               "isopeptide": "cyc/isopeptide_bond_success"}
DIST_COL = {"mainchain": "cyc/mainchain_cn_dist_pred_A",
            "disulfide": "cyc/disulfide_sg_dist_pred_A",
            "isopeptide": "cyc/isopeptide_n_c_dist_pred_A"}
# Edit fields carried into the output next to the Rosetta metrics, so the figure can slice
# dG by grid point / closure / retention without re-joining against the edits JSONL.
CARRY = ("run_key", "example_id", "cyc_type", "t_ca_start", "t_lat_start", "seed",
         "requested_type_satisfied", "contact_retention", "ca_rmsd_to_input_A",
         "n_substitutions", "seq_identity", "peptide_length", "pred_cyc_i", "pred_cyc_j",
         "frame_residual_A", "guidance_w", "guidance_loss", "guidance_schedule",
         "guidance_exclude_termini", "guidance_mode", "guid_ca_disp_A")


def load_edits(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Refuse, never skip: a NUL-corrupted row means two writers shared a file,
                # and dropping it silently turns data loss into a plausible-looking table.
                raise SystemExit(f"FATAL: corrupt row {p}:{lineno}. Do not trust this run.")
    return rows


def edit_tag(row: dict) -> str:
    """The filename scripts/sdedit_cyclize.py wrote for this edit.

    `pdb_tag` is authoritative when present: a guided edit appends its guidance settings to
    the filename, and reconstructing the tag from the grid columns alone would resolve every
    lambda arm to the UNGUIDED file -- i.e. silently score the wrong structure.
    """
    tag = row.get("pdb_tag")
    if tag:
        return str(tag)
    return (f"{row['example_id']}_{row['cyc_type']}_tca{row['t_ca_start']}"
            f"_tlat{row['t_lat_start']}_s{row['seed']}")


def arm_key(row: dict) -> str:
    """Which guidance arm a row belongs to; "" for the unguided arm."""
    if not row.get("guidance_w"):
        return ""
    return (f"{row.get('guidance_loss')}|{row['guidance_w']}|{row.get('guidance_schedule')}"
            f"|{row.get('guidance_exclude_termini')}|{row.get('guidance_mode')}")


def complex_path(row: dict, complex_dir: Path | None) -> Path | None:
    """The after-complex for this edit: the path the sampler recorded, else the same name
    under --complex-dir (which is how a moved/copied run is still scorable)."""
    p = row.get("complex_pdb")
    if p and Path(p).is_file():
        return Path(p)
    if complex_dir is not None:
        cand = complex_dir / f"{edit_tag(row)}__complex.pdb"
        if cand.is_file():
            return cand
    return None


def is_closed(row: dict) -> bool:
    col = SUCCESS_COL.get(row.get("cyc_type", ""))
    return bool(col and row.get(col) == 1 and row.get("requested_type_satisfied") == 1)


def done_keys(out_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            done.add((r["kind"], r["key"]))
        except (json.JSONDecodeError, KeyError):
            continue  # leave it in the file for a human; just do not count it as done
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edits", nargs="+", required=True, help="edits_shard*.jsonl from the arm.")
    ap.add_argument("--out", required=True, help="Output JSONL (appended; resumable).")
    ap.add_argument("--complex-dir", default=None,
                    help="Fallback location of the *__complex.pdb files, if the paths recorded "
                         "in the edits rows no longer resolve.")
    ap.add_argument("--metadata", default=None,
                    help="Optional parquet, used only to supply the input complex / binder chain "
                         "for rows that did not record them.")
    # Rosetta FastRelax is minutes per complex. These bound the bill.
    ap.add_argument("--only-closed", action="store_true",
                    help="Score only edits whose requested ring actually closed.")
    ap.add_argument("--only-scorable", action="store_true",
                    help="Score only edits where the model proposed the requested chemistry.")
    ap.add_argument("--max-per-example", type=int, default=0, help="Cap after-scores per input (0=all).")
    ap.add_argument("--examples", nargs="+", default=None,
                    help="Restrict scoring to these example_ids (e.g. for a targeted viz pass). "
                         "An id present in --examples but absent from the edits is a FATAL error, "
                         "not a silent empty run.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve rows and structures, print the bill, score nothing.")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    complex_dir = Path(args.complex_dir) if args.complex_dir else None

    rows = [r for r in load_edits([Path(p) for p in args.edits]) if r.get("status") == "ok"]
    if not rows:
        # Gate failures exit 0 so an afterok chain still runs and reports them.
        print(f"GATE: no successful edits in {args.edits}. Nothing to score. Exiting 0.", flush=True)
        return 0
    if args.examples:
        want = set(args.examples)
        have = {str(r.get("example_id")) for r in rows}
        absent = want - have
        if absent:
            raise SystemExit(f"FATAL: --examples not in edits: {sorted(absent)}")
        rows = [r for r in rows if str(r.get("example_id")) in want]
        print(f"--examples filter: {len(rows)} rows over {len(want)} example(s)", flush=True)
    if args.only_scorable or args.only_closed:
        rows = [r for r in rows if r.get("requested_type_satisfied") == 1]
    if args.only_closed:
        rows = [r for r in rows if is_closed(r)]
    if not rows:
        print("GATE: every edit was filtered out by --only-scorable/--only-closed. Exiting 0.",
              flush=True)
        return 0

    scorable = [(r, complex_path(r, complex_dir)) for r in rows]
    missing = [r for r, p in scorable if p is None]
    scorable = [(r, p) for r, p in scorable if p is not None]
    if not scorable:
        print(f"GATE: none of the {len(rows)} edits has an after-complex on disk.\n"
              "GATE: the arm was run without `--complex-pdb-dir`, and the complex CANNOT be\n"
              "GATE: rebuilt offline -- the CPSea loader's frame is redrawn per process, so a\n"
              "GATE: peptide-only PDB has no recoverable placement in its receptor.\n"
              "GATE: re-run the SDEdit arm with --complex-pdb-dir. Exiting 0.", flush=True)
        return 0
    if missing:
        print(f"WARNING: {len(missing)} of {len(rows)} edits have no complex on disk and are "
              f"not scored (first: {edit_tag(missing[0])})", flush=True)

    # Input (before) structure per example. Prefer what the sampler recorded; fall back to the
    # metadata the arm was run on.
    before_of: dict[str, tuple[str, str, list[str]]] = {}
    for r, _ in scorable:
        ex = r["example_id"]
        if ex in before_of:
            continue
        inp, chain = r.get("input_complex_pdb"), r.get("binder_chain")
        tgt = r.get("target_chains")
        if inp and chain and tgt:
            before_of[ex] = (inp, chain, [c for c in str(tgt).split(",") if c])
    if args.metadata:
        import pandas as pd
        meta = pd.read_parquet(args.metadata)
        for _, m in meta.iterrows():
            ex = str(m["example_id"])
            if ex in before_of or "path" not in m:
                continue
            chain = str(m.get("binder_chain_id", "B"))
            from complex_frame import chain_of, read_atom_lines
            if not Path(str(m["path"])).is_file():
                continue
            chains = sorted({chain_of(l) for l in read_atom_lines(m["path"])} - {chain})
            before_of[ex] = (str(m["path"]), chain, chains)

    by_example: dict[str, list[tuple[dict, Path]]] = {}
    for r, p in scorable:
        by_example.setdefault(r["example_id"], []).append((r, p))
    # Shard by EXAMPLE so a `before` is scored once, in one place, by one job.
    examples = sorted(by_example)[args.shard_index::args.shard_count]
    def _capped(e):
        per_arm: dict[str, int] = {}
        for r, _ in by_example[e]:
            per_arm[arm_key(r)] = per_arm.get(arm_key(r), 0) + 1
        cap = args.max_per_example
        return sum(min(v, cap) if cap else v for v in per_arm.values())

    n_after = sum(_capped(e) for e in examples)
    print(f"{len(scorable)} scorable edits over {len(by_example)} inputs; shard "
          f"{args.shard_index}/{args.shard_count} takes {len(examples)} inputs "
          f"= {len(examples)} before + {n_after} after structures", flush=True)

    if args.dry_run:
        for e in examples:
            b = before_of.get(e)
            print(f"  {e}: before={'MISSING' if not b else b[0]}  after={len(by_example[e])}",
                  flush=True)
        print("DRY RUN OK: scored nothing.", flush=True)
        return 0

    if not is_pyrosetta_available():
        print("PyRosetta unavailable; nothing scored. Exiting 0.", flush=True)
        return 0

    done = done_keys(out_path)
    if done:
        print(f"resuming: {len(done)} structures already scored", flush=True)

    n_scored = 0
    t_wall = time.time()
    with out_path.open("a") as fh:

        def emit(kind: str, key: str, payload: dict) -> None:
            nonlocal n_scored
            fh.write(json.dumps({"kind": kind, "key": key, **payload}) + "\n")
            fh.flush()
            done.add((kind, key))
            n_scored += 1

        for example_id in examples:
            # --- before -------------------------------------------------------------------
            if ("before", example_id) not in done:
                b = before_of.get(example_id)
                if b is None:
                    # Without the before there is no delta for this peptide -- say so loudly,
                    # and still score the afters (their absolute dG is still worth having).
                    print(f"  WARNING {example_id}: no input complex recorded; no BEFORE score",
                          flush=True)
                else:
                    inp, chain, tgt = b
                    metrics = compute_rosetta_interface_metrics_single(
                        pdb_path=inp, binder_chain=chain, target_chains=tgt)
                    emit("before", example_id,
                         {"example_id": example_id, "pdb": inp, **metrics})
                    print(f"  {example_id} BEFORE dG_separated="
                          f"{metrics.get('binder_rosetta_dG_separated')}", flush=True)

            # --- after --------------------------------------------------------------------
            # The cap is per (input, guidance arm), not per input: with several lambda arms
            # in one results dir a flat cap sorted by run_key would spend the whole budget on
            # whichever arm sorts first, and the paired before/after comparison across arms
            # -- the entire point of the guided sweep -- would have no data for the rest.
            n_this: dict[str, int] = {}
            for row, cpx in sorted(by_example[example_id], key=lambda rp: rp[0]["run_key"]):
                arm = arm_key(row)
                if args.max_per_example and n_this.get(arm, 0) >= args.max_per_example:
                    continue
                n_this[arm] = n_this.get(arm, 0) + 1
                if ("after", row["run_key"]) in done:
                    continue
                chain = row.get("binder_chain") or (before_of.get(example_id) or (None, "B", []))[1]
                tgt = ([c for c in str(row.get("target_chains", "")).split(",") if c]
                       or (before_of.get(example_id) or (None, None, []))[2])
                if not tgt:
                    print(f"  SKIP {row['run_key']}: no receptor chains known", flush=True)
                    continue
                # Declare the ring only when it is ACTUALLY CLOSED. Two ways to get this
                # wrong, both measured: on an abstention the predicted (i, j) describe a
                # different bond; and on a satisfied-but-open edit (smoke run: mainchain
                # termini 13.5 A and 24.7 A apart, both declared) Rosetta is told two atoms
                # are covalently bonded and FastRelax then drags the gap shut -- the dG that
                # comes back describes a structure the model never produced. An open ring is
                # scored as what it is: a linear peptide in a ring-ish pose.
                declare = is_closed(row)
                metrics = compute_rosetta_interface_metrics_single(
                    pdb_path=str(cpx), binder_chain=chain, target_chains=tgt,
                    cyclization_type=row["cyc_type"] if declare else None,
                    cyclization_i=int(row["pred_cyc_i"]) if declare else None,
                    cyclization_j=int(row["pred_cyc_j"]) if declare else None,
                )
                payload = {k: row[k] for k in CARRY if k in row}
                dist_col = DIST_COL.get(row.get("cyc_type", ""))
                if dist_col and dist_col in row:
                    payload["ring_dist_A"] = row[dist_col]
                payload.update(pdb=str(cpx), closed=int(is_closed(row)),
                               ring_declared=int(declare), **metrics)
                emit("after", row["run_key"], payload)
                if n_scored % 10 == 0:
                    print(f"  {n_scored} structures scored ({time.time() - t_wall:.0f}s)",
                          flush=True)

    print(f"done: {n_scored} scored, {time.time() - t_wall:.0f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
