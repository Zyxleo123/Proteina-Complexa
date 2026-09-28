"""Re-score accepted / near-miss candidates with many independent repeats.

Why this stage exists
---------------------
A pass at the energy bar is only a result if the margin to the bar exceeds the measurement
noise. Measured on the first real run: the median within-candidate spread over 3 independent
relaxation seeds is 8.4 REU, and 60% of candidates have a spread wider than the whole +/-5 REU
bar. The first accepted candidate passed at ddG +4.67 with a repeat spread of 1.04 REU -- a
margin of 0.33 REU, i.e. inside its own noise. Reporting that as a success without saying so
would be overstating what was measured.

Selecting the minimum over a large shortlist makes this worse, not better: with N draws from a
distribution whose spread exceeds the bar, something will land under the bar by luck alone. The
defence is to re-measure the SELECTED candidates with a much larger repeat count and ask whether
the median still passes -- which is a different question from whether any single draw did.

What it does
------------
Reads the search/SDEdit rows, picks the candidates worth confirming (accepted, or within
`--near-miss-reu` of the bar on energy while passing everything else), rebuilds each from its
dumped PDB, re-declares its ring, and runs `--repeats` independent relax/score cycles against a
paired LP baseline scored the same number of times. Reports the median, the spread, an
interquartile range and whether the pass survives.

Rebuilding from the dumped PDB is deliberate and it is why the ring is re-declared: no PDB
writer here emits LINK records, so a dumped macrocycle re-loads as a linear peptide and Rosetta
retypes the anchors as ordinary free termini. Scoring it as-loaded would silently describe an
open topology -- the same defect this arm had to fix in `rosetta_energy`. The chemistry and
anchor identities come from the candidate's own row, and the declaration is verified.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_contract as C            # noqa: E402
import cp_evaluate as E            # noqa: E402
import cp_search as S              # noqa: E402
import cp_topology as T            # noqa: E402

MAX_DDG = C.MAX_DDG_REU


def load_rows(paths: list[Path]) -> list[dict]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.glob("rows_*.jsonl")) + sorted(p.glob("rows.jsonl"))
            files += sorted(p.glob("*/rows.jsonl"))
        elif p.is_file():
            files.append(p)
    if not files:
        raise SystemExit(f"FATAL: no rows found under {[str(p) for p in paths]}")
    latest: dict[str, dict] = {}
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"FATAL: corrupt row {f}:{lineno}")
            key = row.get("row_key")
            if key:
                latest[key] = row          # last wins, so retries supersede failures
    return list(latest.values())


def worth_confirming(row: dict, near_miss_reu: float) -> str | None:
    """`"accepted"`, `"near_miss"`, or None.

    A near miss is a candidate that fails ONLY on energy and only by `near_miss_reu`. A
    candidate that also fails RMSD or contacts is not a near miss -- more relax repeats cannot
    move a geometric criterion, so re-scoring it would burn compute to reconfirm a rejection.
    """
    if row.get("stage") != "post_relax" or row.get("status") != "ok":
        return None
    if not row.get("candidate_pdb"):
        return None
    if row.get("accepted"):
        return "accepted"
    reasons = set(row.get("fail_reasons") or [])
    if reasons != {"ddg_interface"}:
        return None
    try:
        ddg = float(row.get("ddg_interface_reu"))
    except (TypeError, ValueError):
        return None
    if ddg != ddg:
        return None
    return "near_miss" if ddg <= MAX_DDG + near_miss_reu else None


def confirm_one(row: dict, target, repeats: int, relax_repeats: int) -> dict:
    """Re-score one candidate `repeats` times from independent seeds."""
    import pyrosetta

    pdb = row["candidate_pdb"]
    chem = T.chemistry(row["chemistry"])
    pose = pyrosetta.pose_from_pdb(pdb)
    bp = T.binder_positions(pose, target.binder_chain)
    if len(bp) != target.length:
        return {"status": "length_mismatch",
                "error": f"dumped candidate L={len(bp)} vs reference L={target.length}"}
    i_pos, j_pos = bp[0], bp[-1]
    aa_i, aa_j = pose.residue(i_pos).name3(), pose.residue(j_pos).name3()

    # The dumped PDB has no LINK record, so the ring must be re-declared or the "confirmation"
    # would score an open peptide. Verified, not assumed.
    try:
        atoms = T.resolve_ring_atoms(chem, aa_i, aa_j)
        T.declare_ring_bond(pose, i_pos, j_pos, chem, atoms)
    except Exception as exc:                                          # noqa: BLE001
        return {"status": "redeclare_failed",
                "error": f"{type(exc).__name__}: {exc}"[:250],
                "anchors_in_dump": f"{aa_i}/{aa_j}"}
    if not pose.residue(i_pos).is_bonded(j_pos):
        return {"status": "redeclare_failed", "error": "ring not bonded after re-declaration"}

    geom = T.ring_bond_geometry(pose, i_pos, j_pos, chem, atoms)
    window = {"k_nterm": row.get("k_nterm", -1), "m_cterm": row.get("m_cterm", -1),
              "n_released": row.get("n_released", 0),
              "released_local": row.get("released_local") or [],
              "released_pose": [], "loop_order": [],
              "anchor_lower": None, "anchor_upper": None}

    class _Spec:
        chemistry = row["chemistry"]
        aa_i = None
        aa_j = None
        k_nterm = row.get("k_nterm", -1)
        m_cterm = row.get("m_cterm", -1)
        extra_mutations: dict = {}

        def label(self):
            return row.get("spec_label", "?")

    spec = _Spec()
    spec.aa_i, spec.aa_j = aa_i, aa_j

    rep_pose, stats = E.refine_candidate_repeats(
        pose, target, i_pos, j_pos, chem, atoms,
        repeats=repeats, relax_repeats=relax_repeats,
        seed_base=7_000_000)
    lp = E.lp_baseline_metrics(target, repeats=repeats, relax_repeats=relax_repeats,
                               seed_base=8_000_000)
    post = E.evaluate_candidate(target, rep_pose, spec, atoms, window, i_pos, j_pos,
                                stage="confirm", lp_baseline=lp, with_energy=True)
    post.update(stats)

    cp_all = [x for x in (stats.get("cp_dG_separated_all") or []) if x is not None]
    lp_all = [x for x in (lp.get("lp_dG_separated_all") or []) if x is not None]

    def iqr(xs):
        if len(xs) < 4:
            return float("nan")
        q = sorted(xs)
        return q[int(0.75 * (len(q) - 1))] - q[int(0.25 * (len(q) - 1))]

    ddg = post.get("ddg_interface_reu")
    margin = (MAX_DDG - float(ddg)) if isinstance(ddg, (int, float)) and ddg == ddg else float("nan")
    combined = ((stats.get("cp_dG_separated_spread") or 0.0)
                + (lp.get("lp_dG_separated_spread") or 0.0))
    post.update({
        "status": "ok",
        "confirm_repeats": repeats,
        "cp_dG_iqr": round(iqr(cp_all), 3) if cp_all else float("nan"),
        "lp_dG_iqr": round(iqr(lp_all), 3) if lp_all else float("nan"),
        "ddg_margin_to_bar_reu": round(margin, 3) if margin == margin else float("nan"),
        "ddg_combined_spread_reu": round(combined, 3),
        # The verdict this stage exists to produce.
        "confirmed_pass": bool(post.get("accepted") and margin == margin and margin > combined),
        "pass_within_noise": bool(post.get("accepted") and margin == margin
                                  and margin <= combined),
        "original_ddg_reu": row.get("ddg_interface_reu"),
        "original_selection": row.get("_selection"),
        "bond_length_A_redeclared": geom.get("bond_length_A"),
        "source_row_key": row.get("row_key"),
        "candidate_pdb_in": pdb,
    })
    return post


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True,
                    help="Search/SDEdit rows directories or files to select candidates from.")
    ap.add_argument("--metadata", required=True, help="Staged LP parquet (for the reference LP).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=15,
                    help="Independent relax/score repeats per candidate AND per LP baseline.")
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--near-miss-reu", type=float, default=5.0,
                    help="Also confirm candidates that fail ONLY on energy, by up to this much.")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="Cap (0 = no cap). Best margin first, so a cap keeps the best.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    rows = load_rows([Path(p) for p in args.rows])
    picked: list[dict] = []
    for r in rows:
        kind = worth_confirming(r, args.near_miss_reu)
        if kind:
            r = dict(r)
            r["_selection"] = kind
            picked.append(r)
    picked.sort(key=lambda r: (0 if r["_selection"] == "accepted" else 1,
                               float(r.get("ddg_interface_reu", 1e9))))
    if args.max_candidates:
        picked = picked[: args.max_candidates]
    mine = picked[args.shard_index::args.shard_count]
    n_acc = sum(1 for r in picked if r["_selection"] == "accepted")
    print(f"{len(rows)} rows -> {len(picked)} worth confirming "
          f"({n_acc} accepted, {len(picked) - n_acc} energy-only near misses); "
          f"shard {args.shard_index}/{args.shard_count} takes {len(mine)}", flush=True)

    if args.dry_run:
        for r in mine:
            print(f"  {r['_selection']:10s} {r['example_id']:22s} {r.get('chemistry'):10s} "
                  f"ddG={r.get('ddg_interface_reu')} spread={r.get('cp_dG_separated_spread')} "
                  f"{Path(str(r.get('candidate_pdb'))).name}", flush=True)
        print("DRY RUN OK: ran no Rosetta.", flush=True)
        return 0
    if not mine:
        print("nothing to confirm in this shard; exiting 0.", flush=True)
        return 0

    meta = pd.read_parquet(args.metadata)
    ref = {str(m["example_id"]): (str(m["path"]), str(m["binder_chain_id"]))
           for _, m in meta.iterrows()}

    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init
    _ensure_pyrosetta_init()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets: dict[str, object] = {}
    t0 = time.time()
    n_conf = n_noise = 0
    with out_path.open("a") as fh:
        for r in mine:
            ex = r["example_id"]
            if ex not in ref:
                fh.write(json.dumps({"status": "no_reference", "example_id": ex,
                                     "source_row_key": r.get("row_key")}) + "\n")
                continue
            if ex not in targets:
                pdb, chain = ref[ex]
                targets[ex] = S.load_target(ex, pdb, chain)
            try:
                row = confirm_one(r, targets[ex], args.repeats, args.relax_repeats)
            except Exception as exc:                                  # noqa: BLE001
                row = {"status": "confirm_failed",
                       "error": f"{type(exc).__name__}: {exc}"[:300]}
            row.setdefault("example_id", ex)
            row.setdefault("chemistry", r.get("chemistry"))
            row.setdefault("spec_label", r.get("spec_label"))
            row["_selection"] = r["_selection"]
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if row.get("confirmed_pass"):
                n_conf += 1
            if row.get("pass_within_noise"):
                n_noise += 1
            print(f"  {ex:22s} {r.get('chemistry','?'):10s} {r['_selection']:10s} "
                  f"ddG {r.get('ddg_interface_reu')} -> {row.get('ddg_interface_reu')} "
                  f"(margin {row.get('ddg_margin_to_bar_reu')} vs spread "
                  f"{row.get('ddg_combined_spread_reu')})  "
                  f"{'CONFIRMED' if row.get('confirmed_pass') else ('WITHIN NOISE' if row.get('pass_within_noise') else row.get('status'))}",
                  flush=True)

    print(f"done in {time.time() - t0:.0f}s: {n_conf} confirmed, {n_noise} pass-but-within-noise "
          f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
