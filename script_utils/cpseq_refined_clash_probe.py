#!/usr/bin/env python3
"""Does Rosetta relaxation resolve the overlaps the gate rejects on?

The validity check runs on the RAW predicted structure, before refinement -- deliberately,
so that a pass cannot be something the relax rescued (`scorer` docstring). The
consequence, measured in README_CPSEQ_PEPBENCH_SCALEUP sec. 11, is that peptide-receptor
overlap is the single largest rejector: sole cause for 3,457 of 11,953 folds, closest
contact a median 1.94 A.

That raises a question the stored rows cannot answer, because clash counts are only ever
computed on raw coordinates: would those structures still overlap after the refinement
the pipeline already runs? If they would not, the gate is judging the wrong structure for
deciding what is shippable -- while remaining the right one for judging the predictor.

This re-relaxes a stratified sample straight from the retained raw complexes (which is
what the `rescorable` tier promises) and recomputes the clash counts on the refined
output. Stratified by the raw closest contact, because a 2.15 A contact and a 1.5 A
interpenetration are different physical claims and averaging them would hide which one
relaxation can fix.

Nothing here re-predicts: the raw complex comes off disk and only Rosetta runs.

Usage:
    python script_utils/cpseq_refined_clash_probe.py --out-dir $ZFS/cpseq/seed_ladder \
        --per-band 40 --jobs 6 --out refined_clash_probe.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import retention as retmod  # noqa: E402
from proteinfoundation.cpseq.evaluate import BINDER_CHAIN  # noqa: E402
from proteinfoundation.cpseq.geometry import read_chain, validity_report  # noqa: E402


def closest_contact_A(pdb: Path, binder_chain: str, receptor_chains: list[str]) -> float | None:
    """Closest peptide-receptor heavy-atom distance, with no cutoff applied.

    `geometry.clashes` reports `min_inter_dist_A` only for pairs already under 2.2 A, so
    it is null exactly when a structure passes -- which makes "passed" and "passed
    comfortably" indistinguishable. A relax that leaves every contact at 2.21 A has
    satisfied the check without fixing the physics, and that has to be visible.
    """
    import numpy as np

    binder = read_chain(pdb, binder_chain)
    rec_xyz = []
    for ch in receptor_chains:
        for _, atoms in read_chain(pdb, ch):
            rec_xyz.extend(atoms.values())
    if not binder or not rec_xyz:
        return None
    rec = np.asarray(rec_xyz)
    best = None
    for _, atoms in binder:
        for x in atoms.values():
            d = float(np.min(np.linalg.norm(rec - x, axis=1)))
            if best is None or d < best:
                best = d
    return round(best, 3) if best is not None else None

# The bands are the physical distinctions, not quantiles: 2.0-2.2 A is where a hydrogen
# bond is plausible and an element-blind cutoff is suspect, below 1.8 A is
# interpenetration that no threshold choice admits.
BANDS = (("2.10-2.20", 2.10, 2.20), ("2.00-2.10", 2.00, 2.10),
         ("1.80-2.00", 1.80, 2.00), ("below-1.80", 0.0, 1.80))


def min_dist(reason: str) -> float | None:
    m = re.search(r"min dist ([\d.]+) A", str(reason))
    return float(m.group(1)) if m else None


def classify(row: dict) -> tuple[str, float] | None:
    """(band, raw closest contact) for folds rejected ONLY on peptide-receptor overlap."""
    reasons = row.get("validity_reasons") or []
    if not reasons or row.get("valid") or row.get("technical_failure"):
        return None
    if not all("peptide-receptor clash" in str(r) for r in reasons):
        return None
    dists = [d for d in (min_dist(r) for r in reasons) if d is not None]
    if not dists:
        return None
    d = min(dists)
    for name, lo, hi in BANDS:
        if lo <= d < hi:
            return name, d
    return None


def _ca(pdb: Path, chains: list[str]):
    """CA coordinates for `chains`, in file order."""
    import numpy as np

    xyz = []
    for ch in chains:
        for _, atoms in read_chain(pdb, ch):
            if "CA" in atoms:
                xyz.append(atoms["CA"])
    return np.asarray(xyz)


def displacement_split(raw: Path, refined: Path, binder_chain: str,
                       receptor_chains: list[str]) -> dict:
    """Receptor deformation and peptide motion, with rigid-body drift removed.

    A plain before/after RMSD cannot tell internal deformation from the whole complex
    translating during relax, and those mean opposite things: drift is harmless
    bookkeeping, deformation of an experimentally derived receptor is the relax rebuilding
    the target to accommodate the pose. Superimposing on the receptor separates them, and
    then the peptide's residual motion is its movement RELATIVE to the receptor, which is
    what "did the pose change" actually asks.
    """
    import numpy as np

    from proteinfoundation.cpseq.retention import kabsch

    out: dict = {}
    r0, r1 = _ca(raw, receptor_chains), _ca(refined, receptor_chains)
    b0, b1 = _ca(raw, [binder_chain]), _ca(refined, [binder_chain])
    if len(r0) != len(r1) or len(r0) < 3 or len(b0) != len(b1) or not len(b0):
        return out
    R, t, fit = kabsch(r1, r0)
    out["receptor_internal_rmsd_A"] = round(float(fit), 3)
    out["peptide_rmsd_receptor_frame_A"] = round(
        float(np.sqrt((((b1 @ R.T + t) - b0) ** 2).sum(1).mean())), 3
    )
    # How much of the raw before/after RMSD was the complex simply moving.
    out["receptor_rigid_drift_A"] = round(
        float(np.linalg.norm(r1.mean(0) - r0.mean(0))), 3
    )
    return out


def one(task: dict) -> dict:
    """Re-relax one retained raw complex and recount its clashes. Runs in a subprocess."""
    from proteinfoundation.cpseq.scorer import score_structure

    pred_dir = Path(task["pred_dir"])
    out = dict(task)
    try:
        # The raw complex is gzipped under `rescorable`; this is the tier's own restore
        # path, so the scorer is handed a real PDB exactly as a rescore would.
        retmod.restore_raw_structures(pred_dir)
        raw = Path(task["raw_pdb_path"])
        if not raw.exists():
            out["status"] = "raw_missing"
            return out
        stem = raw.with_suffix("")
        sc = score_structure(
            raw,
            binder_chain=BINDER_CHAIN,
            receptor_chains=task["receptor_chains"],
            chemistry=task["chemistry"],
            requested_sequence=task["sequence"],
            refined_out_path=f"{stem}_probe_refined.pdb",
            isolated_out_path=f"{stem}_probe_isolated.pdb",
        )
        out["score_status"] = sc.get("status")
        refined = Path(f"{stem}_probe_refined.pdb")
        if not refined.exists():
            out["status"] = "no_refined"
            return out
        val = validity_report(
            refined,
            binder_chain=BINDER_CHAIN,
            receptor_chains=task["receptor_chains"],
            chemistry=task["chemistry"],
            requested_sequence=task["sequence"],
        )
        cl = val.get("clashes") or {}
        disp = sc.get("refinement_displacement") or {}
        out |= {
            "status": "ok",
            "refined_n_inter": cl.get("n_inter_clashes"),
            "refined_n_intra": cl.get("n_intra_clashes"),
            # Unconditional closest contacts, so "resolved" can be told apart from
            # "left sitting just over the bar".
            "raw_closest_A": closest_contact_A(raw, BINDER_CHAIN, task["receptor_chains"]),
            "refined_closest_A": closest_contact_A(
                refined, BINDER_CHAIN, task["receptor_chains"]
            ),
            # Whether refinement moved the peptide, so a fix can be told apart from the
            # peptide being shoved aside -- which would resolve the overlap by discarding
            # the pose the model predicted, and is the reason the gate reads raw
            # coordinates in the first place.
            "binder_ca_rmsd_A": disp.get("binder_ca_rmsd_A"),
            "receptor_ca_rmsd_A": disp.get("receptor_ca_rmsd_A"),
        }
        out |= displacement_split(raw, refined, BINDER_CHAIN, task["receptor_chains"])
        # Clean up the probe's own products; the retained tree is left as it was.
        for p in (refined, Path(f"{stem}_probe_isolated.pdb")):
            p.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        out["status"] = f"error: {type(e).__name__}: {e}"[:300]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-band", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="refined_clash_probe.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cohort = json.loads((out_dir / "cohort.json").read_text())
    sources = cohort["parent_sources"]

    # Receptor chains per complex, from each corpus's own manifest.
    import pandas as pd
    chains: dict[str, list[str]] = {}
    for ddir in sorted(set(sources.values())):
        t = pd.read_csv(Path(ddir) / "manifest" / "targets.csv")
        for r in t.to_dict("records"):
            pid = str(r["parent_id"])
            if pid in sources:
                chains[pid] = sorted(set(json.loads(r["chain_map"]).values()))

    pools: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted((out_dir / "state").glob("*/outcomes.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = classify(row)
            if c is None:
                continue
            band, d = c
            pid = row["parent_id"]
            raw = row.get("raw_pdb_path")
            if not raw or pid not in chains:
                continue
            pools[band].append({
                "parent_id": pid, "band": band, "raw_min_inter_A": d,
                "sequence": row["sequence"], "chemistry": row.get("chemistry", "mainchain"),
                "prediction_seed": row.get("prediction_seed"),
                "raw_n_inter": row.get("n_inter_clashes"),
                "raw_pdb_path": raw,
                "pred_dir": str(Path(raw).parents[4]),
                "receptor_chains": chains[pid],
            })

    rng = random.Random(args.seed)
    tasks = []
    for name, _, _ in BANDS:
        p = pools.get(name, [])
        rng.shuffle(p)
        tasks += p[: args.per_band]
    print(f"[probe] pools: " + ", ".join(f"{n}={len(pools.get(n, []))}" for n, _, _ in BANDS))
    print(f"[probe] re-relaxing {len(tasks)} structure(s) on {args.jobs} process(es); "
          f"no prediction is re-run")
    if not tasks:
        print("[probe] nothing to do")
        return 1

    done = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(one, tasks), 1):
            done.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"[probe] {i}/{len(tasks)}")

    # Report per band: did the overlap survive the refinement the pipeline already runs?
    print()
    import statistics as st

    def med(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return st.median(v) if v else None

    def fmt(v, w=6, p=2):
        return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"

    print(f"{'raw closest':12s} {'n':>4} {'overlap gone':>12} {'raw':>6} {'refined':>8} "
          f"{'receptor':>9} {'rigid':>7} {'peptide vs':>11}")
    print(f"{'(band, A)':12s} {'':>4} {'after relax':>12} {'closest contact':>15} "
          f"{'deformed':>9} {'drift':>7} {'receptor':>11}")
    summary = {}
    for name, _, _ in BANDS:
        rows = [r for r in done if r["band"] == name and r.get("status") == "ok"]
        if not rows:
            print(f"{name:12s} {0:>4}")
            continue
        clear = [r for r in rows if (r.get("refined_n_inter") or 0) == 0]
        summary[name] = {
            "n": len(rows), "resolved": len(clear),
            "resolved_frac": len(clear) / len(rows),
            "median_raw_closest_A": med(rows, "raw_closest_A"),
            "median_refined_closest_A": med(rows, "refined_closest_A"),
            "median_binder_ca_rmsd_A": med(rows, "binder_ca_rmsd_A"),
            "median_receptor_ca_rmsd_A": med(rows, "receptor_ca_rmsd_A"),
            "median_receptor_internal_rmsd_A": med(rows, "receptor_internal_rmsd_A"),
            "median_receptor_rigid_drift_A": med(rows, "receptor_rigid_drift_A"),
            "median_peptide_rmsd_receptor_frame_A": med(
                rows, "peptide_rmsd_receptor_frame_A"),
        }
        print(f"{name:12s} {len(rows):>4} {len(clear):>7}/{len(rows):<4} "
              f"{fmt(med(rows,'raw_closest_A'),6)} {fmt(med(rows,'refined_closest_A'),8)} "
              f"{fmt(med(rows,'receptor_internal_rmsd_A'),9)} "
              f"{fmt(med(rows,'receptor_rigid_drift_A'),7)} "
              f"{fmt(med(rows,'peptide_rmsd_receptor_frame_A'),11)}")

    bad = collections.Counter(r.get("status") for r in done if r.get("status") != "ok")
    if bad:
        print(f"\nnot measured: {dict(bad)}")

    path = out_dir / args.out
    path.write_text(json.dumps({"per_band": summary, "rows": done}, indent=2))
    print(f"\n[probe] -> {path}")
    print("'0 clashes' = no peptide-receptor contact under 2.2 A remains after relax, so "
          "the fold would pass the overlap check if the check were applied to the refined "
          "structure instead of the raw prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
