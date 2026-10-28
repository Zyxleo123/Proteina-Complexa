#!/usr/bin/env python3
"""Within-condition noise vs between-condition difference, for the seq-cond generator.

Reads the four runs `scripts/cpseq_seqcond_determinism.sbatch` produces and answers two
questions that have to be settled before any searched-vs-native number is readable:

**Is `--seq-source` inert when it requests the native peptides?** It is a content-level
no-op (0 substitutions, every write read back in-process), so if `|inject_a - inject_b|`
and `|control_a - control_b|` are the same size and the between-condition gap is no larger,
the plumbing costs nothing. If injected-native sits outside the controls' spread, the
plumbing perturbs something -- allocator state, RNG draw order -- and that has to be found
before it is mistaken for a design effect.

**What is the noise floor?** Two identical runs of this eval were measured 0.104 apart in
`cond_exact_frac`. A searched-vs-native difference below the within-condition spread is not
readable at this sample count, whichever way it points.

The comparison is only valid inside one GPU model -- the first attempt at this measurement
was confounded by two runs landing on different nodes -- so this REFUSES to compare runs
whose recorded hostname and GPU differ, rather than printing a number that averages two
pieces of hardware.

Energies are not involved here; this reads the generator's own closure and compliance
metrics. No GPU.

Usage:
    python script_utils/cpseq_seqcond_determinism_report.py \\
        --root /data/user_data/yixiz/cpseq/transfer/determinism
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

METRICS = [
    ("cyc/mainchain_cn_bond_success", "N-C bond success"),
    ("cyc/mainchain_cn_dist_pred_A", "N-C distance, A"),
    ("cyc/cyc_cb_window_success", "CB 3-8 A window"),
    ("cyc/cyc_cb_dist_pred_A", "CB distance, A"),
    ("seq/cond_recovery", "sequence recovery"),
    ("seq/cond_exact_frac", "exact-sequence frac"),
    ("iface/burial", "interface burial"),
    ("iface/n_clash", "interface clashes"),
    ("place/ca_rmsd_nm", "placement RMSD, nm"),
    ("div/ca_rmsd_nm", "sample diversity, nm"),
]

INVARIANTS = [
    ("cyc/n_valid_mainchain", "valid mainchain samples"),
    ("seq/n_cond_res", "conditioned residues"),
]


def read_run(d: Path) -> dict | None:
    files = sorted(glob.glob(str(d / "closure_rows_*.jsonl")))
    if not files:
        return None
    rows = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return None
    # One batch covers all 12 complexes at --n-complexes 12; if a run ever holds more,
    # averaging them would hide which batch moved, so the batch count is reported.
    metrics: dict[str, float] = {}
    for key in [k for k, _ in METRICS] + [k for k, _ in INVARIANTS]:
        vals = [r["metrics"][f"val_gen/{key}"] for r in rows
                if isinstance(r.get("metrics", {}).get(f"val_gen/{key}"), (int, float))]
        if vals:
            metrics[key] = sum(vals) / len(vals)
    node = {}
    np_ = d / "node.json"
    if np_.exists():
        node = json.loads(np_.read_text())
    return {
        "name": d.name,
        "n_batches": len(rows),
        "metrics": metrics,
        "seq_source": rows[0].get("seq_source"),
        "seq_source_label": rows[0].get("seq_source_label"),
        "requested": rows[0].get("requested"),
        "node": node,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--control", default="control_a,control_b")
    ap.add_argument("--inject", default="inject_a,inject_b")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-mixed-gpu", action="store_true",
                    help="Compare anyway when the runs did not share one GPU model. The "
                         "resulting spread mixes hardware with run-to-run noise.")
    args = ap.parse_args()

    root = Path(args.root)
    names = [n.strip() for n in (args.control + "," + args.inject).split(",") if n.strip()]
    runs = {}
    for n in names:
        r = read_run(root / n)
        if r is None:
            print(f"[determinism] {n}: no closure rows yet")
            continue
        runs[n] = r
    if len(runs) < 2:
        print("[determinism] need at least two completed runs")
        return 1

    # Hardware invariant, enforced rather than assumed.
    sigs = {n: (r["node"].get("hostname"), r["node"].get("gpu")) for n, r in runs.items()}
    distinct = {s for s in sigs.values() if s != (None, None)}
    if len(distinct) > 1:
        print("[determinism] runs did NOT share one node/GPU:")
        for n, s in sigs.items():
            print(f"  {n}: host={s[0]} gpu={s[1]}")
        if not args.allow_mixed_gpu:
            print("[determinism] refusing to compare; a 400-step ODE diverges across GPU "
                  "models at a fixed seed, so this spread would not be run-to-run noise. "
                  "Re-run on one node, or pass --allow-mixed-gpu to see it anyway.")
            return 1
        print("[determinism] --allow-mixed-gpu: proceeding, spread mixes hardware in")

    ctl = [n for n in args.control.split(",") if n.strip() in runs]
    inj = [n for n in args.inject.split(",") if n.strip() in runs]

    # Provenance: an inject run must actually have injected, and only native peptides.
    for n in inj:
        req = runs[n].get("requested") or []
        if not req:
            print(f"[determinism] {n} records no injection; it is not an inject run")
            return 1
        subs = {r.get("n_substitutions") for r in req if r.get("injected")}
        if subs != {0}:
            print(f"[determinism] {n} injected non-native sequences ({sorted(subs)} "
                  "substitutions); this measures a design, not the plumbing")
            return 1
    for n in ctl:
        if runs[n].get("seq_source"):
            print(f"[determinism] {n} passed --seq-source; it is not a control")
            return 1

    print(f"[determinism] {len(ctl)} control + {len(inj)} injected-native run(s)")
    host, gpu = next(iter(distinct)) if distinct else (None, None)
    print(f"[determinism] host={host} gpu={gpu}")
    print("[determinism] injected runs requested each parent's OWN native peptide "
          "(0 substitutions), so any difference is the plumbing, not a design\n")

    print(f"{'invariant':28} " + " ".join(f"{n:>11}" for n in ctl + inj) + "   exact")
    for key, label in INVARIANTS:
        vals = [runs[n]["metrics"].get(key) for n in ctl + inj]
        shown = " ".join("          -" if v is None else f"{v:>11.4f}" for v in vals)
        ok = len({v for v in vals if v is not None}) == 1
        print(f"{label:28} {shown}   {'yes' if ok else 'NO'}")

    print()
    hdr = f"{'metric':24}" + " ".join(f"{n:>11}" for n in ctl + inj)
    print(hdr + f" {'ctl spread':>11} {'inj spread':>11} {'between':>11} {'readable':>9}")
    out_rows = []
    for key, label in METRICS:
        cv = [runs[n]["metrics"].get(key) for n in ctl]
        iv = [runs[n]["metrics"].get(key) for n in inj]
        if any(v is None for v in cv + iv):
            continue
        c_spread = max(cv) - min(cv) if len(cv) > 1 else float("nan")
        i_spread = max(iv) - min(iv) if len(iv) > 1 else float("nan")
        between = abs(sum(iv) / len(iv) - sum(cv) / len(cv))
        floor = max(x for x in (c_spread, i_spread) if x == x) if (
            c_spread == c_spread or i_spread == i_spread) else float("nan")
        # "Readable" means the between-condition gap stands clear of the within-condition
        # spread. For a no-op it should NOT: a plumbing change that shows up above the
        # noise is a plumbing problem.
        verdict = "-" if floor != floor else ("ABOVE" if between > floor else "within")
        vals = " ".join(f"{v:>11.4f}" for v in cv + iv)
        print(f"{label:24}{vals} {c_spread:>11.4f} {i_spread:>11.4f} {between:>11.4f} "
              f"{verdict:>9}")
        out_rows.append({"metric": key, "label": label, "control": cv, "inject": iv,
                         "control_spread": c_spread, "inject_spread": i_spread,
                         "between": between, "verdict": verdict})

    above = [r["label"] for r in out_rows if r["verdict"] == "ABOVE"]
    print()
    if above:
        print(f"[determinism] {len(above)}/{len(out_rows)} metrics show a between-condition "
              f"gap LARGER than the within-condition spread: {', '.join(above)}")
        print("[determinism] requesting the native peptide is a content-level no-op, so a gap "
              "above the noise points at the injection path (allocator or RNG draw order), "
              "not at the sequences. Worth finding before reading any design effect.")
    else:
        print(f"[determinism] no metric's between-condition gap exceeds the within-condition "
              f"spread: --seq-source is inert at this sample count, as it should be")
    print("[determinism] NOISE FLOOR -- the within-condition spreads above are the smallest "
          "searched-vs-native difference this eval can resolve at these settings. Raise "
          "--n-repeat until the floor sits below the effect you need to see.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "root": str(root), "host": host, "gpu": gpu,
            "control": ctl, "inject": inj,
            "n_batches": {n: runs[n]["n_batches"] for n in runs},
            "rows": out_rows,
            "metrics_above_noise": above,
        }, indent=1) + "\n")
        print(f"[determinism] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
