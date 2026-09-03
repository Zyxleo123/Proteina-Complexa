"""Did the soft-closure projection actually buy closure? Paired arms, one table.

Reads the two SDEdit arms produced by scripts/soft_closure_sdedit.sbatch -- `projected` and
`control` -- which ran the SAME grid, the SAME seeds and the SAME checkpoint on the SAME
peptides, differing only in whether the input was projected first. That pairing is the whole
point: the original sweep scored these peptides at one seed on a wider grid, so comparing
against it directly would confound the projection with the grid.

Two traps this script is written around
---------------------------------------
1. An abstained edit carries NaN, and NaN is TRUTHY in Python. Filtering rows with a bare
   truth test scores a cell where the model never proposed the requested chemistry as a
   100% success. Every rate here is gated on `requested_type_satisfied` first.
2. A projected peptide has SEVERAL replicas while the control has one structure. Reporting
   raw per-edit rates would hand the projected arm more attempts for free. The headline is
   therefore per-PEPTIDE (did any attempt close), with the per-attempt rate reported beside
   it and the attempt counts printed so the difference is visible rather than buried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SUCCESS_COL = {"mainchain": "cyc/mainchain_cn_bond_success",
               "disulfide": "cyc/disulfide_bond_success",
               "isopeptide": "cyc/isopeptide_bond_success"}
DIST_COL = {"mainchain": "cyc/mainchain_cn_dist_pred_A",
            "disulfide": "cyc/disulfide_sg_dist_pred_A",
            "isopeptide": "cyc/isopeptide_n_c_dist_pred_A"}


def load_arm(arm_dir: Path, arm: str) -> pd.DataFrame:
    files = sorted(arm_dir.glob("edits_shard*.jsonl"))
    if not files:
        return pd.DataFrame()
    rows = []
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Refuse, never skip: a NUL-corrupted row means concurrent writers shared a
                # file, and silently dropping it turns data loss into a plausible-looking table.
                raise SystemExit(f"FATAL: corrupt row {f}:{lineno}. Do not trust this run.")
    df = pd.DataFrame(rows)
    df["arm"] = arm
    # "LNR_1jrr_A_P__proj03" -> "LNR_1jrr_A_P"; the control's ids have no suffix to strip.
    df["parent"] = df["example_id"].str.split("__proj").str[0]
    return df


def rates(df: pd.DataFrame, cyc: str) -> dict:
    ok = df[df["status"] == "ok"]
    # Gate FIRST. Rows where the model proposed a different chemistry never had their geometry
    # measured, so they are abstentions, not failures -- counting them either way is a lie.
    scorable = ok[ok.get("requested_type_satisfied", 0) == 1]
    succ = SUCCESS_COL[cyc]
    closed = scorable[scorable[succ] == 1] if succ in scorable else scorable.iloc[:0]
    return {
        "n_edits": int(len(ok)),
        "n_scorable": int(len(scorable)),
        "abstention_rate": float(1 - len(scorable) / len(ok)) if len(ok) else float("nan"),
        "per_attempt_closure": float(len(closed) / len(scorable)) if len(scorable) else float("nan"),
        "n_peptides": int(ok["parent"].nunique()),
        "n_peptides_closed": int(closed["parent"].nunique()),
        "per_peptide_closure": (float(closed["parent"].nunique() / ok["parent"].nunique())
                                if ok["parent"].nunique() else float("nan")),
        "best_dist_A": (float(scorable[DIST_COL[cyc]].min())
                        if DIST_COL[cyc] in scorable and len(scorable) else float("nan")),
        "median_contact_retention": (float(scorable["contact_retention"].median())
                                     if "contact_retention" in scorable and len(scorable)
                                     else float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="Holds projected/ and control/.")
    ap.add_argument("--proj-dir", required=True, help="The projection run, for its summary.")
    ap.add_argument("--out", default=None, help="Defaults to <results-dir>/summary.json")
    ap.add_argument("--cyc-type", default="mainchain", choices=sorted(SUCCESS_COL))
    args = ap.parse_args()

    res = Path(args.results_dir)
    proj = load_arm(res / "projected", "projected")
    ctrl = load_arm(res / "control", "control")
    if ctrl.empty:
        raise SystemExit(f"FATAL: no control rows under {res / 'control'} -- without the "
                         f"paired arm there is nothing to compare the projection against.")

    out = {"cyc_type": args.cyc_type, "arms": {}}
    proj_summary = Path(args.proj_dir) / "projection_summary.json"
    if proj_summary.is_file():
        out["projection"] = json.loads(proj_summary.read_text())

    if proj.empty:
        out["arms"]["projected"] = None
        print("The projected arm produced no edits -- the projection accepted no replica.\n"
              "Reporting the control arm alone; the comparison is not available.\n", flush=True)
    else:
        out["arms"]["projected"] = rates(proj, args.cyc_type)
    out["arms"]["control"] = rates(ctrl, args.cyc_type)

    # ---- per-peptide table ------------------------------------------------------------
    succ, dist = SUCCESS_COL[args.cyc_type], DIST_COL[args.cyc_type]
    per = {}
    for name, df in (("control", ctrl), ("projected", proj)):
        if df.empty:
            continue
        ok = df[df["status"] == "ok"]
        sc = ok[ok.get("requested_type_satisfied", 0) == 1]
        for parent, g in ok.groupby("parent"):
            gs = sc[sc["parent"] == parent]
            e = per.setdefault(parent, {})
            e[f"{name}_attempts"] = int(len(ok[ok["parent"] == parent]))
            e[f"{name}_scorable"] = int(len(gs))
            e[f"{name}_closed"] = int((gs[succ] == 1).sum()) if succ in gs and len(gs) else 0
            e[f"{name}_best_A"] = (float(gs[dist].min()) if dist in gs and len(gs) else float("nan"))
    out["per_peptide"] = per

    hdr = (f"{'peptide':22s} {'ctrl clo/scor':>14s} {'ctrl best':>10s} "
           f"{'proj clo/scor':>14s} {'proj best':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for parent in sorted(per, key=lambda k: -per[k].get("control_best_A", 0) if
                         per[k].get("control_best_A", 0) == per[k].get("control_best_A", 0) else 0):
        e = per[parent]
        print(f"{parent:22s} "
              f"{e.get('control_closed', 0):6d}/{e.get('control_scorable', 0):<7d} "
              f"{e.get('control_best_A', float('nan')):10.2f} "
              f"{e.get('projected_closed', 0):6d}/{e.get('projected_scorable', 0):<7d} "
              f"{e.get('projected_best_A', float('nan')):10.2f}")

    print("\n" + json.dumps(out["arms"], indent=2))
    dest = Path(args.out) if args.out else res / "summary.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
