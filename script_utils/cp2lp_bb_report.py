"""The arm comparison: completed-LP quality, binned by terminal gap AND by edit size.

READS ONLY FILES ALREADY ON DISK, so a threshold can change and every table and figure
regenerates without sampling or packing anything.

WHY BOTH BINNINGS, AND WHY NEITHER ALONE WILL DO.

  * A pooled mean rewards an arm for NOT EDITING. The export gate in the previous generation
    was measured to select for non-edits -- the control's 85.5% acceptance was a 0.5 A
    identity map -- so "arm X scores higher" is uninterpretable until the comparison is made
    at matched edit size.
  * A gap binning is needed for the same reason in the other direction: opening the ring is
    the task, and an arm that opens it less has an easier packing problem.

So every quality column is reported inside EDIT bins and inside GAP bins, and the headline
claim is only ever made where the bins overlap.

ALSO REPORTED, because each answers a question a single success rate cannot:

  all attempts vs accepted attempts, with counts     -- an acceptance rate of 1/1 is not 50/50
  packing success CONDITIONAL on raw backbone validity -- separates "the generator emitted a
                                                        broken chain" from "the packer failed"
  success per source at the fixed sampling budget    -- what a user of the model would get
  best valid quality per source                      -- what the model can do at all
  the source-torsion reference row                   -- template drift and near-identity
  within-source diversity                            -- a dead noise path is invisible elsewhere

A LOWER BACKBONE DISCRIMINATOR ACCURACY IS NOT A WIN and is not reported as one: `d_acc` is a
training diagnostic and appears here only if it was logged, labelled as such.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Terminal-gap bins, nm. Kept the same as the ongoing experiments' so a row here can be put
#: beside one from the SDEdit/de-novo tables without re-binning.
GAP_BINS = [(0.0, 0.45), (0.45, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, math.inf)]

#: Edit-size bins, nm of CA RMSD from the source. The lowest bin exists specifically to make a
#: near-identity visible rather than averaged away: 0.05 nm is the 0.5 A scale the previous
#: generation's export gate was measured to select for.
EDIT_BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, math.inf)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packed-dir", required=True, help="directory of packed__*.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt-tag", default=None, help="restrict to one milestone")
    p.add_argument("--min-open-gap-nm", type=float, default=0.45)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def load_rows(packed_dir: Path, ckpt_tag: str | None) -> list[dict]:
    """Every packed row. A corrupt line is REFUSED, never skipped."""
    rows = []
    files = sorted(packed_dir.glob("packed__*.jsonl"))
    if not files:
        raise SystemExit(f"GATE: no packed__*.jsonl under {packed_dir} -- the packing stage never ran.")
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception as exc:
                raise SystemExit(
                    f"REFUSING to report: {f}:{i} is corrupt ({exc}). Concurrent appends to "
                    "one file NUL-corrupt rows here, and a skipped row makes every "
                    "denominator wrong. Re-run the packing stage with one file per job."
                )
            if ckpt_tag and r.get("ckpt_tag") != ckpt_tag:
                continue
            rows.append(r)
    return rows


def selected(row: dict) -> dict | None:
    """The representative packing outcome of one attempt, by the predeclared order."""
    for r in row.get("results", []):
        if r.get("seed") == row.get("selected_seed") and r.get("ok"):
            return r
    return None


def joint_success(row: dict, sel: dict | None, min_gap_nm: float) -> tuple[bool, str]:
    """The joint usable-triplet gate, from the existing calibrated criteria.

    Five conditions, all necessary. A triplet that fails any one of them is not usable as a
    training pair for the downstream task, which is what this whole arm exists to produce:

      nontrivial opening   -- the peptide is actually open, not stacked end-on-end
      intact linear backbone -- RAW, from the generator, before the packer touched it
      exact source sequence
      acceptable completed-structure sterics
      interface preservation -- the source's contacts largely survived
    """
    if sel is None:
        return False, "packing_failed"
    if float(row.get("terminal_gap_nm", 0.0)) < min_gap_nm:
        return False, "not_open"
    if float(row.get("bb_chain_intact", 0.0)) < 1.0:
        return False, "raw_backbone_broken"
    if not sel.get("sequence_exact", True):
        return False, "sequence_mismatch"
    if int(sel.get("clash_receptor_count", 0)) > 0 or int(sel.get("clash_self_count", 0)) > 0:
        return False, "clash"
    ret = sel.get("contact_retention", float("nan"))
    if ret == ret and float(ret) < 0.5:
        return False, "contacts_lost"
    return True, ""


def _bin(value: float, bins) -> str:
    for lo, hi in bins:
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})" if hi != math.inf else f">={lo:g}"
    return "nan"


def _agg(vals: list[float]) -> dict:
    v = [x for x in vals if x == x]
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": len(v), "median": float(np.median(a)), "mean": float(a.mean()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}


def main() -> int:  # noqa: C901
    args = parse_args()
    rows = load_rows(Path(args.packed_dir), args.ckpt_tag)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[report] {len(rows)} packed attempts")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[str(r.get("arm", "?"))].append(r)

    report: dict = {"ckpt_tag": args.ckpt_tag, "n_rows": len(rows), "arms": {}}

    print("\n=== ALL ATTEMPTS vs ACCEPTED, with counts ===")
    hdr = f"{'arm':<16s} {'attempts':>8s} {'packed':>7s} {'valid':>6s} {'joint':>6s} {'joint%':>7s}"
    print(hdr)
    for arm, rs in sorted(by_arm.items()):
        n = len(rs)
        n_packed = sum(1 for r in rs if selected(r) is not None)
        n_valid = sum(1 for r in rs if (s := selected(r)) and s.get("valid"))
        js = [joint_success(r, selected(r), args.min_open_gap_nm) for r in rs]
        n_joint = sum(1 for ok, _ in js if ok)
        print(f"{arm:<16s} {n:>8d} {n_packed:>7d} {n_valid:>6d} {n_joint:>6d} "
              f"{100.0 * n_joint / max(n, 1):>6.1f}%")
        why = defaultdict(int)
        for ok, reason in js:
            if not ok:
                why[reason] += 1
        a: dict = {
            "attempts": n, "packed_ok": n_packed, "valid": n_valid, "joint_success": n_joint,
            "joint_success_frac": n_joint / max(n, 1),
            "failure_reasons": dict(sorted(why.items(), key=lambda kv: -kv[1])),
        }

        # PACKING SUCCESS CONDITIONAL ON RAW BACKBONE VALIDITY. Without the conditioning,
        # "packing failed" and "the generator emitted a broken chain" are one number, and the
        # two call for completely different work.
        intact = [r for r in rs if float(r.get("bb_chain_intact", 0.0)) >= 1.0]
        broken = [r for r in rs if float(r.get("bb_chain_intact", 0.0)) < 1.0]
        a["raw_backbone_intact"] = len(intact)
        a["raw_backbone_broken"] = len(broken)
        for label, subset in (("given_intact", intact), ("given_broken", broken)):
            if subset:
                a[f"pack_valid_{label}"] = sum(
                    1 for r in subset if (s := selected(r)) and s.get("valid")
                ) / len(subset)

        # PER SOURCE at the fixed budget: what a user actually gets from N attempts.
        per_src: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            per_src[str(r.get("example_id"))].append(r)
        a["n_sources"] = len(per_src)
        a["sources_with_any_joint"] = sum(
            1 for v in per_src.values()
            if any(joint_success(r, selected(r), args.min_open_gap_nm)[0] for r in v)
        )
        a["success_per_source_frac"] = a["sources_with_any_joint"] / max(len(per_src), 1)

        # BEST VALID QUALITY PER SOURCE: the model's ceiling, not its average.
        best = []
        for v in per_src.values():
            cands = [
                s.get("dG_separated") for r in v
                if (s := selected(r)) and s.get("valid") and s.get("dG_separated") == s.get("dG_separated")
            ]
            if cands:
                best.append(min(cands))
        a["best_valid_dG_per_source"] = _agg(best)

        a["within_source_diversity_nm"] = _agg([r.get("within_source_diversity_nm", float("nan")) for r in rs])
        a["edit_ca_rmsd_nm"] = _agg([r.get("edit_ca_rmsd_nm", float("nan")) for r in rs])
        a["terminal_gap_nm"] = _agg([r.get("terminal_gap_nm", float("nan")) for r in rs])
        a["contact_retention"] = _agg([
            s.get("contact_retention", float("nan")) for r in rs if (s := selected(r))
        ])
        a["peptide_score_per_res"] = _agg([
            s.get("peptide_score_per_res", float("nan")) for r in rs if (s := selected(r))
        ])
        a["pack_seconds"] = _agg([
            s.get("pack_seconds", float("nan")) for r in rs if (s := selected(r))
        ])
        report["arms"][arm] = a

    for arm, rs in sorted(by_arm.items()):
        why = report["arms"][arm]["failure_reasons"]
        if why:
            print(f"  {arm:<16s} failures: {why}")

    # ------------------------------------------------------------------ binned tables
    for label, key, bins in (
        ("EDIT-BINNED (CA RMSD from source, nm)", "edit_ca_rmsd_nm", EDIT_BINS),
        ("GAP-BINNED (terminal gap, nm)", "terminal_gap_nm", GAP_BINS),
    ):
        print(f"\n=== {label} ===")
        print("A headline claim is only made where BOTH binnings agree -- a pooled gain can "
              "always be explained by one arm editing less.")
        names = sorted(by_arm)
        bin_labels = [f"[{lo:g},{hi:g})" if hi != math.inf else f">={lo:g}" for lo, hi in bins]
        print(f"{'bin':<14s}" + "".join(f"{n[:14]:>18s}" for n in names))
        table: dict = {}
        for bl in bin_labels:
            cells = []
            for arm in names:
                sub = [r for r in by_arm[arm] if _bin(float(r.get(key, float("nan")) or 0.0), bins) == bl]
                if not sub:
                    cells.append(f"{'-':>18s}")
                    continue
                nj = sum(1 for r in sub if joint_success(r, selected(r), args.min_open_gap_nm)[0])
                cells.append(f"{nj}/{len(sub)} ({100.0 * nj / len(sub):.0f}%)".rjust(18))
                table.setdefault(bl, {})[arm] = {"n": len(sub), "joint": nj}
            print(f"{bl:<14s}" + "".join(cells))
        report[f"binned_{key}"] = table

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[report] wrote {out / 'report.json'}")

    if args.plot:
        _plot(report, by_arm, out, args)
    return 0


def _plot(report: dict, by_arm: dict, out: Path, args) -> None:
    """Joint success by edit bin and by gap bin, side by side. Redrawable from report.json."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[report] no matplotlib ({exc}); skipping the figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (key, bins, title) in zip(axes, (
        ("edit_ca_rmsd_nm", EDIT_BINS, "by edit size (CA RMSD from source, nm)"),
        ("terminal_gap_nm", GAP_BINS, "by terminal gap (nm)"),
    )):
        table = report.get(f"binned_{key}", {})
        labels = list(table)
        arms = sorted(by_arm)
        w = 0.8 / max(len(arms), 1)
        for j, arm in enumerate(arms):
            ys = [
                100.0 * table[bl][arm]["joint"] / table[bl][arm]["n"]
                if arm in table.get(bl, {}) and table[bl][arm]["n"] else 0.0
                for bl in labels
            ]
            ns = [table.get(bl, {}).get(arm, {}).get("n", 0) for bl in labels]
            xs = np.arange(len(labels)) + j * w
            ax.bar(xs, ys, width=w, label=arm)
            for x, y, nn in zip(xs, ys, ns):
                if nn:
                    ax.text(x, y + 1, str(nn), ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(labels)) + 0.4 - w / 2)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("joint usable-triplet success (%)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    # The per-bin n is printed on every bar, because a 100% bar over n=1 and one over n=40
    # are not the same result and a bar chart hides the difference.
    fig.suptitle(
        f"CP->LP backbone arms, {report.get('ckpt_tag') or 'all milestones'} "
        "(bar labels are n; claims need both panels to agree)", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "arm_comparison.png", dpi=150)
    print(f"[report] wrote {out / 'arm_comparison.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
