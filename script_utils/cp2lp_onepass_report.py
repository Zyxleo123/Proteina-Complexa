"""The five-arm comparison: valid yield, retention, geometry, clashes, per-parent variation.

Reads exported JSONL rows (and, for diversity, the exported PDBs) and writes one table, one
figure and one JSON summary. CPU only and fully redrawable: nothing here regenerates a
peptide, so a threshold can change and the table can be rebuilt in seconds.

THREE RULES THIS REPORT ENFORCES, each because the opposite is the easy mistake.

*Raw attempts and accepted exports are both reported, always.* An arm with one excellent
survivor and near-zero yield must be recognisable, and it is not if the table shows only
the accepted population -- that arm would top every quality column. Every quality quantity
therefore appears twice, over all attempts and over accepted exports, with the acceptance
rate beside them.

*Rejected rows are never dropped, and a corrupt row is never skipped.* Concurrent
O_APPEND to a shared file NUL-corrupts rows on this filesystem; a reader that skips
unparseable lines turns lost data into a silently smaller denominator. This REFUSES,
naming the file and the line.

*Every addon is compared against the plain one-pass GAN.* The table's delta columns are
against ``--reference-arm``, never against the previous addon, and the no-adversary arm is
carried through as the separate question of whether the adversary is worth having at all.

Run (CPU):
    .venv/bin/python script_utils/cp2lp_onepass_report.py \\
        --arm onepass_gan=$TRIPLETS/onepass_gan__last-EMA \\
        --arm onepass_gan_ar1=$TRIPLETS/onepass_gan_ar1__last-EMA \\
        --out-dir $REPORT/onepass_last-EMA
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

#: Gates are calibrated, not guesses. 0.02 nm on every (i, i+1) bond passes 97% of real
#: crystal LPs through the same code (median deviation 0.0033 nm), and the clash threshold
#: is 0.20 nm because 0.27 nm rejects 43 of 60 native complexes.
CLASH_THRESHOLD_NM = 0.20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="An arm and the directory holding its triplets*.jsonl. Repeatable.",
    )
    p.add_argument("--real-reference", default=None, help="Directory with real_reference*.jsonl, if measured.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--reference-arm", default="onepass_gan", help="The arm every addon is compared against.")
    p.add_argument("--control-arm", default="onepass_contactonly", help="The no-adversary arm.")
    p.add_argument("--no-diversity", action="store_true", help="Skip within-parent diversity (needs the PDBs).")
    return p.parse_args()


# ------------------------------------------------------------------------ loading


def load_rows(directory: Path, pattern: str = "triplets*.jsonl") -> list[dict]:
    """Every row under ``directory``, refusing corrupt lines rather than skipping them."""
    files = sorted(directory.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    rows: list[dict] = []
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            if "\x00" in line:
                raise ValueError(
                    f"{f}:{lineno} contains a NUL byte. Concurrent O_APPEND to one file "
                    "corrupts rows on this filesystem; one manifest per shard is the fix. "
                    "REFUSING rather than skipping -- a skipped row is a silently smaller "
                    "denominator on every rate in this table."
                )
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{f}:{lineno} is not valid JSON ({exc}); refusing to skip it.") from exc
    return rows


def load_summary(directory: Path) -> dict:
    f = directory / "generate_summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


def ca_from_pdb(path: Path, chain: str = "B") -> np.ndarray | None:
    """CA coordinates of one chain, in Angstrom. ``None`` if the file is absent."""
    if not path.exists():
        return None
    xs = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            xs.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(xs, dtype=float) if xs else None


# ------------------------------------------------------------------------ statistics


def _finite(values) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return a[np.isfinite(a)]


def _q(values, key: str) -> dict[str, float]:
    a = _finite(values)
    if a.size == 0:
        return {f"{key}_n": 0}
    return {
        f"{key}_n": int(a.size),
        f"{key}_median": float(np.median(a)),
        f"{key}_mean": float(a.mean()),
        f"{key}_p10": float(np.percentile(a, 10)),
        f"{key}_p90": float(np.percentile(a, 90)),
    }


def arm_stats(rows: list[dict], directory: Path, with_diversity: bool) -> dict:
    """Every column of the comparison for one arm."""
    n = len(rows)
    acc = [r for r in rows if r.get("accepted")]
    out: dict = {
        "n_attempts": n,
        "n_accepted": len(acc),
        "acceptance_rate": (len(acc) / n) if n else 0.0,
    }

    # ---- per-gate pass rates. Derived from the explicit gate booleans when present, and
    # from the reason strings otherwise, so manifests written before the gates were recorded
    # still read correctly.
    def gate(key: str, reason: str) -> float:
        got = [r["gates"][key] for r in rows if isinstance(r.get("gates"), dict) and key in r["gates"]]
        if got:
            return float(np.mean(got))
        return float(np.mean([reason not in (r.get("reject_reasons") or []) for r in rows])) if n else 0.0

    out["gate_sequence_exact_rate"] = gate("sequence_ok", "sequence_mismatch")
    out["gate_open_chain_rate"] = gate("linear_ok", "not_linear")
    out["gate_chain_intact_rate"] = gate("chain_intact", "backbone_broken")

    hist: dict[str, int] = defaultdict(int)
    for r in rows:
        for reason in r.get("reject_reasons") or []:
            hist[reason] += 1
    out["reject_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))

    # ---- quality, over ALL attempts and over ACCEPTED exports. Both, always: an arm with
    # one excellent survivor and no yield tops every accepted-only column.
    fields = [
        ("seq_identity_to_source", "seq_identity"),
        ("peptide_bond_mae_nm", "bond_mae_nm"),
        ("terminal_gap_nm", "terminal_gap_nm"),
        ("terminal_bond_nm", "terminal_bond_nm"),
        ("clash_min_inter_nm", "min_pocket_dist_nm"),
        ("clash_intra_sum_nm", "intra_clash_nm"),
        ("contact_retention_frac", "contact_retention"),
        ("ca_rmsd_from_source_nm", "ca_drift_nm"),
        ("interface_ca_rmsd_from_source_nm", "iface_drift_nm"),
    ]
    for src, label in fields:
        out.update({f"all_{k}": v for k, v in _q([r["scores"].get(src) for r in rows], label).items()})
        out.update({f"acc_{k}": v for k, v in _q([r["scores"].get(src) for r in acc], label).items()})

    clash_vals = _finite([r["scores"].get("clash_min_inter_nm") for r in rows])
    out["all_clash_fraction"] = float((clash_vals < CLASH_THRESHOLD_NM).mean()) if clash_vals.size else None
    clash_acc = _finite([r["scores"].get("clash_min_inter_nm") for r in acc])
    out["acc_clash_fraction"] = float((clash_acc < CLASH_THRESHOLD_NM).mean()) if clash_acc.size else None

    # ---- terminal gap BY LENGTH. The reachable separation scales with chain length, so a
    # pooled median is partly a statement about the manifest's length mix.
    by_len: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        g = r["scores"].get("terminal_gap_nm")
        if g is not None and math.isfinite(g):
            by_len[int(r["peptide_length"])].append(g)
    out["terminal_gap_by_length"] = {
        str(L): {"n": len(v), "median": float(np.median(v))} for L, v in sorted(by_len.items())
    }

    # ---- PER PARENT: yield, and the best retention among that parent's seeds. The second is
    # what a triplet-manufacturing pipeline actually gets to keep, since it may search seeds
    # and take the best; the first says how often the search finds anything at all.
    per_parent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_parent[str(r["source_id"])].append(r)
    yields, best_ret, n_valid_per_parent = [], [], []
    for sid, rs in per_parent.items():
        ok = [r for r in rs if r.get("accepted")]
        yields.append(len(ok) / max(1, len(rs)))
        n_valid_per_parent.append(len(ok))
        rets = _finite([r["scores"].get("contact_retention_frac") for r in ok])
        if rets.size:
            best_ret.append(float(rets.max()))
    out["n_parents"] = len(per_parent)
    out["parents_with_any_valid"] = int(sum(1 for v in n_valid_per_parent if v > 0))
    out["coverage"] = out["parents_with_any_valid"] / max(1, out["n_parents"])
    out.update(_q(yields, "per_parent_valid_yield"))
    out.update(_q(best_ret, "per_parent_best_valid_retention"))

    # ---- WITHIN-PARENT DIVERSITY, over VALID LPs only. Diversity among broken chains is
    # distortion, not variety: a generator that emits different kinds of wreckage would
    # otherwise score as the most diverse arm in the table.
    if with_diversity:
        divs = []
        for sid, rs in per_parent.items():
            coords = []
            for r in rs:
                if not r.get("accepted") or not r.get("gen_lp_pdb"):
                    continue
                ca = ca_from_pdb(Path(r["gen_lp_pdb"]))
                if ca is not None and ca.shape[0] >= 3:
                    coords.append(ca)
            if len(coords) < 2:
                continue
            L = min(c.shape[0] for c in coords)
            # Distance-matrix RMSD: needs no superposition, and two samples differing only
            # by a rigid motion in the pocket frame are correctly scored as identical in
            # CONFORMATION. Pose differences are reported separately as ca_drift.
            dms = [np.linalg.norm(c[:L, None, :] - c[None, :L, :], axis=-1) for c in coords]
            pair = [
                float(np.sqrt(((dms[i] - dms[j]) ** 2).mean()))
                for i in range(len(dms))
                for j in range(i + 1, len(dms))
            ]
            if pair:
                divs.append(float(np.mean(pair)) / 10.0)  # A -> nm
        out.update(_q(divs, "within_parent_diversity_nm"))
        out["n_parents_with_2plus_valid"] = len(divs)

    # ---- cost, from the generation summary. Reported SEPARATELY from quality: the addons
    # require different computation, and folding cost into a quality verdict hides which
    # one is being paid for.
    summary = load_summary(directory)
    for key in (
        "g_updates",
        "seconds_per_output",
        "peak_gpu_gib",
        "n_generation_calls_per_sample",
        "mode",
        "addon",
        "adversarial",
        "ckpt",
        "n_parents_seen",
    ):
        if key in summary:
            out[f"cost_{key}" if key in ("seconds_per_output", "peak_gpu_gib") else key] = summary[key]
    return out


def real_reference_stats(directory: Path) -> dict | None:
    """The real bound LP column, measured by the same code that measures the generated ones."""
    try:
        rows = load_rows(directory, pattern="real_reference*.jsonl")
    except FileNotFoundError:
        return None
    out = {"n_attempts": len(rows), "n_accepted": None, "acceptance_rate": None}
    out["gate_chain_intact_rate"] = float(np.mean([bool(r["scores"].get("chain_intact")) for r in rows]))
    for src, label in (
        ("peptide_bond_mae_nm", "bond_mae_nm"),
        ("terminal_gap_nm", "terminal_gap_nm"),
        ("clash_min_inter_nm", "min_pocket_dist_nm"),
        ("clash_intra_sum_nm", "intra_clash_nm"),
    ):
        out.update({f"all_{k}": v for k, v in _q([r["scores"].get(src) for r in rows], label).items()})
    return out


# ------------------------------------------------------------------------- rendering

#: (label, key, "higher"|"lower"|None). None means "read it, do not rank it".
TABLE = [
    ("G updates", "g_updates", None),
    ("attempts", "n_attempts", None),
    ("accepted", "n_accepted", None),
    ("acceptance rate", "acceptance_rate", "higher"),
    ("  gate: exact sequence", "gate_sequence_exact_rate", "higher"),
    ("  gate: open chain", "gate_open_chain_rate", "higher"),
    ("  gate: chain intact", "gate_chain_intact_rate", "higher"),
    ("bond MAE nm (all)", "all_bond_mae_nm_median", "lower"),
    ("bond MAE nm (accepted)", "acc_bond_mae_nm_median", "lower"),
    ("terminal gap nm (all)", "all_terminal_gap_nm_median", None),
    ("terminal C-N nm (all)", "all_terminal_bond_nm_median", "higher"),
    ("min pocket dist nm (all)", "all_min_pocket_dist_nm_median", "higher"),
    ("clash fraction (all)", "all_clash_fraction", "lower"),
    ("intra clash nm (all)", "all_intra_clash_nm_median", "lower"),
    ("seq identity (all)", "all_seq_identity_median", "higher"),
    ("contact retention (all)", "all_contact_retention_median", "higher"),
    ("contact retention (accepted)", "acc_contact_retention_median", "higher"),
    ("CA drift nm (all)", "all_ca_drift_nm_median", None),
    ("interface drift nm (all)", "all_iface_drift_nm_median", None),
    ("parents with any valid", "parents_with_any_valid", "higher"),
    ("coverage", "coverage", "higher"),
    ("per-parent yield (median)", "per_parent_valid_yield_median", "higher"),
    ("best valid retention (median)", "per_parent_best_valid_retention_median", "higher"),
    ("within-parent diversity nm", "within_parent_diversity_nm_median", None),
    ("seconds / output", "cost_seconds_per_output", "lower"),
    ("peak GPU GiB", "cost_peak_gpu_gib", "lower"),
    ("net calls / sample", "n_generation_calls_per_sample", None),
]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def _short(name: str) -> str:
    """Column header. The `onepass_` prefix is the same on every arm and its 8 characters
    are exactly what pushes `gan_nce` and `gan_noise` past the column width into the same
    ambiguous stub."""
    return name.replace("onepass_", "")


def render_table(stats: dict[str, dict], order: list[str], real: dict | None) -> str:
    cols = [_short(a) for a in order] + (["real LP"] if real else [])
    lines = []
    header = "| " + "metric".ljust(30) + " | " + " | ".join(c.ljust(14)[:14] for c in cols) + " |"
    lines.append(header)
    lines.append("|" + "-" * 32 + "|" + "|".join(["-" * 16] * len(cols)) + "|")
    for label, key, _better in TABLE:
        vals = [fmt(stats[a].get(key)) for a in order]
        if real:
            vals.append(fmt(real.get(key)))
        lines.append("| " + label.ljust(30) + " | " + " | ".join(v.ljust(14)[:14] for v in vals) + " |")
    return "\n".join(lines)


def render_deltas(stats: dict[str, dict], reference: str, order: list[str]) -> str:
    """Each addon minus the plain one-pass GAN. Never addon minus previous addon."""
    if reference not in stats:
        return f"(no reference arm {reference!r} in this set; no deltas)"
    lines = [f"Deltas against {_short(reference)} (positive = the addon's value is larger):", ""]
    others = [a for a in order if a != reference]
    lines.append("| " + "metric".ljust(30) + " | " + " | ".join(_short(o).ljust(14)[:14] for o in others) + " |")
    lines.append("|" + "-" * 32 + "|" + "|".join(["-" * 16] * len(others)) + "|")
    for label, key, better in TABLE:
        if better is None or not isinstance(stats[reference].get(key), (int, float)):
            continue
        base = stats[reference][key]
        cells = []
        for o in others:
            v = stats[o].get(key)
            if not isinstance(v, (int, float)):
                cells.append("-")
                continue
            d = v - base
            good = (d > 0) if better == "higher" else (d < 0)
            mark = "+" if good else ("=" if d == 0 else "-")
            cells.append(f"{d:+.4f} {mark}")
        lines.append("| " + label.ljust(30) + " | " + " | ".join(c.ljust(14)[:14] for c in cells) + " |")
    return "\n".join(lines)


def render_figure(stats: dict[str, dict], order: list[str], out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # a missing plotting stack must not lose the table
        logger.warning(f"matplotlib unavailable ({exc!r}); table and JSON written, figure skipped.")
        return False

    panels = [
        ("acceptance rate", "acceptance_rate"),
        ("contact retention\n(accepted)", "acc_contact_retention_median"),
        ("bond MAE nm\n(all attempts)", "all_bond_mae_nm_median"),
        ("min pocket dist nm\n(all attempts)", "all_min_pocket_dist_nm_median"),
        ("per-parent yield", "per_parent_valid_yield_median"),
        ("within-parent\ndiversity nm", "within_parent_diversity_nm_median"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    x = np.arange(len(order))
    for ax, (title, key) in zip(axes.ravel(), panels):
        vals = [stats[a].get(key) for a in order]
        vals = [v if isinstance(v, (int, float)) and math.isfinite(v) else 0.0 for v in vals]
        ax.bar(x, vals, color=["#777777"] + ["#4C78A8"] * (len(order) - 1))
        ax.set_xticks(x)
        ax.set_xticklabels([_short(a) for a in order], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("CP->LP one-pass editor: five arms, identical parents and seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Path] = {}
    for spec in args.arm:
        if "=" not in spec:
            logger.error(f"--arm expects NAME=DIR, got {spec!r}")
            return 1
        name, path = spec.split("=", 1)
        arms[name] = Path(path)

    stats: dict[str, dict] = {}
    for name, path in arms.items():
        try:
            rows = load_rows(path)
        except FileNotFoundError as exc:
            logger.warning(f"{name}: {exc} -- skipping this arm.")
            continue
        stats[name] = arm_stats(rows, path, with_diversity=not args.no_diversity)
        logger.info(
            f"{name}: {stats[name]['n_accepted']}/{stats[name]['n_attempts']} accepted "
            f"({100 * stats[name]['acceptance_rate']:.1f}%) over {stats[name]['n_parents']} parents"
        )
    if not stats:
        logger.error("No arm produced readable rows; nothing to report.")
        return 1

    # The control first, then the reference, then the addons: the reading order is "does the
    # adversary help at all" before "does this addition help the adversary".
    order = [a for a in (args.control_arm, args.reference_arm) if a in stats]
    order += [a for a in sorted(stats) if a not in order]

    real = real_reference_stats(Path(args.real_reference)) if args.real_reference else None

    # Matched-budget check. The arms are only comparable at equal generator updates, and a
    # mismatch is a caveat on the whole table rather than a detail.
    budgets = {a: stats[a].get("g_updates") for a in order}
    parents = {a: stats[a].get("n_parents") for a in order}
    warn = []
    if len({b for b in budgets.values() if b is not None}) > 1:
        warn.append(f"G-UPDATE MISMATCH: {budgets}. The arms are not budget-matched; every quality delta is confounded.")
    if len({p for p in parents.values() if p is not None}) > 1:
        warn.append(f"PARENT-SET MISMATCH: {parents}. The arms were not evaluated on identical parents.")

    table = render_table(stats, order, real)
    deltas = render_deltas(stats, args.reference_arm, order)
    text = [
        "CP->LP ONE-PASS EDITOR: FIVE-ARM COMPARISON",
        "",
        "Raw attempts AND accepted exports are both shown. An arm with one excellent",
        "survivor and near-zero yield is visible in `acceptance rate` and `coverage`, not",
        "in the accepted-only quality columns.",
        "",
        table,
        "",
        deltas,
        "",
        "HOW TO READ THIS:",
        "  * An addon is worth keeping only if it improves valid LP quality AND retention at",
        "    useful yield over the plain one-pass GAN. A lower discriminator accuracy, a nicer",
        "    GAN loss curve or greater terminal separation alone is not sufficient.",
        f"  * {args.control_arm} answers a different question: whether adversarial learning",
        "    helps at all. An addon that beats the GAN while the GAN loses to the control has",
        "    improved a term that should not be switched on.",
        "  * `within-parent diversity` counts only VALID LPs. Diversity among broken chains is",
        "    distortion, not variety.",
        "  * `CA drift` and `interface drift` are measured IN THE RECEPTOR FRAME with no",
        "    superposition, so they include pose change. That is deliberate: aligning the",
        "    peptide first would hide a sample that slid off its binding mode.",
        "  * Cost is reported separately from quality. The addons require different",
        "    computation and folding that into the verdict hides what is being paid for.",
    ]
    if warn:
        text = ["!! " + w for w in warn] + [""] + text
        for w in warn:
            logger.warning(w)

    (out_dir / "comparison.md").write_text("\n".join(text) + "\n")
    (out_dir / "comparison.json").write_text(
        json.dumps({"arms": stats, "real_lp": real, "order": order, "warnings": warn}, indent=2)
    )
    drew = render_figure(stats, order, out_dir / "comparison.png")

    print("\n".join(text))
    logger.info(f"wrote {out_dir / 'comparison.md'}, comparison.json{', comparison.png' if drew else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
