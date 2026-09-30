"""Compare real and generated bound linear peptides from an exported triplet set.

Answers the four questions that decide whether the generated LPs are usable:

terminal separation
    did the ring actually open, and does the resulting separation look like a real bound
    LP's rather than like a cut macrocycle? The source CP's own gap is carried alongside,
    so the move is visible per example and not just in aggregate.

geometry and clashes
    is the backbone intact, are the peptide bonds the right length, and does the peptide
    overlap its pocket? Scored against the real LPs measured by the same code.

CP-contact retention
    how much of the source CP's binding mode survived. This is the term the ablation is
    about, so it is reported per arm and not only pooled.

diversity across seeds
    do different noise draws for one CP give different peptides, or has the generator
    collapsed? Measured on the CA distance matrix, which needs no superposition and
    correctly calls two rigidly-displaced copies identical.

CPU only, and reads nothing but the exported JSONL rows and PDBs -- so the figure can be
redrawn after a threshold changes without regenerating a single peptide.

Example:
    .venv/bin/python script_utils/cp2lp_report.py \\
        --generated $STORE/cp2lp_gan_v1/triplets \\
        --real $STORE/cp2lp_reference \\
        --label gan --out-dir $STORE/cp2lp_gan_v1/report
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated", action="append", required=True,
                   help="Directory of generated triplets (repeatable, once per arm).")
    p.add_argument("--label", action="append", default=None,
                   help="Arm label, matched positionally to --generated.")
    p.add_argument("--real", default=None, help="Directory of real_reference_*.jsonl rows.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--accepted-only", action="store_true", default=True,
                   help="Restrict generated statistics to samples that passed the export gate.")
    p.add_argument("--all-samples", dest="accepted_only", action="store_false",
                   help="Include rejected samples (acceptance rate is reported either way).")
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def read_jsonl_dir(root: Path, pattern: str) -> list[dict]:
    """Reads every matching JSONL under ``root``, REFUSING corrupt lines rather than skipping.

    Concurrent appends to one file NUL-corrupt rows on this filesystem. Silently dropping
    a bad line turns data loss into a quietly smaller sample, so a corrupt row is a hard
    error naming the file -- the caller can then decide, with the loss visible.
    """
    rows: list[dict] = []
    files = sorted(root.rglob(pattern))
    for f in files:
        for lineno, line in enumerate(f.read_text(errors="replace").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if "\x00" in line:
                raise ValueError(f"NUL-corrupted row at {f}:{lineno} -- concurrent append. Refusing to average over it.")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at {f}:{lineno}: {e}") from e
    return rows


def stats(values: list[float]) -> dict:
    """Median/IQR summary. Median rather than mean: these distributions have long tails."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None, "min": None, "max": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "q1": float(np.percentile(v, 25)),
        "q3": float(np.percentile(v, 75)),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def read_ca_from_pdb(path: Path, chain: str = "B") -> np.ndarray:
    """CA coordinates (angstrom) of one chain. Tiny parser -- no biotite import for this."""
    coords = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(coords, dtype=float)


def distance_matrix_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two CA distance matrices. Alignment-free conformational difference."""
    if a.shape != b.shape or a.shape[0] < 2:
        return float("nan")
    da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
    iu = np.triu_indices(a.shape[0], k=1)
    return float(np.sqrt(np.mean((da[iu] - db[iu]) ** 2)))


def seed_diversity(rows: list[dict]) -> dict:
    """Per-source-CP spread across samples, in angstrom, plus a sequence-collapse check."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("gen_lp_pdb"):
            by_source[r["source_id"]].append(r)

    per_source_struct: list[float] = []
    per_source_unique_seq: list[float] = []
    for _, group in by_source.items():
        if len(group) < 2:
            continue
        cas = []
        for r in group:
            p = Path(r["gen_lp_pdb"])
            if p.exists():
                ca = read_ca_from_pdb(p)
                if ca.size:
                    cas.append(ca)
        pairs = [
            distance_matrix_rmsd(cas[i], cas[j])
            for i in range(len(cas))
            for j in range(i + 1, len(cas))
        ]
        pairs = [x for x in pairs if not math.isnan(x)]
        if pairs:
            per_source_struct.append(float(np.mean(pairs)))
        seqs = {r["sequence"] for r in group}
        per_source_unique_seq.append(len(seqs) / len(group))

    return {
        "n_sources_with_multiple_samples": len(per_source_struct),
        "ca_distmat_rmsd_ang": stats(per_source_struct),
        "unique_sequence_fraction": stats(per_source_unique_seq),
    }


def summarise_arm(rows: list[dict], accepted_only: bool) -> dict:
    total = len(rows)
    accepted = [r for r in rows if r.get("accepted")]
    use = accepted if accepted_only else rows

    def sc(key: str) -> list[float]:
        return [r.get("scores", {}).get(key) for r in use]

    reject_hist: dict[str, int] = {}
    for r in rows:
        for reason in r.get("reject_reasons", []):
            reject_hist[reason] = reject_hist.get(reason, 0) + 1

    return {
        "n_samples": total,
        "n_accepted": len(accepted),
        "acceptance_rate": (len(accepted) / total) if total else 0.0,
        "reject_histogram": reject_hist,
        "scored_on": "accepted_only" if accepted_only else "all_samples",
        "terminal_gap_nm": stats(sc("terminal_gap_nm")),
        "src_terminal_gap_nm": stats(sc("src_terminal_gap_nm")),
        "terminal_gap_delta_nm": stats(sc("terminal_gap_delta_nm")),
        "contact_retention_frac": stats(sc("contact_retention_frac")),
        "peptide_bond_mae_nm": stats(sc("peptide_bond_mae_nm")),
        "clash_inter_sum_nm": stats(sc("clash_inter_sum_nm")),
        "clash_min_inter_nm": stats(sc("clash_min_inter_nm")),
        "chain_intact_frac": float(np.mean([bool(r.get("scores", {}).get("chain_intact")) for r in use])) if use else None,
        "seq_identity_to_source": stats(sc("seq_identity_to_source")),
        "diversity": seed_diversity(use),
    }


def summarise_real(rows: list[dict]) -> dict:
    def sc(key: str) -> list[float]:
        return [r.get("scores", {}).get(key) for r in rows]

    return {
        "n_samples": len(rows),
        "terminal_gap_nm": stats(sc("terminal_gap_nm")),
        "peptide_bond_mae_nm": stats(sc("peptide_bond_mae_nm")),
        "clash_inter_sum_nm": stats(sc("clash_inter_sum_nm")),
        "clash_min_inter_nm": stats(sc("clash_min_inter_nm")),
        "chain_intact_frac": float(np.mean([bool(r.get("scores", {}).get("chain_intact")) for r in rows])) if rows else None,
    }


def _fmt(s: dict | None, scale: float = 1.0, digits: int = 2) -> str:
    if not s or s.get("median") is None:
        return "--"
    return f"{s['median'] * scale:.{digits}f} [{s['q1'] * scale:.{digits}f}, {s['q3'] * scale:.{digits}f}]"


def write_markdown(out: Path, arms: dict, real: dict | None) -> None:
    lines = [
        "# CP -> LP sample report",
        "",
        "Medians with [Q1, Q3]. Lengths in angstrom unless stated. `real` is measured by the",
        "same code on the AE round trip of real bound linear peptides, so the columns are",
        "comparable; a difference is a difference between peptides, not between pipelines.",
        "",
        "| metric | " + " | ".join(arms) + (" | real |" if real else " |"),
        "|---|" + "---|" * (len(arms) + (1 if real else 0)),
    ]

    def row(name: str, getter, scale=1.0, digits=2):
        cells = [_fmt(getter(arms[a]), scale, digits) for a in arms]
        if real:
            cells.append(_fmt(getter(real), scale, digits))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    row("terminal separation (A)", lambda d: d.get("terminal_gap_nm"), 10.0)
    row("source CP gap (A)", lambda d: d.get("src_terminal_gap_nm"), 10.0)
    row("gap opened (A)", lambda d: d.get("terminal_gap_delta_nm"), 10.0)
    row("peptide-bond MAE (A)", lambda d: d.get("peptide_bond_mae_nm"), 10.0, 3)
    row("min peptide-pocket dist (A)", lambda d: d.get("clash_min_inter_nm"), 10.0)
    row("CP contact retention", lambda d: d.get("contact_retention_frac"), 1.0, 3)
    row("sequence identity to CP", lambda d: d.get("seq_identity_to_source"), 1.0, 3)
    row("seed diversity, CA distmat RMSD (A)",
        lambda d: (d.get("diversity") or {}).get("ca_distmat_rmsd_ang"))

    lines += ["", "| scalar | " + " | ".join(arms) + (" | real |" if real else " |"),
              "|---|" + "---|" * (len(arms) + (1 if real else 0))]
    for name, key in [("samples", "n_samples"), ("accepted", "n_accepted")]:
        cells = [str(arms[a].get(key, "--")) for a in arms]
        if real:
            cells.append(str(real.get(key, "--")) if key == "n_samples" else "--")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = [f"{arms[a]['acceptance_rate'] * 100:.1f}%" for a in arms]
    if real:
        cells.append("--")
    lines.append("| acceptance rate | " + " | ".join(cells) + " |")
    cells = [
        f"{arms[a]['chain_intact_frac'] * 100:.1f}%" if arms[a].get("chain_intact_frac") is not None else "--"
        for a in arms
    ]
    if real:
        cells.append(
            f"{real['chain_intact_frac'] * 100:.1f}%" if real.get("chain_intact_frac") is not None else "--"
        )
    lines.append("| backbone intact | " + " | ".join(cells) + " |")

    lines += ["", "## Rejections", ""]
    for a in arms:
        hist = arms[a]["reject_histogram"]
        lines.append(f"- **{a}**: " + (", ".join(f"{k}={v}" for k, v in sorted(hist.items())) or "none"))

    out.write_text("\n".join(lines) + "\n")


def draw_figure(out: Path, arms: dict, real: dict | None, raw: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = list(arms) + (["real"] if real else [])

    def series(arm: str, key: str) -> list[float]:
        rows = raw[arm]
        return [
            r["scores"][key] * 10.0
            for r in rows
            if r.get("scores", {}).get(key) is not None and not math.isnan(r["scores"][key])
        ]

    for ax, key, title in [
        (axes[0], "terminal_gap_nm", "Terminal separation (A)"),
        (axes[1], "clash_min_inter_nm", "Min peptide-pocket distance (A)"),
        (axes[2], "peptide_bond_mae_nm", "Peptide-bond MAE (A)"),
    ]:
        data = [series(n, key) for n in names]
        data = [d if d else [np.nan] for d in data]
        # `labels=` was renamed `tick_labels=` in matplotlib 3.9; set the ticks by hand so
        # the report does not depend on which side of that rename the env is on.
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[1].axhline(2.0, color="crimson", ls="--", lw=1, label="clash gate 2.0 A")
    axes[1].legend(fontsize=8)
    fig.suptitle("CP -> LP: generated vs real bound linear peptides")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.label or []
    if len(labels) < len(args.generated):
        labels += [Path(g).name for g in args.generated[len(labels):]]

    arms, raw = {}, {}
    for label, gdir in zip(labels, args.generated):
        rows = read_jsonl_dir(Path(gdir), "triplets*.jsonl")
        if not rows:
            raise SystemExit(f"No triplets*.jsonl rows under {gdir}")
        arms[label] = summarise_arm(rows, args.accepted_only)
        raw[label] = [r for r in rows if r.get("accepted")] if args.accepted_only else rows

    real = None
    if args.real:
        real_rows = read_jsonl_dir(Path(args.real), "real_reference*.jsonl")
        if real_rows:
            real = summarise_real(real_rows)
            raw["real"] = real_rows

    summary = {"arms": arms, "real": real}
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    write_markdown(out_dir / "report.md", arms, real)
    if not args.no_figure:
        draw_figure(out_dir / "report.png", arms, real, raw)

    print((out_dir / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
