"""Diagnose the provenance-paired LP -> CP set, and render the report.

Reads only saved artefacts -- the pair shards written by `lp2cp_provenance.py pair`, plus a
cached reference-gap array -- so every figure can be redrawn without rebuilding a single
pair. The reference gaps are computed once and cached to npz for exactly that reason.

The five questions, in the order they can invalidate the build:

  1. **Coverage.** How many pairs exist, and why the rest do not. A skip reason that
     dominates is a bug or a data-root problem, not attrition.
  2. **Selection bias on terminal gap** -- the one that would waste the build. CPSea's
     fragments were selected *because they were cyclizable*, which is selection on the
     target variable, so their parents may sit at small terminal gaps while real LP binders
     sit at 16-21 A. `protfrag` is the sharp control here: it is AFDB-derived like CPSea's
     parents, while `pepbench` is real bound LP binders. If the provenance LPs track
     protfrag and not pepbench, that IS the bias, quantified.
  3. **The mutation budget, which now comes free.** The parent-vs-product sequence delta is
     a per-pair edit label. Stage 0 found the budget unenforceable at inference; supervised
     from a coupling it does not need enforcing. Reported with WHERE the mutations sit,
     because the anchor-identity finding predicts the termini.
  4. **Transport amplitude.** How far the peptide actually has to move. Below ~1 A the
     before-state has collapsed onto the after-state and the pair teaches copying.
  5. **Alignment sanity.** The receptor fit residual across CPSea's two relaxation passes.
     A heavy tail here means the parent and product are not the same pocket.

Quoting rule this report enforces on itself: every median prints its slice and its n. A
pooled median is not any set's median.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BAND_LO, BAND_HI = 20.0, 45.0  # the 2.1 coverage band: real LP binders' terminal gaps
AMPLITUDE_FLOOR = 2.0  # below this a "before" state collapses back onto its "after"


# ------------------------------------------------------------------------------- loading


def load_pairs(pairs_dir: Path) -> list[dict]:
    """Read every shard. A NUL-corrupted line is fatal, never skipped."""
    rows: list[dict] = []
    shards = sorted(pairs_dir.glob("pairs_shard*.jsonl"))
    if not shards:
        return rows
    for shard in shards:
        with shard.open() as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                if "\x00" in line:
                    raise SystemExit(
                        f"FATAL: NUL byte at {shard}:{ln}. Concurrent appends corrupted "
                        "this shard; rebuild it rather than reporting around it."
                    )
                rows.append(json.loads(line))
        print(f"  {shard.name}: {sum(1 for _ in shard.open()):,} lines")
    return rows


def reference_gaps(lp_metadata: Path, sample: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Terminal CA-CA gap per reference complex, by dataset_source.

    Uses `peptide_profile.end_to_end` -- the same function the pre-check suite's
    `e2e_ca_A` came from -- so these numbers are comparable to the ones already recorded
    rather than a second, subtly different measurement of the same thing.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd

    from peptide_profile import end_to_end, load_complex

    df = pd.read_parquet(lp_metadata, columns=["example_id", "path", "dataset_source"])
    out: dict[str, list[float]] = {}
    rng = np.random.default_rng(seed)
    for src, grp in df.groupby("dataset_source"):
        if len(grp) > sample:
            grp = grp.iloc[rng.choice(len(grp), size=sample, replace=False)]
        vals: list[float] = []
        for rec in grp.to_dict("records"):
            cx = load_complex(rec["path"], rec["example_id"])
            if cx is None or cx.length < 2:
                continue
            vals.append(end_to_end(cx.CA)[0])
        out[str(src)] = vals
        print(f"  reference {src:10s}: {len(vals):,} complexes profiled")
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def lnr_gaps(lnr_root: Path, sample: int, seed: int = 0) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from peptide_profile import end_to_end, load_complex

    listing = lnr_root / "test.txt"
    if not listing.exists():
        return np.asarray([], dtype=float)
    names = [n.strip() for n in listing.read_text().split() if n.strip()]
    rng = np.random.default_rng(seed)
    if len(names) > sample:
        names = [names[i] for i in rng.choice(len(names), size=sample, replace=False)]
    vals: list[float] = []
    for n in names:
        for cand in (lnr_root / "pdbs" / f"{n}.pdb", lnr_root / "pdbs" / n):
            if cand.exists():
                cx = load_complex(cand, n)
                if cx is not None and cx.length >= 2:
                    vals.append(end_to_end(cx.CA)[0])
                break
    print(f"  reference lnr       : {len(vals):,} complexes profiled")
    return np.asarray(vals, dtype=float)


# ------------------------------------------------------------------------------ reporting


def _stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "mean": float(a.mean()),
        "frac_in_band": float(np.mean((a >= BAND_LO) & (a <= BAND_HI))),
    }


def _row(name: str, s: dict) -> str:
    if s["n"] == 0:
        return f"| {name} | 0 | - | - | - | - |"
    return (
        f"| {name} | {s['n']:,} | {s['median']:.2f} | {s['p25']:.2f} | "
        f"{s['p75']:.2f} | {s['frac_in_band']:.3f} |"
    )


def build_report(rows: list[dict], refs: dict[str, np.ndarray], out_dir: Path) -> dict:
    built = [r for r in rows if r.get("skip_reason") is None]
    skipped = [r for r in rows if r.get("skip_reason") is not None]

    def arr(key: str, src=built) -> np.ndarray:
        return np.asarray(
            [r[key] for r in src if r.get(key) is not None], dtype=float
        )

    lp_gap = arr("lp_term_ca_A")
    cp_gap = arr("cp_term_ca_A")
    amp = arr("lp_cp_ca_rmsd_A")
    rec_rmsd = arr("receptor_rmsd_A")
    n_mut = arr("n_mut")

    L: list[str] = []
    L.append("# Provenance-paired LP -> CP set: diagnosis\n")
    L.append(
        "Pairs are CPSea complexes whose peptide has been replaced by the same residue\n"
        "range taken from the parent AlphaFold model, rigid-aligned on the shared\n"
        "receptor. The receptor is byte-identical on both sides of every pair.\n"
    )

    # --- 1 coverage ---------------------------------------------------------------------
    L.append("\n## 1. Coverage\n")
    L.append(f"- attempted: **{len(rows):,}**")
    L.append(f"- built:     **{len(built):,}**")
    L.append(f"- skipped:   **{len(skipped):,}**\n")
    if skipped:
        L.append("| skip reason | n | share of attempted |")
        L.append("|---|---|---|")
        for reason, n in Counter(
            str(r["skip_reason"]).split(":")[0] for r in skipped
        ).most_common():
            L.append(f"| `{reason}` | {n:,} | {n / max(len(rows), 1):.3f} |")
        L.append("")
        L.append(
            "> `parent_not_fetched` is download coverage, not a data problem. Any other\n"
            "> reason dominating the table is a bug: read it before reading section 2.\n"
        )

    # --- 2 the selection-bias question --------------------------------------------------
    L.append("\n## 2. Terminal gap: do these LPs look like real LP binders?\n")
    L.append(
        f"`frac_in_band` is the share inside {BAND_LO:.0f}-{BAND_HI:.0f} A, the coverage\n"
        "target reinstated by the terminal-only finding. CPSea natives sit at 0% in band.\n"
    )
    L.append("\n| slice | n | median | p25 | p75 | frac in band |")
    L.append("|---|---|---|---|---|---|")
    L.append(_row("**LP side (provenance parents)**", _stats(lp_gap)))
    L.append(_row("CP side (CPSea product)", _stats(cp_gap)))
    for src in sorted(refs):
        label = {
            "pepbench": "pepbench (real bound LP binders)",
            "protfrag": "protfrag (AFDB fragments -- the control)",
            "lnr": "LNR (the benchmark)",
        }.get(src, src)
        L.append(_row(label, _stats(refs[src])))
    L.append("")

    verdict = "UNDECIDED"
    if lp_gap.size >= 50 and "pepbench" in refs and refs["pepbench"].size >= 50:
        lp_med = float(np.median(lp_gap))
        pb_med = float(np.median(refs["pepbench"]))
        lp_band = float(np.mean((lp_gap >= BAND_LO) & (lp_gap <= BAND_HI)))
        pb_band = float(np.mean((refs["pepbench"] >= BAND_LO) & (refs["pepbench"] <= BAND_HI)))
        if lp_band >= 0.5 * pb_band:
            verdict = "GAP COVERAGE OK"
            L.append(
                f"**{verdict}** -- provenance LPs reach {lp_band:.1%} in band against\n"
                f"pepbench's {pb_band:.1%} (medians {lp_med:.1f} vs {pb_med:.1f} A). The\n"
                "selection bias did not collapse the source marginal.\n"
            )
        else:
            verdict = "GAP COVERAGE BIASED"
            L.append(
                f"**{verdict}** -- provenance LPs reach only {lp_band:.1%} in band against\n"
                f"pepbench's {pb_band:.1%} (medians {lp_med:.1f} vs {pb_med:.1f} A). The\n"
                "pairs cover the easy regime. Add a marginal term on real LPs, or augment\n"
                "gaps with the extension move, before reading a transport result as\n"
                "transferable.\n"
            )
    else:
        L.append("**UNDECIDED** -- too few pairs or reference complexes to compare.\n")

    # --- 3 the mutation budget ----------------------------------------------------------
    L.append("\n## 3. Mutation budget (supervised, from the coupling)\n")
    if n_mut.size:
        L.append(f"- pairs with a sequence delta computed: **{int(n_mut.size):,}**")
        L.append(f"- median edits: **{np.median(n_mut):.0f}**, mean {n_mut.mean():.2f}")
        for k in (0, 1, 2, 3):
            L.append(f"- k = {k}: {float(np.mean(n_mut == k)):.3f}")
        L.append(f"- k > 3: {float(np.mean(n_mut > 3)):.3f}\n")

        by_chem: dict[str, list[float]] = {}
        for r in built:
            if r.get("n_mut") is not None:
                by_chem.setdefault(str(r["cyclization_type"]), []).append(float(r["n_mut"]))
        L.append("| chemistry | n | median edits | frac k=0 | frac k<=2 |")
        L.append("|---|---|---|---|---|")
        for chem in sorted(by_chem):
            v = np.asarray(by_chem[chem])
            L.append(
                f"| {chem} | {v.size:,} | {np.median(v):.0f} | "
                f"{float(np.mean(v == 0)):.3f} | {float(np.mean(v <= 2)):.3f} |"
            )
        L.append("")

        # Where the edits sit. The anchor-identity finding predicts the termini; if the
        # edits are spread uniformly, the curation did something else and the free budget
        # labels are not anchor labels.
        at_term = 0
        tot = 0
        for r in built:
            pos = r.get("mut_positions")
            if not pos:
                continue
            lo, hi = int(r["res_start"]), int(r["res_end"])
            for p in pos:
                tot += 1
                if int(p) in (lo, hi):
                    at_term += 1
        if tot:
            L.append(
                f"- of {tot:,} individual edits, **{at_term / tot:.1%}** sit at a terminal\n"
                f"  residue (position `res_start` or `res_end`).\n"
            )
    else:
        L.append("No sequence deltas available.\n")

    # --- 4 transport amplitude ----------------------------------------------------------
    L.append("\n## 4. Transport amplitude\n")
    if amp.size:
        s = _stats(amp)
        L.append(f"- receptor-aligned peptide CA RMSD, LP -> CP: median **{s['median']:.2f} A**")
        L.append(f"  (p25 {s['p25']:.2f}, p75 {s['p75']:.2f}, n {s['n']:,})")
        below = float(np.mean(amp < AMPLITUDE_FLOOR))
        L.append(f"- below the {AMPLITUDE_FLOOR:.0f} A floor: **{below:.1%}**\n")
        if below > 0.5:
            L.append(
                "> More than half the pairs move the peptide less than the amplitude floor.\n"
                "> Those pairs can be satisfied by copying the input, so they supervise\n"
                "> little. Consider filtering on amplitude before training.\n"
            )
    else:
        L.append("No amplitudes available.\n")

    # --- 5 alignment sanity -------------------------------------------------------------
    L.append("\n## 5. Alignment sanity (receptor fit across CPSea's relaxation)\n")
    if rec_rmsd.size:
        s = _stats(rec_rmsd)
        L.append(f"- receptor CA RMSD after fit: median **{s['median']:.3f} A**")
        L.append(f"  (p25 {s['p25']:.3f}, p75 {s['p75']:.3f}, n {s['n']:,})")
        L.append(f"- above 2 A: {float(np.mean(rec_rmsd > 2.0)):.3f}")
        L.append(
            "\n> This is the distance CPSea's two OpenMM passes moved the pocket. A heavy\n"
            "> tail means the parent and the product are not the same pocket, and those\n"
            "> pairs put the LP peptide in a frame its receptor never occupied.\n"
        )
    else:
        L.append("No receptor fits available.\n")

    summary = {
        "n_attempted": len(rows),
        "n_built": len(built),
        "n_skipped": len(skipped),
        "skip_reasons": dict(
            Counter(str(r["skip_reason"]).split(":")[0] for r in skipped)
        ),
        "gap_verdict": verdict,
        "lp_term_ca_A": _stats(lp_gap),
        "cp_term_ca_A": _stats(cp_gap),
        "reference_term_ca_A": {k: _stats(v) for k, v in refs.items()},
        "n_mut": _stats(n_mut),
        "amplitude_ca_rmsd_A": _stats(amp),
        "receptor_fit_rmsd_A": _stats(rec_rmsd),
        "band": [BAND_LO, BAND_HI],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(L))
    return summary


def make_figures(rows: list[dict], refs: dict[str, np.ndarray], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (no figures: {type(exc).__name__}: {exc})")
        return

    built = [r for r in rows if r.get("skip_reason") is None]

    def arr(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in built if r.get(key) is not None], dtype=float)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ECDFs, not histograms: the question is what share sits above a threshold, and an
    # ECDF answers that by eye without a bin-width argument.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [("LP side (provenance)", arr("lp_term_ca_A")), ("CP side (CPSea)", arr("cp_term_ca_A"))]
    series += [(f"ref: {k}", v) for k, v in sorted(refs.items())]
    for label, v in series:
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        x = np.sort(v)
        ax.plot(x, np.arange(1, x.size + 1) / x.size, label=f"{label} (n={x.size:,})")
    ax.axvspan(BAND_LO, BAND_HI, alpha=0.12, color="grey")
    ax.set_xlabel("terminal CA-CA gap (A)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(f"Terminal gap; shaded = {BAND_LO:.0f}-{BAND_HI:.0f} A coverage band")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "terminal_gap_ecdf.png", dpi=150)
    plt.close(fig)

    for key, xlabel, fname, vline in (
        ("n_mut", "sequence edits, parent -> product", "mutation_budget.png", None),
        ("lp_cp_ca_rmsd_A", "peptide CA RMSD, LP -> CP (A)", "amplitude.png", AMPLITUDE_FLOOR),
        ("receptor_rmsd_A", "receptor CA RMSD after fit (A)", "receptor_fit.png", 2.0),
    ):
        v = arr(key)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(v, bins=min(60, max(10, int(np.sqrt(v.size)))), color="#4477aa")
        if vline is not None:
            ax.axvline(vline, color="#cc3311", ls="--", lw=1.5, label=f"{vline:g} A")
            ax.legend(fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pairs")
        ax.set_title(f"{xlabel}  (n={v.size:,}, median {np.median(v):.2f})")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    print(f"  figures -> {fig_dir}")


# ---------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lp-metadata", default="")
    ap.add_argument("--lnr-root", default="")
    ap.add_argument("--ref-sample", type=int, default=3000)
    ap.add_argument("--cache", default="", help="npz of reference gaps; written if absent")
    args = ap.parse_args(argv)

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)

    print("--- pair shards")
    rows = load_pairs(pairs_dir)
    if not rows:
        print("=" * 70)
        print(f"GATE: no pair shards in {pairs_dir}.")
        print("      The `pair` stage produced nothing -- read ITS log, most likely the")
        print("      AFDB fetch was blocked. Exiting 0 so dependents are not stranded.")
        print("=" * 70)
        return 0
    print(f"  total records: {len(rows):,}")

    # Reference gaps are cached so the figures can be redrawn without re-profiling
    # thousands of complexes through mdtraj.
    cache = Path(args.cache) if args.cache else out_dir / "reference_gaps.npz"
    refs: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as z:
            refs = {k: z[k] for k in z.files}
        print(f"--- reference gaps from cache: {cache} ({list(refs)})")
    else:
        print("--- reference gaps (profiling; cached afterwards)")
        if args.lp_metadata and Path(args.lp_metadata).exists():
            refs.update(reference_gaps(Path(args.lp_metadata), args.ref_sample))
        else:
            print(f"  (no LP metadata at {args.lp_metadata!r}; skipping pepbench/protfrag)")
        if args.lnr_root and Path(args.lnr_root).exists():
            v = lnr_gaps(Path(args.lnr_root), args.ref_sample)
            if v.size:
                refs["lnr"] = v
        if refs:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **refs)
            print(f"  cached -> {cache}")

    print()
    summary = build_report(rows, refs, out_dir)
    make_figures(rows, refs, out_dir)
    print()
    print(f"report  -> {out_dir / 'report.md'}")
    print(f"summary -> {out_dir / 'summary.json'}")
    print(f"VERDICT : {summary['gap_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
