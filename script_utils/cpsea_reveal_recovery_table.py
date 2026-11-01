#!/usr/bin/env python3
"""Sequence recovery as a function of reveal fraction, read off the val_gen dumps.

`aggregate_lp2cp_testeval.py` reads closure and dG and ignores the `val_gen/seq/*` keys, so
the reveal sweep has no reader. This is it: CPU, reads only `closure_rows_*.jsonl`, so the
table is redrawable without resampling.

The question it answers is the one every recovery number on record so far could not:
`cond_recovery` has only ever been measured at `keep_frac = 1.0`, where `n_free_res = 0` and
`free_recovery` is NaN -- a copy test. Below 1.0 the two separate: `cond_recovery` stays a
compliance check on the handed positions, and `free_recovery` becomes a genuine prediction
number on the positions the model had to fill.

Three things it refuses to do, each of which would manufacture a number.

**Rates are re-weighted by their own population, never averaged over batches.** A batch at
keep_frac 0.25 contributes three times the free positions of one at 0.75, and the per-batch
`free_recovery` values are rates over different denominators. Averaging them weights a batch
of 4 free residues equally with one of 40. `n_cond_res` / `n_free_res` are logged beside
every rate precisely so the pooled rate can be reconstructed, and that is what is reported.

**NaN is a population of zero, not a miss.** `sequence_recovery_metrics` returns NaN rather
than 0.0 wherever nothing was measurable -- `free_recovery` at full reveal is the designed
case. A filter that treats NaN as a number scores the copy cell as 0% prediction, and NaN is
truthy in Python, so the naive guard is the wrong way round.

**No baseline is invented.** A recovery of 0.3 means nothing without P(r | pos, L, chemistry),
which is a separate corpus pass; this prints the measured rate and the paired-seed spread, and
leaves the margin to whoever has the prior. The spread matters because two no-injection runs
with identical settings have already differed by 0.104 in `cond_exact_frac`: any difference
smaller than the paired cells' gap is not readable here.

Usage:
    python script_utils/cpsea_reveal_recovery_table.py \
        --root /zfsauton/scratch/yixiz/cpsea_seqcond_reveal
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SEQ_RATES = ("cond_recovery", "free_recovery", "all_recovery")

# Closure keys as the dump actually spells them -- verified against an existing
# closure_rows_*.jsonl, not assumed: the success rate is `<chem>_bond_success` but the
# population is `n_valid_<chem>`, which is not the same stem. Chemistries are kept SEPARATE;
# pooling them produces a rate for a mixture that no sample belongs to.
CHEMS = {
    "mainchain": ("val_gen/cyc/mainchain_cn_bond_success", "val_gen/cyc/n_valid_mainchain"),
    "isopeptide": ("val_gen/cyc/isopeptide_bond_success", "val_gen/cyc/n_valid_isopeptide"),
    "disulfide": ("val_gen/cyc/disulfide_bond_success", "val_gen/cyc/n_valid_disulfide"),
}

# Composition diagnostics, logged on every pass (conditioned or not). They describe WHAT the
# model emitted rather than whether it matched, which is the only handle on "did the free
# positions drift toward a different residue distribution".
COMP = ("val_gen/seq/entropy_bits", "val_gen/seq/top1_frac", "val_gen/seq/n_types")


def _finite(x) -> bool:
    """True only for a real number. NaN is truthy, so `if x` is not this test."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def load_cell(cell_dir: Path) -> dict:
    """Pool one cell's batches into population-weighted rates."""
    num = {k: 0.0 for k in SEQ_RATES}
    den = {k: 0.0 for k in SEQ_RATES}
    cnum = {k: 0.0 for k in CHEMS}
    cden = {k: 0.0 for k in CHEMS}
    comp: dict[str, list[float]] = {k: [] for k in COMP}
    exact_vals: list[float] = []
    n_batches = 0
    keep_frac = seed = ckpt_step = None

    for f in sorted(cell_dir.glob("closure_rows_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A truncated tail is a partial write from a job still running, not a reason
                # to drop the whole file.
                continue
            m = row.get("metrics") or {}
            n_batches += 1
            if keep_frac is None:
                keep_frac = row.get("keep_frac")
                seed = row.get("seed")
                # The run records its own checkpoint step, so which checkpoint produced the
                # table is read off the result rather than asserted from the launch command.
                ckpt_step = row.get("ckpt_global_step")

            n_cond = m.get("val_gen/seq/n_cond_res")
            n_free = m.get("val_gen/seq/n_free_res")
            pops = {
                "cond_recovery": n_cond,
                "free_recovery": n_free,
                "all_recovery": (n_cond or 0) + (n_free or 0) if _finite(n_cond) or _finite(n_free) else None,
            }
            for k in SEQ_RATES:
                r, n = m.get(f"val_gen/seq/{k}"), pops[k]
                # Both must be real: a rate with no population, or a population with no rate,
                # contributes nothing rather than a zero.
                if _finite(r) and _finite(n) and float(n) > 0:
                    num[k] += float(r) * float(n)
                    den[k] += float(n)

            ex = m.get("val_gen/seq/cond_exact_frac")
            if _finite(ex):
                # The row records no batch size, so this is an UNWEIGHTED mean over batches.
                # That is exact while batches are equal-sized and off only by the final
                # partial batch -- stating it beats weighting by a hardcoded 16 the dump
                # never confirms.
                exact_vals.append(float(ex))

            # Closure, reported beside recovery: a sequence the model honoured but could not
            # build a ring from is not a success, and the two move independently.
            for chem, (vkey, nkey) in CHEMS.items():
                v, n = m.get(vkey), m.get(nkey)
                if _finite(v) and _finite(n) and float(n) > 0:
                    cnum[chem] += float(v) * float(n)
                    cden[chem] += float(n)

            for k in COMP:
                if _finite(m.get(k)):
                    comp[k].append(float(m[k]))

    out = {
        "cell": cell_dir.name,
        "keep_frac": keep_frac,
        "seed": seed,
        "ckpt_global_step": ckpt_step,
        "n_batches": n_batches,
    }
    for k in SEQ_RATES:
        out[k] = (num[k] / den[k]) if den[k] > 0 else float("nan")
        out[f"{k}_n"] = int(den[k])
    out["cond_exact_frac"] = (sum(exact_vals) / len(exact_vals)) if exact_vals else float("nan")
    for chem in CHEMS:
        out[f"closure_{chem}"] = (cnum[chem] / cden[chem]) if cden[chem] > 0 else float("nan")
        out[f"closure_{chem}_n"] = int(cden[chem])
    for k in COMP:
        short = k.rsplit("/", 1)[-1]
        out[short] = (sum(comp[k]) / len(comp[k])) if comp[k] else float("nan")
    return out


def fmt(x, nd=4) -> str:
    if not _finite(x):
        return "  --  "
    return f"{float(x):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dir holding one subdir per reveal cell.")
    ap.add_argument("--json-out", default=None, help="Also write the rows as JSON.")
    args = ap.parse_args()

    root = Path(args.root)
    cells = [d for d in sorted(root.iterdir()) if d.is_dir() and any(d.glob("closure_rows_*.jsonl"))]
    if not cells:
        print(f"No cell with closure rows under {root} yet.")
        return 0

    rows = [load_cell(d) for d in cells]
    rows.sort(key=lambda r: (-(r["keep_frac"] or 0), r["seed"] or 0))

    steps = {r["ckpt_global_step"] for r in rows if r["ckpt_global_step"] is not None}
    print(f"checkpoint global_step: {steps if len(steps) != 1 else steps.pop()}")
    hdr = (f"{'cell':<12} {'kf':>5} {'sd':>3} {'nb':>4} {'cond_rec':>9} {'free_rec':>9} {'n_free':>7} "
           f"{'exact':>7} {'clos_mc':>8} {'clos_iso':>8} {'clos_ss':>8} {'ent_bits':>8} {'top1':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['cell']:<12} {fmt(r['keep_frac'],2):>5} {str(r['seed']):>3} {r['n_batches']:>4} "
            f"{fmt(r['cond_recovery']):>9} {fmt(r['free_recovery']):>9} {r['free_recovery_n']:>7} "
            f"{fmt(r['cond_exact_frac']):>7} {fmt(r['closure_mainchain']):>8} "
            f"{fmt(r['closure_isopeptide']):>8} {fmt(r['closure_disulfide']):>8} "
            f"{fmt(r['entropy_bits'],3):>8} {fmt(r['top1_frac'],3):>6}"
        )

    # The paired cells are the noise floor: same complexes, same reveal fraction, different
    # sampling seed. Any curve feature smaller than this gap is not readable at this n.
    by_kf: dict[float, list[dict]] = {}
    for r in rows:
        if _finite(r["keep_frac"]):
            by_kf.setdefault(round(float(r["keep_frac"]), 3), []).append(r)
    for kf, group in sorted(by_kf.items()):
        if len(group) < 2:
            continue
        for k in ("free_recovery", "cond_recovery", "cond_exact_frac"):
            vals = [g[k] for g in group if _finite(g[k])]
            if len(vals) >= 2:
                print(f"paired spread @ keep_frac={kf}: {k} {max(vals)-min(vals):.4f} "
                      f"({', '.join(f'{v:.4f}' for v in vals)})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
