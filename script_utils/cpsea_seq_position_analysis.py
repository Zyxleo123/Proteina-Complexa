#!/usr/bin/env python3
"""Positional recovery, BLOSUM scoring, and the fair prior -- all from the per-position dump.

A pooled `free_recovery` cannot be decomposed after the fact, and the decomposition is the
result. The ring anchors sit at the termini and their identity is largely forced by the
chemistry (a disulfide anchor must be CYS), so a pooled number can be carried by exactly the
positions that are a lookup rather than a prediction. This reads
`<cell>/seq_per_position/seq_pos_*.npz` and reports:

    1. positional recovery -- per index from the N terminus and from the C terminus, and
       terminal (pos 0 / L-1) against interior. THE headline split.
    2. BLOSUM62 -- mean over free positions, mean over the free positions exact match got
       WRONG, and the conservative fraction, each against a null drawn from the panel's own
       composition. Exact match charges a full miss for ILE where the native has LEU; this
       says whether the misses are conservative or random.
    3. P(r | pos, L, chemistry) argmax accuracy on EXACTLY the free positions -- the honest
       bar, fit on the corpus TRAIN split only.

All CPU, all from files already on disk, so every number here is redrawable without resampling.

WHAT IT REFUSES TO DO.

**It will not compare a prior to a model across different position populations.** The prior is
evaluated on the same (pos, length, chemistry, truth) tuples the model was scored on, read out
of the same dump, never on a corpus-wide average.

**It will not silently assume the two chemistry encodings agree.** The corpus writes its own
`cyc_type` and the dump carries the batch's `cyclization_type`; these agreeing is an
assumption, so both histograms are printed and a disjoint overlap is called out. A wrong
mapping would quietly swap the per-chemistry priors and still produce a plausible table.

**It will not score a backed-off cell as a free miss.** A (pos, L, cyc) cell the train split
never saw backs off to (pos, L) and then to the global argmax, which is what
`cpseq_mlm_data.baseline_accuracy` does -- without the backoff the prior is understated on
rare lengths and the model's margin is re-inflated.

Usage:
    python script_utils/cpsea_seq_position_analysis.py \
        --root /zfsauton/scratch/yixiz/cpsea_seqcond_reveal_v2 \
        --corpus-dir /zfsauton/scratch/yixiz/cpseq_mlm/corpus
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proteinfoundation.cpseq.seq_inject import _AA_BY_INDEX  # noqa: E402
from proteinfoundation.eval.sequence_similarity import (  # noqa: E402
    blosum62_by_restype_index,
    blosum_null_expectation,
)

AA = list(_AA_BY_INDEX)


def load_cell(cell: Path) -> dict[str, np.ndarray] | None:
    files = sorted(glob.glob(str(cell / "seq_per_position" / "seq_pos_*.npz")))
    if not files:
        return None
    acc: dict[str, list[np.ndarray]] = {}
    for f in files:
        with np.load(f) as z:
            for k in z.files:
                acc.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def fit_prior(corpus_dir: Path) -> tuple[dict, dict, str, Counter]:
    """P(r|pos,L,cyc) and P(r|pos,L) from the corpus TRAIN split, plus the global argmax."""
    poscm: dict[tuple[int, int, int], Counter] = {}
    posm: dict[tuple[int, int], Counter] = {}
    glob_c: Counter = Counter()
    types: Counter = Counter()
    for f in sorted(glob.glob(str(corpus_dir / "train_shard*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = row.get("pep_seq") or ""
                ct = row.get("cyc_type")
                if ct is None:
                    continue
                types[int(ct)] += 1
                L = len(seq)
                for pos, ch in enumerate(seq):
                    if ch not in AA:
                        continue
                    poscm.setdefault((pos, L, int(ct)), Counter())[ch] += 1
                    posm.setdefault((pos, L), Counter())[ch] += 1
                    glob_c[ch] += 1
    best = max(glob_c, key=glob_c.get) if glob_c else "L"
    return poscm, posm, best, types


def prior_accuracy(d: dict[str, np.ndarray], sel: np.ndarray, poscm, posm, best) -> dict:
    """Argmax accuracy of the backed-off prior on exactly the selected positions."""
    pos, L, ct, true = d["pos"][sel], d["length"][sel], d["cyc_type"][sel], d["true_aa"][sel]
    hit_pc = hit_p = hit_g = 0
    n_backoff = 0
    for i in range(len(true)):
        truth = AA[int(true[i])]
        c = poscm.get((int(pos[i]), int(L[i]), int(ct[i])))
        if not c:
            n_backoff += 1
            c = posm.get((int(pos[i]), int(L[i])))
        hit_pc += int(truth == (max(c, key=c.get) if c else best))
        p = posm.get((int(pos[i]), int(L[i])))
        hit_p += int(truth == (max(p, key=p.get) if p else best))
        hit_g += int(truth == best)
    n = max(len(true), 1)
    return {
        "n": int(len(true)),
        "prior_pos_chem_acc": hit_pc / n,
        "prior_pos_acc": hit_p / n,
        "prior_global_acc": hit_g / n,
        "backoff_frac": n_backoff / n,
        "global_argmax": best,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus-dir", default=None, help="Fit the prior from this corpus's TRAIN shards.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    cells = [d for d in sorted(root.iterdir()) if d.is_dir()]
    mat = blosum62_by_restype_index()

    poscm = posm = best = None
    corpus_types: Counter = Counter()
    if args.corpus_dir:
        print(f"Fitting prior from {args.corpus_dir} (train shards)...", flush=True)
        poscm, posm, best, corpus_types = fit_prior(Path(args.corpus_dir))
        print(f"  prior cells: {len(poscm)} (pos,L,cyc), {len(posm)} (pos,L); global argmax {best}")
        print(f"  corpus cyc_type histogram: {dict(sorted(corpus_types.items()))}")

    out_rows = []
    for cell in cells:
        d = load_cell(cell)
        if d is None:
            continue
        free = ~d["revealed"].astype(bool)
        if free.sum() == 0:
            print(f"\n=== {cell.name}: no free positions (full reveal) -- compliance cell only")
            continue
        correct = d["pred_aa"] == d["true_aa"]
        L, pos = d["length"], d["pos"]
        terminal = (pos == 0) | (pos == L - 1)

        print(f"\n=== {cell.name}   n_free={int(free.sum())}")
        print(f"  dump cyc_type histogram: {dict(sorted(Counter(d['cyc_type'].tolist()).items()))}")
        if corpus_types and not (set(d["cyc_type"].tolist()) & set(corpus_types)):
            print("  !! dump and corpus chemistry encodings are DISJOINT -- the prior below is "
                  "keyed on a type the corpus never saw and every cell will back off.")

        rec_all = float(correct[free].mean())
        f_term = free & terminal
        f_int = free & ~terminal
        print(f"  free_recovery        {rec_all:.4f}")
        print(f"  terminal (pos 0/L-1) {float(correct[f_term].mean()) if f_term.sum() else float('nan'):.4f}  n={int(f_term.sum())}")
        print(f"  interior             {float(correct[f_int].mean()) if f_int.sum() else float('nan'):.4f}  n={int(f_int.sum())}")

        # Per-index, from both ends: the N-terminal and C-terminal anchors are different
        # positions and pooling them hides an asymmetry if there is one.
        print("  per-position recovery (from N):", end="")
        for p in range(min(8, int(pos.max()) + 1)):
            s = free & (pos == p)
            if s.sum():
                print(f" {p}:{correct[s].mean():.2f}({int(s.sum())})", end="")
        print()
        rev = L - 1 - pos
        print("  per-position recovery (from C):", end="")
        for p in range(min(8, int(rev.max()) + 1)):
            s = free & (rev == p)
            if s.sum():
                print(f" -{p}:{correct[s].mean():.2f}({int(s.sum())})", end="")
        print()

        # BLOSUM, each against a null drawn from this panel's own composition.
        comp = np.bincount(d["true_aa"], minlength=20).astype(np.float64)
        sc = mat[d["pred_aa"], d["true_aa"]]
        miss = free & ~correct
        null_free = blosum_null_expectation(d["true_aa"][free], comp)
        null_miss = blosum_null_expectation(d["true_aa"][miss], comp)
        print(f"  BLOSUM free          {sc[free].mean():+.3f}   null {null_free:+.3f}   "
              f"margin {sc[free].mean()-null_free:+.3f}")
        print(f"  BLOSUM on misses     {sc[miss].mean():+.3f}   null {null_miss:+.3f}   "
              f"margin {sc[miss].mean()-null_miss:+.3f}   n={int(miss.sum())}")
        print(f"  conservative (>=0)   {float((sc[free] >= 0).mean()):.4f}      positive (>0) "
              f"{float((sc[free] > 0).mean()):.4f}")

        row = {
            "cell": cell.name, "n_free": int(free.sum()), "free_recovery": rec_all,
            "terminal_recovery": float(correct[f_term].mean()) if f_term.sum() else None,
            "interior_recovery": float(correct[f_int].mean()) if f_int.sum() else None,
            "blosum_free": float(sc[free].mean()), "blosum_free_null": null_free,
            "blosum_miss": float(sc[miss].mean()), "blosum_miss_null": null_miss,
            "conservative_frac": float((sc[free] >= 0).mean()),
        }

        if poscm is not None:
            pa = prior_accuracy(d, free, poscm, posm, best)
            print(f"  PRIOR P(r|pos,L,cyc) {pa['prior_pos_chem_acc']:.4f}   "
                  f"P(r|pos,L) {pa['prior_pos_acc']:.4f}   global('{pa['global_argmax']}') "
                  f"{pa['prior_global_acc']:.4f}   backoff {pa['backoff_frac']:.2%}")
            print(f"  >>> MODEL MARGIN over the fair prior: "
                  f"{rec_all - pa['prior_pos_chem_acc']:+.4f}")
            row.update(pa)
            row["margin_over_fair_prior"] = rec_all - pa["prior_pos_chem_acc"]

            # The margin MUST be split the same way the model was, or it repeats the error this
            # project already made once: a terminal anchor is forced by the chemistry, so the
            # prior gets it nearly free and a pooled margin can hide that the model's apparent
            # lead lives entirely in positions that were never a prediction. Report where the
            # margin actually comes from.
            for name, s in (("terminal", f_term), ("interior", f_int)):
                if int(s.sum()) == 0:
                    continue
                ps = prior_accuracy(d, s, poscm, posm, best)
                mrec = float(correct[s].mean())
                print(f"    {name:8s} model {mrec:.4f}  prior {ps['prior_pos_chem_acc']:.4f}  "
                      f"margin {mrec - ps['prior_pos_chem_acc']:+.4f}  n={int(s.sum())}")
                row[f"{name}_prior"] = ps["prior_pos_chem_acc"]
                row[f"{name}_margin"] = mrec - ps["prior_pos_chem_acc"]
        out_rows.append(row)

    if args.json_out and out_rows:
        Path(args.json_out).write_text(json.dumps(out_rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
