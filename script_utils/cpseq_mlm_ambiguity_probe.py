"""Estimate the irreducible accuracy and loss of the cpseq_mlm task from the data alone.

THE CLAIM UNDER TEST. The arm's docs argue a hidden-position loss near 2.4 nats is
intrinsically high because many amino acids are interchangeable. Interchangeability is
plausible, but it does not by itself establish that 2.4 is near the best achievable loss.
Nothing about the model is needed to settle this: the ceiling for a predictor is a property
of its INPUT SET, and both the model's inputs and the data are on disk.

HOW. Group corpus positions by the context a predictor can see, then ask how much the
recorded residue still varies inside a group. Three groupings, in increasing strictness:

  chem_prior   (pos, L, cyc)                   -- what the chemistry-aware lookup sees.
  model        (pocket, L, cyc, pos)           -- what the MODEL sees. This grouping's
                                                  ceiling is the number the claim needs.
  cluster      (cluster_id, L, cyc, pos)       -- coarser than the model's inputs, so it
                                                  OVER-groups and its ceiling is a LOWER
                                                  bound on the model's.

Two estimates per grouping, biased in OPPOSITE directions, which is why both are printed:

  * LOO-argmax accuracy -- for each position, predict the argmax of its group EXCLUDING
    itself. Leave-one-out because the plug-in version is degenerate: a group of size 1 has
    max-probability 1.0 and would report a 100% ceiling from a single observation. LOO
    estimates from m-1 samples, so it UNDERSTATES the ceiling the true conditional
    distribution would reach.
  * Conditional entropy -- plug-in, plus the Miller-Madow correction (+(K-1)/2m). Plug-in
    entropy is biased DOWN on small groups, i.e. it overstates how determined the residue
    is and so flatters the ceiling claim; the corrected figure is the conservative one.

Results are reported by GROUP-SIZE STRATUM, not pooled, because both biases shrink with m:
if the ceiling rises with group size, the pooled number is an underestimate and the trend
is the thing to read.

WHAT WOULD REFUTE THE CEILING CLAIM. A model-grouping conditional entropy well below the
~2.4 nats the run reports, or a LOO-argmax accuracy well above its 26.7%, means headroom
exists for a predictor with exactly the inputs this model already has -- so the plateau is
not a property of the task.

WHAT THIS CANNOT SAY, AND A TRAP MEASURED WHILE BUILDING IT. If exact pocket duplicates are
rare the model grouping has little support, and the probe must report that rather than
extrapolate from a handful of groups. Coverage is therefore printed FIRST, and the cluster
grouping is the fallback whose bound direction is known.

Worse, a group of IDENTICAL rows is not evidence of a determined task. The pocket here is
extracted by radius around the peptide, so two rows sharing a pocket string to the
character are near-duplicate complexes rather than two different peptides against one
target -- and on the smoke corpus that is exactly what the model grouping found: 6 shared
contexts out of 5994, reporting LOO 0.80 and H 0.19, i.e. "nearly deterministic", purely
from duplicate entries. Any group offering only ONE distinct peptide sequence is therefore
dropped before estimating: a ceiling is about alternative answers, and a group with one
answer repeated has none. `--min-distinct` controls this and `distinct_dropped_*` reports
how much it removed, because that count is itself the measurement of how much of the
apparent determinism was redundancy.

IMPLEMENTATION NOTE -- why this streams in two passes. Keyed naively, the model grouping
has one key per position, which is ~26M keys over this corpus and tens of GB of dicts, and
almost all of them are singletons that carry no information. Instead: pass 1 counts ROWS
per row-level context, pass 2 accumulates per-position residues only for contexts seen
more than once. Rows are never retained and the pocket string is reduced to a digest on
read, so peak memory is the key tables rather than 2.5M pocket sequences.

Usage:
    python script_utils/cpseq_mlm_ambiguity_probe.py \
        --corpus-dir /zfsauton/scratch/yixiz/cpseq_mlm/corpus \
        --out .../ambiguity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "script_utils") not in sys.path:
    sys.path.insert(0, str(_REPO / "script_utils"))

from cpseq_mlm_data import AA20, assert_one_shard_grouping  # noqa: E402

# Strata on group size m. The ceiling estimates are only trustworthy in the larger ones;
# the small strata are reported so the trend with m is visible rather than hidden.
STRATA = ((2, 2), (3, 4), (5, 9), (10, 29), (30, 99), (100, 10**9))

GROUPINGS = ("chem_prior", "model", "cluster")


def pocket_digest(pocket_seq: str) -> str:
    """Stable 16-byte digest of a pocket sequence.

    blake2b and not `hash()`: Python salts string hashing per process (PYTHONHASHSEED), so
    `hash()` would make this probe's grouping non-reproducible between runs, and its 64-bit
    space collides often enough at 2.5M keys to silently merge two different pockets into
    one group -- which would manufacture exactly the ambiguity the probe is measuring.
    """
    return hashlib.blake2b(pocket_seq.encode(), digest_size=16).hexdigest()


def pep_digest(pep_seq: str) -> str:
    """Short stable digest of a peptide, used only to count DISTINCT answers per group.

    8 bytes rather than 16 because millions of these are held in the pass-1 table and the
    only question asked of them is equality. A collision would make two different peptides
    look identical, which drops a group from the estimate rather than inventing one -- the
    conservative direction.
    """
    return hashlib.blake2b(pep_seq.encode(), digest_size=8).hexdigest()


def row_key(grouping: str, d: dict) -> tuple | None:
    """The ROW-level part of a grouping key (everything but the position)."""
    L = len(d["pep_seq"])
    cyc = int(d["cyc_type"])
    if grouping == "chem_prior":
        return (L, cyc)
    if grouping == "model":
        return (pocket_digest(d["pocket_seq"]), L, cyc)
    if grouping == "cluster":
        cl = d.get("cluster_id")
        return None if cl is None else (cl, L, cyc)
    raise ValueError(grouping)


def iter_rows(files: list[Path]):
    """Stream corpus rows as raw dicts.

    Mirrors `cpseq_mlm_data.load_corpus`'s refusal to skip a malformed line: concurrent
    appends on this filesystem were measured to NUL-corrupt rows, and a reader that skips
    corruption turns "a third of my data vanished" into a silent change in the answer.
    """
    assert_one_shard_grouping([str(p) for p in files])
    for p in files:
        with open(p) as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{p}:{lineno} is not valid JSON ({e}). Refusing to skip it -- "
                        f"see the corpus builder on shard output corruption."
                    ) from e


def stratum_of(m: int) -> tuple[int, int] | None:
    for lo, hi in STRATA:
        if lo <= m <= hi:
            return (lo, hi)
    return None


def analyse(counts: dict[tuple, dict[str, int]]) -> dict:
    """LOO-argmax accuracy and conditional entropy, by group-size stratum and pooled.

    Groups of size 1 carry no information about ambiguity -- there is nothing to vary --
    so they are excluded from every estimate and counted separately. Including them would
    drive the ceiling toward 1.0 purely by counting singletons.
    """
    per_stratum: dict[tuple[int, int], dict] = {
        s: {"n_groups": 0, "n_pos": 0, "loo_hit": 0, "h_sum": 0.0, "h_mm_sum": 0.0}
        for s in STRATA
    }
    n_singleton_groups = n_singleton_pos = 0
    pooled = {"n_groups": 0, "n_pos": 0, "loo_hit": 0, "h_sum": 0.0, "h_mm_sum": 0.0}

    for c in counts.values():
        m = sum(c.values())
        if m < 2:
            n_singleton_groups += 1
            n_singleton_pos += m
            continue
        st = stratum_of(m)
        if st is None:
            continue

        # Leave-one-out argmax. For a member with residue r the held-out counts are c with
        # c[r] reduced by one; ties break by AA20 order, deterministically, so the result
        # does not depend on dict iteration order.
        loo_hit = 0
        for r, cr in c.items():
            held = dict(c)
            held[r] = cr - 1
            best = max(AA20, key=lambda a: (held.get(a, 0), -AA20.index(a)))
            if best == r:
                loo_hit += cr

        h = 0.0
        for cr in c.values():
            p = cr / m
            if p > 0:
                h -= p * math.log(p)
        # Miller-Madow: plug-in entropy from m samples is biased down by about (K-1)/2m,
        # with K the number of outcomes actually observed.
        k_obs = sum(1 for cr in c.values() if cr > 0)
        h_mm = h + (k_obs - 1) / (2 * m)

        for bucket in (per_stratum[st], pooled):
            bucket["n_groups"] += 1
            bucket["n_pos"] += m
            bucket["loo_hit"] += loo_hit
            bucket["h_sum"] += h * m
            bucket["h_mm_sum"] += h_mm * m

    def finish(b: dict) -> dict:
        n = b["n_pos"]
        if n == 0:
            return {"n_groups": b["n_groups"], "n_positions": 0}
        return {
            "n_groups": b["n_groups"],
            "n_positions": n,
            "loo_argmax_acc": b["loo_hit"] / n,
            "cond_entropy_nats": b["h_sum"] / n,
            "cond_entropy_nats_mm": b["h_mm_sum"] / n,
            "cond_perplexity_mm": math.exp(b["h_mm_sum"] / n),
        }

    return {
        "pooled": finish(pooled),
        "by_stratum": {f"{lo}-{hi}": finish(per_stratum[(lo, hi)]) for lo, hi in STRATA},
        "n_singleton_groups": n_singleton_groups,
        "n_singleton_positions": n_singleton_pos,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default="/zfsauton/scratch/yixiz/cpseq_mlm/corpus")
    ap.add_argument("--globs", default="train_*.jsonl,val_*.jsonl",
                    help="comma-separated; both splits by default, since a CEILING is a "
                         "property of the task and more rows mean more duplicate contexts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    files: list[Path] = []
    for g in args.globs.split(","):
        g = g.strip()
        if g:
            files.extend(sorted(corpus_dir.glob(g)))
    if not files:
        raise SystemExit(f"FATAL: no shards matching {args.globs} under {corpus_dir}")
    print(f"shards: {len(files)}")

    # --- pass 1: how many ROWS share each row-level context, and whether those rows offer
    # more than one distinct peptide. Only contexts seen more than once can exhibit
    # variation, and for the model grouping that is expected to be a small minority -- so
    # this pass is what makes pass 2 affordable.
    #
    # Each entry is [n_rows, first_pep_digest, saw_a_different_one]. A set of digests would
    # answer the same question and cost ~200 bytes per context on top of millions of
    # contexts; the only query is "is there a second distinct peptide", which one flag
    # answers.
    row_info: dict[str, dict[tuple, list]] = {g: {} for g in GROUPINGS}
    n_rows = 0
    for d in iter_rows(files):
        n_rows += 1
        pd = pep_digest(d["pep_seq"])
        for g in GROUPINGS:
            k = row_key(g, d)
            if k is None:
                continue
            e = row_info[g].get(k)
            if e is None:
                row_info[g][k] = [1, pd, False]
            else:
                e[0] += 1
                if not e[2] and pd != e[1]:
                    e[2] = True
    print(f"pass 1: rows={n_rows}")
    multi: dict[str, set[tuple]] = {}
    dropped: dict[str, dict[str, int]] = {}
    for g in GROUPINGS:
        shared = {k: e for k, e in row_info[g].items() if e[0] >= 2}
        multi[g] = {k for k, e in shared.items() if e[2]}
        dup_only = {k: e for k, e in shared.items() if not e[2]}
        dropped[g] = {
            "contexts_shared": len(shared),
            "contexts_one_distinct_peptide": len(dup_only),
            "rows_in_those": sum(e[0] for e in dup_only.values()),
        }
        print(f"  {g:<11} row-contexts={len(row_info[g]):<9} "
              f"shared={len(shared):<8} usable (>=2 distinct peptides)={len(multi[g]):<8} "
              f"dropped as duplicates={len(dup_only)} ({dropped[g]['rows_in_those']} rows)")
    del row_info

    # --- pass 2: per-position residue counts, restricted to shared contexts
    counts: dict[str, dict[tuple, dict[str, int]]] = {g: defaultdict(dict) for g in GROUPINGS}
    n_positions = 0
    for d in iter_rows(files):
        pep = d["pep_seq"]
        keys = {g: row_key(g, d) for g in GROUPINGS}
        for pos, ch in enumerate(pep):
            if ch not in AA20:
                continue
            n_positions += 1
            for g in GROUPINGS:
                k = keys[g]
                if k is None or k not in multi[g]:
                    continue
                dd = counts[g][(*k, pos)]
                dd[ch] = dd.get(ch, 0) + 1
    print(f"pass 2: positions={n_positions}")

    result = {
        "corpus_dir": str(corpus_dir),
        "globs": args.globs,
        "n_rows": n_rows,
        "n_positions": n_positions,
        "groupings": {},
    }
    for g in GROUPINGS:
        res = analyse(counts[g])
        res["n_keys"] = len(counts[g])
        res["n_row_contexts_usable"] = len(multi[g])
        res["duplicate_filter"] = dropped[g]
        # Coverage is against ALL corpus positions, including those whose context was a
        # singleton and so never entered pass 2: that is the honest denominator for "how
        # much of the task does this estimate speak for".
        res["coverage_of_positions"] = res["pooled"].get("n_positions", 0) / max(n_positions, 1)
        result["groupings"][g] = res

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== coverage: how much of the corpus sits in a group that can show variation ===")
    for g in GROUPINGS:
        r = result["groupings"][g]
        f = r["duplicate_filter"]
        print(f"  {g:<11} position-keys={r['n_keys']:<9} "
              f"coverage={r['coverage_of_positions']:.3f}  "
              f"(singleton position-groups: {r['n_singleton_groups']}; "
              f"{f['contexts_one_distinct_peptide']}/{f['contexts_shared']} shared contexts "
              f"dropped as pure duplicates)")
    print("  Low coverage for `model` means exact pocket duplicates are rare and its")
    print("  ceiling rests on few groups; read `cluster` as the lower bound instead.")
    print("  A large duplicate drop means the apparent determinism of that grouping was")
    print("  dataset redundancy, not a property of the task.")

    for g in GROUPINGS:
        r = result["groupings"][g]
        p = r["pooled"]
        if not p.get("n_positions"):
            print(f"\n=== {g}: no group of size >= 2; nothing estimable ===")
            continue
        print(f"\n=== {g}: ceiling for a predictor seeing exactly this context ===")
        print(f"  pooled   LOO-argmax acc {p['loo_argmax_acc']:.4f}   "
              f"H {p['cond_entropy_nats']:.4f} nats (MM {p['cond_entropy_nats_mm']:.4f}, "
              f"ppl {p['cond_perplexity_mm']:.2f})")
        print(f"  {'group size':<12}{'n_pos':>10}{'LOO acc':>10}{'H (MM)':>10}")
        for s in STRATA:
            d = r["by_stratum"][f"{s[0]}-{s[1]}"]
            if not d.get("n_positions"):
                continue
            label = f"{s[0]}-{s[1]}" if s[1] < 10**9 else f">={s[0]}"
            print(f"  {label:<12}{d['n_positions']:>10}{d['loo_argmax_acc']:>10.4f}"
                  f"{d['cond_entropy_nats_mm']:>10.4f}")
        print("  LOO acc rising with group size means the pooled figure UNDERSTATES the")
        print("  ceiling; read the largest stratum that still has support.")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
