#!/usr/bin/env python3
"""Does partial sequence conditioning DESIGN the unrevealed residues, or copy the LP there?

The question this answers is whether partial-reveal generation is a genuine proposal
mechanism or a reparameterisation of "fold the LP sequence". It matters because the arm
only earns its place if the unrevealed positions come back as something the LP did not
already say.

The discriminating statistic is **overall sequence recovery against the reveal fraction**:

    designing  ->  recovery ~= keep_frac + (1 - keep_frac) * p_chance
    copying    ->  recovery ~= 1.0  regardless of keep_frac

with `p_chance` the amino-acid marginal, ~0.05-0.15 on these peptides. At keep_frac 0.5
that is ~0.53-0.58 if the model designs, against ~1.0 if it reproduces the LP. The gap is
wide enough that a modest sample settles it.

Caveat that bounds the conclusion: the reveal mask is drawn at RANDOM per example by
`sample_sequence_conditioning_mask`, so this script cannot know which positions were
revealed. It therefore measures recovery POOLED over revealed and unrevealed positions and
compares that pooled number to the two predictions above -- it cannot report per-position
behaviour. Controlled masks (the `[b, n]` form) are what would allow that, and plumbing
them to inference is the follow-up this probe is meant to justify or kill.

Attribution to a parent is by receptor sequence, reusing `cpseq_score_seqcond`'s matcher
(exact, then containment, because the loader crops receptors past 256 residues).

Usage:
    python script_utils/cpseq_partial_reveal_probe.py \
        --dataset-dir /data/user_data/yixiz/cpseq/tranche1 \
        --dump kf1.00=<dir>/rosetta_dump --dump kf0.50=<dir>/rosetta_dump
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_score_seqcond import _match_parent, parent_index  # noqa: E402

from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402
from proteinfoundation.cpseq.geometry import read_chain, sequence_of  # noqa: E402


def virtual_cb(res: dict) -> np.ndarray:
    N, CA, C = res["N"], res["CA"], res["C"]
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def aa_marginal_recovery(ref: str, pool: Counter) -> float:
    """Chance recovery: the probability a residue drawn from `pool` matches `ref`."""
    total = sum(pool.values()) or 1
    return sum(pool[a] / total for a in ref) / max(len(ref), 1)


def probe(dump: Path, index: dict, chain: str = "B") -> list[dict]:
    rows = []
    for man in sorted(glob.glob(str(dump / "step_*" / "manifest_rank*.jsonl"))):
        step_dir = Path(man).parent
        for line in Path(man).read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            pdb = step_dir / r["pdb"]
            if not pdb.exists():
                continue
            target_chains = [c for c in str(r.get("target_chains", "A")) if c.strip()]
            parsed = parse_structure(pdb)
            rec = "".join(chain_sequence(parsed, c) for c in target_chains)
            info = _match_parent(rec, index)
            if info is None:
                rows.append({"status": "unmatched_receptor"})
                continue
            res = read_chain(pdb, chain)
            if len(res) < 2:
                continue
            got = sequence_of(res)
            ref = info["reference_sequence"]
            if len(got) != len(ref):
                rows.append({"status": f"length_mismatch:{len(got)}!={len(ref)}"})
                continue
            f, l = res[0][1], res[-1][1]
            try:
                cb0 = f["CB"] if "CB" in f else virtual_cb(f)
                cbL = l["CB"] if "CB" in l else virtual_cb(l)
                cb = float(np.linalg.norm(cb0 - cbL))
            except KeyError:
                cb = None
            n_match = sum(1 for a, b in zip(ref, got) if a == b)
            rows.append({
                "status": "ok",
                "parent_id": info["parent_id"],
                "lp_sequence": ref,
                "generated_sequence": got,
                "L": len(ref),
                "recovery": n_match / len(ref),
                "n_mutations": len(ref) - n_match,
                "budget": len(ref) // 2,
                "within_budget": (len(ref) - n_match) <= len(ref) // 2,
                "nc_A": round(float(np.linalg.norm(f["N"] - l["C"])), 3),
                "cb_A": None if cb is None else round(cb, 3),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--dump", action="append", required=True, metavar="LABEL=PATH",
                    help="label should encode the reveal fraction, e.g. kf0.50=<path>")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    index = parent_index(args.dataset_dir)
    all_rows: dict[str, list[dict]] = {}
    for spec in args.dump:
        label, path = spec.split("=", 1)
        all_rows[label] = probe(Path(path), index)

    # Chance baseline from the pooled generated residue composition: what recovery a
    # random residue drawn from the model's own output distribution would give.
    pool = Counter()
    for rows in all_rows.values():
        for r in rows:
            if r.get("status") == "ok":
                pool.update(r["generated_sequence"])

    print(f"{'run':10s} {'n':>4s} {'recovery':>9s} {'if_design':>10s} {'if_copy':>8s} "
          f"{'verdict':>10s} {'n_mut med':>10s} {'in_budget':>10s} {'N-C med':>8s} {'frozen':>7s}")
    print('-' * 104)
    summary = []
    for label, rows in all_rows.items():
        ok = [r for r in rows if r.get("status") == "ok"]
        if not ok:
            bad = Counter(r.get("status") for r in rows)
            print(f"{label:10s} {'0':>4s}  no usable rows: {dict(bad)}")
            continue
        try:
            kf = float(label.replace("kf", ""))
        except ValueError:
            kf = float("nan")
        rec = float(np.mean([r["recovery"] for r in ok]))
        chance = float(np.mean([aa_marginal_recovery(r["lp_sequence"], pool) for r in ok]))
        if_design = kf + (1 - kf) * chance
        verdict = "-"
        if not math.isnan(kf):
            verdict = "DESIGNS" if abs(rec - if_design) < abs(rec - 1.0) else "COPIES"
        nc = [r["nc_A"] for r in ok]
        frozen = float(np.mean([1.1 <= r["nc_A"] <= 1.6 for r in ok]))
        print(f"{label:10s} {len(ok):4d} {rec:9.3f} {if_design:10.3f} {1.0:8.3f} "
              f"{verdict:>10s} {np.median([r['n_mutations'] for r in ok]):10.1f} "
              f"{np.mean([r['within_budget'] for r in ok]):10.3f} "
              f"{np.median(nc):8.3f} {frozen:7.3f}")
        summary.append({"run": label, "keep_frac": kf, "n": len(ok), "recovery": round(rec, 4),
                        "expected_if_designing": round(if_design, 4), "chance": round(chance, 4),
                        "verdict": verdict,
                        "median_n_mutations": float(np.median([r["n_mutations"] for r in ok])),
                        "frac_within_budget": round(float(np.mean([r["within_budget"] for r in ok])), 4),
                        "median_nc_A": round(float(np.median(nc)), 3),
                        "frozen_nc_pass": round(frozen, 4)})

    print("\nResidue composition of generated peptides (pooled):")
    tot = sum(pool.values()) or 1
    print("  " + "  ".join(f"{a}:{pool[a]/tot:.3f}" for a, _ in pool.most_common(10)))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"summary": summary,
             "pooled_residue_fractions": {a: round(c / tot, 5) for a, c in pool.most_common()},
             "per_sample": all_rows}, indent=2))
        print(f"\n[probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
