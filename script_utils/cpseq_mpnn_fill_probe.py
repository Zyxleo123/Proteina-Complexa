#!/usr/bin/env python3
"""Does ProteinMPNN fill a mask informatively? The same instrument the other two failed.

Before any folding. `README_CPSEQ_MASK_ARM.md` §4 and §6 killed two candidate fillers with
one measurement -- recovery of the native residue at HIDDEN positions, against the
positional prior -- and it costs minutes. Running it on MPNN before spending GPU-days on
refolds is the whole point of having the instrument.

    CPSea unmasking LM   argmax native 2/35, per-position perplexity 16.3 of 20
    seq-cond co-generator free_recovery 0.044-0.058 vs a 0.127 positional prior
    ProteinMPNN          <- this script

Scored on hidden positions ONLY. A revealed position is pinned by `fixed_positions` and
comes back by construction; counting it would be the copy test that made `cond_recovery`
1.000 look like a capability.

The prior is fitted on CPSea and evaluated on THIS cohort's own peptides, so the bar
matches the sequences being scored rather than a different tranche.

Recovering the native is not the design objective -- a proposer should eventually disagree
with the native on purpose. What this separates is a filler that carries position-specific
information from one that does not, which is the thing that sank the other two.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from cpseq_mask_mpnn_fill import reference_backbone  # noqa: E402
from cpseq_mpnn_propose import (  # noqa: E402
    PEPTIDE_CHAIN,
    run_mpnn,
    thread_sequence_onto_backbone,
)
from cpseq_seqcond_free_baseline import fit  # noqa: E402

from proteinfoundation.cpseq.search import budget_for  # noqa: E402

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="the run directory")
    ap.add_argument("--cpsea-shard", required=True, help="fits the priors only")
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--masks-per-parent", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--model-name", default="v_48_020")
    ap.add_argument("--omit-aas", default="XC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parents", default="")
    args = ap.parse_args()

    cohort = json.loads((args.out_dir / "cohort.json").read_text())["cohort"]
    if args.parents:
        want = set(args.parents.split(","))
        cohort = [p for p in cohort if p in want]

    work = args.out_dir / "mask_fill" / "mpnn_probe"
    work.mkdir(parents=True, exist_ok=True)

    train = [json.loads(l)["pep_seq"] for l in Path(args.cpsea_shard).read_text().splitlines()]
    predict, overall = fit(train, 20)
    marginal_residue = overall.most_common(1)[0][0]

    rng = random.Random(args.seed)
    hit = tot = 0
    pos_hit = marg_hit = 0
    per_parent = []
    noop_masks = 0
    n_masks_total = 0

    for pid in cohort:
        journal = args.out_dir / "state" / pid / "outcomes.jsonl"
        if not journal.exists():
            continue
        original = None
        for line in journal.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("original_sequence"):
                original = row["original_sequence"]
                break
        if not original:
            continue
        L = len(original)
        bb = reference_backbone(args.out_dir, pid, original, work)
        if bb is None:
            print(f"[probe] {pid}: no reference backbone; excluded")
            continue
        threaded = work / f"{pid}_threaded.pdb"
        thread_sequence_onto_backbone(
            bb, threaded, peptide_chain=PEPTIDE_CHAIN, original=original
        )

        p_hit = p_tot = 0
        for k in range(args.masks_per_parent):
            # Half the peptide, which is also the search's own budget ceiling, so the
            # regime matches the one a proposal actually runs in.
            size = max(1, budget_for(L))
            mask = sorted(rng.sample(range(L), size))
            fixed = [i for i in range(L) if i not in mask]
            designs = run_mpnn(
                pdb_path=threaded, fixed_positions=fixed,
                out_dir=work / f"{pid}_probe{k}", n_seq=1,
                temperature=args.temperature, seed=args.seed * 1000 + k,
                model_name=args.model_name, omit_aas=args.omit_aas,
            )
            n_masks_total += 1
            seq = next((d.get("sequence") for d in designs
                        if d.get("status") == "ok" and d.get("sequence")), None)
            if not seq or len(seq) != L:
                continue
            if seq == original:
                noop_masks += 1
            for i in mask:
                if original[i] not in AA20:
                    continue
                p_tot += 1
                p_hit += seq[i] == original[i]
                pred, _src = predict(i, L)
                pos_hit += pred == original[i]
                marg_hit += marginal_residue == original[i]
        hit += p_hit
        tot += p_tot
        per_parent.append({"parent_id": pid, "length": L, "hidden": p_tot,
                           "recovered": p_hit,
                           "rate": round(p_hit / p_tot, 4) if p_tot else None})
        print(f"[probe] {pid:24s} L={L:2d}  recovered {p_hit:3d}/{p_tot:3d}"
              f"  {'%.3f' % (p_hit / p_tot) if p_tot else '-'}", flush=True)

    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_parents": len(per_parent),
        "n_hidden_positions": tot,
        "mpnn_free_recovery": round(hit / tot, 4) if tot else None,
        "positional_prior": round(pos_hit / tot, 4) if tot else None,
        "marginal_prior": round(marg_hit / tot, 4) if tot else None,
        "marginal_residue": marginal_residue,
        "noop_masks": noop_masks,
        "n_masks": n_masks_total,
        "noop_rate": round(noop_masks / n_masks_total, 4) if n_masks_total else None,
        "mpnn_model": args.model_name,
        "temperature": args.temperature,
        "mask_regime": "floor(L/2) positions, uniform, matching the search budget",
        "per_parent": per_parent,
        "note": (
            "Hidden positions only. Recovery of the NATIVE residue is not the design "
            "objective; it separates a filler that carries position-specific information "
            "from one that does not."
        ),
    }
    print(f"\nMPNN free_recovery {out['mpnn_free_recovery']}  "
          f"positional prior {out['positional_prior']}  "
          f"marginal {out['marginal_prior']}  over {tot} hidden positions")
    print(f"no-op masks (MPNN reproduced the original exactly): "
          f"{noop_masks}/{n_masks_total}")
    args.result.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
