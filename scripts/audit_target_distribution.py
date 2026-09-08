"""Report the receptor/peptide statistics the denoiser actually consumes, per target set.

Why this exists
---------------
A staged target set can be silently out-of-distribution in ways that never show up as an
error: the model reads `effective_chain_id` and `pos_in_segment` (via `chain_idx_pair` and
`rel_seq_sep`), and those come from how the receptor is FRAGMENTED, not from anything in the
metadata. Measured on the LNR sets, that one axis was worth +22 to +34 points of ring
closure. This script prints every statistic on that axis for any number of staged parquets
side by side, so a new set can be checked against the training distribution BEFORE the GPU
arm is launched.

Segments are computed with the featurizer's own rule (segment_utils.compute_segment_info):
a new segment starts on a chain change, a numbering jump, or a C_{i-1}->N_i distance above
2.0 A. The physical term dominates in practice -- a spatial crop leaves real backbone gaps --
so numbering alone is usually redundant. See docs/README_TARGET_DISTRIBUTION_MATCHING.md.

Usage:
    python scripts/audit_target_distribution.py \
        --set "CPSea (reference)=CPSea_data/control/cpsea_val_control.parquet" \
        --set "LNR 14A=CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet"
"""

from __future__ import annotations

import argparse
import gzip
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

CN_BREAK_CUTOFF = 2.0
BACKBONE = {"N", "CA", "C", "O"}


def parse(path: str, binder_chain: str):
    """-> (receptor residues, peptide residues, n_altloc_chars, n_low_occupancy, n_dup_atoms).

    Heavy atoms only, so the numbers mean the same thing for files that carry hydrogens
    and files that do not."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rec, pep = OrderedDict(), OrderedDict()
    n_alt = n_occ = 0
    seen = Counter()
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = (line[76:78].strip() or line[12:16].strip()[:1])
            if element == "H":
                continue
            if line[16] != " ":
                n_alt += 1
            try:
                if float(line[54:60]) < 1.0:
                    n_occ += 1
            except ValueError:
                pass
            name = line[12:16].strip()
            key = (line[21], int(line[22:26]), line[26])
            seen[key + (name,)] += 1
            target = pep if line[21] == binder_chain else rec
            entry = target.setdefault(key, {"atoms": set(), "xyz": {}})
            entry["atoms"].add(name)
            if name in ("C", "N"):
                entry["xyz"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    n_dup = sum(v - 1 for v in seen.values() if v > 1)
    return rec, pep, n_alt, n_occ, n_dup


def segment_runs(res) -> list[int]:
    """Run lengths under the featurizer's rule (chain change OR numbering jump OR C-N break)."""
    keys = list(res)
    runs, run = [], 1
    for i in range(1, len(keys)):
        prev, curr = res[keys[i - 1]], res[keys[i]]
        same_chain = keys[i][0] == keys[i - 1][0]
        num_jump = keys[i][1] != keys[i - 1][1] + 1
        c, n = prev["xyz"].get("C"), curr["xyz"].get("N")
        physical = c is None or n is None or float(np.linalg.norm(np.array(c) - np.array(n))) > CN_BREAK_CUTOFF
        if (not same_chain) or num_jump or physical:
            runs.append(run)
            run = 1
        else:
            run += 1
    runs.append(run)
    return runs


def audit(name: str, parquet: str) -> dict:
    df = pd.read_parquet(parquet)
    segs, maxrun, tgtlen, peplen, fullbb, natoms, nchains = [], [], [], [], [], [], []
    runs_all: list[int] = []
    n_alt_files = n_occ_files = n_dup_total = 0
    for _, row in df.iterrows():
        rec, pep, n_alt, n_occ, n_dup = parse(str(row["path"]), str(row["binder_chain_id"]))
        if not rec or not pep:
            continue
        runs = segment_runs(rec)
        runs_all += runs
        segs.append(len(runs))
        maxrun.append(max(runs))
        tgtlen.append(len(rec))
        peplen.append(len(pep))
        fullbb.append(np.mean([BACKBONE <= v["atoms"] for v in rec.values()]))
        natoms.append(np.mean([len(v["atoms"]) for v in rec.values()]))
        nchains.append(len({k[0] for k in rec}))
        n_alt_files += n_alt > 0
        n_occ_files += n_occ > 0
        n_dup_total += n_dup
    runs_arr = np.array(runs_all)
    return {
        "set": name,
        "n": len(segs),
        "segments": np.median(segs),
        "max_run": np.median(maxrun),
        "seg_med": np.median(runs_arr),
        "seg_p90": np.percentile(runs_arr, 90),
        "frag<5": np.mean(runs_arr < 5) * 100,
        "tgt_len": np.median(tgtlen),
        "pep_len": np.median(peplen),
        "fullBB%": np.mean(fullbb) * 100,
        "at/res": np.mean(natoms),
        "chains": np.median(nchains),
        "altloc_files": n_alt_files,
        "occ<1_files": n_occ_files,
        "dup_atoms": n_dup_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, metavar="NAME=PARQUET",
                    help="Repeatable. The FIRST --set is treated as the reference distribution.")
    args = ap.parse_args()

    results = []
    for spec in args.set:
        if "=" not in spec:
            raise SystemExit(f"--set must be NAME=PARQUET, got {spec!r}")
        name, parquet = spec.split("=", 1)
        results.append(audit(name, parquet))

    cols = ["set", "n", "segments", "max_run", "seg_med", "seg_p90", "frag<5", "tgt_len", "pep_len"]
    def cell(v) -> str:
        return v if isinstance(v, str) else (str(v) if isinstance(v, int) else f"{v:.1f}")

    widths = {c: max(len(c), max(len(cell(r[c])) for r in results)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in results:
        print("  ".join(cell(r[c]).ljust(widths[c]) for c in cols))

    print("\nintegrity / provenance (these should not differ from the reference):")
    for r in results:
        print(f"  {r['set']:24s} fullBB {r['fullBB%']:5.1f}%  heavy-at/res {r['at/res']:4.1f}  "
              f"chains {r['chains']:.1f}  altloc-files {r['altloc_files']:3d}  occ<1-files {r['occ<1_files']:3d}  "
              f"DUPLICATE ATOMS {r['dup_atoms']}")

    ref = results[0]
    print(f"\ndeltas vs reference '{ref['set']}' (|.| that matters: max_run, frag<5, segments):")
    for r in results[1:]:
        print(f"  {r['set']:24s} segments {r['segments']-ref['segments']:+5.1f}  max_run {r['max_run']-ref['max_run']:+6.1f}  "
              f"frag<5 {r['frag<5']-ref['frag<5']:+5.1f}pp  tgt_len {r['tgt_len']-ref['tgt_len']:+6.1f}  "
              f"pep_len {r['pep_len']-ref['pep_len']:+5.1f}")
    if any(r["dup_atoms"] for r in results):
        print("\nWARNING: duplicate atoms present -- altloc copies are reaching the featurizer. Fix staging.")


if __name__ == "__main__":
    main()
