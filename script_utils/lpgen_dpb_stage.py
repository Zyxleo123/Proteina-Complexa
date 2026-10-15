#!/usr/bin/env python
"""Stage LNR complexes for DiffPepDock WITHOUT letting it rewrite anything we care about.

Why this script exists at all
-----------------------------
DiffPepBuilder's docking preprocessing edits its inputs. Two separate sites:

1. ``normalize_receptor_inputs`` forces the ligand chain to be ``'A'`` and, when that
   requires a remap, calls ``renumber_rec_chain(file_path, receptor_info_path,
   in_place=True, ...)`` -- it **overwrites the input PDB and the receptor-info JSON on
   disk** (``experiments/preprocess_utils.py:211-215``).
2. ``get_motif_center_pos``, which IS on the ligand path (called from
   ``process_batch_dock.py:112``), writes ``infile.replace(".pdb", "_processed.pdb")`` --
   a **sibling file next to every input** (``preprocess_utils.py:283, 351``).

Site 1 is avoidable and this script avoids it. ``_build_chain_map(chain_ids, ligand="A")``
returns the identity map (``preprocess_utils.py:48-57``), the caller then skips the remap
because ``any(chain_map[c] != c)`` is False, and ``renumber_rec_chain`` itself early-returns
on an identity map (``:78-79``). So by writing the peptide as chain ``'A'`` up front, we hit
two independent guards and **no input file is ever modified**.

Site 2 is not avoidable without patching upstream, which is why staging into a scratch
directory is still required: the run will litter ``*_processed.pdb`` beside each input.
That is also why ``--out-dir`` is refused if it sits under the data root -- our complexes
are what ``lpgen_manifest.parquet`` points RFD3 at, and one arm's preprocessing writing
into another arm's training data is a silent cross-arm contamination. Note the chain swap
would not even trip the manifest's length/sequence checks, because only the labels move.

``--verify`` re-hashes the staged files afterwards, so "nothing was rewritten" is a
measurement rather than a belief.

Source data
-----------
LNR ships raw multi-chain PDBs plus ``test.txt``, a TSV of
``pdb_id <tab> receptor_chain <tab> peptide_chain <tab> flag`` with 93 rows -- the same 93
complexes UniMoMo reports its peptide numbers on. The 4th column is 0 throughout. Three
PDBs in ``pdbs/`` are absent from ``test.txt`` (3ask, 3v7d, 5gr9) and are not part of the
benchmark, so they are not staged.

This reads the raw files and writes fresh two-chain complexes, so the LNR source tree is
only ever read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Keep the verbatim-ATOM reader rather than reimplementing PDB parsing: lines are passed
# through untouched apart from the columns we deliberately restamp, so coordinates,
# occupancies, b-factors and element columns cannot be corrupted by reformatting.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpgen_export_native import _is_heavy  # noqa: E402

# The 20 standard residues. DiffPepBuilder maps anything else through
# `restype_3to1.get(res_name, "<unk>")` and raises outright on a numeric residue name, so a
# non-standard residue is better dropped here, loudly and counted, than passed through.
STANDARD = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

PEPTIDE_CHAIN = "A"   # DiffPepBuilder's required ligand chain; see module docstring
RECEPTOR_CHAIN = "B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lnr-dir", default="/zfsauton/scratch/yixiz/LNR",
                   help="LNR root holding pdbs/ and test.txt (read-only)")
    p.add_argument("--out-dir", required=True,
                   help="staging directory; must NOT be under --data-root")
    p.add_argument("--data-root", default="/zfsauton/scratch/yixiz/LPData",
                   help="refuse to stage inside this tree (RFD3 reads complexes from it)")
    p.add_argument("--limit", type=int, default=-1, help=">0 stages only the first N rows")
    p.add_argument("--min-peptide-len", type=int, default=4,
                   help="reject shorter peptides; LNR's own range starts at 4")
    p.add_argument("--no-clean", action="store_true",
                   help="keep stale pdbs/processed cache; a re-stage is then NOT authoritative")
    p.add_argument("--verify", action="store_true",
                   help="re-hash staged files against staged_manifest.json and report drift")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


#: Backbone frame atoms. A residue missing any of these is dropped, because upstream
#: dereferences them unguarded -- `preprocess_utils.py:318` does `j["CA"].coord` for every
#: receptor residue inside the hotspot scan, so one CA-less residue anywhere in the pocket
#: raises `KeyError: 'CA'` and kills the whole batch (MEASURED, job 84140: 89 complexes
#: crashed partway through preprocessing; the 3-complex smoke had no such residue).
#: Crystal structures have these routinely -- disordered residues modelled side-chain-only.
FRAME_ATOMS = ("N", "CA", "C")


def read_chain(pdb_path: Path, chain: str) -> tuple[list, dict, dict]:
    """Group one chain's heavy ATOM lines by residue, verbatim, in file order.

    Returns (ordered residue keys, key -> lines, drop counts).

    Altlocs: only the blank or 'A' alternate is kept. Keeping every altloc would emit two
    atoms with the same name in one residue, which is exactly the duplicate-atom condition
    that makes atomworks refuse a file and would give ESM a residue of ambiguous identity.

    Residues lacking a complete N/CA/C frame are dropped -- see FRAME_ATOMS. The CALLER
    must decide what an incomplete PEPTIDE residue means: silently dropping one shortens
    the peptide and changes the sequence, which would break the k=0 contract.
    """
    res: dict = {}
    order: list = []
    drops = {"hetatm_or_other_chain": 0, "hydrogen": 0, "altloc": 0, "nonstandard": 0,
             "incomplete_backbone": 0}
    for line in open(pdb_path, "r"):
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            drops["hetatm_or_other_chain"] += 1
            continue
        if not _is_heavy(line):
            drops["hydrogen"] += 1
            continue
        if line[16] not in (" ", "A"):
            drops["altloc"] += 1
            continue
        if line[17:20].strip() not in STANDARD:
            drops["nonstandard"] += 1
            continue
        key = line[22:27]           # resseq + icode
        if key not in res:
            res[key] = []
            order.append(key)
        res[key].append(line)

    incomplete = [k for k in order
                  if not set(FRAME_ATOMS) <= {ln[12:16].strip() for ln in res[k]}]
    for k in incomplete:
        drops["incomplete_backbone"] += len(res.pop(k))
    order = [k for k in order if k not in set(incomplete)]
    return order, res, drops


def write_complex(path: Path, pep: tuple, rec: tuple) -> tuple[dict, dict]:
    """Write peptide as chain A then receptor as chain B, restamping only the chain column.

    Peptide FIRST and as 'A' is load-bearing, not cosmetic: `_get_chain_ids` returns chains
    in file order, so this yields chain_ids == ['A', 'B'] and therefore the identity chain
    map -- which is what keeps upstream's in-place rewrite from ever running.

    TER/END go out to the column spec (record name padded to 6, TER carrying the final
    residue's name/chain/number). A bare "TER\\n" is silently an unrecognized record to any
    parser matching line[0:6], and chain termination is exactly what a PDB reader uses to
    tell the two chains apart.

    Residues are RENUMBERED sequentially per chain and insertion codes are dropped. This is
    not tidying -- it defuses an upstream collision. `resid_unique(res)` returns
    ``f"{res.get_parent().id}{res.id[1]}"`` (preprocess_utils.py:254-258), i.e. chain plus
    residue NUMBER with the insertion code discarded, and `rec_keep` built from it is handed
    to `ResSelector`. So with insertion codes present, `B 100` and `B 100A` share the key
    `B100`, selecting one pulls in both, and the written pocket ends up LARGER than the
    `mask_rec` the ESM embedding is built from -- tripping
    ``AssertionError: ESM embedding length 102 does not match sequence length 104``
    (MEASURED, job 84142 on 1fiw; 4 of 80 staged complexes carry insertion codes:
    1fiw, 2hwl, 2qa9, 5fxc). Sequential numbering makes that key injective by construction.

    The original numbering is returned so the manifest keeps it and a pose can still be
    mapped back to the source residue.
    """
    serial = 1
    counts = {}
    renumber: dict = {}
    with open(path, "w") as fh:
        for chain_id, (order, res) in ((PEPTIDE_CHAIN, pep), (RECEPTOR_CHAIN, rec)):
            last = None
            last_num = None
            n = 0
            for new_num, key in enumerate(order, start=1):
                renumber[f"{chain_id}{new_num}"] = key.strip()
                for line in res[key]:
                    # Restamp serial (1-11), chain id (22), resSeq (23-26) and blank the
                    # insertion code (27). Everything from column 28 on -- coordinates,
                    # occupancy, b-factor, element -- is passed through verbatim.
                    fh.write(f"ATOM  {serial:>5d}{line[11:21]}{chain_id}"
                             f"{new_num:>4d} {line[27:]}")
                    serial += 1
                    n += 1
                    last = line
                    last_num = new_num
            if last is not None:
                fh.write(f"TER   {serial:>5d}      {last[17:20]} "
                         f"{chain_id}{last_num:>4d} \n")
                serial += 1
            counts[chain_id] = n
        fh.write("END   \n")
    return counts, renumber


def sequence_of(order: list, res: dict) -> str:
    return "".join(THREE_TO_ONE[res[k][0][17:20].strip()] for k in order)


def main() -> int:
    args = parse_args()
    lnr = Path(args.lnr_dir)
    out = Path(args.out_dir).resolve()
    data_root = Path(args.data_root).resolve()

    # Hard guard, not advice. Staging inside the data root would drop *_processed.pdb
    # siblings into the directory RFD3 reads its training complexes from.
    if out == data_root or data_root in out.parents:
        print(f"FATAL: --out-dir {out} is inside the data root {data_root}.\n"
              f"       DiffPepBuilder's preprocessing writes sibling files next to every\n"
              f"       input, and RFD3 reads complexes from that tree.", file=sys.stderr)
        return 2

    index = lnr / "test.txt"
    if not index.is_file():
        print(f"FATAL: missing LNR index {index}", file=sys.stderr)
        return 2

    manifest_path = out / "staged_manifest.json"

    if args.verify:
        if not manifest_path.is_file():
            print(f"FATAL: nothing to verify, no {manifest_path}", file=sys.stderr)
            return 2
        man = json.loads(manifest_path.read_text())
        changed, missing = [], []
        for row in man["staged"]:
            p = Path(row["path"])
            if not p.is_file():
                missing.append(row["id"])
            elif sha256(p) != row["sha256"]:
                changed.append(row["id"])
        extra = sorted(q.name for q in out.glob("*_processed.pdb"))
        print(f"[verify] staged files checked : {len(man['staged'])}")
        print(f"[verify] hash CHANGED         : {len(changed)} {changed[:8]}")
        print(f"[verify] missing              : {len(missing)} {missing[:8]}")
        print(f"[verify] upstream siblings    : {len(extra)} (*_processed.pdb, expected)")
        if changed or missing:
            print("[verify] FAILED: an input was rewritten or removed. The identity-chain-map\n"
                  "         guard did not hold; do not trust results from this run.",
                  file=sys.stderr)
            return 1
        print("[verify] PASSED: no staged input was modified.")
        return 0

    out.mkdir(parents=True, exist_ok=True)

    # Make a re-stage AUTHORITATIVE. process_batch_dock's membership is effectively the
    # directory listing, so a leftover complex from an earlier, larger stage would silently
    # enlarge the set -- and leftover *_processed.pdb siblings from a previous run would be
    # read as inputs in their own right. The processed/ cache goes too: it is keyed on these
    # files and would otherwise describe the old set.
    if not args.no_clean:
        stale = sorted(out.glob("*.pdb"))
        for q in stale:
            q.unlink()
        proc = out / "processed"
        if proc.is_dir():
            for q in sorted(proc.rglob("*")):
                if q.is_file():
                    q.unlink()
            for q in sorted((d for d in proc.rglob("*") if d.is_dir()), reverse=True):
                q.rmdir()
            proc.rmdir()
        if stale:
            print(f"[stage] cleaned {len(stale)} stale pdb(s) and the processed/ cache")

    rows = []
    for raw in index.read_text().splitlines():
        parts = raw.split()
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))
    if args.limit > 0:
        rows = rows[: args.limit]

    staged, skipped = [], []
    for pdb_id, rec_chain, pep_chain in rows:
        src = lnr / "pdbs" / f"{pdb_id}.pdb"
        if not src.is_file():
            skipped.append({"id": pdb_id, "reason": "pdb_absent"})
            continue
        pep_order, pep_res, pep_drops = read_chain(src, pep_chain)
        rec_order, rec_res, rec_drops = read_chain(src, rec_chain)
        # An incomplete RECEPTOR residue is dropped (it is pocket context, and upstream
        # cannot read it anyway). An incomplete PEPTIDE residue is different in kind: the
        # whole point of k=0 is that the generated sequence equals the input sequence, and
        # dropping a residue shortens the peptide and changes that sequence silently. So
        # reject the complex instead, and say so.
        if pep_drops["incomplete_backbone"]:
            skipped.append({"id": pdb_id, "reason": "peptide_residue_missing_backbone",
                            "atoms_dropped": pep_drops["incomplete_backbone"]})
            continue
        if len(pep_order) < args.min_peptide_len:
            skipped.append({"id": pdb_id, "reason": "peptide_too_short",
                            "n_res": len(pep_order)})
            continue
        if not rec_order:
            skipped.append({"id": pdb_id, "reason": "receptor_chain_empty",
                            "chain": rec_chain})
            continue
        dst = out / f"{pdb_id}.pdb"
        counts, renumber = write_complex(dst, (pep_order, pep_res), (rec_order, rec_res))
        staged.append({
            "id": pdb_id,
            "path": str(dst),
            "src": str(src),
            "src_receptor_chain": rec_chain,
            "src_peptide_chain": pep_chain,
            "peptide_seq": sequence_of(pep_order, pep_res),
            "peptide_len": len(pep_order),
            "receptor_len": len(rec_order),
            "atoms_peptide": counts[PEPTIDE_CHAIN],
            "atoms_receptor": counts[RECEPTOR_CHAIN],
            "renumbered_from": renumber,
            "dropped_peptide": pep_drops,
            "dropped_receptor": rec_drops,
            "sha256": sha256(dst),
        })

    # Upstream reads this keyed on the PDB basename. lig_chain is "A" for every entry, which
    # is the whole point: it makes the chain map an identity and the rewrite unreachable.
    info_path = out / "receptor_info.json"
    info_path.write_text(json.dumps({r["id"]: {"lig_chain": PEPTIDE_CHAIN}
                                     for r in staged}, indent=2) + "\n")

    manifest = {
        "lnr_dir": str(lnr),
        "out_dir": str(out),
        "peptide_chain": PEPTIDE_CHAIN,
        "receptor_chain": RECEPTOR_CHAIN,
        "n_index_rows": len(rows),
        "n_staged": len(staged),
        "n_skipped": len(skipped),
        "receptor_info": str(info_path),
        "receptor_info_sha256": sha256(info_path),
        "staged": staged,
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lens = sorted(r["peptide_len"] for r in staged)
    print(f"[stage] index rows      : {len(rows)}")
    print(f"[stage] staged          : {len(staged)} -> {out}")
    print(f"[stage] skipped         : {len(skipped)} {skipped[:5]}")
    if lens:
        print(f"[stage] peptide lengths : min {lens[0]}  median "
              f"{lens[len(lens) // 2]}  max {lens[-1]}")
    def _tot(field):
        return sum(r["dropped_peptide"][field] + r["dropped_receptor"][field]
                   for r in staged)
    print(f"[stage] dropped atoms   : nonstandard {_tot('nonstandard')}, "
          f"altloc {_tot('altloc')}, incomplete_backbone {_tot('incomplete_backbone')}")
    print(f"[stage] receptor_info   : {info_path}")
    print(f"[stage] manifest        : {manifest_path}")
    print("[stage] after preprocessing, prove nothing was rewritten with:")
    print(f"[stage]   python script_utils/lpgen_dpb_stage.py --out-dir {out} --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
