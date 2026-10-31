"""Pocket sequences for the LLM-pilot complexes, in the format the MLM was trained on.

Uses `cpseq_build_mlm_corpus.extract_pocket` rather than re-deriving the contact rule, so
the radius (8.0 A heavy-atom), the chain roles and the `/` gap rule are the ones the
checkpoint saw. The native complex is the source, not a Boltz-2 fold: the pocket is a
property of the target and must be identical for every candidate of that parent.

ONE DEPARTURE FROM A BARE `build_row` CALL, AND IT IS NOT OPTIONAL. Training ran on
preprocessed CPSea PDBs, which carry no waters and no ligands. These are raw LNR crystal
structures, and `parse_pdb_residues` reads HETATM as well as ATOM, so chain B of 1elr
comes back as `?MEEVD?????????????` -- the 5-mer plus thirteen waters. Left alone, those
waters are part of the "binder" the 8 A cutoff is measured against, and the pocket grows
to include whatever they touch: 50 residues rather than the 46 the target manifest
records. So both chains are restricted to canonical amino acids, and the binder is
further restricted to the contiguous run that spells the reference sequence.
"""

import csv
import json
import sys

sys.path.insert(0, "/home/yixiz/Proteina-Complexa/script_utils")

from cpseq_build_mlm_corpus import _THREE_TO_ONE, extract_pocket, parse_pdb_residues

PARENTS_CSV = "/data/user_data/yixiz/cpseq/tranche1/manifest/parents.csv"
PILOT = ("lnr_1elr", "lnr_2v8x", "lnr_6seo")
RADIUS = 8.0


def canonical(residues):
    return [r for r in residues if r["resname"] in _THREE_TO_ONE]


def locate(residues, reference):
    """The contiguous run of `residues` spelling `reference`, or None.

    Contiguous in the residue list, not in `resseq`: a crystal may number the peptide
    with a gap, and the reference sequence is what the search mutates.
    """
    letters = "".join(_THREE_TO_ONE[r["resname"]] for r in residues)
    i = letters.find(reference)
    if i < 0:
        return None
    return residues[i : i + len(reference)]


rows = {r["parent_id"]: r for r in csv.DictReader(open(PARENTS_CSV))}
out = {}
ok = True
for pid in PILOT:
    r = rows[pid]
    ref = r["reference_sequence"]
    chains = parse_pdb_residues(r["structure_path"])
    binder_chain = r["peptide_chain"]

    binder_all = canonical(chains[binder_chain])
    binder = locate(binder_all, ref)
    receptor = [
        res
        for cid, rs in chains.items()
        if cid != binder_chain
        for res in canonical(rs)
    ]
    if binder is None:
        print(f"{pid}: FATAL reference {ref} not found in chain {binder_chain}")
        ok = False
        continue

    pocket_seq, resids = extract_pocket(receptor, binder, RADIUS)
    n_frag = pocket_seq.count("/") + 1 if pocket_seq else 0
    out[pid] = {
        "parent_id": pid,
        "pdb_id": r["pdb_id"],
        "structure_path": r["structure_path"],
        "binder_chain": binder_chain,
        "receptor_chains": r["receptor_chains"],
        "reference_sequence": ref,
        "peptide_length": len(ref),
        "budget": len(ref) // 2,
        "cyc_type": 0,  # mainchain: the search cyclises head-to-tail
        "pocket_seq": pocket_seq,
        "n_pocket_res": len(resids),
        "n_fragments": n_frag,
        "pocket_resids": resids,
        "pocket_radius_A": RADIUS,
        "manifest_n_pocket_residues": int(r["n_pocket_residues"]),
    }
    print(f"{pid}  L={len(ref):2d} budget={len(ref) // 2}  pep={ref}")
    print(
        f"    pocket {len(resids)} res in {n_frag} frags "
        f"(target manifest records {r['n_pocket_residues']})"
    )
    print(f"    {pocket_seq}")

if not ok:
    sys.exit(1)
dest = sys.argv[1]
with open(dest, "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote", dest)
