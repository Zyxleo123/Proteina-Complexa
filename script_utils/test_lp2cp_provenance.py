"""Integration test for the LP/CP provenance pairing, with no network and no AFDB.

The pairing's real risk is not the provenance regex -- `selftest` covers that -- it is the
geometry: matching receptor residues by id, EXCLUDING the excised peptide range from the
anchor set, applying the fitted transform to the parent slice, and writing a two-chain
complex back out. None of that is exercised by a parser test.

So this synthesises a parent from a real CPSea complex: merge the receptor and the binder
into one chain (which is what the parent looked like before excision), apply a known rigid
transform, and hand it to `build_pair` as if it had been downloaded. The pairing must then
undo that transform exactly, which makes every quantity predictable:

    receptor_rmsd_A  ~ 0      the transform is exactly invertible
    n_mut            = 0      same residues on both sides
    lp_term_ca_A     ~ cp_term_ca_A   the LP slice is the CP binder, rigidly moved back

Then the same parent with ONLY its peptide displaced. The receptor fit must stay ~0, which
is the test that the peptide range is genuinely excluded from the anchors -- if it were
included, a rigid test would pass anyway and the bug would only show up as quietly
inflated receptor RMSDs on real data.

Run:  .venv/bin/python script_utils/test_lp2cp_provenance.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lp2cp_provenance import (  # noqa: E402
    BINDER_CHAIN,
    RECEPTOR_CHAIN,
    build_pair,
    parse_example_id,
)

# A row whose provenance, chains and residue numbering were all read off disk by hand.
CP_PDB = Path(
    "/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea/preprocessed/processed/train/"
    "AF-A0A009E759-F1_0_89_101_relaxed_relaxed.pdb"
)
EXAMPLE_ID = "AF-A0A009E759-F1_0_89_101_relaxed_relaxed"


def _rigid(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A rotation well away from identity, plus a large translation."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:  # keep it a rotation, not a reflection
        q[:, 0] *= -1
    return q, rng.normal(scale=25.0, size=3)


def _synth_parent(cp_path: Path, dest: Path, displace_peptide_by: float = 0.0) -> None:
    """Write a single-chain 'parent' = receptor + binder, rigidly moved."""
    from biotite.structure.io.pdb import PDBFile

    cp = PDBFile.read(str(cp_path)).get_structure(model=1)
    merged = cp.copy()
    if displace_peptide_by:
        mask = merged.chain_id == BINDER_CHAIN
        merged.coord[mask] += np.array([displace_peptide_by, 0.0, 0.0])
    merged.chain_id[:] = RECEPTOR_CHAIN  # one chain, as an AFDB model has

    rot, trans = _rigid()
    merged.coord = merged.coord @ rot.T + trans

    pdb = PDBFile()
    pdb.set_structure(merged)
    pdb.write(str(dest))


def main() -> int:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    if not CP_PDB.exists():
        print(f"SKIP: CPSea structure not on this filesystem: {CP_PDB}")
        print("      (the parser tests live in `lp2cp_provenance.py selftest`)")
        return 0

    prov = parse_example_id(EXAMPLE_ID)
    assert prov is not None
    lo, hi = prov.res_start, prov.res_end

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        models = tmp / "models"
        models.mkdir()
        parent = models / f"{prov.accession}-model_v4.pdb"
        _synth_parent(CP_PDB, parent)

        row = {
            "example_id": EXAMPLE_ID,
            "accession": prov.accession,
            "res_start": lo,
            "res_end": hi,
            "cyclization_type": "other",
            "cluster_id": "test",
            "path": str(CP_PDB),
            "version": "v4",
        }

        rec = build_pair(row, models, tmp / "lp_pdb", min_anchor=10)
        print("--- rigid parent (must reproduce the CP binder exactly) ---")
        for k in (
            "skip_reason",
            "n_receptor_common",
            "receptor_rmsd_A",
            "n_mut",
            "lp_term_ca_A",
            "cp_term_ca_A",
            "lp_cp_ca_rmsd_A",
            "lp_cp_ca_max_A",
        ):
            print(f"  {k:<20s} {rec[k]}")

        check(rec["skip_reason"] is None, f"unexpected skip: {rec['skip_reason']}")
        check(rec["n_receptor_common"] == 70, f"expected 70 anchors, got {rec['n_receptor_common']}")
        check(
            rec["receptor_rmsd_A"] is not None and rec["receptor_rmsd_A"] < 1e-3,
            f"receptor fit should be exact, got {rec['receptor_rmsd_A']}",
        )
        check(rec["n_mut"] == 0, f"same residues both sides, got n_mut={rec['n_mut']}")
        check(rec["lp_seq"] == rec["cp_seq"], "sequences should match on a rigid parent")
        check(len(rec["lp_seq"] or "") == hi - lo + 1, f"lp_seq length != {hi - lo + 1}")
        check(
            rec["lp_term_ca_A"] is not None
            and abs(rec["lp_term_ca_A"] - rec["cp_term_ca_A"]) < 1e-3,
            f"terminal gaps should agree: {rec['lp_term_ca_A']} vs {rec['cp_term_ca_A']}",
        )
        check(
            rec["lp_cp_ca_rmsd_A"] is not None and rec["lp_cp_ca_rmsd_A"] < 1e-2,
            f"a rigid parent should need no transport, got {rec['lp_cp_ca_rmsd_A']} A",
        )

        # The written complex must be re-readable and carry both chains, or nothing
        # downstream can load it.
        from biotite.structure.io.pdb import PDBFile

        out = PDBFile.read(rec["lp_pdb"]).get_structure(model=1)
        chains = set(np.unique(out.chain_id).tolist())
        check(chains == {RECEPTOR_CHAIN, BINDER_CHAIN}, f"written chains = {chains}")
        n_bind = len({int(r) for r in out[out.chain_id == BINDER_CHAIN].res_id})
        check(n_bind == hi - lo + 1, f"written binder has {n_bind} residues")
        # And the LP peptide must land in the CP receptor's frame, not the parent's.
        cp = PDBFile.read(str(CP_PDB)).get_structure(model=1)
        lp_bind = out[out.chain_id == BINDER_CHAIN]
        cp_bind = cp[cp.chain_id == BINDER_CHAIN]
        d = float(
            np.linalg.norm(lp_bind.coord.mean(axis=0) - cp_bind.coord.mean(axis=0))
        )
        check(d < 1e-2, f"LP peptide is {d:.3f} A from the CP peptide; frame not restored")

        # --- peptide-only displacement: the receptor fit must be unaffected -----------
        parent2 = models / f"{prov.accession}X-model_v4.pdb"
        _synth_parent(CP_PDB, parent2, displace_peptide_by=8.0)
        row2 = dict(row, accession=f"{prov.accession}X", example_id=EXAMPLE_ID + "X")
        rec2 = build_pair(row2, models, tmp / "lp_pdb", min_anchor=10)
        print("--- peptide displaced 8 A (receptor fit must stay exact) ---")
        print(f"  receptor_rmsd_A      {rec2['receptor_rmsd_A']}")
        print(f"  lp_term_ca_A         {rec2['lp_term_ca_A']}")
        print(f"  lp_cp_ca_rmsd_A      {rec2['lp_cp_ca_rmsd_A']}")
        check(
            rec2["receptor_rmsd_A"] is not None and rec2["receptor_rmsd_A"] < 1e-3,
            "peptide displacement leaked into the receptor fit -- the excised range is "
            f"NOT excluded from the anchors (rmsd={rec2['receptor_rmsd_A']})",
        )
        # The displacement measure must read the size we injected. A measure that merely
        # "looks nonzero" would pass while being scaled wrong, which is how an amplitude
        # floor gets checked against the wrong units.
        check(
            rec2["lp_cp_ca_rmsd_A"] is not None
            and abs(rec2["lp_cp_ca_rmsd_A"] - 8.0) < 1e-2,
            f"injected 8 A of peptide displacement, measured {rec2['lp_cp_ca_rmsd_A']} A",
        )

        # --- a missing parent must be recorded, not raised --------------------------
        rec3 = build_pair(
            dict(row, accession="AF-NOSUCH-F1"), models, tmp / "lp_pdb", min_anchor=10
        )
        check(rec3["skip_reason"] == "parent_not_fetched", f"got {rec3['skip_reason']}")
        check(rec3["example_id"] == EXAMPLE_ID, "skipped record lost its identity")

        # --- an impossible anchor requirement must skip, not crash ------------------
        rec4 = build_pair(row, models, tmp / "lp_pdb", min_anchor=10_000)
        check(
            str(rec4["skip_reason"]).startswith("too_few_receptor_anchors"),
            f"got {rec4['skip_reason']}",
        )

    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nOK: alignment, peptide-range exclusion, sequence delta, writer, skip paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
