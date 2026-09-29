"""Shape and scoring tests for the FlowEdit driver, on REAL batches. No GPU, no checkpoint.

These exist because a shape bug cost a full GPU round trip: `receptor_arrays` stripped the
batch dim only when it was 1, so at B > 1 a 3-D `target_mask` reached `np.nonzero`, which
returned three index arrays instead of two, and every one of 18 candidates died with
"too many values to unpack (expected 2)". Nothing about that needed a GPU to find, so it is
tested here.

The other test is the invariant worth having: scoring a peptide against ITSELF must come out
perfect -- retention 1.0, 0 substitutions, 0 A RMSD. If the contract cannot recognise an
identical structure, no number it produces about a real edit means anything.

Run:  PYTHONPATH=src .venv/bin/python script_utils/test_flowedit_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from proteinfoundation.datasets.structure_data import structure_collate_fn  # noqa: E402
from proteinfoundation.editing import contract as C  # noqa: E402
from proteinfoundation.eval.cyclic_reconstruction_metrics import NM_TO_ANG  # noqa: E402

DATA = Path("/zfsauton2/home/yixiz/Proteina-Complexa/CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet")
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from eval_ae_roundtrip import build_dataset
    from flowedit_cyclize import CA_IDX, binder_endpoints, receptor_arrays

    ds, _ = build_dataset("example/training_cpsea_peptide_smoke", str(DATA), keep_rotation=False)
    sample = ds[0]

    for B in (1, 2, 4):
        print(f"\nreceptor_arrays at B={B}")
        batch = structure_collate_fn([sample] * B)
        xyz, aa, m = receptor_arrays(batch)
        check(f"B={B}: receptor xyz is [Lt, 37, 3]", xyz.dim() == 3, str(tuple(xyz.shape)))
        check(f"B={B}: receptor aatype is [Lt]", aa.dim() == 1, str(tuple(aa.shape)))
        check(f"B={B}: receptor atom mask is [Lt, 37]", m is not None and m.dim() == 2,
              "None" if m is None else str(tuple(m.shape)))
        check(f"B={B}: lengths agree", xyz.shape[0] == aa.shape[0] == m.shape[0],
              f"{xyz.shape[0]}/{aa.shape[0]}/{m.shape[0]}")

    print("\nnonzero unpacking on the real receptor mask")
    batch = structure_collate_fn([sample] * 2)
    xyz, aa, m = receptor_arrays(batch)
    import numpy as np
    try:
        idx, slot = np.nonzero(m.cpu().numpy().astype(bool))
        check("np.nonzero on the receptor mask unpacks into 2", True, f"{len(idx)} atoms")
    except ValueError as exc:
        check("np.nonzero on the receptor mask unpacks into 2", False, str(exc))

    print("\nscoring a peptide against itself must be perfect")
    mask = (batch["mask"].bool() if "mask" in batch
            else batch["coord_mask"][..., CA_IDX].bool())
    keep = mask[0].cpu().numpy().astype(bool)
    in_xyz = batch["coords_nm"][0] * NM_TO_ANG
    in_aa = batch["residue_type"][0]
    in_am = batch["coord_mask"][0].bool()
    i, j = binder_endpoints(mask[0])

    row = C.evaluate(
        before_xyz_A=in_xyz[keep], before_aatype=in_aa[keep],
        after_xyz_A=in_xyz[keep], after_aatype=in_aa[keep],
        tgt_xyz_A=xyz, tgt_aatype=aa,
        cyc_type="mainchain", anchor_i=i, anchor_j=j, ddg_REU=None,
        before_atom_mask=in_am[keep], after_atom_mask=in_am[keep],
        tgt_atom_mask=m,
    )
    check("identity: retention is 1.0", row["contact_retention"] == 1.0,
          str(row["contact_retention"]))
    check("identity: 0 substitutions", row["n_substitutions"] == 0)
    check("identity: CA RMSD is 0", row["ca_rmsd_A"] == 0.0, str(row["ca_rmsd_A"]))
    check("identity: interface CA RMSD is 0", row["interface_ca_rmsd_A"] == 0.0)
    check("the input has real contacts to retain", row["n_contacts_before"] > 0,
          f"{row['n_contacts_before']} pairs, {row['n_interface_residues']} iface residues")
    # The native LINEAR peptide must NOT be reported as a closed ring -- it has no bond.
    check("a linear native is not a closed ring", row["pass_ring"] is False,
          f"N-C = {row['bond_dist_A']:.2f} A")
    check("no Rosetta yet => not accepted", row["accepted"] is False)

    # ---- the clash threshold, calibrated against the natives themselves ----------
    # A gate that rejects real crystal complexes is measuring the instrument, so the
    # threshold is asserted against them rather than chosen for looking strict. Measured on
    # all 60: 1.4 A passes 60/60, 2.0 A passes 59/60, 2.2 A passes only 54/60, and the median
    # native's closest non-bonded heavy-atom pair sits at 2.64 A. 2.0 A is therefore well
    # below a normal native contact while still catching a steric impossibility.
    #
    # The one native that fails at 2.0 A is LNR_1bjr_E_I, whose closest peptide-receptor pair
    # is 1.45 A -- a genuine crystal close contact, not a bug here. It is ds[0], which is why
    # a single-example check on ds[0] would report the gate as broken.
    print("\nclash threshold against natives (subset)")
    n_ex = min(20, len(ds))
    passed, failed_ids = 0, []
    for k in range(n_ex):
        s_k = ds[k]
        if s_k is None:
            continue
        bk = structure_collate_fn([s_k])
        mk = (bk["mask"].bool() if "mask" in bk else bk["coord_mask"][..., CA_IDX].bool())
        kk = mk[0].cpu().numpy().astype(bool)
        xk = bk["coords_nm"][0] * NM_TO_ANG
        ak = bk["residue_type"][0]
        amk = bk["coord_mask"][0].bool()
        tx, ta, tmk = receptor_arrays(bk)
        ik, jk = binder_endpoints(mk[0])
        cl = C.severe_clashes(xk[kk], ak[kk], tx, ta, ring=(ik, jk),
                              pep_atom_mask=amk[kk], tgt_atom_mask=tmk)
        if cl["severe_clashes"] <= C.CONTRACT["max_severe_clashes"]:
            passed += 1
        else:
            failed_ids.append(str(ds.metadata.iloc[k]["example_id"]))
    check(f"at 2.0 A, >= {n_ex - 1}/{n_ex} natives pass the clash gate", passed >= n_ex - 1,
          f"{passed}/{n_ex} passed; rejected {failed_ids}")
    check("the only rejected native is the known 1.45 A crystal contact",
          set(failed_ids) <= {"LNR_1bjr_E_I"}, str(failed_ids))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("all shape/scoring checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
