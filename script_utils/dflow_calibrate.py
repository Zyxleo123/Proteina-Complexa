"""Does the acceptance contract pass the NATIVE crystal structures it is applied to?

Why this runs before the experiment, not after
---------------------------------------------
The binder evaluation on this project once scored all ten runs at 0.0 because it applied
miniprotein thresholds to macrocycles; the pass rate looked like a model result and was an
instrument artifact. The cheapest guard against repeating that is to feed the instrument the
answer and check it agrees: score each native LP peptide against ITSELF.

A native scored against itself must come out at identity 1.0, zero substitutions, contact
retention 1.0 and CA RMSD 0.0 -- those are arithmetic and a failure there is a code bug. The
two that are NOT arithmetic, and are the reason this exists:

    gate_clash        `clash_min_A` is a judgement call. Real crystal structures carry short
                      heavy-atom contacts (a strong hydrogen bond is 2.7-2.9 A, a salt bridge
                      shorter), so a threshold set for "severe overlap" can reject natives.
                      If natives fail it, the threshold is wrong, not the natives.
    n_interface_res   the contract has separate whole-peptide (2 A) and interface (1 A) RMSD
                      clauses. For a 10-mer buried in a pocket those two sets may be the same
                      residues, in which case the 1 A clause simply supersedes the 2 A one and
                      the report should say so rather than presenting them as independent.

The ring gate is EXPECTED to fail here: these inputs are linear peptides, which is the whole
premise. That failure is the positive control -- it says the ring clause is actually being
evaluated and is not passing vacuously.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "script_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dflow_accept import accept  # noqa: E402
from dflow_losses import CONTRACT, NM_TO_ANG  # noqa: E402
from proteinfoundation.cyclization.constants import NAME_TO_CYCLIZATION_TYPE  # noqa: E402
from proteinfoundation.utils.pdb_utils import from_pdb_string  # noqa: E402


def _parse(path, chain):
    prot = from_pdb_string(Path(path).read_text(), chain_id=chain)
    return (
        torch.as_tensor(prot.atom_positions, dtype=torch.float32) / NM_TO_ANG,
        torch.as_tensor(prot.atom_mask, dtype=torch.float32).bool(),
        torch.as_tensor(prot.aatype, dtype=torch.long),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cyc-type", default="disulfide",
                    help="Which ring to REQUEST of a native linear peptide. Only affects the "
                         "ring gate, which is expected to fail either way.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_parquet(args.metadata)
    if args.limit:
        meta = meta.head(args.limit)
    cyc_type = NAME_TO_CYCLIZATION_TYPE[args.cyc_type]

    rows = []
    for _, r in meta.iterrows():
        pdb = str(r["path"])
        bchain = str(r.get("binder_chain_id", "B"))
        try:
            pep, pep_m, pep_aa = _parse(pdb, bchain)
            chains = sorted({
                ln[21] for ln in Path(pdb).read_text().splitlines()
                if ln.startswith(("ATOM", "HETATM")) and ln[21] != bchain
            })
            parts = [_parse(pdb, c) for c in chains]
            tgt = torch.cat([p[0] for p in parts], dim=0)
            tgt_m = torch.cat([p[1] for p in parts], dim=0)
            out = accept(
                pep_coors_nm=pep, pep_amask=pep_m, pep_aatype=pep_aa,
                ref_coors_nm=pep.clone(), ref_amask=pep_m, ref_aatype=pep_aa.clone(),
                tgt_coors_nm=tgt, tgt_amask=tgt_m, cyc_type=cyc_type,
            )
            out["example_id"] = str(r["example_id"])
            out["status"] = "ok"
            out["n_target_chains"] = len(chains)
        except Exception as exc:  # noqa: BLE001
            out = {"example_id": str(r["example_id"]), "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(rows)} natives scored\n")
    if not ok:
        raise SystemExit("FATAL: no native could be scored -- the instrument is broken.")

    print("--- arithmetic identities (a failure here is a code bug) ---")
    for k, want in (("seq_identity", 1.0), ("n_substitutions", 0), ("contact_retention", 1.0)):
        bad = [r["example_id"] for r in ok if r.get(k) != want]
        print(f"  {k} == {want}: {len(ok) - len(bad)}/{len(ok)}"
              + (f"  VIOLATIONS: {bad[:5]}" if bad else ""))
    bad = [r["example_id"] for r in ok if r["ca_rmsd_A"] > 1e-4]
    print(f"  ca_rmsd_A == 0: {len(ok) - len(bad)}/{len(ok)}"
          + (f"  VIOLATIONS: {bad[:5]}" if bad else ""))

    print("\n--- the judgement calls ---")
    n_clash = sum(1 for r in ok if not r["gate_clash"])
    mins = sorted(r["min_heavy_dist_A"] for r in ok)
    imins = sorted(r["min_intra_heavy_dist_A"] for r in ok)
    print(f"  gate_clash passes: {len(ok) - n_clash}/{len(ok)} at clash_min_A="
          f"{CONTRACT['clash_min_A']}")
    print(f"  native min heavy dist to receptor (A): min={mins[0]:.2f} "
          f"p05={mins[max(0, len(mins) // 20)]:.2f} median={st.median(mins):.2f}")
    print(f"  native min intra-peptide heavy dist (A): min={imins[0]:.2f} "
          f"median={st.median(imins):.2f}")
    if n_clash:
        print(f"  ==> {n_clash} NATIVE structures fail the clash gate. The threshold is set "
              "for the wrong thing; it must be loosened to below the native floor before any "
              "arm is judged by it.")
        for thr in (2.6, 2.5, 2.4, 2.3, 2.2, 2.0):
            n = sum(1 for r in ok if r["min_heavy_dist_A"] >= thr
                    and r["min_intra_heavy_dist_A"] >= thr)
            print(f"      clash_min_A={thr}: {n}/{len(ok)} natives would pass")

    frac = [r["n_interface_residues"] / r["peptide_length"] for r in ok]
    print(f"\n  interface residues / peptide length: median={st.median(frac):.2f} "
          f"min={min(frac):.2f} max={max(frac):.2f}")
    same = sum(1 for r in ok if r["n_interface_residues"] == r["peptide_length"])
    print(f"  peptides whose interface IS the whole peptide: {same}/{len(ok)}")
    if same > len(ok) // 2:
        print("  ==> for most of these peptides the interface-RMSD clause (1 A) simply "
              "supersedes the whole-peptide clause (2 A); they are not independent tests and "
              "the report must not present them as two.")

    print("\n--- positive control ---")
    n_ring = sum(1 for r in ok if r["gate_ring_closed"])
    print(f"  gate_ring_closed on LINEAR natives: {n_ring}/{len(ok)} (expected ~0; a high "
          "number would mean the ring clause passes vacuously)")
    n_joint = sum(1 for r in ok if r["gate_joint_structural"])
    print(f"  gate_joint_structural: {n_joint}/{len(ok)} (expected 0: they have no ring)")


if __name__ == "__main__":
    main()
