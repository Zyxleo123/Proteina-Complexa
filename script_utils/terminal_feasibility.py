"""A provable lower bound on the pose distortion terminal cyclization costs.

The point
---------
`README_GENKIC_LP2CP.md` reports an r=0.903 correlation between an LP's terminal gap and the
best interface RMSD any closure achieved. A correlation over a finite search cannot distinguish
"we did not search hard enough" from "this cannot be done". This module settles that with
arithmetic instead.

To form a terminal bond, the two anchor atoms must end up within the chemistry's bonding
distance. Write `D` for their current separation and `reach` for the largest separation at which
the bond can exist. If the anchor residue at each end moves by `d_0` and `d_last`, the triangle
inequality gives

    d_0 + d_last  >=  D - reach

and since CA RMSD over the peptide is `sqrt( (1/L) * sum_i d_i^2 )`, dropping every non-terminal
term and minimising `d_0^2 + d_last^2` subject to the sum (equal split) gives

    RMSD  >=  (D - reach) / sqrt(2 L)                                      (whole peptide)
    iRMSD >=  (D - reach) / sqrt(2 * n_interface)     if BOTH termini are interface residues

The second bound is the sharper one when it applies, because `n_interface <= L` and the interface
bar is half the whole-peptide bar. When a terminus makes no receptor contact it does not enter
the interface sum at all, and that bound simply does not constrain it -- which is why the two are
reported separately rather than combined.

Both are LOWER bounds on a quantity the contract caps, so a target whose bound exceeds the cap is
infeasible by geometry, for every conceivable search budget. That is a different and much stronger
statement than "the search did not find one", and it is the honest way to report the 37 targets the
terminal route cannot reach.

What this does NOT prove
------------------------
The bound is necessary, not sufficient: clearing it does not mean a closure exists, only that
none is excluded by this argument. Bond angles, sterics, rama and the receptor all impose more.
So "feasible" here means "not provably impossible", and the deep search is what decides those.

Pure geometry, read straight from the staged PDBs -- no Rosetta, no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Largest anchor-atom separation at which each chemistry's bond can exist, and which atoms the
# separation is measured between. Mainchain bonds backbone N to backbone C, so its reach is the
# amide bond itself; the side-chain chemistries bond through side chains, so the backbone-frame
# question is how far apart the CB atoms may be.
CHEMISTRY_REACH = {
    # name: (atom on residue i, atom on residue j, reach in Angstrom)
    "mainchain": ("N", "C", 1.33),
    "disulfide": ("CB", "CB", 5.0),
    "isopeptide": ("CB", "CB", 9.5),
}

HEAVY_CONTACT_CUTOFF_A = 4.5
MAX_CA_RMSD_A = 2.0
MAX_INTERFACE_CA_RMSD_A = 1.0


def read_chain_atoms(pdb: str) -> dict[str, list[dict]]:
    """{chain: [ {reskey, resn, atoms: {name: (x,y,z)}} ]} in file order, heavy atoms only."""
    chains: dict[str, list[dict]] = {}
    index: dict[tuple[str, str], dict] = {}
    for line in open(pdb):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper()
        name = line[12:16].strip()
        if element == "H" or (not element and name.startswith("H")):
            continue
        ch = line[21]
        key = line[22:27]
        rec = index.get((ch, key))
        if rec is None:
            rec = {"reskey": key, "resn": line[17:20].strip(), "atoms": {}}
            index[(ch, key)] = rec
            chains.setdefault(ch, []).append(rec)
        rec["atoms"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return chains


def interface_locals(binder: list[dict], receptor: list[dict],
                     cutoff: float = HEAVY_CONTACT_CUTOFF_A) -> set[int]:
    """Binder-local indices with any heavy atom within `cutoff` of any receptor heavy atom."""
    # Coarse screen on a per-residue centroid keeps this linear enough on big receptors.
    def centroid(r):
        pts = list(r["atoms"].values())
        return tuple(sum(p[k] for p in pts) / len(pts) for k in range(3)) if pts else (0, 0, 0)

    b_cent = [centroid(r) for r in binder]
    near = []
    for r in receptor:
        c = centroid(r)
        if any(math.dist(c, bc) <= cutoff + 16.0 for bc in b_cent):
            near.append(r)
    out: set[int] = set()
    for li, br in enumerate(binder):
        bp = list(br["atoms"].values())
        hit = False
        for rr in near:
            for a in bp:
                for c in rr["atoms"].values():
                    if math.dist(a, c) <= cutoff:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            out.add(li)
    return out


def anchor_separation(binder: list[dict], atom_i: str, atom_j: str) -> float | None:
    """Distance between `atom_i` on the first residue and `atom_j` on the last."""
    first, last = binder[0]["atoms"], binder[-1]["atoms"]
    pi = first.get(atom_i) or first.get("CA")
    pj = last.get(atom_j) or last.get("CA")
    if pi is None or pj is None:
        return None
    return math.dist(pi, pj)


def analyse(example_id: str, pdb: str, binder_chain: str) -> dict:
    chains = read_chain_atoms(pdb)
    if binder_chain not in chains:
        return {"example_id": example_id, "status": "no_binder_chain",
                "chains_present": sorted(chains)}
    binder = chains[binder_chain]
    receptor = [r for ch, rs in chains.items() if ch != binder_chain for r in rs]
    if not receptor:
        return {"example_id": example_id, "status": "no_receptor"}

    L = len(binder)
    iface = interface_locals(binder, receptor)
    termini_at_interface = (0 in iface) and (L - 1 in iface)

    row = {
        "example_id": example_id, "status": "ok", "peptide_length": L,
        "n_interface_residues": len(iface),
        "terminus_0_is_interface": 0 in iface,
        "terminus_last_is_interface": (L - 1) in iface,
        "both_termini_at_interface": termini_at_interface,
        "sequence_resn": [r["resn"] for r in binder],
    }

    best = None
    for chem, (ai, aj, reach) in CHEMISTRY_REACH.items():
        D = anchor_separation(binder, ai, aj)
        if D is None:
            row[f"{chem}_status"] = "anchor_atom_missing"
            continue
        travel = max(0.0, D - reach)
        rmsd_min = travel / math.sqrt(2.0 * L)
        # The interface bound only bites when BOTH termini are interface residues; otherwise a
        # terminus's displacement never enters the interface sum.
        irmsd_min = (travel / math.sqrt(2.0 * len(iface))
                     if termini_at_interface and iface else 0.0)
        feasible = (rmsd_min <= MAX_CA_RMSD_A) and (irmsd_min <= MAX_INTERFACE_CA_RMSD_A)
        row.update({
            f"{chem}_anchor_sep_A": round(D, 2),
            f"{chem}_required_travel_A": round(travel, 2),
            f"{chem}_rmsd_lower_bound_A": round(rmsd_min, 3),
            f"{chem}_irmsd_lower_bound_A": round(irmsd_min, 3),
            f"{chem}_not_excluded": bool(feasible),
            # How much slack (or deficit) against the binding bar, in Angstrom of RMSD.
            f"{chem}_slack_A": round(min(MAX_CA_RMSD_A - rmsd_min,
                                         MAX_INTERFACE_CA_RMSD_A - irmsd_min), 3),
        })
        if feasible and (best is None or rmsd_min < row[f"{best}_rmsd_lower_bound_A"]):
            best = chem
    row["best_chemistry_not_excluded"] = best
    row["any_chemistry_not_excluded"] = best is not None
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True, help="JSONL, one row per target.")
    ap.add_argument("--examples", nargs="+", default=None)
    args = ap.parse_args()

    import pandas as pd

    meta = pd.read_parquet(args.metadata)
    if args.examples:
        meta = meta[meta["example_id"].astype(str).isin(set(args.examples))]

    rows = []
    for _, m in meta.iterrows():
        rows.append(analyse(str(m["example_id"]), str(m["path"]), str(m["binder_chain_id"])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: r.get("mainchain_anchor_sep_A", 1e9))
    print(f"{len(ok)} targets analysed\n")
    print(f"{'target':22s} {'L':>3s} {'ifc':>4s} {'trm':>4s} "
          f"{'mainchain':>18s} {'disulfide':>18s} {'isopeptide':>18s}  verdict")
    print(f"{'':22s} {'':>3s} {'':>4s} {'':>4s} "
          f"{'sep -> minRMSD':>18s} {'sep -> minRMSD':>18s} {'sep -> minRMSD':>18s}")
    for r in ok:
        cells = []
        for chem in ("mainchain", "disulfide", "isopeptide"):
            sep = r.get(f"{chem}_anchor_sep_A")
            lb = r.get(f"{chem}_rmsd_lower_bound_A")
            if sep is None:
                cells.append(f"{'--':>18s}")
            else:
                mark = "ok" if r.get(f"{chem}_not_excluded") else "XX"
                cells.append(f"{sep:6.1f} -> {lb:5.2f} {mark:>2s}")
        verdict = (f"REACHABLE via {r['best_chemistry_not_excluded']}"
                   if r["any_chemistry_not_excluded"] else "PROVABLY INFEASIBLE")
        trm = ("both" if r["both_termini_at_interface"]
               else ("one" if r["terminus_0_is_interface"] or r["terminus_last_is_interface"]
                     else "none"))
        print(f"{r['example_id']:22s} {r['peptide_length']:3d} {r['n_interface_residues']:4d} "
              f"{trm:>4s} " + " ".join(cells) + f"  {verdict}")

    n_ok = sum(1 for r in ok if r["any_chemistry_not_excluded"])
    print(f"\nnot excluded by the bound: {n_ok}/{len(ok)}")
    print(f"provably infeasible       : {len(ok) - n_ok}/{len(ok)}")
    print("\nThe bound is NECESSARY, not sufficient: 'not excluded' means no closure is ruled")
    print("out by this argument alone, not that one exists. Sterics, angles, rama and the")
    print("receptor all add further constraints, which only a search can settle.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
