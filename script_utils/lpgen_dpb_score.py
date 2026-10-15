#!/usr/bin/env python
"""Score DiffPepDock poses against the staged natives: the k=0 reference baseline.

Writes one JSONL row per pose plus a JSON summary, so the figures and tables are
re-derivable without re-running the GPU job.

What is measured, and why each one
----------------------------------
* **sequence identity to the staged input.** At k=0 this must be exactly 1.0 for every
  pose -- `mask_lig_seq=False` makes the writer emit `gt_aatype`. It is computed rather
  than trusted because it is the contract the whole similarity axis is defined against: if
  it drifts below 1.0 at k=0, every later "k substitutions" number is measured from an
  unknown baseline.
* **raw CA RMSD** (no superposition). This is the number that matters for docking: it
  answers "is the peptide in the right place in the pocket", and it is the one that would
  collapse to ~0 if the run were echoing the crystal pose instead of re-docking.
* **superposed (Kabsch) CA RMSD.** Conformation only, placement factored out. Raw minus
  superposed separates "wrong pose" from "wrong fold".
* **interface CA RMSD**, raw, over only those peptide residues contacting the receptor in
  the NATIVE structure. Whole-peptide RMSD is dominated by flexible termini that carry no
  binding signal; this project has already measured that interface RMSD supersedes
  whole-peptide RMSD on half of a comparable set.

Per-target aggregation is **best-of-N**, not mean: the downstream procedure generates
several candidates and selects, so the mean over samples describes a procedure nobody runs.
Both are reported.

This is pose accuracy only. It is NOT the success criterion -- that is Rosetta dG through
the shared evaluator, which needs side chains packed first, since this model exports
backbone + CB only (it predicts frames plus 1 psi and 2 chi).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="DiffPepDock eval_dir")
    p.add_argument("--stage-dir", required=True, help="staging dir holding staged_manifest.json")
    p.add_argument("--out-dir", required=True, help="where to write rows + summary")
    p.add_argument("--contact-cutoff", type=float, default=5.0,
                   help="native heavy-atom distance defining an interface peptide residue")
    p.add_argument("--success-a", type=float, default=2.0,
                   help="raw interface CA RMSD threshold counted as a hit")
    return p.parse_args()


def read_pdb(path: str) -> dict:
    """chain -> {reskey: {atomname: xyz}} in file order."""
    out: dict = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        ch = line[21]
        key = line[22:27]
        name = line[12:16].strip()
        xyz = np.array([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
        out.setdefault(ch, {}).setdefault(key, {})[name] = xyz
        out[ch][key]["_resname"] = line[17:20].strip()  # type: ignore[assignment]
    return out


def seq_of(chain: dict) -> str:
    return "".join(THREE_TO_ONE.get(r["_resname"], "X") for r in chain.values())


def ca_array(chain: dict) -> np.ndarray:
    return np.array([r["CA"] for r in chain.values() if "CA" in r])


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, W = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ W))
    Pr = Pc @ (V @ np.diag([1.0, 1.0, d]) @ W)
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


def interface_mask(native: dict, cutoff: float) -> np.ndarray:
    """True for peptide residues with any heavy atom within `cutoff` of the receptor."""
    pep = native["A"]
    rec_xyz = np.array([xyz for res in native.get("B", {}).values()
                        for name, xyz in res.items() if name != "_resname"])
    mask = []
    for res in pep.values():
        atoms = np.array([xyz for name, xyz in res.items() if name != "_resname"])
        if len(rec_xyz) == 0 or len(atoms) == 0:
            mask.append(False)
            continue
        d = np.linalg.norm(atoms[:, None, :] - rec_xyz[None, :, :], axis=-1)
        mask.append(bool((d <= cutoff).any()))
    return np.array(mask)


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(args.stage_dir) / "staged_manifest.json").read_text())
    by_id = {r["id"]: r for r in man["staged"]}

    rows, problems = [], []
    for pid, meta in sorted(by_id.items()):
        native = read_pdb(meta["path"])
        if "A" not in native:
            problems.append({"id": pid, "reason": "no_chain_A_in_staged"})
            continue
        nat_seq = seq_of(native["A"])
        nat_ca = ca_array(native["A"])
        iface = interface_mask(native, args.contact_cutoff)

        poses = sorted(glob.glob(os.path.join(args.run_dir, "**", f"{pid}_*_sample_*.pdb"),
                                 recursive=True))
        if not poses:
            problems.append({"id": pid, "reason": "no_poses_found"})
            continue
        for pose in poses:
            gen = read_pdb(pose)
            if "A" not in gen:
                problems.append({"id": pid, "pose": pose, "reason": "no_chain_A_in_pose"})
                continue
            gen_seq = seq_of(gen["A"])
            gen_ca = ca_array(gen["A"])
            n = min(len(nat_ca), len(gen_ca))
            if n == 0 or len(gen_seq) != len(nat_seq):
                problems.append({"id": pid, "pose": pose, "reason": "length_mismatch",
                                 "n_native": len(nat_seq), "n_generated": len(gen_seq)})
                continue
            P, Q = gen_ca[:n], nat_ca[:n]
            ident = sum(a == b for a, b in zip(gen_seq, nat_seq)) / len(nat_seq)
            raw = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
            m = iface[:n]
            if m.any():
                iface_raw = float(np.sqrt(((P[m] - Q[m]) ** 2).sum(1).mean()))
            else:
                iface_raw = float("nan")
            rows.append({
                "id": pid, "pose": os.path.basename(pose), "n_res": int(n),
                "seq_identity": ident,
                "seq_identical": bool(gen_seq == nat_seq),
                "raw_ca_rmsd": raw,
                "superposed_ca_rmsd": kabsch_rmsd(P, Q),
                "interface_raw_ca_rmsd": iface_raw,
                "n_interface_res": int(m.sum()),
            })

    with open(out / "poses.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # ---------------------------------------------------------------- summary
    def med(vals):
        v = [x for x in vals if x == x]      # drop NaN
        return float(np.median(v)) if v else float("nan")

    ids = sorted({r["id"] for r in rows})
    per_target = []
    for pid in ids:
        rs = [r for r in rows if r["id"] == pid]
        # PAIRWISE diversity among this target's poses, measured directly between poses.
        # Spread of RMSD-TO-NATIVE does NOT establish this: several poses can each sit
        # ~5 A from the native while being far from each other. And it is load-bearing for
        # the k>0 sweep, because "generate N candidates and select" is only worth its
        # compute if the N differ.
        div = []
        pose_cas = []
        paths = {os.path.basename(p): p
                 for p in glob.glob(os.path.join(args.run_dir, "**", f"{pid}_*_sample_*.pdb"),
                                    recursive=True)}
        for r in rs:
            p = paths.get(r["pose"])
            if p:
                pose_cas.append(ca_array(read_pdb(p)["A"]))
        for i in range(len(pose_cas)):
            for j in range(i + 1, len(pose_cas)):
                n = min(len(pose_cas[i]), len(pose_cas[j]))
                if n:
                    d = pose_cas[i][:n] - pose_cas[j][:n]
                    div.append(float(np.sqrt((d ** 2).sum(1).mean())))
        per_target.append({
            "id": pid,
            "n_poses": len(rs),
            "all_seq_identical": all(r["seq_identical"] for r in rs),
            "best_raw": min(r["raw_ca_rmsd"] for r in rs),
            "median_raw": med([r["raw_ca_rmsd"] for r in rs]),
            "best_interface": min((r["interface_raw_ca_rmsd"] for r in rs
                                   if r["interface_raw_ca_rmsd"] == r["interface_raw_ca_rmsd"]),
                                  default=float("nan")),
            "pairwise_diversity_median": med(div),
            "pairwise_diversity_max": float(max(div)) if div else float("nan"),
        })

    n_ident = sum(1 for r in rows if r["seq_identical"])
    summary = {
        "run_dir": args.run_dir,
        "stage_dir": args.stage_dir,
        "contact_cutoff_a": args.contact_cutoff,
        "success_threshold_a": args.success_a,
        "n_targets": len(ids),
        "n_poses": len(rows),
        "n_problems": len(problems),
        "problems": problems[:40],
        # The k=0 contract. Anything but 1.000 invalidates the similarity axis.
        "seq_identity_mean": float(np.mean([r["seq_identity"] for r in rows])) if rows else None,
        "frac_poses_seq_identical": n_ident / len(rows) if rows else None,
        "n_targets_all_identical": sum(1 for t in per_target if t["all_seq_identical"]),
        "median_raw_ca_rmsd": med([r["raw_ca_rmsd"] for r in rows]),
        "median_superposed_ca_rmsd": med([r["superposed_ca_rmsd"] for r in rows]),
        "median_interface_raw_ca_rmsd": med([r["interface_raw_ca_rmsd"] for r in rows]),
        "median_best_of_n_raw": med([t["best_raw"] for t in per_target]),
        "median_best_of_n_interface": med([t["best_interface"] for t in per_target]),
        # Low diversity means N samples cost N x compute for ~1 sample of information.
        "median_pairwise_diversity": med([t["pairwise_diversity_median"] for t in per_target]),
        "frac_poses_under_threshold": (
            sum(1 for r in rows if r["interface_raw_ca_rmsd"] <= args.success_a) / len(rows)
            if rows else None),
        "frac_targets_any_under_threshold": (
            sum(1 for t in per_target if t["best_interface"] <= args.success_a) / len(per_target)
            if per_target else None),
        "per_target": per_target,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[score] targets {summary['n_targets']}  poses {summary['n_poses']}  "
          f"problems {summary['n_problems']}")
    print(f"[score] k=0 CONTRACT: poses with identical sequence "
          f"{summary['frac_poses_seq_identical']:.4f}  "
          f"({summary['n_targets_all_identical']}/{summary['n_targets']} targets all-identical)")
    print(f"[score] median raw CA RMSD        : {summary['median_raw_ca_rmsd']:.2f} A")
    print(f"[score] median superposed CA RMSD : {summary['median_superposed_ca_rmsd']:.2f} A")
    print(f"[score] median interface raw RMSD : {summary['median_interface_raw_ca_rmsd']:.2f} A")
    print(f"[score] best-of-N raw (median)    : {summary['median_best_of_n_raw']:.2f} A")
    print(f"[score] best-of-N interface       : {summary['median_best_of_n_interface']:.2f} A")
    print(f"[score] poses <= {args.success_a} A interface  : "
          f"{summary['frac_poses_under_threshold']:.3f}")
    print(f"[score] targets with any hit      : "
          f"{summary['frac_targets_any_under_threshold']:.3f}")
    print(f"[score] pairwise pose diversity   : "
          f"{summary['median_pairwise_diversity']:.2f} A (median over targets)")
    if problems:
        print(f"[score] problems (first 5): {problems[:5]}")
    print(f"[score] rows    -> {out / 'poses.jsonl'}")
    print(f"[score] summary -> {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
