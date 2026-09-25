"""Scoring of perturbed poses, before and after minimization (pre-build check 1, stage C).

For every pose this computes, on BOTH the pre- and post-minimization structure:

  * Rosetta interface dG                 -- the quantity the filter would ideally use
  * `contact_retention`                  -- the cheap quantity being calibrated against it
  * interface contact Jaccard distance   -- from the native pose
  * peptide CA-RMSD from native
  * buried SASA fraction

The native reference is each complex's own pose 00, which is the identity transform, so
"distance from native" is measured against the same structure the perturbation started
from rather than against a re-read of the original file.

**No superposition anywhere.**  Receptor coordinates are shared by construction -- only the
peptide was moved, and the receptor was frozen during minimization -- so CA-RMSD is
displacement WITHIN the binding site.  A Kabsch fit would hide exactly the motion being
measured.

**FastRelax is off.**  Relaxing before scoring would undo the perturbation being calibrated,
which is the one thing the measurement cannot survive; and at roughly 10k scorings it would
also dominate the job. The pose is scored as given.

Runs in `.venv` (mdtraj + pyrosetta).  Reads the PDBs `.venv_openmm` wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from script_utils import precheck_config  # noqa: E402


def load_heavy(path: str):
    """(peptide CA, per-residue peptide heavy atoms, receptor heavy atoms, residue ids).

    Hydrogens are dropped: the minimized structures carry them and the inputs do not, so
    including them would make a pre/post comparison partly a comparison of atom counts.
    """
    import mdtraj as md

    traj = md.load(path)
    top = traj.topology
    if top.n_chains < 2:
        return None
    xyz = traj.xyz[0] * 10.0

    def heavy(res):
        return [a.index for a in res.atoms
                if a.element is not None and a.element.symbol != "H"]

    pep_res = list(top.chain(1).residues)
    rec_res = list(top.chain(0).residues)
    pep_heavy = [xyz[heavy(r)] for r in pep_res if heavy(r)]
    rec_heavy = [xyz[heavy(r)] for r in rec_res if heavy(r)]
    ca = []
    for r in pep_res:
        hit = [a.index for a in r.atoms if a.name == "CA"]
        if hit:
            ca.append(xyz[hit[0]])
    if not pep_heavy or not rec_heavy or not ca:
        return None
    return (np.array(ca), pep_heavy, rec_heavy,
            [int(r.resSeq) for r in rec_res if heavy(r)])


def ca_contacts(pep_ca: np.ndarray, rec_heavy: list[np.ndarray], cutoff_A: float) -> set:
    """{(peptide residue, receptor residue)} by CA-to-receptor-CA distance.

    Matched to `sdedit_cyclize.py`'s CONTACT_CUTOFF_NM = 1.0 and to `kinematic_ceiling`, so
    a retention measured here is the same quantity those report.
    """
    rec_ca = np.array([r.mean(axis=0) for r in rec_heavy])   # centroid stands in for CA
    d = np.linalg.norm(pep_ca[:, None, :] - rec_ca[None, :, :], axis=-1)
    pi, ri = np.where(d < cutoff_A)
    return {(int(p), int(r)) for p, r in zip(pi, ri)}


def heavy_contacts(pep_heavy: list[np.ndarray], rec_heavy: list[np.ndarray],
                   cutoff_A: float) -> set:
    """{(peptide residue, receptor residue)} with any heavy-atom pair within `cutoff_A`."""
    out = set()
    for i, p in enumerate(pep_heavy):
        for j, r in enumerate(rec_heavy):
            if np.min(np.linalg.norm(p[:, None, :] - r[None, :, :], axis=-1)) < cutoff_A:
                out.add((i, j))
    return out


def buried_fraction(pep_heavy: list[np.ndarray], rec_heavy: list[np.ndarray],
                    cutoff_A: float) -> float:
    """Fraction of peptide residues with any heavy atom within `cutoff_A` of the receptor.

    A residue-level proxy for buried SASA. Used instead of a Shrake-Rupley recomputation
    because it is defined identically on the pre- and post-minimization structures without
    depending on how hydrogens were placed.
    """
    if not pep_heavy:
        return float("nan")
    n = sum(1 for p in pep_heavy
            if any(np.min(np.linalg.norm(p[:, None, :] - r[None, :, :], axis=-1)) < cutoff_A
                   for r in rec_heavy))
    return n / len(pep_heavy)


def rosetta_dg(path: str) -> float:
    from proteinfoundation.evaluation.rosetta_energy import (
        compute_rosetta_interface_metrics_single,
    )
    try:
        m = compute_rosetta_interface_metrics_single(path, binder_chain="B",
                                                     target_chains=["A"])
        v = m.get("binder_rosetta_dG_separated")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def score_structure(path: str, native, ccfg: dict, with_rosetta: bool) -> dict:
    """Every metric for one structure, against its complex's native pose."""
    cur = load_heavy(path)
    if cur is None:
        return {}
    ca, pep_h, rec_h, _ = cur
    n_ca, n_pep_h, n_rec_h, _ = native

    cut_ca = float(ccfg["contact_cutoff_A"])
    cut_hv = float(ccfg["heavy_cutoff_A"])

    nat_ca_set = ca_contacts(n_ca, n_rec_h, cut_ca)
    cur_ca_set = ca_contacts(ca, rec_h, cut_ca)
    nat_hv = heavy_contacts(n_pep_h, n_rec_h, cut_hv)
    cur_hv = heavy_contacts(pep_h, rec_h, cut_hv)

    union = len(nat_hv | cur_hv)
    inter = len(nat_hv & cur_hv)
    m = min(len(ca), len(n_ca))

    return {
        "contact_retention": (len(nat_ca_set & cur_ca_set) / len(nat_ca_set)
                              if nat_ca_set else float("nan")),
        "n_contacts_ca": len(cur_ca_set),
        "n_contacts_heavy": len(cur_hv),
        "jaccard": inter / union if union else float("nan"),
        "jaccard_distance": 1.0 - (inter / union) if union else float("nan"),
        "ca_rmsd_A": float(np.sqrt(((ca[:m] - n_ca[:m]) ** 2).sum(-1).mean())),
        "ca_max_dev_A": float(np.sqrt(((ca[:m] - n_ca[:m]) ** 2).sum(-1)).max()),
        "buried_frac": buried_fraction(pep_h, rec_h, cut_hv),
        "rosetta_dG": rosetta_dg(path) if with_rosetta else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--minimized-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-rosetta", action="store_true",
                    help="Geometry only. For smoke runs; a scored row without dG cannot "
                         "answer the calibration question.")
    args = ap.parse_args()

    cfg = precheck_config.load(args.config)
    ccfg = cfg["decoy_calibration"]["score"]
    with_rosetta = not args.no_rosetta

    rows = []
    for f in sorted(Path(args.minimized_dir).glob("minimized_shard*.jsonl")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"{f}:{n}: unparseable row ({exc}). Refusing to continue.")
    if not rows:
        raise SystemExit(f"no minimized rows under {args.minimized_dir}")

    by_example: dict[str, list] = {}
    for r in rows:
        by_example.setdefault(r["example_id"], []).append(r)
    examples = sorted(by_example)[args.shard::args.n_shards]
    if args.limit:
        examples = examples[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    done.add((d["example_id"], d["pose_index"]))
                except Exception:
                    raise SystemExit(f"{out_path}: unparseable row while resuming")

    t0 = time.perf_counter()
    n_ok = 0
    with out_path.open("a") as fh:
        for ex in examples:
            poses = sorted(by_example[ex], key=lambda r: r["pose_index"])
            # Pose 00 pre-minimization IS the native: the identity transform, before any
            # relaxation. Every distance for this complex is measured against it.
            native_row = next((p for p in poses if int(p["pose_index"]) == 0), None)
            if native_row is None:
                print(f"  skip {ex}: no pose 00, so no native reference", flush=True)
                continue
            native = load_heavy(native_row["pre_pdb"])
            if native is None:
                print(f"  skip {ex}: native unreadable", flush=True)
                continue

            for p in poses:
                if (ex, p["pose_index"]) in done:
                    continue
                rec = {k: p[k] for k in p if not k.endswith("_pdb")}
                for phase, key in (("pre", "pre_pdb"), ("post", "post_pdb")):
                    s = score_structure(p[key], native, ccfg, with_rosetta)
                    rec.update({f"{phase}_{k}": v for k, v in s.items()})
                # The collapse measure: how far the minimized pose ended up from native,
                # against how far it started. Below the configured threshold it collapsed.
                rec["collapsed"] = int(
                    float(rec.get("post_ca_rmsd_A", np.inf))
                    <= float(cfg["decoy_calibration"]["analysis"]["collapse_rmsd_A"]))
                fh.write(json.dumps(rec, default=float) + "\n")
                fh.flush()
                n_ok += 1
            print(f"  {ex}: {len(poses)} poses ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"wrote {out_path}: {n_ok} poses scored, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
