"""Pose-decoy sampling, stage A (pre-build check 1).

Selects native holo complexes stratified by peptide length and interface burial, builds the
stratified pose ladder for each, and writes one complex PDB per pose plus a manifest of the
transform parameters.

Stratification is not decoration.  A regression of interface dG on `contact_retention` fit
on whatever the metadata happened to order first would be dominated by one regime -- short
peptides, or shallow interfaces -- and the band bounds it produced would only be valid
there.  The strata are the product of the two edge lists in the config; each cell is filled
to an equal quota and any cell that could not be filled is REPORTED rather than silently
back-filled from a neighbouring cell.

Only the PEPTIDE moves.  The receptor is written unchanged in every pose, so the frozen
pocket in stage B and the interface frame in this stage refer to the same coordinates.

Runs in `.venv`.  CPU only; no OpenMM and no PyRosetta on this path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script_utils import precheck_config  # noqa: E402
from script_utils.peptide_profile import load_complex, sasa_terms  # noqa: E402
from script_utils.pose_decoys import (  # noqa: E402
    build_ladder, detect_ring_bond, interface_frame,
)
from script_utils.precheck_profile import select_rows  # noqa: E402


def _bin_index(value: float, edges: list[float]) -> int:
    """Half-open bins (lo, hi]; -1 when the value falls outside every bin."""
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if lo < value <= hi:
            return i
    return -1


def choose_complexes(cfg: dict, pool_factor: int = 4) -> tuple[pd.DataFrame, dict]:
    """Stratified selection by (peptide length, interface burial).

    Burial has to be MEASURED, so a candidate pool several times the target size is
    profiled and then drawn down into the strata.  Profiling the whole set to fill 300
    slots would cost hours for a number used only to bin.
    """
    dcfg = cfg["decoy_calibration"]
    n_target = int(dcfg["n_complexes"])
    seed = int(dcfg.get("seed", 0))
    len_edges = [float(e) for e in dcfg["strata"]["peptide_length_edges"]]
    bur_edges = [float(e) for e in dcfg["strata"]["burial_edges"]]

    pool = select_rows(dcfg["metadata"], dcfg.get("source_filter", ""),
                       n_target * pool_factor, seed)

    rows = []
    t0 = time.perf_counter()
    for n, (_, r) in enumerate(pool.iterrows(), 1):
        cx = load_complex(r["path"], example_id=str(r["example_id"]))
        if cx is None:
            continue
        try:
            buried, _ = sasa_terms(cx)
        except Exception:
            continue
        if not np.isfinite(buried):
            continue
        rows.append({"example_id": str(r["example_id"]), "path": str(r["path"]),
                     "cluster_id": r.get("cluster_id"),
                     "peptide_length": int(cx.length), "buried_sasa_frac": float(buried),
                     "len_bin": _bin_index(float(cx.length), len_edges),
                     "bur_bin": _bin_index(float(buried), bur_edges)})
        if n % 100 == 0:
            print(f"  pool {n}/{len(pool)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df = df[(df["len_bin"] >= 0) & (df["bur_bin"] >= 0)]
    if df.empty:
        raise SystemExit("no complexes survived burial profiling")

    n_cells = (len(len_edges) - 1) * (len(bur_edges) - 1)
    quota = max(1, n_target // n_cells)
    rng = np.random.default_rng(seed)

    picked, cells = [], {}
    for (li, bi), g in df.groupby(["len_bin", "bur_bin"]):
        take = g.iloc[rng.permutation(len(g))[:quota]]
        picked.append(take)
        cells[f"len{li}_bur{bi}"] = {"available": int(len(g)), "quota": quota,
                                     "taken": int(len(take))}
    out = pd.concat(picked).reset_index(drop=True)

    # Any shortfall is reported, and only then topped up from outside the strata -- so the
    # report can say whether the final set is actually balanced or merely the right size.
    short = {k: v for k, v in cells.items() if v["taken"] < v["quota"]}
    if len(out) < n_target:
        rest = df[~df["example_id"].isin(set(out["example_id"]))]
        if len(rest):
            extra = rest.iloc[rng.permutation(len(rest))[:n_target - len(out)]]
            out = pd.concat([out, extra]).reset_index(drop=True)

    meta = {"n_pool_profiled": int(len(df)), "n_cells": n_cells, "quota_per_cell": quota,
            "cells": cells, "underfilled_cells": short, "n_selected": int(len(out)),
            "n_target": n_target,
            "topped_up_outside_strata": int(max(0, len(out) - sum(
                c["taken"] for c in cells.values())))}
    return out, meta


def write_poses(cx, poses, out_dir: Path, example_id: str) -> list[dict]:
    """One complex PDB per pose.  Only peptide atoms move."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top = cx.traj.topology
    pep_atoms = np.array([a.index for r in top.residues if r.index in set(cx.pep_res_index)
                          for a in r.atoms], dtype=int)
    centre = cx.CA.mean(axis=0)

    rows = []
    for p in poses:
        traj = cx.traj[0]
        xyz = traj.xyz.copy()                       # [1, n_atoms, 3] in nm
        moved = p.apply(xyz[0][pep_atoms] * 10.0, centre) / 10.0
        xyz[0, pep_atoms, :] = moved.astype(xyz.dtype)
        traj = traj.__class__(xyz, top)

        name = f"{example_id}__pose{p.index:02d}.pdb"
        traj.save_pdb(str(out_dir / name))
        row = {"example_id": example_id, "pose_pdb": str(out_dir / name)}
        row.update(p.as_row())
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--selection", default="",
                    help="Reuse a previously written selection.csv instead of re-choosing. "
                         "Every shard MUST be given the same selection, or the shards are "
                         "sampling different complexes.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = precheck_config.load(args.config)
    dcfg = cfg["decoy_calibration"]
    pcfg = dcfg["poses"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_path = Path(args.selection) if args.selection else (out_dir / "selection.csv")
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        print(f"reusing selection: {sel_path} ({len(sel)} complexes)")
    else:
        sel, meta = choose_complexes(cfg)
        sel.to_csv(sel_path, index=False)
        (out_dir / "selection_strata.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote selection: {sel_path} ({len(sel)} complexes)")
        if meta["underfilled_cells"]:
            print(f"  UNDERFILLED strata cells: {meta['underfilled_cells']}")

    if args.limit:
        sel = sel.head(args.limit)
    sel = sel.iloc[args.shard::args.n_shards]

    pdb_dir = out_dir / "poses"
    manifest = out_dir / f"manifest_shard{args.shard}.jsonl"

    done: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["example_id"])
                except Exception:
                    raise SystemExit(f"{manifest}: unparseable row while resuming; "
                                     "delete it and rerun rather than skipping rows")

    t0 = time.perf_counter()
    n_ok = 0
    with manifest.open("a") as fh:
        for _, r in sel.iterrows():
            ex = str(r["example_id"])
            if ex in done:
                continue
            cx = load_complex(r["path"], example_id=ex)
            if cx is None:
                continue
            frame = interface_frame(cx.pep_heavy, cx.rec_heavy, cx.CA)
            ring = detect_ring_bond(cx)
            if frame.n_contacts == 0:
                print(f"  skip {ex}: no heavy-atom contacts at the interface", flush=True)
                continue
            poses = build_ladder(frame, int(pcfg["n_per_complex"]),
                                 [float(x) for x in pcfg["translation_A"]],
                                 [float(x) for x in pcfg["rotation_deg"]],
                                 seed=int(dcfg.get("seed", 0)) + abs(hash(ex)) % 10_000)
            for row in write_poses(cx, poses, pdb_dir, ex):
                row.update({
                    "native_pdb": str(r["path"]),
                    "peptide_length": int(cx.length),
                    "buried_sasa_frac_native": float(r.get("buried_sasa_frac", np.nan)),
                    "n_interface_contacts": frame.n_contacts,
                    "frame_normal": [float(x) for x in frame.normal],
                    # Rotation centre, needed with (tx,ty,tz) and the axis/angle to
                    # reconstruct the transform on the hydrogenated system downstream.
                    "centre_x_A": float(cx.CA.mean(axis=0)[0]),
                    "centre_y_A": float(cx.CA.mean(axis=0)[1]),
                    "centre_z_A": float(cx.CA.mean(axis=0)[2]),
                })
                # Carried so the OpenMM stage can hold the ring closed without parsing
                # CONECT: that parser lives behind the torch stack, which .venv_openmm
                # does not have.
                row.update(ring or {"ring_atom_i": -1, "ring_atom_j": -1})
                fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  [{n_ok}] {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"wrote {manifest}: {n_ok} complexes x {pcfg['n_per_complex']} poses "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
