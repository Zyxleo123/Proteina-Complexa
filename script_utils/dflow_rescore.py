"""Re-score any arm's written complexes through ONE acceptance instrument.

Why every reported number comes from here
----------------------------------------
The GPU driver computes acceptance from its own tensors, which is what it selects on. But the
comparison this experiment exists to make -- D-Flow against SDEdit against guided SDEdit --
must not depend on which code path produced each arm's numbers. The older arms were scored by
a different function with a different contact definition (CA-CA within 10 A) and no clash,
sequence-budget or interface-RMSD test at all, so their own columns cannot be compared to
anything here.

So this reads PDBs, not tensors, and it reads them the same way for every arm: the candidate
peptide and the receptor from the written complex (which is already in the input PDB's frame),
the original LP from that example's `input_complex_pdb`. Identical parser, identical
`accept()`, identical thresholds. An arm's own reported columns are carried through untouched
for reference but never used as the verdict.

Ring declaration is reported, not scored
----------------------------------------
`accept()` can gate on a declared covalent ring, and the D-Flow writer emits and verifies a
CONECT record. The earlier SDEdit runs were written before that existed, so every one of
their complexes would fail that gate for a reason that is about the writer and not about the
structure. Including it would make the baselines look worse on a plumbing difference, so this
records `file_ring_declared` as its own column and leaves it OUT of the joint gate; the joint
gate's ring clause is `gate_ring_closed`, which is measured from the anchor atoms and is
equally available for every arm. (Rosetta is unaffected either way: `_declare_cyclic_bond`
takes explicit endpoints and never reads CONECT.)

Frames
------
No superposition, anywhere. `write_complex` puts the candidate peptide in the input PDB's
frame and copies the receptor verbatim, so the candidate, the original and the receptor are
already in one frame and a subtraction is the honest displacement inside the binding site.
The `frame_residual_A` the writer recorded is the check that this held; rows whose residual
was too large never got a complex written in the first place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "script_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dflow_accept import accept  # noqa: E402
from dflow_losses import NM_TO_ANG  # noqa: E402
from dflow_pdb import verify_ring  # noqa: E402
from proteinfoundation.cyclization.constants import NAME_TO_CYCLIZATION_TYPE  # noqa: E402
from proteinfoundation.utils.pdb_utils import from_pdb_string  # noqa: E402

_CACHE: dict = {}


def _parse(path: str, chain: str | None):
    """(atom37 nm, atom mask, aatype, residue numbers) for one chain of a PDB. Cached."""
    key = (str(path), chain)
    if key in _CACHE:
        return _CACHE[key]
    prot = from_pdb_string(Path(path).read_text(), chain_id=chain)
    if prot.aatype.shape[0] == 0:
        raise ValueError(f"no residues parsed from {path} chain {chain!r}")
    out = (
        torch.as_tensor(prot.atom_positions, dtype=torch.float32) / NM_TO_ANG,
        torch.as_tensor(prot.atom_mask, dtype=torch.float32).bool(),
        torch.as_tensor(prot.aatype, dtype=torch.long),
        torch.as_tensor(prot.residue_index, dtype=torch.long),
    )
    # Bounded: one entry per (file, chain), and input complexes are shared across candidates
    # so the cache is what keeps this stage from re-parsing the same receptor thousands of
    # times. Candidate complexes are each read once, so they are not cached.
    if "input" in str(path) or chain is None or len(_CACHE) < 4096:
        _CACHE[key] = out
    return out


def _receptor(complex_path: str, binder_chain: str, target_chains: list[str]):
    # Parsing per target chain keeps the receptor to exactly the chains the arm declared,
    # rather than "everything that is not the binder" -- which would silently absorb a
    # crystallographic waters chain or a second copy of the peptide.
    if not target_chains:
        raise ValueError(f"no target chains declared for {complex_path}")
    parts = [_parse(complex_path, c) for c in target_chains]
    coors = torch.cat([p[0] for p in parts], dim=0)
    amask = torch.cat([p[1] for p in parts], dim=0)
    return coors, amask


def rescore_row(r: dict, arm: str, declared_gate: bool = False) -> dict:
    cyc_name = str(r.get("cyc_type_name") or r.get("cyc_type"))
    if cyc_name not in NAME_TO_CYCLIZATION_TYPE:
        raise ValueError(f"unknown cyclization type {cyc_name!r}")
    cyc_type = NAME_TO_CYCLIZATION_TYPE[cyc_name]
    bchain = str(r.get("binder_chain", "B"))
    tchains = [c for c in str(r.get("target_chains", "A")).split(",") if c]

    pep, pep_m, pep_aa, pep_res = _parse(r["complex_pdb"], bchain)
    ref, ref_m, ref_aa, _ = _parse(r["input_complex_pdb"], bchain)
    tgt, tgt_m = _receptor(r["input_complex_pdb"], bchain, tchains)

    i = int(r.get("cyc_i", 0))
    j = int(r.get("cyc_j", pep.shape[0] - 1))
    v = verify_ring(r["complex_pdb"], int(pep_res[i]), int(pep_res[j]), cyc_type, chain=bchain)

    out = accept(
        pep_coors_nm=pep, pep_amask=pep_m, pep_aatype=pep_aa,
        ref_coors_nm=ref, ref_amask=ref_m, ref_aatype=ref_aa,
        tgt_coors_nm=tgt, tgt_amask=tgt_m,
        cyc_type=cyc_type, cyc_i=i, cyc_j=j,
        protect=r.get("protect"),
        ring_declared=bool(v["declared"]) if declared_gate else None,
    )
    out["file_ring_declared"] = bool(v["declared"])
    out["file_bond_dist_A"] = v["dist_A"]
    out["arm"] = arm
    out["complex_pdb"] = r["complex_pdb"]
    out["input_complex_pdb"] = r["input_complex_pdb"]
    # Carried through so the report can group the baselines by their own knobs, and so a
    # D-Flow row keeps its restart/stage identity. Never used as a verdict.
    for k in ("run_key", "example_id", "pdb_tag", "stage", "restart", "init_mode",
              "objective_tag", "seed", "t_ca_start", "t_lat_start", "guidance_w",
              "guidance_loss", "graft_anchors", "sampler_mode", "opt_nsteps", "eval_nsteps",
              "opt_steps", "objective", "grid_consistent", "coarse_gate_ring_closed",
              "opt_seconds", "opt_loss_first", "opt_loss_best", "input_peptide_length",
              "input_nc_gap_angstrom", "input_cluster_id", "frame_residual_A"):
        if k in r:
            out[k] = r[k]
    out.setdefault("run_key", r.get("pdb_tag"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--arm", required=True,
                    help="Label for this arm, e.g. dflow_source / sdedit / sdedit_guided.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default=None, choices=[None, "opt", "init"],
                    help="D-Flow rows only: restrict to one stage.")
    ap.add_argument("--declared-gate", action="store_true",
                    help="Include the CONECT ring declaration in the joint gate. Off by "
                         "default: the older SDEdit arms were written before the declaring "
                         "writer existed, and gating on it would penalize them for plumbing.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--examples-file", default=None,
                    help="Restrict to the example_ids in this file (blank lines and '#' "
                         "comments ignored). The baselines were run over all 60 LNR peptides "
                         "and a large t-grid; the comparison is only meaningful on the "
                         "peptides both arms actually attempted, so this scopes them to the "
                         "same set instead of comparing 8 peptides against 60.")
    args = ap.parse_args()

    keep: set[str] | None = None
    if args.examples_file:
        text = Path(args.examples_file).read_text().splitlines()
        keep = {ln.strip() for ln in text if ln.strip() and not ln.lstrip().startswith("#")}
        if not keep:
            raise SystemExit(f"FATAL: --examples-file is empty: {args.examples_file}")

    rows = []
    for p in args.rows:
        path = Path(p)
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"FATAL: corrupt row {path}:{lineno}")
    cands = [r for r in rows if r.get("status") == "ok" and r.get("complex_pdb")]
    if keep is not None:
        before = len(cands)
        cands = [r for r in cands if str(r.get("example_id")) in keep]
        print(f"examples filter: {before} -> {len(cands)} rows over {len(keep)} example(s)",
              flush=True)
        if not cands:
            # A filter that matches nothing is almost always a staging mismatch (the arm ran
            # on different example_ids), and an empty arm in the report is indistinguishable
            # from an arm that scored zero.
            raise SystemExit(
                f"FATAL: --examples-file matched no row in arm '{args.arm}'. The arm's "
                "example_ids do not overlap the requested set."
            )
    if args.stage:
        cands = [r for r in cands if r.get("stage") == args.stage]
    cands.sort(key=lambda r: str(r.get("run_key") or r.get("pdb_tag")))
    if args.limit:
        cands = cands[: args.limit]
    mine = cands[args.shard_index :: args.shard_count]
    print(f"arm={args.arm}: {len(rows)} rows -> {len(cands)} with complexes; shard "
          f"{args.shard_index}/{args.shard_count} takes {len(mine)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["rescore_key"])
    if done:
        print(f"resuming: {len(done)} done", flush=True)

    n = n_fail = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for r in mine:
            key = f"{args.arm}|{r.get('run_key') or r.get('pdb_tag')}"
            if key in done:
                continue
            try:
                row = rescore_row(r, args.arm, declared_gate=args.declared_gate)
                row["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                row = {"status": "failed", "arm": args.arm,
                       "example_id": r.get("example_id"),
                       "cyc_type_name": r.get("cyc_type_name") or r.get("cyc_type"),
                       "complex_pdb": r.get("complex_pdb"),
                       "error": f"{type(exc).__name__}: {exc}"}
            row["rescore_key"] = key
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n} rescored ({time.time() - t0:.0f}s, {n_fail} failed)", flush=True)
            if args.max_seconds and (time.time() - t0) > args.max_seconds:
                print(f"  budget reached after {n}; resubmit to continue", flush=True)
                break
    print(f"done: {n} rescored, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)
    if n and n_fail == n:
        raise SystemExit(f"FATAL: every one of {n} rescores failed.")


if __name__ == "__main__":
    main()
