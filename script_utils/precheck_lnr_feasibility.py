"""LNR ring-closure feasibility frontier (pre-build check 2).

The question: for each LNR target, does a ring-closing solution exist that preserves the
input interface -- and if so, how much of the peptide has to move to reach it?

Feasibility is meaningless unscoped.  Release the whole peptide and almost everything
closes, with the interface gone.  So nothing here reports a boolean; it reports a frontier
over two swept parameters:

  * `k` -- how many residues are released from their input conformation.  The complement,
    a contiguous held window `[a, b]`, stays at its native bound position.
  * `tolerance` -- the interface retention a solution must still satisfy.

A (target, k, tolerance) cell is feasible when some window with `L - (b - a + 1) <= k`
closes AND retains at least `tolerance` of the interface.  The tolerance axis is applied in
the report rather than stored, because it is a threshold on a number this file already
emits and storing the cross product would multiply the rows by the grid for no information.

Two things this shares byte-for-byte with the Milestone 1.5 ceiling audit, on purpose, so
the numbers can be divided into each other: the CA-CA 10 A contact definition, and the
analytic/torsion closure solver in `kinematic_ceiling.py`.  A frontier computed against a
different cutoff is a different quantity.

## What `k` means, and why it is the axis rather than the retention ratio

`README_M15_CEILING_AUDIT.md` records the trap at length: retention counted this way is a
RATIO with peptide length in its denominator, so a trend in it against anything correlated
with length is arithmetic.  `k` is a residue count and has no such contamination.  The
retention numbers are emitted and are the tolerance axis, but any claim about *what makes
closure expensive* is read off `k`.

## What the held window can and cannot express

The released set is the two flanks of a contiguous held window.  Releasing an interior loop
while pinning both termini is NOT in the sweep.  For head-to-tail closure that restriction
costs nothing -- pinning both termini fixes the very atoms the bond has to join, so those
windows are feasible only when the ring is already closed -- but for a bridged chemistry it
is a real restriction, and the report says so rather than presenting the frontier as
exhaustive.

Runs in `.venv` (numpy / scipy / pandas / pyarrow / mdtraj).  No OpenMM, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from script_utils import precheck_config  # noqa: E402
from script_utils.kinematic_ceiling import (  # noqa: E402
    bridge_spec,
    ca_contact_set,
    feasible_analytic,
    feasible_torsion,
    mainchain_spec,
    retention_ceiling,
)
from script_utils.m15_ceiling_audit import MIN_BRIDGE_SEP, load_staged  # noqa: E402
from script_utils.peptide_profile import anchor_residues, load_complex  # noqa: E402


# --------------------------------------------------------------------------------------
# The frontier for one (complex, closure spec)
# --------------------------------------------------------------------------------------
def _candidates_at_k(CA, contacts, anchor_contacts, rec_ca, spec, k, tau, cutoff,
                     atom_i, atom_j, with_retention: bool = True):
    """Every analytically-feasible window that releases exactly `k` residues.

    Returned as dicts with keys a, b, ret_all_strict, ret_all_perm, ret_anchor_strict,
    ret_anchor_perm.  Unsorted -- the caller sorts by whichever retention it is selecting
    on, and there are two of those.

    `with_retention=False` skips the retention accounting entirely.  Anchor-pair selection
    only needs to know WHETHER a window closes, and retention is the expensive half of this
    loop, so computing it there would dominate the whole job for a number nothing reads.
    """
    L = int(CA.shape[0])
    wlen = L - k
    out = []
    for a in range(0, k + 1):
        b = a + wlen - 1
        ok, _, _ = feasible_analytic(CA, spec, a, b, tau, None, atom_i, atom_j)
        if not ok:
            continue
        if with_retention:
            s_all, p_all = retention_ceiling(contacts, rec_ca, CA, a, b, tau, cutoff)
            s_anc, p_anc = retention_ceiling(anchor_contacts, rec_ca, CA, a, b, tau, cutoff)
        else:
            s_all = p_all = s_anc = p_anc = float("nan")
        out.append({"a": a, "b": b, "ret_all_strict": s_all, "ret_all_perm": p_all,
                    "ret_anchor_strict": s_anc, "ret_anchor_perm": p_anc})
    return out


def frontier_for_spec(st, contacts, anchor_contacts, spec, cfg, atom_i, atom_j,
                      refine: bool = True, max_solves: int = 60) -> list[dict]:
    """One row per `k`, from 0 (hold everything) to L-1 (hold a single residue).

    `k = L` -- release the entire peptide -- is not swept.  It is feasible by construction
    for every chemistry and retains nothing, so it is the trivial endpoint the brief warns
    against rather than a measurement.

    Torsion confirmation uses two exact prunes, both from the same monotonicity: holding
    FEWER residues can never make closure harder.

      * a window contained in a known SUCCESS closes for free -- no solve;
      * a window containing a known FAILURE fails for free -- no solve.

    Without the first, confirming the whole frontier would re-solve every level below the
    cheapest feasible one.
    """
    CA, N, C = st["CA"], st["N"], st["C"]
    L = int(CA.shape[0])
    tau, cutoff = cfg["tau_max_deg"], cfg["contact_cutoff_A"]
    tol_A = float((cfg.get("refine") or {}).get("tol_A", cfg["mainchain_tol_A"]))
    n_restarts = int((cfg.get("refine") or {}).get("n_restarts", 6))
    seed = int((cfg.get("refine") or {}).get("seed", 0))
    max_seconds = float((cfg.get("refine") or {}).get("max_seconds", 15.0))

    state = {"succeeded": [], "failed": [], "solves": 0, "exhausted": False}

    def _closes(a: int, b: int) -> tuple[bool, float, bool]:
        """(closes, residual, free) with both exact prunes applied."""
        if any(sa <= a and b <= sb for sa, sb in state["succeeded"]):
            return True, float("nan"), True          # sub-window of a known success
        if any(a <= fa and fb <= b for fa, fb in state["failed"]):
            return False, float("nan"), True         # superset of a known failure
        if state["solves"] >= max_solves:
            state["exhausted"] = True
            return False, float("nan"), True
        state["solves"] += 1
        ok, resid = feasible_torsion(N, CA, C, spec, a, b, n_restarts=n_restarts,
                                     seed=seed, tol_A=tol_A, max_seconds=max_seconds)
        state["succeeded" if ok else "failed"].append((a, b))
        return ok, resid, False

    def _select(cands, key, prefix, row, do_refine=True):
        """Best window by `key`, analytically and then torsion-confirmed, into `row`."""
        ranked = sorted(cands, key=lambda c: (-_finite(c[key]), c["a"]))
        if not ranked:
            for suffix in ("a", "b"):
                row[f"an_{prefix}_{suffix}"] = -1
                row[f"tor_{prefix}_{suffix}"] = -1
            for m in ("ret_all_strict", "ret_all_perm", "ret_anchor_strict", "ret_anchor_perm"):
                row[f"an_{prefix}_{m}"] = float("nan")
                row[f"tor_{prefix}_{m}"] = float("nan")
            row[f"tor_{prefix}_feasible"] = False
            row[f"tor_{prefix}_residual_A"] = float("nan")
            return

        best = ranked[0]
        row[f"an_{prefix}_a"], row[f"an_{prefix}_b"] = best["a"], best["b"]
        for m in ("ret_all_strict", "ret_all_perm", "ret_anchor_strict", "ret_anchor_perm"):
            row[f"an_{prefix}_{m}"] = best[m]

        row[f"tor_{prefix}_feasible"] = False
        row[f"tor_{prefix}_a"] = row[f"tor_{prefix}_b"] = -1
        row[f"tor_{prefix}_residual_A"] = float("nan")
        for m in ("ret_all_strict", "ret_all_perm", "ret_anchor_strict", "ret_anchor_perm"):
            row[f"tor_{prefix}_{m}"] = float("nan")
        if not do_refine:
            return
        for c in ranked:
            ok, resid, _ = _closes(c["a"], c["b"])
            if ok:
                row[f"tor_{prefix}_feasible"] = True
                row[f"tor_{prefix}_a"], row[f"tor_{prefix}_b"] = c["a"], c["b"]
                row[f"tor_{prefix}_residual_A"] = resid
                for m in ("ret_all_strict", "ret_all_perm",
                          "ret_anchor_strict", "ret_anchor_perm"):
                    row[f"tor_{prefix}_{m}"] = c[m]
                return
            if state["exhausted"]:
                return

    by_k = {k: _candidates_at_k(CA, contacts, anchor_contacts, rec_ca=st["rec_ca"],
                                spec=spec, k=k, tau=tau, cutoff=cutoff,
                                atom_i=atom_i, atom_j=atom_j)
            for k in range(0, L)}

    def _level_closes(k: int) -> bool:
        for c in sorted(by_k[k], key=lambda c: (-_finite(c["ret_all_strict"]), c["a"])):
            if _closes(c["a"], c["b"])[0]:
                return True
            if state["exhausted"]:
                break
        return False

    # Binary search for the cheapest feasible k, exploiting the same monotonicity as the
    # prunes: holding fewer residues can never make closure harder, so torsion feasibility
    # is non-decreasing in k and has a single threshold.
    #
    # This is not a micro-optimisation.  The expensive operation is PROVING a level
    # infeasible -- that runs every restart of every candidate -- and walking k upward from
    # 0 pays it at every level below the threshold.  Binary search pays it O(log L) times.
    k_star = None
    if refine:
        lo, hi = 0, L - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _level_closes(mid):
                k_star, hi = mid, mid - 1
            else:
                lo = mid + 1
            if state["exhausted"]:
                break

    rows: list[dict] = []
    for k in range(0, L):
        cands = by_k[k]
        row = {
            "k": k,
            "window_len": L - k,
            "n_windows_at_k": k + 1,
            "n_analytic_feasible_at_k": len(cands),
            "analytic_feasible": bool(cands),
            # True when this level was never probed directly: its infeasibility follows
            # from the threshold rather than from its own solves.  Distinguishing the two
            # keeps "measured infeasible" separate from "implied infeasible".
            "torsion_inferred": bool(refine and k_star is not None and k < k_star),
        }
        # Below the threshold nothing closes, so the torsion walk is skipped -- but the
        # ANALYTIC frontier at these levels is still a real measurement and is still
        # reported.  Blanking both would throw away the upper bound along with the lower.
        do_refine = refine and not row["torsion_inferred"]
        # TWO optima per k, because they are different questions and they disagree.
        # Selecting on all-contact retention can return a window that retains ZERO anchors
        # -- measured on LNR_1bjr mainchain at k=4 -- and the anchors are the part of the
        # interface the task is defined to preserve.  Reporting only `all` would have
        # scored that cell as a solution that keeps 55% of the interface when the part that
        # mattered was entirely gone.
        _select(cands, "ret_all_strict", "all", row, do_refine)
        _select(cands, "ret_anchor_strict", "anc", row, do_refine)
        row["torsion_feasible"] = bool(row["tor_all_feasible"] or row["tor_anc_feasible"])
        row["solves_cumulative"] = state["solves"]
        row["refine_exhausted"] = state["exhausted"]
        rows.append(row)
    return rows


def _finite(v) -> float:
    """NaN sorts last rather than poisoning the comparison."""
    v = float(v)
    return v if v == v else float("-inf")


def _best_bridge_pair(st, contacts, spec_windows, typ, cfg, atom_pairs_cap: int = 0):
    """The (i, j) that closes with the FEWEST released residues, ties to best retention.

    Choosing on `k` rather than on peak retention is deliberate: the frontier's question is
    how much the peptide has to move, so the right anchor pair is the cheapest one, not the
    one whose best-case retention is highest at an unaffordable `k`.

    Returns (i, j, k_min) or None when no pair closes at any `k`.
    """
    CA = st["CA"]
    L = int(CA.shape[0])
    tau, cutoff = cfg["tau_max_deg"], cfg["contact_cutoff_A"]
    w = spec_windows[typ]
    best = None
    pairs = [(i, j) for i in range(L) for j in range(i + MIN_BRIDGE_SEP, L)]
    if atom_pairs_cap:
        pairs = pairs[:atom_pairs_cap]
    for i, j in pairs:
        bspec = bridge_spec(typ, i, j, w["ca_lo"], w["ca_hi"])
        for k in range(0, L):
            # Feasibility only: retention is not consulted until the winning pair is known,
            # and computing it for every (pair, k) is what made this the whole job's cost.
            cands = _candidates_at_k(CA, contacts, contacts, st["rec_ca"], bspec, k, tau,
                                     cutoff, CA[i], CA[j], with_retention=False)
            if not cands:
                continue
            # Tie-break on retention only among pairs that already tie on k, so the
            # expensive quantity is computed for a handful of pairs rather than all of them.
            ret = max(retention_ceiling(contacts, st["rec_ca"], CA, c["a"], c["b"],
                                        tau, cutoff)[0] for c in cands)
            cand = (k, -ret, i, j)
            if best is None or cand < best:
                best = cand
            break
    if best is None:
        return None
    k_min, neg_ret, i, j = best
    return i, j, k_min


# --------------------------------------------------------------------------------------
# Per-complex driver
# --------------------------------------------------------------------------------------
def audit_one(row_meta: dict, windows: dict, cfg: dict, acfg: dict) -> list[dict]:
    """Every (chemistry, k) frontier row for one staged complex."""
    path = Path(row_meta["path"])
    st = load_staged(path)
    if st is None:
        return []
    CA, N, C = st["CA"], st["N"], st["C"]
    L = int(CA.shape[0])
    contacts = ca_contact_set(CA, st["rec_ca"], cfg["contact_cutoff_A"])
    if not contacts:
        return []

    # Anchors come from buried surface area, which needs the all-atom complex -- a second
    # read of the same file.  Kept separate from `load_staged` rather than folded into it
    # so the geometry path stays byte-identical to the Milestone 1.5 audit.
    cx = load_complex(path, example_id=str(row_meta.get("example_id")))
    n_anchors = int(acfg.get("n_anchors", 3))
    anchors = anchor_residues(cx, n_anchors) if cx is not None else []
    anchor_contacts = {(p, r) for (p, r) in contacts if p in set(anchors)}
    if not anchor_contacts:
        # Every anchor is a non-contacting residue.  Fall back to the full contact set so
        # the anchor columns stay defined, and say so in the row rather than emitting a
        # silent NaN that would read as "measured and zero".
        anchor_contacts = contacts

    base = {
        "example_id": row_meta.get("example_id"),
        "pdb_id": row_meta.get("pdb_id"),
        "path": str(path),
        "peptide_length": L,
        "receptor_length": int(st["rec_ca"].shape[0]),
        "nc_gap_A": float(np.linalg.norm(N[0] - C[L - 1])),
        "term_ca_gap_A": float(np.linalg.norm(CA[0] - CA[L - 1])),
        "n_contacts_ca10": len(contacts),
        "n_anchor_contacts": len(anchor_contacts),
        "anchors": ",".join(str(a) for a in anchors),
        "anchor_fallback_to_all": int(anchor_contacts is contacts),
    }

    refine = bool((cfg.get("refine") or {}).get("enabled", True))
    max_solves = int(acfg.get("max_solves_per_spec", 60))
    wanted = set(acfg.get("chemistries") or ("mainchain", "disulfide", "isopeptide"))
    out: list[dict] = []

    # ---- head-to-tail.  This is the chemistry the brief's validity control is stated in:
    # the three never-closed SDEdit targets are named by their N-to-C terminal gap.
    spec = mainchain_spec(L, cfg["mainchain_tol_A"])
    for r in [] if "mainchain" not in wanted else frontier_for_spec(st, contacts, anchor_contacts, spec, cfg, N[0], C[L - 1],
                               refine=refine, max_solves=max_solves):
        out.append({**base, "chemistry": "mainchain", "bridge_i": 0, "bridge_j": L - 1, **r})

    # ---- bridged chemistries, reported as secondary.  The anchor pair is chosen ONCE, on
    # cheapest-closure, and then held fixed across the frontier; sweeping (i, j) jointly
    # with k would report the envelope of many different molecules as one target's frontier.
    if acfg.get("bridged", True):
        for typ in ("disulfide", "isopeptide"):
            if typ not in windows or typ not in wanted:
                continue
            picked = _best_bridge_pair(st, contacts, windows, typ, cfg)
            if picked is None:
                out.append({**base, "chemistry": typ, "bridge_i": -1, "bridge_j": -1,
                            "k": -1, "window_len": -1, "n_windows_at_k": 0,
                            "n_analytic_feasible_at_k": 0, "analytic_feasible": False,
                            "torsion_feasible": False, "no_hostable_pair": 1})
                continue
            i, j, _ = picked
            bspec = bridge_spec(typ, i, j, windows[typ]["ca_lo"], windows[typ]["ca_hi"])
            for r in frontier_for_spec(st, contacts, anchor_contacts, bspec, cfg,
                                       CA[i], CA[j], refine=refine, max_solves=max_solves):
                out.append({**base, "chemistry": typ, "bridge_i": i, "bridge_j": j,
                            "no_hostable_pair": 0, **r})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", required=True,
                    help="bridge_windows.json from the Milestone 1.5 calibration")
    ap.add_argument("--out", required=True, help="JSONL, one row per (complex, chemistry, k)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-solves", type=int, default=0,
                    help="Override lnr_feasibility.max_solves_per_spec. Smoke runs only: a "
                         "low budget makes levels report INFEASIBLE that a full budget "
                         "would close, so a capped run must never be reported as a result.")
    ap.add_argument("--chemistries", default="",
                    help="Space-separated subset of mainchain/disulfide/isopeptide. "
                         "Empty = all. Smoke runs only, for the same reason.")
    args = ap.parse_args()

    full = precheck_config.load(args.config)
    cfg = full["geometry"]
    acfg = dict(full.get("lnr_feasibility") or {})
    if args.max_solves:
        acfg["max_solves_per_spec"] = args.max_solves
    if args.chemistries:
        acfg["chemistries"] = args.chemistries.split()
    windows = json.loads(Path(args.windows).read_text())["windows"]

    meta = acfg.get("metadata") or full["audit"]["sets"]["lnr"]["metadata"]
    df = pq.read_table(meta).to_pandas()
    if args.limit:
        df = df.head(args.limit)
    # Stable stride partition: a relaunch with a different shard count re-partitions, but a
    # relaunch of the SAME shard owns the same rows, which is what resumability needs.
    df = df.iloc[args.shard::args.n_shards]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip complexes already written.  A frontier is minutes, not seconds, and a
    # wall-clock kill should cost the tail rather than the whole shard.
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["example_id"])
            except Exception:
                raise SystemExit(
                    f"{out_path}: unparseable row while resuming. Refusing to continue -- "
                    f"a corrupt shard silently becomes missing data. Delete it and rerun.")

    t0 = time.perf_counter()
    n_done = 0
    with out_path.open("a") as fh:
        for _, r in df.iterrows():
            if str(r.get("example_id")) in done:
                continue
            for row in audit_one(r.to_dict(), windows, cfg, acfg):
                fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            n_done += 1
            print(f"[{n_done}/{len(df) - len(done)}] {r.get('example_id')} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"wrote {out_path} ({n_done} complexes, {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
