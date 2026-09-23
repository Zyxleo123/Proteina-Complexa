"""Milestone 1.5 -- geometric ceiling audit.

Answers one question before any pose-decoy pair is built: at a given terminal gap, how
much of a peptide's native bound interface can survive ring closure AT ALL?

Without that number a measured contact retention is uninterpretable.  A retention of 0.40
is a failure if the ceiling is 0.95 and a near-perfect result if the ceiling is 0.45, and
nothing measured so far distinguishes those two readings.

Three sub-commands:

  calibrate  from CPSea natives, derive the CA-CA / CB-CB distance windows that a real
             disulfide or isopeptide bridge actually occupies.  Asserting those windows
             from textbook chemistry would make every downstream ceiling depend on a
             guess; measuring them on 1,162 real disulfides does not.
  audit      per-complex ceiling scan over a staged metadata parquet (LNR or PepBench).
  profile    the section-3 bridge-span profile: every (i, j) CB-CB distance that could
             host a bridge, not just the terminal gap.

Runs in `.venv` (numpy / scipy / pandas / pyarrow).  No OpenMM, no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from build_lnr_metadata import parse_pdb_chains, virtual_cb  # noqa: E402
from script_utils import m15_config  # noqa: E402
from script_utils.kinematic_ceiling import (  # noqa: E402
    CA_CONTACT_CUTOFF_A, HEAVY_CONTACT_CUTOFF_A, TAU_MAX_DEG, bridge_spec, ca_contact_set,
    feasible_torsion, mainchain_spec, rank_feasible_windows, refine_with_torsion,
    scan_windows,
)

BACKBONE = ("N", "CA", "C")
MIN_BRIDGE_SEP = 3   # |i - j| >= 3; closer pairs are trivially adjacent, not a bridge


# ---------------------------------------------------------------------------- loading
def load_staged(path: Path) -> dict | None:
    """Peptide backbone + CB and receptor CA from a staged chain-A/chain-B complex."""
    residues, _ = parse_pdb_chains(path, {"A", "B"})
    pep, rec = residues["B"], residues["A"]
    if len(pep) < 4 or len(rec) < 3:
        return None
    keys = list(pep)
    if any(not set(BACKBONE) <= set(pep[k]) for k in keys):
        return None

    N = np.array([pep[k]["N"][1] for k in keys], dtype=float)
    CA = np.array([pep[k]["CA"][1] for k in keys], dtype=float)
    C = np.array([pep[k]["C"][1] for k in keys], dtype=float)
    CB = np.array([virtual_cb(pep[k]) for k in keys], dtype=float)

    rec_keys = [k for k in rec if "CA" in rec[k]]
    rec_ca = np.array([rec[k]["CA"][1] for k in rec_keys], dtype=float)

    pep_heavy = [np.array([xyz for _, (_, xyz) in pep[k].items()], dtype=float) for k in keys]
    rec_heavy = np.concatenate(
        [np.array([xyz for _, (_, xyz) in rec[k].items()], dtype=float) for k in rec_keys])

    return {
        "N": N, "CA": CA, "C": C, "CB": CB,
        "resnames": [k[2] for k in keys],
        "resseq": [int(k[0]) for k in keys],
        "rec_ca": rec_ca,
        "rec_resseq": [int(k[0]) for k in rec_keys],
        "pep_heavy": pep_heavy,
        "rec_heavy": rec_heavy,
    }


def n_heavy_contacts(pep_heavy: list[np.ndarray], rec_heavy: np.ndarray,
                     cutoff: float = HEAVY_CONTACT_CUTOFF_A) -> int:
    """Peptide residues with any heavy atom within `cutoff` of the receptor."""
    return sum(1 for atoms in pep_heavy
               if np.min(np.linalg.norm(atoms[:, None, :] - rec_heavy[None, :, :], axis=-1)) < cutoff)


# ------------------------------------------------------------------------ calibration
def calibrate(args) -> None:
    """Measure what distance a real bridge occupies, rather than asserting one."""
    from proteinfoundation.cyclization.constants import DISULFIDE, ISOPEPTIDE, MAINCHAIN
    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    cal = m15_config.load(args.config)["calibration"]
    args.metadata = args.metadata or m15_config.resolve(cal["metadata"])
    args.max_examples = int(cal["max_examples"])
    args.pad_A = float(cal["pad_A"])
    args.seed = int(cal["seed"])

    name_of = {MAINCHAIN: "mainchain", DISULFIDE: "disulfide", ISOPEPTIDE: "isopeptide"}
    df = pq.read_table(args.metadata, columns=["example_id", "path", "peptide_length",
                                               "cyclization_type", "binder_chain_id"]).to_pandas()
    rng = np.random.default_rng(args.seed)
    if len(df) > args.max_examples:
        df = df.iloc[rng.choice(len(df), args.max_examples, replace=False)]

    rows, n_fail = [], 0
    for rec in df.itertuples(index=False):
        try:
            st = load_staged(Path(rec.path))
        except Exception:
            n_fail += 1
            continue
        if st is None:
            n_fail += 1
            continue
        lab = infer_cyclization_label(
            str(rec.path), st["resseq"], binder_length_hint=int(rec.peptide_length),
            binder_chain_id=str(rec.binder_chain_id or "B"),
            cyclization_type_hint=str(rec.cyclization_type),
        )
        if not lab.get("has_cyclization") or int(lab.get("type", -1)) not in name_of:
            continue
        i, j = int(lab["i"]), int(lab["j"])
        L = len(st["CA"])
        if not (0 <= i < L and 0 <= j < L and i != j):
            continue
        rows.append({
            "example_id": rec.example_id,
            "type": name_of[int(lab["type"])],
            "i": i, "j": j, "sep": abs(j - i), "peptide_length": L,
            "ca_ca_A": float(np.linalg.norm(st["CA"][i] - st["CA"][j])),
            "cb_cb_A": float(np.linalg.norm(st["CB"][i] - st["CB"][j])),
            "nc_gap_A": float(np.linalg.norm(st["N"][0] - st["C"][L - 1])),
        })

    if not rows:
        raise SystemExit("calibration found no CONECT-resolved cyclizations -- refusing to "
                         "write windows derived from nothing")
    obs_df = pd.DataFrame(rows)
    out: dict[str, object] = {
        "source_metadata": str(args.metadata),
        "n_sampled": int(len(df)),
        "n_unreadable": int(n_fail),
        "n_labelled": int(len(obs_df)),
        "pad_A": float(args.pad_A),
        "windows": {},
        "observed": {},
    }
    for typ, grp in obs_df.groupby("type"):
        q = lambda col, p: float(np.percentile(grp[col], p))  # noqa: E731
        obs = {
            "n": int(len(grp)),
            "ca_ca": {p: q("ca_ca_A", p) for p in (1, 5, 25, 50, 75, 95, 99)},
            "cb_cb": {p: q("cb_cb_A", p) for p in (1, 5, 25, 50, 75, 95, 99)},
            "sep_median": float(grp["sep"].median()),
        }
        out["observed"][typ] = obs
        # [p1, p99] plus a pad: the window must admit every real bridge, because a window
        # that excludes real chemistry would understate the ceiling -- the one direction a
        # ceiling must never err in.
        out["windows"][typ] = {
            "ca_lo": max(0.0, q("ca_ca_A", 1) - args.pad_A),
            "ca_hi": q("ca_ca_A", 99) + args.pad_A,
            "cb_lo": max(0.0, q("cb_cb_A", 1) - args.pad_A),
            "cb_hi": q("cb_cb_A", 99) + args.pad_A,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    obs_df.to_parquet(args.out.with_suffix(".rows.parquet"), index=False)
    print(json.dumps(out["windows"], indent=2))
    print(f"[calibrate] {len(obs_df)} labelled bridges from {len(df)} sampled -> {args.out}")


# ------------------------------------------------------------------------- the audit
def bridge_profile(CB: np.ndarray, windows: dict) -> dict:
    """Section 3: the distribution over every (i, j) CB-CB distance that could host a bridge.

    The chain-terminal N-to-C gap is the wrong discriminator for a bridged macrocycle --
    the bonded atoms are interior side chains and the termini are free tails.  This is the
    feature that says whether a linear peptide is cyclizable at all, and where.
    """
    L = len(CB)
    pairs = [(float(np.linalg.norm(CB[i] - CB[j])), i, j)
             for i in range(L) for j in range(i + MIN_BRIDGE_SEP, L)]
    if not pairs:
        return {"n_bridge_pairs": 0}
    d = np.array([p[0] for p in pairs])
    out = {
        "n_bridge_pairs": len(pairs),
        "bridge_span_min_A": float(d.min()),
        "bridge_span_p25_A": float(np.percentile(d, 25)),
        "bridge_span_med_A": float(np.median(d)),
        "bridge_span_p75_A": float(np.percentile(d, 75)),
        "bridge_span_max_A": float(d.max()),
    }
    for typ, w in windows.items():
        if typ == "mainchain":
            continue
        lo, hi = w["cb_lo"], w["cb_hi"]
        inside = [(dd, i, j) for dd, i, j in pairs if lo <= dd <= hi]
        out[f"n_pairs_in_{typ}_window"] = len(inside)
        mid = 0.5 * (lo + hi)
        best = min(pairs, key=lambda t: abs(t[0] - mid))
        out[f"best_{typ}_cb_A"] = best[0]
        out[f"best_{typ}_i"] = best[1]
        out[f"best_{typ}_j"] = best[2]
        out[f"{typ}_hostable"] = int(bool(inside))
    return out


def _refine(res, st, contacts, rec_ca, spec, cfg, tau, atom_i, atom_j):
    """Confirm the analytic ceiling in torsion space.

    The analytic pass over-licenses at the boundary and does so by a lot: on a real
    10-mer it calls closure feasible with 1 free residue where the backbone needs 5,
    because a short flank's reachable set is a cone about the incoming chain direction and
    not the sphere the annulus model allows.  Left unrefined, the reported ceiling is too
    high and a genuine retention deficit hides behind it.

    The analytic pass is still the pre-filter, because it is a true upper bound -- anything
    it rejects is really infeasible -- so this can only ever remove windows.
    """
    r = cfg.get("refine") or {}
    if not r.get("enabled", True) or not res.feasible_any:
        return res
    cands = rank_feasible_windows(st["CA"], contacts, rec_ca, spec, tau,
                                  cfg["contact_cutoff_A"],
                                  atom_i_xyz=atom_i, atom_j_xyz=atom_j)
    return refine_with_torsion(
        res, cands, st["N"], st["CA"], st["C"], spec,
        max_solves=int(r.get("max_solves", 40)),
        n_restarts=int(r.get("n_restarts", 6)),
        seed=int(r.get("seed", 0)),
        tol_A=float(r.get("tol_A", cfg["mainchain_tol_A"])),
        max_seconds=float(r.get("max_seconds", 15.0)),
    )


def audit_one(st: dict, windows: dict, cfg: dict) -> dict:
    """Ceiling scan for one complex, across every closure chemistry."""
    N, CA, C, CB = st["N"], st["CA"], st["C"], st["CB"]
    L = len(CA)
    rec_ca = st["rec_ca"]
    contacts = ca_contact_set(CA, rec_ca, cfg["contact_cutoff_A"])
    tau = cfg["tau_max_deg"]

    row: dict[str, object] = {
        "peptide_length": L,
        "receptor_length": len(rec_ca),
        "nc_gap_A": float(np.linalg.norm(N[0] - C[L - 1])),
        "term_ca_gap_A": float(np.linalg.norm(CA[0] - CA[L - 1])),
        "n_contacts_ca10": len(contacts),
        "n_interface_residues_heavy45": n_heavy_contacts(st["pep_heavy"], st["rec_heavy"],
                                                         cfg["heavy_cutoff_A"]),
    }
    row.update(bridge_profile(CB, windows))

    # ---- head-to-tail: the endpoints are fixed by the chemistry, so only the window varies
    spec = mainchain_spec(L, cfg["mainchain_tol_A"])
    # N(0) and C(L-1) are passed explicitly: a held window holds its backbone, so an
    # in-window terminal atom is pinned rather than free within its bond length.
    res = scan_windows(CA, contacts, rec_ca, spec, tau, cfg["contact_cutoff_A"],
                       atom_i_xyz=N[0], atom_j_xyz=C[L - 1])
    res = _refine(res, st, contacts, rec_ca, spec, cfg, tau, N[0], C[L - 1])
    for k, v in res.__dict__.items():
        if k != "spec_name":
            row[f"mainchain_{k}"] = v

    # Same scan at a realistic rather than extremal backbone extension.  If the headline
    # ceiling only survives at tau = 150 deg it is an artifact of the idealisation, and the
    # report has to say so rather than quote the permissive number alone.
    tau_s = cfg.get("tau_sensitivity_deg")
    if tau_s:
        res_s = scan_windows(CA, contacts, rec_ca, spec, tau_s, cfg["contact_cutoff_A"],
                             atom_i_xyz=N[0], atom_j_xyz=C[L - 1])
        row["mainchain_ceiling_strict_tau_sens"] = res_s.ceiling_strict
        row["mainchain_max_feasible_window_len_tau_sens"] = res_s.max_feasible_window_len
        row["mainchain_feasible_any_tau_sens"] = res_s.feasible_any

    # ---- bridged: the anchors are NOT fixed, so the ceiling is a max over (i, j) too.
    # Reporting only the metadata's "best" pair would answer a narrower question than the
    # one that matters -- whether SOME bridge preserves the interface, not whether the
    # closest-to-ideal one does.
    for typ in ("disulfide", "isopeptide"):
        w = windows.get(typ)
        if w is None:
            continue
        best = None
        for i in range(L):
            for j in range(i + MIN_BRIDGE_SEP, L):
                bspec = bridge_spec(typ, i, j, w["ca_lo"], w["ca_hi"])
                # The bridged spec is already stated CA-to-CA, so the CA positions ARE
                # the bonded-atom positions and no extra pinning is needed.
                r = scan_windows(CA, contacts, rec_ca, bspec, tau, cfg["contact_cutoff_A"],
                                 atom_i_xyz=CA[i], atom_j_xyz=CA[j])
                if not r.feasible_any:
                    continue
                if best is None or r.ceiling_strict > best[0].ceiling_strict:
                    best = (r, i, j)
        if best is None:
            row[f"{typ}_feasible_any"] = False
            row[f"{typ}_ceiling_strict"] = float("nan")
            row[f"{typ}_ceiling_permissive"] = float("nan")
            continue
        r, i, j = best
        # A bridged ceiling of 1.0 means some (i, j) pair is ALREADY at bond distance, so
        # nothing has to move -- but only if those two residues can carry the chemistry.
        # Record whether they natively do: a ceiling that silently assumes free mutation of
        # the anchors is a different claim from one that does not, and isopeptide failure
        # is about half anchor identity rather than geometry.
        need = {"disulfide": ({"CYS"}, {"CYS"}),
                "isopeptide": ({"LYS"}, {"ASP", "GLU", "ASN", "GLN"})}[typ]
        names = st["resnames"]
        row[f"{typ}_anchor_native_compatible"] = int(
            (names[i] in need[0] and names[j] in need[1])
            or (names[j] in need[0] and names[i] in need[1]))
        bspec = bridge_spec(typ, i, j, w["ca_lo"], w["ca_hi"])
        r = _refine(r, st, contacts, rec_ca, bspec, cfg, tau, CA[i], CA[j])
        for k, v in r.__dict__.items():
            if k != "spec_name":
                row[f"{typ}_{k}"] = v
        row[f"{typ}_anchor_i"] = i
        row[f"{typ}_anchor_j"] = j
    return row


def audit(args) -> None:
    cfg = m15_config.load(args.config)["geometry"]
    windows = json.loads(Path(args.windows).read_text())["windows"]

    df = pq.read_table(args.metadata).to_pandas()
    if args.source_filter and "dataset_source" in df.columns:
        df = df[df["dataset_source"] == args.source_filter]
    if args.max_examples and len(df) > args.max_examples:
        rng = np.random.default_rng(args.seed)
        df = df.iloc[np.sort(rng.choice(len(df), args.max_examples, replace=False))]
    df = df.reset_index(drop=True)
    shard = df.iloc[args.shard::args.num_shards]
    print(f"[audit] {args.label}: shard {args.shard}/{args.num_shards} -> {len(shard)} of {len(df)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip = 0, 0
    with args.out.open("w") as fh:
        for rec in shard.itertuples(index=False):
            try:
                st = load_staged(Path(rec.path))
            except Exception as exc:  # noqa: BLE001
                fh.write(json.dumps({"example_id": rec.example_id, "status": "unreadable",
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                n_skip += 1
                continue
            if st is None:
                fh.write(json.dumps({"example_id": rec.example_id,
                                     "status": "incomplete_backbone"}) + "\n")
                n_skip += 1
                continue
            row = audit_one(st, windows, cfg)
            row.update({
                "example_id": rec.example_id, "status": "ok", "label": args.label,
                "cluster_id": getattr(rec, "cluster_id", None),
                "meta_nc_gap_A": float(getattr(rec, "nc_gap_angstrom", float("nan"))),
            })
            if args.validate_frac > 0:
                row.update(validate_row(st, row, cfg, args))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_ok += 1
    print(f"[audit] wrote {n_ok} rows ({n_skip} skipped) -> {args.out}")


def validate_row(st: dict, row: dict, cfg: dict, args) -> dict:
    """Spot-check the analytic verdict against a real torsion-space solve.

    Sampled at the decision boundary on purpose: agreement on obviously-open and
    obviously-closed cases proves nothing, and the boundary is where a ceiling is set.
    """
    rng = np.random.default_rng(abs(hash(row["example_id"])) % (2 ** 31))
    if rng.random() > args.validate_frac:
        return {}
    L = row["peptide_length"]
    edge = int(row.get("mainchain_max_feasible_window_len", 0))
    if edge <= 0 or edge >= L:
        return {"validator_status": "no_boundary"}
    spec = mainchain_spec(L, cfg["mainchain_tol_A"])
    out: dict[str, object] = {"validator_status": "ok"}
    # One window just inside the analytic boundary, one just outside it.
    for tag, wlen in (("inside", edge), ("outside", min(L, edge + 1))):
        a = max(0, (L - wlen) // 2)
        b = min(L - 1, a + wlen - 1)
        ok_t, resid = feasible_torsion(st["N"], st["CA"], st["C"], spec, a, b,
                                       n_restarts=args.validate_restarts, seed=args.seed,
                                       tol_A=cfg["mainchain_tol_A"])
        from script_utils.kinematic_ceiling import feasible_analytic
        ok_a, _, _ = feasible_analytic(st["CA"], spec, a, b, cfg["tau_max_deg"],
                                       atom_i_xyz=st["N"][0], atom_j_xyz=st["C"][L - 1])
        out[f"validator_{tag}_window"] = [a, b]
        out[f"validator_{tag}_analytic"] = bool(ok_a)
        out[f"validator_{tag}_torsion"] = bool(ok_t)
        out[f"validator_{tag}_residual_A"] = float(resid)
        # The bound must be permissive.  A case the solver closes but the analytic model
        # rejects means the ceiling is UNDERstated, which would let a real deficit hide.
        out[f"validator_{tag}_under_licensed"] = bool(ok_t and not ok_a)
    return out


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="derive bridge distance windows from CPSea natives")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--metadata", type=Path, default=None,
                   help="overrides calibration.metadata from the config")
    c.set_defaults(func=calibrate)

    a = sub.add_parser("audit", help="per-complex ceiling scan")
    a.add_argument("--metadata", type=Path, required=True)
    a.add_argument("--windows", type=Path, required=True)
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--label", default="lnr")
    a.add_argument("--source-filter", default="")
    a.add_argument("--max-examples", type=int, default=0)
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--num-shards", type=int, default=1)
    a.add_argument("--validate-frac", type=float, default=0.0)
    a.add_argument("--validate-restarts", type=int, default=6)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=audit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
