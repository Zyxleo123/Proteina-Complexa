#!/usr/bin/env python3
"""Score RFD3 validation generations against the native peptides.

WHY. The in-loop design metrics score the whole predicted structure -- 550 receptor atoms
against ~64 peptide atoms -- for *physical plausibility*: clash-free, connected, chemically
valid. None of their 29 columns says whether the peptide landed in the right place. So they
cannot answer the only question that matters here, and they already produced a result that
cannot be interpreted without this: four of five degraded significantly over 12 epochs, which
is either harm or an irrelevance depending on what happened to binding accuracy.

This reads the dumped structures and measures accuracy directly. It is offline CPU work and
retroactive: every epoch's structures are on disk, so no retraining is needed.

FOUR THINGS THAT SILENTLY CORRUPT THIS COMPARISON, all measured on real output:

1.  **Residue ids are off by one.** The generated CIF is 1-based (chain A 1..74, B 1..6); our
    native PDBs are 0-based (0..73, 0..5). Matching on `res_id` directly aligns the wrong
    residues and quietly scores a minority of atoms.
2.  **Sidechain atoms are renamed to virtual atoms.** The output carries `V0, V1, V2...`
    where the native has `CG, CD, CE...` (the trainer sets `cleanup_virtual_atoms: False`),
    so sidechains cannot be matched by name at all. Only `N, CA, C, O, CB` survive in both,
    which is why scoring is backbone-level. Residue *names* do match, so sequence is
    checkable.
3.  **The whole structure is recentred.** The receptor is rigidly fixed by the contig spec,
    yet its raw coordinate deviation from the native is a near-uniform 1.88 A -- RFD3 sets
    the origin to the COM of the fixed motif. Superposing on the receptor drops that to
    0.019 A. Without superposition a peptide RMSD comes out ~12 A when ~10 A of it is the
    global offset, and the error is invisible because it looks like a plausible bad number.
4.  **Atom ORDER differs** between generated and native, so no element-wise comparison is
    valid even when the counts match exactly (they do: 550/64 in both).

`receptor_rmsd_after_fit` is reported for every row as a QC gate. It must be ~0; anything
larger means the receptor was not actually held fixed and that row's peptide RMSD is
meaningless rather than merely bad.

METRIC NAMES MATCH `lpgen_dpb_score.py` so the arms can be compared at all:
  `raw_ca_rmsd`            CA RMSD in the NATIVE FRAME -- placement and conformation together.
                           For RFD3 this requires the receptor superposition described above:
                           the output is recentred, so its "raw" frame is not the native one.
                           For a docking arm the poses are already in that frame.
  `superposed_ca_rmsd`     Kabsch on the PEPTIDE ONLY -- conformation with placement factored
                           out. raw minus superposed separates "wrong pose" from "wrong fold".
  `interface_raw_ca_rmsd`  raw CA RMSD over the native-contacting residues only.
A hit is `interface_raw_ca_rmsd <= 2.0 A`, the same threshold the other arm uses.

WHAT IS REPORTED. Per generated model: peptide CA and backbone RMSD after receptor
superposition; INTERFACE RMSD over the peptide residues that contact the receptor in the
native (this repository's own finding is that interface RMSD supersedes whole-peptide RMSD on
half of a comparable set); receptor-peptide clash count at 2.0 A (calibrated against native
crystals -- 2.7 A rejects 43 of 60 natives, 2.0 A keeps 59); and sequence identity against
the native, which is the control: the `fullseq` task is *given* the sequence, so anything
below 1.0 there is a bug, not a result.

Run (see scripts/lpgen_score_val.sbatch):
    .venv/bin/python script_utils/lpgen_score_val_structures.py \\
        --pred-root $ZFS/lpgen/runs/lp_finetune_v1/val_structures/predictions \\
        --panel     $ZFS/LPData/preprocessed/metadata_lpgen/lpgen_val_panel.parquet \\
        --out       $ZFS/lpgen/runs/lp_finetune_v1/val_scores.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

#: Atoms whose names survive in both the generated CIF and the native PDB. Everything else
#: in the output sidechain is a virtual atom (`V0`, `V1`, ...).
BACKBONE = ("N", "CA", "C", "O", "CB")

#: Native peptide residues within this distance of any receptor heavy atom define the
#: interface. 5.0 A is the usual contact shell for an interface-RMSD definition.
INTERFACE_CUTOFF_A = 5.0

#: Clash distance, calibrated against native crystals in this project: 2.7 A rejects 43 of
#: 60 natives, 2.0 A keeps 59. Scoring generations at 2.7 A would call most real complexes
#: clashing.
CLASH_A = 2.0

#: Generated res_id minus this equals the native res_id. Measured, see the module docstring.
RES_ID_OFFSET = -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pred-root", required=True,
                   help="val_structures/predictions, containing epoch_NNN/<dataset>/*.cif.gz")
    p.add_argument("--panel", required=True, help="lpgen_val_panel.parquet (for native paths)")
    p.add_argument("--out", required=True, help="Per-model CSV to write.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=-1, help="Score only N files (smoke).")
    p.add_argument("--contigs", default=None,
                   help="lpgen_val_contigs.json. Supplying it splits sequence identity into "
                        "the positions whose identity was GIVEN (a copy check that must read "
                        "1.0) and the positions that were HIDDEN (the only real measurement). "
                        "Without it only the pooled, mostly-copied number is available.")
    return p.parse_args()


def read_native(path: str) -> dict:
    """Backbone atoms of a native complex, keyed by (chain, res_id, atom_name)."""
    from biotite.structure.io.pdb import PDBFile

    arr = PDBFile.read(path).get_structure(model=1)
    return arr


def read_generated(path: Path):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    with gzip.open(path, "rt") as fh:
        return get_structure(CIFFile.read(fh), model=1)


def backbone_index(arr, offset: int) -> dict:
    return {
        (str(c), int(r) + offset, str(a)): i
        for i, (c, r, a) in enumerate(zip(arr.chain_id, arr.res_id, arr.atom_name))
        if str(a) in BACKBONE
    }


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Rotation taking centred ``mobile`` onto centred ``target``, plus both centroids."""
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, mc, tc


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def score_one(task: tuple) -> dict:
    """Score one generated model. Returns a row dict; never raises."""
    (pred_path, native_path, epoch, dataset, example_id, model_id, pep_chain,
     released_ids) = task
    out = {
        "epoch": epoch, "dataset": dataset, "example_id": example_id,
        "model": model_id, "task": "partialseq" if "partialseq" in example_id else "fullseq",
        "status": "ok", "reason": "",
    }
    try:
        gen = read_generated(Path(pred_path))
        nat = read_native(native_path)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "unreadable"
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    gi = backbone_index(gen, RES_ID_OFFSET)
    ni = backbone_index(nat, 0)
    common = sorted(set(gi) & set(ni))
    rec = [k for k in common if k[0] != pep_chain]
    pep = [k for k in common if k[0] == pep_chain]
    out["n_matched_backbone"] = len(common)
    out["n_pep_backbone"] = len(pep)

    if len(rec) < 12 or not pep:
        out["status"] = "insufficient_match"
        out["reason"] = f"{len(rec)} receptor and {len(pep)} peptide backbone atoms matched"
        return out

    gr = gen.coord[[gi[k] for k in rec]]
    nr = nat.coord[[ni[k] for k in rec]]
    out["receptor_rmsd_raw"] = rmsd(gr, nr)

    rot, mc, tc = kabsch(gr, nr)
    fit = lambda x: (x - mc) @ rot.T + tc  # noqa: E731
    # QC gate: the receptor is supposed to be rigidly fixed, so this must be ~0.
    out["receptor_rmsd_after_fit"] = rmsd(fit(gr), nr)

    gp = fit(gen.coord[[gi[k] for k in pep]])
    npep = nat.coord[[ni[k] for k in pep]]
    out["pep_bb_rmsd"] = rmsd(gp, npep)
    ca = [i for i, k in enumerate(pep) if k[2] == "CA"]
    if ca:
        # Native frame: placement + conformation.
        out["raw_ca_rmsd"] = rmsd(gp[ca], npep[ca])
        # Peptide-only Kabsch: conformation alone. The difference between the two is the
        # placement error, which is what tells a wrong pose from a wrong fold.
        if len(ca) >= 3:
            r2, m2, t2 = kabsch(gp[ca], npep[ca])
            out["superposed_ca_rmsd"] = rmsd((gp[ca] - m2) @ r2.T + t2, npep[ca])

    # Interface: native peptide residues contacting the receptor, scored on those residues
    # only. Defined on the NATIVE so it is the same residue set at every epoch.
    nat_rec = nat[np.asarray(nat.chain_id, dtype=str) != pep_chain]
    iface_res = set()
    for res_id in {k[1] for k in pep}:
        sel = (np.asarray(nat.chain_id, dtype=str) == pep_chain) & (nat.res_id == res_id)
        if not sel.any():
            continue
        d = np.linalg.norm(
            nat.coord[sel][:, None, :] - nat_rec.coord[None, :, :], axis=-1
        )
        if d.min() <= INTERFACE_CUTOFF_A:
            iface_res.add(res_id)
    out["n_interface_res"] = len(iface_res)
    if iface_res:
        sub = [i for i, k in enumerate(pep) if k[1] in iface_res]
        out["pep_iface_rmsd"] = rmsd(gp[sub], npep[sub])
        sub_ca = [i for i, k in enumerate(pep) if k[1] in iface_res and k[2] == "CA"]
        if sub_ca:
            out["interface_raw_ca_rmsd"] = rmsd(gp[sub_ca], npep[sub_ca])

    # Receptor-peptide clash in the GENERATED pose, in its own frame (no fit needed: both
    # chains come from the same file).
    gen_pep_all = gen[np.asarray(gen.chain_id, dtype=str) == pep_chain]
    gen_rec_all = gen[np.asarray(gen.chain_id, dtype=str) != pep_chain]
    if gen_pep_all.array_length() and gen_rec_all.array_length():
        d = np.linalg.norm(
            gen_pep_all.coord[:, None, :] - gen_rec_all.coord[None, :, :], axis=-1
        )
        out["n_clash_2A"] = int((d < CLASH_A).sum())
        out["min_rec_pep_dist"] = float(d.min())
        # A peptide that drifted off the protein entirely is a distinct failure from one
        # that clashes, and the two look the same in an RMSD.
        out["n_contacts_5A"] = int((d < 5.0).sum())

    # SEQUENCE IDENTITY, SPLIT. The pooled number is ~99% a copy of an input: every position
    # not in `released_ids` had its identity handed to the model as a one-hot input feature,
    # so scoring it measures the wiring, not the model. Reporting the two separately is the
    # difference between a tautology and a measurement -- pooled identity read 1.0000 on the
    # fullseq task while accuracy on the single hidden residue sat at chance.
    gpep_res = {
        int(r) + RES_ID_OFFSET: str(nm)
        for c, r, nm in zip(gen.chain_id, gen.res_id, gen.res_name)
        if str(c) == pep_chain
    }
    npep_res = {
        int(r): str(nm)
        for c, r, nm in zip(nat.chain_id, nat.res_id, nat.res_name)
        if str(c) == pep_chain
    }
    shared = set(gpep_res) & set(npep_res)
    if shared:
        out["seq_identity"] = sum(gpep_res[k] == npep_res[k] for k in shared) / len(shared)
        out["n_seq_compared"] = len(shared)

        hidden = shared & set(released_ids)
        given = shared - set(released_ids)
        # Only emitted when the split is actually known, so an absent column reads as
        # "not measured" rather than silently as zero.
        if hidden:
            out["n_hidden"] = len(hidden)
            out["seq_identity_hidden"] = (
                sum(gpep_res[k] == npep_res[k] for k in hidden) / len(hidden))
        if given:
            out["n_given"] = len(given)
            out["seq_identity_given"] = (
                sum(gpep_res[k] == npep_res[k] for k in given) / len(given))
        # A released position the generated model never produced means the hidden-position
        # metric would be computed over the wrong set, so flag it instead of averaging it in.
        missing = set(released_ids) - shared
        if missing:
            out["reason"] = (out["reason"] + "; " if out["reason"] else "") + \
                f"released ids absent from the comparison: {sorted(missing)}"
    return out


def main() -> int:
    args = parse_args()
    panel = pd.read_parquet(args.panel)
    native_by_id = dict(zip(panel.example_id, panel.path))
    pep_by_id = dict(zip(panel.example_id, panel.peptide_chain_id))

    # Which peptide positions were HIDDEN per spec. `select_unfixed_sequence` is a
    # comma-separated list of chain+res_id tokens, e.g. "B2,B7"; absent for the fullseq
    # task, where nothing was hidden.
    released_by_spec: dict = {}
    if args.contigs:
        specs = json.loads(Path(args.contigs).read_text())
        for key, spec in specs.items():
            sel = spec.get("select_unfixed_sequence")
            if not sel:
                continue
            ids = set()
            for tok in str(sel).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                digits = "".join(ch for ch in tok if ch.isdigit() or ch == "-")
                if digits:
                    ids.add(int(digits))
            if ids:
                released_by_spec[key] = frozenset(ids)
        logger.info(f"contigs: {len(released_by_spec)} of {len(specs)} specs hide "
                    f"sequence positions "
                    f"({sum(len(v) for v in released_by_spec.values())} positions total)")
        if not released_by_spec:
            logger.warning("no spec hides any position: the hidden-position column will be "
                           "empty and only the copy check is available")
    else:
        logger.warning("no --contigs: sequence identity will be POOLED over given and hidden "
                       "positions alike, which is ~99% a copy of the input and must not be "
                       "quoted as a capability")

    root = Path(args.pred_root)
    pat = re.compile(r"^(?P<eid>.+?)__(?P<task>fullseq|partialseq)_model_(?P<model>\d+)$")
    tasks, skipped = [], []
    for cif in sorted(root.glob("epoch_*/*/*.cif.gz")):
        epoch = int(cif.parent.parent.name.split("_")[1])
        dataset = cif.parent.name
        m = pat.match(cif.name[: -len(".cif.gz")])
        if not m:
            skipped.append((cif.name, "filename does not parse"))
            continue
        eid = m.group("eid")
        if eid not in native_by_id:
            skipped.append((cif.name, f"no native for {eid}"))
            continue
        spec_key = f"{eid}__{m.group('task')}"
        tasks.append((
            str(cif), native_by_id[eid], epoch, dataset,
            spec_key, int(m.group("model")), str(pep_by_id[eid]),
            released_by_spec.get(spec_key, frozenset()),
        ))

    if args.limit > 0:
        tasks = tasks[: args.limit]
    logger.info(f"scoring {len(tasks):,} generated models with {args.workers} workers")
    if skipped:
        logger.warning(f"skipped {len(skipped)}: e.g. {skipped[:3]}")
    if not tasks:
        raise SystemExit("FATAL: nothing to score")

    with Pool(args.workers) as pool:
        rows = pool.map(score_one, tasks, chunksize=16)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df[df.status == "ok"]
    logger.info(f"scored {len(ok):,} of {len(df):,}; statuses {df.status.value_counts().to_dict()}")
    if ok.empty:
        raise SystemExit("FATAL: no row scored")

    # QC first: if the receptor did not stay fixed, every RMSD below is meaningless.
    bad_fit = ok[ok.receptor_rmsd_after_fit > 0.5]
    logger.info(f"QC receptor_rmsd_after_fit: median {ok.receptor_rmsd_after_fit.median():.4f} A, "
                f"{len(bad_fit)} row(s) above 0.5 A")
    if len(bad_fit):
        logger.warning("those rows' peptide RMSDs are NOT trustworthy")

    # SEQUENCE, SPLIT. The pooled median is a copy score; report it only next to the split
    # so it cannot be mistaken for a capability.
    logger.info(f"sequence identity POOLED (mostly copied, not a capability): "
                f"median {ok.seq_identity.median():.4f}")
    if "seq_identity_given" in ok:
        g = ok.seq_identity_given.dropna()
        if len(g):
            bad = int((g < 0.9999).sum())
            logger.info(f"  COPY CHECK  given positions: median {g.median():.4f} over "
                        f"{len(g):,} models -- must be 1.0")
            if bad:
                logger.warning(f"  {bad} model(s) got a GIVEN identity wrong: the "
                               f"conditioning is not being honoured, so the hidden-position "
                               f"number below is not interpretable either")
    if "seq_identity_hidden" in ok:
        h = ok.seq_identity_hidden.dropna()
        if len(h):
            logger.info(f"  MEASUREMENT hidden positions: mean {h.mean():.4f} over "
                        f"{len(h):,} models "
                        f"(compare against the most-common-amino-acid prior, ~9-10%)")
            for ep, ge in ok.groupby("epoch"):
                he = ge.seq_identity_hidden.dropna()
                if len(he):
                    logger.info(f"    epoch {ep}: {h.name} mean {he.mean():.4f}  n {len(he):,}")
    else:
        logger.warning("no hidden-position column: pass --contigs to measure the sequence "
                       "task rather than the copy")

    # Headline numbers in the same shape the other arm reports, per validation epoch, so a
    # trend across checkpoints is readable and an arm comparison is possible.
    logger.info("per-epoch summary (medians over models; hit = interface_raw <= 2.0 A):")
    for ep, g in ok.groupby("epoch"):
        per_spec = g.groupby("example_id")
        best_raw = per_spec.raw_ca_rmsd.min()
        best_iface = per_spec.interface_raw_ca_rmsd.min()
        hit = (g.interface_raw_ca_rmsd <= 2.0).mean()
        any_hit = (best_iface <= 2.0).mean()
        logger.info(
            f"  epoch {ep}: raw {g.raw_ca_rmsd.median():6.2f}  superposed "
            f"{g.superposed_ca_rmsd.median():5.2f}  iface {g.interface_raw_ca_rmsd.median():6.2f}"
            f"  bestN_raw {best_raw.median():6.2f}  bestN_iface {best_iface.median():6.2f}"
            f"  poses<=2A {hit:.3f}  targets_any_hit {any_hit:.3f}"
            f"  clash2A {g.n_clash_2A.median():5.0f}  contacts5A {g.n_contacts_5A.median():5.0f}"
        )
    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
