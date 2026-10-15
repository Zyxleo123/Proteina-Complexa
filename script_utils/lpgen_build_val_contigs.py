#!/usr/bin/env python3
"""Build RFD3 contig specs for in-loop generative validation on the LP panel.

WHY THIS EXISTS. RFD3's validation is not a held-out loss -- `AADesignTrainer.validation_step`
runs a rollout and scores design metrics, and its datasets are `ContigJsonDataset` over a
contig JSON. So "generate some during validation to see whether the learning reaches our
goal" means: hand the validator a contig spec per panel complex and let it sample.

That is also the only honest instrument here. The training metrics cannot answer the
question: after two epochs the only thing that moved was `Seq Recovery` rising to 0.988,
which measures the model reproducing a sequence our own condition *revealed* to it, while
MSE and lDDT were flat to within 0.0 and 0.08 standard errors. This project has been misled
by teacher-forced numbers before.

HOW THE TASK IS EXPRESSED. The eventual goal is: given a target and a peptide SEQUENCE,
produce a bound peptide STRUCTURE. In contig terms that is two independent switches:

  * `contig` includes the peptide (`B<lo>-<hi>`), and every region drawn from the input has
    a **fixed sequence** by default -- so the peptide's identity is given.
  * `select_fixed_atoms` is a **dictionary that explicitly releases the peptide** with an
    empty value -- so the peptide's **coordinates are not fixed** and must be generated.

MEASURED, because the obvious spellings are silent no-ops. Fixed-coordinate atom counts for
one 74+6-residue complex (614 atoms, 64 of them peptide):

    select_fixed_atoms: "A0-73"            -> 614 fixed   (no effect)
    select_fixed_atoms omitted             -> 614 fixed   (identical)
    select_fixed_atoms: {"A0-73": "ALL"}   -> 614 fixed   (identical)
    select_fixed_atoms: False              ->   0 fixed   (releases the RECEPTOR too)
    contig with a designed region `6`      -> 550 fixed, but peptide SEQUENCE lost
    select_fixed_atoms: {"B0-5": ""}       -> 550 fixed, peptide free, sequence kept  <-- this
    select_fixed_atoms: {"A0-73":"ALL","B0-5":""}  -> same, and self-documenting

Listing a region does not restrict fixing to it: input regions are fixed by default and a
selection does not subtract. Only an **explicit empty value** releases atoms -- the
mini-language documents this as `LIG: ''` "selects no atoms (i.e. unfixes the atoms)".
Getting this wrong produces a validation example with every atom fixed, so the model
generates nothing, and the failure surfaces far away as
`arrays used as indices must be of integer (or boolean) type` from a design metric handed an
empty array (`get_token_starts` early-exits with a float64 `np.array([])`).

Those are exactly the two annotations `LPBinderCondition` sets during training
(`is_motif_atom_with_fixed_seq` on the peptide, `is_motif_atom`/`..._with_fixed_coord` on the
receptor only), so validation asks for the same thing training taught.

TWO SPECS PER COMPLEX, matching the plan's two training tasks:

  ``<id>__fullseq``     every peptide identity given; only geometry generated.
  ``<id>__partialseq``  a deterministic ~10% of peptide identities released via
                        `select_unfixed_sequence`, so those identities are generated too.

The released positions are chosen from a hash of the example id, NOT at random: validation
has to be the *same* set at every epoch or the metric is not comparable across checkpoints.

THE ORI TOKEN IS SUPPLIED, and omitting it was a measured error worth ~18 A of RMSD.
`ori_token` appears ONLY in rfd3's inference code -- never in a training transform. Training
centres on the diffused region (`center_option: diffuse`, i.e. the peptide); inference with
no `ori_token` logs "Setting origin as COM of fixed motif" and centres on the RECEPTOR. The
docs state the diffused region's COM lands within ~5 A of the ORI token, so the model was
being told to build the peptide at the receptor's centroid.

Measured on this panel: receptor-COM to native-peptide-COM is a median of 17.8 A (range
8.1-25.0), and the first validation runs produced a median raw CA RMSD of ~21.3 A with a
peptide-only superposed RMSD of just ~3.1 A -- right fold, wrong place, displaced by almost
exactly that offset. The numbers were measuring obedience to a bad instruction, not the
model.

`ori_token` is therefore set to the native peptide's centre of mass, which is what
`center_option: diffuse` supplied during training. Note what this makes validation mean: it
measures "GIVEN the binding site, can the model build the pose", not "can the model find the
site". That is the right question for a pose benchmark and the wrong one to quote as de novo
site discovery. `infer_ori_strategy: "hotspots"` is the alternative, but it requires
`select_hotspots`, and we train with `add_ppi_hotspots: 0.0` -- so it would introduce a
conditioning channel the model never saw.

NO HOTSPOTS, deliberately. Training runs with `add_ppi_hotspots: 0.0` because the eventual
inference input is target plus sequence and nothing else. Supplying hotspots here would
measure a different, easier task than the one being trained.

Run:
    .venv/bin/python script_utils/lpgen_build_val_contigs.py \\
        --panel $ZFS/LPData/preprocessed/metadata_lpgen/lpgen_val_panel.parquet \\
        --out   $ZFS/LPData/preprocessed/metadata_lpgen/lpgen_val_contigs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--panel", required=True, help="lpgen_val_panel.parquet")
    p.add_argument("--out", required=True, help="Contig JSON to write.")
    p.add_argument("--unfixed-frac", type=float, default=0.10,
                   help="Fraction of peptide identities released in the partialseq spec. "
                        "The training task reveals 80-95%%, so ~10%% released matches its "
                        "middle.")
    p.add_argument("--tasks", default="fullseq,partialseq")
    p.add_argument("--ori-mode", default="pocket",
                   choices=("pocket", "native_pep", "none"),
                   help="How to place the diffusion prior. 'pocket' derives it from RECEPTOR "
                        "atoms lining the site (declared input); 'native_pep' uses the native "
                        "peptide COM (uses the answer's location -- learning check only); "
                        "'none' omits it and lets rfd3 fall back to the receptor centroid "
                        "(measured ~17.8 A off -- the original bug).")
    p.add_argument("--ori-push", type=float, default=6.0,
                   help="Angstroms to push the pocket centroid outward along the "
                        "core->pocket vector. Calibrated on this panel: median distance to "
                        "the native peptide COM is 6.89 A at 0, 3.27 A at 6, 5.23 A at 10 "
                        "(rfd3's own stock value). 0 disables the push.")
    p.add_argument("--pocket-cutoff", type=float, default=10.0,
                   help="Receptor atoms within this distance of the peptide line the pocket. "
                        "Matches POCKET_CUTOFF_A in lpgen_export_native.py.")
    return p.parse_args()


def chain_ranges(pdb_path: str) -> dict:
    """Per-chain (min_res_id, max_res_id, n_distinct) from ATOM records."""
    seen: dict = {}
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            seen.setdefault(line[21], set()).add(int(line[22:26]))
    return {ch: (min(v), max(v), len(v)) for ch, v in seen.items()}


def released_positions(example_id: str, res_ids: list, n: int) -> list:
    """Deterministically pick ``n`` peptide residues whose identity is released.

    Hash-derived rather than random: a validation set that changes between epochs cannot be
    compared across checkpoints, which is the whole point of validating periodically.
    Interior positions are preferred -- the termini are where the eventual edit task is
    expected to act, so releasing them would conflate the two.
    """
    if n <= 0 or not res_ids:
        return []
    interior = res_ids[1:-1] or res_ids
    h = hashlib.sha256(example_id.encode()).digest()
    order = sorted(range(len(interior)), key=lambda i: h[i % len(h)] * 256 + i)
    return sorted(interior[i] for i in order[:min(n, len(interior))])



def pocket_ori(atom_array, pep_chain: str, cutoff: float, push: float):
    """Origin for the diffusion prior, derived from RECEPTOR atoms only.

    WHY NOT THE NATIVE PEPTIDE COM. That is the answer's location. Using it makes validation
    a pose-reconstruction check and nothing more -- it can never support a claim about
    placing a peptide on a target, because the target's answer was supplied. The pocket is
    declared input (the plan's inference input is "target/pocket plus CP sequence"), and in
    real design the site is chosen by the user, not discovered. So the origin is computed
    from the receptor residues lining the site; the peptide is used ONLY to select which
    receptor atoms those are, which is the same information a user supplies by naming
    hotspots.

    This mirrors rfd3's own `infer_ori_from_hotspots`: hotspot centroid pushed outward along
    the vector from the protein core, so the prior starts just off the surface where a binder
    sits rather than inside the protein. `push` is calibrated rather than taken from rfd3's
    stock 10.0 -- measured on this panel, distance to the native peptide COM is 6.89 A at
    push 0, 3.27 A at 6, and 5.23 A at 10, so the stock value overshoots here.

    Returns ``(ori, info)``; ``ori`` is None when the direction is degenerate.
    """
    import numpy as np

    chain = np.asarray(atom_array.chain_id, dtype=str)
    pep = atom_array[chain == pep_chain]
    rec = atom_array[chain != pep_chain]
    if pep.array_length() == 0 or rec.array_length() == 0:
        return None, "no peptide or no receptor atoms"

    d = np.linalg.norm(rec.coord[:, None, :] - pep.coord[None, :, :], axis=-1)
    near = d.min(axis=1) < cutoff
    if not near.any():
        return None, f"no receptor atom within {cutoff} A of the peptide"

    pocket_c = rec.coord[near].mean(0)
    core_c = rec.coord.mean(0)
    v = pocket_c - core_c
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        # A pocket whose centroid coincides with the protein core has no outward direction.
        # Normalising here would divide by ~0 and emit NaNs, which rfd3 would silently
        # translate the whole structure by -- so refuse and let the caller fall back.
        return None, "pocket centroid coincides with receptor core: no outward direction"

    ori = pocket_c + push * (v / n)
    return [float(x) for x in ori], f"|core->pocket| = {n:.2f} A, {int(near.sum())} pocket atoms"

def main() -> int:
    args = parse_args()
    panel = pd.read_parquet(args.panel)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    specs: dict = {}
    from collections import Counter
    ori_modes: Counter = Counter()
    skipped = []

    for row in panel.itertuples():
        rng = chain_ranges(row.path)
        pep_ch = str(row.peptide_chain_id)
        rec_chs = sorted(c for c in rng if c != pep_ch)
        if pep_ch not in rng or not rec_chs:
            skipped.append((row.example_id, f"chains present: {sorted(rng)}"))
            continue

        # A contig range `A{lo}-{hi}` names every id in between, so a gap would silently
        # request residues that do not exist. Counts come from the manifest, so this also
        # cross-checks the two.
        bad = False
        for ch, (lo, hi, n) in rng.items():
            if hi - lo + 1 != n:
                skipped.append((row.example_id, f"chain {ch} residue ids are not contiguous "
                                                f"({lo}-{hi} spans {hi-lo+1} but has {n})"))
                bad = True
                break
        if bad:
            continue
        if rng[pep_ch][2] != int(row.peptide_length):
            skipped.append((row.example_id, f"peptide {rng[pep_ch][2]} residues != manifest "
                                            f"{row.peptide_length}"))
            continue

        rec_parts = [f"{c}{rng[c][0]}-{rng[c][1]}" for c in rec_chs]
        pep_lo, pep_hi, pep_n = rng[pep_ch]
        pep_part = f"{pep_ch}{pep_lo}-{pep_hi}"
        # Receptor, chain break, peptide. The peptide is IN the contig so its sequence is
        # given; it is absent from select_fixed_atoms so its coordinates are generated.
        contig = ",".join(rec_parts + ["/0", pep_part])
        fixed = ",".join(rec_parts)

        # Receptor explicitly fixed, peptide explicitly RELEASED. Both sides stated so the
        # intent is readable and does not depend on which default applies.
        base = {
            "input": str(row.path),
            "contig": contig,
            "select_fixed_atoms": {fixed: "ALL", pep_part: ""},
        }

        # ORI: where the diffusion prior is dropped. Omitting it is NOT neutral -- rfd3 then
        # centres on the fixed motif, i.e. the receptor centroid, measured 17.8 A (median)
        # from the real site, and the peptide condenses inside the protein.
        if args.ori_mode != "none":
            import numpy as _np
            from biotite.structure.io.pdb import PDBFile as _PDB
            _a = _PDB.read(str(row.path)).get_structure(model=1)
            if args.ori_mode == "native_pep":
                _pep = _a[_np.asarray(_a.chain_id, dtype=str) == pep_ch]
                base["ori_token"] = [float(x) for x in _pep.coord.mean(0)]
                ori_modes["native_pep"] += 1
            else:
                _ori, _why = pocket_ori(_a, pep_ch, args.pocket_cutoff, args.ori_push)
                if _ori is None:
                    # Fall back rather than emit a NaN origin, and say so: a silent fallback
                    # would put a handful of specs on a different convention than the rest of
                    # the panel and make the curve incomparable across complexes.
                    _pep = _a[_np.asarray(_a.chain_id, dtype=str) == pep_ch]
                    base["ori_token"] = [float(x) for x in _pep.coord.mean(0)]
                    ori_modes["fallback_native_pep"] += 1
                    logger.warning(f"{row.example_id}: pocket ORI unavailable ({_why}); "
                                   f"fell back to the native peptide COM")
                else:
                    base["ori_token"] = _ori
                    ori_modes["pocket"] += 1

        if "fullseq" in tasks:
            specs[f"{row.example_id}__fullseq"] = dict(base)

        if "partialseq" in tasks:
            n_rel = max(1, round(pep_n * args.unfixed_frac))
            rel = released_positions(row.example_id, list(range(pep_lo, pep_hi + 1)), n_rel)
            if rel:
                specs[f"{row.example_id}__partialseq"] = dict(
                    base, select_unfixed_sequence=",".join(f"{pep_ch}{r}" for r in rel)
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))

    logger.info(f"panel rows: {len(panel)}, specs written: {len(specs)} -> {out}")
    for t in tasks:
        logger.info(f"  {t}: {sum(1 for k in specs if k.endswith('__' + t))}")
    if skipped:
        logger.warning(f"skipped {len(skipped)}:")
        for eid, why in skipped[:8]:
            logger.warning(f"    {eid}: {why}")
    # The ORI composition decides what the whole panel MEASURES, so state it rather than
    # leaving it to be inferred from the numbers later.
    logger.info(f"ori_mode={args.ori_mode} push={args.ori_push} cutoff={args.pocket_cutoff}: "
                f"{dict(ori_modes)}")
    if args.ori_mode == "native_pep":
        logger.warning("ORI is the NATIVE PEPTIDE COM: this panel measures pose "
                       "reconstruction given the answer's location. Do not quote it as "
                       "site placement.")
    if ori_modes.get("fallback_native_pep"):
        logger.warning(f"{ori_modes['fallback_native_pep']} spec(s) fell back to the native "
                       f"peptide COM, so the panel MIXES two ORI conventions; those "
                       f"complexes are not comparable with the rest.")
    if not specs:
        raise SystemExit("FATAL: no specs written")
    k = next(iter(specs))
    logger.info(f"example spec [{k}]: {json.dumps(specs[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
