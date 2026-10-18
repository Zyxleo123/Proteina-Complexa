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

NO HOTSPOTS, deliberately -- SUPERSEDED 2026-10-15, see `--hotspot-mode`. The original
reasoning was that training runs with `add_ppi_hotspots: 0.0` because the eventual inference
input is target plus sequence and nothing else, so supplying hotspots would measure an easier
task than the one being trained. That is still true of the BASE arm, and `--hotspot-mode off`
(the default) still builds exactly the panel described above for it.

What changed is the task. Without a retention objective the paired LP structure is
decorative: if the generated peptide may bind anywhere on the target, the LP's binding mode
carried no information and its sequence alone would have been a sufficient input. So hotspots
become part of the task rather than a leak, and `datasets/lpgen_hotspot.yaml` trains with
RFD3's own stock dosage. This builder therefore grew a panel that can measure whether the
channel is OBEYED -- see HOTSPOT ARMS below.

HOTSPOT ARMS (`--hotspot-mode native|obedience`). `fraction_hotspots_contacted` on its own
cannot support a claim of obedience, because the hotspots are drawn FROM the native interface
and the model would score well by binding the pocket it was going to bind anyway. Only a
contrast does:

  ``__hs_native``           hotspots from the native interface; ori from that same patch.
  ``__hs_none``             NO hotspots; ori from the native patch. The paired control --
                            identical in every other respect, so the difference against
                            ``hs_native`` is the channel's entire contribution.
  ``__hs_decoy``            hotspots on a patch on the FAR SIDE of the receptor, ori from
                            that patch. "Can the peptide be sent somewhere else at all."
  ``__hs_decoy_orinative``  decoy hotspots but the NATIVE ori. The hotspot request and the
                            diffusion prior now disagree, which is the only arm that
                            isolates the channel's pull from the prior's. Expect it to be
                            hard; a non-zero shift toward the decoy is still real evidence.

THE ORI IS WHY THE DECOY ARMS NEED CARE. `ori_token` is where the prior is dropped, and it
beats `infer_ori_strategy` outright (`set_com`, utils/inference.py:463) -- so a decoy arm that
kept the native `ori_token` and merely added `infer_ori_strategy: hotspots` would silently
ignore the strategy (it does log a warning) and the test would be rigged. Equally, a decoy arm
whose ori is derived from the decoy patch proves little on its own: the peptide would land
there because the PRIOR put it there, not because the hotspots asked. That is exactly why both
decoy arms exist and why neither is quoted alone.

The ori for every hotspot arm is computed HERE, from the requested patch, with this panel's
calibrated 6 A push -- not by `infer_ori_strategy: hotspots`, whose `DISTANCE_ABOVE_HOTSPOTS`
is hard-coded to 10.0 (utils/inference.py:435). Measured on this panel, 10 A lands 5.23 A from
the native peptide COM against 3.27 A at 6 A, so the stock value overshoots here. Using our
own value keeps the hotspot panel on the same ori convention as the existing pocket panel.

Run:
    .venv/bin/python script_utils/lpgen_build_val_contigs.py \\
        --panel $ZFS/LPData/preprocessed/metadata_lpgen/lpgen_val_panel.parquet \\
        --out   $ZFS/LPData/preprocessed/metadata_lpgen/lpgen_val_contigs.json

    # the in-loop hotspot panel (native + paired control)
    ... --hotspot-mode native   --out .../lpgen_val_contigs_hotspot.json
    # the full obedience panel, for the reval path
    ... --hotspot-mode obedience --out .../lpgen_val_contigs_hs_obedience.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    p.add_argument("--hotspot-mode", default="off",
                   choices=("off", "native", "obedience"),
                   help="'off' keeps the original no-hotspot panel (the base arm). 'native' "
                        "adds the hs_native and hs_none arms -- the pair that measures "
                        "whether giving hotspots changes anything. 'obedience' adds the two "
                        "decoy arms as well; that panel belongs on the reval path, not "
                        "in-loop, because it quadruples the rollout cost.")
    p.add_argument("--hotspot-cutoff", type=float, default=4.5,
                   help="Interchain heavy-atom cutoff defining the native hotspot set. MUST "
                        "match `ppi_hotspot_max_distance` in datasets/lpgen_hotspot.yaml AND "
                        "`PPIMetrics.distance_cutoff`: conditioning wider than scoring makes "
                        "a perfectly obedient model look disobedient.")
    p.add_argument("--hotspot-frac", type=float, default=0.2,
                   help="Fraction of the ground-truth hotspot atoms to request, as "
                        "ceil(n*frac). Matches the training config's "
                        "`max_ppi_hotspots_frac_to_provide`, but note the DELIBERATE "
                        "difference: training draws randint(0, ceil(n*frac)) -- uniform below "
                        "the ceiling and sometimes zero -- while validation requests the "
                        "ceiling exactly, every epoch. A request that changed between epochs "
                        "could not be compared across checkpoints, and a request that is "
                        "sometimes empty would silently drop specs from the metric (PPIMetrics "
                        "skips examples with no hotspots). So the val request is slightly "
                        "denser than the mean training request, on purpose.")
    p.add_argument("--decoy-radius", type=float, default=8.0,
                   help="Radius of the decoy patch around its seed atom. 8 A gives a patch of "
                        "comparable size to a peptide interface without wrapping around the "
                        "protein.")
    p.add_argument("--decoy-min-separation", type=float, default=15.0,
                   help="A decoy patch atom must be at least this far from every native "
                        "hotspot atom. Guards the whole point of the decoy arm: a patch that "
                        "brushes the native site would be partly satisfiable by binding "
                        "natively, and the arm would under-report disobedience.")
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


# ----------------------------------------------------------------- hotspot arms
# All of these are deterministic functions of the structure and the example id. Nothing here
# draws a random number: a validation request that changed between epochs could not be
# compared across checkpoints, which is the only reason to validate periodically at all.

def native_hotspot_mask(atom_array, pep_chain: str, cutoff: float):
    """Receptor atoms within ``cutoff`` of any peptide atom -- RFD3's hotspot definition.

    Mirrors `get_hotspot_atoms` (ppi_transforms.py:44) in substance: interchain contacts at a
    heavy-atom cutoff, on the non-binder side only. Written out rather than imported because
    this builder runs in the PROJECT venv, where rfd3 is not installed.
    """
    import numpy as np

    chain = np.asarray(atom_array.chain_id, dtype=str)
    is_pep = chain == pep_chain
    coord = np.asarray(atom_array.coord, dtype=float)
    finite = np.isfinite(coord).all(axis=-1)

    out = np.zeros(len(chain), dtype=bool)
    pep_idx = np.where(is_pep & finite)[0]
    rec_idx = np.where((~is_pep) & finite)[0]
    if pep_idx.size == 0 or rec_idx.size == 0:
        return out
    d = np.linalg.norm(coord[rec_idx][:, None, :] - coord[pep_idx][None, :, :], axis=-1)
    out[rec_idx] = d.min(axis=1) <= cutoff
    return out


def deterministic_subsample(mask, n: int, key: str):
    """Keep ``n`` of the True positions in ``mask``, chosen from a hash of ``key``.

    Hash-ordered rather than ``np.random.choice`` so the same complex asks for the same atoms
    at every epoch and in every rerun of this builder.
    """
    import numpy as np

    idx = np.where(mask)[0]
    if n >= idx.size:
        return mask.copy()
    out = np.zeros_like(mask)
    if n <= 0 or idx.size == 0:
        return out
    h = hashlib.sha256(key.encode()).digest()
    order = sorted(range(idx.size), key=lambda i: (h[i % len(h)], i))
    out[idx[sorted(order[:n])]] = True
    return out


def decoy_hotspot_mask(atom_array, pep_chain: str, native_mask, radius: float,
                       min_separation: float):
    """A receptor patch on the FAR SIDE of the target, as a negative control.

    Construction, deterministic at every step:
      1. the direction from the receptor core to the native hotspot centroid;
      2. receptor atoms in the OPPOSITE hemisphere of that direction;
      3. among those, the one FARTHEST FROM THE CORE is the seed -- farthest-from-centroid is
         a surface atom by construction, which avoids seeding the patch inside the protein
         where no peptide could ever satisfy it. There is no RASA available here, so this
         stands in for solvent exposure;
      4. receptor atoms within ``radius`` of the seed;
      5. every patch atom within ``min_separation`` of ANY native hotspot is dropped.

    Step 5 is what makes the arm a control rather than a weaker version of the native arm: a
    patch that brushes the native interface is partly satisfiable by binding natively, so the
    model would score as obedient without having moved.

    Returns ``(mask, info)``; ``mask`` is None when no acceptable patch exists.
    """
    import numpy as np

    chain = np.asarray(atom_array.chain_id, dtype=str)
    coord = np.asarray(atom_array.coord, dtype=float)
    finite = np.isfinite(coord).all(axis=-1)
    rec = (chain != pep_chain) & finite
    if not rec.any() or not native_mask.any():
        return None, "no receptor atoms or no native hotspots"

    core = coord[rec].mean(0)
    v = coord[native_mask].mean(0) - core
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        return None, "native patch centroid coincides with the receptor core"
    v = v / nv

    rec_idx = np.where(rec)[0]
    proj = (coord[rec_idx] - core) @ v
    far_side = rec_idx[proj < 0]
    if far_side.size == 0:
        return None, "no receptor atom lies opposite the native patch"

    seed = far_side[int(np.argmax(np.linalg.norm(coord[far_side] - core, axis=1)))]
    d_seed = np.linalg.norm(coord[rec_idx] - coord[seed], axis=1)
    patch = np.zeros(len(chain), dtype=bool)
    patch[rec_idx[d_seed <= radius]] = True

    # Enforce the separation from the native site.
    nat_idx = np.where(native_mask)[0]
    pidx = np.where(patch)[0]
    if pidx.size and nat_idx.size:
        dmin = np.linalg.norm(
            coord[pidx][:, None, :] - coord[nat_idx][None, :, :], axis=-1
        ).min(axis=1)
        patch[pidx[dmin < min_separation]] = False

    if not patch.any():
        return None, (f"decoy patch empty after enforcing {min_separation} A separation "
                      f"from the native site")
    sep = float(np.linalg.norm(
        coord[np.where(patch)[0]][:, None, :] - coord[nat_idx][None, :, :], axis=-1
    ).min())
    return patch, (f"{int(patch.sum())} atoms, min separation from native site {sep:.1f} A")


def patch_ori(atom_array, patch_mask, pep_chain: str, push: float):
    """Origin derived from an arbitrary receptor patch, pushed outward from the core.

    The same construction as `pocket_ori` -- centroid of the patch, displaced along the
    core->patch vector -- so the hotspot arms and the existing pocket panel sit on ONE ori
    convention and a difference between them is not a difference in where the prior was
    dropped. Deliberately not `infer_ori_strategy: hotspots`, whose push is hard-coded to
    10.0 A against this panel's calibrated 6.0.
    """
    import numpy as np

    chain = np.asarray(atom_array.chain_id, dtype=str)
    coord = np.asarray(atom_array.coord, dtype=float)
    finite = np.isfinite(coord).all(axis=-1)
    rec = (chain != pep_chain) & finite
    if not rec.any() or not patch_mask.any():
        return None, "no receptor atoms or empty patch"

    patch_c = coord[patch_mask].mean(0)
    core_c = coord[rec].mean(0)
    v = patch_c - core_c
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None, "patch centroid coincides with the receptor core"
    return [float(x) for x in (patch_c + push * (v / n))], f"|core->patch| = {n:.2f} A"


def hotspot_selection(atom_array, mask) -> dict:
    """``{"<chain><res_id>": [atom names]}`` for the selected atoms.

    This is the ATOM-level spelling of `select_hotspots`, which matters: training marks
    individual atoms (`is_atom_level_hotspot`, and the upstream note at ppi_transforms.py:39
    is explicit that RFD3 differs from RFD here), so a residue-level request would hand the
    model a denser pattern than it ever saw. The key format is exactly what
    `apply_selections` matches on -- ``selection.get(f"{chain_id}{res_id}")``,
    input_parsing.py:508 -- and the value may be a list of atom names (parsing.py:159).
    """
    import numpy as np

    chain = np.asarray(atom_array.chain_id, dtype=str)
    res_id = np.asarray(atom_array.res_id)
    name = np.asarray(atom_array.atom_name, dtype=str)
    sel: dict = {}
    for i in np.where(mask)[0]:
        sel.setdefault(f"{chain[i]}{res_id[i]}", []).append(str(name[i]))
    return sel


def main() -> int:
    args = parse_args()
    panel = pd.read_parquet(args.panel)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    specs: dict = {}
    from collections import Counter, defaultdict
    ori_modes: Counter = Counter()
    arm_counts: Counter = Counter()
    hs_stats: dict = defaultdict(list)
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

        # The structure is needed by the ORI block and by the hotspot arms, so load it once.
        _a = None
        if args.ori_mode != "none" or args.hotspot_mode != "off":
            import numpy as _np
            from biotite.structure.io.pdb import PDBFile as _PDB
            _a = _PDB.read(str(row.path)).get_structure(model=1)

        # ORI: where the diffusion prior is dropped. Omitting it is NOT neutral -- rfd3 then
        # centres on the fixed motif, i.e. the receptor centroid, measured 17.8 A (median)
        # from the real site, and the peptide condenses inside the protein.
        if args.ori_mode != "none":
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

        if args.hotspot_mode == "off":
            if "fullseq" in tasks:
                specs[f"{row.example_id}__fullseq"] = dict(base)

            if "partialseq" in tasks:
                n_rel = max(1, round(pep_n * args.unfixed_frac))
                rel = released_positions(row.example_id, list(range(pep_lo, pep_hi + 1)), n_rel)
                if rel:
                    specs[f"{row.example_id}__partialseq"] = dict(
                        base, select_unfixed_sequence=",".join(f"{pep_ch}{r}" for r in rel)
                    )
        else:
            # ---------------------------------------------------------- hotspot arms
            # Every arm here is `fullseq` (the peptide's identity is given) so that the ONLY
            # axis the panel varies is the hotspot request. Mixing in the partial-sequence
            # task as well would confound "did it obey the hotspots" with "did it also have
            # to invent identities", and the sequence half of the task is separately known to
            # sit at chance.
            nat_all = native_hotspot_mask(_a, pep_ch, args.hotspot_cutoff)
            n_gt = int(nat_all.sum())
            if n_gt == 0:
                skipped.append((row.example_id,
                                f"no receptor atom within {args.hotspot_cutoff} A of the "
                                f"peptide: no hotspot can be defined"))
                continue

            n_req = max(1, math.ceil(n_gt * args.hotspot_frac))
            nat = deterministic_subsample(nat_all, n_req, f"{row.example_id}|native")
            nat_ori, _nw = patch_ori(_a, nat, pep_ch, args.ori_push)
            if nat_ori is None:
                skipped.append((row.example_id, f"native hotspot ori unavailable: {_nw}"))
                continue

            # Calibration, reported rather than assumed: how far the hotspot-derived prior
            # sits from where the native peptide actually is. The existing pocket panel
            # measures a median 3.27 A, and a hotspot panel much worse than that would be
            # measuring its own origin rather than the model.
            _pep_com = _a[_np.asarray(_a.chain_id, dtype=str) == pep_ch].coord.mean(0)
            hs_stats["native_ori_to_pep_com"].append(
                float(_np.linalg.norm(_np.asarray(nat_ori) - _pep_com))
            )
            hs_stats["n_gt"].append(n_gt)
            hs_stats["n_requested"].append(int(nat.sum()))

            hs_base = dict(base)
            hs_base["ori_token"] = nat_ori

            specs[f"{row.example_id}__hs_native"] = dict(
                hs_base, select_hotspots=hotspot_selection(_a, nat)
            )
            # The paired control: same ori, same everything, NO hotspot request. Without this
            # arm a high retention number is unattributable.
            specs[f"{row.example_id}__hs_none"] = dict(hs_base)
            arm_counts["hs_native"] += 1
            arm_counts["hs_none"] += 1

            if args.hotspot_mode == "obedience":
                dec_all, _dw = decoy_hotspot_mask(
                    _a, pep_ch, nat_all, args.decoy_radius, args.decoy_min_separation
                )
                if dec_all is None:
                    # Not a skip: the native pair is still valid and informative for this
                    # complex. Dropping the whole row would silently shrink the panel and
                    # make the two arms span different complexes.
                    logger.warning(f"{row.example_id}: no decoy patch ({_dw}); "
                                   f"native arms kept, decoy arms omitted")
                    arm_counts["decoy_unavailable"] += 1
                else:
                    # SAME NUMBER of requested atoms as the native arm, so the metric's
                    # denominator matches and the two are comparable.
                    dec = deterministic_subsample(dec_all, int(nat.sum()),
                                                  f"{row.example_id}|decoy")
                    dec_ori, _dow = patch_ori(_a, dec, pep_ch, args.ori_push)
                    if dec_ori is None:
                        logger.warning(f"{row.example_id}: decoy ori unavailable ({_dow}); "
                                       f"decoy arms omitted")
                        arm_counts["decoy_unavailable"] += 1
                    else:
                        dec_sel = hotspot_selection(_a, dec)
                        specs[f"{row.example_id}__hs_decoy"] = dict(
                            base, ori_token=dec_ori, select_hotspots=dec_sel
                        )
                        # Decoy request against the NATIVE prior: the only arm where the two
                        # channels disagree, and therefore the only one that isolates the
                        # hotspot channel's pull from the prior's placement.
                        specs[f"{row.example_id}__hs_decoy_orinative"] = dict(
                            hs_base, select_hotspots=dec_sel
                        )
                        arm_counts["hs_decoy"] += 1
                        arm_counts["hs_decoy_orinative"] += 1
                        hs_stats["decoy_ori_to_pep_com"].append(
                            float(_np.linalg.norm(_np.asarray(dec_ori) - _pep_com))
                        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))

    logger.info(f"panel rows: {len(panel)}, specs written: {len(specs)} -> {out}")
    if args.hotspot_mode == "off":
        for t in tasks:
            logger.info(f"  {t}: {sum(1 for k in specs if k.endswith('__' + t))}")
    else:
        import numpy as _np
        for arm in ("hs_native", "hs_none", "hs_decoy", "hs_decoy_orinative"):
            n = sum(1 for k in specs if k.endswith("__" + arm))
            if n:
                logger.info(f"  {arm}: {n}")
        if arm_counts.get("decoy_unavailable"):
            # Stated rather than left to be noticed: the decoy arms then cover FEWER
            # complexes than the native arms, so the two cannot be compared by taking a
            # mean over each arm independently -- pair them by example id.
            logger.warning(
                f"  {arm_counts['decoy_unavailable']} complex(es) have NO decoy arm, so the "
                f"decoy and native arms span different complexes. Compare them PAIRED by "
                f"example id, not as two independent means.")
        for key, label in (("n_gt", "ground-truth hotspot atoms"),
                           ("n_requested", "requested hotspot atoms"),
                           ("native_ori_to_pep_com", "native-hotspot ORI to native pep COM (A)"),
                           ("decoy_ori_to_pep_com", "decoy ORI to native pep COM (A)")):
            v = hs_stats.get(key) or []
            if v:
                a = _np.asarray(v, dtype=float)
                logger.info(f"  {label}: median {_np.median(a):.2f} "
                            f"[{a.min():.2f}, {a.max():.2f}]")
        # The instrument check. The existing pocket panel lands a median 3.27 A from the
        # native peptide COM; a hotspot-derived prior much worse than that would make the
        # hotspot arms measure their own origin rather than the model.
        v = hs_stats.get("native_ori_to_pep_com") or []
        if v:
            med = float(_np.median(_np.asarray(v, dtype=float)))
            if med > 6.0:
                logger.warning(
                    f"  native-hotspot ORI sits a median {med:.2f} A from the native peptide "
                    f"COM, against 3.27 A for the pocket panel. The prior is the dominant "
                    f"term at that distance; lower --ori-push or widen --hotspot-frac before "
                    f"reading obedience off this panel.")
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
