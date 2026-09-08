"""Track-asymmetric SDEdit: turn a bound LINEAR peptide into a cyclic analogue, minimally.

The idea this implements
------------------------
The CPSea flow is a product space over two tracks:

    bb_ca           CA coordinates (R^3)  -- the POSE
    local_latents   8-dim/residue         -- SEQUENCE + sidechains + local geometry

and `t` is sampled INDEPENDENTLY per track during training (`shared_groups: []`). So a
state built at (t_ca = 0.4, t_latent = 1.0) is something the model was actually trained on,
not an off-manifold hack. That turns SDEdit here from a 1-D noise sweep into a 2-D grid
whose axes have distinct physical meanings:

    t_ca_start      how much the BACKBONE may rearrange (1.0 = frozen pose)
    t_lat_start     the SUBSTITUTION BUDGET (1.0 = sequence frozen, zero substitutions)

The corner (t_ca < 1, t_lat = 1) is the minimal edit: sequence and sidechains preserved
exactly, backbone free to move enough to close a ring. Lowering t_lat buys the model
freedom to place chemistry-required anchors (e.g. two CYS for a disulfide) at the cost of
substitutions -- which this script MEASURES rather than dictates.

Why substitutions are not pre-applied
-------------------------------------
Mutating a residue before encoding would mean handing the AE a residue whose sidechain
atoms are masked out -- an input it never saw in training. Letting the latent track
regenerate identity keeps every state on-manifold, and makes the substitution count an
outcome of the noise level instead of a hand-set constant.

Preservation is measured, not assumed
-------------------------------------
The receptor is fixed conditioning in a target-centred frame, so the sampled peptide is
already in the input's frame: CA displacement is a direct subtraction, with no superposition
step that could hide movement inside the binding site.

Outputs one JSONL row per (example, cyclization type, t_ca, t_lat, seed). Resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import torch

from proteinfoundation.cyclization.inference import attach_cyclization_prediction
from proteinfoundation.datasets.structure_data import structure_collate_fn
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    C_IDX,
    NM_TO_ANG,
    N_IDX,
    cyclic_geometry_metrics,
    get_cb_position,
)
from proteinfoundation.eval.sampled_binder_metrics import _as_residue_mask, atom37_mask_from_aatype
from proteinfoundation.flow_matching.product_space_flow_matcher import (
    get_gt,
    get_schedule,
    get_schedule_tsr_safe,
)
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting

sys.path.insert(0, str(Path(__file__).parent))
from sdedit_guidance import SimilarityGuidance, guidance_tag, parse_loss_spec  # noqa: E402
from sdedit_guidance import add_cli_args as add_guidance_args  # noqa: E402
from anchor_graft import graft_anchors, graft_tag  # noqa: E402
from anchor_graft import add_cli_args as add_graft_args  # noqa: E402

CA_IDX = 1
CONTACT_CUTOFF_NM = 1.0  # CA-CA; the coarse contact definition used for retention accounting

from proteinfoundation.cyclization.constants import CYCLIZATION_TYPE_TO_NAME

TYPE_NAMES = {"mainchain": 0, "disulfide": 1, "isopeptide": 2}


def _resolve_sampling_cfg(cfg):
    """Returns the node holding `args`/`model`, whether at the top level or nested under the
    config's directory package (hydra makes "pipeline/model_sampling" land at cfg.pipeline)."""
    if "args" in cfg and "model" in cfg:
        return cfg
    for key in list(cfg.keys()):
        child = cfg[key]
        if hasattr(child, "keys") and "args" in child and "model" in child:
            return child
    raise SystemExit(
        f"FATAL: sampling config has no node with both `args` and `model`; top-level keys: {list(cfg.keys())}"
    )


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


# `vf_to_score` asserts t < 1 strictly, so the frozen-track corner cannot use t = 1 exactly.
T_START_MAX = 1.0 - 1e-3


def build_edit_schedule(fm, nsteps: int, sampling_model_args: dict, t_start: dict[str, float]):
    """Per-track schedules compressed into [t_start[mode], 1], mirroring `full_simulation`.

    Remapping (t_start + (1 - t_start) * t) rather than slicing the full schedule keeps the
    schedule's shape AND gives both tracks the same number of steps -- required, because
    `partial_simulation` advances both tracks on one shared step index. `gt` (the noise
    injection schedule) is recomputed from the remapped times, not carried over.
    """
    ts, gt = {}, {}
    for dm in fm.data_modes:
        args_dm = sampling_model_args[dm]
        schedule_func = (
            get_schedule_tsr_safe
            if args_dm["simulation_step_params"]["sampling_mode"] == "vf_tsr"
            else get_schedule
        )
        base = schedule_func(mode=args_dm["schedule"]["mode"], nsteps=int(nsteps), p1=args_dm["schedule"]["p"])
        # t_start = 1 is the frozen-track corner (start from the clean input). Taken
        # literally the remap collapses to a constant vector of ones, and every step then
        # trips the strict `t < 1` assert in `vf_to_score`. Clamp to just inside the open
        # interval: the track is still effectively frozen, but the schedule is non-degenerate.
        t0 = min(float(t_start[dm]), T_START_MAX)
        ts[dm] = t0 + (1.0 - t0) * base
        gt[dm] = get_gt(
            t=ts[dm][:-1],
            mode=args_dm["gt"]["mode"],
            param=args_dm["gt"]["p"],
            clamp_val=args_dm["gt"]["clamp_val"],
        )
    return ts, gt


def contact_set(pep_ca_nm, tgt_ca_nm, pep_mask, tgt_mask, cutoff=CONTACT_CUTOFF_NM):
    """Set of (peptide_idx, target_idx) CA pairs within `cutoff` nm."""
    d = torch.cdist(pep_ca_nm, tgt_ca_nm)  # [Lp, Lt]
    valid = pep_mask[:, None] & tgt_mask[None, :]
    hits = (d < cutoff) & valid
    return {(int(i), int(j)) for i, j in zip(*torch.where(hits))}


# Bond-atom closure (C-N, SG-SG, NZ-CG) is the real criterion, but it is only DEFINED where
# the sampled sequence happens to carry the required anchors. In the frozen-sequence corner
# the head abstains on ~78% of rows, so "0 of 13 closed" is a statement about 13 of 59
# samples and silent about the rest.
#
# These CB/backbone distances need no anchor and are therefore defined on EVERY row, which
# turns "unscorable" into a number.
#
# MEASURED, AND THE ANSWER IS NO: they do NOT predict bond closure. Validated on 242 scorable
# de novo samples (uncondrelax_20260906_170925) against the bond-atom criterion:
#
#     term_cb_dist_A       closed median 4.66 A   open median 4.83 A   AUC 0.462
#     min_cb_pair_dist_A   closed median 4.19 A   open median 4.28 A   AUC 0.430
#     term_nc_dist_A       closed median 6.38 A   open median 7.38 A   AUC 0.402
#
# AUC 0.5 is chance, so none of these separates closed from open. Do NOT use them as a
# closure criterion or as a substitute where the head abstains.
#
# They are still worth recording, because that null result IS the finding: the RING SHAPE is
# formed either way -- the ends come together at ~4-5 A whether or not a bond exists. Closure
# therefore fails on anchor IDENTITY (the model placing the wrong residue at i/j), not on
# gross geometry, which matches the separate measurement that isopeptide failure is ~50%
# anchor identity with NZ-CG already at 1.3 A when the residues are right. It is also why
# `cyc_cb_window_success` saturates and why the summarizer drops it.
def cb_closure_geometry(sample_coors_nm, sample_aatype, mask, b=0):
    """Anchor-free closure geometry for one sample. Distances in Angstrom."""
    m = mask[b].bool()
    idx = torch.nonzero(m, as_tuple=False).flatten()
    if idx.numel() < 2:
        return {}
    coors = sample_coors_nm[b]                      # [n, 37, 3] nm
    a37_mask = atom37_mask_from_aatype(sample_aatype)[b] & m[:, None]
    cb, cb_is_real = get_cb_position(coors, a37_mask)   # [n, 3], [n]

    first, last = int(idx[0]), int(idx[-1])
    out = {
        # Terminal ring geometry: what a head-to-tail macrocycle would have to close.
        "term_cb_dist_A": float(torch.norm(cb[last] - cb[first]) * NM_TO_ANG),
        "term_nc_dist_A": float(torch.norm(coors[last, C_IDX] - coors[first, N_IDX]) * NM_TO_ANG),
        "term_ca_dist_A": float(torch.norm(coors[last, CA_IDX] - coors[first, CA_IDX]) * NM_TO_ANG),
        "term_cb_is_real": int(bool(cb_is_real[first]) and bool(cb_is_real[last])),
    }
    # Best CB-CB over any pair separated by >=3 residues: "did it form a ring ANYWHERE",
    # independent of both the anchors and the termini. |i-j|>=3 excludes trivially-close
    # neighbours that every extended chain already satisfies.
    cbv = cb[idx]                                    # [L, 3]
    L = cbv.shape[0]
    d = torch.cdist(cbv, cbv) * NM_TO_ANG            # [L, L]
    sep = (torch.arange(L, device=d.device)[:, None] - torch.arange(L, device=d.device)[None, :]).abs()
    far = sep >= 3
    if bool(far.any()):
        out["min_cb_pair_dist_A"] = float(d[far].min())
        flat = int(torch.argmin(torch.where(far, d, torch.full_like(d, float("inf")))))
        out["min_cb_pair_i"] = int(idx[flat // L])
        out["min_cb_pair_j"] = int(idx[flat % L])
    return out


def score_edit(sample_coors_nm, sample_aatype, batch, mask, model, gen_samples, sample_prots,
               requested_type_idx):
    """Closure + preservation metrics for one edited peptide."""
    out: dict[str, float] = {}
    b = 0
    m = mask[b].bool()
    L = int(m.sum())

    gt_coors = batch["coords_nm"][b]        # [n, 37, 3] input peptide
    gt_aatype = batch["residue_type"][b]

    # --- preservation: pose. No superposition -- receptor frame is shared by construction,
    # so this is displacement WITHIN the binding site, which a Kabsch fit would hide.
    pep_ca_in = gt_coors[m][:, CA_IDX, :]
    pep_ca_out = sample_coors_nm[b][m][:, CA_IDX, :]
    out["ca_rmsd_to_input_A"] = float(torch.sqrt(((pep_ca_out - pep_ca_in) ** 2).sum(-1).mean()) * NM_TO_ANG)
    out["ca_max_dev_to_input_A"] = float(torch.sqrt(((pep_ca_out - pep_ca_in) ** 2).sum(-1)).max() * NM_TO_ANG)

    # --- preservation: sequence / substitution budget
    # Measured against the batch's `residue_type`, i.e. against whatever was ENCODED -- so
    # under --graft-anchors this is what the SAMPLER changed, with the graft already in the
    # baseline. The `*_vs_native` pair below is measured against the untouched crystal
    # sequence and therefore includes the grafted anchors, which is the number to quote as
    # the total cost of the edit. Reporting only the first would make a grafted arm look
    # one or two mutations cheaper than it really is.
    seq_in, seq_out = gt_aatype[m].long(), sample_aatype[b][m].long()
    n_sub = int((seq_in != seq_out).sum())
    out["n_substitutions"] = n_sub
    out["seq_identity"] = float((seq_in == seq_out).float().mean())
    native = batch.get("residue_type_native")
    if native is not None:
        seq_nat = native[b][m].long()
        out["n_substitutions_vs_native"] = int((seq_nat != seq_out).sum())
        out["seq_identity_vs_native"] = float((seq_nat == seq_out).float().mean())
    out["peptide_length"] = L

    # --- preservation: interface contacts
    x_target, target_mask = batch.get("x_target"), batch.get("target_mask")
    if x_target is not None and target_mask is not None:
        # `target_mask` is ATOM-level [B, T, 37] in compact mode, not the [B, T] its name
        # suggests. Reuse the codebase's own normaliser -- indexing it as a residue mask
        # silently misaligns every interface number.
        tmask = _as_residue_mask(target_mask.bool())[b]          # [T]
        tgt_ca = x_target[b][:, CA_IDX, :] if x_target[b].dim() == 3 else x_target[b]
        if tmask.shape[0] != tgt_ca.shape[0]:
            raise RuntimeError(
                f"target mask/coords length mismatch: {tmask.shape[0]} vs {tgt_ca.shape[0]}"
            )
        tgt_ca = tgt_ca[tmask]
        ones_p = torch.ones(L, dtype=torch.bool, device=m.device)
        ones_t = torch.ones(tgt_ca.shape[0], dtype=torch.bool, device=m.device)
        before = contact_set(pep_ca_in, tgt_ca, ones_p, ones_t)
        after = contact_set(pep_ca_out, tgt_ca, ones_p, ones_t)
        out["n_contacts_input"] = len(before)
        out["n_contacts_edited"] = len(after)
        out["contact_retention"] = float(len(before & after) / len(before)) if before else float("nan")

    # --- closure geometry that does NOT depend on the head finding anchors
    out.update(cb_closure_geometry(sample_coors_nm, sample_aatype, mask, b=b))

    # --- closure: the model's own predicted (i, j, type) on its own sequence
    cyc = attach_cyclization_prediction(model, gen_samples, batch, dict(sample_prots))
    if "pred_cyclization_i" in cyc:
        meta = {"i": cyc["pred_cyclization_i"].long(), "j": cyc["pred_cyclization_j"].long(),
                "type": cyc["pred_cyclization_type"].long(),
                "has_cyclization": torch.ones_like(cyc["pred_cyclization_i"], dtype=torch.bool)}
        raw = cyclic_geometry_metrics(
            pred_atom37=sample_coors_nm, gt_atom37=sample_coors_nm,
            atom37_mask=atom37_mask_from_aatype(sample_aatype) & mask[..., None],
            seq_tokens=sample_aatype.long(), cyclization_metadata=meta, prefix="cyc",
        )
        for k, v in raw.items():
            if "_gt_A" in k or "_abs_err_A" in k:
                continue  # meaningless here: gt_atom37 is the sample itself
            out[k] = float(v)
        out["pred_cyc_i"] = int(cyc["pred_cyclization_i"][b])
        out["pred_cyc_j"] = int(cyc["pred_cyclization_j"][b])
        out["pred_cyc_type"] = int(cyc["pred_cyclization_type"][b])
        out["requested_type_satisfied"] = int(int(cyc["pred_cyclization_type"][b]) == requested_type_idx)
    return out


def load_abstained_cells(paths, min_rate: float) -> dict:
    """(example_id, cyc_type, t_ca, t_lat) -> abstention rate, from reference edit rows.

    Abstention is `requested_type_satisfied == 0`: the cyclization head emitted a NULL edge
    because the decoded sequence admitted no candidate pair for the requested chemistry. It is
    read as an int, never for truthiness -- an abstained row carries NaN in its
    `cyc/*_bond_success` field, and NaN is truthy, which has already scored a 100%-abstention
    cell as 100% closure once.

    Failed rows are dropped: they carry no verdict either way, and counting them as
    "attempted" would quietly shrink the population this run is supposed to target.
    """
    seen: dict = {}
    n_rows = n_failed = 0
    for path in paths:
        for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"FATAL: corrupt row {path}:{lineno}")
            if r.get("status") == "failed":
                n_failed += 1
                continue
            key = (str(r["example_id"]), str(r["cyc_type"]),
                   float(r["t_ca_start"]), float(r["t_lat_start"]))
            tot, abst = seen.get(key, (0, 0))
            seen[key] = (tot + 1, abst + int(int(r.get("requested_type_satisfied") or 0) == 0))
            n_rows += 1
    rates = {k: a / t for k, (t, a) in seen.items()}
    kept = {k: v for k, v in rates.items() if v >= min_rate}
    print(f"abstention reference: {n_rows} rows ({n_failed} failed, dropped) over "
          f"{len(rates)} cells; {len(kept)} at abstention >= {min_rate}", flush=True)
    return rates, kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--ckpt-name", default="last-EMA.ckpt")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config-name", default="example/training_cpsea_peptide_smoke")
    ap.add_argument("--sampling-config", default="pipeline/model_sampling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb-dir", default=None, help="If set, writes each edited peptide as a PDB.")
    ap.add_argument("--complex-pdb-dir", default=None,
                    help="If set, ALSO writes receptor+edited-peptide complexes in the input "
                         "PDB's frame -- the only place they can be built (see "
                         "scripts/complex_frame.py: the loader frame is per-process, so this "
                         "cannot be reconstructed after the run). Needed for Rosetta dG.")
    ap.add_argument("--complex-align-tol-A", type=float, default=0.05,
                    help="Max Kabsch residual of the INPUT peptide onto its own crystal pose "
                         "before the complex is refused rather than written misplaced.")
    ap.add_argument("--cyc-types", nargs="+", default=["disulfide", "isopeptide", "mainchain"],
                    help="Requested cyclization types. LNR geometry says disulfide/isopeptide "
                         "are the tractable regime and mainchain is the stress case.")
    ap.add_argument("--t-ca", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8],
                    help="Backbone-track start times. Lower = more rearrangement allowed.")
    ap.add_argument("--t-lat", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.4],
                    help="Latent-track start times = the substitution budget. 1.0 freezes sequence.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--native-type", action="store_true",
                    help="Ignore --cyc-types: run ONE arm per example using that example's OWN "
                         "native cyclization type (the batch's cyclization_type_cond, set by "
                         "CyclizationLabelTransform). This is what val_generation does, and it is "
                         "the only apples-to-apples control for a val_gen closure number. Examples "
                         "with no usable native type are SKIPPED, never coerced to mainchain.")
    ap.add_argument("--nsteps", type=int, default=0, help="0 = the design sampler's own nsteps.")
    ap.add_argument("--limit", type=int, default=0, help="Score at most N input peptides (0 = all).")
    ap.add_argument("--examples", nargs="+", default=None,
                    help="Run only these example_ids. Unknown ids are a FATAL error, not a "
                         "silent empty run -- a typo would otherwise look like a clean pass.")
    # Sharding partitions the INPUT PEPTIDES, and each shard writes its own --out file.
    # Concurrent O_APPEND to one shared file has NUL-corrupted rows on this cluster before,
    # so shards must never share an output path.
    ap.add_argument("--abstained-from", nargs="+", default=None,
                    help="One or more edits_shard*.jsonl from a REFERENCE run. Restricts this "
                         "run to the (example, type, t_ca, t_lat) cells the reference "
                         "ABSTAINED on (requested_type_satisfied == 0) -- the population a "
                         "closure-guidance arm exists to move. A requested cell the reference "
                         "never ran is FATAL, not a silent skip.")
    ap.add_argument("--abstained-min-rate", type=float, default=1.0,
                    help="Keep a cell only if its reference abstention rate is at least this. "
                         "1.0 = the cell abstained on every reference seed.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_guidance_args(ap)
    add_graft_args(ap)
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve configs, dataset and grid, then exit before loading the "
                         "checkpoint. Catches config/plumbing errors without a GPU.")
    args = ap.parse_args()

    from eval_ae_roundtrip import build_dataset  # same loader path as the AE go/no-go

    # Parse the spec here so a typo dies before the dataset build, not 3 GB of checkpoint later.
    gtag = "|".join(t for t in (guidance_tag(args), graft_tag(args)) if t)
    if args.graft_anchors:
        print(f"anchor grafting ON (sidechain={args.graft_sidechain}"
              f"{', keep_cb=' + str(args.graft_keep_cb) if args.graft_sidechain == 'truncate' else ''}"
              "): the input peptide's two terminal residues are set to the requested "
              "chemistry's anchors before encoding, so the head cannot abstain for lack of a "
              "candidate pair.", flush=True)
        if args.graft_sidechain == "truncate":
            print("WARNING: truncate mode keeps only backbone+CB, which IS alanine's complete "
                  "atom set -- the AE decodes these endpoints back as ALA and the graft is "
                  "erased. This is the negative control, not a run to draw conclusions from.",
                  flush=True)
    if guidance_tag(args):
        print(f"similarity guidance ON: {parse_loss_spec(args.guidance_loss)} w={args.guidance_w} "
              f"schedule={args.guidance_schedule}^{args.guidance_schedule_pow} "
              f"exclude_termini={args.guidance_exclude_termini} mode={args.guidance_mode}",
              flush=True)
    else:
        print("similarity guidance OFF (--guidance-w 0): sampler runs with no hook", flush=True)

    unknown = [t for t in args.cyc_types if t not in TYPE_NAMES]
    if unknown:
        raise SystemExit(f"FATAL: unknown cyclization type(s) {unknown}; expected {sorted(TYPE_NAMES)}")

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    dataset, _ = build_dataset(args.config_name, args.metadata, keep_rotation=False)
    meta = dataset.metadata
    all_indices = list(range(len(dataset)))
    if args.examples:
        want = list(dict.fromkeys(args.examples))
        pos = {str(e): i for i, e in enumerate(meta["example_id"])}
        missing = [e for e in want if e not in pos]
        if missing:
            raise SystemExit(f"FATAL: example(s) not in {args.metadata}: {missing}")
        all_indices = [pos[e] for e in want]
    # Abstention filter: resolved BEFORE the checkpoint load, and BEFORE sharding, so the
    # shards partition the peptides that actually have work rather than an unfiltered list
    # (which would leave most shards with nothing to do and one with everything).
    abstained = None
    if args.abstained_from:
        _rates, abstained = load_abstained_cells(args.abstained_from, args.abstained_min_rate)
        want_types = set(args.cyc_types)
        keep_examples = {e for (e, ct, _tca, _tlat) in abstained if ct in want_types}
        pos = {str(e): i for i, e in enumerate(meta["example_id"])}
        unknown = sorted(keep_examples - set(pos))
        if unknown:
            raise SystemExit(f"FATAL: the abstention reference names example(s) absent from "
                             f"{args.metadata}: {unknown[:5]} -- wrong metadata for this run.")
        all_indices = [i for i in all_indices if str(meta.iloc[i]["example_id"]) in keep_examples]
        if not all_indices:
            raise SystemExit("FATAL: no peptide has an abstaining cell in the requested "
                             "chemistries -- an empty run would look like a clean pass.")
        print(f"abstention filter: {len(all_indices)} of {len(meta)} peptides retained",
              flush=True)
    if args.limit:
        all_indices = all_indices[: args.limit]
    my_indices = all_indices[args.shard_index :: args.shard_count]
    print(f"{len(dataset)} input peptides from {args.metadata}; "
          f"shard {args.shard_index}/{args.shard_count} takes {len(my_indices)}", flush=True)

    # Resolve the sampling config BEFORE loading the checkpoint. The flow checkpoint is ~3 GB;
    # a config typo discovered after that load costs a full model load per failed job.
    with hydra.initialize("../configs", version_base=hydra.__version__):
        samp = hydra.compose(config_name=args.sampling_config)
    # A config named by a PATH ("pipeline/model_sampling") composes with its directory as the
    # package, so the payload lands at cfg.pipeline.*, not the top level. Descend to whichever
    # node actually carries `args`/`model` instead of assuming either shape.
    samp = _resolve_sampling_cfg(samp)
    sampler_args, sampling_model_args = samp.args, samp.model
    nsteps = args.nsteps or int(sampler_args.nsteps)
    self_cond = bool(sampler_args.self_cond)
    n_recycle = int(samp.get("n_recycle", 0))
    print(f"sampler: nsteps={nsteps} self_cond={self_cond} n_recycle={n_recycle}", flush=True)

    if args.dry_run:
        n_cells = len(my_indices) * len(args.cyc_types) * len(args.t_ca) * len(args.t_lat)
        if abstained is not None:
            # The filter is applied per CELL inside the run loop, so the cartesian product
            # badly over-counts a filtered run -- and a job sized off that number asks for
            # hours it does not need. Count the surviving cells here, and check the same
            # coverage the run loop would die on rather than discovering it on a GPU.
            ids = {str(meta.iloc[i]["example_id"]) for i in my_indices}
            cells = [(e, c, float(a), float(l)) for e in ids for c in args.cyc_types
                     for a in args.t_ca for l in args.t_lat]
            missing = [c for c in cells if c not in _rates]
            if missing:
                raise SystemExit(
                    f"FATAL: {len(missing)} of {len(cells)} requested cells are absent from "
                    f"the abstention reference (e.g. {missing[:3]}). The reference run is "
                    "incomplete for this grid -- wait for it to finish, or narrow the grid.")
            n_cells = sum(1 for c in cells if c in abstained)
            print(f"abstention filter: {n_cells} of {len(cells)} cells retained", flush=True)
        n_edits = n_cells * len(args.seeds)
        print(f"DRY RUN OK: {len(my_indices)} peptides x {len(args.cyc_types)} chemistries x "
              f"{len(args.t_ca)} t_ca x {len(args.t_lat)} t_lat x {len(args.seeds)} seeds "
              f"= {n_edits} edits", flush=True)
        return

    ckpt_file = os.path.join(args.ckpt_path, args.ckpt_name)
    if not os.path.isfile(ckpt_file):
        raise SystemExit(f"FATAL: checkpoint not found: {ckpt_file}")
    print(f"flow checkpoint: {ckpt_file}", flush=True)
    model = Proteina.load_from_checkpoint(
        ckpt_file, strict=False, autoencoder_ckpt_path=os.environ.get("CPSEA_AE_CKPT_PATH")
    )
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    if model.autoencoder is None:
        raise SystemExit("FATAL: checkpoint has no autoencoder; local_latents editing is impossible.")

    if not getattr(model, "cyclization_type_conditioning", False):
        print("WARNING: this checkpoint has cyclization.type_conditioning OFF -- the requested "
              "type cannot reach the denoiser and every arm collapses to the same run.", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for lineno, line in enumerate(out_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["run_key"])
            except (json.JSONDecodeError, KeyError):
                raise SystemExit(f"FATAL: corrupt row {out_path}:{lineno}. Delete and rerun.")
    if done:
        print(f"resuming: {len(done)} runs already done", flush=True)
    if args.pdb_dir:
        Path(args.pdb_dir).mkdir(parents=True, exist_ok=True)
    if args.complex_pdb_dir:
        Path(args.complex_pdb_dir).mkdir(parents=True, exist_ok=True)
        if "path" not in meta_columns_of(dataset):
            raise SystemExit("FATAL: --complex-pdb-dir needs a `path` column in the metadata "
                             "(the input complex PDB); this parquet has none.")

    n_done = n_fail = 0
    t_wall = time.time()
    with out_path.open("a") as fh:
        for idx in my_indices:
            example_id = str(meta.iloc[idx]["example_id"])
            sample = dataset[idx]
            if sample is None:
                print(f"  SKIP {example_id}: failed to load", flush=True)
                continue
            base_batch = to_device(structure_collate_fn([sample]), args.device)
            mask = (base_batch["mask"].bool() if "mask" in base_batch
                    else base_batch["coord_mask"][..., CA_IDX].bool())
            base_batch["mask"] = mask

            # The loader -> crystal transform, recovered HERE and nowhere else. Two loads of
            # one example land in frames that differ by a rigid transform (the receptor crop
            # is redrawn each time), so a peptide written without its receptor can never be
            # re-united with it afterwards. See scripts/complex_frame.py.
            frame = None
            if args.complex_pdb_dir:
                frame = build_frame(base_batch, mask, meta.iloc[idx], args.complex_align_tol_A)
                if frame is None:
                    print(f"  {example_id}: no complexes written (frame fit refused)", flush=True)

            if args.native_type:
                # One arm only, and it is whatever chemistry THIS example actually has.
                nt = base_batch.get("cyclization_type_cond")
                has_cyc = base_batch.get("has_cyclization")
                usable = (nt is not None and int(nt[0]) in CYCLIZATION_TYPE_TO_NAME
                          and (has_cyc is None or bool(has_cyc[0])))
                if not usable:
                    got = None if nt is None else int(nt[0])
                    print(f"  SKIP {example_id}: no usable native cyclization type "
                          f"(cyclization_type_cond={got})", flush=True)
                    continue
                arm_types = [CYCLIZATION_TYPE_TO_NAME[int(nt[0])]]
            else:
                arm_types = list(args.cyc_types)

            for cyc_name in arm_types:
                type_idx = TYPE_NAMES[cyc_name]
                for t_ca in args.t_ca:
                    for t_lat in args.t_lat:
                        for seed in args.seeds:
                            if abstained is not None:
                                cell = (example_id, cyc_name, float(t_ca), float(t_lat))
                                if cell not in _rates:
                                    raise SystemExit(
                                        f"FATAL: cell {cell} is not in the abstention "
                                        "reference. Skipping it silently would make this arm "
                                        "an unmeasured subset of the grid it is compared to.")
                                if cell not in abstained:
                                    continue
                            run_key = f"{example_id}|{cyc_name}|{t_ca}|{t_lat}|{seed}"
                            if gtag:
                                run_key += f"|{gtag}"
                            if run_key in done:
                                continue
                            try:
                                row = run_one(model, base_batch, mask, cyc_name, type_idx,
                                              t_ca, t_lat, seed, nsteps, self_cond,
                                              sampler_args, sampling_model_args, args,
                                              example_id=example_id, n_recycle=n_recycle,
                                              frame=frame, native_type=args.native_type)
                            except Exception as exc:  # noqa: BLE001
                                n_fail += 1
                                # Stamp the grid point on the failure too. Without it a dead
                                # cell has t_ca_start/t_lat_start = None and vanishes from
                                # every groupby, so the summary shows a clean grid.
                                fh.write(json.dumps({"run_key": run_key, "example_id": example_id,
                                                     "status": "failed", "cyc_type": cyc_name,
                                                     "t_ca_start": t_ca, "t_lat_start": t_lat,
                                                     "seed": seed,
                                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                                fh.flush()
                                print(f"  FAIL {run_key}: {type(exc).__name__}: {exc}", flush=True)
                                continue
                            row.update(run_key=run_key, example_id=example_id, status="ok",
                                       cyc_type=cyc_name, t_ca_start=t_ca, t_lat_start=t_lat, seed=seed)
                            for col in ("peptide_length", "nc_gap_angstrom", "best_ss_cb_dist",
                                        "best_iso_cb_dist", "n_cys"):
                                if col in meta.columns:
                                    v = meta.iloc[idx][col]
                                    row[f"input_{col}"] = v.item() if hasattr(v, "item") else v
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                            n_done += 1
                            if n_done % 25 == 0:
                                print(f"  {n_done} edits ({time.time() - t_wall:.0f}s)", flush=True)

    print(f"done: {n_done} edits, {n_fail} failed, {time.time() - t_wall:.0f}s", flush=True)

    # Per-edit failures are caught and written as rows, so without this the script exits 0
    # even when EVERY edit failed -- which is how a smoke run whose 2/2 edits died on the
    # `t < 1` assert still cleared the way for a 1416-edit sweep three minutes later.
    # A wholly-failed shard must break the afterok chain: its summary would be meaningless.
    if n_done == 0 and n_fail > 0:
        raise SystemExit(f"FATAL: all {n_fail} edits failed, none succeeded -- see the errors above.")


def meta_columns_of(dataset):
    return list(dataset.metadata.columns)


def build_frame(base_batch, mask, meta_row, tol_A):
    """Loader -> input-PDB transform for this example, or None if it cannot be trusted."""
    from complex_frame import crystal_transform

    input_pdb = Path(str(meta_row["path"]))
    if not input_pdb.is_file():
        print(f"  WARNING: input PDB missing: {input_pdb}; no complexes for this example", flush=True)
        return None
    binder_chain = str(meta_row.get("binder_chain_id", "B"))
    loader_ca = (base_batch["coords_nm"][0][mask[0]][:, CA_IDX, :].detach().cpu().numpy()
                 * NM_TO_ANG).astype("float64")
    try:
        r, t, offset, resid, receptor_lines, target_chains = crystal_transform(
            loader_ca, input_pdb, binder_chain)
    except RuntimeError as exc:
        print(f"  WARNING: frame fit failed ({exc}); no complexes for this example", flush=True)
        return None
    if resid > tol_A:
        # Refuse rather than write a complex in the wrong place: a misplaced peptide does not
        # look broken downstream, it looks like a bad interface energy.
        print(f"  WARNING: frame residual {resid:.3f} A > {tol_A} A for {input_pdb.name}; "
              f"no complexes for this example", flush=True)
        return None
    return {"r": r, "t": t, "offset": offset, "residual_A": resid,
            "receptor_lines": receptor_lines, "target_chains": target_chains,
            "binder_chain": binder_chain, "input_pdb": str(input_pdb)}


def run_one(model, base_batch, mask, cyc_name, type_idx, t_ca, t_lat, seed, nsteps,
            self_cond, sampler_args, sampling_model_args, args, example_id="x", n_recycle=0,
            frame=None, native_type=False):
    """One edit: encode the input, re-noise each track to its own start time, integrate, score."""
    L.seed_everything(seed, workers=True)
    fm = model.fm

    batch = {k: v for k, v in base_batch.items() if k not in ("x_1", "x_0", "x_t", "t", "x_sc")}
    batch["mask"] = mask

    # The requested type MUST be stamped explicitly. An LNR batch already carries a
    # `cyclization_type_cond` of UNSPECIFIED (CyclizationLabelTransform finds no CONECT ring
    # in a linear peptide), and `apply_cyclization_type_conditioning` returns early whenever
    # that key exists -- so relying on it would silently run every arm unconditioned.
    bs = mask.shape[0]
    if native_type:
        # Native arm: the batch already carries the example's real type and it must survive
        # untouched -- overwriting it is precisely the bug this control exists to exclude,
        # so verify rather than trust.
        if "cyclization_type_cond" not in batch:
            raise RuntimeError("native-type arm requires cyclization_type_cond in the batch")
        if int(batch["cyclization_type_cond"][0]) != int(type_idx):
            raise RuntimeError(
                f"native type mismatch: batch={int(batch['cyclization_type_cond'][0])} "
                f"caller={type_idx}")
    else:
        batch["cyclization_type_cond"] = torch.full((bs,), int(type_idx), dtype=torch.long,
                                                    device=mask.device)

    # Terminal anchor graft, BEFORE the encode: the latents this edit preserves are the
    # latents of the grafted peptide, so the graft has to happen while the sequence is still
    # an input rather than after it has become a latent. See scripts/anchor_graft.py.
    graft_info: dict = {}
    if getattr(args, "graft_anchors", False):
        batch, graft_info = graft_anchors(batch, mask, int(type_idx),
                                          keep_cb=bool(int(args.graft_keep_cb)),
                                          sidechain=args.graft_sidechain)

    # Clean sample: bb_ca from the input CA trace, local_latents from the AE posterior MEAN
    # (deterministic -- posterior sampling would add noise this experiment is trying to control).
    with torch.no_grad():
        batch = add_clean_samples(batch, model.cfg_exp.product_flowmatcher,
                                  autoencoder=model.autoencoder, local_latent_target="mean")
        x_1 = fm._apply_mask(batch["x_1"], mask) if hasattr(fm, "_apply_mask") else batch["x_1"]

        # Clamp ONCE, here, so the interpolated start state `x_t` and the schedule below
        # agree on where each track begins -- clamping only inside build_edit_schedule
        # would start the latent track at t=1 but step it from 1-eps.
        t_start = {"bb_ca": min(float(t_ca), T_START_MAX),
                   "local_latents": min(float(t_lat), T_START_MAX)}
        missing = set(fm.data_modes) - set(t_start)
        if missing:
            raise RuntimeError(f"no start time supplied for data mode(s) {missing}")

        # No `training=` kwarg here: the product-space sample_noise reads `self.training`,
        # which `model.eval()` has already set False (this matters -- it gates the
        # stochastic-centering branch, so passing it explicitly would diverge from full_simulation).
        x_0 = fm.sample_noise(n=mask.shape[1], shape=(bs,), mask=mask, device=mask.device)
        t_vec = {dm: torch.full((bs,), float(t_start[dm]), device=mask.device) for dm in fm.data_modes}
        x_t = fm.interpolate(x_0=x_0, x_1=x_1, t=t_vec, mask=mask)

        ts, gt = build_edit_schedule(fm, nsteps, sampling_model_args, t_start)
        ts = {k: v.to(mask.device) for k, v in ts.items()}
        gt = {k: v.to(mask.device) for k, v in gt.items()}
        sim_params = {dm: sampling_model_args[dm]["simulation_step_params"] for dm in fm.data_modes}
        predict_fn = partial(model.predict_for_sampling, n_recycle=n_recycle)

        # Similarity guidance: gradient of an explicit LP-vs-CP loss, added to the state after
        # every Euler step. `guidance_fn=None` at w=0 means the sampler takes the untouched
        # pre-guidance code path, so the w=0 arm stays comparable with the earlier sweeps.
        guid = None
        if getattr(args, "guidance_w", 0.0):
            guid = SimilarityGuidance(
                fm=fm, predict_for_sampling=predict_fn, batch=batch, mask=mask,
                losses=parse_loss_spec(args.guidance_loss), weight=float(args.guidance_w),
                schedule=args.guidance_schedule, schedule_pow=float(args.guidance_schedule_pow),
                exclude_termini=int(args.guidance_exclude_termini), mode=args.guidance_mode,
                t_start=t_start, max_disp_A=float(args.guidance_max_disp_A),
                # Closure terms decode the predicted clean sample and score the bond of the
                # REQUESTED chemistry -- `type_idx` here, the same value stamped into
                # `cyclization_type_cond` above, so the guidance and the denoiser are being
                # asked for the same ring. `allow_asn_gln` must match the validity mask that
                # decides abstention, or anchor_ce would optimise for a pair the head rejects.
                autoencoder=model.autoencoder, cyc_type_idx=int(type_idx),
                allow_asn_gln=bool(getattr(model, "cyclization_allow_asn_gln_isopeptide", True)),
                cb_window_A=tuple(args.guidance_cb_window_A),
                stride=int(getattr(args, "guidance_stride", 1) or 1),
            )

        gen_samples, _ = fm.partial_simulation(
            batch=batch, x=x_t, x_1_pred=None, mask=mask,
            predict_for_sampling=predict_fn,
            start_step=0, end_step=nsteps, self_cond=self_cond, ts=ts, gt=gt,
            simulation_step_params=sim_params, device=mask.device,
            guidance_w=float(sampler_args.get("guidance_w", 1.0)), ag_ratio=0.0,
            guidance_fn=guid,
        )

        sample_prots = sample_formatting(
            x=gen_samples, extra_info={"mask": mask}, ret_mode="coors37_n_aatype",
            data_modes=list(model.cfg_exp.product_flowmatcher), autoencoder=model.autoencoder,
        )
        # sample_formatting returns ANGSTROM; every metric below is in nm. Convert once, here.
        coors_nm = sample_prots["coors"] / NM_TO_ANG
        aatype = sample_prots["residue_type"]

        row = score_edit(coors_nm, aatype, batch, mask, model, gen_samples, sample_prots, type_idx)

    # Settings go on EVERY row (0/None when off) so the summary can group by them without
    # inventing defaults for the unguided arm; the diagnostics say whether guidance moved
    # anything at all, which is how a silently inert arm gets caught.
    row.update(graft_anchors=int(getattr(args, "graft_anchors", False)))
    row.update(graft_info)
    row.update(guidance_w=float(getattr(args, "guidance_w", 0.0)),
               guidance_loss=args.guidance_loss if guid else None,
               guidance_schedule=args.guidance_schedule if guid else None,
               guidance_schedule_pow=float(args.guidance_schedule_pow) if guid else None,
               guidance_exclude_termini=int(args.guidance_exclude_termini) if guid else None,
               guidance_mode=args.guidance_mode if guid else None,
               guidance_stride=int(getattr(args, "guidance_stride", 1) or 1) if guid else None)
    if guid is not None:
        row.update(guid.stats())

    # "_" here, not the run_key's "|": this one becomes a FILENAME.
    gtag = "_".join(t for t in (guidance_tag(args), graft_tag(args)) if t)
    tag = f"{example_id}_{cyc_name}_tca{t_ca}_tlat{t_lat}_s{seed}"
    if gtag:
        # Without this every lambda arm writes the same PDB filename and the last one wins.
        tag += f"_{gtag}"
    if args.pdb_dir or (args.complex_pdb_dir and frame is not None):
        from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

        # The complex needs the peptide's PDB text anyway, so it is written once and reused;
        # with --complex-pdb-dir alone it goes to a scratch file next to the complexes.
        pep_dir = Path(args.pdb_dir) if args.pdb_dir else Path(args.complex_pdb_dir) / "_peptides"
        pep_dir.mkdir(parents=True, exist_ok=True)
        pep_path = pep_dir / f"{tag}.pdb"
        write_prot_to_pdb(
            (coors_nm[0][mask[0]] * NM_TO_ANG).cpu().numpy(),
            str(pep_path),
            aatype=aatype[0][mask[0]].cpu().numpy(),
            overwrite=True, no_indexing=True,
        )
        row["peptide_pdb"] = str(pep_path)
        # Authoritative filename stem for the downstream Rosetta stage: a guided edit's name
        # cannot be reconstructed from the grid columns alone.
        row["pdb_tag"] = tag
        if args.complex_pdb_dir and frame is not None:
            from complex_frame import read_atom_lines, write_complex

            cpx = Path(args.complex_pdb_dir) / f"{tag}__complex.pdb"
            write_complex(cpx, frame["receptor_lines"], read_atom_lines(pep_path),
                          frame["binder_chain"], frame["r"], frame["t"])
            row["complex_pdb"] = str(cpx)
            row["input_complex_pdb"] = frame["input_pdb"]
            row["binder_chain"] = frame["binder_chain"]
            row["target_chains"] = ",".join(frame["target_chains"])
            row["frame_residual_A"] = round(float(frame["residual_A"]), 4)
    return row


if __name__ == "__main__":
    main()
