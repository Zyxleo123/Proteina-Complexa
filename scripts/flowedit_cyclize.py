#!/usr/bin/env python3
"""FlowEdit LP -> CP: translate a bound linear peptide into a cyclic one.

Runs the FlowEdit ODE (arXiv:2412.08629, see `proteinfoundation.editing.flowedit` for the
method and the two adaptations this model forces) on real bound linear peptides, decodes
each result, and grades it against the fixed acceptance contract in
`proteinfoundation.editing.contract`. Writes one JSONL row per candidate; resumable by
`run_key`.

The checkpoint dependency, stated plainly
-----------------------------------------
FlowEdit needs a SOURCE velocity field and a TARGET velocity field over the SAME
representation. Here both are one network under two conditionings: `cyclization_type_cond`
= LINEAR for the source, the requested chemistry for the target. That single key drives both
the type embedding and the ring positional encoding (LINEAR yields the all-zero ring
feature), so the receptor conditioning is identical between the two branches by
construction, not by careful matching.

That means the checkpoint must be one that was TRAINED on the LINEAR token -- i.e. an
LP-mixing arm on the shared LP+CP autoencoder. A CP-only checkpoint predates LINEAR, its
embedding row for it is an untrained zero, and `CyclizationTypeSeqFeat.forward` raises
rather than silently returning a zero vector. `--require-trained-linear` (default on) turns
that into an up-front refusal with a readable message instead of a crash mid-integration.

What "the original LP" means here, and why two references are recorded
---------------------------------------------------------------------
The contract compares against the INPUT peptide. But the latent track is an autoencoder
code, so decoding an UNEDITED latent already differs from the crystal by the AE's
reconstruction error. If the AE alone mis-decodes two residues, the contract's <= 2
substitution budget is spent before any editing happens.

So every row carries both:

  * `*` (no prefix) -- versus the true input. This is the contract, and what is reported.
  * `ae_rt_*` -- the AE round trip alone: decode `x_1` with no edit. This is the FLOOR. A
    substitution count at or above the budget here means the arm cannot pass for reasons
    that have nothing to do with FlowEdit, and saying so is the difference between a
    measurement and a misattribution.

CA RMSD needs no such care: `bb_ca`'s clean sample is `coords_nm[:, :, 1, :]`, the true
input CA trace, so the CA comparison never passes through the AE.

Frames
------
Everything is read in the `coords_nm` frame and converted to Angstrom by *10: the binder
input (`coords_nm`), the receptor (`x_target`, which is `coords_nm[target_mask]`), and the
decoder output (already Angstrom, same frame). `coords` and `coords_nm` are NOT one tensor
scaled by 10 -- they are different rigid frames -- so nothing here touches `coords`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import lightning as L
import torch

from proteinfoundation.cyclization.constants import (
    LINEAR,
    NAME_TO_CYCLIZATION_TYPE,
)
from proteinfoundation.datasets.structure_data import structure_collate_fn
from proteinfoundation.editing import contract as C
from proteinfoundation.editing.flowedit import flowedit_simulate, ode_step_params
from proteinfoundation.eval.cyclic_reconstruction_metrics import NM_TO_ANG
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting

sys.path.insert(0, str(Path(__file__).parent))
from sdedit_cyclize import (  # noqa: E402  (path must be set first)
    CA_IDX,
    T_START_MAX,
    _resolve_sampling_cfg,
    build_edit_schedule,
    to_device,
)

TYPE_NAMES = dict(NAME_TO_CYCLIZATION_TYPE)


# --------------------------------------------------------------------------- helpers


def binder_endpoints(mask_row: torch.Tensor) -> tuple[int, int]:
    """(first, last) valid residue: the terminal anchors.

    CPSea is terminal-only -- every training ring bonds at (0, L-1) and the design path
    derives endpoints the same way -- so these are the only anchors the model can honour.
    Holding them fixed is also what keeps "terminal bond placement unchanged" true in the
    primary comparison.
    """
    idx = torch.nonzero(mask_row, as_tuple=False).flatten()
    if len(idx) == 0:
        raise ValueError("empty binder mask")
    return int(idx[0]), int(idx[-1])


def _strip_batch(t):
    """Drop a leading singleton batch dim if present."""
    return t[0] if (hasattr(t, "dim") and t.dim() >= 3 and t.shape[0] == 1) else t


def receptor_arrays(batch) -> tuple:
    """Receptor atom37 (Angstrom), aatype and present-atom mask, in the coords_nm frame."""
    if "x_target" not in batch:
        raise KeyError(
            "batch has no `x_target`; contacts need the receptor. Is "
            "ExtractTargetCoordinatesTransform in the transform stack?"
        )
    xyz = _strip_batch(batch["x_target"]) * NM_TO_ANG
    aa = _strip_batch(batch["seq_target"])
    m = batch.get("target_mask")
    m = _strip_batch(m) if m is not None else None
    if aa.dim() > 1:
        aa = aa[0]
    return xyz, aa, m


def decode(model, x, mask):
    """Decode a product-space sample to (atom37 Angstrom, aatype), per batch row."""
    prots = sample_formatting(
        x=x,
        extra_info={"mask": mask},
        ret_mode="coors37_n_aatype",
        data_modes=list(model.cfg_exp.product_flowmatcher),
        autoencoder=model.autoencoder,
    )
    return prots["coors"], prots["residue_type"]


def make_noise_fn(fm, *, n: int, mask_row: torch.Tensor, device, seeds: list[int]):
    """Per-candidate coupled-noise draws.

    Each row's noise comes from its OWN seed, so a candidate's result does not depend on
    which batch it shared or where it sat in it. Drawing the whole batch from one generator
    would make every number reproducible only at the same batch size and position, which
    quietly breaks resume (a resumed shard batches whatever is left).
    """
    gens = [torch.Generator().manual_seed(int(s) & 0x7FFF_FFFF) for s in seeds]
    fork_devices = [device] if torch.device(device).type == "cuda" else []

    def noise_fn(step: int, draw: int):
        parts = []
        for g in gens:
            sub = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item())
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(sub)
                parts.append(fm.sample_noise(n=n, shape=(1,), mask=mask_row, device=device))
        return {dm: torch.cat([p[dm] for p in parts], dim=0) for dm in fm.data_modes}

    return noise_fn


def resolve_protected(spec: str, iface: set[int], anchors: tuple[int, int]) -> list[int]:
    """Positions that must not change.

    `interface` protects the original interface residues EXCEPT the two anchors: a terminal
    disulfide needs CYS at both ends, so protecting the anchors makes the request
    unsatisfiable rather than strict, and an unsatisfiable gate measures nothing.
    """
    spec = (spec or "").strip()
    if not spec or spec.lower() == "none":
        return []
    if spec.lower() == "interface":
        return sorted(iface - set(anchors))
    out = []
    for tok in spec.replace(",", " ").split():
        out.append(int(tok))
    return sorted(set(out))


# --------------------------------------------------------------------------- one batch


def run_setting(
    model,
    sample,
    *,
    cyc_name: str,
    type_idx: int,
    t_ca: float,
    t_lat: float,
    edit_end_frac: float,
    n_avg: int,
    seeds: list[int],
    nsteps: int,
    self_cond: bool,
    guidance_w: float,
    sampling_model_args: dict,
    n_recycle: int,
    protected_spec: str,
    device,
    method: str = "flowedit",
) -> list[dict]:
    """One (example, chemistry, setting) cell across `seeds`, batched into one forward set.

    `method` selects the sampler, and both options run through ONE loop so the arms differ
    only where they are meant to:
      * `flowedit` -- start at the clean input, integrate the velocity DIFFERENCE.
      * `sdedit`   -- start at the NOISED input, integrate the plain target field.
    Same schedule, same integrator, same decode, same contract. A separate SDEdit script
    would also differ in incidental ways, and those would land in the comparison.
    """
    fm = model.fm
    B = len(seeds)

    # The repo's own collate does the batching, so every field is shaped the way the model
    # expects rather than by a hand-rolled tensor repeat.
    batch = to_device(structure_collate_fn([sample] * B), device)
    mask = batch["mask"].bool() if "mask" in batch else batch["coord_mask"][..., CA_IDX].bool()
    batch["mask"] = mask
    n = mask.shape[1]

    i_anchor, j_anchor = binder_endpoints(mask[0])

    # Input (true) binder arrays, and the receptor. All coords_nm frame -> Angstrom.
    in_xyz = batch["coords_nm"][0] * NM_TO_ANG
    in_aa = batch["residue_type"][0]
    in_atom_mask = batch["coord_mask"][0].bool() if "coord_mask" in batch else None
    tgt_xyz, tgt_aa, tgt_atom_mask = receptor_arrays(batch)

    base = {k: v for k, v in batch.items() if k not in ("x_1", "x_0", "x_t", "t", "x_sc")}
    base["mask"] = mask

    with torch.no_grad():
        # Clean source. Latent target is the posterior MEAN: posterior sampling would add
        # noise this experiment exists to control.
        base = add_clean_samples(
            base, model.cfg_exp.product_flowmatcher,
            autoencoder=model.autoencoder, local_latent_target="mean",
        )
        x_src = fm._apply_mask(base["x_1"], mask)

        # The AE round-trip floor: decode the UNEDITED latent. One decode per cell.
        rt_xyz, rt_aa = decode(model, x_src, mask)

        # Source and target differ in exactly one key. Separate dicts because the loop
        # writes x_t/t/x_sc into each.
        batch_src = dict(base)
        batch_tar = dict(base)
        batch_src["cyclization_type_cond"] = torch.full((B,), LINEAR, dtype=torch.long, device=device)
        batch_tar["cyclization_type_cond"] = torch.full((B,), int(type_idx), dtype=torch.long, device=device)

        t_start = {"bb_ca": min(float(t_ca), T_START_MAX),
                   "local_latents": min(float(t_lat), T_START_MAX)}
        missing = set(fm.data_modes) - set(t_start)
        if missing:
            raise RuntimeError(f"no start time supplied for data mode(s) {missing}")

        # Same schedule construction as the SDEdit arm, so the two arms differ in the
        # sampler and nothing else.
        ts, _gt = build_edit_schedule(fm, nsteps, sampling_model_args, t_start)
        ts = {k: v.to(device) for k, v in ts.items()}
        step_params = ode_step_params(fm.data_modes, sampling_model_args)

        edit_end_step = int(round(float(edit_end_frac) * nsteps))
        noise_fn = make_noise_fn(fm, n=n, mask_row=mask[:1], device=device, seeds=seeds)

        x_init = None
        if method == "sdedit":
            # The SDEdit control: the state IS the noised input, and no delta is applied.
            # The noise comes from the same per-candidate generator the delta would have
            # used, so a seed means the same thing in both arms.
            x0 = noise_fn(0, 0)
            t_vec = {dm: torch.full((B,), float(t_start[dm]), device=device)
                     for dm in fm.data_modes}
            x_init = fm.interpolate(x_0=x0, x_1=x_src, t=t_vec, mask=mask)
            edit_end_step = 0
        elif method != "flowedit":
            raise ValueError(f"unknown method {method!r}; expected 'flowedit' or 'sdedit'")

        out = flowedit_simulate(
            fm,
            batch_src=batch_src,
            batch_tar=batch_tar,
            x_src=x_src,
            x_init=x_init,
            mask=mask,
            ts=ts,
            step_params=step_params,
            predict_for_sampling=partial(model.predict_for_sampling, n_recycle=n_recycle),
            nsteps=nsteps,
            edit_end_step=edit_end_step,
            n_avg=int(n_avg),
            guidance_w=float(guidance_w),
            self_cond=bool(self_cond),
            noise_fn=noise_fn,
        )
        ed_xyz, ed_aa = decode(model, out["x"], mask)

    # ---- score each candidate -------------------------------------------------
    keep = mask[0].cpu().numpy().astype(bool)
    rows = []
    for b, seed in enumerate(seeds):
        # Contacts of the true input define the interface, so protection and the
        # interface-RMSD subset are both anchored on the input, not on a prediction.
        c_in = C.heavy_atom_contacts(in_xyz[keep], in_aa[keep], tgt_xyz, tgt_aa,
                                     pep_atom_mask=None if in_atom_mask is None else in_atom_mask[keep],
                                     tgt_atom_mask=tgt_atom_mask)
        iface = C.interface_residues(c_in)
        protected = resolve_protected(protected_spec, iface, (i_anchor, j_anchor))

        row = C.evaluate(
            before_xyz_A=in_xyz[keep], before_aatype=in_aa[keep],
            after_xyz_A=ed_xyz[b][keep], after_aatype=ed_aa[b][keep],
            tgt_xyz_A=tgt_xyz, tgt_aatype=tgt_aa,
            cyc_type=cyc_name, anchor_i=i_anchor, anchor_j=j_anchor,
            protected=protected, ddg_REU=None,
            before_atom_mask=None if in_atom_mask is None else in_atom_mask[keep],
            after_atom_mask=None, tgt_atom_mask=tgt_atom_mask,
        )

        # The AE round-trip floor, same code path, no edit.
        rt = C.evaluate(
            before_xyz_A=in_xyz[keep], before_aatype=in_aa[keep],
            after_xyz_A=rt_xyz[b][keep], after_aatype=rt_aa[b][keep],
            tgt_xyz_A=tgt_xyz, tgt_aatype=tgt_aa,
            cyc_type=cyc_name, anchor_i=i_anchor, anchor_j=j_anchor,
            protected=protected, ddg_REU=None,
            before_atom_mask=None if in_atom_mask is None else in_atom_mask[keep],
            after_atom_mask=None, tgt_atom_mask=tgt_atom_mask,
        )
        for k in ("seq_identity", "n_substitutions", "ca_rmsd_A", "interface_ca_rmsd_A",
                  "contact_retention", "substitutions"):
            row[f"ae_rt_{k}"] = rt[k]
        # Budget already consumed by the autoencoder, before any editing.
        row["ae_rt_consumes_budget"] = bool(
            rt["n_substitutions"] > C.CONTRACT["max_substitutions"]
        )

        row.update(
            seed=int(seed), t_ca_start=float(t_ca), t_lat_start=float(t_lat),
            edit_end_frac=float(edit_end_frac), edit_end_step=edit_end_step,
            n_avg=int(n_avg), nsteps=int(nsteps), guidance_w=float(guidance_w),
            self_cond=bool(self_cond), method=str(method),
        )
        rows.append((row, ed_xyz[b][keep].detach().cpu(), ed_aa[b][keep].detach().cpu()))
    return rows


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-ckpt", required=True, help="Flow checkpoint FILE (an LP-mixing arm).")
    ap.add_argument("--ae-ckpt", default=None,
                    help="Autoencoder checkpoint. MUST be the AE this flow was trained "
                         "against (the shared LP+CP AE); mixing them reinterprets the "
                         "latent track silently.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config-name", default="example/training_cpsea_peptide_smoke")
    ap.add_argument("--sampling-config", default="pipeline/model_sampling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cyc-types", nargs="+", default=["mainchain", "disulfide", "isopeptide"])
    ap.add_argument("--t-ca", nargs="+", type=float, default=[0.8],
                    help="Per-track edit start for bb_ca. 1.0 freezes the backbone.")
    ap.add_argument("--t-lat", nargs="+", type=float, default=[0.6],
                    help="Per-track edit start for local_latents. 1.0 freezes the sequence.")
    ap.add_argument("--edit-end-frac", nargs="+", type=float, default=[1.0],
                    help="Fraction of nsteps to keep applying the velocity difference "
                         "(the paper's n_min). 1.0 edits the whole interval; below it the "
                         "plain target field finishes.")
    ap.add_argument("--n-avg", nargs="+", type=int, default=[1],
                    help="Coupled-noise draws averaged per step. Each costs 2 forwards.")
    ap.add_argument("--nsteps", type=int, default=0, help="0 = the sampling config's own nsteps.")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--batch-seeds", type=int, default=8,
                    help="Candidates per forward batch. Per-candidate RNG makes results "
                         "independent of this value.")
    ap.add_argument("--guidance-w", type=float, default=1.0,
                    help="CFG weight inside each branch. 1.0 is faithful FlowEdit.")
    ap.add_argument("--self-cond", default=None, type=lambda s: s.lower() in ("1", "true", "yes"),
                    help="Default: follow the sampling config.")
    ap.add_argument("--method", default="flowedit", choices=["flowedit", "sdedit"],
                    help="flowedit = integrate the velocity difference from the clean "
                         "input; sdedit = integrate the plain target field from the noised "
                         "input. Both share this script's decode and contract.")
    ap.add_argument("--protected", default="none",
                    help="'none', 'interface', or explicit 0-based indices.")
    ap.add_argument("--pdb-dir", default=None, help="Write peptide PDBs for kept candidates.")
    ap.add_argument("--write-if", default="geom", choices=["accepted", "geom", "all", "none"],
                    help="Which candidates get a PDB. 'geom' = passed every geometry and "
                         "sequence gate (Rosetta has not run yet at this point).")
    ap.add_argument("--require-trained-linear", default=True,
                    type=lambda s: s.lower() in ("1", "true", "yes"),
                    help="Refuse a checkpoint whose LINEAR embedding row is untrained.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    sys.path.insert(0, str(Path(__file__).parent))
    from eval_ae_roundtrip import build_dataset

    import hydra
    with hydra.initialize(version_base=None, config_path="../configs"):
        samp = hydra.compose(config_name=args.sampling_config)
    samp = _resolve_sampling_cfg(samp)
    sampler_args, sampling_model_args = samp.args, samp.model
    nsteps = args.nsteps or int(sampler_args.nsteps)
    self_cond = bool(sampler_args.self_cond) if args.self_cond is None else bool(args.self_cond)
    n_recycle = int(samp.get("n_recycle", 0))

    dataset, _ = build_dataset(args.config_name, args.metadata, keep_rotation=False)
    meta = dataset.metadata
    all_idx = list(range(len(dataset)))
    if args.limit:
        all_idx = all_idx[: args.limit]
    my_idx = all_idx[args.shard_index :: args.shard_count]

    settings = [
        (tc, tl, ee, na)
        for tc in args.t_ca for tl in args.t_lat
        for ee in args.edit_end_frac for na in args.n_avg
    ]
    n_cand = len(my_idx) * len(args.cyc_types) * len(settings) * len(args.seeds)
    # Forward count per candidate is NOT nsteps: the delta branch costs 2 forwards per
    # noise draw (source field + target field), and only the tail past `edit_end_step`
    # costs one. Getting this wrong understates the job by exactly the factor that decides
    # whether the sweep fits in its walltime.
    per_cell = len(my_idx) * len(args.cyc_types) * len(args.seeds)
    fwd = 0
    for (_tc, _tl, ee, na) in settings:
        # The sdedit arm applies no delta (edit_end_step is forced to 0), so it costs one
        # forward per step regardless of edit_end_frac or n_avg. Billing it at the FlowEdit
        # rate would overstate the control by 2x and make the comparison of COST wrong.
        e = 0 if args.method == "sdedit" else int(round(min(max(float(ee), 0.0), 1.0) * nsteps))
        fwd += (e * 2 * int(na) + (nsteps - e)) * per_cell
    print(f"{len(my_idx)}/{len(all_idx)} peptides x {len(args.cyc_types)} chemistries x "
          f"{len(settings)} settings x {len(args.seeds)} seeds = {n_cand} candidates "
          f"({fwd / 1000:.0f}k network forwards, {fwd / max(n_cand, 1):.0f}/candidate), "
          f"nsteps={nsteps} self_cond={self_cond} batch_seeds={args.batch_seeds}",
          flush=True)
    if args.dry_run:
        print("DRY RUN OK", flush=True)
        return

    if args.ae_ckpt:
        os.environ["CPSEA_AE_CKPT_PATH"] = args.ae_ckpt
    if not os.path.isfile(args.flow_ckpt):
        raise SystemExit(f"FATAL: flow checkpoint not found: {args.flow_ckpt}")
    print(f"flow checkpoint: {args.flow_ckpt}", flush=True)
    print(f"autoencoder    : {os.environ.get('CPSEA_AE_CKPT_PATH', '(from config)')}", flush=True)

    model = Proteina.load_from_checkpoint(
        args.flow_ckpt, strict=False,
        autoencoder_ckpt_path=os.environ.get("CPSEA_AE_CKPT_PATH"),
    )
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    if model.autoencoder is None:
        raise SystemExit("FATAL: checkpoint has no autoencoder; latent editing is impossible.")
    if not getattr(model, "cyclization_type_conditioning", False):
        raise SystemExit(
            "FATAL: this checkpoint has cyclization.type_conditioning OFF. The requested "
            "chemistry cannot reach the denoiser, so the source and target fields would be "
            "identical and every edit would be a no-op by construction."
        )

    # Refuse a pre-LINEAR checkpoint up front rather than crashing mid-integration.
    if args.require_trained_linear:
        bad = []
        for name, mod in model.named_modules():
            trained = getattr(mod, "_trained_num_types", None)
            if trained is not None and trained <= LINEAR:
                bad.append(f"{name} (trained with {trained} conditioning types)")
        if bad:
            raise SystemExit(
                "FATAL: this checkpoint predates the LINEAR conditioning token, so its "
                "source field does not exist:\n  " + "\n  ".join(bad) +
                "\nFlowEdit needs a model trained on LINEAR (an LP-mixing arm on the shared "
                "LP+CP autoencoder). Pass --require-trained-linear=false only to inspect "
                "machinery; any result from it is not a FlowEdit result."
            )

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
        print(f"resuming: {len(done)} candidates already done", flush=True)
    if args.pdb_dir:
        Path(args.pdb_dir).mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_written = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for idx in my_idx:
            example_id = str(meta.iloc[idx]["example_id"])
            sample = dataset[idx]
            if sample is None:
                print(f"  SKIP {example_id}: failed to load", flush=True)
                continue

            for cyc_name in args.cyc_types:
                if cyc_name not in TYPE_NAMES:
                    raise SystemExit(f"FATAL: unknown chemistry {cyc_name!r}")
                type_idx = TYPE_NAMES[cyc_name]
                for (t_ca, t_lat, ee, na) in settings:
                    tag = f"{t_ca}|{t_lat}|{ee}|{na}"
                    todo = [s for s in args.seeds
                            if f"{args.method}|{example_id}|{cyc_name}|{tag}|{s}" not in done]
                    if not todo:
                        continue
                    for s0 in range(0, len(todo), max(1, args.batch_seeds)):
                        chunk = todo[s0 : s0 + max(1, args.batch_seeds)]
                        keys = [f"{args.method}|{example_id}|{cyc_name}|{tag}|{s}"
                                for s in chunk]
                        L.seed_everything(int(chunk[0]), workers=True)
                        try:
                            results = run_setting(
                                model, sample, cyc_name=cyc_name, type_idx=type_idx,
                                t_ca=t_ca, t_lat=t_lat, edit_end_frac=ee, n_avg=na,
                                seeds=chunk, nsteps=nsteps, self_cond=self_cond,
                                guidance_w=args.guidance_w,
                                sampling_model_args=sampling_model_args,
                                n_recycle=n_recycle, protected_spec=args.protected,
                                device=args.device, method=args.method,
                            )
                        except Exception as exc:  # noqa: BLE001
                            n_fail += len(chunk)
                            for k, s in zip(keys, chunk):
                                fh.write(json.dumps({
                                    "run_key": k, "example_id": example_id, "status": "failed",
                                    "method": args.method, "cyc_type": cyc_name,
                                    "t_ca_start": t_ca, "t_lat_start": t_lat,
                                    "edit_end_frac": ee, "n_avg": na, "seed": s,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }) + "\n")
                            fh.flush()
                            print(f"  FAIL {keys[0]}..: {type(exc).__name__}: {exc}", flush=True)
                            continue

                        for (row, xyz, aa), k in zip(results, keys):
                            row.update(run_key=k, example_id=example_id, status="ok")
                            for col in ("peptide_length", "receptor_length", "nc_gap_angstrom"):
                                if col in meta.columns:
                                    v = meta.iloc[idx][col]
                                    row[f"input_{col}"] = v.item() if hasattr(v, "item") else v
                            want = (
                                args.write_if == "all"
                                or (args.write_if == "geom" and row["pass_geometry_and_sequence"])
                                or (args.write_if == "accepted" and row["accepted"])
                            )
                            if args.pdb_dir and want:
                                p = Path(args.pdb_dir) / (k.replace("|", "_") + ".pt")
                                torch.save({"coors_A": xyz, "aatype": aa,
                                            "example_id": example_id, "run_key": k}, p)
                                row["candidate_file"] = str(p)
                                n_written += 1
                            fh.write(json.dumps(row) + "\n")
                            n_ok += 1
                        fh.flush()
                        if n_ok % 50 < len(chunk):
                            rate = n_ok / max(time.time() - t0, 1e-6)
                            print(f"  {n_ok} candidates, {n_written} kept "
                                  f"({time.time() - t0:.0f}s, {rate:.2f}/s)", flush=True)

    print(f"done: {n_ok} candidates, {n_fail} failed, {n_written} kept, "
          f"{time.time() - t0:.0f}s", flush=True)
    # A wholly-failed shard must break an afterok chain rather than exit 0 with a file of
    # error rows that a summary job would happily aggregate into nothing.
    if n_ok == 0 and n_fail > 0:
        raise SystemExit(f"FATAL: all {n_fail} candidates failed -- see the errors above.")


if __name__ == "__main__":
    main()
