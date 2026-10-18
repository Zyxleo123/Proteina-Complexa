"""D-Flow for LP -> CP: optimize the source state of a frozen CP generator, not its weights.

The move
--------
The pretrained target-conditioned CPSea flow is left exactly as it is. What gets optimized
is the CONTINUOUS STATE the deterministic sampler starts from, by differentiating through
the whole integration trajectory and the autoencoder decode into a loss defined on the FINAL
decoded structure. Gradient descent on the input of a frozen generative map -- D-Flow
(Ben-Hamu et al., ICML 2024). See `scripts/dflow_rollout.py` for why the rollout needed its
own loop (checkpointing, and a deterministic integrator in place of the production SDE) and
`scripts/dflow_losses.py` for the objective and the acceptance contract.

This is NOT the per-step guidance in `scripts/sdedit_guidance.py`. That reads a gradient at
each Euler step and discards the graph, so it cannot trade a worse step now for a better
final structure. Here one graph spans the entire rollout.

Two initializations, labelled separately and never pooled
--------------------------------------------------------
    source   t_start = (0, 0). The free variable is the source noise itself; the rollout is
             the full production schedule. This is D-Flow as published, and the peptide is
             generated de novo under the receptor's conditioning.
    sdedit   t_start = (t_ca, t_lat) > 0. The free variable is an intermediate SDEdit state
             built from the ENCODED input peptide. This is an adaptation, not the published
             method, and it inherits SDEdit's own bias toward the input -- so its numbers
             are reported as their own arm.

Restarts are independent rollouts, not a batch dimension: each carries its own source state
and its own retained graph, and the graph is the binding memory constraint.

Every restart is scored twice, and the pair is the point
-------------------------------------------------------
`stage="init"` is the restart's UNOPTIMIZED source rolled out and scored; `stage="opt"` is
the same source after optimization. Same seed, same schedule, same everything else. Without
that pairing a headline number cannot distinguish "the optimizer worked" from "16 restarts
found a good sample", which is the first thing anyone should ask of a method whose selling
point is optimization.

Coarse-grid optimization, fine-grid verdict
------------------------------------------
Optimization runs on `--opt-nsteps` (short, so the graph fits and iterations are cheap) and
every reported metric is recomputed on `--eval-nsteps` (the production grid). Both grids'
ring closure is recorded: a candidate that closes only on the coarse grid is an integration
artifact, flagged `grid_consistent=False`, and must not be counted.
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

from proteinfoundation.cyclization.constants import CYCLIZATION_TYPE_TO_NAME
from proteinfoundation.datasets.structure_data import structure_collate_fn
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.sample_utils import add_clean_samples

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script_utils"))
from dflow_losses import (  # noqa: E402
    CONTRACT,
    NM_TO_ANG,
    Objective,
    objective_tag,
    parse_weights,
)
from dflow_pdb import RingUndeclarable, declare_ring, peptide_serial_offset, verify_ring  # noqa: E402
from dflow_rollout import DiffRollout, decode_final  # noqa: E402

from dflow_accept import accept  # noqa: E402

CA_IDX = 1
TYPE_NAMES = {"mainchain": 0, "disulfide": 1, "isopeptide": 2}


def _resolve_sampling_cfg(cfg):
    """The node holding `args`/`model`, top level or nested under the config's package."""
    if "args" in cfg and "model" in cfg:
        return cfg
    for key in list(cfg.keys()):
        child = cfg[key]
        if hasattr(child, "keys") and "args" in child and "model" in child:
            return child
    raise SystemExit(
        f"FATAL: sampling config has no node with both `args` and `model`; keys: {list(cfg.keys())}"
    )


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


# ---------------------------------------------------------------------------- optimization
def optimize_source(rollout, objective, ae, mask, z_init, args, log_prefix=""):
    """Adam on the source state. Returns (best_z, history).

    Separate learning rates per track because the two are not comparable quantities: bb_ca
    is CA displacement in nm, local_latents is an 8-dim unitless code. One shared lr means
    whichever track happens to carry larger numbers absorbs the whole step.

    `best` is by OBJECTIVE value, not by gate count. Selecting on the gates would tune the
    stopping point to the acceptance test and inflate it; the objective is what is being
    descended, so it is what "best" means.
    """
    z = {dm: v.detach().clone().requires_grad_(True) for dm, v in z_init.items()}
    groups = []
    for dm, v in z.items():
        lr = args.lr_ca if dm == "bb_ca" else args.lr_lat
        groups.append({"params": [v], "lr": float(lr)})
    opt = torch.optim.Adam(groups)
    sched = None
    if args.lr_decay < 1.0:
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(args.lr_decay))

    best = {"loss": float("inf"), "z": {dm: v.detach().clone() for dm, v in z.items()}, "it": -1}
    history = []
    for it in range(int(args.opt_steps)):
        opt.zero_grad(set_to_none=True)
        x_final, _ = rollout(z, grad=True)
        decoded = decode_final(ae, x_final, mask)
        total, terms, diag = objective(decoded, z=z, z_init=z_init)
        if not torch.isfinite(total):
            print(f"{log_prefix}  it {it}: objective is not finite, stopping this restart",
                  flush=True)
            break
        total.backward()
        gnorm = float(
            torch.nn.utils.clip_grad_norm_([v for v in z.values()], float(args.grad_clip))
        )
        if gnorm == 0.0:
            # A dead gradient is a result, not a warning to skip past: it means every active
            # term is flat here, and continuing would just burn the budget.
            print(f"{log_prefix}  it {it}: zero gradient on the source state, stopping",
                  flush=True)
            history.append({"it": it, "loss": float(total.detach()), "grad_norm": 0.0, **terms})
            break
        opt.step()
        if sched is not None:
            sched.step()
        loss_v = float(total.detach())
        history.append({"it": it, "loss": loss_v, "grad_norm": gnorm, **terms,
                        **{k: v for k, v in diag.items() if isinstance(v, (int, float))}})
        if loss_v < best["loss"]:
            best = {"loss": loss_v, "z": {dm: v.detach().clone() for dm, v in z.items()}, "it": it}
        if args.log_every and (it % args.log_every == 0 or it == int(args.opt_steps) - 1):
            print(f"{log_prefix}  it {it:3d} loss {loss_v:.4f} |g| {gnorm:.3e} "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(terms.items())), flush=True)
    return best, history


# ------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate_candidate(rollout, ae, mask, batch, z, objective, cyc_type):
    """Roll out `z` without a graph, decode, and score every contract metric."""
    x_final, _ = rollout(z, grad=False)
    decoded = decode_final(ae, x_final, mask)
    total, terms, diag = objective(decoded, z=z, z_init=z)
    m = mask[0].bool()
    row = accept(
        pep_coors_nm=decoded["coors_nm"][0][m],
        pep_amask=(decoded["atom_mask"].bool() & mask[..., None])[0][m],
        pep_aatype=decoded["residue_type"][0][m].long(),
        ref_coors_nm=batch["coords_nm"][0][m],
        ref_amask=(batch["coord_mask"][0][m].bool() if "coord_mask" in batch
                   else torch.ones_like(decoded["atom_mask"][0][m]).bool()),
        ref_aatype=objective.ref_aatype[0][m].long(),
        tgt_coors_nm=objective.tgt,
        tgt_amask=objective.tgt_amask,
        cyc_type=cyc_type,
        cyc_i=objective.i,
        cyc_j=objective.j,
        protect=[int(p) for p in objective.protect] if objective.protect is not None else None,
    )
    row["objective"] = float(total)
    row.update({f"loss/{k}": v for k, v in terms.items()})
    row.update({f"diag/{k}": v for k, v in diag.items()})
    return row, decoded


def write_candidate(decoded, mask, batch, objective, cyc_type, tag, args, frame):
    """Peptide PDB (+ complex) with the ring DECLARED, verified by reading the file back."""
    from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

    m = mask[0].bool()
    coors_A = (decoded["coors_nm"][0][m] * NM_TO_ANG).cpu().numpy()
    aatype = decoded["residue_type"][0][m].cpu().numpy()
    out: dict = {}

    pep_dir = Path(args.pdb_dir) if args.pdb_dir else Path(args.complex_pdb_dir) / "_peptides"
    pep_dir.mkdir(parents=True, exist_ok=True)
    pep_path = pep_dir / f"{tag}.pdb"
    write_prot_to_pdb(coors_A, str(pep_path), aatype=aatype, overwrite=True, no_indexing=True)
    out["peptide_pdb"] = str(pep_path)
    out["pdb_tag"] = tag

    # write_prot_to_pdb numbers residues 1..L, so endpoint k is resSeq k+1.
    res_i, res_j = objective.i + 1, objective.j + 1
    aa_i, aa_j = int(aatype[objective.i]), int(aatype[objective.j])
    try:
        declare_ring(pep_path, res_i, res_j, cyc_type, aa_i, aa_j)
    except RingUndeclarable as exc:
        # Not fatal, and deliberately not silent: an undeclared ring is a failed gate that
        # the report needs to be able to count, and the reason is the diagnosis.
        out["ring_declare_error"] = f"{type(exc).__name__}: {exc}"
    v = verify_ring(pep_path, res_i, res_j, cyc_type)
    out["gate_ring_declared"] = bool(v["declared"])
    out["file_bond_dist_A"] = v["dist_A"]
    out["file_bond_in_window"] = bool(v["in_window"])
    if v.get("reason"):
        out["ring_verify_reason"] = v["reason"]

    if args.complex_pdb_dir and frame is not None:
        from complex_frame import read_atom_lines, write_complex

        cpx = Path(args.complex_pdb_dir) / f"{tag}__complex.pdb"
        write_complex(cpx, frame["receptor_lines"], read_atom_lines(pep_path),
                      frame["binder_chain"], frame["r"], frame["t"])
        # write_complex renumbers every serial, so the peptide's CONECT has to be rebuilt
        # against the complex's numbering rather than copied over.
        try:
            off = peptide_serial_offset(cpx, frame["binder_chain"])
            declare_ring(cpx, res_i, res_j, cyc_type, aa_i, aa_j, chain=frame["binder_chain"])
            cv = verify_ring(cpx, res_i, res_j, cyc_type, chain=frame["binder_chain"])
            out["complex_ring_declared"] = bool(cv["declared"])
            out["complex_serial_offset"] = off
        except RingUndeclarable as exc:
            out["complex_ring_declared"] = False
            out["complex_ring_error"] = f"{type(exc).__name__}: {exc}"
        out["complex_pdb"] = str(cpx)
        out["input_complex_pdb"] = frame["input_pdb"]
        out["binder_chain"] = frame["binder_chain"]
        out["target_chains"] = ",".join(frame["target_chains"])
        out["frame_residual_A"] = round(float(frame["residual_A"]), 4)
    return out


def build_frame(base_batch, mask, meta_row, tol_A):
    """Loader -> input-PDB transform. Recoverable HERE and nowhere else: two loads of one
    example land in frames differing by a rigid transform, so a peptide written without its
    receptor can never be re-united with it. See scripts/complex_frame.py."""
    from complex_frame import crystal_transform

    input_pdb = Path(str(meta_row["path"]))
    if not input_pdb.is_file():
        print(f"  WARNING: input PDB missing: {input_pdb}; no complexes", flush=True)
        return None
    binder_chain = str(meta_row.get("binder_chain_id", "B"))
    loader_ca = (base_batch["coords_nm"][0][mask[0]][:, CA_IDX, :].detach().cpu().numpy()
                 * NM_TO_ANG).astype("float64")
    try:
        r, t, offset, resid, receptor_lines, target_chains = crystal_transform(
            loader_ca, input_pdb, binder_chain)
    except RuntimeError as exc:
        print(f"  WARNING: frame fit failed ({exc}); no complexes", flush=True)
        return None
    if resid > tol_A:
        print(f"  WARNING: frame residual {resid:.3f} A > {tol_A} A for {input_pdb.name}; "
              "no complexes", flush=True)
        return None
    return {"r": r, "t": t, "offset": offset, "residual_A": resid,
            "receptor_lines": receptor_lines, "target_chains": target_chains,
            "binder_chain": binder_chain, "input_pdb": str(input_pdb)}


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--ckpt-name", default="last-EMA.ckpt")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config-name", default="example/training_cpsea_peptide_smoke")
    ap.add_argument("--sampling-config", default="pipeline/model_sampling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--complex-pdb-dir", default=None,
                    help="Receptor+peptide complexes in the input PDB's frame, needed for "
                         "Rosetta. The ring CONECT is rebuilt against the complex numbering.")
    ap.add_argument("--z-dir", default=None,
                    help="Save each candidate's optimized source state (a few KB). Without "
                         "it a finalist cannot be re-rolled on a different grid later.")
    ap.add_argument("--complex-align-tol-A", type=float, default=0.05)
    ap.add_argument("--cyc-types", nargs="+", default=["mainchain", "disulfide", "isopeptide"])

    g = ap.add_argument_group("initialization")
    g.add_argument("--init-mode", default="source", choices=["source", "sdedit"],
                   help="source: optimize the source noise, t_start=(0,0) -- D-Flow as "
                        "published. sdedit: optimize an intermediate SDEdit state, a "
                        "separately-labelled adaptation.")
    g.add_argument("--t-ca", type=float, default=0.8,
                   help="sdedit mode only: backbone-track start time.")
    g.add_argument("--t-lat", type=float, default=0.4,
                   help="sdedit mode only: latent-track start time (the substitution budget).")

    g = ap.add_argument_group("sampler")
    g.add_argument("--sampler-mode", default="vf", choices=["vf", "sc"],
                   help="vf: the probability-flow ODE, deterministic with no noise term -- "
                        "what D-Flow assumes, and the default. sc: the PRODUCTION SDE with "
                        "its noise sequence frozen by reseeding, which is the sampler every "
                        "earlier number on this repo was produced with.")
    g.add_argument("--opt-nsteps", type=int, default=30,
                   help="Integration steps during optimization. Short on purpose.")
    g.add_argument("--eval-nsteps", type=int, default=0,
                   help="Integration steps for every reported metric. 0 = the design "
                        "sampler's own nsteps (the production grid).")
    g.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Keep every step's activations instead of recomputing them. Faster "
                        "per iteration, and out of memory at any useful --opt-nsteps.")

    g = ap.add_argument_group("optimization")
    g.add_argument("--restarts", type=int, default=16)
    g.add_argument("--opt-steps", type=int, default=60)
    g.add_argument("--lr-ca", type=float, default=0.02)
    g.add_argument("--lr-lat", type=float, default=0.05)
    g.add_argument("--lr-decay", type=float, default=1.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--loss", default="",
                   help="Comma list with optional :weight, e.g. "
                        "'anchor_ce:3,anchor_cb,bond_fb,seq_ce:2,ca_hinge,iface_hinge:2,"
                        "contact_hinge,clash,prior:0.05'. Empty = the defaults.")
    g.add_argument("--protect", nargs="+", type=int, default=None,
                   help="Peptide-local positions that may not mutate. Enforced as a hard "
                        "gate; the loss only weights them.")
    g.add_argument("--protect-w", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=0, help="Base seed; restart k uses seed+k.")
    g.add_argument("--log-every", type=int, default=10)

    g = ap.add_argument_group("run control")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--examples", nargs="+", default=None)
    g.add_argument("--shard-index", type=int, default=0)
    g.add_argument("--shard-count", type=int, default=1)
    g.add_argument("--max-seconds", type=float, default=0.0,
                   help="Stop cleanly after this much wall clock and exit 0, so the job is "
                        "resubmittable rather than restarted. 0 = no limit.")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dry-run", action="store_true",
                   help="Resolve configs, dataset, objective and grid, then exit before the "
                        "checkpoint load. Catches plumbing errors without a GPU.")
    args = ap.parse_args()

    weights = parse_weights(args.loss)
    obj_tag = objective_tag(weights)
    unknown = [t for t in args.cyc_types if t not in TYPE_NAMES]
    if unknown:
        raise SystemExit(f"FATAL: unknown cyclization type(s) {unknown}; expected {sorted(TYPE_NAMES)}")
    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")
    if args.init_mode == "source" and (args.t_ca != 0.8 or args.t_lat != 0.4):
        print("NOTE: --t-ca/--t-lat are ignored in --init-mode source (t_start is (0, 0)).",
              flush=True)

    print(f"objective: {weights}", flush=True)
    print(f"tag: {obj_tag}", flush=True)

    from eval_ae_roundtrip import build_dataset

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
    if args.limit:
        all_indices = all_indices[: args.limit]
    my_indices = all_indices[args.shard_index :: args.shard_count]
    print(f"{len(dataset)} peptides from {args.metadata}; shard {args.shard_index}/"
          f"{args.shard_count} takes {len(my_indices)}", flush=True)

    with hydra.initialize("../configs", version_base=hydra.__version__):
        samp = hydra.compose(config_name=args.sampling_config)
    samp = _resolve_sampling_cfg(samp)
    sampler_args, sampling_model_args = samp.args, samp.model
    eval_nsteps = args.eval_nsteps or int(sampler_args.nsteps)
    self_cond = bool(sampler_args.self_cond)
    n_recycle = int(samp.get("n_recycle", 0))
    guidance_w = float(sampler_args.get("guidance_w", 1.0))
    print(f"sampler: mode={args.sampler_mode} opt_nsteps={args.opt_nsteps} "
          f"eval_nsteps={eval_nsteps} self_cond={self_cond} n_recycle={n_recycle}", flush=True)

    if args.dry_run:
        n = len(my_indices) * len(args.cyc_types) * args.restarts
        print(f"DRY RUN OK: {len(my_indices)} peptides x {len(args.cyc_types)} chemistries x "
              f"{args.restarts} restarts = {n} optimizations, "
              f"{args.opt_steps} updates each", flush=True)
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
        print("WARNING: cyclization.type_conditioning is OFF in this checkpoint -- the "
              "requested type cannot reach the denoiser and every chemistry arm collapses "
              "to the same run.", flush=True)
    allow_asn_gln = bool(getattr(model, "cyclization_allow_asn_gln_isopeptide", True))

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
        print(f"resuming: {len(done)} rows already done", flush=True)
    for d in (args.pdb_dir, args.complex_pdb_dir, args.z_dir):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    fm = model.fm
    predict_fn = partial(model.predict_for_sampling, n_recycle=n_recycle)
    t_start = ({dm: 0.0 for dm in fm.data_modes} if args.init_mode == "source"
               else {"bb_ca": float(args.t_ca), "local_latents": float(args.t_lat)})

    n_rows = n_fail = 0
    t_wall = time.time()
    budget_hit = False
    with out_path.open("a") as fh:
        for idx in my_indices:
            if budget_hit:
                break
            example_id = str(meta.iloc[idx]["example_id"])
            sample = dataset[idx]
            if sample is None:
                print(f"  SKIP {example_id}: failed to load", flush=True)
                continue
            base_batch = to_device(structure_collate_fn([sample]), args.device)
            mask = (base_batch["mask"].bool() if "mask" in base_batch
                    else base_batch["coord_mask"][..., CA_IDX].bool())
            base_batch["mask"] = mask
            frame = None
            if args.complex_pdb_dir:
                frame = build_frame(base_batch, mask, meta.iloc[idx], args.complex_align_tol_A)

            for cyc_name in args.cyc_types:
                if budget_hit:
                    break
                cyc_type = TYPE_NAMES[cyc_name]
                batch = {k: v for k, v in base_batch.items()
                         if k not in ("x_1", "x_0", "x_t", "t", "x_sc")}
                batch["mask"] = mask
                # The requested type MUST be stamped explicitly. An LNR batch already carries
                # UNSPECIFIED (no CONECT ring in a linear peptide) and
                # `apply_cyclization_type_conditioning` returns early whenever the key exists,
                # so relying on it runs every arm unconditioned.
                batch["cyclization_type_cond"] = torch.full(
                    (1,), cyc_type, dtype=torch.long, device=mask.device
                )
                with torch.no_grad():
                    batch = add_clean_samples(
                        batch, model.cfg_exp.product_flowmatcher,
                        autoencoder=model.autoencoder, local_latent_target="mean",
                    )
                    x_1 = (fm._apply_mask(batch["x_1"], mask) if hasattr(fm, "_apply_mask")
                           else batch["x_1"])

                try:
                    objective = Objective(
                        batch=batch, mask=mask, cyc_type=cyc_type, weights=weights,
                        allow_asn_gln=allow_asn_gln, protect=args.protect,
                        protect_w=args.protect_w,
                    )
                except SystemExit as exc:
                    # An unsatisfiable request for THIS peptide (e.g. a protected endpoint) is
                    # a recorded outcome, not a reason to kill the shard.
                    print(f"  SKIP {example_id}/{cyc_name}: {exc}", flush=True)
                    fh.write(json.dumps({
                        "run_key": f"{example_id}|{cyc_name}|{args.init_mode}|{obj_tag}|skip",
                        "example_id": example_id, "cyc_type_name": cyc_name,
                        "status": "unsatisfiable", "error": str(exc)}) + "\n")
                    fh.flush()
                    continue

                common = dict(
                    fm=fm, predict_for_sampling=predict_fn, batch=batch, mask=mask,
                    t_start=t_start, sampling_model_args=sampling_model_args,
                    guidance_w=guidance_w, sampler_mode=args.sampler_mode,
                    self_cond=self_cond,
                )
                roll_opt = DiffRollout(nsteps=int(args.opt_nsteps),
                                       grad_checkpoint=not args.no_grad_checkpoint,
                                       noise_seed=args.seed, **common)
                roll_eval = DiffRollout(nsteps=int(eval_nsteps), grad_checkpoint=False,
                                        noise_seed=args.seed, **common)

                for restart in range(int(args.restarts)):
                    base_key = f"{example_id}|{cyc_name}|{args.init_mode}|{obj_tag}|r{restart}"
                    if f"{base_key}|opt" in done:
                        continue
                    seed = int(args.seed) + restart
                    L.seed_everything(seed, workers=True)
                    t_restart = time.time()
                    try:
                        z_init = roll_opt.init_source(x_1=x_1, seed=seed)

                        # Paired control: the same source, unoptimized, on the eval grid.
                        row_init, dec_init = evaluate_candidate(
                            roll_eval, model.autoencoder, mask, batch, z_init, objective, cyc_type
                        )

                        best, history = optimize_source(
                            roll_opt, objective, model.autoencoder, mask, z_init, args,
                            log_prefix=f"  [{example_id}/{cyc_name}/r{restart}]",
                        )
                        z_best = best["z"]

                        # The verdict grid, plus the coarse grid the optimizer actually saw:
                        # a ring that closes only on the coarse grid is an artifact.
                        row_opt, dec_opt = evaluate_candidate(
                            roll_eval, model.autoencoder, mask, batch, z_best, objective, cyc_type
                        )
                        row_coarse, _ = evaluate_candidate(
                            roll_opt, model.autoencoder, mask, batch, z_best, objective, cyc_type
                        )
                    except torch.cuda.OutOfMemoryError as exc:
                        n_fail += 1
                        torch.cuda.empty_cache()
                        fh.write(json.dumps({"run_key": f"{base_key}|opt",
                                             "example_id": example_id, "status": "failed",
                                             "cyc_type_name": cyc_name, "restart": restart,
                                             "error": f"OOM: {exc}"}) + "\n")
                        fh.flush()
                        print(f"  FAIL {base_key}: OOM", flush=True)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        n_fail += 1
                        fh.write(json.dumps({"run_key": f"{base_key}|opt",
                                             "example_id": example_id, "status": "failed",
                                             "cyc_type_name": cyc_name, "restart": restart,
                                             "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                        fh.flush()
                        print(f"  FAIL {base_key}: {type(exc).__name__}: {exc}", flush=True)
                        continue

                    row_opt["coarse_gate_ring_closed"] = bool(row_coarse["gate_ring_closed"])
                    row_opt["coarse_bond_dist_A"] = row_coarse["bond_dist_A"]
                    row_opt["coarse_objective"] = row_coarse["objective"]
                    row_opt["grid_consistent"] = bool(
                        row_coarse["gate_ring_closed"] == row_opt["gate_ring_closed"]
                    )

                    stamp = dict(
                        example_id=example_id, cyc_type_name=cyc_name, status="ok",
                        init_mode=args.init_mode, objective_tag=obj_tag, restart=restart,
                        seed=seed, sampler_mode=args.sampler_mode,
                        opt_nsteps=int(args.opt_nsteps), eval_nsteps=int(eval_nsteps),
                        opt_steps=int(args.opt_steps),
                        t_ca_start=t_start["bb_ca"], t_lat_start=t_start["local_latents"],
                    )
                    for col in ("peptide_length", "receptor_length", "nc_gap_angstrom",
                                "n_cys", "cluster_id"):
                        if col in meta.columns:
                            v = meta.iloc[idx][col]
                            stamp[f"input_{col}"] = v.item() if hasattr(v, "item") else v

                    for stage, row, dec in (("init", row_init, dec_init), ("opt", row_opt, dec_opt)):
                        row.update(stamp)
                        row["run_key"] = f"{base_key}|{stage}"
                        row["stage"] = stage
                        if stage == "opt":
                            row["opt_best_it"] = best["it"]
                            row["opt_loss_first"] = history[0]["loss"] if history else None
                            row["opt_loss_best"] = best["loss"]
                            row["opt_n_updates"] = len(history)
                            row["opt_seconds"] = round(time.time() - t_restart, 2)
                        if args.pdb_dir or args.complex_pdb_dir:
                            tag = f"{example_id}_{cyc_name}_{args.init_mode}_r{restart}_{stage}"
                            row.update(write_candidate(dec, mask, batch, objective, cyc_type,
                                                       tag, args, frame))
                        fh.write(json.dumps(row, default=str) + "\n")
                        fh.flush()
                        n_rows += 1

                    if args.z_dir:
                        torch.save(
                            {"z": {k: v.cpu() for k, v in z_best.items()},
                             "z_init": {k: v.cpu() for k, v in z_init.items()},
                             "meta": stamp, "history": history},
                            Path(args.z_dir) / f"{base_key.replace('|', '__')}.pt",
                        )

                    if args.max_seconds and (time.time() - t_wall) > args.max_seconds:
                        print(f"  budget of {args.max_seconds:.0f}s reached after {n_rows} rows; "
                              "stopping cleanly -- resubmit to continue", flush=True)
                        budget_hit = True
                        break

    elapsed = time.time() - t_wall
    print(f"done: {n_rows} rows, {n_fail} failed, {elapsed:.0f}s", flush=True)
    print("COST_SUMMARY " + json.dumps({
        "rows": n_rows, "failed": n_fail, "seconds": round(elapsed, 1),
        "restarts": int(args.restarts), "opt_steps": int(args.opt_steps),
        "opt_nsteps": int(args.opt_nsteps), "eval_nsteps": int(eval_nsteps),
        # Network evaluations are the honest unit of compute here: a restart costs
        # opt_steps * opt_nsteps forward-equivalents for the rollout, roughly double that
        # again for the backward under checkpointing, plus two eval rollouts.
        "nn_evals_per_restart_approx": int(args.opt_steps) * int(args.opt_nsteps) * 3
        + 2 * int(eval_nsteps) + int(args.opt_nsteps),
        "budget_hit": budget_hit,
    }), flush=True)

    # Per-restart failures are written as rows, so without this the script exits 0 even when
    # every optimization died -- which would release the afterok chain behind a stage that
    # produced nothing.
    if n_rows == 0 and n_fail > 0:
        raise SystemExit(f"FATAL: all {n_fail} restarts failed, none succeeded.")


if __name__ == "__main__":
    main()
