"""DDIB-style INVERSION of a bound linear peptide under the public Proteina-Complexa flow.

The idea
--------
SDEdit destroys information isotropically: it interpolates the clean sample toward a
Gaussian draw, so at t = 0.6 the peptide is "the input plus 40% of white noise". Nothing
about that perturbation knows what a protein is.

DDIB replaces the noise with the MODEL. Integrate the probability-flow ODE BACKWARDS,

    dx/dt = v_theta(x, t),   t : 1 -> t*

using the source model (here the public `complexa.ckpt`, trained on ordinary bound
peptides). The state x_{t*} is then a model-consistent, deterministically invertible
encoding of THIS peptide -- every intermediate lies on a trajectory the source model
believes in, and re-integrating forward under the source model returns the input. Handing
x_{t*} to a different (cyclic) model forward integrates the same latent through a different
prior, which is the DDIB translation.

What this script measures (part 1 only: does the hope stand?)
-------------------------------------------------------------
The hope is that somewhere along the reverse trajectory the peptide's TERMINI come
together far enough that the cyclic generator can close a ring from there -- and that they
do so before the pose is destroyed. Those two things fight, so both are recorded at every
recorded time, along with the paired SDEdit control at the same t:

    ca_end_gap_A        CA(first) - CA(last) distance, the closure feasibility number
    nc_gap_A            N(first) - C(last), the actual mainchain bond distance (needs decode)
    ca_rmsd_to_input_A  how much pose was spent to get there
    contact_retention   how much of the receptor interface survived
    rg_A                radius of gyration -- separates "termini met" from "everything collapsed"

Each of those appears three times per row, suffixed:
    `_xt`      the state actually handed onward (what the cyclic model would start from)
    `_x1pred`  the source model's clean-sample prediction at that t (what it "means")
    `_sdedit`  the SDEdit control: interpolate(noise, input, t), zero extra NFE

If the reverse ODE is worth anything, `_xt`/`_x1pred` must beat `_sdedit` on gap-at-equal-
damage. If it does not, the whole DDIB arm dies here for the price of one reverse sweep.

Two knobs of substance
----------------------
`--invert-tracks`  The flow is a product space over `bb_ca` (CA coords) and `local_latents`
    (8-dim/residue, the AE's sequence+sidechain code). `t` is sampled independently per
    track, so freezing one at t=1 while inverting the other is in-distribution, not a hack.
    This matters for the eventual translation: the CPSea AE was FINETUNED from the public
    one, so the two models do NOT share a latent space -- but they do share `bb_ca` exactly
    (raw CA coordinates). An inversion restricted to `bb_ca` is therefore transferable to
    the cyclic model as-is, while a `local_latents` inversion is not. Both arms are run so
    the cost of the restriction is measured rather than assumed.

`--roundtrip-ts`  At these times, re-integrate FORWARD to t=1 under the same source model
    and report `roundtrip_ca_rmsd_A`. This is the invertibility check: a DDIB latent that
    does not return the input is not an encoding of the input, and every downstream number
    would be measuring discretization error instead of translation.

Inversion is run as a pure ODE: `sampling_mode` is forced to "vf" and the noise-injection
schedule `gt` to zero. The stock sampler's "sc" mode adds `sqrt(2*gt*dt)` noise, which is
neither invertible nor even real-valued at the negative dt a reverse pass takes.

Outputs one JSONL row per (example, track set, seed, recorded t). Resumable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import torch

from proteinfoundation.datasets.structure_data import structure_collate_fn
from proteinfoundation.eval.cyclic_reconstruction_metrics import NM_TO_ANG
from proteinfoundation.eval.sampled_binder_metrics import _as_residue_mask
from proteinfoundation.flow_matching.product_space_flow_matcher import (
    get_schedule,
    get_schedule_tsr_safe,
)
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting

sys.path.insert(0, str(Path(__file__).parent))
from sdedit_cyclize import (  # noqa: E402  (path must be set first)
    CA_IDX,
    T_START_MAX,
    _resolve_sampling_cfg,
    contact_set,
    to_device,
)

N_IDX, C_IDX = 0, 2  # atom37 backbone slots


def build_reverse_schedule(data_modes, nsteps: int, sampling_model_args: dict, invert: set[str]):
    """Descending time grid for the tracks being inverted; a frozen constant for the rest.

    The grid is the sampler's OWN schedule, flipped. Using the same discretization forward
    and backward is what makes the inversion a numerical inverse rather than a different
    integrator applied to the same field. The top is clamped just inside 1: the network
    never saw t = 1 during training, and `vf_to_score` asserts t < 1 strictly.

    A frozen track gets a constant vector, so its `dt` is exactly 0 at every step and the
    Euler update leaves it at its clean value -- the (t_ca < 1, t_lat = 1) corner the
    product flow was actually trained on, not a special case bolted onto the loop.
    """
    ts, gt = {}, {}
    for dm in data_modes:
        args_dm = sampling_model_args[dm]
        schedule_func = (
            get_schedule_tsr_safe
            if args_dm["simulation_step_params"]["sampling_mode"] == "vf_tsr"
            else get_schedule
        )
        base = schedule_func(
            mode=args_dm["schedule"]["mode"], nsteps=int(nsteps), p1=args_dm["schedule"]["p"]
        )
        base = torch.clamp(base, max=T_START_MAX)
        if dm in invert:
            ts[dm] = torch.flip(base, dims=[0]).contiguous()  # 1-eps -> 0
        else:
            ts[dm] = torch.full_like(base, T_START_MAX)
        # Deterministic probability-flow ODE: no noise injection anywhere.
        gt[dm] = torch.zeros(len(ts[dm]) - 1)
    return ts, gt


def ode_step_params(data_modes, sampling_model_args: dict) -> dict:
    """The sampler's step params with `sampling_mode` forced to the plain ODE.

    The stock config samples in "sc" mode, whose SDE branch takes `sqrt(2*gt*sc_scale_noise*dt)`.
    Under a reverse pass dt < 0, so that is a NaN, and even with dt > 0 the injected noise
    would make the map non-invertible. Deep-copied because the config node is shared with
    whatever else holds this config.
    """
    params = copy.deepcopy({dm: dict(sampling_model_args[dm]["simulation_step_params"]) for dm in data_modes})
    for dm in params:
        params[dm]["sampling_mode"] = "vf"
        params[dm]["center_every_step"] = False  # would re-centre the binder out of its receptor frame
    return params


def _ends(mask_b: torch.Tensor) -> tuple[int, int]:
    idx = torch.nonzero(mask_b, as_tuple=False).flatten()
    return int(idx[0]), int(idx[-1])


def ca_geometry(ca_nm, ca_in_nm, mask_b, tgt_ca_nm, contacts_in, suffix: str) -> dict:
    """CA-only geometry of one state. No superposition: the receptor frame is shared, so a
    Kabsch fit would hide exactly the movement inside the binding site we are pricing."""
    i, j = _ends(mask_b)
    sel = mask_b.bool()
    ca, ca_in = ca_nm[sel], ca_in_nm[sel]
    out = {
        f"ca_end_gap_A{suffix}": float(torch.linalg.norm(ca_nm[i] - ca_nm[j]) * NM_TO_ANG),
        f"ca_rmsd_to_input_A{suffix}": float(torch.sqrt(((ca - ca_in) ** 2).sum(-1).mean()) * NM_TO_ANG),
        f"rg_A{suffix}": float(torch.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()) * NM_TO_ANG),
    }
    if tgt_ca_nm is not None:
        ones_p = torch.ones(ca.shape[0], dtype=torch.bool, device=ca.device)
        ones_t = torch.ones(tgt_ca_nm.shape[0], dtype=torch.bool, device=ca.device)
        now = contact_set(ca, tgt_ca_nm, ones_p, ones_t)
        out[f"n_contacts{suffix}"] = len(now)
        out[f"contact_retention{suffix}"] = (
            float(len(contacts_in & now) / len(contacts_in)) if contacts_in else float("nan")
        )
    return out


def decode_geometry(model, x, mask, seq_in, suffix: str) -> dict:
    """All-atom quantities that only exist after an AE decode: the true N-C bond distance
    (the number a mainchain ring has to close) and the sequence the latent track now codes."""
    prots = sample_formatting(
        x=x, extra_info={"mask": mask}, ret_mode="coors37_n_aatype",
        data_modes=list(model.cfg_exp.product_flowmatcher), autoencoder=model.autoencoder,
    )
    coors_nm = prots["coors"] / NM_TO_ANG
    aatype = prots["residue_type"]
    i, j = _ends(mask[0])
    out = {
        f"nc_gap_A{suffix}": float(
            torch.linalg.norm(coors_nm[0][i, N_IDX] - coors_nm[0][j, C_IDX]) * NM_TO_ANG
        ),
        f"seq_identity{suffix}": float((aatype[0][mask[0]].long() == seq_in).float().mean()),
    }
    return out


def simulate(fm, model, batch, x, mask, ts, gt, step_params, start, end, self_cond, n_recycle,
             guidance_w):
    return fm.partial_simulation(
        batch=batch, x=x, x_1_pred=None, mask=mask,
        predict_for_sampling=partial(model.predict_for_sampling, n_recycle=n_recycle),
        start_step=start, end_step=end, self_cond=self_cond, ts=ts, gt=gt,
        simulation_step_params=step_params, device=mask.device,
        guidance_w=guidance_w, ag_ratio=0.0,
    )


def stop_indices(ts_invert: torch.Tensor, targets: list[float]) -> list[tuple[int, float]]:
    """Map each requested time onto the nearest grid index, de-duplicated and ordered along
    the reverse pass. Recording is snapped to the grid rather than interpolated so the state
    reported is one the integrator actually visited."""
    seen, stops = set(), []
    for target in targets:
        k = int(torch.argmin((ts_invert - float(target)).abs()))
        if k in seen or k == 0:
            continue
        seen.add(k)
        stops.append((k, float(ts_invert[k])))
    return sorted(stops)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True, help="Source flow checkpoint FILE (e.g. .../complexa.ckpt).")
    ap.add_argument("--ae-ckpt", default=None,
                    help="Autoencoder for the source model. Must be the AE this flow was "
                         "trained against (the public flow -> complexa_ae.ckpt); mixing them "
                         "silently reinterprets the latent track.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config-name", default="example/training_cpsea_peptide_smoke")
    ap.add_argument("--sampling-config", default="pipeline/model_sampling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--invert-tracks", nargs="+", default=["bb_ca"],
                    help="Which product-space tracks the reverse ODE moves. Others stay clean "
                         "at t=1. Pass 'bb_ca' and/or 'local_latents'.")
    ap.add_argument("--record-ts", nargs="+", type=float,
                    default=[0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
                    help="Times along the reverse trajectory to score.")
    ap.add_argument("--roundtrip-ts", nargs="+", type=float, default=[],
                    help="Times at which to also re-integrate FORWARD to t=1 and report the "
                         "CA RMSD back to the input (the invertibility check). Costs a second "
                         "pass per listed time, so keep the list short.")
    ap.add_argument("--no-decode", action="store_true",
                    help="Skip the AE decode, so only CA-level geometry is recorded (no nc_gap).")
    ap.add_argument("--self-cond", action="store_true",
                    help="Self-condition during inversion. OFF by default: the x_sc chain is "
                         "built from the forward pass's own history and breaks invertibility.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="Only affects the SDEdit control (the inversion is deterministic).")
    ap.add_argument("--nsteps", type=int, default=0, help="0 = the design sampler's own nsteps.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve configs, dataset and schedule, then exit before the "
                         "multi-GB checkpoint load.")
    args = ap.parse_args()

    from eval_ae_roundtrip import build_dataset

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    dataset, _ = build_dataset(args.config_name, args.metadata, keep_rotation=False)
    meta = dataset.metadata
    all_indices = list(range(len(dataset)))
    if args.limit:
        all_indices = all_indices[: args.limit]
    my_indices = all_indices[args.shard_index :: args.shard_count]
    print(f"{len(dataset)} input peptides from {args.metadata}; "
          f"shard {args.shard_index}/{args.shard_count} takes {len(my_indices)}", flush=True)

    # Config first, checkpoint after: a typo here must cost seconds, not a 3 GB load.
    with hydra.initialize("../configs", version_base=hydra.__version__):
        samp = _resolve_sampling_cfg(hydra.compose(config_name=args.sampling_config))
    sampler_args, sampling_model_args = samp.args, samp.model
    nsteps = args.nsteps or int(sampler_args.nsteps)
    n_recycle = int(samp.get("n_recycle", 0))
    guidance_w = float(sampler_args.get("guidance_w", 1.0))
    print(f"sampler: nsteps={nsteps} self_cond={args.self_cond} n_recycle={n_recycle} "
          f"(inversion forces sampling_mode=vf, gt=0)", flush=True)

    if args.dry_run:
        # Build the reverse grid here too, from the config's own data-mode keys. The real
        # path needs the loaded model for `fm.data_modes`, and a schedule/stop bug that only
        # surfaces after a multi-GB load is exactly what this flag exists to prevent.
        dms = list(sampling_model_args.keys())
        unknown = [t for t in args.invert_tracks if t not in dms]
        if unknown:
            raise SystemExit(f"FATAL: unknown track(s) {unknown}; sampling config has {dms}")
        ts_d, gt_d = build_reverse_schedule(dms, nsteps, sampling_model_args, set(args.invert_tracks))
        ode_step_params(dms, sampling_model_args)
        ref_d = sorted(set(args.invert_tracks))[0]
        stops_d = stop_indices(ts_d[ref_d], args.record_ts)
        rt_d = stop_indices(ts_d[ref_d], args.roundtrip_ts)
        print(f"reverse grid on {ref_d}: t[0]={ts_d[ref_d][0]:.4f} -> t[-1]={ts_d[ref_d][-1]:.4f}, "
              f"{len(stops_d)} stops at {[round(t, 3) for _, t in stops_d]}", flush=True)
        print(f"roundtrip stops: {[round(t, 3) for _, t in rt_d]}", flush=True)
        print(f"DRY RUN OK: {len(my_indices)} peptides x tracks={args.invert_tracks} x "
              f"{len(args.seeds)} seeds x {len(stops_d)} recorded times", flush=True)
        return

    if not os.path.isfile(args.flow_ckpt):
        raise SystemExit(f"FATAL: flow checkpoint not found: {args.flow_ckpt}")
    ae_ckpt = args.ae_ckpt or os.environ.get("CPSEA_AE_CKPT_PATH")
    print(f"flow: {args.flow_ckpt}\nAE:   {ae_ckpt}", flush=True)
    model = Proteina.load_from_checkpoint(args.flow_ckpt, strict=False, autoencoder_ckpt_path=ae_ckpt)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    if model.autoencoder is None:
        raise SystemExit("FATAL: checkpoint has no autoencoder; the local_latents track is undefined.")

    fm = model.fm
    unknown = [t for t in args.invert_tracks if t not in fm.data_modes]
    if unknown:
        raise SystemExit(f"FATAL: unknown track(s) {unknown}; model data modes are {list(fm.data_modes)}")
    invert = set(args.invert_tracks)
    track_tag = "+".join(sorted(invert))

    ts, gt = build_reverse_schedule(fm.data_modes, nsteps, sampling_model_args, invert)
    ts = {k: v.to(args.device) for k, v in ts.items()}
    gt = {k: v.to(args.device) for k, v in gt.items()}
    step_params = ode_step_params(fm.data_modes, sampling_model_args)
    ref = sorted(invert)[0]
    stops = stop_indices(ts[ref].cpu(), args.record_ts)
    rt_stops = {k for k, _ in stop_indices(ts[ref].cpu(), args.roundtrip_ts)}
    print(f"inverting {track_tag}; {len(stops)} recorded stops at t="
          f"{[round(t, 3) for _, t in stops]}", flush=True)

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
        print(f"resuming: {len(done)} rows already written", flush=True)

    n_rows = n_fail = 0
    t_wall = time.time()
    with out_path.open("a") as fh:
        for idx in my_indices:
            example_id = str(meta.iloc[idx]["example_id"])
            for seed in args.seeds:
                run_key = f"{example_id}|{track_tag}|{seed}"
                if f"{run_key}|{round(stops[-1][1], 4)}" in done:
                    continue
                sample = dataset[idx]
                if sample is None:
                    print(f"  SKIP {example_id}: failed to load", flush=True)
                    break
                batch = to_device(structure_collate_fn([sample]), args.device)
                mask = (batch["mask"].bool() if "mask" in batch
                        else batch["coord_mask"][..., CA_IDX].bool())
                batch["mask"] = mask
                try:
                    rows = invert_one(model, fm, batch, mask, ts, gt, step_params, stops,
                                      rt_stops, seed, args, n_recycle, guidance_w, invert)
                except Exception as exc:  # noqa: BLE001
                    n_fail += 1
                    fh.write(json.dumps({"run_key": f"{run_key}|failed", "example_id": example_id,
                                         "status": "failed", "invert_tracks": track_tag,
                                         "seed": seed,
                                         "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                    fh.flush()
                    print(f"  FAIL {run_key}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                for row in rows:
                    row.update(run_key=f"{run_key}|{row['t']}", example_id=example_id,
                               status="ok", invert_tracks=track_tag, seed=seed, nsteps=nsteps)
                    for col in ("peptide_length", "nc_gap_angstrom", "best_ss_cb_dist",
                                "best_iso_cb_dist", "n_cys"):
                        if col in meta.columns:
                            v = meta.iloc[idx][col]
                            row[f"input_{col}"] = v.item() if hasattr(v, "item") else v
                    fh.write(json.dumps(row) + "\n")
                    n_rows += 1
                fh.flush()
                print(f"  {example_id}: {len(rows)} stops ({time.time() - t_wall:.0f}s)", flush=True)

    print(f"done: {n_rows} rows, {n_fail} failed peptides, {time.time() - t_wall:.0f}s", flush=True)
    if n_rows == 0 and n_fail > 0:
        raise SystemExit(f"FATAL: all {n_fail} peptides failed -- see the errors above.")


@torch.no_grad()
def invert_one(model, fm, batch, mask, ts, gt, step_params, stops, rt_stops, seed, args,
               n_recycle, guidance_w, invert):
    """One reverse pass, scored at every requested stop. Returns a list of rows."""
    L.seed_everything(seed, workers=True)

    batch = {k: v for k, v in batch.items() if k not in ("x_1", "x_0", "x_t", "t", "x_sc")}
    batch["mask"] = mask
    # The source model is the public linear-peptide flow: it has no cyclization conditioning,
    # and the LNR transform stack stamps these keys regardless. Drop them so nothing is
    # quietly interpreted by a head that does not exist here.
    for k in [k for k in batch if k.startswith("cyclization")]:
        batch.pop(k)

    batch = add_clean_samples(batch, model.cfg_exp.product_flowmatcher,
                              autoencoder=model.autoencoder, local_latent_target="mean")
    x_1 = {dm: v.clone() for dm, v in batch["x_1"].items()}

    # Reference quantities, computed once: the input pose, its interface, its sequence.
    ca_in = batch["coords_nm"][0][:, CA_IDX, :]
    seq_in = batch["residue_type"][0][mask[0]].long()
    tgt_ca = None
    x_target, target_mask = batch.get("x_target"), batch.get("target_mask")
    if x_target is not None and target_mask is not None:
        # `target_mask` is ATOM-level [B, T, 37] in compact mode despite the name.
        tmask = _as_residue_mask(target_mask.bool())[0]
        tgt = x_target[0][:, CA_IDX, :] if x_target[0].dim() == 3 else x_target[0]
        if tmask.shape[0] != tgt.shape[0]:
            raise RuntimeError(f"target mask/coords mismatch: {tmask.shape[0]} vs {tgt.shape[0]}")
        tgt_ca = tgt[tmask]
    ones_t = None if tgt_ca is None else torch.ones(tgt_ca.shape[0], dtype=torch.bool, device=mask.device)
    contacts_in = set()
    if tgt_ca is not None:
        ones_p = torch.ones(int(mask[0].sum()), dtype=torch.bool, device=mask.device)
        contacts_in = contact_set(ca_in[mask[0]], tgt_ca, ones_p, ones_t)

    i0, j0 = _ends(mask[0])
    input_row = {
        "input_ca_end_gap_A": float(torch.linalg.norm(ca_in[i0] - ca_in[j0]) * NM_TO_ANG),
        "input_nc_gap_A": float(
            torch.linalg.norm(batch["coords_nm"][0][i0, N_IDX] - batch["coords_nm"][0][j0, C_IDX])
            * NM_TO_ANG),
        "input_n_contacts": len(contacts_in),
        "peptide_length": int(mask[0].sum()),
    }

    # The SDEdit control shares this noise draw across all stops, exactly as an SDEdit sweep
    # would: the arms differ in the map applied to the input, not in the noise they saw.
    x_0 = fm.sample_noise(n=mask.shape[1], shape=(mask.shape[0],), mask=mask, device=mask.device)

    x = {dm: v.clone() for dm, v in x_1.items()}
    rows, step = [], 0
    for stop_idx, t_stop in stops:
        x, x_1_pred = simulate(fm, model, batch, x, mask, ts, gt, step_params, step, stop_idx,
                               args.self_cond, n_recycle, guidance_w)
        step = stop_idx

        row = {"t": round(t_stop, 4), "step": stop_idx, **input_row}
        row.update(ca_geometry(x["bb_ca"][0], ca_in, mask[0], tgt_ca, contacts_in, "_xt"))
        row.update(ca_geometry(x_1_pred["bb_ca"][0], ca_in, mask[0], tgt_ca, contacts_in, "_x1pred"))

        # Paired SDEdit control at the SAME t, on the SAME tracks: interpolate toward the
        # Gaussian instead of integrating the model. Zero network evaluations.
        t_vec = {dm: torch.full((mask.shape[0],), float(t_stop if dm in invert else T_START_MAX),
                                device=mask.device) for dm in fm.data_modes}
        x_sd = fm.interpolate(x_0=x_0, x_1=x_1, t=t_vec, mask=mask)
        row.update(ca_geometry(x_sd["bb_ca"][0], ca_in, mask[0], tgt_ca, contacts_in, "_sdedit"))

        if not args.no_decode:
            # Decoding x_t itself is meaningless (a partially noised latent is not a code for
            # any structure), so the all-atom numbers are read off the model's CLEAN prediction.
            row.update(decode_geometry(model, x_1_pred, mask, seq_in, "_x1pred"))

        if stop_idx in rt_stops:
            # Forward re-integration on the same grid: ts flipped back over [0, stop_idx].
            ts_fwd = {dm: torch.flip(ts[dm][: stop_idx + 1], dims=[0]).contiguous() for dm in fm.data_modes}
            gt_fwd = {dm: torch.zeros(stop_idx, device=mask.device) for dm in fm.data_modes}
            x_back, _ = simulate(fm, model, batch, {dm: v.clone() for dm, v in x.items()}, mask,
                                 ts_fwd, gt_fwd, step_params, 0, stop_idx, args.self_cond,
                                 n_recycle, guidance_w)
            back = x_back["bb_ca"][0][mask[0]]
            row["roundtrip_ca_rmsd_A"] = float(
                torch.sqrt(((back - ca_in[mask[0]]) ** 2).sum(-1).mean()) * NM_TO_ANG)
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
