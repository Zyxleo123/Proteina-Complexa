"""Re-run selected SDEdit edits and dump the INTEGRATION TRAJECTORY, frame by frame.

Why this exists
---------------
The sweep (`scripts/sdedit_cyclize.py`) wrote one metrics row per edit and threw the
structures away, so "what does linear -> cyclic actually look like" is unanswerable from
its output. This re-runs a handful of named cases and saves every intermediate state.

Reproducing the sweep exactly
-----------------------------
`partial_simulation`'s Euler loop is `for step in range(start_step, end_step)` with all
state carried in `x`, `x_1_pred` and `batch["x_sc"]` -- no setup or teardown. So calling it
one step at a time, threading those three through, is bit-identical to one call, provided
nothing else consumes RNG in between (`simulation_step` injects noise from the global
stream). That is why frames are stashed as raw CPU tensors inside the loop and decoded
only AFTER it finishes: an autoencoder forward pass per frame, interleaved with the
integration, would shift the noise stream and silently produce a different trajectory than
the one the sweep scored.

Two states are saved per step, because they answer different questions:
    x_t        the actual (noisy) state being integrated
    x_1_pred   the model's running clean-sample prediction -- its current belief about the
               finished peptide, and the legible one to animate

Distances are recorded with the same definitions `scripts/build_lnr_metadata.py` uses for
the input peptides (N of the first residue to C of the last; CB to CB), so a frame's number
is directly comparable to that peptide's `nc_gap_angstrom`.
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
import numpy as np
import torch
from openfold.np import residue_constants as rc

sys.path.insert(0, str(Path(__file__).parent))

from sdedit_cyclize import (  # noqa: E402
    CA_IDX,
    TYPE_NAMES,
    _resolve_sampling_cfg,
    build_edit_schedule,
    score_edit,
    to_device,
)

from proteinfoundation.datasets.structure_data import structure_collate_fn  # noqa: E402
from proteinfoundation.eval.cyclic_reconstruction_metrics import NM_TO_ANG  # noqa: E402
from proteinfoundation.eval.sampled_binder_metrics import (  # noqa: E402
    _as_residue_mask,
    atom37_mask_from_aatype,
)
from proteinfoundation.proteina import Proteina  # noqa: E402
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting  # noqa: E402

N_IDX, C_IDX, CB_IDX = rc.atom_order["N"], rc.atom_order["C"], rc.atom_order["CB"]


def parse_case(spec: str) -> dict:
    """"example_id:chem:t_ca:t_lat:seed" -> dict. Fails loudly on a malformed spec rather
    than silently dropping a case the caller asked to render."""
    parts = spec.split(":")
    if len(parts) != 5:
        raise SystemExit(f"FATAL: bad --cases spec {spec!r}; want example_id:chem:t_ca:t_lat:seed")
    example_id, chem, t_ca, t_lat, seed = parts
    if chem not in TYPE_NAMES:
        raise SystemExit(f"FATAL: unknown chemistry {chem!r} in {spec!r}; expected {sorted(TYPE_NAMES)}")
    return {
        "example_id": example_id,
        "chem": chem,
        "t_ca": float(t_ca),
        "t_lat": float(t_lat),
        "seed": int(seed),
    }


def decode(model, x, mask, data_modes):
    """One product-space state -> (coords_A [L,37,3], aatype [L]) for the masked residues."""
    prots = sample_formatting(
        x=x, extra_info={"mask": mask}, ret_mode="coors37_n_aatype",
        data_modes=data_modes, autoencoder=model.autoencoder,
    )
    m = mask[0].bool()
    return prots["coors"][0][m].float().cpu().numpy(), prots["residue_type"][0][m].long().cpu().numpy()


def closing_distances(coords_A, aatype):
    """Terminal N->C and CB->CB, matching `build_lnr_metadata.py`'s definitions.

    These are the distances the model was actually conditioned to close: with no
    `cyclization_i`/`cyclization_j` in the batch, the ring positional encoding falls back to
    the binder termini (see nn/feature_factory/pair_feats.py). NaN when the atom is absent
    from the sampled residue, which is itself the finding for the disulfide arm.
    """
    valid = atom37_mask_from_aatype(torch.from_numpy(aatype)).numpy()
    out = {}
    for name, (ai, ri, aj, rj) in {
        "nc": (N_IDX, 0, C_IDX, -1),
        "cbcb": (CB_IDX, 0, CB_IDX, -1),
    }.items():
        if valid[ri, ai] and valid[rj, aj]:
            out[name] = float(np.linalg.norm(coords_A[ri, ai] - coords_A[rj, aj]))
        else:
            out[name] = float("nan")
    return out


def run_case(model, dataset, idx, case, nsteps, self_cond, sampler_args,
             sampling_model_args, n_recycle, out_dir):
    """One edit, integrated step by step, every intermediate state saved."""
    sample = dataset[idx]
    if sample is None:
        raise RuntimeError(f"dataset failed to load {case['example_id']}")
    device = next(model.parameters()).device
    base = to_device(structure_collate_fn([sample]), device)
    mask = (base["mask"].bool() if "mask" in base else base["coord_mask"][..., CA_IDX].bool())
    base["mask"] = mask

    # --- setup: identical to sdedit_cyclize.run_one, in the same order (RNG-sensitive) ---
    L.seed_everything(case["seed"], workers=True)
    fm = model.fm
    batch = {k: v for k, v in base.items() if k not in ("x_1", "x_0", "x_t", "t", "x_sc")}
    batch["mask"] = mask
    bs = mask.shape[0]
    batch["cyclization_type_cond"] = torch.full(
        (bs,), int(TYPE_NAMES[case["chem"]]), dtype=torch.long, device=device
    )

    frames_xt, frames_x1 = [], []
    with torch.no_grad():
        batch = add_clean_samples(batch, model.cfg_exp.product_flowmatcher,
                                  autoencoder=model.autoencoder, local_latent_target="mean")
        x_1 = fm._apply_mask(batch["x_1"], mask) if hasattr(fm, "_apply_mask") else batch["x_1"]

        t_start = {"bb_ca": case["t_ca"], "local_latents": case["t_lat"]}
        x_0 = fm.sample_noise(n=mask.shape[1], shape=(bs,), mask=mask, device=device)
        t_vec = {dm: torch.full((bs,), float(t_start[dm]), device=device) for dm in fm.data_modes}
        x = fm.interpolate(x_0=x_0, x_1=x_1, t=t_vec, mask=mask)

        ts, gt = build_edit_schedule(fm, nsteps, sampling_model_args, t_start)
        ts = {k: v.to(device) for k, v in ts.items()}
        gt = {k: v.to(device) for k, v in gt.items()}
        sim_params = {dm: sampling_model_args[dm]["simulation_step_params"] for dm in fm.data_modes}

        # --- integrate one step at a time. Stash RAW tensors only: decoding here would
        # advance the RNG stream and desync this trajectory from the sweep's sample.
        x_1_pred = None
        frames_xt.append({k: v.detach().cpu() for k, v in x.items()})
        for step in range(nsteps):
            x, x_1_pred = fm.partial_simulation(
                batch=batch, x=x, x_1_pred=x_1_pred, mask=mask,
                predict_for_sampling=partial(model.predict_for_sampling, n_recycle=n_recycle),
                start_step=step, end_step=step + 1, self_cond=self_cond, ts=ts, gt=gt,
                simulation_step_params=sim_params, device=device,
                guidance_w=float(sampler_args.get("guidance_w", 1.0)), ag_ratio=0.0,
            )
            frames_xt.append({k: v.detach().cpu() for k, v in x.items()})
            frames_x1.append({k: v.detach().cpu() for k, v in x_1_pred.items()})

        # --- decode everything now that the integration is finished
        dms = list(model.cfg_exp.product_flowmatcher)
        dec_xt = [decode(model, to_device(f, device), mask, dms) for f in frames_xt]
        dec_x1 = [decode(model, to_device(f, device), mask, dms) for f in frames_x1]

        # final metrics, through the sweep's own scorer, so the case is verifiable
        prots = sample_formatting(x=x, extra_info={"mask": mask}, ret_mode="coors37_n_aatype",
                                  data_modes=dms, autoencoder=model.autoencoder)
        row = score_edit(prots["coors"] / NM_TO_ANG, prots["residue_type"], batch, mask,
                         model, x, prots, TYPE_NAMES[case["chem"]])

    m = mask[0].bool()
    inp_coords = (base["coords_nm"][0][m] * NM_TO_ANG).float().cpu().numpy()
    inp_aatype = base["residue_type"][0][m].long().cpu().numpy()

    receptor = None
    if base.get("x_target") is not None and base.get("target_mask") is not None:
        tmask = _as_residue_mask(base["target_mask"].bool())[0]
        xt = base["x_target"][0]
        tgt = (xt[:, CA_IDX, :] if xt.dim() == 3 else xt)[tmask]
        receptor = (tgt * NM_TO_ANG).float().cpu().numpy()

    tag = f"{case['example_id']}_{case['chem']}_tca{case['t_ca']}_tlat{case['t_lat']}_s{case['seed']}"
    payload = {
        "input_coords": inp_coords, "input_aatype": inp_aatype,
        "xt_coords": np.stack([c for c, _ in dec_xt]), "xt_aatype": np.stack([a for _, a in dec_xt]),
        "x1_coords": np.stack([c for c, _ in dec_x1]), "x1_aatype": np.stack([a for _, a in dec_x1]),
        "ts_bb_ca": ts["bb_ca"].cpu().numpy(), "ts_local_latents": ts["local_latents"].cpu().numpy(),
    }
    if receptor is not None:
        payload["receptor_ca"] = receptor

    inp_d = closing_distances(inp_coords, inp_aatype)
    meta = {
        **case, "tag": tag, "nsteps": nsteps, "n_residues": int(m.sum()),
        "input_nc_A": inp_d["nc"], "input_cbcb_A": inp_d["cbcb"],
        "x1_nc_A": [closing_distances(c, a)["nc"] for c, a in dec_x1],
        "x1_cbcb_A": [closing_distances(c, a)["cbcb"] for c, a in dec_x1],
        "final_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()},
    }
    payload["meta_json"] = json.dumps(meta)

    out = Path(out_dir) / f"{tag}.npz"
    np.savez_compressed(out, **payload)
    return meta, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--ckpt-name", default="last-EMA.ckpt")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config-name", default="example/training_cpsea_peptide_smoke")
    ap.add_argument("--sampling-config", default="pipeline/model_sampling")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", nargs="+", required=True,
                    help="example_id:chem:t_ca:t_lat:seed (repeatable)")
    ap.add_argument("--nsteps", type=int, default=0, help="0 = the design sampler's own nsteps.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from eval_ae_roundtrip import build_dataset

    cases = [parse_case(c) for c in args.cases]
    dataset, _ = build_dataset(args.config_name, args.metadata, keep_rotation=False)
    meta_df = dataset.metadata
    by_id = {str(r["example_id"]): i for i, (_, r) in enumerate(meta_df.iterrows())}
    missing = [c["example_id"] for c in cases if c["example_id"] not in by_id]
    if missing:
        raise SystemExit(f"FATAL: example_id(s) not in {args.metadata}: {missing}")

    with hydra.initialize("../configs", version_base=hydra.__version__):
        samp = _resolve_sampling_cfg(hydra.compose(config_name=args.sampling_config))
    sampler_args, sampling_model_args = samp.args, samp.model
    nsteps = args.nsteps or int(sampler_args.nsteps)
    self_cond = bool(sampler_args.self_cond)
    n_recycle = int(samp.get("n_recycle", 0))
    print(f"sampler: nsteps={nsteps} self_cond={self_cond} n_recycle={n_recycle}", flush=True)
    print(f"{len(cases)} case(s); {nsteps + 1} frames each", flush=True)

    if args.dry_run:
        for c in cases:
            print(f"  DRY {c['example_id']} {c['chem']} t_ca={c['t_ca']} t_lat={c['t_lat']} seed={c['seed']}")
        print("DRY RUN OK", flush=True)
        return

    ckpt = Path(args.ckpt_path) / args.ckpt_name
    if not ckpt.is_file():
        raise SystemExit(f"FATAL: checkpoint not found: {ckpt}")
    model = Proteina.load_from_checkpoint(
        str(ckpt), strict=False, autoencoder_ckpt_path=os.environ.get("CPSEA_AE_CKPT_PATH")
    )
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    if model.autoencoder is None:
        raise SystemExit("FATAL: checkpoint has no autoencoder; cannot decode frames.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in cases:
        tag = f"{c['example_id']}_{c['chem']}_tca{c['t_ca']}_tlat{c['t_lat']}_s{c['seed']}"
        if (out_dir / f"{tag}.npz").exists():
            print(f"  skip (done): {tag}", flush=True)
            continue
        meta, path = run_case(model, dataset, by_id[c["example_id"]], c, nsteps, self_cond,
                              sampler_args, sampling_model_args, n_recycle, out_dir)
        fm_ = meta["final_metrics"]
        print(f"  {meta['tag']}\n"
              f"      input N-C {meta['input_nc_A']:.2f} A -> final {meta['x1_nc_A'][-1]:.2f} A"
              f" | retention {fm_.get('contact_retention', float('nan')):.2f}"
              f" | rmsd {fm_.get('ca_rmsd_to_input_A', float('nan')):.2f} A"
              f" | wrote {path.name}", flush=True)
    print(f"done: {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
