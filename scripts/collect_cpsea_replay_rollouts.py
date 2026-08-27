#!/usr/bin/env python3
"""Offline reward-weighted-replay rollout collection for CPSea (Stage 4).

Loads a frozen checkpoint, samples K candidates for each TRAINING receptor (never
val/test -- uses `dm.train_dataloader()`, backed by `cpsea_train.parquet`, see
`configs/dataset/unified/cpsea_peptide.yaml`), decodes each candidate, scores it
with `cyclization.scoring.score_cyclization`, and appends the scored terminal
endpoints (`x1_ca`, `z1` -- pre-decode, exactly what Stage 5's
`flow_loss_from_clean_target` trains on) to a `ReplayBuffer` saved to disk.

Each training receptor is conditioned on and scored against its OWN native
cyclization label (`cyclization_i/j/type`, read straight off the dataloader
batch) -- the model is asked to reproduce the linkage that receptor's binder
actually has, not an arbitrary one.

Mirrors `scripts/eval_cyc_type_conditioning.py`'s checkpoint/datamodule loading
pattern (`Proteina.load_from_checkpoint` + `configure_inference` + `instantiate`
the Hydra datamodule), extended to call `Proteina.generate()` directly (single-
pass, bypassing `predict_step`/search -- see the implementation plan's Decision 2)
and to persist the pre-decode endpoints rather than only a decoded structure.

Usage:
    python scripts/collect_cpsea_replay_rollouts.py \\
        --ckpt /path/to/last-EMA.ckpt \\
        --config example/training_cpsea_peptide_cyc_typecond \\
        --out $ZFS/CPSea/replay_buffers/<run> \\
        --k 10 --n-batches 50 --nsteps 200 --gate-chirality
"""

from __future__ import annotations

import argparse
import os

import torch
from omegaconf import OmegaConf

# Must land before any structure_data / atomworks encoding path: patches
# atom_array_to_encoding with atomize_token (used by CPSea parquet loaders).
import proteinfoundation.patches.atomworks_patches  # noqa: F401


def _to_str_list(value, batch_size: int) -> list[str]:
    """Normalizes a batch column that may be a list[str] or a tensor into list[str]."""
    if isinstance(value, torch.Tensor):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * batch_size


def _repeat_batch(batch: dict, k: int) -> dict:
    """Tiles every per-example entry in `batch` k times along the batch dim.

    Produces K independent candidates per original receptor in one forward pass
    (`repeat_interleave`, so all K copies of receptor 0 come before receptor 1's,
    matching how `group_ids` are built below for stratified/grouped weighting).
    """
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.repeat_interleave(k, dim=0)
        elif isinstance(value, (list, tuple)):
            out[key] = [v for v in value for _ in range(k)]
        else:
            out[key] = value
    return out


def _slice_batch(batch: dict, start: int, end: int) -> dict:
    """Slice a collated batch along the batch dimension [start:end]."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[start:end]
        elif isinstance(value, (list, tuple)):
            out[key] = list(value[start:end])
        else:
            out[key] = value
    return out


def _extract_receptor_condition(batch_slice: dict) -> dict:
    """Pulls every receptor-conditioning tensor (any key containing "target") out of a 1-example batch slice.

    Matches the "target" substring convention `handle_target_dropout` and
    `FilterTargetResiduesTransform` already use elsewhere for this exact
    purpose, rather than hardcoding a field list that would silently go stale
    if the dataset pipeline adds/renames a target feature.
    """
    cond = {}
    for key, value in batch_slice.items():
        if "target" not in key.lower() or not torch.is_tensor(value):
            continue
        t = value[0].detach().cpu()
        if t.is_floating_point():
            t = t.to(torch.float16)
        cond[key] = t
    return cond


def _cat_dicts(parts: list[dict]) -> dict:
    """Concatenate a list of batch-dicts along dim 0 (tensors) / list-extend (lists)."""
    if not parts:
        return {}
    out = {}
    for key in parts[0]:
        vals = [p[key] for p in parts]
        if torch.is_tensor(vals[0]):
            out[key] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], (list, tuple)):
            out[key] = [x for v in vals for x in v]
        else:
            out[key] = vals[0]
    return out


def collect(
    ckpt_path: str,
    config_name: str,
    out_dir: str,
    k: int,
    n_batches: int,
    nsteps: int,
    gate_chirality: bool,
    gate_angle: bool,
    gate_dihedral: bool,
    gate_clash: bool,
    max_size: int,
    seed: int,
    gen_chunk: int = 2,
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from proteinfoundation.cyclization.scoring import REWARD_VERSION, score_cyclization
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.replay.buffer import ReplayBuffer
    from proteinfoundation.utils.sample_utils import sample_formatting

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        data_cfg = compose(config_name=config_name, overrides=["+single=true"])

    # Use the design-pipeline sampler (val_generation.design_sampling), NOT empty
    # bb_ca/local_latents dicts and NOT generation.model.ode. The monomer ode block
    # sets bb_ca.center_every_step=True, which is wrong for CPSea binders (target-
    # centered); empty dicts crash on missing simulation_step_params.
    design = data_cfg.val_generation.design_sampling
    if design is None:
        raise RuntimeError(
            f"{config_name} has no val_generation.design_sampling; add "
            "`- /pipeline/model_sampling@val_generation.design_sampling` to its defaults."
        )
    inf_cfg = OmegaConf.create(OmegaConf.to_container(design, resolve=True))
    inf_cfg.args.nsteps = nsteps

    model = Proteina.load_from_checkpoint(ckpt_path, map_location=device).eval().to(device)
    model.configure_inference(inf_cfg, nn_ag=None)

    dm = instantiate(data_cfg.dataset.unified.datamodule)
    dm.setup("fit")

    checkpoint_tag = os.path.basename(ckpt_path)

    try:
        buffer = ReplayBuffer.load(out_dir, expected_reward_version=REWARD_VERSION)
        print(f"Resuming existing buffer at {out_dir} ({len(buffer)} entries).")
    except FileNotFoundError:
        buffer = ReplayBuffer(max_size=max_size)
        buffer.reward_version = REWARD_VERSION

    if gen_chunk < 1:
        raise ValueError(f"gen_chunk must be >= 1, got {gen_chunk}")
    print(f"Generation microbatch size gen_chunk={gen_chunk} (K={k} candidates/receptor).")

    n_written = 0
    type_counts: dict[int, int] = {}
    zero_variance_groups = 0
    total_groups = 0

    for batch_idx, batch in zip(range(n_batches), dm.train_dataloader()):
        batch = {k_: (v.to(device) if torch.is_tensor(v) else v) for k_, v in batch.items()}
        required = ("mask", "cyclization_i", "cyclization_j", "cyclization_type", "has_cyclization")
        if any(r not in batch for r in required):
            print(f"batch {batch_idx}: missing cyclization labels, skipping.")
            continue

        bs = batch["mask"].shape[0]
        example_ids = _to_str_list(batch.get("example_id", [f"batch{batch_idx}_ex{i}" for i in range(bs)]), bs)
        cluster_ids = _to_str_list(batch.get("cluster_id", example_ids), bs)

        # Per-receptor + chunked K: a single `generate(batch * K)` OOMs on 24GB GPUs
        # because binder+target pair features are O(B * N^2) and train batches already
        # pad to N~250. Match train's effective B (~2) via gen_chunk.
        entries = []
        reward_means = []
        for receptor_idx in range(bs):
            single = _slice_batch(batch, receptor_idx, receptor_idx + 1)
            # Stored once per receptor, not once per candidate: the K generated
            # candidates below all share the same receptor, so `_repeat_batch`
            # would otherwise duplicate these (larger) receptor tensors K times.
            buffer.add_receptor_conditions({example_ids[receptor_idx]: _extract_receptor_condition(single)})
            repeated = _repeat_batch(single, k)

            gen_parts = []
            with torch.no_grad():
                for start in range(0, k, gen_chunk):
                    end = min(start + gen_chunk, k)
                    gen_parts.append(model.generate(_slice_batch(repeated, start, end)))
                gen_samples = _cat_dicts(gen_parts)
                decoded = sample_formatting(
                    x=gen_samples,
                    extra_info={"mask": repeated["mask"]},
                    ret_mode="atom37_nm_with_atom_mask",
                    data_modes=["bb_ca", "local_latents"],
                    autoencoder=model.autoencoder,
                )
                linkage_metadata = {
                    "i": repeated["cyclization_i"],
                    "j": repeated["cyclization_j"],
                    "type": repeated["cyclization_type"],
                    "has_cyclization": repeated["has_cyclization"],
                }
                scores = score_cyclization(
                    atom37=decoded["coors_nm"],
                    atom37_mask=decoded["atom_mask"],
                    aatype=decoded["residue_type"],
                    binder_mask=repeated["mask"],
                    linkage_metadata=linkage_metadata,
                    gate_chirality=gate_chirality,
                    gate_angle=gate_angle,
                    gate_dihedral=gate_dihedral,
                    gate_clash=gate_clash,
                )

            for row in range(k):
                length = int(repeated["mask"][row].sum().item())
                entry = {
                    "target_or_dataset_id": example_ids[receptor_idx],
                    "cluster_id": cluster_ids[receptor_idx],
                    "linkage_type": int(repeated["cyclization_type"][row].item()),
                    "linkage_sites": (
                        int(repeated["cyclization_i"][row].item()),
                        int(repeated["cyclization_j"][row].item()),
                    ),
                    "peptide_length": length,
                    "x1_ca": gen_samples["bb_ca"][row, :length].detach().cpu().to(torch.float16),
                    "z1": gen_samples["local_latents"][row, :length].detach().cpu().to(torch.float16),
                    "binder_mask": repeated["mask"][row, :length].detach().cpu().bool(),
                    "raw_reward": float(scores["reward"][row].item()),
                    "reward_components": {
                        "success": bool(scores["success"][row].item()),
                        "near_success": bool(scores["near_success"][row].item()),
                        "distance_error": float(scores["distance_error"][row].item()),
                        "chirality_valid": bool(scores["chirality_valid"][row].item()),
                        "clash_count": float(scores["clash_count"][row].item()),
                    },
                    "collector_checkpoint": checkpoint_tag,
                    "reward_version": REWARD_VERSION,
                    "random_seed": seed,
                }
                entries.append(entry)
                type_counts[entry["linkage_type"]] = type_counts.get(entry["linkage_type"], 0) + 1

            group_rewards = scores["reward"]
            reward_means.append(float(group_rewards.mean().item()))
            total_groups += 1
            if float(group_rewards.std().item()) < 1e-6:
                zero_variance_groups += 1

        buffer.append(entries)
        n_written += len(entries)

        if batch_idx % 5 == 0:
            buffer.save(out_dir)
            mean_r = sum(reward_means) / max(len(reward_means), 1)
            print(
                f"batch {batch_idx}: buffer={len(buffer)} entries, "
                f"reward_mean={mean_r:.3f}, "
                f"by_type={type_counts}"
            )

    buffer.save(out_dir)
    zero_var_frac = zero_variance_groups / max(total_groups, 1)
    print(f"\nDone. Wrote {n_written} entries to buffer at {out_dir} ({len(buffer)} total).")
    print(f"By linkage type: {type_counts}")
    print(
        f"Zero-reward-variance groups: {zero_variance_groups}/{total_groups} "
        f"({zero_var_frac:.1%}) -- these provide no relative preference signal "
        "for group-relative (GeoCycler-style) weighting."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Frozen checkpoint to collect from (e.g. last-EMA.ckpt).")
    ap.add_argument(
        "--config",
        default="example/training_cpsea_peptide_cyc_typecond",
        help="Hydra config name whose dataset.unified.datamodule to instantiate (train split).",
    )
    ap.add_argument("--out", required=True, help="Replay buffer output directory (ReplayBuffer.save target).")
    ap.add_argument("--k", type=int, default=10, help="Candidates sampled per training receptor.")
    ap.add_argument("--n-batches", type=int, default=50, help="Number of training-receptor batches to collect from.")
    ap.add_argument("--nsteps", type=int, default=200, help="Flow ODE integration steps.")
    ap.add_argument("--gate-chirality", action="store_true", help="Require chirality validity for `success`.")
    ap.add_argument("--gate-angle", action="store_true", help="Require the angle term to pass for `success`.")
    ap.add_argument("--gate-dihedral", action="store_true", help="Require the dihedral term to pass for `success`.")
    ap.add_argument("--gate-clash", action="store_true", help="Require zero clashes for `success`.")
    ap.add_argument("--max-size", type=int, default=10_000, help="ReplayBuffer max_size (new buffers only).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--gen-chunk",
        type=int,
        default=2,
        help="Max candidates per generate() call. Default 2 matches train batch_size on 24GB GPUs; "
        "raise on larger GPUs. Full batch*K without chunking OOMs (pair features ~O(B N^2)).",
    )
    args = ap.parse_args()

    collect(
        ckpt_path=args.ckpt,
        config_name=args.config,
        out_dir=args.out,
        k=args.k,
        n_batches=args.n_batches,
        nsteps=args.nsteps,
        gate_chirality=args.gate_chirality,
        gate_angle=args.gate_angle,
        gate_dihedral=args.gate_dihedral,
        gate_clash=args.gate_clash,
        max_size=args.max_size,
        seed=args.seed,
        gen_chunk=args.gen_chunk,
    )


if __name__ == "__main__":
    main()
