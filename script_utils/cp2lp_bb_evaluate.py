"""GPU stage of the checkpoint evaluation: generate the panel's backbones and DUMP them.

This job does the GPU half and stops. It writes, per attempt, the generated backbone AND the
receptor AND the source backbone into one npz, plus a JSONL row of the RAW backbone
measurements. The CPU packing job reads those; the report job reads what packing writes.

WHY THE RECEPTOR IS IN EVERY DUMP, rather than reloaded later by example id. The loader's
frame is PER-PROCESS: two loads of one example were measured to differ by ~2.7 A, so a
peptide saved without its receptor can never be re-united with the right one. Writing the
complex at sampling time is the only version of this that is correct, and it is cheap.

WHY THE SPLIT AT ALL. Packing is CPU work -- 3 Rosetta runs per attempt -- and asking for a
GPU to do it would queue it behind real training. Splitting also makes the figures redrawable:
a threshold can change and the report regenerates from these rows without sampling a peptide.

PAIRED SEEDS ACROSS ARMS. The seed list is a configured constant, not a draw, so attempt k of
source s in Arm A and in Arm B come from the same noise draw. Without that, a difference
between the arms at a fixed budget is partly a difference between two sets of random numbers.

A SOURCE-TORSION BASELINE is generated alongside, under arm name `source_torsion`: the source
backbone's own torsions rebuilt through the same builder and packed through the same routine.
It exposes template drift and trivial near-identity results, and it is NOT a third arm -- it
trains nothing and is reported as a reference row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

#: Reported as an arm in the manifest, generated like one, trained like nothing.
SOURCE_TORSION_ARM = "source_torsion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the arm's config, e.g. example/training_cp2lp_bb_torsion")
    p.add_argument("--ckpt", required=True, help="checkpoint to evaluate")
    p.add_argument("--ckpt-tag", required=True, help="label for this milestone, e.g. g005300-EMA")
    p.add_argument("--manifest", required=True, help="panel parquet from cp2lp_eval_manifest.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default="0,1,2,3", help="PAIRED across arms; 4 attempts per source")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--source-torsion-baseline", action="store_true",
        help="also dump the source-torsion reconstruction as a reference arm",
    )
    return p.parse_args()


def main() -> int:  # noqa: C901
    args = parse_args()
    load_dotenv(".env")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name=args.config)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.cp2lp.mode) != "backbone":
        logger.error(f"this evaluator is for cp2lp.mode=backbone, got {cfg.cp2lp.mode!r}")
        return 1

    arm = str(cfg.cp2lp.backbone.arm)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    out = Path(args.out_dir)
    (out / "dumps").mkdir(parents=True, exist_ok=True)

    cfg.dataset.unified.datamodule.metadata_file = args.manifest
    cfg.dataset.unified.datamodule.val_metadata_file = args.manifest
    # The panel is the WHOLE dataset for this job, and every row must be used exactly once.
    cfg.dataset.unified.datamodule.source_fractions = None
    cfg.dataset.unified.datamodule.batch_size = int(args.batch)
    cfg.dataset.datamodule.num_workers = 2
    cfg.dataset.unified.datamodule.num_workers = 2
    if cfg.get("val_generation") is not None:
        cfg.val_generation.enabled = False
    cfg.cp2lp.max_cp_per_step = int(args.batch)
    cfg.cp2lp.samples_per_cp = 1  # the seed loop supplies the repeats, paired across arms

    from proteinfoundation.cp2lp.backbone import (
        backbone_to_torsions,
        backbone_validity,
        extract_backbone,
        resolve_chain_geometry,
        torsions_to_backbone,
    )
    from proteinfoundation.cp2lp.data import slice_batch, split_roles
    from proteinfoundation.cp2lp.editor import sample_editor_noise
    from proteinfoundation.cp2lp.geometry import safe_norm, terminal_gap_nm
    from proteinfoundation.cp2lp.losses import pairwise_ca_rmsd_nm
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    logger.info(f"loading {args.ckpt}")
    model = CP2LPGenerator(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    other = [k for k in missing if not k.startswith(("autoencoder.",))]
    logger.info(f"loaded: {len(other)} unexpected-missing, {len(unexpected)} unexpected")
    if other:
        logger.warning(f"missing keys the checkpoint should have carried: {other[:8]}")
    del ckpt
    device = torch.device(args.device)
    model.to(device).eval()

    _, datamodule = load_data_module(cfg, is_cluster_run=True)
    datamodule.setup("fit")

    rows: list[dict] = []
    n_attempts = 0
    t_start = time.time()
    per_source_ca: dict[str, list[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in datamodule.train_dataloader():
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = add_clean_samples(
                batch, cfg.product_flowmatcher, model.autoencoder,
                local_latent_target=cfg.get("local_latent_target", "sample"),
            )
            cyclic, _ = split_roles(batch)
            if not cyclic.any():
                continue
            cp = slice_batch(batch, cyclic, batch["mask"].shape[0])
            ids = list(cp.get("example_id", [f"row{i}" for i in range(cp["mask"].shape[0])]))

            for seed in seeds:
                bs, n = cp["mask"].shape
                ref = cp["x_1"][model._data_modes[0]]
                gen = torch.Generator(device=ref.device).manual_seed(int(seed))
                noise = sample_editor_noise(
                    bs, n, model.onepass_noise_channels, ref.device, ref.dtype, gen
                )
                gen_batch, decoded, extras = model.generate_backbone(cp, noise=noise)
                mask = gen_batch["mask"].bool()
                gen_bb, src_bb = extras["gen_bb"], extras["src_bb"]

                variants = {arm: gen_bb}
                if args.source_torsion_baseline:
                    internal = backbone_to_torsions(src_bb, mask)
                    geom = resolve_chain_geometry(
                        internal,
                        str(cfg.cp2lp.backbone.angle_source),
                        gen_batch.get("residue_type", None),
                        length_source=str(cfg.cp2lp.backbone.length_source),
                    )
                    variants[SOURCE_TORSION_ARM] = torsions_to_backbone(
                        internal["phi"], internal["psi"], internal["omega"],
                        internal["root"], mask, geom,
                    )

                for vname, vbb in variants.items():
                    v = backbone_validity(vbb, mask)
                    gap = terminal_gap_nm(*_atom37(vbb, mask))
                    src_gap = terminal_gap_nm(*_atom37(src_bb, mask))
                    d = safe_norm(vbb - src_bb, dim=-1)
                    ca = safe_norm(
                        vbb[:, :, 1] - src_bb[:, :, 1], dim=-1
                    ) * mask.float()
                    for i in range(vbb.shape[0]):
                        L = int(mask[i].sum())
                        ex = str(ids[i]) if i < len(ids) else f"row{i}"
                        tag = f"{vname}__{args.ckpt_tag}__{ex}__s{seed}"
                        npz = out / "dumps" / f"{tag}.npz"
                        # The receptor travels WITH the peptide. See the module docstring:
                        # the loader frame is per-process and a peptide saved alone cannot be
                        # re-united with its receptor.
                        np.savez_compressed(
                            npz,
                            gen_bb=vbb[i, :L].float().cpu().numpy(),
                            src_bb=src_bb[i, :L].float().cpu().numpy(),
                            pep_aatype=gen_batch["residue_type"][i, :L].cpu().numpy(),
                            target_atom37_ang=(
                                gen_batch["x_target"][i] * 10.0
                            ).float().cpu().numpy(),
                            target_aatype=gen_batch.get(
                                "seq_target", torch.zeros_like(gen_batch["seq_target_mask"])
                            )[i].cpu().numpy(),
                            target_mask=gen_batch["seq_target_mask"][i].cpu().numpy(),
                        )
                        row = {
                            "arm": vname, "ckpt_tag": args.ckpt_tag, "example_id": ex,
                            "seed": int(seed), "n_residues": L, "npz": str(npz),
                            "terminal_gap_nm": float(gap[i]),
                            "src_terminal_gap_nm": float(src_gap[i]),
                            "edit_bb_rmsd_nm": float(
                                (d[i, :L] ** 2).mean().sqrt()
                            ),
                            "edit_ca_rmsd_nm": float((ca[i, :L] ** 2).mean().sqrt()),
                            "edit_ca_max_nm": float(ca[i, :L].max()),
                            **{k: float(x[i]) for k, x in v.items()},
                        }
                        rows.append(row)
                        per_source_ca.setdefault(f"{vname}__{ex}", []).append(
                            vbb[i, :L, 1].float().cpu()
                        )
                        n_attempts += 1
            logger.info(f"  {n_attempts} attempts so far")

    # WITHIN-SOURCE DIVERSITY, over the paired seeds of one source. A one-pass editor with a
    # dead noise path produces the same LP every time, and that is invisible in any per-
    # attempt column -- it only shows up as this number being zero.
    div: dict[str, float] = {}
    for key, cas in per_source_ca.items():
        if len(cas) < 2:
            continue
        stack = torch.stack(cas, dim=0)
        pw = [
            float(((stack[a] - stack[b]) ** 2).sum(-1).mean().sqrt())
            for a in range(len(cas)) for b in range(a + 1, len(cas))
        ]
        div[key] = float(np.mean(pw)) if pw else 0.0
    for r in rows:
        r["within_source_diversity_nm"] = div.get(f"{r['arm']}__{r['example_id']}", float("nan"))

    jsonl = out / f"attempts__{arm}__{args.ckpt_tag}.jsonl"
    with open(jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "arm": arm, "ckpt": args.ckpt, "ckpt_tag": args.ckpt_tag,
        "manifest": args.manifest, "seeds": seeds,
        "n_attempts": n_attempts, "n_sources": len({r["example_id"] for r in rows}),
        "gpu_seconds": time.time() - t_start,
        "source_torsion_baseline": bool(args.source_torsion_baseline),
    }
    (out / f"summary__{arm}__{args.ckpt_tag}.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {n_attempts} attempts to {jsonl}")
    print(json.dumps(summary, indent=2))
    return 0


def _atom37(bb: torch.Tensor, mask: torch.Tensor):
    from proteinfoundation.cp2lp.backbone import backbone_to_atom37

    a, _ = backbone_to_atom37(bb, mask)
    return a, mask


if __name__ == "__main__":
    raise SystemExit(main())
