#!/usr/bin/env python3
"""Phase 1.2 / 1.3 diagnostic: quantify the frozen AE's posterior on CPSea peptides.

Goal
----
The CPSea finetune loss floors ~2.0, entirely in the `local_latents` modality
(~50x the bb_ca loss). The flow model regresses against a *freshly sampled* VAE
latent

    z = mean + exp(log_scale) * eps,   eps ~ N(0, I)

so its per-dimension MSE is lower-bounded by the injected posterior variance
`exp(2*log_scale)`. If the frozen encoder (`complexa_ae.ckpt`, trained on 50-256 aa
chains) is uncertain on short (5-16 aa) cyclic peptides, `log_scale` is large and
~2.0 is an *irreducible* noise floor that no amount of flow training / positional
embedding can remove. This script measures exactly that.

What it reports (read-only; loads AE only, never trains):
  * per-dim scale = exp(log_scale): mean/median/quantiles, stratified by peptide length
  * mean magnitude and sampled-z magnitude
  * per-component KL(q||N(0,1)) and the fraction of latent dims that are "active"
    (KL > 0.1) vs collapsed to the prior (uninformative noise)
  * decode reconstruction RMSD (angstrom) from `mean` and from sampled `z`
    (all-atom / backbone / sidechain), which tells us if the latent even carries
    recoverable sidechain information for peptides
  * data sanity (Phase 1.3): binder residue_pdb_idx contiguity, presence of
    x_target / chains, peptide length distribution

Decision rule (printed as a verdict):
  * large peptide scale (near/above the prior 1.0) + many collapsed dims + poor
    sidechain reconstruction  => the ~2.0 floor is an AE noise floor => fix is the
    AE (Phase 2 finetune), not the flow model or positional embeddings.
  * small scale + good reconstruction => floor is flow-side / data-scale.

Runs on GPU (AE forward + atomworks dataloader). See
`~/slurm/cpsea/cpsea_ae_latent_diag.sh` for the launcher.

Usage
-----
    source env.sh
    python script_utils/diagnose_ae_latents.py \
        --config-name example/training_cpsea_peptide_smoke \
        --split val --num-batches 40 --batch-size 8 \
        --out $PROTEINA_ZFS_PATH/training_runs/diagnostics/ae_latent_diag

Optional matched-length control (e.g. a longer-chain parquet) via
    --metadata /path/train.parquet --val-metadata /path/val.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]

# .env values (CKPT_PATH, CPSEA_DATA_PATH, ...) are shell-local; the app loads them via
# python-dotenv. Do the same so ${oc.env:...} interpolations resolve for a child process.
load_dotenv(REPO / ".env")

# atom37 (openfold) order: 0=N, 1=CA, 2=C, 3=CB, 4=O, ...
BACKBONE_ATOMS = [0, 1, 2, 4]  # N, CA, C, O
CA_IDX = 1
KL_ACTIVE_THRESH = 0.1  # matches AutoEncoder training-time logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="example/training_cpsea_peptide_smoke",
                   help="Hydra config to compose (defines dataset transforms + AE ckpt path).")
    # NOTE: the val StructureDataset is built with is_inference=True (binder coords may be
    # withheld for generation). Use `train` (is_inference=False) to encode real binder
    # coordinates -- this is exactly what training_step feeds to the AE.
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--num-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (raise for faster CPU loading).")
    p.add_argument("--metadata", default=None, help="Override train metadata parquet (control set).")
    p.add_argument("--val-metadata", default=None, help="Override val metadata parquet (control set).")
    p.add_argument("--out", default=None,
                   help="Output dir for JSON + PNGs (default: $PROTEINA_ZFS_PATH/training_runs/diagnostics/ae_latent_diag).")
    p.add_argument("--tag", default=None, help="Optional label for output files (e.g. 'peptide' or 'control').")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def move_to_device(obj, device):
    """Recursively move tensors in nested dict/list to device; leave others as-is."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def masked_flatten(v: torch.Tensor, res_mask: torch.Tensor) -> torch.Tensor:
    """v: [b, n, d] -> [num_valid_residues, d] using res_mask [b, n] (bool)."""
    return v[res_mask.bool()]  # [num_valid, d]


def rmsd_angstrom(pred_nm: torch.Tensor, gt_nm: torch.Tensor, atom_mask: torch.Tensor) -> float:
    """Per-atom RMSD in angstrom over masked atoms. pred/gt: [b, n, A, 3], atom_mask: [b, n, A] bool."""
    if atom_mask.sum() == 0:
        return float("nan")
    diff2 = ((pred_nm - gt_nm) ** 2).sum(dim=-1)  # [b, n, A]  (nm^2)
    msd = diff2[atom_mask.bool()].mean().item()  # nm^2
    return math.sqrt(max(msd, 0.0)) * 10.0  # nm -> angstrom


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import proteinfoundation.patches.atomworks_patches  # noqa: F401  (required for atomize_token)
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    from proteinfoundation.train import load_data_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}")

    # The CPSea training configs are `# @package _global_` with their own `dataset:` block,
    # so `++dataset.datamodule.*` overrides do not merge cleanly. Compose minimally and mutate
    # the datamodule object after instantiation (setup/train_dataloader read these attributes).
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config_name, overrides=["+single=true", "+nolog=true"])

    ae_ckpt = cfg.get("autoencoder_ckpt_path", None)
    if ae_ckpt is None:
        raise SystemExit("cfg.autoencoder_ckpt_path missing; this config does not use local_latents.")
    print(f"[diag] loading AE from {ae_ckpt}")
    ae = AutoEncoder.load_from_checkpoint(ae_ckpt)
    ae = ae.to(device).eval()
    for prm in ae.parameters():
        prm.requires_grad_(False)
    print(f"[diag] latent_dim={ae.latent_dim}")

    _, dm = load_data_module(cfg, is_cluster_run=False)
    # Overrides applied directly on the datamodule (avoids Hydra struct issues).
    dm.num_workers = args.num_workers
    dm.batch_size = args.batch_size
    dm.pin_memory = False
    if args.metadata:
        dm.metadata_file = args.metadata
    if args.val_metadata:
        dm.val_metadata_file = args.val_metadata
    print(f"[diag] train metadata: {dm.metadata_file}")
    dm.setup("fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()

    # Accumulators
    scale_all, mean_all, z_all, kl_all = [], [], [], []  # per-residue, per-dim [*, d]
    active_frac_per_res = []  # fraction of dims active per residue
    per_len = defaultdict(lambda: {"scale": [], "kl_active_frac": [], "sc_rmsd_mean": [],
                                   "sc_rmsd_z": [], "bb_rmsd_mean": [], "allatom_rmsd_mean": []})
    lengths = []
    # data sanity
    idx_noncontig = 0
    n_structs = 0
    has_x_target = 0
    has_chains = 0

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.num_batches:
                break
            batch = move_to_device(batch, device)

            # residue (binder) mask from CA presence
            coord_mask = batch["coord_mask"].bool()  # [b, n, 37]
            res_mask = coord_mask[..., CA_IDX]  # [b, n]
            coords_nm = batch["coords_nm"]  # [b, n, 37, 3]
            # The CPSea dataloader batch has no `mask`/`mask_dict`; AE.encode + its features
            # fall back to batch["mask"] when mask_dict is absent (see seq_feats.py). Provide it
            # so encode does not dereference a missing mask_dict.
            if "mask" not in batch:
                batch["mask"] = res_mask

            # data sanity
            has_x_target += int("x_target" in batch)
            has_chains += int("chains" in batch)
            if "residue_pdb_idx" in batch:
                rpi = batch["residue_pdb_idx"]  # [b, n]
                for b in range(rpi.shape[0]):
                    valid = res_mask[b]
                    idx = rpi[b][valid]
                    n_structs += 1
                    if idx.numel() >= 2:
                        d = (idx[1:] - idx[:-1])
                        if not torch.all(d == 1):
                            idx_noncontig += 1
            else:
                n_structs += int(res_mask.shape[0])

            # ---- encode ----
            enc = ae.encode(batch)
            mean = enc["mean"]           # [b, n, d]
            log_scale = enc["log_scale"]  # [b, n, d]
            z = enc["z_latent"]          # [b, n, d]
            scale = torch.exp(log_scale)  # sigma
            kl = 0.5 * (scale ** 2 + mean ** 2 - 1.0 - 2.0 * log_scale)  # [b, n, d]

            # ---- decode (from mean and from sampled z) ----
            ca_nm = coords_nm[..., CA_IDX, :] * res_mask[..., None]  # [b, n, 3]
            dec_mean = ae.decode(z_latent=mean, ca_coors_nm=ca_nm, mask=res_mask)
            dec_z = ae.decode(z_latent=z, ca_coors_nm=ca_nm, mask=res_mask)

            # reconstruction masks over ground-truth present atoms within binder
            gt_atom_mask = coord_mask & res_mask[..., None]  # [b, n, 37]
            bb_mask = torch.zeros_like(gt_atom_mask)
            bb_mask[..., BACKBONE_ATOMS] = True
            bb_mask = bb_mask & gt_atom_mask
            sc_mask = gt_atom_mask.clone()
            sc_mask[..., BACKBONE_ATOMS] = False  # sidechain = all present non-backbone atoms

            # ---- per-residue accumulation ----
            sm = masked_flatten(scale, res_mask).float().cpu().numpy()
            mm = masked_flatten(mean, res_mask).float().cpu().numpy()
            zm = masked_flatten(z, res_mask).float().cpu().numpy()
            km = masked_flatten(kl, res_mask).float().cpu().numpy()
            scale_all.append(sm)
            mean_all.append(mm)
            z_all.append(zm)
            kl_all.append(km)
            active_frac_per_res.append((km > KL_ACTIVE_THRESH).mean(axis=1))  # [num_valid]

            # ---- per-structure, stratified by peptide length ----
            for b in range(res_mask.shape[0]):
                m = res_mask[b]
                L = int(m.sum().item())
                if L == 0:
                    continue
                lengths.append(L)
                s_b = torch.exp(log_scale[b][m]).float().cpu().numpy()  # [L, d]
                k_b = kl[b][m].float().cpu().numpy()
                per_len[L]["scale"].append(float(s_b.mean()))
                per_len[L]["kl_active_frac"].append(float((k_b > KL_ACTIVE_THRESH).mean()))

                gm = gt_atom_mask[b:b + 1]
                bm = bb_mask[b:b + 1]
                cm = sc_mask[b:b + 1]
                per_len[L]["allatom_rmsd_mean"].append(
                    rmsd_angstrom(dec_mean["coors_nm"][b:b + 1], coords_nm[b:b + 1], gm))
                per_len[L]["bb_rmsd_mean"].append(
                    rmsd_angstrom(dec_mean["coors_nm"][b:b + 1], coords_nm[b:b + 1], bm))
                per_len[L]["sc_rmsd_mean"].append(
                    rmsd_angstrom(dec_mean["coors_nm"][b:b + 1], coords_nm[b:b + 1], cm))
                per_len[L]["sc_rmsd_z"].append(
                    rmsd_angstrom(dec_z["coors_nm"][b:b + 1], coords_nm[b:b + 1], cm))

            n_done += 1
            if n_done % 5 == 0:
                print(f"[diag] processed {n_done}/{args.num_batches} batches")

    if not scale_all:
        raise SystemExit("No batches processed; check dataloader / split.")

    scale_all = np.concatenate(scale_all, axis=0)      # [R, d]
    mean_all = np.concatenate(mean_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    kl_all = np.concatenate(kl_all, axis=0)
    active_frac_per_res = np.concatenate(active_frac_per_res, axis=0)  # [R]
    lengths = np.array(lengths)

    d = scale_all.shape[1]
    # per-dim active fraction across all residues (a dim is "active" if KL>thresh on that residue)
    per_dim_active = (kl_all > KL_ACTIVE_THRESH).mean(axis=0)  # [d]
    near_prior_frac = float(np.mean((scale_all > 0.9) & (scale_all < 1.1)))  # dims collapsed to prior

    def flat_metric(key):
        vals = []
        for L in per_len:
            vals.extend(per_len[L][key])
        return np.array([v for v in vals if not (isinstance(v, float) and math.isnan(v))])

    summary = {
        "config": {
            "config_name": args.config_name,
            "split": args.split,
            "num_batches": n_done,
            "batch_size": args.batch_size,
            "ae_ckpt": str(ae_ckpt),
            "latent_dim": int(d),
            "tag": args.tag,
            "metadata_override": args.metadata,
            "val_metadata_override": args.val_metadata,
        },
        "n_residues": int(scale_all.shape[0]),
        "peptide_length": summarize(lengths.astype(float)),
        "scale_exp_log_scale": summarize(scale_all.reshape(-1)),
        "mean_abs": summarize(np.abs(mean_all).reshape(-1)),
        "z_abs": summarize(np.abs(z_all).reshape(-1)),
        "kl_per_component": summarize(kl_all.reshape(-1)),
        "kl_active_frac_per_residue": summarize(active_frac_per_res),
        "per_dim_active_frac": [float(x) for x in per_dim_active],
        "near_prior_frac_scale_0p9_1p1": near_prior_frac,
        "expected_sigma_sq_mean": float(np.mean(scale_all ** 2)),  # ~ irreducible per-dim floor scale
        "recon_rmsd_ang": {
            "allatom_from_mean": summarize(flat_metric("allatom_rmsd_mean")),
            "backbone_from_mean": summarize(flat_metric("bb_rmsd_mean")),
            "sidechain_from_mean": summarize(flat_metric("sc_rmsd_mean")),
            "sidechain_from_sampled_z": summarize(flat_metric("sc_rmsd_z")),
        },
        "by_length": {
            str(L): {
                "n": len(per_len[L]["scale"]),
                "scale_mean": float(np.mean(per_len[L]["scale"])) if per_len[L]["scale"] else None,
                "kl_active_frac": float(np.mean(per_len[L]["kl_active_frac"])) if per_len[L]["kl_active_frac"] else None,
                "sidechain_rmsd_from_mean": float(np.nanmean(per_len[L]["sc_rmsd_mean"])) if per_len[L]["sc_rmsd_mean"] else None,
            }
            for L in sorted(per_len)
        },
        "data_sanity": {
            "n_structs": n_structs,
            "residue_pdb_idx_noncontiguous": idx_noncontig,
            "batches_with_x_target": has_x_target,
            "batches_with_chains": has_chains,
            "num_batches": n_done,
        },
    }

    # ---- verdict ----
    sigma_mean = summary["scale_exp_log_scale"]["mean"]
    active_mean = summary["kl_active_frac_per_residue"]["mean"]
    sc_rmsd = summary["recon_rmsd_ang"]["sidechain_from_mean"].get("mean", float("nan"))
    verdict = []
    if sigma_mean > 0.8:
        verdict.append(f"HIGH posterior sigma (mean={sigma_mean:.3f}, prior=1.0): latent carries large "
                       f"irreducible noise -> supports AE noise-floor hypothesis.")
    else:
        verdict.append(f"LOW/MODERATE posterior sigma (mean={sigma_mean:.3f}): noise floor unlikely to explain ~2.0.")
    verdict.append(f"KL-active dim fraction (mean per residue) = {active_mean:.3f} "
                   f"({'few informative dims' if active_mean < 0.5 else 'many informative dims'}).")
    verdict.append(f"Sidechain reconstruction RMSD (from mean) = {sc_rmsd:.2f} A "
                   f"({'poor -> latent uninformative for peptides' if sc_rmsd > 1.5 else 'reasonable'}).")
    summary["verdict"] = verdict

    # ---- outputs ----
    out_dir = Path(args.out) if args.out else Path(
        os.environ.get("PROTEINA_ZFS_PATH", str(REPO))) / "training_runs" / "diagnostics" / "ae_latent_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    json_path = out_dir / f"ae_latent_diag{tag}_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ AE LATENT DIAGNOSTIC ================")
    print(f"residues={summary['n_residues']}  latent_dim={d}  peptide_len "
          f"median={summary['peptide_length']['median']:.0f} "
          f"[{summary['peptide_length']['min']:.0f},{summary['peptide_length']['max']:.0f}]")
    print(f"scale=exp(log_scale): mean={sigma_mean:.3f} median={summary['scale_exp_log_scale']['median']:.3f} "
          f"p05={summary['scale_exp_log_scale']['p05']:.3f} p95={summary['scale_exp_log_scale']['p95']:.3f}")
    print(f"E[sigma^2]={summary['expected_sigma_sq_mean']:.3f}   near-prior dim frac={near_prior_frac:.3f}")
    print(f"KL/comp mean={summary['kl_per_component']['mean']:.3f}   "
          f"active-dim frac/res mean={active_mean:.3f}")
    print(f"recon RMSD (A): allatom={summary['recon_rmsd_ang']['allatom_from_mean'].get('mean', float('nan')):.2f} "
          f"backbone={summary['recon_rmsd_ang']['backbone_from_mean'].get('mean', float('nan')):.2f} "
          f"sidechain(mean)={sc_rmsd:.2f} sidechain(z)={summary['recon_rmsd_ang']['sidechain_from_sampled_z'].get('mean', float('nan')):.2f}")
    print(f"data sanity: noncontig_idx={idx_noncontig}/{n_structs}  "
          f"x_target={has_x_target}/{n_done}  chains={has_chains}/{n_done}")
    print("verdict:")
    for v in verdict:
        print("  - " + v)
    print(f"\n[diag] wrote {json_path}")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].hist(scale_all.reshape(-1), bins=60, color="steelblue")
        axes[0, 0].axvline(1.0, color="red", ls="--", label="prior sigma=1")
        axes[0, 0].set_title("scale = exp(log_scale) (per residue-dim)")
        axes[0, 0].set_xlabel("sigma"); axes[0, 0].legend()

        axes[0, 1].hist(kl_all.reshape(-1), bins=60, color="darkorange")
        axes[0, 1].axvline(KL_ACTIVE_THRESH, color="red", ls="--", label=f"active>{KL_ACTIVE_THRESH}")
        axes[0, 1].set_title("KL per component"); axes[0, 1].set_xlabel("KL"); axes[0, 1].legend()

        Ls = sorted(per_len)
        axes[1, 0].plot(Ls, [np.mean(per_len[L]["scale"]) for L in Ls], "o-", label="mean sigma")
        axes[1, 0].plot(Ls, [np.mean(per_len[L]["kl_active_frac"]) for L in Ls], "s-", label="active-dim frac")
        axes[1, 0].set_title("by peptide length"); axes[1, 0].set_xlabel("length"); axes[1, 0].legend()

        axes[1, 1].plot(Ls, [np.nanmean(per_len[L]["sc_rmsd_mean"]) for L in Ls], "o-", label="sidechain (mean)")
        axes[1, 1].plot(Ls, [np.nanmean(per_len[L]["sc_rmsd_z"]) for L in Ls], "s-", label="sidechain (z)")
        axes[1, 1].set_title("recon sidechain RMSD by length"); axes[1, 1].set_xlabel("length")
        axes[1, 1].set_ylabel("RMSD (A)"); axes[1, 1].legend()

        fig.tight_layout()
        png_path = out_dir / f"ae_latent_diag{tag}_{args.split}.png"
        fig.savefig(png_path, dpi=120)
        print(f"[diag] wrote {png_path}")
    except Exception as e:  # plotting is best-effort
        print(f"[diag] plotting skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
