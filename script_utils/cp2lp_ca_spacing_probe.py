"""Why the chain is not intact: measure the CA track the decoder is actually handed.

THE QUESTION. The one-pass editor exports at ~1% `chain_intact` in every adversarial arm
while the control exports at ~79%, and the 0.02 nm bond tolerance is NOT the culprit -- it
is calibrated on a plateau and natives sit 6x inside it (docs/README_CP2LP.md). So the
generated backbones are genuinely broken. This probe asks which input breaks them.

THE MECHANISM UNDER TEST. The editor emits a per-residue Cartesian residual and adds it to
the CA track (`editor.apply_residuals`: `edited[bb_ca] = src[bb_ca] + delta`), then a FROZEN
decoder builds N, C and O from that track. Three facts make the CA track the suspect:

* CA(i)-CA(i+1) is the most rigid quantity in a peptide backbone. MEASURED here on 13,837
  bonded pairs from 101 native crystal complexes: **3.809 +- 0.048 A**, p1 3.741, p99 3.885,
  99.8% trans. With ideal bond lengths and angles the span is a function of omega alone and
  its **ceiling is 3.804 A** -- CA-CA cannot exceed that, at any omega, ever.
* nothing in the loss set constrains it. `losses.open_chain_geometry_loss` scores
  C(i)-N(i+1) and intra-residue N-CA / CA-C on the DECODED atom37, i.e. downstream of the
  frozen decoder, so the cause is reachable only through the decoder's Jacobian.
* the decoder is independently CA-dominated. From the run's own four-way decode ablation:
  clean CA + clean latents -> 95.2% mainchain C-N success; clean latents + CA perturbed by
  1.95 A MAE -> **12.5%**; clean CA + perturbed latents -> 35.9%.

WHAT THIS PROBE ADDS, AND WHY THE EXISTING NUMBER IS NOT ENOUGH. `delta_ca_rms_nm` is
logged, but it is the RMS of ONE Cartesian COMPONENT over all residues -- it cannot
distinguish the edit the task wants from the edit that breaks the chain. Translating or
smoothly bending the peptide moves every CA and changes no CA-CA distance; only the
INCOHERENT, high-spatial-frequency part of the residual stretches the virtual bond. The
independent-residual upper bound (sqrt(2) * sigma_component = 2.42 A of CA-CA change for
the gan arm, 50 sigma) is clearly too pessimistic, because the realized decoded C-N MAE is
only 0.344 A. So the residual must be substantially correlated across neighbours, and the
open question is how much incoherence is left. That is what decides the fix:

    small incoherent part  -> a direct CA-CA term (short gradient path, no decoder in
                              between) is enough;
    large incoherent part  -> the free Cartesian residual cannot express a chain-preserving
                              edit and the parameterisation has to change.

So the probe reports the decomposition rather than a single magnitude:

    sigma_component          the logged quantity, reproduced as a cross-check
    ds_firstorder_rms        RMS of (delta_{i+1} - delta_i) . u_i, the first-order CA-CA
                             length change -- the chain-breaking projection, exactly
    ds_exact_rms             |edited bond| - |source bond|, without the linearisation
    incoherence              ds_exact_rms / (sqrt(2) * sigma_component); 1.0 = residuals
                             independent across neighbours, ~0 = pure coherent motion

THE COUNTERFACTUAL IS THE VERDICT. Measuring correlation would leave the direction of
causation open, so the probe also PROJECTS the edited CA track back onto the constraint
(every virtual bond rescaled to 3.809 A by damped iterative relaxation), re-decodes with
the SAME latents, and re-scores. Reading:

    chain_intact recovers, projection RMSD small  -> CA spacing IS the cause and a CA-CA
                                                    loss is the fix
    chain_intact recovers, projection RMSD large  -> cause confirmed, but the edit is far
                                                    from any valid chain: reparameterise
    chain_intact does not recover                 -> the latents are off-manifold too and
                                                    the CA track is not the whole story

`--reference-mode` runs the same measurement on real bound LPs through the identical code
path, which is what makes the generated CA-CA numbers comparable to anything.

NOT A TRAINING CHANGE. This file only reads; it imports the cp2lp modules and does not
modify them, because five arms are mid-flight against this working tree and a requeue
re-reads it.

Run via `scripts/cp2lp_ca_probe.sbatch`; see `scripts/submit_cp2lp_ca_probe.sh`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

#: Native CA(i)-CA(i+1) separation in nm. MEASURED on 13,837 bonded pairs from the 101
#: native complexes in `calibration_native/`: median 3.809 A, sd 0.048 A, p1 3.741,
#: p99 3.885, 99.77% trans-like (>3.4 A). The ideal-geometry ceiling at omega=180 is
#: 3.804 A, so this is a rod, not a range.
NATIVE_CA_CA_NM = 0.3809
NATIVE_CA_CA_SD_NM = 0.0048

#: Window admitting ~98% of natives (p1..p99). Used only for the "outside window" fraction.
CA_CA_P1_NM = 0.3741
CA_CA_P99_NM = 0.3885


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="training run dir (holds exp_config json)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cp-parquet", required=True)
    p.add_argument("--out", required=True, help="output JSON summary path")
    p.add_argument("--rows-out", default=None, help="optional per-bond JSONL (subsampled)")
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--proj-iters", type=int, default=600)
    p.add_argument("--proj-omega", type=float, default=0.5, help="relaxation factor")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--reference-mode", action="store_true", help="measure real LPs, no editor")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- geometry


def bond_lengths(ca: torch.Tensor) -> torch.Tensor:
    """[b, n-1] consecutive CA-CA distances."""
    v = ca[:, 1:] - ca[:, :-1]
    return torch.sqrt((v * v).sum(-1) + 1e-12)


def bend_angles_deg(ca: torch.Tensor) -> torch.Tensor:
    """[b, n-2] CA(i-1)-CA(i)-CA(i+1) angles in degrees.

    The SECOND constraint on a CA trace, and the one a bond-length projection leaves
    untouched. A trace can have every CA-CA at exactly 3.81 A and still be impossible,
    because the bend angle is bounded below by sterics (two CAs cannot fold onto each
    other) and above by the extended-chain limit.
    """
    a = ca[:, :-2] - ca[:, 1:-1]
    b = ca[:, 2:] - ca[:, 1:-1]
    an = a / torch.sqrt((a * a).sum(-1, keepdim=True) + 1e-12)
    bn = b / torch.sqrt((b * b).sum(-1, keepdim=True) + 1e-12)
    cos = (an * bn).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def bend_triple_mask(mask: torch.Tensor) -> torch.Tensor:
    """[b, n-2] mask of (i-1, i, i+1) triples with all three residues present."""
    m = mask.bool()
    return m[:, :-2] & m[:, 1:-1] & m[:, 2:]


def project_ca_spacing(
    ca: torch.Tensor,
    bond_mask: torch.Tensor,
    d0: float,
    iters: int,
    omega: float,
) -> torch.Tensor:
    """Rescale every bonded CA-CA separation to ``d0`` by damped iterative relaxation.

    Each bond pushes its two endpoints symmetrically, so the update introduces no net
    translation and the peptide stays where the editor put it. Simultaneous (Jacobi) updates
    on a chain overshoot, hence ``omega`` < 1; convergence is reported by the caller as the
    residual max |L - d0| rather than assumed.

    Padded residues are untouched: their bonds are masked out of the update.
    """
    p = ca.clone()
    m = bond_mask[..., None].to(p.dtype)
    for _ in range(iters):
        v = p[:, 1:] - p[:, :-1]
        L = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
        corr = omega * 0.5 * (L - d0) * (v / L) * m
        upd = torch.zeros_like(p)
        upd[:, :-1] = upd[:, :-1] + corr
        upd[:, 1:] = upd[:, 1:] - corr
        p = p + upd
    return p


def masked_stats(x: torch.Tensor, m: torch.Tensor) -> dict:
    """Summary of ``x`` over the True entries of ``m``. Both [*]; returns plain floats."""
    v = x[m.bool()].detach().float().cpu()
    if v.numel() == 0:
        return {"n": 0}
    s = v.sort().values
    def q(f: float) -> float:
        return float(s[min(s.numel() - 1, int(f * s.numel()))])
    return {
        "n": int(v.numel()),
        "mean": float(v.mean()),
        "sd": float(v.std(unbiased=False)),
        "min": float(s[0]),
        "p1": q(0.01),
        "median": q(0.50),
        "p99": q(0.99),
        "max": float(s[-1]),
    }


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    den = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / den)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def rank(x: torch.Tensor) -> torch.Tensor:
        idx = x.argsort()
        r = torch.empty_like(x, dtype=torch.float64)
        r[idx] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
        return r
    return pearson(rank(a), rank(b))


# --------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    cfg_files = sorted(run_dir.glob("*exp_config*.json")) or sorted(run_dir.glob("**/*exp_config*.json"))
    if not cfg_files:
        logger.error(f"no exp_config json under {run_dir}")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proteinfoundation.cp2lp.geometry import (
        C_IDX,
        IDEAL_PEPTIDE_BOND_NM,
        N_IDX,
        PEPTIDE_BOND_TOL_NM,
        open_chain_bond_pair_mask,
    )
    from proteinfoundation.cp2lp.losses import open_chain_geometry_loss
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    cfg_exp = OmegaConf.create(json.load(cfg_files[0].open()))
    OmegaConf.set_struct(cfg_exp, False)
    dm = cfg_exp.dataset.unified.datamodule
    dm.metadata_file = str(args.cp_parquet)
    dm.val_metadata_file = str(args.cp_parquet)
    dm.val_filters = None
    # Same reason as cp2lp_generate.py: the inherited three-way training mix names sources
    # the single-source eval parquet does not have, and setup() raises before the model loads.
    dm.source_fractions = None
    if cfg_exp.get("rollout_finetune") is not None:
        cfg_exp.rollout_finetune.enabled = False
    if cfg_exp.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    _, datamodule = load_data_module(cfg_exp, is_cluster_run=True)
    datamodule.setup("validate")

    model = CP2LPGenerator(cfg_exp)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    g_updates = int(model.g_updates) if hasattr(model, "g_updates") else None
    logger.info(
        f"loaded {Path(args.ckpt).name} global_step={ck.get('global_step')} g_updates={g_updates} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    del ck
    device = torch.device(args.device)
    model.to(device).eval()

    if getattr(model, "cp2lp_mode", "rollout") != "onepass" and not args.reference_mode:
        logger.error("this probe measures the one-pass editor's residual; run-dir is not onepass.")
        return 1

    loader = datamodule.val_dataloader()
    rows: list[dict] = []
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in (
        "src_bond", "edit_bond", "proj_bond",
        "src_bend", "edit_bend", "proj_bend",
        "ds_first", "ds_exact",
        "cn_dev", "cn_dev_proj",
        "ca_dev_edit",
    )}
    # THE 2x2, plus the two projected arms. Deliberately the same shape as the run's own
    # `val/decode_*` ablation, so the two can be read side by side: that one varies the FLOW
    # model's prediction, this one varies the EDITOR's residual, and the question -- which
    # decoder input carries the damage -- is identical.
    GRID = ("edit_ca__edit_z", "src_ca__edit_z", "edit_ca__src_z", "src_ca__src_z",
            "proj_ca__edit_z", "proj_ca__src_z")
    grid_intact = {k: [] for k in GRID}
    grid_cn_med = {k: [] for k in GRID}
    per_example = {k: [] for k in ("intact", "intact_proj", "proj_rmsd_nm", "gap_edit_nm", "gap_proj_nm")}
    sig_sq_sum = torch.zeros((), dtype=torch.float64)
    sig_n = torch.zeros((), dtype=torch.float64)
    proj_resid_max = 0.0

    for bidx, batch in enumerate(loader):
        if bidx >= args.n_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            ref = add_clean_samples(batch, cfg_exp.product_flowmatcher, model.autoencoder)
            mask = ref["mask"].bool()
            bond_mask = open_chain_bond_pair_mask(mask)

            if args.reference_mode:
                # Real LPs: the encoded clean state IS the measurement. No editor.
                src_ca = ref["x_1"]["bb_ca"]
                edit_ca = src_ca
                latents = ref["x_1"]["local_latents"]
                src_latents = latents
                delta = torch.zeros_like(src_ca)
                decoded = model._decode({"bb_ca": src_ca, "local_latents": latents}, mask)
            else:
                gen_batch, decoded, extras = model.generate_onepass(ref)
                mask = gen_batch["mask"].bool()
                bond_mask = open_chain_bond_pair_mask(mask)
                src_ca = gen_batch["x_t"]["bb_ca"]
                edit_ca = extras["edited"]["bb_ca"]
                latents = extras["edited"]["local_latents"]
                src_latents = gen_batch["x_t"]["local_latents"]
                delta = extras["deltas"]["bb_ca"]

            # ---- CA-CA spacing, source vs edited
            src_bond = bond_lengths(src_ca)
            edit_bond = bond_lengths(edit_ca)

            # ---- the chain-breaking projection of the residual, first order and exact
            u = (src_ca[:, 1:] - src_ca[:, :-1])
            u = u / torch.sqrt((u * u).sum(-1, keepdim=True) + 1e-12)
            ds_first = ((delta[:, 1:] - delta[:, :-1]) * u).sum(-1)
            ds_exact = edit_bond - src_bond

            # ---- sigma per Cartesian component, reproducing the logged metric exactly
            sig_sq_sum += (delta[mask] ** 2).sum().double().cpu()
            sig_n += (mask.sum() * 3).double().cpu()

            # ---- decoded peptide bond deviation, per bond
            cn = decoded["coors_nm"]
            d_cn = torch.sqrt(
                ((cn[:, :-1, C_IDX, :] - cn[:, 1:, N_IDX, :]) ** 2).sum(-1) + 1e-12
            )
            cn_dev = (d_cn - IDEAL_PEPTIDE_BOND_NM).abs()

            # ---- the counterfactual: valid CA spacing, SAME latents, re-decode
            proj_ca = project_ca_spacing(
                edit_ca, bond_mask, NATIVE_CA_CA_NM, args.proj_iters, args.proj_omega
            )
            proj_bond = bond_lengths(proj_ca)
            proj_resid_max = max(
                proj_resid_max, float((proj_bond - NATIVE_CA_CA_NM).abs()[bond_mask].max())
            )
            # translation-align so the projection does not move the peptide in the pocket
            mf = mask[..., None].to(proj_ca.dtype)
            nres = mf.sum(1).clamp(min=1.0)
            shift = ((edit_ca * mf).sum(1) - (proj_ca * mf).sum(1)) / nres
            proj_ca = proj_ca + shift[:, None, :]
            decoded_proj = model._decode({"bb_ca": proj_ca, "local_latents": latents}, mask)
            cnp = decoded_proj["coors_nm"]
            d_cnp = torch.sqrt(
                ((cnp[:, :-1, C_IDX, :] - cnp[:, 1:, N_IDX, :]) ** 2).sum(-1) + 1e-12
            )
            cn_dev_proj = (d_cnp - IDEAL_PEPTIDE_BOND_NM).abs()

            # ---- per-example chain_intact, through the SAME function training uses
            _, m_edit = open_chain_geometry_loss(cn, decoded["atom_mask_eff"], mask)
            _, m_proj = open_chain_geometry_loss(cnp, decoded_proj["atom_mask_eff"], mask)
            intact = ((cn_dev <= PEPTIDE_BOND_TOL_NM) | ~bond_mask).all(dim=1).float()
            intact_p = ((cn_dev_proj <= PEPTIDE_BOND_TOL_NM) | ~bond_mask).all(dim=1).float()
            # cross-check the per-example recomputation against the training metric
            if abs(float(intact.mean()) - float(m_edit["geom_chain_intact_frac"])) > 1e-5:
                logger.warning(
                    f"batch {bidx}: local intact {float(intact.mean()):.4f} != "
                    f"loss-side {float(m_edit['geom_chain_intact_frac']):.4f}"
                )

            # ---- THE 2x2: which decoder input carries the damage?
            # Scored through the same all-bonds test the export gate uses, so every cell is
            # comparable to `chain_intact` rather than to a loss value.
            for name, (ca_in, z_in) in {
                "edit_ca__edit_z": (edit_ca, latents),
                "src_ca__edit_z": (src_ca, latents),
                "edit_ca__src_z": (edit_ca, src_latents),
                "src_ca__src_z": (src_ca, src_latents),
                "proj_ca__edit_z": (proj_ca, latents),
                "proj_ca__src_z": (proj_ca, src_latents),
            }.items():
                dec = model._decode({"bb_ca": ca_in, "local_latents": z_in}, mask)
                cc = dec["coors_nm"]
                dd = torch.sqrt(
                    ((cc[:, :-1, C_IDX, :] - cc[:, 1:, N_IDX, :]) ** 2).sum(-1) + 1e-12
                )
                dv = (dd - IDEAL_PEPTIDE_BOND_NM).abs()
                ok = ((dv <= PEPTIDE_BOND_TOL_NM) | ~bond_mask).all(dim=1).float()
                grid_intact[name].append(ok.cpu())
                grid_cn_med[name].append(dv[bond_mask].cpu())

            # how far the projection had to move the CA track
            sq = (((proj_ca - edit_ca) ** 2).sum(-1) * mask).sum(1) / mask.sum(1).clamp(min=1)
            proj_rmsd = torch.sqrt(sq + 1e-12)

            def gap(ca):
                lengths = mask.sum(1)
                i = torch.arange(ca.shape[0], device=ca.device)
                return torch.sqrt(
                    ((ca[i, (lengths - 1).clamp(min=0)] - ca[i, 0]) ** 2).sum(-1) + 1e-12
                )

            tri = bend_triple_mask(mask)
            acc["src_bend"].append(bend_angles_deg(src_ca)[tri].cpu())
            acc["edit_bend"].append(bend_angles_deg(edit_ca)[tri].cpu())
            acc["proj_bend"].append(bend_angles_deg(proj_ca)[tri].cpu())
            acc["src_bond"].append(src_bond[bond_mask].cpu())
            acc["edit_bond"].append(edit_bond[bond_mask].cpu())
            acc["proj_bond"].append(proj_bond[bond_mask].cpu())
            acc["ds_first"].append(ds_first[bond_mask].cpu())
            acc["ds_exact"].append(ds_exact[bond_mask].cpu())
            acc["cn_dev"].append(cn_dev[bond_mask].cpu())
            acc["cn_dev_proj"].append(cn_dev_proj[bond_mask].cpu())
            acc["ca_dev_edit"].append((edit_bond - NATIVE_CA_CA_NM).abs()[bond_mask].cpu())
            per_example["intact"].append(intact.cpu())
            per_example["intact_proj"].append(intact_p.cpu())
            per_example["proj_rmsd_nm"].append(proj_rmsd.cpu())
            per_example["gap_edit_nm"].append(gap(edit_ca).cpu())
            per_example["gap_proj_nm"].append(gap(proj_ca).cpu())

            if args.rows_out and len(rows) < args.max_rows:
                bi, pi = bond_mask.nonzero(as_tuple=True)
                for k in range(min(bi.numel(), args.max_rows - len(rows))):
                    b_, p_ = int(bi[k]), int(pi[k])
                    rows.append({
                        "batch": bidx, "example": b_, "bond": p_,
                        "src_ca_ca_nm": float(src_bond[b_, p_]),
                        "edit_ca_ca_nm": float(edit_bond[b_, p_]),
                        "ds_first_nm": float(ds_first[b_, p_]),
                        "cn_dev_nm": float(cn_dev[b_, p_]),
                        "cn_dev_proj_nm": float(cn_dev_proj[b_, p_]),
                    })
        logger.info(f"batch {bidx}: intact {float(intact.mean()):.4f} -> projected {float(intact_p.mean()):.4f}")

    if not acc["src_bond"]:
        logger.error("no batches measured")
        return 1
    cat = {k: torch.cat(v) for k, v in acc.items() if v}
    pex = {k: torch.cat(v) for k, v in per_example.items() if v}
    sigma_c = float(torch.sqrt(sig_sq_sum / sig_n.clamp(min=1)))
    indep_pred = math.sqrt(2.0) * sigma_c
    ds_exact_rms = float(cat["ds_exact"].double().pow(2).mean().sqrt())
    ds_first_rms = float(cat["ds_first"].double().pow(2).mean().sqrt())

    out = {
        "run_dir": str(run_dir),
        "ckpt": args.ckpt,
        "g_updates": g_updates,
        "reference_mode": bool(args.reference_mode),
        "n_bonds": int(cat["src_bond"].numel()),
        "n_examples": int(pex["intact"].numel()),
        "native_ca_ca_nm": NATIVE_CA_CA_NM,
        "native_ca_ca_sd_nm": NATIVE_CA_CA_SD_NM,
        "ca_ca_nm": {
            "source": masked_stats(cat["src_bond"], torch.ones_like(cat["src_bond"], dtype=torch.bool)),
            "edited": masked_stats(cat["edit_bond"], torch.ones_like(cat["edit_bond"], dtype=torch.bool)),
            "projected": masked_stats(cat["proj_bond"], torch.ones_like(cat["proj_bond"], dtype=torch.bool)),
        },
        # The second CA-track constraint, which the bond-length projection does NOT fix.
        "bend_angle_deg": {
            "source": masked_stats(cat["src_bend"], torch.ones_like(cat["src_bend"], dtype=torch.bool)),
            "edited": masked_stats(cat["edit_bend"], torch.ones_like(cat["edit_bend"], dtype=torch.bool)),
            "projected": masked_stats(cat["proj_bend"], torch.ones_like(cat["proj_bend"], dtype=torch.bool)),
        },
        # Which decoder input carries the damage. Same shape as the run's `val/decode_*`.
        "decode_grid": {
            k: {
                "chain_intact_frac": float(torch.cat(v).mean()),
                "cn_dev_median_nm": float(torch.cat(grid_cn_med[k]).median()),
                "cn_dev_p99_nm": float(torch.cat(grid_cn_med[k]).sort().values[
                    min(torch.cat(grid_cn_med[k]).numel() - 1, int(0.99 * torch.cat(grid_cn_med[k]).numel()))
                ]),
            }
            for k, v in grid_intact.items() if v
        },
        "ca_ca_outside_native_window_frac": {
            "source": float(((cat["src_bond"] < CA_CA_P1_NM) | (cat["src_bond"] > CA_CA_P99_NM)).float().mean()),
            "edited": float(((cat["edit_bond"] < CA_CA_P1_NM) | (cat["edit_bond"] > CA_CA_P99_NM)).float().mean()),
        },
        # The decomposition this probe exists for.
        "residual_decomposition": {
            "sigma_component_nm": sigma_c,
            "independent_prediction_nm": indep_pred,
            "ds_firstorder_rms_nm": ds_first_rms,
            "ds_exact_rms_nm": ds_exact_rms,
            "incoherence_frac": (ds_exact_rms / indep_pred) if indep_pred > 0 else None,
            "ca_ca_change_in_native_sd": ds_exact_rms / NATIVE_CA_CA_SD_NM,
        },
        "cn_dev_nm": {
            "edited": masked_stats(cat["cn_dev"], torch.ones_like(cat["cn_dev"], dtype=torch.bool)),
            "projected": masked_stats(cat["cn_dev_proj"], torch.ones_like(cat["cn_dev_proj"], dtype=torch.bool)),
        },
        "correlation_ca_dev_vs_cn_dev": {
            "pearson": pearson(cat["ca_dev_edit"], cat["cn_dev"]),
            "spearman": spearman(cat["ca_dev_edit"], cat["cn_dev"]),
        },
        "counterfactual": {
            "chain_intact_frac": float(pex["intact"].mean()),
            "chain_intact_frac_projected": float(pex["intact_proj"].mean()),
            "projection_ca_rmsd_nm": float(pex["proj_rmsd_nm"].mean()),
            "projection_residual_max_nm": proj_resid_max,
            "terminal_gap_nm": float(pex["gap_edit_nm"].mean()),
            "terminal_gap_nm_projected": float(pex["gap_proj_nm"].mean()),
        },
        "proj_iters": args.proj_iters,
        "proj_omega": args.proj_omega,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info(json.dumps(out["residual_decomposition"], indent=2))
    logger.info(json.dumps(out["counterfactual"], indent=2))
    logger.info(f"wrote {args.out}")
    if args.rows_out and rows:
        Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.rows_out).open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        logger.info(f"wrote {len(rows)} bond rows -> {args.rows_out}")
    if proj_resid_max > 1e-3:
        logger.warning(
            f"projection did not fully converge (max |L-d0| = {proj_resid_max:.5f} nm); "
            "raise --proj-iters before reading the counterfactual."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
