"""The trade-off that decides whether this architecture can do the task: extension vs intact.

WHY THE MATCHED-SIZE PROBE IS NOT ENOUGH. `cp2lp_internal_edit_probe.py` compares the two edit
parameterisations at equal RMS displacement, and it showed the internal space keeps the CA
trace perfectly native (CA-CA sd unchanged, bend angles inside the window) while the Cartesian
space destroys it. But it uses a RANDOM edit, and a random torsional edit barely extends a
peptide: at sigma_c = 0.1705 it moved the terminal gap only 0.352 -> 0.507 nm while
`chain_intact` fell to 0.078. That understates the parameterisation badly, because the editor
does not need a random edit -- it needs a COORDINATED one. Driving every virtual torsion
toward trans takes a 16-mer to ~5 nm end to end, which is the move the task actually wants.

So the useful measurement is not "damage at a given edit size" but **the curve of
`chain_intact` against terminal gap along a coordinated extension**, which is the trade-off
the trained editor would ride. This script walks the source CP toward the extended
(beta-strand-like) corner of internal-coordinate space by a fraction `f`, decoding at each
step with the SOURCE latents, and reports both axes.

WHAT EACH OUTCOME MEANS. The target is the real bound-LP terminal gap, median 2.14 nm, against
a source CP's 0.35 nm.

* `chain_intact` stays high out to 2.14 nm -> the architecture can do the task, and the
  remaining problem is only that the editor has to LEARN this path. The fix is the
  parameterisation plus training.
* `chain_intact` collapses well before 2.14 nm -> a valid CA trace is not sufficient, because
  the latents must track the torsions and these are the SOURCE latents. The fix then needs the
  latent channel too, and this curve says how much edit the frozen decoder tolerates before
  the latents go stale.

The second outcome is the one the matched-size probe already hinted at, which is why this is
worth measuring rather than assuming: that probe reached `chain_intact` 0.078 with source
latents and a perfectly valid trace, so CA validity was demonstrably not the whole story.

A `latent_oracle` column -- the same walk decoded with latents RE-ENCODED from the decoded
structure -- would separate "the decoder cannot render this trace" from "...not with THESE
latents". It is **not implemented**, deliberately. A faithful re-encode has to rebuild the
encoder's whole input batch, and `coords` and `coords_nm` are DIFFERENT RIGID FRAMES (~19 A
apart) that different feature factories read independently, so overriding only the nm
coordinates would hand the encoder a structure half in one frame and half in the other and
return latents that mean nothing. The column would look authoritative and be wrong, which is
worse than its absence. Everything below uses the SOURCE latents, which makes every number a
FLOOR for the parameterisation rather than a prediction: a trained editor co-edits the latents,
and the control arm is already measured doing so (its edited latents recover +0.234 of
`chain_intact` over a source-latent substitution).
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

#: Real bound-LP terminal gap, median, nm. The target the walk is measured against.
REAL_LP_TERMINAL_GAP_NM = 2.14

#: The extended corner of CA internal-coordinate space. MEASURED in
#: `test_cp2lp_ca_internal.test_secondary_structure_corners_reproduce_published_rises`:
#: theta 120 deg with tau 180 deg reproduces the beta-strand rise of 3.33 A per residue.
EXTENDED_THETA_DEG = 120.0
EXTENDED_TAU_DEG = 180.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cp-parquet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--fractions", default="0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    cfgs = sorted(run_dir.glob("*exp_config*.json")) or sorted(run_dir.glob("**/*exp_config*.json"))
    if not cfgs:
        logger.error(f"no exp_config json under {run_dir}")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proteinfoundation.cp2lp.ca_internal import apply_internal_residual, ca_to_internal
    from proteinfoundation.cp2lp.geometry import (
        C_IDX,
        IDEAL_PEPTIDE_BOND_NM,
        N_IDX,
        PEPTIDE_BOND_TOL_NM,
        open_chain_bond_pair_mask,
    )
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cp2lp_ca_spacing_probe import bend_angles_deg, bend_triple_mask, bond_lengths, masked_stats

    cfg = OmegaConf.create(json.load(cfgs[0].open()))
    OmegaConf.set_struct(cfg, False)
    dm = cfg.dataset.unified.datamodule
    dm.metadata_file = str(args.cp_parquet)
    dm.val_metadata_file = str(args.cp_parquet)
    dm.val_filters = None
    dm.source_fractions = None
    if cfg.get("rollout_finetune") is not None:
        cfg.rollout_finetune.enabled = False
    if cfg.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    _, datamodule = load_data_module(cfg, is_cluster_run=True)
    datamodule.setup("validate")
    model = CP2LPGenerator(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=False)
    del ck
    device = torch.device(args.device)
    model.to(device).eval()

    fractions = [float(x) for x in args.fractions.split(",")]
    rows = {f: {k: [] for k in ("intact", "gap", "cn_dev", "ca_ca", "bend", "disp")}
            for f in fractions}

    for bidx, batch in enumerate(datamodule.val_dataloader()):
        if bidx >= args.n_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            ref = add_clean_samples(batch, cfg.product_flowmatcher, model.autoencoder)
            mask = ref["mask"].bool()
            bond_mask = open_chain_bond_pair_mask(mask)
            tri = bend_triple_mask(mask)
            src_ca = ref["x_1"]["bb_ca"]
            latents = ref["x_1"]["local_latents"]
            b, n, _ = src_ca.shape
            mf = mask.to(src_ca.dtype)
            internal = ca_to_internal(src_ca)

            # The coordinated direction: toward the extended corner, per angle. Taking the
            # SHORT way round on tau matters -- a naive difference can ask for a +350 deg turn
            # where -10 deg was meant, which is the same conformation by a wildly different path
            # and would misreport the edit size.
            want_theta = torch.full_like(internal["theta"], math.radians(EXTENDED_THETA_DEG))
            want_tau = torch.full_like(internal["tau"], math.radians(EXTENDED_TAU_DEG))
            d_theta_full = want_theta - internal["theta"]
            d_tau_full = torch.remainder(want_tau - internal["tau"] + math.pi, 2 * math.pi) - math.pi

            def gap(ca):
                L = mask.sum(1)
                i = torch.arange(b, device=ca.device)
                return torch.sqrt(((ca[i, (L - 1).clamp(min=0)] - ca[i, 0]) ** 2).sum(-1) + 1e-12)

            for f in fractions:
                ca, _ = apply_internal_residual(
                    src_ca, mask, delta_theta=d_theta_full * f, delta_tau=d_tau_full * f
                )
                dec = model._decode({"bb_ca": ca, "local_latents": latents}, mask)
                c = dec["coors_nm"]
                d = torch.sqrt(((c[:, :-1, C_IDX, :] - c[:, 1:, N_IDX, :]) ** 2).sum(-1) + 1e-12)
                dev = (d - IDEAL_PEPTIDE_BOND_NM).abs()
                intact = ((dev <= PEPTIDE_BOND_TOL_NM) | ~bond_mask).all(dim=1).float()

                r = rows[f]
                r["intact"].append(intact.cpu())
                r["gap"].append(gap(ca).cpu())
                r["cn_dev"].append(dev[bond_mask].cpu())
                r["ca_ca"].append(bond_lengths(ca)[bond_mask].cpu())
                r["bend"].append(bend_angles_deg(ca)[tri].cpu())
                r["disp"].append(
                    torch.sqrt((((ca - src_ca) ** 2).sum(-1) * mf).sum() / mf.sum().clamp(min=1) / 3.0
                               ).reshape(1).cpu()
                )
        logger.info(f"batch {bidx} done")

    if not rows[fractions[0]]["intact"]:
        logger.error("no batches measured")
        return 1

    ones = lambda t: torch.ones_like(t, dtype=torch.bool)  # noqa: E731
    out = {
        "run_dir": str(run_dir),
        "ckpt": args.ckpt,
        "real_lp_terminal_gap_nm": REAL_LP_TERMINAL_GAP_NM,
        "extended_corner": {"theta_deg": EXTENDED_THETA_DEG, "tau_deg": EXTENDED_TAU_DEG},
        "points": [],
    }
    for f in fractions:
        cat = {k: torch.cat(v) for k, v in rows[f].items()}
        out["points"].append({
            "fraction": f,
            "chain_intact_frac": float(cat["intact"].mean()),
            "terminal_gap_nm": float(cat["gap"].mean()),
            "terminal_gap_nm_median": float(cat["gap"].median()),
            "rms_ca_displacement_nm": float(cat["disp"].mean()),
            "cn_dev_nm": masked_stats(cat["cn_dev"], ones(cat["cn_dev"])),
            "ca_ca_nm": masked_stats(cat["ca_ca"], ones(cat["ca_ca"])),
            "bend_deg": masked_stats(cat["bend"], ones(cat["bend"])),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info(f"{'frac':>6s} {'gap nm':>8s} {'intact':>8s} {'disp':>7s} "
                f"{'CA-CA sd':>9s} {'bend min':>9s} {'cn med':>8s}")
    for p in out["points"]:
        logger.info(
            f"{p['fraction']:6.2f} {p['terminal_gap_nm']:8.3f} {p['chain_intact_frac']:8.4f} "
            f"{p['rms_ca_displacement_nm']:7.4f} {p['ca_ca_nm']['sd']:9.5f} "
            f"{p['bend_deg']['min']:9.1f} {p['cn_dev_nm']['median']:8.4f}"
        )
    reached = [p for p in out["points"] if p["terminal_gap_nm"] >= REAL_LP_TERMINAL_GAP_NM]
    if reached:
        logger.info(
            f"at the real-LP gap of {REAL_LP_TERMINAL_GAP_NM} nm, chain_intact is "
            f"{reached[0]['chain_intact_frac']:.4f} (fraction {reached[0]['fraction']})"
        )
    else:
        logger.warning(
            f"the walk never reached {REAL_LP_TERMINAL_GAP_NM} nm "
            f"(max {max(p['terminal_gap_nm'] for p in out['points']):.3f}); "
            "extend --fractions past 1.0 or the extended corner is not enough on its own."
        )
    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
