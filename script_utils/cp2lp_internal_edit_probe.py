"""Does the internal-coordinate edit space actually fix `chain_intact`? Matched-size test.

THE EXPERIMENT. The diagnosis (docs/README_CP2LP.md, "Why the chain is not intact") says the
CA track is the damage channel and that a free Cartesian residual cannot avoid wrecking it:
~30-40% of the residual is incoherent, the trace leaves both of its native windows, and
`chain_intact` is 0.0000 against an AE round-trip ceiling of 0.9844. The proposed fix is to
move the residual into CA-trace internal coordinates
(:mod:`proteinfoundation.cp2lp.ca_internal`), where a valid trace is the only reachable output.

That proposal is testable WITHOUT training anything, which is the point of this script. Take
real source CA traces, apply a RANDOM edit of a matched size in each parameterisation, decode
both with the same frozen decoder and the same source latents, and compare. Holding edit size
and decoder fixed isolates the parameterisation from the learning:

    rigid only            a pure rigid motion. The floor -- it changes no internal coordinate,
                          so it must score at the ceiling, and if it does not the harness is
                          wrong rather than the parameterisation.
    cartesian (sigma_c)   iid per-residue Cartesian residual at the gan arm's measured
                          sigma_component = 0.1705 nm. Must reproduce the ~0 chain_intact the
                          trained arm shows; if it does not, the magnitude is not the cause and
                          the whole diagnosis is wrong.
    internal (matched)    bend/torsion deltas scaled by bisection until the RMS CA
                          displacement equals the Cartesian arm's. Same amount of motion, in
                          the constrained space.

The comparison is only meaningful because the sizes are matched, so the script reports the
achieved RMS displacement for every arm and refuses to be read if the match failed.

A random edit is NOT a trained edit, and this measures a ceiling on the parameterisation
rather than a prediction of final quality: it says whether an edit of the required size CAN be
chain-preserving, not whether the editor will learn a useful one. The honest claim it supports
is "the Cartesian space makes this impossible and the internal space does not".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

#: The gan arm's measured per-component residual RMS at g=5300. See the dose-response table.
DEFAULT_TARGET_SIGMA_C_NM = 0.1705


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cp-parquet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--target-sigma-c", type=float, default=DEFAULT_TARGET_SIGMA_C_NM)
    p.add_argument("--sigma-sweep", default=None,
                   help="comma-separated sigma_c values; sweeps instead of a single point. "
                        "The curve is the point: it says how much edit the decoder tolerates "
                        "at all, which a single magnitude cannot.")
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
    from proteinfoundation.cp2lp.ca_internal import (
        apply_internal_residual,
        ca_to_internal,
    )
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

    sigmas = ([float(x) for x in args.sigma_sweep.split(",")]
              if args.sigma_sweep else [args.target_sigma_c])
    ARMS = ("source", "rigid_only", "cartesian", "internal_matched")
    sweep_out = []
    for sigma in sigmas:
        acc = {a: {k: [] for k in ("intact", "cn_dev", "ca_ca", "bend", "disp", "gap")} for a in ARMS}
        clamp_fracs: list[float] = []

        def score(ca, latents, mask, bond_mask, tri):
            dec = model._decode({"bb_ca": ca, "local_latents": latents}, mask)
            c = dec["coors_nm"]
            d = torch.sqrt(((c[:, :-1, C_IDX, :] - c[:, 1:, N_IDX, :]) ** 2).sum(-1) + 1e-12)
            dev = (d - IDEAL_PEPTIDE_BOND_NM).abs()
            intact = ((dev <= PEPTIDE_BOND_TOL_NM) | ~bond_mask).all(dim=1).float()
            return intact, dev[bond_mask], bond_lengths(ca)[bond_mask], bend_angles_deg(ca)[tri]

        def rms_disp(ca, src, mask):
            return torch.sqrt((((ca - src) ** 2).sum(-1) * mask).sum() / mask.sum().clamp(min=1) / 3.0)

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

                def gap(ca):
                    L = mask.sum(1)
                    i = torch.arange(b, device=ca.device)
                    return torch.sqrt(((ca[i, (L - 1).clamp(min=0)] - ca[i, 0]) ** 2).sum(-1) + 1e-12)

                # ---- cartesian arm at the gan arm's measured magnitude
                cart = src_ca + torch.randn_like(src_ca) * sigma * mask[..., None]
                target = rms_disp(cart, src_ca, mf)

                # ---- internal arm, bisected to the SAME RMS displacement
                dth0 = torch.randn(b, max(0, n - 2), device=device)
                dta0 = torch.randn(b, max(0, n - 3), device=device)
                lo, hi = 0.0, 4.0
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    cand, _ = apply_internal_residual(
                        src_ca, mask, delta_theta=dth0 * mid, delta_tau=dta0 * mid
                    )
                    if rms_disp(cand, src_ca, mf) < target:
                        lo = mid
                    else:
                        hi = mid
                internal, idiag = apply_internal_residual(
                    src_ca, mask, delta_theta=dth0 * lo, delta_tau=dta0 * lo
                )
                clamp_fracs.append(float(idiag["internal_bend_clamp_frac"]))

                # ---- rigid-only floor, matched the same way (translation along a random axis)
                axis = torch.randn(b, 3, device=device)
                axis = axis / axis.norm(dim=-1, keepdim=True)
                rigid, _ = apply_internal_residual(
                    src_ca, mask,
                    delta_theta=torch.zeros(b, max(0, n - 2), device=device),
                    delta_tau=torch.zeros(b, max(0, n - 3), device=device),
                    delta_trans=axis * float(target) * (3.0 ** 0.5),
                )

                for name, ca in (("source", src_ca), ("rigid_only", rigid),
                                 ("cartesian", cart), ("internal_matched", internal)):
                    it, dv, bl, bn = score(ca, latents, mask, bond_mask, tri)
                    acc[name]["intact"].append(it.cpu())
                    acc[name]["cn_dev"].append(dv.cpu())
                    acc[name]["ca_ca"].append(bl.cpu())
                    acc[name]["bend"].append(bn.cpu())
                    acc[name]["disp"].append(rms_disp(ca, src_ca, mf).reshape(1).cpu())
                    acc[name]["gap"].append(gap(ca).cpu())
            logger.info(
                f"batch {bidx}: cart intact {float(acc['cartesian']['intact'][-1].mean()):.3f}  "
                f"internal {float(acc['internal_matched']['intact'][-1].mean()):.3f}"
            )

        if not acc["source"]["intact"]:
            logger.error("no batches measured")
            return 1

        ones = lambda t: torch.ones_like(t, dtype=torch.bool)  # noqa: E731
        out = {
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "target_sigma_c_nm": sigma,
            "n_examples": int(torch.cat(acc["source"]["intact"]).numel()),
            "internal_bend_clamp_frac_mean": sum(clamp_fracs) / max(1, len(clamp_fracs)),
            "arms": {},
        }
        for a in ARMS:
            cat = {k: torch.cat(v) for k, v in acc[a].items()}
            out["arms"][a] = {
                "chain_intact_frac": float(cat["intact"].mean()),
                "rms_ca_displacement_nm": float(cat["disp"].mean()),
                "cn_dev_nm": masked_stats(cat["cn_dev"], ones(cat["cn_dev"])),
                "ca_ca_nm": masked_stats(cat["ca_ca"], ones(cat["ca_ca"])),
                "bend_deg": masked_stats(cat["bend"], ones(cat["bend"])),
                "terminal_gap_nm": float(cat["gap"].mean()),
            }

        # The match is a PRECONDITION for reading the comparison, so it is checked, not assumed.
        dc = out["arms"]["cartesian"]["rms_ca_displacement_nm"]
        di = out["arms"]["internal_matched"]["rms_ca_displacement_nm"]
        rel = abs(di - dc) / max(dc, 1e-9)
        out["size_match_rel_error"] = rel
        out["size_match_ok"] = bool(rel < 0.05)

        out["sigma_c_nm"] = sigma
        sweep_out.append(out)
    out = {"sweep": sweep_out} if len(sweep_out) > 1 else sweep_out[0]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for point in sweep_out:
        logger.info(f"--- sigma_c = {point['sigma_c_nm']:.4f} nm")
        for a in ARMS:
            v = point["arms"][a]
            logger.info(
                f"  {a:18s} intact {v['chain_intact_frac']:.4f}  "
                f"disp {v['rms_ca_displacement_nm']:.4f} nm  "
                f"CA-CA sd {v['ca_ca_nm']['sd']:.5f}  bend min {v['bend_deg']['min']:.1f}  "
                f"cn med {v['cn_dev_nm']['median']:.4f}  gap {v['terminal_gap_nm']:.3f}"
            )
        if not point["size_match_ok"]:
            logger.warning(
                f"  SIZE MATCH FAILED (rel err {point['size_match_rel_error']:.1%}) -- the arms "
                "are not comparable at this sigma; do not read their intact columns together."
            )
    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
