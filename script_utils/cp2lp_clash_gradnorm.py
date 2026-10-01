"""Weighted per-term GENERATOR GRADIENT NORMS, with the clash term scored both ways.

WHY THIS EXISTS. The clash-normalisation arm was justified from a table of weighted scalar
loss shares (docs/README_CP2LP.md: clash 0.0002, 0.06% of the objective). A scalar share is
NOT a gradient share -- a term can be numerically tiny and still steer the step if its
gradient is large, and a term can be numerically large and nearly flat. The decision to
change the normalisation therefore has to be checked in gradient units, which is what this
measures.

WHAT MAKES IT PAIRED, AND WHY THAT MATTERS. Both normalisations are scored on the SAME
decoded output from the SAME editor forward, in one process. `inter_norm` changes only how
a fixed set of pairwise overlap depths is reduced to a scalar, so nothing upstream of the
reduction differs: the two numbers come from one structure, one noise draw, one batch. A
before/after taken from two separate runs would be confounded by resampling alone -- the
batch is 4 samples and the doc already records a 1.64-1.82 spread in terminal gap from
window choice. Here the only difference is the denominator.

WHAT IS REPORTED, per term, weighted by the weight training actually applies:

  value       the scalar, for continuity with the loss-share table
  grad_norm   the global L2 norm of d(weight * term)/d(trunk params)
  frac_total  grad_norm as a fraction of the norm of the SUMMED gradient

`frac_total` can exceed 1 and the per-term fractions need not sum to 1: gradient norms are
not additive, and two terms that disagree partially cancel. It is a magnitude comparison,
not a decomposition. Cosine against the summed gradient is reported beside it because that
is what says whether a term's pull survives the sum or is cancelled by the others.

The adversarial term is included because it is the thing everything else competes with --
95% of the scalar objective at g=10300 -- and the question "does clash now have a voice" is
meaningless except against it.

Run (GPU):
    .venv/bin/python script_utils/cp2lp_clash_gradnorm.py \
        --config example/training_cp2lp_onepass_internal_clashnorm \
        --ckpt <store>/cp2lp_onepass_internal_clashnorm/checkpoints/last.ckpt \
        --batches 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="example/training_cp2lp_onepass_internal_clashnorm")
    p.add_argument("--ckpt", default=None, help="Full Lightning checkpoint to measure at.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--batches",
        type=int,
        default=4,
        help="Ordinary training batches. A few: this is a measurement of where the gradient "
        "is, not a quality estimate, and each batch costs a full generator forward plus one "
        "backward per term.",
    )
    p.add_argument("--out", default=None, help="Write the per-batch rows here as JSON.")
    return p.parse_args()


def grad_norm(scalar: torch.Tensor, params: list[torch.Tensor]) -> tuple[float, list | None]:
    """Global L2 norm of d(scalar)/d(params), and the raw grads for cosine work.

    `allow_unused=True` distinguishes a term that does not reach the trunk at all (None
    everywhere -> norm 0.0, reported as such) from one that reaches it with a small value.
    Those are different findings and collapsing them would hide a miswiring.
    """
    if not torch.is_tensor(scalar) or not scalar.requires_grad:
        return 0.0, None
    g = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    flat = [x.reshape(-1) for x in g if x is not None]
    if not flat:
        return 0.0, None
    cat = torch.cat(flat)
    return float(cat.norm()), g


def cosine(a: list | None, b: list | None) -> float:
    """Cosine between two per-parameter gradient lists, aligned by position."""
    if a is None or b is None:
        return float("nan")
    av, bv = [], []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        av.append(x.reshape(-1))
        bv.append(y.reshape(-1))
    if not av:
        return float("nan")
    ac, bc = torch.cat(av), torch.cat(bv)
    denom = ac.norm() * bc.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((ac @ bc) / denom)


def main() -> int:  # noqa: C901 -- one linear measurement; splitting it hides the order
    args = parse_args()
    load_dotenv(REPO / ".env")

    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        cfg = compose(config_name=args.config)

    print("=" * 78)
    print("WEIGHTED PER-TERM GENERATOR GRADIENT NORMS")
    print("=" * 78)
    print(f"config   : {args.config}")
    print(f"run_name : {cfg.run_name}")
    print(f"clash_inter_norm (config): {cfg.cp2lp.losses.get('clash_inter_norm', 'all_pairs')}")

    # The measurement must not be perturbed by the things a smoke turns on.
    cfg.dataset.datamodule.num_workers = 2
    cfg.dataset.unified.datamodule.num_workers = 2
    cfg.cp2lp.max_g_updates = 0
    cfg.cp2lp.eval_checkpoint_g_updates = []
    cfg.cp2lp.feature_scales.min_examples = 1
    if cfg.get("val_generation") is not None:
        cfg.val_generation.enabled = False
    if cfg.get("rollout_finetune") is not None:
        cfg.rollout_finetune.enabled = False

    from proteinfoundation.cp2lp.data import slice_batch, split_roles
    from proteinfoundation.cp2lp.discriminator import hinge_g_loss
    from proteinfoundation.cp2lp.losses import clash_loss
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    model = CP2LPGenerator(cfg)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        logger.error("--ckpt is required: this measures a specific training state.")
        return 1
    if not os.path.exists(ckpt_path):
        logger.error(f"checkpoint not found: {ckpt_path}")
        return 1
    # Loaded WITHOUT the pretrain splice and WITHOUT `init_editor_noise_adapter`. This is a
    # mid-run state whose editor heads and noise adapter are already trained; re-initialising
    # the adapter here (as train.py does on a FRESH fine-tune) would measure gradients at
    # weights no run ever had.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"ckpt     : {ckpt_path}")
    print(f"  global_step {ckpt.get('global_step')}   g_updates {int(model.g_updates)}")
    print(f"  missing {len(missing)}   unexpected {len(unexpected)}")
    if missing:
        print(f"  first missing: {list(missing)[:6]}")
    del ckpt

    device = torch.device(args.device)
    model.to(device)
    model.train()

    params = [p for p in model.nn.parameters() if p.requires_grad]
    print(f"trunk params with grad: {len(params)}")

    w = {
        "contact": model.w_contact,
        "sequence": model.w_sequence,
        "geometry": model.w_geometry,
        "clash": model.w_clash,
        "open": model.w_open,
        "adversarial": model.adv_weight,
    }
    print("weights  : " + "  ".join(f"{k}={v}" for k, v in w.items()))
    print(f"clash_threshold_nm: {model.clash_threshold_nm}")

    _, datamodule = load_data_module(cfg, is_cluster_run=True)
    datamodule.setup("fit")
    loader = iter(datamodule.train_dataloader())

    rows = []
    for bi in range(int(args.batches)):
        # Seeded per batch so the generator's noise draw is reproducible and the whole
        # measurement can be repeated exactly.
        torch.manual_seed(1000 + bi)
        try:
            batch = next(loader)
        except StopIteration:
            print(f"loader exhausted after {bi} batches")
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probe = add_clean_samples(dict(batch), cfg.product_flowmatcher, model.autoencoder)
        cyclic, linear = split_roles(probe)
        bs = probe["mask"].shape[0]
        if int(cyclic.sum()) == 0:
            print(f"batch {bi}: no cyclic rows, skipped")
            continue
        cp_batch = slice_batch(probe, cyclic, bs)

        gen_batch, decoded, _extras = model._generate(cp_batch)
        mask = gen_batch["mask"].bool()
        n_res = mask.float().sum(dim=1)

        # The five shaping terms, UNWEIGHTED, from the module's own attribution path -- the
        # same function the training path mirrors, so these are the terms training uses.
        terms = model._generator_loss_terms(gen_batch, decoded)

        target = dict(
            target_atom37=gen_batch["x_target"],
            target_atom_mask=gen_batch["target_mask"],
            target_mask=gen_batch["seq_target_mask"].bool(),
        )
        # THE PAIRED PART: one decoded structure, two reductions.
        clash_both = {}
        clash_metrics = {}
        for norm in ("all_pairs", "peptide_length"):
            c, m = clash_loss(
                atom37=decoded["coors_nm"],
                atom_mask=decoded["atom_mask_eff"],
                pep_mask=mask,
                threshold_nm=model.clash_threshold_nm,
                inter_norm=norm,
                **target,
            )
            clash_both[norm] = c
            clash_metrics[norm] = {k: float(v) for k, v in m.items()}

        scored_terms: dict[str, torch.Tensor] = {
            "contact": w["contact"] * terms["contact"],
            "sequence": w["sequence"] * terms["sequence"],
            "geometry": w["geometry"] * terms["geometry"],
            "open": w["open"] * terms["open"],
            "clash@all_pairs": w["clash"] * clash_both["all_pairs"],
            "clash@peptide_length": w["clash"] * clash_both["peptide_length"],
        }

        # The adversarial term, formed exactly as the training step forms it: D frozen for
        # the forward (so gradient reaches the generator's inputs but not D's weights), and
        # through the same corruption hook, which is a no-op for the addon-free arms.
        if model.adv_enabled:
            fake_feats = model._featurize(decoded, gen_batch, mask)
            with model._frozen_discriminator():
                s, level = model._maybe_corrupt(fake_feats)
                adv = hinge_g_loss(model.discriminator(s, noise_level=level))
            scored_terms["adversarial"] = w["adversarial"] * adv

        # The summed objective in BOTH worlds, for frac_total and the cosines.
        totals = {}
        for norm in ("all_pairs", "peptide_length"):
            t = torch.zeros((), device=device)
            for name, val in scored_terms.items():
                if name.startswith("clash@") and not name.endswith(norm):
                    continue
                t = t + val
            totals[norm] = t

        row: dict = {
            "batch": bi,
            "n_rows": int(mask.shape[0]),
            "lengths": [int(x) for x in n_res.tolist()],
            "terms": {},
            "clash_metrics": clash_metrics,
        }

        tot_g = {}
        for norm in ("all_pairs", "peptide_length"):
            nrm, g = grad_norm(totals[norm], params)
            tot_g[norm] = (nrm, g)
            row[f"total_grad_norm@{norm}"] = nrm
            row[f"total_value@{norm}"] = float(totals[norm])

        for name, val in scored_terms.items():
            nrm, g = grad_norm(val, params)
            # Compare each term against the total of the world it belongs to.
            world = "peptide_length" if name.endswith("peptide_length") else "all_pairs"
            ref_n, ref_g = tot_g[world]
            row["terms"][name] = {
                "value": float(val),
                "grad_norm": nrm,
                "frac_total": (nrm / ref_n) if ref_n > 0 else float("nan"),
                "cos_total": cosine(g, ref_g),
            }

        rows.append(row)

        print("-" * 78)
        print(f"batch {bi}: {int(mask.shape[0])} rows, lengths {row['lengths']}")
        print(f"  {'term':22s} {'weighted':>11s} {'grad_norm':>11s} {'frac_tot':>9s} {'cos_tot':>8s}")
        for name, d in row["terms"].items():
            print(
                f"  {name:22s} {d['value']:11.3e} {d['grad_norm']:11.3e} "
                f"{d['frac_total']:9.3f} {d['cos_total']:8.3f}"
            )
        for norm in ("all_pairs", "peptide_length"):
            print(f"  TOTAL@{norm:18s} {row[f'total_value@{norm}']:11.3e} "
                  f"{row[f'total_grad_norm@{norm}']:11.3e}")
        cm = clash_metrics["all_pairs"]
        print(f"  overlap: sum {cm.get('clash_inter_sum_nm', float('nan')):.4f} nm   "
              f"min_inter {cm.get('clash_min_inter_nm', float('nan')):.4f} nm   "
              f"max_depth {cm.get('clash_inter_max_depth_nm', float('nan')):.4f} nm   "
              f"n_viol {cm.get('clash_inter_n_viol', float('nan')):.2f}")

        model.zero_grad(set_to_none=True)
        del gen_batch, decoded, terms, scored_terms, totals, clash_both

    if not rows:
        logger.error("no batches measured")
        return 1

    # ---------------------------------------------------------------- summary
    print("=" * 78)
    print(f"SUMMARY over {len(rows)} batches (mean [min, max])")
    names = list(rows[0]["terms"].keys())
    print(f"  {'term':22s} {'weighted value':>22s} {'grad_norm':>24s}")
    agg: dict[str, dict] = {}
    for name in names:
        vals = [r["terms"][name]["value"] for r in rows]
        nrms = [r["terms"][name]["grad_norm"] for r in rows]
        fr = [r["terms"][name]["frac_total"] for r in rows]
        cs = [r["terms"][name]["cos_total"] for r in rows]
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        agg[name] = {
            "value_mean": mean(vals), "value_min": min(vals), "value_max": max(vals),
            "grad_mean": mean(nrms), "grad_min": min(nrms), "grad_max": max(nrms),
            "frac_mean": mean(fr), "cos_mean": mean(cs),
        }
        print(f"  {name:22s} {mean(vals):10.3e} [{min(vals):8.2e},{max(vals):8.2e}] "
              f"{mean(nrms):10.3e} [{min(nrms):8.2e},{max(nrms):8.2e}]")

    old, new = agg["clash@all_pairs"], agg["clash@peptide_length"]
    print()
    print("THE NUMBER THIS SCRIPT EXISTS FOR:")
    for label, key in (("scalar value", "value_mean"), ("gradient norm", "grad_mean")):
        o, n = old[key], new[key]
        ratio = (n / o) if o > 0 else float("inf")
        print(f"  clash {label:14s}: {o:.4e} -> {n:.4e}   ({ratio:.1f}x)")
    if "adversarial" in agg:
        a = agg["adversarial"]
        print(f"  adversarial gradient norm : {a['grad_mean']:.4e}")
        for label, d in (("before", old), ("after", new)):
            denom = a["grad_mean"]
            share = (d["grad_mean"] / denom) if denom > 0 else float("nan")
            print(f"    clash/adversarial {label:6s}: {share:.5f}")
    print()
    print("Read `cos_total` beside the norms: a term with a large norm and a cosine near 0")
    print("against the summed gradient is being cancelled, not heard.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"config": args.config, "ckpt": args.ckpt,
                       "weights": {k: float(v) for k, v in w.items()},
                       "rows": rows, "summary": agg}, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
