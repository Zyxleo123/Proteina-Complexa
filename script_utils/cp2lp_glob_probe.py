"""What is `d_acc` actually measuring? Test the discriminator's openness columns.

THE QUESTION. Every CP->LP arm pins `d_acc` near 0.98 and nothing moves it. The clash-norm
arm fixed a real steric defect (interfacial overlap 0.35-0.88x the control at matched edit
size) and `d_acc` did not budge: 0.9872 vs 0.9879. The torsion backbone arm reaches
near-perfect chain geometry (C-N 0.9978, intra 0.9994, tau 0.9998, angles 0.9899) and still
scores 0.9840 -- no better than the ca_internal population at ~1.6 broken bonds per
peptide. Both of the two tells docs/README_CP2LP.md names have now been manipulated, and
neither moved the number. So `d_acc` is reading something else.

THE SUSPECT, and it is in the feature list rather than in the data. `build_features` puts
the terminal gap into the discriminator's global block as an explicit input column:

    glob = [terminal_gap_nm, radius_of_gyration_nm, contact_count, mean_min_heavy_nm]

Real bound LPs sit at ~2.14 nm terminal separation; the arms reach 1.04-1.25 nm from a
source CP's 0.349. Length matching is exact (`d_match_frac` 1.0), so that is a within-pair
comparison, not a length artefact. Radius of gyration is the same deficit measured a second
way: a half-opened ring is more compact than a real extended LP. If D leans on those two
columns then `d_acc` is close to a restatement of the unsolved task -- "did you open the
ring all the way" -- rather than a judgment about realism, and no loss-term experiment can
move it.

WHY THE EXISTING INSTRUMENT CANNOT ANSWER THIS. `d_shortcut_probe_acc` reads
`feats["glob"][:, 2:]` (`discriminator.probe_forward`) -- it excludes exactly these two
columns, and it masks the peptide besides. The README already records the shape of the hole:
"There is no probe for this; `shortcut_probe` masks the peptide, so it is blind to a tell
that lives *in* the peptide."

WHAT THIS PROBE REPORTS. Three measurements, in decreasing order of how much they settle.

  1. DEPENDENCE (the counterfactual, and the verdict). Re-score the SAME length-matched
     real/fake rows with the trained D, holding `glob[:, (0, 1)]` at one pooled median --
     see `glob_ablated_pair` for why the fill must be shared across classes. Reported as
     `d_acc_openblind`, plus the single-column variants. A collapse toward 0.5 proves
     dependence. A null does NOT prove independence: `glob` enters only the head while the
     trunk still sees `geom`/`iface`, from which an openness proxy is partly recoverable, so
     this bounds dependence from below.
  2. SUFFICIENCY. The dumped columns go to `cp2lp_glob_report.py`, which fits logistic
     regressions on the gap alone / gap+Rg / all four, cross-validated, and compares their
     accuracy with D's own on the identical rows.
  3. USAGE, correlationally. Per-example D logits are dumped beside the columns, so the
     report can take Spearman(terminal_gap, d_logit). Sufficiency says a column COULD carry
     the decision; this says D's decision moves with it.

WHAT IS DELIBERATELY NOT RE-DERIVED. The real/fake assembly is the module's own: reals are
decoded and featurized through `_decode`/`_featurize` and pushed to the same reservoir,
fakes come from `_generate`, and the pairing is `reservoir.draw_matched`. Re-implementing
that here would risk measuring a distribution D never saw, which is the one error that would
invalidate the whole probe. The anchor path (`_real_lp_backbone_anchor`) is never called --
it mutates the batch, and in training it deliberately runs after the real-pool snapshot.

No optimizer, no Trainer, no gradient: this is a measurement pass over a frozen checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True, help="training run dir (holds exp_config json)")
    p.add_argument("--ckpt", required=True, help="NON-EMA checkpoint: it carries the trained D")
    p.add_argument("--cp-parquet", required=True, help="metadata with BOTH cyclic and linear rows")
    p.add_argument("--out", required=True, help="output JSON summary path")
    p.add_argument("--rows-out", required=True, help="per-example JSONL for the CPU report")
    p.add_argument("--n-batches", type=int, default=200)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "overrides the run's own batch_size (6 in every arm). At 25%% linear rows a "
            "batch of 6 carries ~1.5 reals, so many batches would be single-role and "
            "skipped; 16 carries ~4 of each. Changes throughput only -- D is scored on "
            "length-matched pairs drawn from the reservoir, not on whole batches."
        ),
    )
    p.add_argument("--max-rows", type=int, default=100000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    cfg_files = sorted(run_dir.glob("*exp_config*.json")) or sorted(
        run_dir.glob("**/*exp_config*.json")
    )
    if not cfg_files:
        logger.error(f"no exp_config json under {run_dir}")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proteinfoundation.cp2lp.data import slice_batch, split_roles
    from proteinfoundation.cp2lp.discriminator import (
        GLOB_COLUMNS,
        GLOB_INTERFACE_COLS,
        GLOB_OPENNESS_COLS,
        discriminator_accuracy,
        glob_ablated_pair,
    )
    from proteinfoundation.cp2lp.conditioning import SRC_CP_KEY
    from proteinfoundation.cp2lp.geometry import residue_lengths, terminal_gap_nm
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    cfg_exp = OmegaConf.create(json.load(cfg_files[0].open()))
    OmegaConf.set_struct(cfg_exp, False)
    dm = cfg_exp.dataset.unified.datamodule
    dm.metadata_file = str(args.cp_parquet)
    dm.val_metadata_file = str(args.cp_parquet)
    # MUST be cleared, and it is not cosmetic: every arm ships
    # `val_filters: ["dataset_source == 'cpsea'"]`, which would drop every PepBench row --
    # i.e. the entire real-LP class -- and the probe would then report "no scorable step"
    # on a parquet that does contain reals.
    dm.val_filters = None
    dm.batch_size = int(args.batch_size)
    # Same reason as cp2lp_generate.py and cp2lp_ca_spacing_probe.py: the inherited
    # three-way training mix names sources a single-source parquet does not have, and
    # setup() raises before the model loads.
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

    # THE ONE LOAD FAILURE THAT WOULD FAKE THE ANSWER. `strict=False` is needed for the
    # frozen AE/flow buffers, but a checkpoint without discriminator weights would leave a
    # RANDOMLY INITIALISED D here -- whose accuracy is ~0.5 whatever the features say, so
    # the ablation would read as "no dependence" from a model that never learned anything.
    # EMA checkpoints are the live risk: they are ~0.6 GB smaller and may track the
    # generator only. Fail loudly rather than measure noise.
    d_missing = [k for k in missing if k.startswith("discriminator.")]
    if d_missing:
        logger.error(
            f"{len(d_missing)} discriminator.* keys MISSING from {Path(args.ckpt).name} "
            f"(first: {d_missing[:3]}). This probe would score an untrained D. Pass the "
            "NON-EMA checkpoint."
        )
        return 1
    g_updates = int(model.g_updates) if hasattr(model, "g_updates") else None
    logger.info(
        f"loaded {Path(args.ckpt).name} global_step={ck.get('global_step')} "
        f"g_updates={g_updates} missing={len(missing)} unexpected={len(unexpected)}"
    )
    del ck

    if not getattr(model, "adv_enabled", False):
        logger.error("this run has no discriminator (adv_enabled=False); nothing to probe.")
        return 1

    device = torch.device(args.device)
    model.to(device).eval()

    # The VALIDATION loader, as in cp2lp_ca_spacing_probe.py. The training loader would
    # bring the weighted source sampler with it (`source_fractions`, cleared above), and
    # shuffling is unnecessary here because cp2lp_glob_mix.py writes the mixed parquet
    # already shuffled -- so sequential reads interleave the two roles anyway.
    loader = datamodule.val_dataloader()
    rows: list[dict] = []
    n_steps = 0
    n_unmatched = 0
    # THE POPULATION AUDIT, not a nicety. If the editor returns a near-identity edit here,
    # the generated class is "a cyclic peptide with the ring bond deleted" rather than an
    # attempted LP, and D's job is easy for a reason that has nothing to do with which
    # features it reads. The arm logs terminal gap 1.044 nm in training, so a gap near the
    # source's 0.349 means this probe is measuring a different generator than the one the
    # scorecard describes -- and the ablation would then be a true statement about the
    # wrong population.
    edit_rms: list[float] = []

    # The interventions. `()` is the baseline -- it runs through the SAME `glob_ablated_pair`
    # path with an empty column set, so the baseline cannot differ from the ablations by
    # anything other than the held columns.
    #
    # THE INTERFACE COLUMNS ARE THE POINT OF THIS SECOND PASS. The first run held (0, 1) and
    # measured an exact null, 0.9817 -> 0.9817, which left columns 2 and 3 as the only places
    # a glob tell can be. `contactblind` is the hypothesis with a mechanism attached: see
    # GLOB_INTERFACE_COLS. `allblind` bounds the whole block at once, so a null there sends
    # the search to the trunk (geom/iface) and ends the glob line of enquiry.
    ABLATIONS = {
        "": (),
        "openblind": GLOB_OPENNESS_COLS,
        "gapblind": (0,),
        "rgblind": (1,),
        "ifaceblind": GLOB_INTERFACE_COLS,
        "contactblind": (2,),
        "closeblind": (3,),
        "allblind": tuple(range(len(GLOB_COLUMNS))),
    }
    # DERIVED, not restated. The accumulator used to carry a hand-written key list that had
    # to match `ABLATIONS` exactly; adding an intervention there and forgetting it here
    # raises a KeyError deep in the batch loop, after the model is loaded and the data is
    # streaming. Deriving it means a new intervention cannot be half-added.
    logits_acc: dict[str, list[torch.Tensor]] = {
        f"{role}{f'_{tag}' if tag else ''}": []
        for tag in ABLATIONS
        for role in ("real", "fake")
    }

    for bidx, batch in enumerate(loader):
        if bidx >= args.n_batches or len(rows) >= args.max_rows:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            batch = add_clean_samples(
                batch,
                cfg_exp.product_flowmatcher,
                getattr(model, "autoencoder", None),
                local_latent_target=cfg_exp.get("local_latent_target", "sample"),
                detach_latent_target_for_flow=bool(
                    cfg_exp.get("detach_latent_target_for_flow", False)
                ),
            )
            bs = batch["mask"].shape[0]
            cyclic_idx, linear_idx = split_roles(batch)
            if not (cyclic_idx.any() and linear_idx.any()):
                continue

            # ---- reals, exactly as `training_step` builds them, into the same reservoir
            lp_batch = slice_batch(batch, linear_idx, bs)
            lp_mask = lp_batch["mask"].bool()
            lp_clean = {
                dmode: lp_batch["x_1"][dmode].detach()
                * lp_mask[..., None].to(lp_batch["x_1"][dmode].dtype)
                for dmode in model._data_modes
            }
            real_feats_live = model._featurize(model._decode(lp_clean, lp_mask), lp_batch, lp_mask)
            pool = model._real_pool_index(lp_batch)
            if pool is not None:
                keep = pool.nonzero(as_tuple=True)[0]
                if not keep.numel():
                    continue
                model.reservoir.push_batch(
                    {k: v[keep] for k, v in real_feats_live.items()},
                    residue_lengths(lp_mask[keep]),
                )
            else:
                model.reservoir.push_batch(real_feats_live, residue_lengths(lp_mask))

            # ---- fakes, through the arm's own generator
            cp_batch = slice_batch(batch, cyclic_idx, bs)
            n_cp = cp_batch["mask"].shape[0]
            n_keep = min(n_cp, model.cp2lp_max_cp)
            cp_batch = slice_batch(
                cp_batch, torch.arange(n_keep, device=batch["mask"].device), n_cp
            )
            if model.cp2lp_samples_per_cp > 1:
                idx = torch.arange(n_keep, device=batch["mask"].device).repeat_interleave(
                    model.cp2lp_samples_per_cp
                )
                cp_batch = slice_batch(cp_batch, idx, n_keep)
            gen_batch, decoded, _extras = model._generate(cp_batch)
            fake_feats = model._featurize(decoded, gen_batch, gen_batch["mask"].bool())

            # ---- the pairing D is scored on
            fake_lengths = residue_lengths(fake_feats["mask"])
            real_feats, matched = model.reservoir.draw_matched(
                fake_lengths.cpu(), device, max_length_slack=model.reservoir_slack
            )
            if real_feats is None:
                n_unmatched += 1
                continue
            keep = matched.to(device).nonzero(as_tuple=True)[0]
            fake_det = {k: v[keep] for k, v in fake_feats.items()}

            # The source CP's own terminal gap, through the same decode, so "did it open the
            # ring" is answerable per row instead of against a remembered constant.
            gen_mask = gen_batch["mask"].bool()
            src_gap_all = terminal_gap_nm(
                model._decode(gen_batch[SRC_CP_KEY], gen_mask)["coors_nm"], gen_mask
            )
            src_gap = src_gap_all[keep].cpu()
            if "delta_ca_rms_nm" in _extras.get("metrics", {}):
                edit_rms.append(float(_extras["metrics"]["delta_ca_rms_nm"]))

            # Diffusion-GAN's controller corrupts features in arm B only; the clean logit is
            # what the other arms' `d_acc` means, so every arm is scored clean here. See
            # `d_acc_clean` in module.py for the same choice on the training side.
            zero = None
            if model.discriminator.noise_level_dim > 0:
                zero = torch.zeros(fake_det["mask"].shape[0], device=device)

            scored: dict[str, torch.Tensor] = {}
            for tag, cols in ABLATIONS.items():
                r_ab, f_ab, _fill = glob_ablated_pair(real_feats, fake_det, cols)
                z_r = None if zero is None else torch.zeros(r_ab["mask"].shape[0], device=device)
                r_log = model.discriminator(r_ab, noise_level=z_r)
                f_log = model.discriminator(f_ab, noise_level=zero)
                suffix = f"_{tag}" if tag else ""
                scored[f"real{suffix}"] = r_log
                scored[f"fake{suffix}"] = f_log
                logits_acc[f"real{suffix}"].append(r_log.cpu())
                logits_acc[f"fake{suffix}"].append(f_log.cpu())

            # ---- per-example rows: the glob columns, the length, and D's own verdict
            for label, feats, tag in (("real", real_feats, "real"), ("fake", fake_det, "fake")):
                glob = feats["glob"].cpu()
                lens = residue_lengths(feats["mask"]).cpu()
                for i in range(glob.shape[0]):
                    row = {
                        "label": 1 if label == "real" else 0,
                        "batch": bidx,
                        "peptide_length": int(lens[i]),
                    }
                    row.update({name: float(glob[i, j]) for j, name in enumerate(GLOB_COLUMNS)})
                    for ab in ABLATIONS:
                        suffix = f"_{ab}" if ab else ""
                        row[f"d_logit{suffix}"] = float(scored[f"{tag}{suffix}"][i])
                    # Only the generated rows have a source; reals are left absent rather
                    # than NaN-filled, because NaN is truthy and a naive filter downstream
                    # would silently count them.
                    if label == "fake":
                        row["src_terminal_gap_nm"] = float(src_gap[i])
                        row["terminal_gap_delta_nm"] = float(
                            row["terminal_gap_nm"] - src_gap[i]
                        )
                    rows.append(row)
            n_steps += 1

    if not n_steps:
        logger.error(
            "no scorable step: every batch was single-role or the reservoir never matched. "
            f"unmatched={n_unmatched}. Check the parquet has BOTH cyclic and linear rows."
        )
        return 1

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "ckpt": str(args.ckpt),
        "g_updates": g_updates,
        "n_steps": n_steps,
        "n_rows": len(rows),
        "n_unmatched_steps": n_unmatched,
        "glob_columns": list(GLOB_COLUMNS),
        "openness_cols": list(GLOB_OPENNESS_COLS),
        "interface_cols": list(GLOB_INTERFACE_COLS),
    }
    for ab in ABLATIONS:
        suffix = f"_{ab}" if ab else ""
        r = torch.cat(logits_acc[f"real{suffix}"])
        f = torch.cat(logits_acc[f"fake{suffix}"])
        summary[f"d_acc{suffix or '_baseline'}"] = discriminator_accuracy(r, f)
        summary[f"d_real_logit{suffix or '_baseline'}"] = float(r.mean())
        summary[f"d_fake_logit{suffix or '_baseline'}"] = float(f.mean())

    fake_gaps = [r["terminal_gap_nm"] for r in rows if r["label"] == 0]
    src_gaps = [r["src_terminal_gap_nm"] for r in rows if r["label"] == 0]
    real_gaps = [r["terminal_gap_nm"] for r in rows if r["label"] == 1]
    summary["gap_generated_median"] = float(np.median(fake_gaps)) if fake_gaps else None
    summary["gap_source_cp_median"] = float(np.median(src_gaps)) if src_gaps else None
    summary["gap_real_lp_median"] = float(np.median(real_gaps)) if real_gaps else None
    summary["gap_travel_median"] = (
        float(np.median(np.asarray(fake_gaps) - np.asarray(src_gaps))) if fake_gaps else None
    )
    summary["delta_ca_rms_nm_mean"] = float(np.mean(edit_rms)) if edit_rms else None
    # A travel under ~0.05 nm is the identity map the export gate is known to select for.
    summary["population_warning"] = (
        None
        if not fake_gaps or float(np.median(np.asarray(fake_gaps) - np.asarray(src_gaps))) > 0.05
        else (
            "NEAR-IDENTITY EDIT: the generated class is barely distinguishable from the "
            "source CP, so every accuracy here describes real-LP-vs-cyclic-peptide and NOT "
            "the arm the scorecard describes. Resolve before reading the ablation."
        )
    )

    base = float(summary["d_acc_baseline"])  # type: ignore[arg-type]
    # Every intervention gets a delta, not just the openness pair. Ranking them is the whole
    # output: the question is no longer "does D read openness" (measured, it does not) but
    # "which column does it read", and that is answered by which delta is largest.
    DEPENDENCE_DROP = 0.05
    for ab in ABLATIONS:
        if not ab:
            continue
        summary[f"delta_{ab}"] = float(summary[f"d_acc_{ab}"]) - base  # type: ignore[arg-type]

    ranked = sorted(
        ((ab, float(summary[f"delta_{ab}"])) for ab in ABLATIONS if ab),  # type: ignore[arg-type]
        key=lambda kv: kv[1],
    )
    summary["ranked_deltas"] = [{"ablation": ab, "delta": d} for ab, d in ranked]
    worst, worst_delta = ranked[0]
    summary["largest_dependence"] = worst
    # The verdict names the column, and says explicitly what a null over the WHOLE block
    # means -- otherwise "no dependence" reads as "D is not using the interface", when what
    # it bounds is only the head. `allblind` is the one that closes the glob question.
    all_delta = float(summary["delta_allblind"])  # type: ignore[arg-type]
    if worst_delta <= -DEPENDENCE_DROP:
        summary["verdict"] = (
            f"DEPENDS on glob: largest drop is `{worst}` ({worst_delta:+.4f}), columns "
            f"{[GLOB_COLUMNS[c] for c in ABLATIONS[worst]]}"
        )
    elif all_delta > -DEPENDENCE_DROP:
        summary["verdict"] = (
            f"NO glob dependence at the head: holding ALL {len(GLOB_COLUMNS)} columns moved "
            f"d_acc by {all_delta:+.4f}. D's tell is in the TRUNK (geom/iface), not in the "
            "global block -- stop probing glob. LOWER BOUND: the trunk may re-derive these."
        )
    else:
        summary["verdict"] = (
            f"diffuse glob dependence: no single intervention reaches {DEPENDENCE_DROP}, but "
            f"allblind moves {all_delta:+.4f}, so the columns are mutually redundant."
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    with Path(args.rows_out).open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    logger.info(f"d_acc baseline={base:.4f}")
    for ab, d in ranked:  # most negative first: the leading candidate reads at the top
        logger.info(
            f"  {ab:14s} d_acc={float(summary[f'd_acc_{ab}']):.4f}  delta={d:+.4f}  "
            f"cols={[GLOB_COLUMNS[c] for c in ABLATIONS[ab]]}"
        )
    logger.info(f"-> {summary['verdict']}")
    logger.info(f"wrote {args.out} and {len(rows)} rows to {args.rows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
