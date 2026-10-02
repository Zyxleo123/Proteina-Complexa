"""A FRESH, HELD-OUT discriminator against one arm's generated peptides.

WHY THE TRAINING `d_acc` IS NOT ENOUGH. The online discriminator is a moving opponent
trained jointly with the generator: its accuracy can fall because the generator got more
realistic, or because D fell behind, got over-regularised, or found the current objective's
gradient uninformative. The three are indistinguishable from the training curve. A
discriminator trained FROM SCRATCH, after the fact, on complexes neither it nor the
generator's own D was fit on, answers the question the curve cannot: is this arm's output
actually harder to tell from real bound linear peptides?

FOUR PROTOCOL RULES, each because its absence gives a confidently wrong number.

**The real class is the AE ROUND TRIP of a real LP, never its crystal coordinates.** This
mirrors what the training path feeds its own D, and the comment there says why: otherwise
"was this decoded?" separates the classes perfectly and the number measures the
autoencoder, not the arm. This is the single easiest way to get a meaningless 1.00 here.

**Every generation attempt counts, and the export gate is not applied.** MEASURED
(docs/README_CP2LP.md, "The export gate selects FOR non-edits"): the gate's pass rate was
438/512 in the control arm against 19/512 in the gan arm, and the accepted subset moved
LESS than its population (1.84 A vs 2.76 A). Scoring accepted exports would compare two
differently-gated populations and would flatter whichever arm rejects more.

**Grouped split on receptor family.** Folds are split on the family key, never on rows.
These sets are family-redundant and a row split puts near-duplicates on both sides, which
drives every accuracy here toward 1.0 for a reason unrelated to the arm.

**A positive control gates the result.** Real LPs against the SOURCE cyclic peptides is a
comparison that MUST come back near 1.0 -- they are a different topology with a closed
ring. If the harness cannot separate those, a near-chance score on the generated peptides
says nothing, and the report says NOT MEASURED rather than "indistinguishable".

Accuracy is reported on balanced classes so 0.5 is chance, with AUC beside it because
accuracy at a 0.5 threshold hides a confident-but-miscalibrated D. Several fresh seeds,
because a single random init of a 3-layer transformer on a few hundred complexes is noisy
and one draw is not a measurement.

Run (GPU):
    .venv/bin/python script_utils/cp2lp_fresh_d.py \\
        --config example/training_cp2lp_onepass_internal_clashnorm \\
        --ckpt <store>/cp2lp_onepass_internal_clashnorm/checkpoints/last.ckpt \\
        --batches 32 --seeds 3 --out fresh_d.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from loguru import logger

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--batches", type=int, default=32,
        help="Eval batches to collect. Each contributes a few generated peptides and a few "
        "real LPs; the fresh D needs a few hundred of each to mean anything.",
    )
    p.add_argument("--seeds", type=int, default=3, help="Fresh D initialisations.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--d-lr", type=float, default=3.0e-4)
    p.add_argument("--test-frac", type=float, default=0.35, help="Fraction of FAMILIES held out.")
    p.add_argument("--out", default=None)
    return p.parse_args()


# ----------------------------------------------------------------- feature bookkeeping


def _to_cpu_rows(feats: dict) -> list[dict]:
    """Split a batched feature dict into per-sample dicts, cropped to each own length.

    Stored per sample rather than as a padded batch because the eval batches have different
    `n`, and padding everything to a global max would make the stored tensors mostly
    padding -- and, worse, would let the padded width differ between the real and fake
    passes, which is a feature the D could read.
    """
    out = []
    b = feats["mask"].shape[0]
    for i in range(b):
        m = feats["mask"][i].bool()
        n = int(m.sum())
        if n <= 1:
            continue  # a 1-residue row has no dihedral or neighbour structure
        row = {}
        for k, v in feats.items():
            if k == "glob":
                row[k] = v[i].detach().float().cpu()
            else:
                row[k] = v[i][m].detach().float().cpu()
        out.append(row)
    return out


def _collate(rows: list[dict], device) -> dict:
    """Pad a list of per-sample dicts back into a batch."""
    n_max = max(r["mask"].shape[0] for r in rows)
    out: dict[str, torch.Tensor] = {}
    keys = [k for k in rows[0] if k != "glob"]
    for k in keys:
        stacked = []
        for r in rows:
            v = r[k]
            pad = n_max - v.shape[0]
            if pad:
                v = F.pad(v, (0, 0) * (v.dim() - 1) + (0, pad)) if v.dim() == 1 else \
                    F.pad(v, (0, 0, 0, pad))
            stacked.append(v)
        out[k] = torch.stack(stacked).to(device)
    out["mask"] = out["mask"].bool()
    out["glob"] = torch.stack([r["glob"] for r in rows]).to(device)
    return out


def _train_fresh_d(
    real_tr, fake_tr, real_te, fake_te, dcfg, contact_cutoff, seed, epochs, lr, device
) -> dict:
    """One fresh discriminator, trained on the train families and scored on the held-out ones."""
    from proteinfoundation.cp2lp.discriminator import PeptideInterfaceDiscriminator

    torch.manual_seed(seed)
    d = PeptideInterfaceDiscriminator(
        hidden_dim=int(dcfg.get("hidden_dim", 192)),
        nlayers=int(dcfg.get("nlayers", 3)),
        nheads=int(dcfg.get("nheads", 6)),
        k_pocket=int(dcfg.get("k_pocket", 8)),
        contact_cutoff_nm=contact_cutoff,
        pocket_identity=str(dcfg.get("pocket_identity", "chem")),
        shortcut_probe=False,  # not needed here, and it would add an unused head
        dropout=float(dcfg.get("dropout", 0.1)),
        noise_level_dim=0,
    ).to(device)
    opt = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.9))

    # BALANCED by construction: equal numbers of real and fake, so 0.5 is chance and
    # accuracy is readable without a prevalence correction.
    n = min(len(real_tr), len(fake_tr))
    rtr, ftr = real_tr[:n], fake_tr[:n]
    nte = min(len(real_te), len(fake_te))
    rte, fte = real_te[:nte], fake_te[:nte]
    if n < 8 or nte < 4:
        return {"n_train_per_class": n, "n_test_per_class": nte, "status": "too few samples"}

    bs = 16
    d.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs].tolist()
            if len(idx) < 2:
                continue
            rb = _collate([rtr[i] for i in idx], device)
            fb = _collate([ftr[i] for i in idx], device)
            lr_logit, lf_logit = d(rb), d(fb)
            # Plain BCE rather than the training hinge: this D is a MEASUREMENT instrument,
            # and a calibrated probability is what the AUC below wants.
            loss = F.binary_cross_entropy_with_logits(
                lr_logit, torch.ones_like(lr_logit)
            ) + F.binary_cross_entropy_with_logits(lf_logit, torch.zeros_like(lf_logit))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    d.eval()
    with torch.no_grad():
        logits, labels = [], []
        for rows, lab in ((rte, 1.0), (fte, 0.0)):
            for s in range(0, len(rows), bs):
                chunk = rows[s : s + bs]
                if len(chunk) < 2:
                    continue
                out = d(_collate(chunk, device)).reshape(-1)
                logits.append(out.cpu())
                labels.append(torch.full_like(out.cpu(), lab))
        lg, lb = torch.cat(logits), torch.cat(labels)
    acc = float(((lg > 0).float() == lb).float().mean())
    # AUC by rank, no sklearn dependency.
    order = lg.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(lg), dtype=torch.float) + 1.0
    n_pos, n_neg = float((lb == 1).sum()), float((lb == 0).sum())
    auc = float("nan")
    if n_pos > 0 and n_neg > 0:
        auc = float((ranks[lb == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {
        "seed": seed, "acc": acc, "auc": auc,
        "n_train_per_class": n, "n_test_per_class": nte, "status": "ok",
    }


def main() -> int:  # noqa: C901
    args = parse_args()
    load_dotenv(REPO / ".env")

    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        cfg = compose(config_name=args.config)

    print("=" * 78)
    print("FRESH HELD-OUT DISCRIMINATOR")
    print("=" * 78)
    print(f"config : {args.config}   run_name {cfg.run_name}")
    print(f"ckpt   : {args.ckpt}")
    print(f"clash_inter_norm: {cfg.cp2lp.losses.get('clash_inter_norm', 'all_pairs')}")

    cfg.dataset.datamodule.num_workers = 2
    cfg.dataset.unified.datamodule.num_workers = 2
    cfg.cp2lp.max_g_updates = 0
    cfg.cp2lp.eval_checkpoint_g_updates = []
    cfg.cp2lp.feature_scales.min_examples = 1
    if cfg.get("val_generation") is not None:
        cfg.val_generation.enabled = False
    if cfg.get("rollout_finetune") is not None:
        cfg.rollout_finetune.enabled = False

    from proteinfoundation.cp2lp.data import family_of, slice_batch, split_roles
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    model = CP2LPGenerator(cfg)
    if not os.path.exists(args.ckpt):
        logger.error(f"checkpoint not found: {args.ckpt}")
        return 1
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  g_updates {int(model.g_updates)}   missing {len(missing)}  unexpected {len(unexpected)}")
    del ckpt

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # The VALIDATION loader: held-out receptor families. Training families must not appear
    # here at all -- the generator was fit on them, and a D trained to separate its outputs
    # on seen complexes measures memorisation.
    _, datamodule = load_data_module(cfg, is_cluster_run=True)
    datamodule.setup("fit")
    try:
        loader = iter(datamodule.val_dataloader())
        which = "val"
    except Exception:
        loader = iter(datamodule.train_dataloader())
        which = "train (NO val loader -- families are NOT held out)"
    print(f"loader : {which}")

    real_rows: list[tuple[str, dict]] = []
    fake_rows: list[tuple[str, dict]] = []
    srccp_rows: list[tuple[str, dict]] = []

    def family_keys(batch, nrows):
        """Family key per row, so the split can be grouped. Falls back loudly."""
        ids = batch.get("example_id", None)
        if isinstance(ids, (list, tuple)) and len(ids) >= nrows:
            return [family_of({"example_id": str(ids[i])}) for i in range(nrows)]
        return [f"__unknown_{i}" for i in range(nrows)]

    for bi in range(int(args.batches)):
        torch.manual_seed(5000 + bi)
        try:
            batch = next(loader)
        except StopIteration:
            print(f"loader exhausted after {bi} batches")
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probe = add_clean_samples(dict(batch), cfg.product_flowmatcher, model.autoencoder)
        cyclic, linear = split_roles(probe)
        bs = probe["mask"].shape[0]
        fams = family_keys(probe, bs)

        with torch.no_grad():
            # ---- FAKE: every attempt, no export gate.
            if int(cyclic.sum()) > 0:
                cp_batch = slice_batch(probe, cyclic, bs)
                cfams = [f for f, c in zip(fams, cyclic.tolist()) if c]
                gen_batch, decoded, _ = model._generate(cp_batch)
                gmask = gen_batch["mask"].bool()
                ff = model._featurize(decoded, gen_batch, gmask)
                rep = gmask.shape[0] // max(len(cfams), 1)
                expanded = [f for f in cfams for _ in range(max(rep, 1))]
                for f, r in zip(expanded, _to_cpu_rows(ff)):
                    fake_rows.append((f, r))

                # ---- POSITIVE CONTROL: the SOURCE cyclic peptides, through the same
                # decode and the same featuriser. A closed ring against real linear
                # peptides must be near-perfectly separable.
                from proteinfoundation.cp2lp.conditioning import SRC_CP_KEY

                if SRC_CP_KEY in gen_batch:
                    src_dec = model._decode(gen_batch[SRC_CP_KEY], gmask)
                    sf = model._featurize(src_dec, gen_batch, gmask)
                    for f, r in zip(expanded, _to_cpu_rows(sf)):
                        srccp_rows.append((f, r))

            # ---- REAL: the AE ROUND TRIP of a real bound LP, exactly as training feeds D.
            if int(linear.sum()) > 0:
                lp_batch = slice_batch(probe, linear, bs)
                lfams = [f for f, c in zip(fams, linear.tolist()) if c]
                lmask = lp_batch["mask"].bool()
                lp_clean = {
                    dm: lp_batch["x_1"][dm].detach() * lmask[..., None].to(lp_batch["x_1"][dm].dtype)
                    for dm in model._data_modes
                }
                rdec = model._decode(lp_clean, lmask)
                rf = model._featurize(rdec, lp_batch, lmask)
                for f, r in zip(lfams, _to_cpu_rows(rf)):
                    real_rows.append((f, r))

        if (bi + 1) % 8 == 0:
            print(f"  batch {bi+1}: real {len(real_rows)}  fake {len(fake_rows)}  srcCP {len(srccp_rows)}")

    print(f"collected: real {len(real_rows)}  fake {len(fake_rows)}  src_cp {len(srccp_rows)}")
    if len(real_rows) < 16 or len(fake_rows) < 16:
        logger.error("too few samples collected; raise --batches")
        return 1

    # ---------------------------------------------------- grouped family split
    import hashlib

    fams_all = sorted({f for f, _ in real_rows} | {f for f, _ in fake_rows})
    unknown = [f for f in fams_all if f.startswith("__unknown_")]
    if unknown:
        print(f"WARNING: {len(unknown)} rows carry no family id; the split for those rows is "
              f"NOT grouped and the numbers below are optimistic.")

    def is_test(fam: str) -> bool:
        h = hashlib.sha256(f"freshd:{fam}".encode()).digest()
        return (int.from_bytes(h[:4], "big") / 2**32) < args.test_frac

    def split(rows):
        tr = [r for f, r in rows if not is_test(f)]
        te = [r for f, r in rows if is_test(f)]
        return tr, te

    real_tr, real_te = split(real_rows)
    fake_tr, fake_te = split(fake_rows)
    src_tr, src_te = split(srccp_rows)
    n_test_fams = sum(1 for f in fams_all if is_test(f))
    print(f"families: {len(fams_all)} total, {n_test_fams} held out "
          f"({len(real_te)} real / {len(fake_te)} fake test rows)")

    dcfg = cfg.cp2lp.get("discriminator", None) or {}
    cc = float(cfg.cp2lp.losses.contact_cutoff_nm)

    results = {"arm": str(cfg.run_name), "ckpt": args.ckpt,
               "clash_inter_norm": str(cfg.cp2lp.losses.get("clash_inter_norm", "all_pairs")),
               "n_real": len(real_rows), "n_fake": len(fake_rows),
               "n_families": len(fams_all), "n_test_families": n_test_fams,
               "generated": [], "positive_control": []}

    print("-" * 78)
    print("GENERATED vs REAL (the measurement)")
    for s in range(int(args.seeds)):
        r = _train_fresh_d(real_tr, fake_tr, real_te, fake_te, dcfg, cc,
                           1234 + s, args.epochs, args.d_lr, device)
        results["generated"].append(r)
        print(f"  seed {1234+s}: acc {r.get('acc', float('nan')):.4f}  "
              f"auc {r.get('auc', float('nan')):.4f}  ({r['status']}, "
              f"train/class {r['n_train_per_class']}, test/class {r['n_test_per_class']})")

    print("-" * 78)
    print("POSITIVE CONTROL -- source CYCLIC peptides vs real LPs (must be near 1.0)")
    if len(src_tr) >= 8 and len(src_te) >= 4:
        for s in range(int(args.seeds)):
            r = _train_fresh_d(real_tr, src_tr, real_te, src_te, dcfg, cc,
                               4321 + s, args.epochs, args.d_lr, device)
            results["positive_control"].append(r)
            print(f"  seed {4321+s}: acc {r.get('acc', float('nan')):.4f}  "
                  f"auc {r.get('auc', float('nan')):.4f}")
    else:
        print("  NOT RUN: no source-CP features collected.")

    ok = [r for r in results["generated"] if r["status"] == "ok"]
    ctl = [r for r in results["positive_control"] if r["status"] == "ok"]
    print("=" * 78)
    if ok:
        accs = [r["acc"] for r in ok]
        aucs = [r["auc"] for r in ok]
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        results["acc_mean"], results["auc_mean"] = mean(accs), mean(aucs)
        print(f"generated vs real : acc {mean(accs):.4f} [{min(accs):.4f}, {max(accs):.4f}]   "
              f"auc {mean(aucs):.4f}")
    if ctl:
        cacc = [r["acc"] for r in ctl]
        results["control_acc_mean"] = sum(cacc) / len(cacc)
        print(f"positive control  : acc {results['control_acc_mean']:.4f}")
        if results["control_acc_mean"] < 0.9:
            print("VERDICT: NOT MEASURED. The harness cannot separate cyclic from real linear")
            print("         peptides, so a low score on the generated set is uninformative.")
        else:
            print("control passed, so the generated-vs-real number above is interpretable.")
    print()
    print("Compare this ACROSS ARMS at matched budget. In isolation it is not a verdict:")
    print("the question is whether the clash-normalised arm scores LOWER than its control.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
