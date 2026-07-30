#!/usr/bin/env python3
"""Audit that the head-to-tail cyclization label survives the trip into the model.

`audit_cpsea_cyclization_endpoints.py` establishes the fact about the *data*
(every CPSea ring closes between the first and last binder residue, and
preprocessing preserves it). This script establishes that the fact actually
reaches the tensors the network consumes, on real batches:

  1. LABEL      `cyclization_i/j` from `CyclizationLabelTransform` equal
                `(0, L_b - 1)` for every labeled sample.
  2. TRAIN PE   `CyclizationGraphPositionalPairFeat` resolves the same endpoints.
  3. DESIGN PE  with `cyclization_i/j` REMOVED from the batch -- which is exactly
                what a design batch looks like -- the feature falls back to the
                termini and produces a *bit-identical* pair tensor. This is the
                train/design parity claim; if it ever breaks, designs are
                conditioned on a different ring than training used.
  4. LOSS       `extract_cyclization_metadata` (bond loss / closure metric) reads
                the same endpoints.
  5. MASK       the validity mask keeps the gold `(i, j, type)` entry, under both
                the type-conditioned and the `terminal_only` settings.

Usage
-----
    source env.sh
    python script_utils/audit_cyclization_endpoint_propagation.py --num-batches 20
    # against the full data root:
    python script_utils/audit_cyclization_endpoint_propagation.py \
        --data-root $CPSEA_FULL_DATA_PATH --num-batches 50
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="example/training_cpsea_peptide_cyc_typecond")
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", default=None, help="Override CPSEA_DATA_PATH.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    if args.data_root:
        os.environ["CPSEA_DATA_PATH"] = args.data_root

    import proteinfoundation.patches.atomworks_patches  # noqa: F401
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from proteinfoundation.cyclization.constants import NO_CYCLIZATION_INDEX, UNSPECIFIED
    from proteinfoundation.cyclization.mask import build_cyclization_validity_mask
    from proteinfoundation.eval.cyclic_reconstruction_metrics import extract_cyclization_metadata
    from proteinfoundation.nn.feature_factory.pair_feats import CyclizationGraphPositionalPairFeat
    from proteinfoundation.train import load_data_module

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config_name, overrides=["+single=true", "+nolog=true"])

    _, dm = load_data_module(cfg, is_cluster_run=False)
    dm.num_workers = args.num_workers
    dm.batch_size = args.batch_size
    dm.pin_memory = False
    dm.setup("fit")
    loader = dm.train_dataloader()
    print(f"[audit] metadata: {dm.metadata_file}")

    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=32)

    n = 0
    n_labeled = 0
    fails = {
        "label_not_terminal": 0,
        "train_pe_endpoint_mismatch": 0,
        "design_pe_endpoint_mismatch": 0,
        "design_pe_tensor_differs": 0,
        "loss_meta_mismatch": 0,
        "gold_masked_out": 0,
        "gold_masked_out_terminal_only": 0,
    }
    examples: list[str] = []

    for batch in itertools.islice(loader, args.num_batches):
        if "cyclization_i" not in batch:
            print("[audit] FAIL: batch carries no cyclization_i -- transform not wired in")
            return 1

        mask = batch["mask"].bool()
        b, L = mask.shape
        lengths = mask.sum(dim=1)
        ci = batch["cyclization_i"].long()
        cj = batch["cyclization_j"].long()
        labeled = batch["has_cyclization"].bool()
        n += b
        n_labeled += int(labeled.sum())

        # ---- 1. LABEL is (0, L_b - 1) ----
        terminal = (ci == 0) & (cj == lengths - 1)
        bad = labeled & ~terminal
        fails["label_not_terminal"] += int(bad.sum())
        for k in torch.nonzero(bad).flatten().tolist()[:3]:
            examples.append(f"label_not_terminal: i={ci[k].item()} j={cj[k].item()} L={lengths[k].item()}")

        # ---- 2. TRAIN-time ring PE endpoints ----
        i_tr, j_tr = feat._resolve_endpoints(batch, mask, b, L, mask.device)
        mism = labeled & ((i_tr != ci) | (j_tr != cj))
        fails["train_pe_endpoint_mismatch"] += int(mism.sum())

        # ---- 3. DESIGN-time: no labels in the batch, only a requested type ----
        design_batch = {k: v for k, v in batch.items() if k not in ("cyclization_i", "cyclization_j")}
        i_de, j_de = feat._resolve_endpoints(design_batch, mask, b, L, mask.device)
        mism_de = labeled & ((i_de != 0) | (j_de != lengths - 1))
        fails["design_pe_endpoint_mismatch"] += int(mism_de.sum())
        with torch.no_grad():
            pe_train = feat(batch)
            pe_design = feat(design_batch)
        # Compare only labeled rows: unlabeled rows have cyclization_type_cond=UNSPECIFIED
        # and are all-zero on both sides anyway.
        rows = torch.nonzero(labeled).flatten()
        if rows.numel():
            if not torch.equal(pe_train[rows], pe_design[rows]):
                diff = (pe_train[rows] - pe_design[rows]).abs().max().item()
                fails["design_pe_tensor_differs"] += int(rows.numel())
                examples.append(f"design_pe_tensor_differs: max|d|={diff:.3g}")

        # ---- 4. LOSS metadata ----
        meta = extract_cyclization_metadata(batch)
        if meta is None:
            fails["loss_meta_mismatch"] += b
        else:
            mism_l = labeled & ((meta["i"] != ci) | (meta["j"] != cj))
            fails["loss_meta_mismatch"] += int(mism_l.sum())

        # ---- 5. VALIDITY MASK keeps the gold entry ----
        aa = batch["residue_type"].long()
        gold_type = batch["cyclization_type"].long().clamp(min=0)
        cond = batch.get("cyclization_type_cond")
        cond = cond.long() if cond is not None else torch.full((b,), UNSPECIFIED)
        for key, terminal_only in (("gold_masked_out", False), ("gold_masked_out_terminal_only", True)):
            valid = build_cyclization_validity_mask(
                aa=aa,
                binder_mask=mask,
                gold_i=ci.clamp(min=0),
                gold_j=cj.clamp(min=0),
                gold_type=gold_type,
                force_gold_valid=False,  # the point is whether the RULES already admit it
                cond_type=cond,
                terminal_only=terminal_only,
            )
            idx = torch.arange(b)
            kept = valid[idx, ci.clamp(min=0), cj.clamp(min=0), gold_type]
            miss = labeled & ~kept
            fails[key] += int(miss.sum())
            for k in torch.nonzero(miss).flatten().tolist()[:2]:
                examples.append(
                    f"{key}: i={ci[k].item()} j={cj[k].item()} type={gold_type[k].item()} "
                    f"aa_i={aa[k, ci[k]].item()} aa_j={aa[k, cj[k]].item()}"
                )

    print(f"\n{'=' * 70}\nCYCLIZATION ENDPOINT PROPAGATION -- {n} samples ({n_labeled} labeled)\n{'=' * 70}")
    for k, v in fails.items():
        print(f"[{'PASS' if v == 0 else 'FAIL'}] {k}: {v}")
    if examples:
        print("\nexamples:")
        for e in examples[:20]:
            print("  " + e)
    unlabeled = n - n_labeled
    if unlabeled:
        print(f"\nnote: {unlabeled} samples unlabeled (has_cyclization=False); excluded from every check above")
    return 0 if all(v == 0 for v in fails.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
