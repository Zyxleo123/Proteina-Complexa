#!/usr/bin/env python3
"""Read-only sanity inspection of the batches fed to the CPSea flow model.

Motivation
----------
We found two data bugs by accident (binder mask spanning the whole complex; the AE
training on the whole complex). This script proactively checks the *actual* tensors the
model consumes, over many real batches, and flags anything suspicious. It never trains,
never writes to the dataset, and loads no model weights (pure dataloader inspection), so
it is safe to run anytime.

What it checks (per loaded batch, aggregated):
  MASK / LENGTH
    - binder `mask` == CA presence (`coord_mask[...,CA]`)         [the bug we just fixed]
    - padded positions (mask False) carry no CA / zero coords
    - binder length in [binder_min, binder_max] (config 5..16)
    - pad efficiency: real binder tokens / padded tokens
  POSITIONAL INDEX (relevant to the advisor's PE question)
    - binder `residue_pdb_idx` contiguity (diff==1 along the chain)
    - offset distribution (do peptides start at 0/1, or keep parent-protein numbering?)
  COORDINATES
    - NaN / Inf in binder `coords_nm` and `x_target`
    - unit scale (nm vs A): consecutive CA-CA distance should be ~0.38 nm
    - coordinate magnitude sanity
  SEQUENCE / ATOMS
    - residue_type in [0,19] for binder and target
    - backbone (N,CA,C,O) completeness on real binder residues
    - atoms-per-residue distribution
  TARGET
    - target length distribution, NaN check, target chain != binder chain
  CYCLIZATION
    - whether any cyclization/CONECT info reaches the batch (known to be dropped)

Usage
-----
    source env.sh
    # full data:
    CPSEA_DATA_PATH=$CPSEA_FULL_DATA_PATH python script_utils/inspect_cpsea_data.py \
        --config-name example/training_cpsea_peptide_smoke --num-batches 15 --batch-size 8
    # AE (peptide-only) pipeline:
    python script_utils/inspect_cpsea_data.py --config-name training_ae_cpsea --ae

Runs on CPU (dataloader only). Prints a report with PASS / WARN / FAIL per check.
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

CA_IDX = 1
BACKBONE = [0, 1, 2, 4]  # N, CA, C, O in atom37


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="example/training_cpsea_peptide_smoke")
    p.add_argument("--ae", action="store_true", help="Use the AE train.py loader (config is a training_ae_* config).")
    p.add_argument("--num-batches", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--binder-min", type=int, default=5)
    p.add_argument("--binder-max", type=int, default=16)
    p.add_argument("--data-root", default=None, help="Override CPSEA_DATA_PATH (e.g. the full-data root).")
    return p.parse_args()


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.status = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add(self, level: str, name: str, detail: str):
        self.status[level] = self.status.get(level, 0) + 1
        self.lines.append(f"[{level:4}] {name}: {detail}")

    def dump(self):
        print("\n================ CPSea DATA SANITY REPORT ================")
        for ln in self.lines:
            print(ln)
        print("---------------------------------------------------------")
        print(f"PASS={self.status['PASS']}  WARN={self.status['WARN']}  FAIL={self.status['FAIL']}")


def q(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "n=0"
    return f"min={a.min():.3g} med={np.median(a):.3g} max={a.max():.3g} mean={a.mean():.3g}"


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    if args.data_root:
        os.environ["CPSEA_DATA_PATH"] = args.data_root

    import proteinfoundation.patches.atomworks_patches  # noqa: F401
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    if args.ae:
        from proteinfoundation.partial_autoencoder.train import load_data_module
    else:
        from proteinfoundation.train import load_data_module

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config_name, overrides=["+single=true", "+nolog=true"])

    _, dm = load_data_module(cfg, is_cluster_run=False)
    dm.num_workers = args.num_workers
    dm.batch_size = args.batch_size
    dm.pin_memory = False
    dm.setup("fit")
    print(f"[inspect] metadata: {dm.metadata_file}")
    loader = dm.train_dataloader()

    rep = Report()

    # accumulators
    n_samples = 0
    mask_ca_violations = 0
    pad_ca_violations = 0
    binder_lens: list[int] = []
    target_lens: list[int] = []
    real_tokens = 0
    total_tokens = 0
    noncontig = 0
    offsets: list[int] = []
    nan_binder = 0
    nan_target = 0
    inf_binder = 0
    ca_ca_dists: list[float] = []
    restype_bad_binder = 0
    restype_bad_target = 0
    bb_incomplete = 0
    atoms_per_res: list[int] = []
    binder_target_chain_overlap = 0
    zero_len = 0
    first_batch_keys: list[str] = []
    cycl_types: list[str] = []
    conect_is_count = None

    for batch in itertools.islice(loader, args.num_batches):
        if not first_batch_keys:
            first_batch_keys = sorted([k for k in batch.keys() if isinstance(k, str)])
            cp = batch.get("conect_pairs_kept")
            if cp is not None:
                # A usable cyclization *edge* feature needs the bonded residue indices; a
                # per-sample scalar/count does not carry that. Detect which we have.
                if torch.is_tensor(cp):
                    conect_is_count = cp.dim() <= 1
                elif isinstance(cp, (list, tuple)) and len(cp):
                    e0 = cp[0]
                    conect_is_count = np.isscalar(e0) or (torch.is_tensor(e0) and e0.dim() == 0)
        if "cyclization_type" in batch and isinstance(batch["cyclization_type"], list):
            cycl_types.extend([str(x) for x in batch["cyclization_type"]])

        m = batch["mask"].bool()  # [b, n]
        cm = batch["coord_mask"].bool()  # [b, n, 37]
        ca = cm[..., CA_IDX]  # [b, n]
        coords = batch["coords_nm"]  # [b, n, 37, 3]
        b, n = m.shape

        # mask vs CA
        mask_ca_violations += int((m & ~ca).sum() + (~m & ca).sum())
        pad_ca_violations += int((~m & ca).sum())

        for i in range(b):
            mi = m[i]
            L = int(mi.sum())
            n_samples += 1
            if L == 0:
                zero_len += 1
                continue
            binder_lens.append(L)
            real_tokens += L
            total_tokens += n

            # residue_pdb_idx contiguity + offset (in the masked order)
            if "residue_pdb_idx" in batch:
                rpi = batch["residue_pdb_idx"][i][mi]
                offsets.append(int(rpi.min()))
                d = (rpi[1:] - rpi[:-1])
                if not torch.all(d == 1):
                    noncontig += 1

            # coord nan/inf on real atoms
            real_atoms = cm[i] & mi[:, None]
            cvals = coords[i][real_atoms]
            if torch.isnan(cvals).any():
                nan_binder += 1
            if torch.isinf(cvals).any():
                inf_binder += 1

            # CA-CA consecutive distance (nm)
            cai = coords[i][mi][:, CA_IDX, :]  # [L, 3]
            if L >= 2:
                dd = torch.linalg.norm(cai[1:] - cai[:-1], dim=-1)
                ca_ca_dists.extend(dd.tolist())

            # residue types
            rt = batch["residue_type"][i][mi]
            if int(rt.min()) < 0 or int(rt.max()) > 19:
                restype_bad_binder += 1

            # backbone completeness + atoms per residue
            bbc = cm[i][mi][:, BACKBONE]  # [L, 4]
            if not bool(bbc.all()):
                bb_incomplete += 1
            atoms_per_res.extend(cm[i][mi].sum(-1).tolist())

            # chain overlap
            if "chains" in batch and "target_chains" in batch and "target_padding_mask" in batch:
                bch = set(batch["chains"][i][mi].tolist())
                tpm = batch["target_padding_mask"][i].bool()
                tch = set(batch["target_chains"][i][tpm].tolist())
                if bch & tch:
                    binder_target_chain_overlap += 1

        # target
        if "target_padding_mask" in batch:
            tpm = batch["target_padding_mask"].bool()
            target_lens.extend(tpm.sum(1).tolist())
            if "x_target" in batch:
                xt = batch["x_target"]
                tm = batch["target_mask"].bool() if "target_mask" in batch else tpm[..., None].expand_as(xt[..., 0])
                tvals = xt[tm]
                if torch.isnan(tvals).any():
                    nan_target += 1
        if "seq_target" in batch and "seq_target_mask" in batch:
            st = batch["seq_target"][batch["seq_target_mask"].bool()]
            if st.numel() and (int(st.min()) < 0 or int(st.max()) > 19):
                restype_bad_target += 1

    # ---- verdicts ----
    def flag(ok_cond, warn_cond, name, detail):
        if ok_cond:
            rep.add("PASS", name, detail)
        elif warn_cond:
            rep.add("WARN", name, detail)
        else:
            rep.add("FAIL", name, detail)

    print(f"[inspect] inspected {n_samples} samples across {args.num_batches} batches")

    flag(mask_ca_violations == 0, mask_ca_violations < n_samples,
         "binder_mask==CA", f"{mask_ca_violations} token mismatches (want 0)")
    flag(pad_ca_violations == 0, False,
         "padded_positions_empty", f"{pad_ca_violations} padded tokens that still have a CA (want 0)")

    bl = np.array(binder_lens) if binder_lens else np.array([0])
    in_range = np.mean((bl >= args.binder_min) & (bl <= args.binder_max)) if binder_lens else 0.0
    flag(in_range == 1.0, in_range > 0.95,
         "binder_length_in_range", f"{100*in_range:.1f}% in [{args.binder_min},{args.binder_max}]  ({q(bl)})")

    eff = real_tokens / max(total_tokens, 1)
    flag(eff > 0.5, eff > 0.2,
         "pad_efficiency", f"{100*eff:.1f}% of binder tokens are real (higher is better)")

    flag(zero_len == 0, zero_len < 3, "no_zero_length_binders", f"{zero_len} samples with 0 binder residues")

    if offsets:
        offs = np.array(offsets)
        contig_ok = noncontig == 0
        starts_at_1 = np.mean(offs <= 1)
        flag(contig_ok, noncontig < 0.05 * len(binder_lens),
             "binder_idx_contiguous", f"{noncontig}/{len(binder_lens)} non-contiguous peptides")
        # This is informational (feeds absolute PE). Not necessarily a bug.
        lvl = "PASS" if starts_at_1 > 0.99 else "WARN"
        rep.add(lvl, "binder_idx_offset",
                f"{100*starts_at_1:.1f}% start at idx<=1; offsets {q(offs)} "
                f"(parent-protein numbering -> absolute PE varies per peptide)")

    flag(nan_binder == 0, False, "binder_coords_finite", f"{nan_binder} samples with NaN, {inf_binder} with Inf")
    flag(nan_target == 0, False, "target_coords_finite", f"{nan_target} batches with NaN in x_target")

    if ca_ca_dists:
        d = np.array(ca_ca_dists)
        # peptide has one chain break only if cyclic-linearized; consecutive bonded ~0.38nm.
        frac_bonded = float(np.mean((d > 0.30) & (d < 0.45)))
        flag(0.30 < np.median(d) < 0.45, frac_bonded > 0.6,
             "ca_ca_distance_nm", f"median={np.median(d):.3f} nm (expect ~0.38); {100*frac_bonded:.0f}% in [0.30,0.45]  -> units look like {'nm' if np.median(d)<1 else 'ANGSTROM?!'}")

    flag(restype_bad_binder == 0, False, "binder_residue_type_range", f"{restype_bad_binder} samples out of [0,19]")
    flag(restype_bad_target == 0, False, "target_residue_type_range", f"{restype_bad_target} batches out of [0,19]")

    flag(bb_incomplete == 0, bb_incomplete < 0.05 * max(len(binder_lens), 1),
         "backbone_complete", f"{bb_incomplete}/{len(binder_lens)} peptides missing an N/CA/C/O somewhere")

    if atoms_per_res:
        apr = np.array(atoms_per_res)
        flag(apr.min() >= 4, apr.min() >= 3,
             "atoms_per_residue", f"{q(apr)} (>=4 expected: backbone N,CA,C,O)")

    if target_lens:
        flag(True, True, "target_length", q(np.array(target_lens)))
    flag(binder_target_chain_overlap == 0, False,
         "binder_target_chain_disjoint", f"{binder_target_chain_overlap} samples share a chain id between binder and target")

    if cycl_types:
        from collections import Counter
        dist = ", ".join(f"{k}={v}" for k, v in Counter(cycl_types).most_common())
        rep.add("PASS", "cyclization_type_available", f"per-peptide category present: {dist}")
    else:
        rep.add("WARN", "cyclization_type_available", "no cyclization_type in batch")
    # The actual bonded residue indices are NOT in the batch (conect_pairs_kept is a count),
    # so a cyclization *edge* feature needs CONECT re-parsing -- EXCEPT head_tail, whose bond
    # is deterministically (first,last) binder residue and can be built from the type alone.
    rep.add("WARN" if conect_is_count else "PASS", "cyclization_bond_pairs",
            "conect_pairs_kept is a count, not residue pairs -> edge feature needs CONECT "
            "re-parse (head_tail is derivable as first<->last)" if conect_is_count
            else "bonded-pair info present")

    rep.add("PASS", "batch_keys", f"{len(first_batch_keys)} keys: {', '.join(first_batch_keys)}")
    rep.dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
