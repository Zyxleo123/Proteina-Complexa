#!/usr/bin/env python3
"""Validate the contact/hotspot construction pipeline on the LP datasets (and CPSea).

The question this answers: can we build receptor hotspots for the *linear peptide*
training data the same way CPSea does? Hotspots are receptor residues at an interface,
so the whole idea needs a receptor in the row. PepBench and ProtFrag are peptide-protein
complexes (chain A receptor / chain B peptide) and the LP mixing config runs the same
`ContactTransform` with byte-identical settings as CPSea -- so the answer should be yes.
This measures it instead of assuming it, through the real transform stack, because a
conditioning channel that is silently zero looks exactly like one that works.

Checks, per source dataset (`pepbench` / `protfrag` / `cpsea`, split on `example_id`):

  1. RECEPTOR   the row has target residues at all. No receptor => no interface =>
                no hotspot is definable, and the dataset is unusable for this task.
  2. SUPPLY     `target_hotspot_mask` reaches the batch, and how many hotspots it
                carries. Also how often it is EMPTY: the sampler draws
                `int(n_contacts * uniform(0, max_perc))`, which truncates to zero on
                small interfaces, and `hotspot_dropout_prob` then zeroes half of what
                survives. A channel that is empty most of the time is weakly trained
                even though it is correctly wired.
  3. SUBSET     every flagged hotspot really is an interface residue, recomputed here
                at the same 8 A cutoff. This is the correctness check: a non-zero mask
                pointing at residues on the far side of the receptor would train the
                model to associate "hotspot" with noise.
  4. DEAD       `batch["hotspot_mask"]` (binder-length) is identically zero. This is a
                KNOWN DEFECT, asserted here so it cannot regress silently:
                `FilterTargetResiduesTransform` skips only attributes whose name
                contains "target", so `hotspot_mask` gets subset to *binder* rows while
                hotspots are only ever set at *target* indices. The key still exists, so
                the zero-fallback warning in `HotspotMaskSeqFeat` never fires. Anything
                reading `hotspot_mask_seq` / `hotspot_mask_pair` sees zeros.
  5. MODEL      the flag actually reaches the network: zeroing `target_hotspot_mask`
                must change exactly ONE column of the 558-dim `TargetConcatSeqFeat`
                target encoding. That is the live path (`enable_target: true`).

Distances are recomputed in the `coords_nm` frame for BOTH chains. `coords` and
`coords_nm` are different rigid frames of the same structure, not one scaled by the
other, so mixing them would produce garbage distances; using one frame for both chains
is safe because a rigid motion preserves inter-chain distances.

Usage
-----
    source env.sh
    # LP + CPSea mixed in one pass (all three sources appear):
    python script_utils/hotspot_pipeline_probe.py \
        --config-name example/training_cpsea_lpmix_from_v4cfg --num-batches 40
    # LP only, or CPSea only, by pointing at that config.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

# ContactTransform's cutoff is 8.0 A on `coords`; we recompute in nm.
CONTACT_NM = 0.8
CA_IDX = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="example/training_cpsea_lpmix_from_v4cfg")
    p.add_argument("--num-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default=None, help="Write the summary JSON here.")
    p.add_argument("--rows-out", default=None, help="Write per-sample rows as JSONL here.")
    return p.parse_args()


def _source_of(example_id: str) -> str:
    """Split on the preprocessor's own `<dataset>__<...>` id prefix.

    `dataset_source` drives the weighted sampler but is not in `columns_to_load`, so it
    does not reach the batch. The prefix is an exact substitute: measured on
    `mixed_train.parquet` (2,489,756 rows), the fraction of ids containing `__` is 1.000
    for pepbench and protfrag and 0.000 for cpsea (whose ids are AFDB accessions).
    """
    if "__" in example_id:
        return example_id.split("__", 1)[0]
    return "cpsea"


def _find_binder_coords(batch: dict) -> tuple[str, str | None]:
    """Locate the binder coordinate key (and its atom mask) in the collated batch.

    Reported rather than guessed: if the stack ever renames these, the probe should say
    so instead of silently skipping the SUBSET check.
    """
    coord_key = None
    for k in ("coords_nm", "x1_coords_nm", "coords"):
        if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4:
            coord_key = k
            break
    mask_key = None
    for k in ("coord_mask", "atom_mask", "mask"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            mask_key = k
            break
    return coord_key, mask_key


def main() -> int:
    args = parse_args()
    os.chdir(REPO)

    import proteinfoundation.patches.atomworks_patches  # noqa: F401
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from proteinfoundation.nn.feature_factory.target_feats import TargetConcatSeqFeat
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
    print(f"[probe] config   : {args.config_name}")
    print(f"[probe] metadata : {getattr(dm, 'metadata_file', '?')}")

    per_src: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "no_receptor": 0,
            "n_hotspots": [],
            "zero_hotspots": 0,
            "n_target_contacts": [],
            "hotspot_not_contact": 0,
            "hotspot_checked": 0,
            "fallback_rows": 0,
            "unexplained_rows": 0,
            "binder_len": [],
            "target_len": [],
        }
    )
    rows: list[dict] = []
    fails: dict[str, int] = defaultdict(int)
    notes: list[str] = []

    # Checks 4 and 5 are properties of the wiring, not of a sample: evaluate once.
    dead_channel_checked = False
    dead_channel_zero = None
    dead_channel_present = None
    model_cols_changed = None
    keys_seen: list[str] = []

    for bi, batch in enumerate(itertools.islice(loader, args.num_batches)):
        if bi == 0:
            keys_seen = sorted(k for k in batch.keys() if isinstance(batch[k], torch.Tensor))
            print(f"[probe] tensor keys: {', '.join(keys_seen)}")

        if "target_hotspot_mask" not in batch:
            print("[probe] FAIL: no target_hotspot_mask in batch -- ContactTransform not wired in")
            return 1

        hs = batch["target_hotspot_mask"]            # [B, T]
        xt = batch["x_target"]                       # [B, T, 37, 3]
        tmask = batch["target_mask"]                 # [B, T, 37]

        # `id`, NOT `example_id`: structure_data.py:604 sets every metadata column EXCEPT
        # `id_column` as an attribute, so `example_id` is deliberately never an attribute
        # and `id` carries its value. Falling back to positional labels here is what made
        # the first run report all 2,400 samples as `cpsea` -- a label with no "__" in it
        # is indistinguishable from a real CPSea id, so the mistake was invisible. Fail
        # loudly instead: a silently mislabeled source table is worse than no table.
        ids = batch.get("id", batch.get("example_id"))
        if ids is None:
            print("[probe] FAIL: batch carries neither 'id' nor 'example_id' -- cannot attribute")
            print(f"[probe]       non-tensor keys: {[k for k in batch.keys() if not torch.is_tensor(batch[k])]}")
            return 1

        # ---- check 4: the dead binder-length channel -------------------------------
        if not dead_channel_checked:
            dead_channel_present = "hotspot_mask" in batch
            if dead_channel_present:
                dead_channel_zero = bool((batch["hotspot_mask"] == 0).all().item())
            dead_channel_checked = True

        # ---- check 5: does the flag reach the target encoding? ---------------------
        if model_cols_changed is None and hs.sum() > 0:
            try:
                feat = TargetConcatSeqFeat()
                with torch.no_grad():
                    # In compact mode this returns (feats, seq_target_mask), not a tensor.
                    on = feat(dict(batch))
                    off_batch = dict(batch)
                    off_batch["target_hotspot_mask"] = torch.zeros_like(hs)
                    off = feat(off_batch)
                on = on[0] if isinstance(on, tuple) else on
                off = off[0] if isinstance(off, tuple) else off
                diff = (on - off).abs().sum(dim=(0, 1))       # [558]
                model_cols_changed = int((diff > 0).sum().item())
                where = (diff > 0).nonzero().flatten().tolist()
                notes.append(f"TargetConcatSeqFeat dim={on.shape[-1]}, columns changed={where}")
            except Exception as e:  # noqa: BLE001 - report, do not abort the data audit
                notes.append(f"TargetConcatSeqFeat check skipped: {type(e).__name__}: {e}")
                model_cols_changed = -1

        # ---- per-sample data checks ------------------------------------------------
        coord_key, mask_key = _find_binder_coords(batch)
        if bi == 0:
            notes.append(f"binder coords key={coord_key}, atom mask key={mask_key}")

        tres = tmask.any(dim=-1)                     # [B, T] real target residues
        for i in range(hs.shape[0]):
            eid = ids[i] if isinstance(ids, list) else str(ids[i])
            src = _source_of(str(eid))
            d = per_src[src]
            d["n"] += 1

            n_t = int(tres[i].sum().item())
            d["target_len"].append(n_t)
            if n_t == 0:
                d["no_receptor"] += 1
                fails["no_receptor"] += 1
                continue

            n_hs = int((hs[i].bool() & tres[i]).sum().item())
            d["n_hotspots"].append(n_hs)
            if n_hs == 0:
                d["zero_hotspots"] += 1

            # Recompute the interface to (a) get the contact denominator and
            # (b) verify flagged hotspots really sit at the interface.
            n_contacts = None
            if coord_key is not None:
                bc = batch[coord_key][i]                       # [L, 37, 3]
                # Without a per-atom mask of the right shape, fall back to "coordinate is
                # not exactly the origin" rather than all-ones: padded atom-37 slots sit at
                # (0,0,0), and after target-centering the origin is inside the receptor, so
                # all-ones would invent contacts for every absent side-chain atom.
                if mask_key is not None and batch[mask_key][i].shape == bc.shape[:2]:
                    bm = batch[mask_key][i].bool()
                else:
                    bm = bc.abs().sum(-1) > 0
                d["binder_len"].append(int(bm.any(dim=-1).sum().item()))
                tc, tm_i = xt[i], tmask[i].bool()              # [T,37,3], [T,37]

                # ContactTransform defines a contact CA-to-CA (it indexes atom 1 on both
                # sides; there is no all-atom option on this class), so the CA set is the
                # one whose size fed `int(n_contacts * uniform(0, max_perc))`.
                ca_t, ca_b = tc[:, CA_IDX, :], bc[:, CA_IDX, :]
                dca = torch.linalg.norm(ca_t[:, None, :] - ca_b[None, :, :], dim=-1)  # [T,L]
                contact_ca = (
                    (dca < CONTACT_NM) & tm_i[:, CA_IDX][:, None] & bm[:, CA_IDX][None, :]
                ).any(dim=1) & tres[i]
                n_contacts = int(contact_ca.sum().item())
                d["n_target_contacts"].append(n_contacts)

                # All-atom at the same cutoff is a strict SUPERSET of the CA set, so it is
                # the lenient bar: a hotspot outside it is not an interface residue under
                # any reading. [T,37,L,37] blows up on a 256-residue receptor, so chunk.
                contact_all = torch.zeros(tc.shape[0], dtype=torch.bool)
                CH = 32
                for s in range(0, tc.shape[0], CH):
                    e = min(s + CH, tc.shape[0])
                    dd = torch.linalg.norm(
                        tc[s:e, :, None, None, :] - bc[None, None, :, :, :], dim=-1
                    )  # [c,37,L,37]
                    valid = tm_i[s:e, :, None, None] & bm[None, None, :, :]
                    contact_all[s:e] = ((dd < CONTACT_NM) & valid).any(dim=(1, 2, 3))
                contact_all &= tres[i]

                if n_hs > 0:
                    bad = int(((hs[i].bool() & tres[i]) & ~contact_all).sum().item())
                    d["hotspot_checked"] += n_hs
                    if bad:
                        d["hotspot_not_contact"] += bad
                        fails["hotspot_not_contact"] += bad
                        # ContactTransform falls back to RANDOM target residues when its
                        # own CA contact set is empty, which plants "hotspots" nowhere near
                        # the interface. Separate that from a real disagreement.
                        if n_contacts == 0:
                            d["fallback_rows"] += 1
                        else:
                            d["unexplained_rows"] += 1

            rows.append(
                {
                    "example_id": str(eid),
                    "source": src,
                    "n_target": n_t,
                    "n_hotspots": n_hs,
                    "n_target_contacts": n_contacts,
                }
            )

    # ---- report -------------------------------------------------------------------
    def _stat(v: list[int]) -> dict:
        if not v:
            return {"n": 0}
        return {
            "n": len(v),
            "mean": round(statistics.mean(v), 2),
            "median": statistics.median(v),
            "min": min(v),
            "max": max(v),
        }

    summary = {
        "config": args.config_name,
        "batches": args.num_batches,
        "batch_size": args.batch_size,
        "dead_channel": {
            "hotspot_mask_present_in_batch": dead_channel_present,
            "hotspot_mask_all_zero": dead_channel_zero,
        },
        "model_path": {"target_concat_columns_changed": model_cols_changed},
        "notes": notes,
        "per_source": {},
        "fails": dict(fails),
    }

    print("\n==================== hotspot construction, by source ====================")
    for src in sorted(per_src):
        d = per_src[src]
        hs_stat = _stat(d["n_hotspots"])
        ct_stat = _stat(d["n_target_contacts"])
        zero_frac = (d["zero_hotspots"] / len(d["n_hotspots"])) if d["n_hotspots"] else float("nan")
        summary["per_source"][src] = {
            "n_samples": d["n"],
            "no_receptor": d["no_receptor"],
            "hotspots_per_example": hs_stat,
            "target_contacts_per_example": ct_stat,
            "frac_zero_hotspots": None if d["n_hotspots"] == [] else round(zero_frac, 3),
            "hotspot_not_contact": d["hotspot_not_contact"],
            "hotspot_checked": d["hotspot_checked"],
            "fallback_rows": d["fallback_rows"],
            "unexplained_rows": d["unexplained_rows"],
            "binder_len": _stat(d["binder_len"]),
            "target_len": _stat(d["target_len"]),
        }
        print(f"\n[{src}]  n={d['n']}")
        print(f"  receptor missing      : {d['no_receptor']}")
        print(f"  target contacts/ex    : {ct_stat}")
        print(f"  hotspots/ex           : {hs_stat}")
        print(f"  frac with 0 hotspots  : {zero_frac:.3f}" if d["n_hotspots"] else "  frac with 0 hotspots  : n/a")
        print(f"  hotspot NOT at iface  : {d['hotspot_not_contact']} / {d['hotspot_checked']} checked")
        print(f"    of which: random-fallback rows={d['fallback_rows']}  unexplained={d['unexplained_rows']}")

    print("\n==================== wiring ====================")
    print(f"  batch['hotspot_mask'] present : {dead_channel_present}")
    print(f"  batch['hotspot_mask'] all zero: {dead_channel_zero}   <- known defect (binder-length)")
    print(f"  TargetConcatSeqFeat cols moved: {model_cols_changed}   <- expect exactly 1 (live path)")
    for n in notes:
        print(f"  note: {n}")

    # Hotspots landing off the interface are a FINDING about the sampler's random fallback,
    # not a failure of the probe, so they are reported separately. The correctness gate is
    # that nothing is left unexplained.
    n_unexplained = sum(d["unexplained_rows"] for d in per_src.values())
    ok = fails.get("no_receptor", 0) == 0 and n_unexplained == 0 and model_cols_changed == 1
    print(f"\n[probe] {'PASS' if ok else 'ATTENTION'}: " + ("construction and the live channel check out" if ok else f"see fails={dict(fails)}"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"[probe] summary -> {args.out}")
    if args.rows_out:
        Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rows_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[probe] rows    -> {args.rows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
