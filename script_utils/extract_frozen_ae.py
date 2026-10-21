#!/usr/bin/env python3
"""Recover a run's frozen autoencoder from its own flow checkpoint.

Some runs pinned their autoencoder by COPYING it into the run directory
(`ae_mode: frozen_pinned`) and recording where it came from in
`frozen_ae_source.txt`. For `cpsea_cyc_ringpe_bond_v4` -- the checkpoint behind the
paper's `batch_design_manifest_v4_paper_rosetta_cyc.jsonl` -- both the copy and the
pointer were lost in the cluster move: neither the local store nor the B2 bucket has
`frozen_ae.ckpt`, so every loader that resolves `autoencoder_ckpt_path` fails and the run
looks unrecoverable.

It is not. A frozen autoencoder is still a submodule of the Lightning module, so its
weights are saved inside the flow checkpoint's own `state_dict` under the `autoencoder.`
prefix. Measured on v4's `last-EMA.ckpt`: 765 `autoencoder.*` tensors, against exactly 765
tensors (381 `encoder.*` + 384 `decoder.*`) in a surviving standalone
`shared_ae_lpcp_128/frozen_ae.ckpt`. Stripping the prefix maps one onto the other.

So this is a **recovery, not a substitution**: the autoencoder written out here is the one
that run actually used, taken from that run's own weights. That distinction matters --
pointing v4 at another run's autoencoder on the strength of a config diff would be an
unprovable guess about which file the copy had been made from, and the latent space the
flow was trained against is exactly what an autoencoder swap would change.

The extraction is verified against a reference standalone autoencoder before anything is
written: identical key sets and identical tensor shapes, or it refuses.

Usage:
    python script_utils/extract_frozen_ae.py \
        --flow-ckpt .../cpsea_cyc_ringpe_bond_v4/checkpoints/last-EMA.ckpt \
        --reference .../shared_ae_lpcp_128/frozen_ae.ckpt \
        --out .../cpsea_cyc_ringpe_bond_v4/frozen_ae.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PREFIX = "autoencoder."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-ckpt", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None,
                    help="a surviving standalone frozen_ae.ckpt to verify key names and "
                         "shapes against. Strongly recommended: without it the extraction "
                         "is unchecked.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"[extract] {args.out} exists; refusing without --force")
        return 1

    ck = torch.load(args.flow_ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("state_dict", ck)
    ae = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    if not ae:
        print(f"[extract] no {PREFIX}* tensors in {args.flow_ckpt} -- this checkpoint does "
              f"not carry its autoencoder, so it cannot be recovered this way")
        return 1
    print(f"[extract] {len(ae)} autoencoder tensor(s) from {args.flow_ckpt.name}")
    print(f"[extract] prefixes: {Counter(k.split('.')[0] for k in ae).most_common()}")

    if args.reference is not None:
        ref_ck = torch.load(args.reference, map_location="cpu", mmap=True, weights_only=False)
        ref = ref_ck.get("state_dict", ref_ck)
        missing = sorted(set(ref) - set(ae))
        extra = sorted(set(ae) - set(ref))
        if missing or extra:
            print(f"[extract] REFUSING: key sets differ from {args.reference.name} -- "
                  f"{len(missing)} missing, {len(extra)} extra")
            for k in (missing[:5] + extra[:5]):
                print(f"    {k}")
            return 1
        bad = [k for k in ref if tuple(ref[k].shape) != tuple(ae[k].shape)]
        if bad:
            print(f"[extract] REFUSING: {len(bad)} tensor shape mismatch(es), e.g. "
                  f"{bad[0]}: {tuple(ae[bad[0]].shape)} vs {tuple(ref[bad[0]].shape)}")
            return 1
        print(f"[extract] verified against {args.reference.name}: "
              f"{len(ref)} keys, all shapes identical")

    # The envelope a loader actually needs. `AutoEncoder.__init__` takes `cfg_ae`, so a
    # state_dict alone fails with
    # `TypeError: AutoEncoder.__init__() missing 1 required positional argument: 'cfg_ae'`.
    #
    # The flow checkpoint does NOT carry the autoencoder's config -- its hparams hold only
    # `cfg_exp`, `store_dir` and `autoencoder_ckpt_path`, i.e. a POINTER to the config that
    # lived in the lost file. So `cfg_ae` is taken from the reference standalone
    # autoencoder, and that is sound for exactly one reason, already checked above: all
    # 765 tensors match the reference in name AND shape, so the two share an architecture.
    # A config describing any other architecture could not have passed that check.
    #
    # What this does and does not claim: the WEIGHTS are this run's own. `cfg_ae` is a
    # construction recipe, and only its `nn_ae` section is load-bearing at inference. Its
    # other sections (dataset, opt, loss, seed, run_name) describe how the REFERENCE
    # autoencoder was trained and are not this run's provenance -- they are carried only
    # because the loader expects a whole config object.
    cfg_ae = None
    if args.reference is not None:
        ref_hp = ref_ck.get("hyper_parameters") or {}
        cfg_ae = ref_hp.get("cfg_ae")
    if cfg_ae is None:
        print("[extract] REFUSING: no cfg_ae available. Pass --reference pointing at a "
              "surviving standalone frozen_ae.ckpt whose hyper_parameters carry one; "
              "without it the written checkpoint cannot be constructed by the loader.")
        return 1

    out_ck = {
        "state_dict": {k: v.clone() for k, v in ae.items()},
        "hyper_parameters": {"cfg_ae": cfg_ae, "store_dir": str(args.out.parent)},
        "hparams_name": "kwargs",
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
        "pytorch-lightning_version": ck.get("pytorch-lightning_version"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, args.out)
    size_gb = args.out.stat().st_size / 1e9
    print(f"[extract] wrote {args.out} ({size_gb:.2f} GB)")

    # Reinstate the provenance pointer the pinning convention used, so the next reader is
    # not left guessing again.
    src = args.out.parent / "frozen_ae_source.txt"
    src.write_text(json.dumps({
        "recovered_from": str(args.flow_ckpt),
        "method": "autoencoder.* submodule extracted from the flow checkpoint's state_dict",
        "n_tensors": len(ae),
        "verified_against": str(args.reference) if args.reference else None,
        "weights_provenance": "this run's own flow checkpoint (autoencoder.* submodule)",
        "cfg_ae_provenance": str(args.reference) if args.reference else None,
        "note": "The WEIGHTS are the autoencoder THIS RUN used, taken from this run's own "
                "checkpoint -- not another run's autoencoder substituted in. The original "
                "pinned copy and its frozen_ae_source.txt were lost in the cluster move. "
                "`cfg_ae` is borrowed from the reference because the flow checkpoint does "
                "not store the autoencoder's config, and is valid as a construction recipe "
                "because all tensors matched the reference in name and shape. Only its "
                "nn_ae section is load-bearing; its dataset/opt/loss/run_name sections "
                "describe the REFERENCE autoencoder's training, not this run's.",
    }, indent=2) + "\n")
    print(f"[extract] provenance -> {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
