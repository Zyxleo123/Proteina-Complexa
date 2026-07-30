"""Materialize the 96 eval-set native complexes into the layout calibrate_rosetta_dg.py reads.

`calibrate_rosetta_dg.py` globs `calibration_native/<task_name>/processed/*.pdb` and derives the
per-target key from the directory name (see its `target_from_native_path`). phase0_baseline.py then
joins a design row to its native bar on `task_name`, so the directory MUST be named exactly the
eval-config key (e.g. `cpsea_eval_6CCA_A_145_150_relaxed`).

The eval natives at `target_path` are already in the receptor=A / binder=B convention the harness
expects (verified: 6CCA -> chain A 103 res receptor, chain B 6 res peptide == its binder_length),
so this is a symlink, not a re-processing step. Symlink (not copy) keeps calibration_native/ small
and makes the source-of-truth obvious.

    python script_utils/link_evalset_natives.py \
        --config configs/targets/cpsea_eval_targets.yaml --root calibration_native

Idempotent: an existing correct symlink is left alone; a wrong one is replaced.
"""

import argparse
import logging
import os
import sys

import yaml

logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/targets/cpsea_eval_targets.yaml")
    parser.add_argument("--root", default="calibration_native")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(open(args.config))["target_dict_cfg"]
    linked, missing = 0, []
    for task_name, entry in cfg.items():
        src = entry["target_path"]
        if not os.path.isfile(src):
            missing.append((task_name, src))
            continue
        dst_dir = os.path.join(args.root, task_name, "processed")
        dst = os.path.join(dst_dir, os.path.basename(src))
        if args.dry_run:
            logger.info("would link %s -> %s", dst, src)
            linked += 1
            continue
        os.makedirs(dst_dir, exist_ok=True)
        # Replace only if the existing link points elsewhere; leave a correct one untouched.
        if os.path.islink(dst) or os.path.exists(dst):
            if os.path.realpath(dst) == os.path.realpath(src):
                linked += 1
                continue
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        linked += 1

    logger.info("linked %d/%d eval natives into %s", linked, len(cfg), args.root)
    if missing:
        # A missing native silently drops that target's bar downstream, so fail loudly.
        logger.error("%d native PDBs missing on disk:", len(missing))
        for name, src in missing[:20]:
            logger.error("  %s -> %s", name, src)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
