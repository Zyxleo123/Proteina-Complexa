#!/usr/bin/env python3
"""Non-GPU smoke checks for CPSea preprocess + training configs.

Usage:
  source env.sh
  python script_utils/test_cpsea_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def main() -> int:
    os.chdir(REPO)
    py = sys.executable
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)  # ensure +single path in train.py tests

    sample_root = REPO / "CPSea_data" / "CPSea_sample_100"
    sample_meta = REPO / "CPSea_data" / "preprocessed_sample100" / "metadata"
    if not (sample_meta / "cpsea_train.parquet").exists():
        print("Running preprocess on sample_100 (5 structures)...")
        run(
            [
                py,
                "script_utils/preprocess_cpsea.py",
                "--pdb-root",
                str(sample_root),
                "--out-dir",
                str(REPO / "CPSea_data" / "preprocessed_sample100_test"),
                "--index",
                "/dev/null",
                "--limit",
                "5",
            ],
            env=env,
        )

    for cfg in [
        "example/training_cpsea_peptide_smoke",
        "example/training_cpsea_peptide_smoke_from_scratch",
    ]:
        run(
            [
                py,
                "-m",
                "proteinfoundation.train",
                f"--config-name={cfg}",
                "--cfg",
                "job",
            ],
            env=env,
        )
        # grep-like sanity: composed config must be flat (not nested under example:)
        out = subprocess.check_output(
            [
                py,
                "-m",
                "proteinfoundation.train",
                f"--config-name={cfg}",
                "--cfg",
                "job",
            ],
            cwd=REPO,
            env=env,
            text=True,
        )
        if out.lstrip().startswith("example:"):
            raise SystemExit(f"Config {cfg} still nested under example: — add # @package _global_")
        if "run_name:" not in out and "run_name" not in out:
            raise SystemExit(f"Config {cfg} missing run_name in --cfg job output")

    print("\n>>> datamodule load test")
    import proteinfoundation.patches.atomworks_patches  # noqa: F401
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    import hydra
    from proteinfoundation.train import load_data_module

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        overrides = [
            "+single=true",
            "+nolog=true",
            f"++dataset.datamodule.metadata_file={sample_meta / 'cpsea_train.parquet'}",
            f"++dataset.datamodule.val_metadata_file={sample_meta / 'cpsea_val.parquet'}",
            "++dataset.datamodule.num_workers=0",
        ]
        for cfg_name in [
            "example/training_cpsea_peptide_smoke",
            "example/training_cpsea_peptide_smoke_from_scratch",
        ]:
            cfg = compose(config_name=cfg_name, overrides=overrides)
            if not hasattr(cfg, "hardware"):
                raise SystemExit(f"{cfg_name}: cfg.hardware missing (check @package _global_)")
            _, dm = load_data_module(cfg, is_cluster_run=False)
            dm.setup("fit")
            sample = dm.train_dataset[0]
            assert sample.binder_chain_id == "B"
            batch = next(iter(dm.train_dataloader()))
            assert "x_target" in batch and "chains" in batch
            print(f"  OK {cfg_name}: train={len(dm.train_dataset)} batch={batch['chains'].shape}")

    print("\nAll non-GPU CPSea smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
