"""Validate Rosetta interface dG as a success criterion, by scoring native complexes.

Motivation
----------
AF2 confidence was just disproven for this molecule class: after calibrating its
thresholds against native structures, 280/298 designs passed (94%), versus 0/298
before -- neither number distinguishes one model from another. The failure was only
visible because the natives were scored through the same harness.

Rosetta dG is the proposed replacement. It has NOT had that check. This script runs
it, so we learn whether dG has resolution *before* an ablation is built on it.

Method
------
Natives are scored with `compute_rosetta_interface_metrics_single` -- the identical
function, chains convention and relax setting the designs went through -- so the
comparison cannot be an artifact of differing Rosetta treatment. That control matters:
the reason AF2 scRMSD looked healthy on designs and terrible on natives is precisely
that the two were not treated alike.

Both the relaxed and unrelaxed dG are reported. FastRelax on an already-relaxed
crystal structure is close to idempotent but not exactly so, and the natives here
arrive pre-relaxed while the designs did not. If the verdict is the same under both
columns, that residual difference does not drive the conclusion.

Interpretation
--------------
The comparison that matters is *within a target*: dG scales with interface size, so
pooling targets would compare interfaces, not models. Designs exist for 1E2T only, so
that is the controlled test; the other natives are reported for context and
normalized by buried area (dG per 100 A^2 dSASA) to make them roughly comparable.

    natives clearly better  -> dG discriminates; usable as the success criterion
    indistinguishable       -> no resolution either; a much bigger problem
    natives WORSE           -> same pathology as AF2 scRMSD; measuring prep, not molecules
"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NATIVE_GLOB = "calibration_native/*/processed/*.pdb"
DESIGN_GLOB = "evaluation_results/search_binder_local_pipeline_cpsea_*/binder_results_*.csv"

DG_COL = "generated_binder_rosetta_dG_separated"
DG_NORELAX_COL = "generated_binder_rosetta_dG_separated_norelax"
DSASA_COL = "generated_binder_rosetta_dSASA_int"

# Natives are laid out receptor=A, binder=B (verified across all five files).
NATIVE_BINDER_CHAIN = "B"
NATIVE_TARGET_CHAINS = ["A"]


def target_from_native_path(path: str) -> str:
    """calibration_native/cpsea_1E2T/processed/... -> cpsea_1E2T"""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def score_natives(paths: list[str]) -> pd.DataFrame:
    from proteinfoundation.evaluation.rosetta_energy import compute_rosetta_interface_metrics_single

    records = []
    for path in paths:
        target = target_from_native_path(path)
        logger.info("Scoring native %s (%s)", target, os.path.basename(path))
        metrics = compute_rosetta_interface_metrics_single(
            pdb_path=path,
            binder_chain=NATIVE_BINDER_CHAIN,
            target_chains=NATIVE_TARGET_CHAINS,
        )
        # Re-key to the `generated_` namespace the design CSVs use, so both sides of the
        # comparison are read through one set of column names.
        row = {f"generated_{k}": v for k, v in metrics.items()}
        row["task_name"] = target
        row["pdb_path"] = path
        records.append(row)
    return pd.DataFrame(records)


def load_designs(design_glob: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(design_glob)):
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError) as e:
            logger.warning("Skipping unreadable %s: %s", path, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def compare(natives: pd.DataFrame, designs: pd.DataFrame) -> dict:
    """Per-target native-vs-design comparison on dG (lower = tighter binding)."""
    report: dict = {}

    for col, label in ((DG_COL, "relaxed"), (DG_NORELAX_COL, "norelax")):
        per_target = {}
        for target in sorted(natives["task_name"].unique()):
            native_vals = _num(natives[natives["task_name"] == target], col)
            if native_vals.empty:
                continue
            native_dg = float(native_vals.iloc[0])

            entry = {"native_dG": round(native_dg, 3)}

            if "task_name" in designs.columns:
                design_vals = _num(designs[designs["task_name"] == target], col)
            else:
                design_vals = pd.Series(dtype=float)

            if not design_vals.empty:
                # Fraction of designs at least as tight as the native. Around 0.5 means
                # dG cannot tell designs from the real thing.
                frac_better = float((design_vals <= native_dg).mean())
                entry.update(
                    {
                        "n_designs": int(len(design_vals)),
                        "design_dG_median": round(float(design_vals.median()), 3),
                        "design_dG_best": round(float(design_vals.min()), 3),
                        "design_dG_worst": round(float(design_vals.max()), 3),
                        "frac_designs_at_least_native": round(frac_better, 4),
                    }
                )
            per_target[target] = entry
        report[label] = per_target

    # Cross-target context only: normalize by buried area, since dG scales with interface size.
    native_dsasa = _num(natives, DSASA_COL)
    native_dg = _num(natives, DG_COL)
    if len(native_dsasa) == len(native_dg) and not native_dg.empty:
        per_area = (native_dg.values / np.maximum(native_dsasa.values, 1e-6)) * 100.0
        report["native_dG_per_100A2"] = {
            t: round(float(v), 4) for t, v in zip(natives["task_name"], per_area, strict=False)
        }

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native-glob", default=NATIVE_GLOB)
    parser.add_argument("--design-glob", default=DESIGN_GLOB)
    parser.add_argument("--out", default=None, help="Write the report as JSON")
    parser.add_argument("--natives-csv", default=None, help="Also dump raw native Rosetta metrics")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from proteinfoundation.evaluation.rosetta_energy import is_pyrosetta_available

    if not is_pyrosetta_available():
        logger.error("PyRosetta is not available; cannot calibrate dG.")
        return 1

    native_paths = sorted(glob.glob(args.native_glob))
    if not native_paths:
        logger.error("No native structures matched %r", args.native_glob)
        return 1

    natives = score_natives(native_paths)
    if args.natives_csv:
        natives.to_csv(args.natives_csv, index=False)
        logger.info("Wrote %s", args.natives_csv)

    designs = load_designs(args.design_glob)
    logger.info("Loaded %d design rows", len(designs))

    report = compare(natives, designs)

    print("\n=== Rosetta dG: native vs design (lower = tighter binding) ===")
    for label in ("relaxed", "norelax"):
        print(f"\n-- dG_separated [{label}] --")
        for target, entry in report.get(label, {}).items():
            if "n_designs" in entry:
                print(
                    f"  {target}: native {entry['native_dG']:>9.2f} | "
                    f"designs n={entry['n_designs']:<4d} "
                    f"median {entry['design_dG_median']:>9.2f} best {entry['design_dG_best']:>9.2f} | "
                    f"{entry['frac_designs_at_least_native'] * 100:.1f}% of designs >= native"
                )
            else:
                print(f"  {target}: native {entry['native_dG']:>9.2f} | (no designs for this target)")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
