"""Derive cyclic-peptide success thresholds from native-peptide calibration runs.

Reads the results of `scripts/run_cpsea_native_calibration.sh` and emits a
`success_thresholds` JSON that the real crystallographic peptides actually pass.

The reference point is deliberately the **native** structure, not the design
distribution. Setting thresholds from designs would define success as "typical of
what the model already produces", which cannot then be evidence that a change
improved anything. Natives give an external, model-independent ceiling: a bar no
real cyclic peptide can clear is a broken bar.

Each threshold is placed at the **worst** native across targets, loosened by a
margin, so every native passes. This is intentionally permissive -- its job is to
stop the criteria vetoing everything, not to be selective. Selectivity comes from
the metrics that do have headroom (scRMSD, ring closure, interface dG).

Usage:
    python script_utils/derive_cyclic_thresholds.py evaluation_results \
        --out configs/cyclic_thresholds.json
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

# (metric, column, comparison direction, scale) as consumed by
# `result_analysis.binder_analysis.filter_by_success_thresholds`.
METRIC_SPECS = [
    ("pLDDT", "self_complex_pLDDT", ">=", 1.0, "complex"),
    ("i_pAE", "self_complex_i_pAE", "<=", 31.0, "complex"),
    ("scRMSD_ca", "self_binder_scRMSD_ca", "<", 1.0, "binder"),
]

# Fractional slack applied away from the worst native, so a threshold sitting exactly
# on the observed value does not fail on floating-point noise or a rerun.
DEFAULT_MARGIN = 0.05


def load_calibration(results_dir: str, pattern: str) -> pd.DataFrame:
    """Concatenate every native-calibration results CSV matching `pattern`."""
    csvs = sorted(glob.glob(os.path.join(results_dir, pattern, "binder_results_*.csv")))
    if not csvs:
        return pd.DataFrame()
    frames = []
    for path in csvs:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as e:
            logger.warning("Skipping unreadable %s: %s", path, e)
            continue
        frame["calib_run"] = os.path.basename(os.path.dirname(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def derive_thresholds(df: pd.DataFrame, margin: float = DEFAULT_MARGIN) -> tuple[dict, dict]:
    """Return (thresholds, report). Metrics absent from `df` are skipped, not guessed."""
    thresholds: dict = {}
    report: dict = {}

    for metric, column, op, scale, prefix in METRIC_SPECS:
        if column not in df.columns:
            logger.warning("Column %s absent from calibration results; skipping %s.", column, metric)
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            logger.warning("No usable values for %s; skipping.", metric)
            continue

        if op == ">=":
            # Natives must clear the bar, so anchor on the WORST (lowest) native.
            worst = float(values.min())
            threshold = worst * (1.0 - margin)
        else:
            worst = float(values.max())
            threshold = worst * (1.0 + margin)

        thresholds[metric] = {
            "threshold": round(threshold * scale, 4),
            "op": op,
            "scale": scale,
            "column_prefix": prefix,
        }
        report[metric] = {
            "n_natives": int(len(values)),
            "native_min": round(float(values.min()), 4),
            "native_median": round(float(values.median()), 4),
            "native_max": round(float(values.max()), 4),
            "worst_native": round(worst, 4),
            "derived_threshold_raw": round(threshold, 4),
        }

    return thresholds, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="Directory holding native_calib_* result folders")
    parser.add_argument(
        "--pattern",
        default="native_calib_*_cycon",
        help="Glob for calibration run folders. Defaults to the cyclic-offset-ON runs, "
        "which are the only ones that folded the actual macrocycle.",
    )
    parser.add_argument("--out", default=None, help="Where to write the thresholds JSON")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_calibration(args.results_dir, args.pattern)
    if df.empty:
        logger.error(
            "No calibration results found under %s matching %r. Run "
            "scripts/run_cpsea_native_calibration.sh first.",
            args.results_dir,
            args.pattern,
        )
        return 1

    thresholds, report = derive_thresholds(df, margin=args.margin)
    if not thresholds:
        logger.error("Calibration results contained none of the expected metric columns.")
        return 1

    logger.info("Native calibration over %d rows from %d run(s):", len(df), df["calib_run"].nunique())
    for metric, stats in report.items():
        logger.info(
            "  %-10s natives[min=%.4f med=%.4f max=%.4f n=%d] -> threshold %s %.4f",
            metric,
            stats["native_min"],
            stats["native_median"],
            stats["native_max"],
            stats["n_natives"],
            thresholds[metric]["op"],
            thresholds[metric]["threshold"],
        )

    payload = {"success_thresholds": thresholds, "calibration_report": report}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Wrote %s", args.out)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
