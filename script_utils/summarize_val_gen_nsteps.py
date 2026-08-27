"""Aggregate `eval_val_gen_nsteps.py` rows into a paired arm-vs-baseline table.

CPU only -- reads JSONL and nothing else, so the table can be regenerated as often as the
question changes without touching a GPU.

THE COMPARISON THIS SUPPORTS
----------------------------
Rows share their noise draw across arms: at a fixed (nsteps, seed, batch) every arm
integrated from the same x_0 on the same complexes. So the arm-vs-baseline difference is
PAIRED, and the pairing is what buys the power -- the between-unit spread (which complexes
landed in batch 3, how hard that seed's noise was) cancels in the difference instead of
being carried into the error bar. The five-arm training-time comparison could not do this:
its validations never shared a draw.

Two rates are reported per condition:

  pooled  = sum(success_i * n_valid_i) / sum(n_valid_i)   -- the closure rate over all
            rings scored, which is the quantity of interest.
  paired  = mean over units of (arm_pooled_unit - baseline_pooled_unit), +- SEM over units.
            Units with no rings of that chemistry on either side are dropped, not zeroed.

READING THE ANSWER
------------------
The question is whether arm B's null is an artefact of scoring at nsteps=400 when it
trained its rollouts at 100. That is answered by the SIGN CHANGE, not by either number
alone: if B - v4 is ~0 at 400 (reproducing the ablation) and clearly positive at 100, the
fidelity gap is real and the arm was measured with the wrong instrument. If B - v4 is ~0 at
both, the fidelity gap is dead as an explanation and arm B's null is about the method.

Also check that the 400-step row REPRODUCES the training-logged val_gen number for each
arm. It is the control. If it does not, something else moved (weights, AE, val split) and
no comparison in this table means anything yet -- but reproduce it CORRECTLY:

    --batches 0,1   and read the `unwtd` column

Training scored n_batches=2 and logged an unweighted mean of the two per-batch rates. The
val loader is unshuffled, so batches 0 and 1 are the same complexes every time, and batches
2+ are complexes training never scored. Measured on the smoke row: batch 0 alone carries
mainchain n=8 against the 18.0 that training logged as its mean -- the composition varies
hugely batch to batch, so a pooled rate over 8 batches will differ from the logged number
for reasons that have nothing to do with the model. The ARM COMPARISON is unaffected: every
arm sees the same batches, so the pairing holds either way.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from glob import glob
from pathlib import Path

LINKAGES = [
    ("mainchain", "val_gen/cyc/mainchain_cn_bond_success", "val_gen/cyc/n_valid_mainchain"),
    ("disulfide", "val_gen/cyc/disulfide_bond_success", "val_gen/cyc/n_valid_disulfide"),
    ("isopeptide", "val_gen/cyc/isopeptide_bond_success", "val_gen/cyc/n_valid_isopeptide"),
]


def _resolve_row_files(spec: str) -> list[Path]:
    """Accept a directory (all *.jsonl in it), a glob, or a single file."""
    p = Path(spec)
    if p.is_dir():
        return sorted(f for f in p.glob("*.jsonl") if not f.name.startswith("smoke"))
    if any(ch in spec for ch in "*?["):
        return sorted(Path(x) for x in glob(spec))
    return [p] if p.exists() else []


def _iter_lines(files: list[Path]):
    for f in files:
        with f.open(errors="replace") as fh:
            yield from fh


def _grid_report(data: dict) -> None:
    """Print rows-per-condition. A ragged grid is the visible symptom of lost rows, and it
    also breaks the pooled columns: pooling over different unit sets per arm compares
    different complexes. The paired column intersects units and stays honest either way."""
    counts = {k: len(v) for k, v in sorted(data.items())}
    if not counts:
        return
    n = set(counts.values())
    print("\nUnits per (arm, nsteps): " + ", ".join(f"{k[0].split('_')[1][:8]}@{k[1]}={v}"
                                                     for k, v in counts.items()))
    if len(n) > 1:
        print("  ^ RAGGED: arms did not complete the same grid. `pooled`/`unwtd` are then NOT\n"
              "    comparable across arms (different complexes). Read only `paired d`, or rescore.")


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=str, default="evaluation_results/val_gen_nsteps",
                    help="A directory of per-arm *.jsonl row files (preferred), a glob, or a single "
                         "file. One file per arm is the contract -- see --allow-corrupt.")
    ap.add_argument("--allow-corrupt", action="store_true",
                    help="Proceed despite unreadable lines. OFF by default and it should stay off: "
                         "silently skipping corrupt lines is exactly how a shared-file run produced "
                         "a complete-looking table from 39%% of its rows.")
    ap.add_argument("--baseline", type=str, default="cpsea_cyc_ringpe_bond_v4",
                    help="Arm label to difference every other arm against.")
    ap.add_argument("--batches", type=str, default=None,
                    help="Restrict to these batch indices, e.g. --batches 0,1. USE THIS FOR THE "
                         "nsteps=400 CONTROL: training's val_gen scored only n_batches=2, so batches "
                         "2+ contain complexes it never saw and a pooled rate over all 8 is not the "
                         "logged number even when the model is identical.")
    ap.add_argument("--json-out", type=str, default=None, help="Optional machine-readable summary.")
    args = ap.parse_args()

    keep_batches = ({int(x) for x in args.batches.split(",") if x.strip()}
                    if args.batches else None)
    files = _resolve_row_files(args.rows)
    if not files:
        print(f"No row files at {args.rows}")
        return 1
    print("Reading: " + ", ".join(f.name for f in files))

    # (label, nsteps) -> unit (seed, batch) -> {linkage: (success, n_valid)}
    data: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    steps_seen: dict[str, set] = defaultdict(set)
    corrupt, dupes = 0, 0
    row_keys: set[tuple] = set()
    for line in _iter_lines(files):
        line = line.strip()
        if not line:
            continue
        if "\x00" in line:
            corrupt += 1
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            corrupt += 1
            continue
        if keep_batches is not None and row["batch_idx"] not in keep_batches:
            continue
        m = row["metrics"]
        unit = (row["seed"], row["batch_idx"])
        rec = {}
        for name, skey, nkey in LINKAGES:
            s, n = m.get(skey), m.get(nkey)
            if _finite(s) and _finite(n) and n > 0:
                rec[name] = (float(s), float(n))
        # Last-write-wins dedupe. Two jobs for the same arm that started before either had
        # written will both compute the same (seed, batch) units; the rows are identical by
        # construction (same weights, same per-batch seed), so collapsing them is safe and
        # keeps duplicate submissions from double-counting units in the paired statistics.
        key = (row["label"], row["ckpt_name"], row["nsteps"], row["seed"], row["batch_idx"])
        if key in row_keys:
            dupes += 1
        row_keys.add(key)
        data[(row["label"], row["nsteps"])][unit] = rec
        steps_seen[row["label"]].add(row.get("ckpt_global_step"))

    if dupes:
        print(f"Note: collapsed {dupes} duplicate row(s) -- the same (arm, nsteps, seed, batch) "
              f"scored more than once. Harmless (identical by construction), but it means the "
              f"scoring jobs were submitted more than once.")
    if corrupt:
        msg = (f"\n*** {corrupt} UNREADABLE LINE(S) across {len(files)} file(s). ***\n"
               "Rows were destroyed on disk -- most likely two writers appended to one file\n"
               "(concurrent O_APPEND is not atomic here; collisions NUL-pad the loser).\n"
               "Any table built from what survived is a biased subsample, not a result.\n"
               "Give each arm its own --out path and rescore.")
        if not args.allow_corrupt:
            print(msg + "\nRefusing to aggregate. Pass --allow-corrupt to override (do not).")
            return 2
        print(msg + "\n--allow-corrupt set: continuing on partial data. DO NOT cite this table.")

    labels = sorted({k[0] for k in data})
    nsteps_list = sorted({k[1] for k in data})
    _grid_report(data)
    print("Checkpoint step per arm: " + ", ".join(f"{lab}={sorted(steps_seen[lab])}" for lab in labels))

    summary = []
    for nsteps in nsteps_list:
        print(f"\n{'='*78}\nnsteps = {nsteps}\n{'='*78}")
        for name, _, _ in LINKAGES:
            print(f"\n  {name}")
            print(f"    {'arm':<38}{'pooled':>9}{'unwtd':>8}{'rings':>8}"
                  f"{'paired d':>11}{'SEM':>8}{'z':>7}{'n':>5}")
            base_units = data.get((args.baseline, nsteps), {})
            for lab in labels:
                units = data.get((lab, nsteps), {})
                if not units:
                    continue
                num = sum(s * n for u in units.values() if name in u for s, n in [u[name]])
                den = sum(n for u in units.values() if name in u for _, n in [u[name]])
                pooled = num / den if den else float("nan")
                # `unwtd` is the UNWEIGHTED mean of per-batch rates. That -- not `pooled` -- is
                # what training logged: log_nan_safe feeds MeanMetric(nan_strategy="ignore") one
                # value per batch with no weight, so the wandb number averages batch rates, not
                # rings. Compare `unwtd` (with --batches 0,1) against the logged value; use
                # `pooled` for everything else, since it is the honest ring-level rate.
                rates = [u[name][0] for u in units.values() if name in u]
                unwtd = sum(rates) / len(rates) if rates else float("nan")
                line = f"    {lab:<38}{pooled:>9.3f}{unwtd:>8.3f}{den:>8.0f}"
                if lab != args.baseline and base_units:
                    diffs = [
                        units[u][name][0] - base_units[u][name][0]
                        for u in units
                        if u in base_units and name in units[u] and name in base_units[u]
                    ]
                    if len(diffs) >= 2:
                        d = sum(diffs) / len(diffs)
                        var = sum((x - d) ** 2 for x in diffs) / (len(diffs) - 1)
                        sem = math.sqrt(var / len(diffs))
                        # A SEM this small means the paired differences were effectively
                        # constant across units; z is then a division-by-noise artefact
                        # (it prints as 1e16 and blows the column), not evidence.
                        z = d / sem if sem > 1e-9 else float("nan")
                        zs = f"{max(-99.9, min(99.9, z)):>+7.1f}" if _finite(z) else f"{'--':>7}"
                        line += f"{d:>+11.3f}{sem:>8.3f}{zs}{len(diffs):>5}"
                        summary.append({"nsteps": nsteps, "linkage": name, "arm": lab,
                                        "pooled": pooled, "unweighted": unwtd,
                                        "paired_delta": d, "sem": sem,
                                        "z": z, "n_units": len(diffs)})
                    else:
                        line += f"{'(unpaired)':>11}"
                print(line)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
