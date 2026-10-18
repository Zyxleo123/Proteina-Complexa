#!/bin/bash
#SBATCH --job-name=stage0_select
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 0, step 1: choose WHICH complexes the cost profile runs on, and write that choice
# down as an artefact.
#
# The plan calls for "about 16 production-setting generations across small and large pockets".
# Pocket size is the cost axis that actually varies here -- LNR pocket crops run 57 to 287
# receptor residues, a 5x spread -- so a mean seconds-per-structure taken over an arbitrary
# handful is not a number Stage 1 can be sized from. This job takes the N smallest and N
# largest receptors by `receptor_length` and records both the ids and their lengths, so the
# report can quote cost AT a pocket size rather than averaged over an unstated mixture.
#
# Selection lives in its own job, not in the submitter, because the submitter only calls
# sbatch; and not in the GPU job, because a 60-row parquet read has no business holding a GPU.
# The written file is also what makes the profile re-runnable against the same peptides after
# the metadata is restaged.
#
# Do NOT sbatch directly -- use scripts/submit_stage0_profile.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
: "${EXAMPLES_FILE:?EXAMPLES_FILE not set}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

mkdir -p "$(dirname "$EXAMPLES_FILE")"
echo "Node: $(hostname 2>/dev/null || echo unknown)"
echo "metadata: ${LNR_METADATA}"
echo "n_small=${PROFILE_N_SMALL:-4}  n_large=${PROFILE_N_LARGE:-4}"

"$PY" - "$LNR_METADATA" "$EXAMPLES_FILE" "${PROFILE_N_SMALL:-4}" "${PROFILE_N_LARGE:-4}" <<'PYEOF'
import json
import sys
from pathlib import Path

import pandas as pd

meta_path, out_path, n_small, n_large = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
df = pd.read_parquet(meta_path)
for col in ("example_id", "receptor_length", "peptide_length"):
    if col not in df.columns:
        raise SystemExit(f"FATAL: {meta_path} has no `{col}` column; cannot size the profile by pocket.")

if len(df) < n_small + n_large:
    raise SystemExit(f"FATAL: {meta_path} has {len(df)} rows, fewer than the {n_small + n_large} asked for.")

# Sort by pocket size, then take both ends. Ties broken by example_id so the selection is
# deterministic across pandas versions -- a profile whose case list silently changes between
# runs cannot be compared with itself.
df = df.sort_values(["receptor_length", "example_id"]).reset_index(drop=True)
small = df.head(n_small)
large = df.tail(n_large)
picked = pd.concat([small, large])

if len(set(picked["example_id"])) != len(picked):
    raise SystemExit("FATAL: the small and large ends overlap; ask for fewer cases per end.")

Path(out_path).write_text(
    "# Stage 0 cost-profile cases: the {} smallest and {} largest pockets in\n"
    "# {}\n".format(n_small, n_large, meta_path)
    + "".join(f"{r.example_id}\n" for r in picked.itertuples())
)

record = {
    "metadata": meta_path,
    "n_small": n_small,
    "n_large": n_large,
    "receptor_length_all": {
        "min": int(df.receptor_length.min()),
        "median": float(df.receptor_length.median()),
        "max": int(df.receptor_length.max()),
    },
    "cases": [
        {
            "example_id": r.example_id,
            "receptor_length": int(r.receptor_length),
            "peptide_length": int(r.peptide_length),
            "pocket_end": "small" if i < n_small else "large",
        }
        for i, r in enumerate(picked.itertuples())
    ],
}
Path(out_path).with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

print(f"wrote {len(picked)} cases -> {out_path}")
for c in record["cases"]:
    print(f"  {c['pocket_end']:>5}  {c['example_id']:<18} receptor={c['receptor_length']:>4}  peptide={c['peptide_length']:>3}")
PYEOF

echo "DONE stage0 selection"
