#!/bin/bash
#SBATCH --job-name=cpsea_valgen_summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Aggregates the JSONL the GPU jobs wrote. NO GPU: this only reads saved rows, and asking
# for a card to compute a mean just queues the table behind real work. Submitted with
# --dependency=afterok on the three scoring jobs by scripts/submit_val_gen_nsteps.sh, and
# re-runnable by hand any time the question changes, since it recomputes from disk.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }

# env.sh FIRST, the job's own env file SECOND -- the job config must win over the repo-wide
# exports in .env. The other order is what let .env shadow a variable in the scoring job and
# produce a silent no-op (job 36051).
source env.sh
source "$ENV_FILE"
: "${OUT_DIR:?OUT_DIR not set in $ENV_FILE}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

# Gate: distinguish "the scoring jobs ran and produced nothing" from "they never ran".
if ! compgen -G "$OUT_DIR/rows_*.jsonl" > /dev/null; then
  echo "GATE FAILED: no $OUT_DIR/rows_*.jsonl files."
  echo "The scoring jobs either never ran or wrote no rows -- check their .out files."
  exit 0
fi

BASE="${BASELINE_LABEL:-cpsea_cyc_ringpe_bond_v4}"

# Two views of the same rows, both cheap, both from disk.
echo "############ CONTROL: batches 0,1 only -- read the 'unwtd' column ############"
echo "# These are the only batches training's val_gen ever scored, and it logged an"
echo "# unweighted mean of per-batch rates. The nsteps=400 numbers here must match the"
echo "# training-logged val_gen closure. If they do not, stop: something other than"
echo "# nsteps moved, and nothing below is interpretable."
"$PY" script_utils/summarize_val_gen_nsteps.py \
  --rows "$OUT_DIR" --baseline "$BASE" --batches 0,1 \
  --json-out "$OUT_DIR/control_summary.json"

echo
echo "############ FULL: all batches -- the actual comparison ############"
"$PY" script_utils/summarize_val_gen_nsteps.py \
  --rows "$OUT_DIR" --baseline "$BASE" \
  --json-out "$OUT_DIR/summary.json"
