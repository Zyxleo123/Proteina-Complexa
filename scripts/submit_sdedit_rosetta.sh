#!/bin/bash
# Submitter for the before/after Rosetta measurement on an SDEdit run. Calls sbatch only.
#
#   bash scripts/submit_sdedit_rosetta.sh --results-dir evaluation_results/<run>/sdedit \
#        [--arm control --arm projected] [--shards 8] [--partition cpu] [--all-edits]
#        [--only-closed] [--max-per-example N] [--metadata file.parquet] [--dry-run]
#
# Stage 1 CPU array  score before + after with Rosetta   (sdedit_rosetta.sbatch)
# Stage 2 CPU        figure + table                      (sdedit_rosetta_plot.sbatch)
#
# One timestamped env file, passed as a POSITIONAL argument -- never `sbatch --export`,
# which sets SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD.
#
# REQUIREMENT: the SDEdit arm must have been run with `--complex-pdb-dir`. The receptor is
# not in the peptide PDBs, and the CPSea loader's frame is redrawn per process, so a complex
# cannot be rebuilt after the fact -- an arm without it has to be re-run.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RESULTS_DIR=""; ARMS=(); PARTITION="cpu"; SHARDS=8; TIME="12:00:00"; PLOT_TIME="00:20:00"
ONLY_SCORABLE=1; ONLY_CLOSED=0; MAX_PER_EXAMPLE=""; METADATA=""; DRYRUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)     RESULTS_DIR="$2";     shift 2 ;;
    --arm)             ARMS+=("$2");         shift 2 ;;
    --partition)       PARTITION="$2";       shift 2 ;;
    --shards)          SHARDS="$2";          shift 2 ;;
    --time)            TIME="$2";            shift 2 ;;
    --metadata)        METADATA="$2";        shift 2 ;;
    --max-per-example) MAX_PER_EXAMPLE="$2"; shift 2 ;;
    # Abstained edits (the model proposed a different chemistry) are excluded by default:
    # their ring was never measured, so a dG for them answers a different question.
    --all-edits)       ONLY_SCORABLE=0;      shift ;;
    --only-closed)     ONLY_CLOSED=1;        shift ;;
    --dry-run)         DRYRUN=1;             shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$RESULTS_DIR" ]] || { echo "FATAL: --results-dir is required" >&2; exit 1; }
[[ -d "$RESULTS_DIR" ]] || { echo "FATAL: no such directory: $RESULTS_DIR" >&2; exit 1; }
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=("")

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="sdrosetta_${STAMP}"
mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_sdedit_rosetta.sh at ${STAMP}.
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)
ROSETTA_SHARDS=${SHARDS}
ROSETTA_ONLY_SCORABLE=${ONLY_SCORABLE}
ROSETTA_ONLY_CLOSED=${ONLY_CLOSED}
${MAX_PER_EXAMPLE:+ROSETTA_MAX_PER_EXAMPLE=${MAX_PER_EXAMPLE}}
${METADATA:+ROSETTA_METADATA=${METADATA}}
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

if [[ $DRYRUN == 1 ]]; then
  echo "--dry-run: resolving locally, submitting nothing."
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  for ARM in "${ARMS[@]}"; do
    EDIT_DIR="${RESULTS_DIR}${ARM:+/$ARM}"
    shopt -s nullglob; EDITS=("${EDIT_DIR}"/edits_shard*.jsonl); shopt -u nullglob
    if [[ ${#EDITS[@]} -eq 0 ]]; then echo "arm ${ARM:-<root>}: no edits under ${EDIT_DIR}"; continue; fi
    "${PYTHON_EXEC:-$REPO/.venv/bin/python}" scripts/score_sdedit_rosetta.py \
      --edits "${EDITS[@]}" --out /dev/null \
      --complex-dir "${EDIT_DIR}/complexes" \
      ${ROSETTA_METADATA:+--metadata "$ROSETTA_METADATA"} \
      $([[ $ONLY_SCORABLE == 1 ]] && echo --only-scorable) \
      $([[ $ONLY_CLOSED == 1 ]] && echo --only-closed) \
      --dry-run
  done
  exit 0
fi

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

for ARM in "${ARMS[@]}"; do
  LABEL="${ARM:-root}"
  SCORE_ID=$(sbatch --parsable \
    --partition="$PARTITION" --time="$TIME" \
    --array=0-$((SHARDS - 1)) --job-name="${RUN_ID}_${LABEL}" \
    scripts/sdedit_rosetta.sbatch "$ENV_FILE" "$ARM")
  echo "stage 1 CPU  score  arm=${LABEL}  job ${SCORE_ID} (${SHARDS} shard(s), ${PARTITION})"

  PLOT_ID=$(sbatch --parsable \
    --partition="$PARTITION" --time="$PLOT_TIME" \
    --dependency=afterok:"$SCORE_ID" --job-name="${RUN_ID}_${LABEL}_fig" \
    scripts/sdedit_rosetta_plot.sbatch "$ENV_FILE" "$ARM")
  echo "stage 2 CPU  figure arm=${LABEL}  job ${PLOT_ID} (afterok:${SCORE_ID})"
  echo "        -> $(cd "$RESULTS_DIR" && pwd)${ARM:+/$ARM}/rosetta/figures/rosetta_before_after.png"
done
