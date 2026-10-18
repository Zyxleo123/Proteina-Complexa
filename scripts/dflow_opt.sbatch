#!/bin/bash
#SBATCH --job-name=dflow_opt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# GPU arm of D-Flow LP -> CP: optimize the source state of the frozen CP generator by
# differentiating through the whole rollout (scripts/dflow_cyclize.py).
#
# Runs as a job ARRAY. Each task takes one shard of the peptides and writes its OWN results
# file: concurrent O_APPEND has NUL-corrupted rows on this cluster (88 of 144 lost), so
# shards never share an output path.
#
# Resumable by construction: dflow_cyclize.py reads the shard file's run_keys on startup and
# skips what is already done, and MAX_SECONDS makes it stop cleanly rather than be killed at
# the time limit. Hitting the wall means resubmitting the same command, not restarting.
#
# Do NOT sbatch directly -- use scripts/submit_dflow.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/_dflow_preamble.sh" "$@"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${FLOW_CKPT_PATH:?FLOW_CKPT_PATH not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"

FLOW_CKPT_NAME="${FLOW_CKPT_NAME:-last-EMA.ckpt}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

dflow_require_file "${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}" "flow checkpoint"
dflow_require_file "${CPSEA_AE_CKPT_PATH}" "AE checkpoint"
dflow_require_file "${LNR_METADATA}" "LNR metadata"
dflow_preflight_python

mkdir -p "$RESULTS_DIR" "${PDB_DIR:-$RESULTS_DIR/pdbs}" "${COMPLEX_DIR:-$RESULTS_DIR/complexes}"

# An explicit example list pins WHICH peptides were measured as an artefact, rather than as
# a number recomputed at read time. An unknown id is FATAL inside dflow_cyclize.py.
EXAMPLES="${EXAMPLES:-}"
if [[ -n "${EXAMPLES_FILE:-}" ]]; then
  dflow_require_file "$EXAMPLES_FILE" "examples file"
  EXAMPLES="$(grep -v '^[[:space:]]*\(#.*\)\?$' "$EXAMPLES_FILE" | tr '\n' ' ')"
  [[ -n "${EXAMPLES// /}" ]] || { echo "FATAL: EXAMPLES_FILE is empty: $EXAMPLES_FILE"; exit 1; }
  echo "examples from ${EXAMPLES_FILE}: $(wc -w <<< "$EXAMPLES") id(s)"
fi

echo "shard ${SHARD_INDEX}/${SHARD_COUNT}"
echo "flow ckpt: ${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}"
echo "AE ckpt:   ${CPSEA_AE_CKPT_PATH}"

# Objective arms. One space-separated entry per arm; each is a `--loss` spec passed through
# verbatim (see parse_weights in scripts/dflow_losses.py). "-" means the script default.
# Arms run SEQUENTIALLY into the same shard file -- run_key carries the objective tag, so
# resume works per arm and a resubmitted job skips whatever already finished.
OBJECTIVES="${OBJECTIVES:--}"

for OBJ in ${OBJECTIVES}; do
  LOSS_ARG=()
  [[ "$OBJ" != "-" ]] && LOSS_ARG=(--loss "$OBJ")
  echo
  echo "=== objective: ${OBJ}"

  # Unquoted on purpose: space-separated lists that must word-split into argv.
  # shellcheck disable=SC2086
  srun "$DFLOW_PY" scripts/dflow_cyclize.py \
    --ckpt-path "$FLOW_CKPT_PATH" \
    --ckpt-name "$FLOW_CKPT_NAME" \
    --metadata "$LNR_METADATA" \
    --out "${RESULTS_DIR}/dflow_shard${SHARD_INDEX}.jsonl" \
    --pdb-dir "${PDB_DIR:-$RESULTS_DIR/pdbs}" \
    --complex-pdb-dir "${COMPLEX_DIR:-$RESULTS_DIR/complexes}" \
    --cyc-types ${CYC_TYPES:-mainchain disulfide isopeptide} \
    --init-mode "${INIT_MODE:-source}" \
    --t-ca "${T_CA:-0.8}" \
    --t-lat "${T_LAT:-0.4}" \
    --sampler-mode "${SAMPLER_MODE:-vf}" \
    --opt-nsteps "${OPT_NSTEPS:-30}" \
    --eval-nsteps "${EVAL_NSTEPS:-0}" \
    --restarts "${RESTARTS:-16}" \
    --opt-steps "${OPT_STEPS:-60}" \
    --lr-ca "${LR_CA:-0.02}" \
    --lr-lat "${LR_LAT:-0.05}" \
    --lr-decay "${LR_DECAY:-1.0}" \
    --grad-clip "${GRAD_CLIP:-1.0}" \
    --seed "${SEED:-0}" \
    --log-every "${LOG_EVERY:-10}" \
    --limit "${PEPTIDE_LIMIT:-0}" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --max-seconds "${MAX_SECONDS:-0}" \
    "${LOSS_ARG[@]}" \
    ${PROTECT:+--protect ${PROTECT}} \
    ${PROTECT_W:+--protect-w "$PROTECT_W"} \
    ${Z_DIR:+--z-dir "$Z_DIR"} \
    ${EXAMPLES:+--examples ${EXAMPLES}}
done

echo "DONE shard ${SHARD_INDEX}"
