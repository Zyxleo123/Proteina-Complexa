#!/bin/bash
#SBATCH --job-name=sdedit_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# GPU arm of the track-asymmetric SDEdit sweep (see scripts/sdedit_cyclize.py).
# Runs as a job ARRAY: each task takes one shard of the input peptides and writes its OWN
# results file. Shards never share an output path -- concurrent O_APPEND has NUL-corrupted
# rows on this cluster.
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_sweep.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${FLOW_CKPT_PATH:?FLOW_CKPT_PATH not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
FLOW_CKPT_NAME="${FLOW_CKPT_NAME:-last-EMA.ckpt}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable"; exit 1; }
[[ -f "${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}" ]] || { echo "FATAL: flow ckpt missing: ${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}"; exit 1; }
[[ -f "${CPSEA_AE_CKPT_PATH}" ]] || { echo "FATAL: AE ckpt missing: ${CPSEA_AE_CKPT_PATH}"; exit 1; }
[[ -f "${LNR_METADATA}" ]] || { echo "FATAL: LNR metadata missing: ${LNR_METADATA} (run the AE round-trip job first, it stages LNR)"; exit 1; }

mkdir -p "$RESULTS_DIR"
echo "shard ${SHARD_INDEX}/${SHARD_COUNT}"
echo "flow ckpt: ${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}"
echo "AE ckpt:   ${CPSEA_AE_CKPT_PATH}"

# Guidance arms. One space-separated entry per arm, fields separated by "|":
#
#     <lambda>|<loss spec>|<schedule>|<schedule pow>|<exclude termini>|<mode>[|<stride>][|<graft>]
#
# "-" in any field means "the script default"; lambda 0 is the UNGUIDED arm and runs the
# sampler with no hook at all. The default below is a single unguided arm, so an env file
# written before guidance existed reproduces the old single-pass behaviour exactly. The
# 7th field (stride) and 8th (graft) are optional for the same reason: a 6-field arm still
# parses, and neither default is stamped into the run_key, so old rows resume unchanged.
#
# graft=1 sets the input peptide's two terminal residues to the requested chemistry's anchors
# BEFORE encoding (scripts/anchor_graft.py), which removes head abstention at the source. The
# new residue gets a COMPLETE side chain at a standard rotamer (scripts/rotamer.py); the
# earlier backbone+CB truncation decoded back as alanine on every grafted endpoint, because
# [N, CA, C, CB, O] is exactly alanine's atom set. GRAFT_SIDECHAIN=truncate restores that as a
# negative control, and carries a different run tag so the two cannot resume onto each other.
# Making it an ARM field rather than a job-level flag is deliberate: the grafted and ungrafted
# runs then share peptides, seeds and job, so the comparison is paired instead of cross-run.
#
# Arms are run SEQUENTIALLY into the same shard file: run_key carries the guidance settings,
# so resume works per arm and a re-submitted job skips whatever already finished.
GUIDANCE_ARMS="${GUIDANCE_ARMS:-0|-|-|-|-|-}"

# Restrict the run to the cells a REFERENCE run abstained on (see --abstained-from). Unset
# = the full grid, exactly as before. The value is a glob of edits_shard*.jsonl and must
# word-split into argv, so it is deliberately unquoted at the call site below.
ABSTAINED_FROM="${ABSTAINED_FROM:-}"
if [[ -n "$ABSTAINED_FROM" ]]; then
  # A glob that matches nothing would pass the literal pattern to python and die there with
  # a confusing FileNotFoundError; say so here instead.
  # An array, not `set --`: the positional parameters still hold this job's env-file
  # argument, and clobbering them here would be a trap for anything added below.
  # shellcheck disable=SC2206
  ABSTAINED_FILES=( $ABSTAINED_FROM )
  [[ -f "${ABSTAINED_FILES[0]}" ]] || {
    echo "FATAL: ABSTAINED_FROM matched no file: $ABSTAINED_FROM"; exit 1; }
  echo "abstention reference: ${#ABSTAINED_FILES[@]} file(s), min rate ${ABSTAINED_MIN_RATE:-1.0}"
fi

for ARM in ${GUIDANCE_ARMS}; do
  IFS="|" read -r G_W G_LOSS G_SCHED G_POW G_K G_MODE G_STRIDE G_GRAFT <<< "$ARM"
  [[ -n "${G_W:-}" ]] || { echo "FATAL: malformed guidance arm: $ARM"; exit 1; }
  [[ "${G_LOSS:--}" == "-" ]] && G_LOSS="ca_mse,contact_hinge"
  [[ "${G_SCHED:--}" == "-" ]] && G_SCHED="decay"
  [[ "${G_POW:--}"  == "-" ]] && G_POW="1.0"
  [[ "${G_K:--}"    == "-" ]] && G_K="1"
  [[ "${G_MODE:--}" == "-" ]] && G_MODE="identity"
  [[ "${G_STRIDE:--}" == "-" || -z "${G_STRIDE:-}" ]] && G_STRIDE="1"
  [[ "${G_GRAFT:--}" == "-" || -z "${G_GRAFT:-}" ]] && G_GRAFT="0"
  GRAFT_ARG=(); [[ "$G_GRAFT" == "1" ]] && GRAFT_ARG=(--graft-anchors)
  echo
  echo "=== arm: w=${G_W} loss=${G_LOSS} schedule=${G_SCHED}^${G_POW} k=${G_K} mode=${G_MODE} stride=${G_STRIDE} graft=${G_GRAFT}"

  # Unquoted on purpose: these are space-separated lists that must word-split into argv.
  # shellcheck disable=SC2086
  srun "$PY" scripts/sdedit_cyclize.py \
    --ckpt-path "$FLOW_CKPT_PATH" \
    --ckpt-name "$FLOW_CKPT_NAME" \
    --metadata "$LNR_METADATA" \
    --out "${RESULTS_DIR}/edits_shard${SHARD_INDEX}.jsonl" \
    ${NATIVE_TYPE:+--native-type} \
  --cyc-types ${CYC_TYPES} \
    --t-ca ${T_CA_GRID} \
    --t-lat ${T_LAT_GRID} \
    --seeds ${SEEDS} \
    --nsteps "${NSTEPS:-0}" \
    --limit "${PEPTIDE_LIMIT:-0}" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --guidance-w "$G_W" \
    --guidance-loss "$G_LOSS" \
    --guidance-schedule "$G_SCHED" \
    --guidance-schedule-pow "$G_POW" \
    --guidance-exclude-termini "$G_K" \
    --guidance-mode "$G_MODE" \
    --guidance-stride "$G_STRIDE" \
    "${GRAFT_ARG[@]}" \
    ${GRAFT_KEEP_CB:+--graft-keep-cb "$GRAFT_KEEP_CB"} \
    ${GRAFT_SIDECHAIN:+--graft-sidechain "$GRAFT_SIDECHAIN"} \
    ${GUIDANCE_MAX_DISP_A:+--guidance-max-disp-A "$GUIDANCE_MAX_DISP_A"} \
    ${GUIDANCE_CB_WINDOW_A:+--guidance-cb-window-A ${GUIDANCE_CB_WINDOW_A}} \
    ${ABSTAINED_FROM:+--abstained-from ${ABSTAINED_FROM}} \
    ${ABSTAINED_FROM:+--abstained-min-rate "${ABSTAINED_MIN_RATE:-1.0}"} \
    ${PDB_DIR:+--pdb-dir "$PDB_DIR"} \
    ${COMPLEX_DIR:+--complex-pdb-dir "$COMPLEX_DIR"}
done

echo "DONE shard ${SHARD_INDEX}"
