# Shared configuration for the cpseq mean-dG proxy-scorer pipeline.
#
# Sourced by every job in the chain. Defaults live HERE and nowhere else, so a job script
# never carries its own copy of a path. Per-submission overrides arrive as POSITIONAL
# arguments, never through `sbatch --export`: an explicit export list sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD with "(user env retrieval failed requeued held)",
# stranding every afterok dependent on Dependency. Arguments travel through the job record
# untouched, so the submitter writes one timestamped env file and passes its path.
#
# Usage inside a job body (NOT via BASH_SOURCE -- slurm copies the script to
# /var/spool/slurm/job<id>/slurm_script, so a path relative to the script points at the
# spool directory and would silently source nothing):
#
#   set -euo pipefail
#   REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
#   source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
#   meanscore_apply_args "$@"
#   meanscore_banner

# --- paths ----------------------------------------------------------------------------
# Everything written goes to scratch, never to home.
export REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
export ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
export PYTHON_EXEC="${PYTHON_EXEC:-${REPO}/.venv/bin/python}"
# Without this, python buffers stdout when it is a file rather than a tty, and a job's
# .out stays empty for hours -- indistinguishable from a hung job, with sacct dead and no
# other way to tell. Measured on jobs 96637_0..5, which printed nothing for 20 minutes
# while working normally.
export PYTHONUNBUFFERED=1
# Shadow the venv's broken torchaudio. `transformers` 5.x reaches it on the path to EVERY
# model class (modeling_utils -> loss.loss_utils -> loss_rnnt -> import torchaudio); the
# guard passes because the package is installed, then the import dies with
#   OSError: .../_torchaudio.abi3.so: undefined symbol: torch_library_impl
# because that extension was built against a different torch than the installed 2.7.0.
# Set here rather than per-job: job 96687 (the ESM probe) died on exactly this because
# only the MLM job carried the export.
export PYTHONPATH="${REPO}/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"

# Per-candidate feature JSONs handed to the scorer with --extra-features. Empty means the
# sequence-only model. Space-separated paths.
export MEANSCORE_EXTRA="${MEANSCORE_EXTRA:-}"
# Suffix so an augmented sweep cannot overwrite the sequence-only one's results.
export MEANSCORE_TAG="${MEANSCORE_TAG:-}"

export MEANSCORE_DIR="${MEANSCORE_DIR:-${ZFS}/cpseq/meanscore}"
# The run directory the folds are read from. ONE run dir on purpose: pooling run dirs
# pools PREDICTION protocols (llm_ladder_pot has the Boltz potentials on), which changes
# the structure and therefore every dG, not just the scoring.
export MEANSCORE_SOURCE="${MEANSCORE_SOURCE:-${ZFS}/cpseq/llm_ladder}"
export MEANSCORE_SCORING_VERSION="${MEANSCORE_SCORING_VERSION:-v3-cis-amide-muco-windows}"
export MEANSCORE_ARMS="${MEANSCORE_ARMS:-random_control,reference,llm_search,llm_mask_random_fill,llm_mask_mpnn_fill,random_mask_mpnn_fill}"
export MEANSCORE_MIN_FOLDS="${MEANSCORE_MIN_FOLDS:-8}"

export MEANSCORE_DATASET="${MEANSCORE_DATASET:-${MEANSCORE_DIR}/dataset_v3.json}"
export MEANSCORE_RELIABILITY="${MEANSCORE_RELIABILITY:-${MEANSCORE_DIR}/reliability_v3.json}"
export MEANSCORE_BASELINE="${MEANSCORE_BASELINE:-${MEANSCORE_DIR}/baseline_v3.json}"

# --- structure-aware (pretrained ProteinMPNN) features -------------------------------
# The native complex per target, with the peptide chain resolved by exact sequence match.
export MEANSCORE_STRUCTURES="${MEANSCORE_STRUCTURES:-${MEANSCORE_DIR}/structures_v3.json}"
# Backbone marginals: one pass per target, scores every candidate. Cheap and definitive.
export MEANSCORE_MPNN_BB="${MEANSCORE_MPNN_BB:-${MEANSCORE_DIR}/mpnn_feats_bb_v3.json}"
# Autoregressive score: one pass per candidate per decoding order. Slow, kept separate so
# a time limit on it cannot cost the cheap features.
export MEANSCORE_MPNN_AR="${MEANSCORE_MPNN_AR:-${MEANSCORE_DIR}/mpnn_feats_ar_v3.json}"
export MEANSCORE_MPNN_MODEL="${MEANSCORE_MPNN_MODEL:-v_48_020}"
# Batch 64 OOMs on a ~400-residue receptor: the batch repeats the whole complex, not just
# the peptide. 8 is measured-safe inside a 24 G job.
export MEANSCORE_MPNN_BATCH="${MEANSCORE_MPNN_BATCH:-8}"
export MEANSCORE_MPNN_DRAW_MARGINAL="${MEANSCORE_MPNN_DRAW_MARGINAL:-8}"
export MEANSCORE_MPNN_DRAW_AR="${MEANSCORE_MPNN_DRAW_AR:-2}"

# --- fold-budget triage (the deployment number) ------------------------------------
# "Does the proxy save folds?", in folds rather than in correlation. Reads saved JSON only.
export MEANSCORE_TRIAGE="${MEANSCORE_TRIAGE:-${MEANSCORE_DIR}/triage_v3.json}"
# The winning feature set from the sweep: composition_delta + ProteinMPNN (bb + AR).
export MEANSCORE_TRIAGE_FEATS="${MEANSCORE_TRIAGE_FEATS:-${MEANSCORE_MPNN_BB} ${MEANSCORE_MPNN_AR}}"
export MEANSCORE_TRIAGE_AGG="${MEANSCORE_TRIAGE_AGG:-trimmed20}"

# --- shipping model -------------------------------------------------------------------
export MEANSCORE_MODEL="${MEANSCORE_MODEL:-${MEANSCORE_DIR}/proxy_model_v3.json}"

# --- protocol-transfer arm (llm_ladder_pot: Boltz potentials ON) ----------------------
# A different PREDICTION protocol on the same targets and the same scoring version, so a
# protocol-level held-out set. Kept strictly separate from MEANSCORE_SOURCE: pooling the
# two would pool instruments, which is what the single-source pin exists to prevent.
export MEANSCORE_POT_SOURCE="${MEANSCORE_POT_SOURCE:-${ZFS}/cpseq/llm_ladder_pot}"
export MEANSCORE_POT_DATASET="${MEANSCORE_POT_DATASET:-${MEANSCORE_DIR}/dataset_pot_v3.json}"
export MEANSCORE_POT_MPNN_BB="${MEANSCORE_POT_MPNN_BB:-${MEANSCORE_DIR}/mpnn_feats_bb_pot_v3.json}"
export MEANSCORE_POT_MPNN_AR="${MEANSCORE_POT_MPNN_AR:-${MEANSCORE_DIR}/mpnn_feats_ar_pot_v3.json}"
export MEANSCORE_TRANSFER="${MEANSCORE_TRANSFER:-${MEANSCORE_DIR}/transfer_v3.json}"
# The pot run has fewer folds per sequence than llm_ladder; keep the floor lower or every
# row is filtered out and the stage reports an empty dataset as if nothing existed.
export MEANSCORE_POT_MIN_FOLDS="${MEANSCORE_POT_MIN_FOLDS:-5}"

# --- learning curve over the number of training targets -------------------------------
export MEANSCORE_CURVE="${MEANSCORE_CURVE:-${MEANSCORE_DIR}/learning_curve_v3.json}"
# Nested permutations per held-out target. The ridge is deterministic, so this is the only
# source of sampling noise and it is what makes the curve readable at small K.
export MEANSCORE_CURVE_DRAWS="${MEANSCORE_CURVE_DRAWS:-24}"

# --- statistics -----------------------------------------------------------------------
export MEANSCORE_K_PER_HALF="${MEANSCORE_K_PER_HALF:-5}"
export MEANSCORE_N_REP="${MEANSCORE_N_REP:-51}"

# --- scorer ---------------------------------------------------------------------------
export MEANSCORE_SEEDS="${MEANSCORE_SEEDS:-5}"
export MEANSCORE_EPOCHS="${MEANSCORE_EPOCHS:-160}"
export MEANSCORE_PROTOCOL="${MEANSCORE_PROTOCOL:-both}"

# The array sweep. One line per task: "<agg> <loss>". The two aggregators are the ones
# the reliability sweep cannot separate (their CIs overlap), so the scorer is trained
# under both rather than the choice being made by assertion.
MEANSCORE_CONFIGS=(
  "trimmed20 both"
  "trimmed20 mse"
  "trimmed20 rank"
  "mean_clip0 both"
  "mean_clip0 mse"
  "mean_clip0 rank"
)
export MEANSCORE_N_CONFIGS="${#MEANSCORE_CONFIGS[@]}"

# --- argument handling ----------------------------------------------------------------
# Any argument that is an existing file is sourced; any VAR=value argument is exported.
# A file handles spaces, so list-valued variables need no comma encoding.
meanscore_apply_args() {
  local a
  for a in "$@"; do
    if [[ -f "$a" ]]; then
      # shellcheck disable=SC1090
      source "$a"
    elif [[ "$a" == *=* ]]; then
      export "${a?}"
    fi
  done
}

meanscore_config_for_task() {
  # Echo "<agg> <loss>" for a 0-based array index; empty if out of range.
  local idx="${1:-0}"
  if (( idx >= 0 && idx < ${#MEANSCORE_CONFIGS[@]} )); then
    echo "${MEANSCORE_CONFIGS[$idx]}"
  fi
}

meanscore_banner() {
  echo "=============================================================="
  echo "job            ${SLURM_JOB_ID:-none}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
  echo "host           $(hostname)"
  echo "started        $(date -Is)"
  echo "partition      ${SLURM_JOB_PARTITION:-?}  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}"
  echo "submit_dir     ${SLURM_SUBMIT_DIR:-unset}"
  echo "python         ${PYTHON_EXEC}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | sed 's/^/gpu            /' || true
  else
    echo "gpu            none (cpu job, by design: a 23k-param MLP on 2k rows does not"
    echo "               need one, and asking for a GPU only queues behind real work)"
  fi
  echo "source_run     ${MEANSCORE_SOURCE}"
  echo "dataset        ${MEANSCORE_DATASET}"
  echo "=============================================================="
  "${PYTHON_EXEC}" -c 'import sys; print("python ok: " + sys.executable)' \
    || { echo "FATAL: python missing at ${PYTHON_EXEC}"; exit 1; }
}
