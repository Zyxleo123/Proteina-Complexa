#!/usr/bin/env bash
#SBATCH --job-name=cpseq_seqcond_det
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_seqcond_det_%j.out
#
# How much does one sequence-conditioned generation run differ from an IDENTICAL one?
#
# Why this exists. Two runs of `eval_val_gen_testset.py` with the same checkpoint, the same
# seed and the same settings were found to disagree by 0.104 in `cond_exact_frac` (0.9375
# vs 0.8333) and 0.16 A in median N-C distance, with no injected sequence involved in
# either. The script sets no determinism flags -- no `torch.use_deterministic_algorithms`,
# no cuDNN determinism -- so that is expected rather than a bug. But it means two things
# were unmeasured and both block the searched-vs-native comparison:
#
#   1. Whether `--seq-source` is inert when it requests the NATIVE sequences. It must be:
#      0 substitutions is a content-level no-op, verified in-process by reading every
#      write back. If injected-native sits outside the spread of two plain controls, the
#      plumbing perturbs something and the finding would be an artifact of the plumbing.
#   2. The noise floor. A searched-vs-native difference smaller than the run-to-run spread
#      of identical runs is not readable at this sample count, whatever its sign.
#
# So: 2 controls (no injection) and 2 injected-native runs, same checkpoint, same seed,
# same node, differing only in whether --seq-source is passed. Within-condition spread is
# the noise floor; between-condition difference is what the plumbing costs.
#
# GPU type is deliberately NOT pinned. The comparison is only valid within one GPU model --
# the first measurement of this was confounded by one run landing on babel-y9-16 and the
# other on babel-x5-28 -- but all four runs here are sequential steps of ONE job, so they
# share a node and a GPU by construction and a pin buys nothing for validity. It was pinned
# to L40S at first and sat PENDING on Priority with 627 jobs ahead of it and one L40S node
# free against seven other GPU nodes; that is a 7x smaller target for no gain.
#
# What the pin would have guarded is a REQUEUE landing on different hardware mid-sequence.
# That case is detected instead of prevented: every run records its hostname and GPU, and
# the report refuses to compare across them rather than averaging two pieces of silicon.
# If it fires, delete the run dirs and resubmit.
#
# Output goes to /data/user_data (NAS), never the session scratchpad: /tmp is node-local
# and the first attempt at this measurement was lost when the allocation moved nodes.
#
# Submit:  sbatch scripts/cpseq_seqcond_determinism.sbatch
# Resume:  identical -- each run has its own out-dir and the eval skips recorded batches
# Report:  python script_utils/cpseq_seqcond_determinism_report.py \
#              --root /data/user_data/yixiz/cpseq/transfer/determinism   (CPU, no GPU)

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

RUN_DIR="${RUN_DIR:-/data/user_data/yixiz/Proteina-Complexa/training_runs/store/cpsea_seqcond_v2}"
CKPT="${CKPT:-$RUN_DIR/checkpoints/last-v3-EMA.ckpt}"
TEST_PARQUET="${TEST_PARQUET:-/data/user_data/yixiz/cpseq/seqcond/lnr_staged/metadata/tranche12_mainchain.parquet}"
DATASET_DIR="${DATASET_DIR:-/data/user_data/yixiz/cpseq/tranche1}"
SEQ_SOURCE="${SEQ_SOURCE:-/data/user_data/yixiz/cpseq/transfer/native12.jsonl}"
ROOT="${ROOT:-/data/user_data/yixiz/cpseq/transfer/determinism}"
N_COMPLEXES="${N_COMPLEXES:-12}"
N_REPEAT="${N_REPEAT:-4}"
SEED="${SEED:-0}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L
echo "request: gpu:1 cpus=${SLURM_CPUS_PER_TASK:-?} mem=24G time=1:30:00"
echo "n_complexes=$N_COMPLEXES n_repeat=$N_REPEAT seed=$SEED"

for p in "$CKPT" "$TEST_PARQUET" "$SEQ_SOURCE" "$DATASET_DIR/manifest/targets.csv"; do
  [ -e "$p" ] || { echo "missing input: $p" >&2; exit 2; }
done

mkdir -p "$ROOT"

# Four runs, sequential so they share one node and one GPU. `control_*` passes no
# --seq-source at all; `inject_*` requests each parent's own native peptide, which is a
# content-level no-op by construction.
for NAME in control_a control_b inject_a inject_b; do
  OUT="$ROOT/$NAME"
  if compgen -G "$OUT/closure_rows_*.jsonl" > /dev/null; then
    echo "[$NAME] already has closure rows; skipping (resume)"
    continue
  fi
  mkdir -p "$OUT"
  # Recorded per run, because the comparison is only meaningful within one GPU model and a
  # requeue can move the job between runs.
  python - "$OUT" <<'PY'
import json, os, subprocess, socket, sys
out = sys.argv[1]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
json.dump({"hostname": socket.gethostname(), "gpu": gpu,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
          open(os.path.join(out, "node.json"), "w"), indent=1)
PY

  EXTRA=()
  case "$NAME" in
    inject_*) EXTRA=(--seq-source "$SEQ_SOURCE"
                     --seq-source-label native
                     --seq-source-dataset-dir "$DATASET_DIR") ;;
  esac

  echo "=== $NAME ==="
  python script_utils/eval_val_gen_testset.py \
    --run-dir "$RUN_DIR" \
    --ckpt "$CKPT" \
    --test-parquet "$TEST_PARQUET" \
    --n-complexes "$N_COMPLEXES" \
    --n-repeat "$N_REPEAT" \
    --seed "$SEED" \
    --num-workers 0 \
    --seq-cond --keep-frac 1.0 \
    "${EXTRA[@]}" \
    --out-dir "$OUT"
done

echo "all four runs done; report with:"
echo "  python script_utils/cpseq_seqcond_determinism_report.py --root $ROOT"
