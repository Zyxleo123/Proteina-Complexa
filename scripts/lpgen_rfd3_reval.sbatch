#!/bin/bash
#SBATCH --job-name=lpgen_reval
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_reval_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_reval_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Re-validate ONE saved checkpoint against the corrected panel. Trains nothing.
#
# WHY THIS EXISTS. The first validation curve was measured through a broken instrument: no
# `ori_token` was supplied, so rfd3 centred the diffusion prior on the fixed motif -- the
# RECEPTOR centroid, a median 17.8 A from the real binding site -- and every peptide
# condensed inside the protein. Training itself was NOT affected (`center_option: diffuse`
# centres on the peptide), so the checkpoints are sound and only the measurement needs
# redoing. Eight checkpoints already exist; re-scoring them costs no retraining.
#
# HOW A VALIDATION-ONLY RUN IS OBTAINED, and the trap that is NOT the way to do it.
# Loading a checkpoint does `current_epoch += 1` and then sets `should_stop` if
# `current_epoch >= max_epochs` (foundry/trainers/fabric.py:429-435) -- the reason a small
# `max_epochs` makes a fine-tune train ZERO steps and exit 0. `prevalidate` does run before
# the `while not self.should_stop` loop (:447 vs :459), so that looks like a free
# validation-only mode. IT IS NOT: `validation_loop` ALSO breaks on `self.should_stop`
# (:628), so a small `max_epochs` aborts validation at batch 0. MEASURED -- job 85088 logged
# "Prevalidating with epoch 586", ran 0 of 64 batches, wrote an empty metrics CSV and then
# died in `pd.concat` with "No objects to concatenate".
#
# So instead: keep `should_stop` FALSE through prevalidation by setting
# `max_epochs = ckpt_epoch + 2`, and neutralise the one training epoch that then starts:
#   limit_train_batches=0        -> train_loop breaks at batch_idx 0 (:533), does no work
#   validate_every_n_epochs=huge -> `should_validate` is False, so the post-epoch validation
#                                   does NOT run and the curve gets no duplicate point
#                                   (prevalidate is unaffected: it is gated only on
#                                   `val_loaders` being non-empty, not on should_validate)
#   checkpoint_every_n_epochs=huge -> no stray checkpoint is written
# After that no-op epoch, current_epoch reaches ckpt_epoch+2 == max_epochs and the run stops.
# Prevalidate also decrements current_epoch for its duration, so structures land in
# `epoch_<N>/` matching the checkpoint rather than N+1.
#
# ONE JOB PER CHECKPOINT, each with its own `paths.output_dir`. Concurrent jobs sharing an
# output tree would append to one metrics.csv, and concurrent O_APPEND is measured to
# NUL-corrupt rows on this cluster.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "reval"

ARM_TREE="${UPSTREAM_DIR}/foundry"
ARM_PY="${ENVS_DIR}/foundry/bin/python"

# Gates exit 0, never non-zero: a non-zero exit here strands the scoring job behind this
# one in DependencyNeverSatisfied with no explanation in any log.
for p in "$ARM_TREE/models/rfd3/src/rfd3/train.py" "$ARM_PY"; do
  [[ -e "$p" ]] || { echo "[reval] GATE missing: $p (run --submit stage / --submit env)" >&2; exit 0; }
done
if [[ -z "${REVAL_CKPT}" ]]; then
  echo "[reval] GATE: REVAL_CKPT is empty -- nothing to re-evaluate" >&2; exit 0
fi
if [[ ! -f "${REVAL_CKPT}" ]]; then
  echo "[reval] GATE missing checkpoint: ${REVAL_CKPT}" >&2; exit 0
fi
if [[ ! -f "${REVAL_CONTIGS}" ]]; then
  echo "[reval] GATE missing contigs: ${REVAL_CONTIGS}" >&2
  echo "[reval] build them with:" >&2
  echo "  python script_utils/lpgen_build_val_contigs.py --panel \$VAL_PANEL \\" >&2
  echo "      --out ${REVAL_CONTIGS} --ori-mode pocket --ori-push 6.0" >&2
  exit 0
fi

CKPT_TAG="$(basename "${REVAL_CKPT}" .ckpt)"

# The epoch the checkpoint came from decides max_epochs, so parse it and REFUSE rather than
# guess: too small silently aborts validation (see the header), too large trains for real.
CKPT_EPOCH="$(sed -n 's/^epoch-0*\([0-9][0-9]*\)$/\1/p' <<<"$CKPT_TAG")"
if [[ -z "$CKPT_EPOCH" ]]; then
  echo "[reval] GATE: cannot parse an epoch from '${CKPT_TAG}' (expected epoch-NNNN.ckpt)." >&2
  echo "[reval] Refusing rather than guessing max_epochs -- see this script's header." >&2
  exit 0
fi
REVAL_MAX_EPOCHS=$((CKPT_EPOCH + 2))

# Derive the output root from the checkpoint's OWN run directory rather than TRAIN_RUN_NAME.
# Those two drifted on the first attempt: weights from lp_finetune_v1 were written into
# lp_scratch_v1_reval because TRAIN_RUN_NAME still held its default.
CKPT_RUN_DIR="$(dirname "$(dirname "${REVAL_CKPT}")")"
CKPT_RUN_NAME="$(basename "${CKPT_RUN_DIR}")"
RUN_DIR="${REVAL_ROOT_OVERRIDE:-${LPGEN_RUNS_DIR}/${CKPT_RUN_NAME}_reval}/${CKPT_TAG}"
mkdir -p "$RUN_DIR"
echo "[reval] ckpt epoch ${CKPT_EPOCH} -> max_epochs ${REVAL_MAX_EPOCHS}"

echo "[reval] checkpoint : ${REVAL_CKPT}"
echo "[reval] contigs    : ${REVAL_CONTIGS}"
echo "[reval] output_dir : ${RUN_DIR}"

# Record WHICH ori convention this run used, in the run directory. A curve assembled later
# from several of these is only comparable if they all used the same one, and the contigs
# file can be rebuilt with a different mode under the same name.
"$ARM_PY" - "$REVAL_CONTIGS" "$RUN_DIR/ori_provenance.json" <<'PYEOF'
import hashlib, json, sys
src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
specs = json.loads(raw)
n_ori = sum(1 for v in specs.values() if "ori_token" in v)
json.dump({"contigs": src,
           "sha256": hashlib.sha256(raw).hexdigest(),
           "n_specs": len(specs),
           "n_with_ori_token": n_ori}, open(dst, "w"), indent=2)
print(f"[reval] contigs sha256 {hashlib.sha256(raw).hexdigest()[:16]}  "
      f"{n_ori}/{len(specs)} specs carry an ori_token")
assert n_ori == len(specs), "some specs have NO ori_token: those fall back to the receptor centroid"
PYEOF

export PYTHONPATH="${REPO}/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$ARM_TREE"
set +e
"$ARM_PY" models/rfd3/src/rfd3/train.py \
  "experiment=${TRAIN_EXPERIMENT}" \
  "hydra.searchpath=[pkg://rfd3.configs,pkg://configs,file://${REPO}/configs/rfd3_lp]" \
  "ckpt_path=${REVAL_CKPT}" \
  "paths.output_dir=${RUN_DIR}" \
  "datasets.val.lp_panel.dataset.data=${REVAL_CONTIGS}" \
  "datasets.val.lp_panel.dataset.eval_every_n=1" \
  "trainer.prevalidate=True" \
  "trainer.max_epochs=${REVAL_MAX_EPOCHS}" \
  "trainer.limit_train_batches=0" \
  "trainer.validate_every_n_epochs=1000000" \
  "trainer.checkpoint_every_n_epochs=1000000" \
  ${EXTRA_REVAL_ARGS:-} 2>&1 | tee "${RUN_DIR}/reval.log"
RC=${PIPESTATUS[0]}
set -e

echo "=================================================================="
echo "[reval] train.py exit code: ${RC}"

# A validation-only run that trained by accident would corrupt the curve it exists to
# measure, so assert the trap behaved as intended.
# One epoch DOES start by design (it is how should_stop stays false through prevalidation),
# but limit_train_batches=0 must make it do nothing. Assert on optimizer steps, not on the
# epoch banner: a re-eval that silently trained would move the weights it is measuring.
N_OPT=$(grep -cE "Optimizer step|optimizer_step|global_step" "${RUN_DIR}/reval.log" 2>/dev/null; true)
if [[ "${N_OPT}" -gt 0 ]]; then
  echo "[reval] ERROR: ${N_OPT} optimizer-step line(s) found -- this run TRAINED." >&2
  echo "[reval] limit_train_batches=0 did not take effect; the measurement is not clean." >&2
  exit 1
fi
if ! grep -q "Prevalidating with epoch ${CKPT_EPOCH}" "${RUN_DIR}/reval.log" 2>/dev/null; then
  echo "[reval] ERROR: prevalidation did not run at the checkpoint's own epoch." >&2
  exit 1
fi
# The empty-CSV failure mode: validation "ran" but scored nothing.
if grep -q "Skipping empty CSV" "${RUN_DIR}/reval.log" 2>/dev/null; then
  echo "[reval] ERROR: the validation metrics CSV was EMPTY -- validation iterated 0 batches." >&2
  echo "[reval] This is the should_stop-aborts-validation mode; check max_epochs." >&2
  exit 1
fi
if ! grep -q "Prevalidating with epoch" "${RUN_DIR}/reval.log" 2>/dev/null; then
  echo "[reval] ERROR: prevalidation never ran, so nothing was measured." >&2
  exit 1
fi

N_CIF=$(find "${RUN_DIR}/val_structures" -name '*.cif.gz' 2>/dev/null | wc -l)
echo "[reval] structures dumped: ${N_CIF}"
if [[ "$N_CIF" -eq 0 ]]; then
  echo "[reval] ERROR: prevalidation ran but dumped nothing -- check callbacks=lp_callbacks" >&2
  exit 1
fi
echo "[reval] run dir: $RUN_DIR"
exit "$RC"
