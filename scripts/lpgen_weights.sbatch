#!/bin/bash
#SBATCH --job-name=lpgen_weights
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_weights_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_weights_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Download the arms' RELEASED checkpoints, verify them, and record what landed.
#
# Reproducing a published number requires the published weights; this stage is what makes
# the reproduction arms possible without training anything.
#
# Runs as a job, not on the login node: compute nodes have outbound network here (measured,
# job 80479) and these are 1.24 GB files.
#
# Checkpoints live on scratch under WEIGHTS_DIR, deliberately OUTSIDE the upstream trees,
# so re-staging an arm (which wipes its tree) does not throw away 2.5 GB of download.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "weights"

mkdir -p "$WEIGHTS_DIR"
MANIFEST_OUT="${WEIGHTS_DIR}/weights_manifest.json"
rows=()
fail=0

# Only act on the arms this submission asked for, so a run for one arm does not pull
# another arm's 1.24 GB file as a side effect.
want_arm() {
  local a
  for a in ${STAGE_ARMS//,/ }; do [[ "$a" == "$1" ]] && return 0; done
  return 1
}

while read -r arm fname url md5; do
  [[ -z "${arm:-}" || "$arm" == \#* ]] && continue
  want_arm "$arm" || { echo "[weights] skip ${arm}/${fname}: not in STAGE_ARMS=${STAGE_ARMS}"; continue; }

  dest="${WEIGHTS_DIR}/${arm}/${fname}"
  mkdir -p "$(dirname "$dest")"
  echo "=================================================================="
  echo "[weights] ${arm}/${fname}"

  if [[ -f "$dest" && "$WEIGHTS_FORCE" != "1" ]]; then
    echo "[weights] present: ${dest} ($(stat -c %s "$dest") bytes)"
  else
    # --fail so an HTTP error is an error. Without it curl happily writes the error BODY
    # to the output file, producing a small HTML file with a .pth name that fails only
    # later, inside torch.load, as an unrelated-looking unpickling error.
    # -L because both Zenodo and GitHub releases redirect to object storage.
    # Download to a .part file and move only on success, so an interrupted job cannot
    # leave a truncated checkpoint that the `-f` check above then treats as complete.
    echo "[weights] GET ${url}"
    # --no-progress-meter, not -s: curl's progress meter is not line-buffered, so into a
    # non-tty .out it writes one ~15 kB single-line smear per file that makes the log
    # unreadable. -S still surfaces the error message if the transfer fails.
    if ! curl -fL -S --no-progress-meter --retry 3 --retry-delay 5 --connect-timeout 30 \
              -o "${dest}.part" "$url"; then
      echo "[weights] FAIL download: ${arm}/${fname}" >&2
      rm -f "${dest}.part"
      rows+=("{\"arm\":\"${arm}\",\"file\":\"${fname}\",\"status\":\"download_failed\",\"url\":\"${url}\"}")
      fail=1
      continue
    fi
    mv "${dest}.part" "$dest"
  fi

  size="$(stat -c %s "$dest")"
  got="$(md5sum "$dest" | awk '{print $1}')"
  if [[ "$md5" == "-" ]]; then
    # No published checksum for this file (the GitHub release does not advertise one), so
    # record what we got rather than claiming it was verified.
    echo "[weights] md5 ${got} (no published checksum to compare against)"
    status="recorded_unverified"
  elif [[ "$got" == "$md5" ]]; then
    echo "[weights] md5 OK ${got}"
    status="verified"
  else
    echo "[weights] FAIL md5 MISMATCH: expected ${md5}, got ${got}" >&2
    echo "[weights] not deleting ${dest}; inspect it (a captured error page is ~KB, not GB)" >&2
    status="md5_mismatch"
    fail=1
  fi
  echo "[weights] ${dest}  ${size} bytes  ${status}"
  rows+=("{\"arm\":\"${arm}\",\"file\":\"${fname}\",\"path\":\"${dest}\",\"bytes\":${size},\"md5\":\"${got}\",\"expected_md5\":\"${md5}\",\"status\":\"${status}\"}")
done <<< "$WEIGHTS_SPEC"

# foundry/RFD3 is not a URL download: it ships its own installer, which knows the checkpoint
# registry and the current RFD3 revision. So it gets a branch of its own rather than a row in
# WEIGHTS_SPEC.
#
# This closes a real gap. The train job's own banner says "the released RFD3 weights are not
# on this cluster", which reads as unobtainable -- but `foundry list-available` offers `rfd3`
# and the CLI is already in the arm's env. Without these weights the LP job trains RFD3 FROM
# RANDOM INIT, which is not what the plan asks for ("Start from the released RFD3 weights;
# use model-only initialization") and which diverged in job 83990.
if want_arm foundry; then
  echo "=================================================================="
  echo "[weights] foundry/rfd3 via its own installer"
  f_env="${ENVS_DIR}/foundry"
  f_cli="${f_env}/bin/foundry"
  f_store="${WEIGHTS_DIR}/foundry"
  if [[ ! -x "$f_cli" ]]; then
    echo "[weights] GATE foundry: no CLI at ${f_cli}; run --submit env --set STAGE_ARMS=foundry first" >&2
  else
    mkdir -p "$f_store"
    # Keep HOME-bound defaults out of it: foundry defaults to ~/.foundry/checkpoints, and
    # home is a quota'd volume that also holds the repository.
    if "$f_cli" install "${FOUNDRY_MODELS:-rfd3}" --checkpoint-dir "$f_store"; then
      echo "[weights] foundry installed -> ${f_store}"
      find "$f_store" -maxdepth 2 -type f \( -name '*.ckpt' -o -name '*.pt' \) \
        -printf '  %p  %s bytes\n' | sort
      # The path the train job needs, printed so it can be copied into TRAIN_CKPT_PATH
      # rather than guessed.
      rfd3_ckpt="$(find "$f_store" -maxdepth 2 -type f -name '*.ckpt' | sort | head -1)"
      if [[ -n "$rfd3_ckpt" ]]; then
        echo "[weights] NEXT: --submit train --set TRAIN_CKPT_PATH=${rfd3_ckpt}"
        rows+=("{\"arm\":\"foundry\",\"file\":\"$(basename "$rfd3_ckpt")\",\"path\":\"${rfd3_ckpt}\",\"bytes\":$(stat -c %s "$rfd3_ckpt"),\"status\":\"installed_by_cli\"}")
      fi
    else
      echo "[weights] FAIL foundry install" >&2
      fail=1
    fi
  fi
fi

# Point each arm's own expected checkpoint directory at the weights store, so upstream code
# resolves its DEFAULT paths without being patched. DiffPepBuilder reads both
# `${BASE_PATH}/experiments/checkpoints/diffpepbuilder_v1.pth` (config/eval.yaml) and
# `{root}/experiments/checkpoints/esm2_*.pt` (PretrainedSequenceEmbedder), i.e. one
# directory inside the tree -- so one symlink serves both and nothing is copied twice.
#
# Done here rather than in the stage job because the stage job does not know WEIGHTS_DIR,
# and refreshed every run so a re-stage (which recreates a real directory) is repaired
# rather than silently shadowing the store. A REAL non-empty directory is left alone: it
# may hold something a human put there, and clobbering 2.5 GB on a guess is not recoverable.
for arm in ${STAGE_ARMS//,/ }; do
  [[ "$arm" == "diffpepbuilder" ]] || continue
  tree="${UPSTREAM_DIR}/${arm}"
  link="${tree}/experiments/checkpoints"
  store="${WEIGHTS_DIR}/${arm}"
  if [[ ! -d "$tree" ]]; then
    echo "[weights] note: ${arm} not staged; skipping the checkpoint-dir symlink" >&2
    continue
  fi
  mkdir -p "$store"
  if [[ -L "$link" ]]; then
    rm -f "$link"
  elif [[ -d "$link" ]]; then
    if [[ -n "$(ls -A "$link" 2>/dev/null)" ]]; then
      echo "[weights] REFUSING to replace non-empty real directory ${link}" >&2
      echo "[weights] move its contents into ${store} and rerun" >&2
      fail=1
      continue
    fi
    rmdir "$link"
  fi
  ln -s "$store" "$link"
  echo "[weights] ${link} -> ${store}"
done

{
  echo "{"
  echo "  \"written_at\": \"$(date -Is)\","
  echo "  \"job\": \"${SLURM_JOB_ID:-none}\","
  echo "  \"arms\": \"${STAGE_ARMS}\","
  echo "  \"weights_dir\": \"${WEIGHTS_DIR}\","
  echo "  \"files\": ["
  for i in "${!rows[@]}"; do
    sep=","; [[ $i -eq $((${#rows[@]} - 1)) ]] && sep=""
    echo "    ${rows[$i]}${sep}"
  done
  echo "  ]"
  echo "}"
} > "$MANIFEST_OUT"

echo "=================================================================="
echo "[weights] manifest -> ${MANIFEST_OUT}"
cat "$MANIFEST_OUT"
if [[ $fail -ne 0 ]]; then
  echo "[weights] at least one file failed; see above" >&2
  exit 1
fi
echo "[weights] done"
