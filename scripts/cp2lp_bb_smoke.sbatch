#!/bin/bash
#SBATCH --job-name=cp2lp_bb_smoke
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:50:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# GPU preflight for the two backbone arms, on the REAL trunk and a REAL batch. Runs BOTH
# arms in one job, so a difference between them cannot be a difference between two jobs.
#
# a6000 and 48G by default: the same class the arms will train on, because pair features are
# O(n_target^2) with a crop to 256 residues and a memory number from a smaller card measures
# the wrong thing. Override for a queue-constrained smoke, e.g.
#   sbatch --partition=debug --nodelist=gpu1 scripts/cp2lp_bb_smoke.sbatch
# which is a legitimate use of `debug` -- a short preflight is what it is for -- but do NOT
# read a memory envelope off a run that was squeezed onto a different card.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-bb_torsion}"
_cp2lp_defaults
_cp2lp_banner "bb-smoke"

cd "$REPO"

# Both arms, always. The whole point of this tranche is a single-variable comparison, and a
# smoke that exercised one arm would leave the other's first contact with the real trunk to
# the training job.
ARMS_TO_SMOKE="${ARMS_TO_SMOKE:-bb_cartesian bb_torsion}"

# A fresh init is a legitimate smoke mode: the backbone heads are new in every case, so
# `--skip-ckpt` still exercises every claim except the splice itself.
EXTRA=()
[[ "${SMOKE_SKIP_CKPT:-0}" == "1" ]] && EXTRA+=(--skip-ckpt)
[[ "${SMOKE_SKIP_TRAINING_STEP:-0}" == "1" ]] && EXTRA+=(--skip-training-step)
[[ -n "${INIT_CKPT_OVERRIDE:-}" ]] && EXTRA+=(--init-ckpt "${INIT_CKPT_OVERRIDE}")

echo "[smoke] CPU-side invariants first -- a broken one makes the GPU numbers meaningless"
"${PY}" -m pytest -q --no-header -p no:warnings \
  script_utils/test_cp2lp_backbone.py \
  script_utils/test_cp2lp_bb_arms.py \
  script_utils/test_cp2lp_bb_losses.py \
  script_utils/test_cp2lp_bb_discriminator.py

rc=0
for arm in ${ARMS_TO_SMOKE}; do
  case "${arm}" in
    bb_cartesian) cfg="example/training_cp2lp_bb_cartesian" ;;
    bb_torsion)   cfg="example/training_cp2lp_bb_torsion" ;;
    *) echo "unknown arm ${arm}" >&2; exit 1 ;;
  esac
  echo
  echo "################################################################"
  echo "# ARM ${arm}  (${cfg})"
  echo "################################################################"
  if ! "${PY}" script_utils/cp2lp_bb_smoke.py --config "${cfg}" --max-cp 2 "${EXTRA[@]}"; then
    echo "ARM ${arm} FAILED" >&2
    rc=1
  fi
done

echo
if [[ "${rc}" -eq 0 ]]; then
  echo "[smoke] both arms PASSED"
else
  echo "[smoke] at least one arm FAILED -- do not launch the pilots" >&2
fi
exit "${rc}"
