#!/bin/bash
#SBATCH --job-name=stage0_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 0, step 2: the GPU half of the cost profile.
#
# This is a WRAPPER around scripts/sdedit_sweep.sbatch, not a copy of it, and that is the
# whole design. A profile is only useful if it measured the code path Stage 1 will actually
# run; a second invocation with its own hand-copied argument list would drift from the
# production arm the first time either changed, and the cost table would then describe a
# pipeline nobody runs. An .sbatch file is an ordinary bash script -- the #SBATCH lines are
# comments -- so running it here executes the real arm inside this allocation, driven by the
# same env file, with PROFILE=1 and EXAMPLES_FILE set by the selection job.
#
# What this wrapper adds on top: the run's provenance (git revision, resolved checkpoints,
# which GPU it landed on, the sampler settings) and the setup/generation split, written to
# profile/run_meta.json. Setup is paid once per job, so the report needs it separately from
# the per-structure figure or every extrapolation is wrong in the same direction.
#
# Do NOT sbatch directly -- use scripts/submit_stage0_profile.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs
REPO="$(pwd)"

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${EXAMPLES_FILE:?EXAMPLES_FILE not set}"
PROFILE_DIR="${RESULTS_DIR}/profile"
mkdir -p "$PROFILE_DIR"

# The selection job is an afterok dependency, so a missing list means it ran and wrote
# nothing -- a real failure, not a gate.
[[ -f "$EXAMPLES_FILE" ]] || { echo "FATAL: no case list at ${EXAMPLES_FILE}"; exit 1; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
GIT_REV="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
echo "git revision: ${GIT_REV} (${GIT_DIRTY} modified file(s) in the working tree)"
echo "GPU: ${GPU_NAME:-unknown}"

# Run the production arm and keep its stdout: PROFILE_SUMMARY comes back on it, and the
# pipefail-safe tee keeps a failure inside the arm a failure here.
RUN_LOG="${PROFILE_DIR}/gpu_run.log"
set +e
PROFILE=1 bash scripts/sdedit_sweep.sbatch "$ENV_FILE" 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e
echo "sdedit arm exit code: ${RC}"

PY="${PYTHON_EXEC:-${REPO}/.venv/bin/python}"
"$PY" - "$RUN_LOG" "${PROFILE_DIR}/run_meta.json" <<PYEOF
import json, sys
from pathlib import Path

run_log, out_json = sys.argv[1], sys.argv[2]

# The arm runs one python process per guidance arm, so there is one PROFILE_SUMMARY line per
# arm. Setup is paid once per PROCESS, so the per-job figure the report needs is the sum, and
# quoting only the last line would understate it by however many arms ran.
summaries = []
for line in Path(run_log).read_text(errors="replace").splitlines():
    if line.startswith("PROFILE_SUMMARY "):
        try:
            summaries.append(json.loads(line[len("PROFILE_SUMMARY "):]))
        except json.JSONDecodeError:
            pass

meta = {
    "git_rev": "${GIT_REV}",
    "git_dirty_files": int("${GIT_DIRTY}"),
    "gpu": "${GPU_NAME:-unknown}",
    "node": "$(hostname 2>/dev/null || echo unknown)",
    "flow_ckpt": "${FLOW_CKPT_PATH:-?}/${FLOW_CKPT_NAME:-last-EMA.ckpt}",
    "ae_ckpt": "${CPSEA_AE_CKPT_PATH:-?}",
    "metadata": "${LNR_METADATA:-?}",
    "examples_file": "${EXAMPLES_FILE}",
    "nsteps": "${NSTEPS:-0} (0 = the sampler config's own, currently 400)",
    "t_ca": "${T_CA_GRID:-?}",
    "t_lat": "${T_LAT_GRID:-?}",
    "seeds": "${SEEDS:-?}",
    "cyc_types": "${CYC_TYPES:-?}",
    "guidance_arms": "${GUIDANCE_ARMS:-0|-|-|-|-|-}",
    "arm_exit_code": ${RC},
    "profile_summaries": summaries,
    "setup_seconds": round(sum(s.get("setup_seconds", 0.0) for s in summaries), 1) if summaries else None,
    "generation_seconds": round(sum(s.get("generation_seconds", 0.0) for s in summaries), 1) if summaries else None,
}
if not summaries:
    meta["warning"] = ("no PROFILE_SUMMARY line in the run log -- the arm ran without "
                       "PROFILE=1, or died before finishing. Per-structure timings will be "
                       "missing from the report.")
Path(out_json).write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PYEOF

exit "$RC"
