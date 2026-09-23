#!/bin/bash
#SBATCH --job-name=m15_time
#SBATCH --output=logs/m15_time_%j.out
#SBATCH --error=logs/m15_time_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# Section 2: the setup/marginal ratio the whole decoy budget rests on.
#
# --cpus-per-task=1 on purpose. OpenMM's CPU platform will happily use every core it is
# given, and a per-replica time measured on 88 threads would not predict throughput under
# the many-single-core layout the real build will use.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
_m15_banner timing
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

[[ -x "$PY_OPENMM" ]] || { echo "FATAL: no openmm venv at $PY_OPENMM" >&2; exit 1; }
export OPENMM_CPU_THREADS=1

cd "$REPO"
mkdir -p "$TIMING_DIR"

read -r META NCPX NREP SIG CPR KP KC CTOL TOL MAXIT RUNGS LADDER PLAT SEED < <("$PY" -c "
import sys; sys.path.insert(0, '.')
from script_utils import m15_config
t = m15_config.load('$M15_CONFIG')['timing']
print(m15_config.resolve(t['metadata']), t['n_complexes'], t['n_replicas'],
      t['torsion_sigma_deg'], t['contacts_per_residue'], t['k_pull'], t['k_contact'],
      t['contact_tol_A'], t['tolerance'], t['max_iterations'], t['ladder_rungs'],
      int(bool(t['time_ladder'])), t['platform'], t['seed'])
")
[[ "$SMOKE" == "1" ]] && { NCPX=2; NREP=3; LADDER=0; }

"$PY_OPENMM" script_utils/m15_replica_timing.py \
  --metadata "$META" \
  --out "${TIMING_DIR}/replica_timing.jsonl" \
  --n-complexes "$NCPX" --n-replicas "$NREP" \
  --torsion-sigma-deg "$SIG" --contacts-per-residue "$CPR" \
  --k-pull "$KP" --k-contact "$KC" --contact-tol-A "$CTOL" \
  --tolerance "$TOL" --max-iterations "$MAXIT" \
  --ladder-rungs "$RUNGS" $( [[ "$LADDER" == "1" ]] && echo --time-ladder ) \
  --platform "$PLAT" --seed "$SEED"

echo "[timing] done -> ${TIMING_DIR}/replica_timing.jsonl"
