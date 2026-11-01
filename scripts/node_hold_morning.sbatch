#!/usr/bin/env bash
#SBATCH --job-name=nodehold_am
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/nodehold_am_%j.out
#
# A node-hold timed to be ALIVE when yixiz wakes up, rather than to start as
# soon as possible.
#
# The problem it solves: on preempt the conditional median wait is ~8.5 h once
# a job has already waited 3 h, so a 2 h hold submitted at night usually lands
# at some hour like 03:00 and has expired by morning. Starting sooner is worth
# nothing if the allocation is gone by the time anyone can type into it.
#
# Two knobs do the work, and they pull against each other:
#
#   --begin   refuses to start before a given time, so a landing is not spent
#             in the middle of the night. This costs almost nothing: the Age
#             priority factor is worth ~3.5 points/hour out of ~2740, so
#             forgoing several hours of accrual is noise. It is close to a free
#             way to aim the landing window.
#   --time    must span from the landing to when you actually want to work, but
#             a longer limit is harder to backfill: a 8 h request needs an 8 h
#             hole, and preempt mostly offers short ones.
#
# So this is submitted more than once with different (begin, time) pairs --
# a wide landing window needing a big hole, and a narrow one needing a small
# hole -- because which bet pays off depends on holes nobody can predict.
#
# Honest limits, which no script can fix:
#   - Nothing guarantees a node at a given hour. 584 jobs outrank these, and
#     Slurm itself returns START_TIME=N/A for them.
#   - preempt is PreemptMode=REQUEUE, so a hold that did land can be killed and
#     requeued at any moment, which ends the interactive session regardless.
#
# This is a deliberate, time-boxed hold with a definite TimeLimit, not a
# `sleep infinity` parked on a GPU: CPU-only at billing=3 (2 CPU + 8 GB), so
# an idle hour costs ~3 billing-units against a two-week spend of 249M.
# Do not add --gres here; a GPU is 16 of a 20-billing job, and an idle GPU is
# the one thing the cluster etiquette actually forbids.
#
# Submit (see the companion comment block in the response for the pairs used):
#   sbatch --begin=2026-11-01T03:00:00 --time=8:00:00 scripts/node_hold_morning.sbatch
#   sbatch --begin=2026-11-01T06:30:00 --time=3:00:00 scripts/node_hold_morning.sbatch

set -uo pipefail   # deliberately not -e: a hiccup must not drop the allocation

STATUS="$HOME/node_hold_status.txt"
STAMP="$(date '+%F %T')"

echo "=== $STAMP node-hold ${SLURM_JOB_ID:-none} landed on $(hostname) ==="
squeue -j "${SLURM_JOB_ID:-0}" -o '%.10i %.4C %.8m %.10L %R' -h 2>/dev/null || true

# Where to go, written to /home so it is readable from the login node without
# an allocation of your own. Refreshed every minute so TIME_LEFT is current
# rather than a stale number from whenever the job started.
write_status() {
  {
    echo "=== node hold: WHERE TO GO ==="
    echo "updated:    $(date '+%F %T')"
    echo "job:        ${SLURM_JOB_ID:-none}"
    echo "node:       $(hostname)"
    echo "time left:  $(squeue -j "${SLURM_JOB_ID:-0}" -h -o '%L' 2>/dev/null || echo '?')"
    echo "ends at:    $(squeue -j "${SLURM_JOB_ID:-0}" -h -o '%e' 2>/dev/null || echo '?')"
    echo "cpus/mem:   $(squeue -j "${SLURM_JOB_ID:-0}" -h -o '%C CPUs / %m' 2>/dev/null || echo '?')"
    echo "gpu:        $(nvidia-smi -L 2>/dev/null | head -2 || echo 'none (CPU-only hold)')"
    echo
    echo "--- attach from the login node ---"
    echo "  ssh $(hostname)        # allowed: you hold an allocation on it"
    echo "  cd /home/yixiz/Proteina-Complexa"
    echo
    echo "--- /data is mounted here, so this works too ---"
    echo "  bash scripts/cpseq_push_b2.sh b2:proteina/cpseq"
    echo
    echo "--- other holds still queued (cancel what you do not need) ---"
    squeue -u "$USER" -h -n nodehold_am -o '  %i %T begin=%S time=%l' 2>/dev/null || true
  } > "$STATUS" 2>&1
}

write_status
echo "--- status written to $STATUS ---"
cat "$STATUS"

# Hold the allocation for the whole TimeLimit, refreshing the status file so
# "time left" stays honest. Slurm ends the job at the limit; there is no
# unbounded sleep here.
while true; do
  sleep 60
  write_status
done
