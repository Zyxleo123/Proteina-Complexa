#!/usr/bin/env bash
#SBATCH --job-name=cpseq_costprobe
#SBATCH --partition=preempt
#SBATCH --qos=qos_preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/costprobe/logs/probe_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/costprobe/logs/probe_%j.out
#
# MOVED CLUSTER 2026-11-03. This was written on babel (-p preempt --account=bapoczos
# --qos=preempt_qos, scratch at /data/user_data/yixiz, logs in /home/yixiz/slurm_logs)
# and none of that exists on autonlab: there is no bapoczos account, `preempt_qos` is
# not a QoS here, and logs must not go to the home volume. The directives above are the
# autonlab equivalents and the body detects its scratch root, the way
# _cpseq_potentials_preamble.sh does, so the script works on either cluster.
#
# PARTITION: `preempt`, not `general`. Both carry a6000s, but every `general` node
# (gpu24-27, gpu29-31) was fully allocated for the whole of 2026-11-03 while `preempt`
# (gpu1, gpu27, gpu29-31 -- all a6000) had cards free, and a probe that never runs
# measures nothing. Each partition here carries its own restricted QoS, so
# `--qos=qos_preempt` is mandatory and not a preference: without it the job is rejected
# with "Invalid qos specification". On autonlab the QoS names are qos_preempt /
# qos_general / qos_debug / qos_legacy; the babel names (preempt_qos, bapoczos account)
# do not exist.
#
# Being preemptible is acceptable here for the reason the header already gives: the
# results file is append-only and every row carries the full configuration that produced
# it, so a preempted-and-requeued probe adds an independent measurement rather than
# corrupting one.
#
# The A6000 pin is kept deliberately -- see the note below on why the card cannot float.
# The v100s on `legacy` are idle far more often, and using them would be the wrong
# saving: the 44 s + 26 s/structure figure this probe extends is an a6000 figure.
#
# Where a Boltz-2 fold's wall time goes, and whether ten seeds can become one
# invocation with forty diffusion samples without changing the answer.
#
# A6000 is pinned, not left to the scheduler: the 44 s + 26 s/structure figure this
# probe is extending was measured on an A6000, and preempt also hands out Blackwell
# RTX_PRO_6000 and others. A timing taken on a different card would not compose with
# the number it is meant to refine.
#
# Must be a job. Run inside the node-hold allocation, boltz's working set is above the
# extern step's ceiling and the process dies to a bare SIGKILL with no traceback --
# which is how the earlier batching comparison managed to "pass" while producing
# nothing at all.
#
# Requeue-safe in the only sense that matters here: the results file is append-only and
# every row carries the full configuration that produced it, so a restarted probe adds
# a second independent measurement rather than corrupting the first.
#
# Submit: sbatch scripts/cpseq_fold_cost_probe.sbatch
#         sbatch scripts/cpseq_fold_cost_probe.sbatch /path/to/other/input.yaml

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/script_utils" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa

# Detected, not hardcoded: this probe was written on babel, where scratch is
# /data/user_data/yixiz, and runs on autonlab, where it is /zfsauton/scratch/yixiz. A
# fixed default would point at a root that does not exist and the first symptom would be
# "no input yaml" rather than "wrong cluster".
if [ -z "${ZFS:-}" ]; then
  if [ -d /data/user_data/yixiz ]; then ZFS=/data/user_data/yixiz
  else ZFS=/zfsauton/scratch/yixiz
  fi
fi
PROBE=$ZFS/cpseq/costprobe
WORK=${WORK:-$PROBE/work}
OUT=${OUT:-$PROBE/results.jsonl}

# Default input: a real production fold with its cached MSA, target template and pocket
# constraint all intact. Overridable as $1.
#
# The babel default was pepbench_corpus_pot/pepbench_1_3_pdb1gff (receptor 414 aa +
# 12-mer); that corpus is not on autonlab. `llm_ladder_pot` is, with 4,680 predictions,
# and it is the potentials-ON run -- which matters here, because `use_potentials` is what
# multiplies the diffusion multiplicity by three and is therefore part of the chunking
# arithmetic this probe exists to test. First input.yaml found is used.
#
# The banner comes FIRST, before the search, because job 95827 died in 35 s with a
# ZERO-BYTE log and no way to tell why: the search was written as
# `find ... | sort | head -1`, `head` exits after one line, `sort` takes SIGPIPE, the
# pipeline returns non-zero, and `set -o pipefail` + `set -e` killed the script before a
# single echo had run. `-print -quit` stops at the first match with no pipe at all, which
# is both correct under pipefail and far faster over 4,680 directories on NFS.
echo "=== cpseq_fold_cost_probe ==="
echo "node:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "request: -p preempt --qos=qos_preempt --gres=gpu:a6000:1 --cpus-per-task=4 --mem=32G --time=2:00:00"
echo "repo:    $REPO"
echo "out:     $OUT"
nvidia-smi -L

if [ -z "${INPUT:-}" ] && [ -n "${1:-}" ]; then INPUT="$1"; fi
if [ -z "${INPUT:-}" ]; then
  echo "searching for a default input.yaml under $ZFS/cpseq/llm_ladder_pot/predictions"
  INPUT=$(find "$ZFS/cpseq/llm_ladder_pot/predictions" -name input.yaml -print -quit \
          2>/dev/null || true)
fi
echo "input:   ${INPUT:-<none found>}"
echo

[ -n "$INPUT" ] && [ -f "$INPUT" ] || {
  echo "FATAL: no input yaml (looked under $ZFS/cpseq/llm_ladder_pot/predictions)" >&2
  exit 1
}

mkdir -p "$WORK" "$(dirname "$OUT")"

# The repo venv, only to run the probe itself; it imports the cpseq adapter (which
# imports nothing from boltz) and shells out to the boltz venv for every prediction.
PY=$REPO/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: no repo venv python at $PY" >&2; exit 1; }

# The probe's own --boltz-python/--boltz-cache defaults are babel paths, so they are
# passed explicitly. Without them the probe would shell out to a python that does not
# exist and every configuration would record a prediction failure rather than a timing.
BOLTZ_PY="${BOLTZ_PYTHON:-$ZFS/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-$ZFS/boltz_cache}"
[ -x "$BOLTZ_PY" ] || { echo "FATAL: no boltz python at $BOLTZ_PY" >&2; exit 1; }
echo "boltz:   $BOLTZ_PY"
echo "cache:   $BOLTZ_CACHE"

# Extra probe flags, word-split, so a phase can be re-run on its own without editing
# this file. The reason it exists: job 95853 spent its two hours on phases 1 and 2 and was
# killed mid-A7, and A7-vs-A6 is the ONE arm that separates the per-invocation fixed cost
# F from the per-input trunk K -- which is the measurement standing between the collapsed
# trunk being admissible and it being the default (see sec. 2 of
# README_CPSEQ_FOLD_PIPELINE_STANDARD). Re-running phase 3 alone is therefore the next
# thing anyone wants from this probe:
#
#     PROBE_EXTRA="--skip-cost --skip-equivalence" sbatch scripts/cpseq_fold_cost_probe.sbatch
#
# Unquoted on purpose, so several flags in one variable become several arguments.
PROBE_EXTRA="${PROBE_EXTRA:-}"
echo "extra:   ${PROBE_EXTRA:-<none>}"

exec "$PY" -u "$REPO/script_utils/cpseq_fold_cost_probe.py" \
  --input-yaml "$INPUT" \
  --work "$WORK" \
  --out "$OUT" \
  --boltz-python "$BOLTZ_PY" \
  --boltz-cache "$BOLTZ_CACHE" \
  $PROBE_EXTRA
