#!/usr/bin/env bash
#SBATCH --job-name=cpseq_kern_ab
#SBATCH --partition=preempt
#SBATCH --qos=qos_preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/kernels/logs/ab_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/kernels/logs/ab_%j.out
#
# Stage 2: does enabling the cuequivariance triangle kernels make a Boltz-2 fold faster,
# and does it change the structure?
#
# Three invocations on ONE input, same seed, same settings, differing only in the kernel
# flag and the env:
#   A  production env, --no_kernels           (what every fold in this project paid for)
#   B  kernels env,    --no_kernels           (control: the env change alone, no flag)
#   C  kernels env,    kernels ENABLED        (the candidate)
#
# B is not padding. Without it a timing difference between A and C could be the env --
# a different cuequivariance in site-packages changes import time and can change which
# code path boltz selects before any kernel runs -- and the whole claim is that only the
# kernel differs. A and B should agree; if they do not, the comparison is void.
#
# a6000 PINNED, and for two reasons this time. The 44 s + 26 s/structure numbers this is
# measured against are a6000 numbers, and the kernels need Ampere or newer: the v100s on
# `legacy` are sm_70 and the ops would either refuse to load or fall back silently, which
# would read as "the kernels do not help". So this job QUEUES for an a6000 rather than
# taking whatever is free.
#
# On `preempt` (gpu1, gpu27, gpu29-31 -- all a6000) rather than `general`, whose seven
# nodes were fully allocated all of 2026-11-03 while preempt had cards free. Each
# partition carries its own restricted QoS, so --qos=qos_preempt is mandatory, not a
# preference: without it the job is rejected with "Invalid qos specification".
# Preemption is survivable here -- each configuration appends its own result row, and a
# requeue re-runs the ones that did not finish.
#
# Stage 1 confirmed on a CPU node (job 95843) that `cuequivariance_torch` 0.10.0 imports
# but `cuequivariance_ops_torch` cannot: `libcuda.so.1: cannot open shared object file`,
# the CUDA DRIVER, which exists only on a GPU node. So the ops import is still unproven
# and this job is where it is decided -- if C fails to import here, on an Ampere card
# with a driver present, the kernels are genuinely unavailable at 0.10.0 and the comment
# in adapters/boltz2.py is right for a reason it did not give.
#
# WHAT THIS DOES NOT DO: change production. `adapters/boltz2.py` still hardcodes
# `--no_kernels` and must keep doing so until this job answers, because the kernel
# setting is NOT in the cache key. `Boltz2Config.to_dict()` carries diffusion_samples,
# recycling_steps, sampling_steps, use_potentials, seed and affinity -- and no kernel
# field -- so `input_hash` cannot tell the two regimes apart. Flip kernels on today and
# all ~29k cached predictions keep their keys while new ones are computed differently,
# which would put two numerics regimes inside the paired potentials comparison whose
# entire design is that exactly one setting moves. Adopting kernels therefore requires
# a cache-key or BOLTZ_REV change FIRST. That is why this is a probe in a sibling env.
#
# GATE: reads stage 1's status file and exits 0 with an explanation if the install did
# not happen, rather than failing and looking like a crash.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
# ONE env, two PYTHONPATHs. Stage 1 installs the kernels into a shadow directory
# instead of cloning the 4.4 GB env (the clone died mid-copy on a full scratch, job
# 95767), so A/B/C all run the SAME production interpreter and differ only in whether
# $SHADOW is on PYTHONPATH and whether --no_kernels is passed. That is a tighter
# comparison than two envs could be: there is no second site-packages to differ.
BOLTZ_ENV="${BOLTZ_ENV:-$ZFS/envs/boltz2/.venv}"
SHADOW="${SHADOW:-$ZFS/cpseq/kernels/site}"
STATUS="${STATUS:-$ZFS/cpseq/kernels/install_status.json}"
BOLTZ_CACHE="${BOLTZ_CACHE:-$ZFS/boltz_cache}"
RUN_ID="${RUN_ID:-kernab_$(date +%Y%m%d_%H%M%S)}"
WORK="${WORK:-$ZFS/cpseq/kernels/$RUN_ID}"
SEED="${SEED:-42}"
DS="${DS:-1}"
RECYCLE="${RECYCLE:-3}"
STEPS="${STEPS:-200}"
POTENTIALS="${POTENTIALS:-1}"

# `-print -quit`, NOT `| sort | head -1`. This job died as 95851 with a zero-byte log
# and exit 141 -- 128+13, SIGPIPE: `head` exits after the first line, `sort` is killed by
# the pipe, `set -o pipefail` makes the assignment fail and `set -e` ends the script
# before a single echo. Identical death in cpseq_fold_cost_probe.sbatch as job 95827.
# `-print -quit` stops at the first match with no pipe, so there is nothing to signal,
# and it returns in 0 s instead of 35 s over 4,680 directories on NFS.
#
# $1 is honoured first, and PINNING THE INPUT IS PREFERRED: directory order is not
# sorted order, so the default search is not reproducible across runs.
if [ -z "${INPUT:-}" ] && [ -n "${1:-}" ]; then INPUT="$1"; fi
if [ -z "${INPUT:-}" ]; then
  INPUT=$(find "$ZFS/cpseq/llm_ladder_pot/predictions" -name input.yaml -print -quit \
          2>/dev/null || true)
fi

mkdir -p "$WORK"

echo "=== cpseq_kernels_ab ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=a6000:1"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true
echo "input:     $INPUT"
echo "work:      $WORK"
echo "settings:  seed=$SEED ds=$DS recycle=$RECYCLE steps=$STEPS potentials=$POTENTIALS"
echo "========================"

if [ ! -f "$STATUS" ]; then
  echo
  echo "GATE: no install status at $STATUS -- stage 1 never ran."
  echo "      This is 'upstream never ran', not 'the kernels failed'. Exiting 0."
  exit 0
fi
cat "$STATUS"
OK=$(python3 -c "import json,sys;print(json.load(open('$STATUS'))['ok'])")
if [ "$OK" != "True" ]; then
  echo
  echo "GATE: stage 1 reported the install did not succeed. Nothing to A/B. Exiting 0."
  exit 0
fi
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "GATE: no input yaml found. Exiting 0."; exit 0; }

run_one() {
  local label="$1" shadow="$2" use_kernels="$3"
  local out="$WORK/$label"
  rm -rf "$out"; mkdir -p "$out"
  local flags=(
    "$BOLTZ_ENV/bin/python" -m boltz.main predict "$INPUT"
    --out_dir "$out"
    --cache "$BOLTZ_CACHE"
    --diffusion_samples "$DS"
    --recycling_steps "$RECYCLE"
    --sampling_steps "$STEPS"
    --seed "$SEED"
    --output_format pdb
    --accelerator gpu
    --devices 1
    --write_full_pae
    --preprocessing-threads 1
  )
  [ "$POTENTIALS" = "1" ] && flags+=(--use_potentials)
  # The one axis. `--no_kernels` is what production passes; omitting it is what lets
  # boltz reach for cuequivariance.
  [ "$use_kernels" = "0" ] && flags+=(--no_kernels)

  echo
  echo "--- $label: shadow=${shadow:-<none>} kernels=$use_kernels ---"
  local t0 t1 rc
  t0=$(date +%s)
  set +e
  # PYTHONPATH is the only env difference between the configurations. Set empty rather
  # than unset for the no-shadow runs so the variable is never inherited from the job.
  PYTHONPATH="$shadow" "${flags[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt"
  rc=$?
  set -e
  t1=$(date +%s)
  local el=$(( t1 - t0 ))
  local n
  n=$(find "$out" -name '*.pdb' | wc -l)
  echo "$label: rc=$rc  wall=${el}s  pdb=$n"
  if [ "$rc" -ne 0 ]; then
    echo "  stderr tail:"
    tail -8 "$out/stderr.txt" | sed 's/^/    /'
  fi
  # One JSON line per configuration, so the comparison can be redrawn without re-folding.
  "$REPO/.venv/bin/python" - "$WORK/results.jsonl" "$label" "${shadow:-none}" \
           "$use_kernels" "$el" "$rc" "$n" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
row = {"label": sys.argv[2], "shadow": sys.argv[3], "kernels": sys.argv[4] == "1",
       "wall_s": int(sys.argv[5]), "rc": int(sys.argv[6]), "n_pdb": int(sys.argv[7])}
with p.open("a") as fh:
    fh.write(json.dumps(row) + "\n")
EOF
}

# A: production exactly as it runs today.
# B: the shadow on PYTHONPATH but --no_kernels still passed. The CONTROL -- it isolates
#    the import from the kernel. If A and B disagree, the comparison is void, because
#    something other than the kernel changed with the path.
# C: the shadow on PYTHONPATH and the kernels enabled. The candidate.
run_one A_prod_nokernels "" 0
run_one B_shadow_nokernels "$SHADOW" 0
run_one C_shadow_kernels "$SHADOW" 1

echo
echo "=== results ==="
cat "$WORK/results.jsonl"

# Structures are compared, not just times. A kernel that is faster and returns a
# different structure is not an optimisation, it is a different model -- and since the
# kernel setting is absent from `input_hash`, that difference would be invisible to
# every cache in this project. RMSD between A and C is the number that says which it is.
echo
echo "=== structural comparison (A vs C) ==="
"$REPO/.venv/bin/python" - "$WORK" <<'EOF'
import sys
from pathlib import Path
work = Path(sys.argv[1])
def one(label):
    hits = sorted(work.glob(f"{label}/**/*.pdb"))
    return hits[0] if hits else None
a, c = one("A_prod_nokernels"), one("C_shadow_kernels")
if not a or not c:
    print(f"  cannot compare: A={a} C={c}")
    raise SystemExit(0)

def ca(p):
    out = {}
    for ln in p.read_text().splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out[(ln[21], int(ln[22:26]))] = (
                float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            )
    return out

A, C = ca(a), ca(c)
shared = sorted(set(A) & set(C))
print(f"  CA atoms: A={len(A)} C={len(C)} shared={len(shared)}")
if not shared:
    raise SystemExit(0)
import math
sq = sum((A[k][i] - C[k][i]) ** 2 for k in shared for i in range(3))
rmsd = math.sqrt(sq / len(shared))
mx = max(math.dist(A[k], C[k]) for k in shared)
print(f"  in-frame CA RMSD A vs C: {rmsd:.4f} A   max dev: {mx:.4f} A")
print("  (no superposition: both come from the same input and seed, so a kernel that")
print("   changes nothing should give ~0. A large value means the kernels change the")
print("   prediction, and `input_hash` cannot tell -- see the header.)")
EOF

echo
echo "=== cpseq_kernels_ab OK ==="
