#!/usr/bin/env bash
#SBATCH --job-name=cpsea_pull
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/user_data/yixiz/logs/cpsea_pull_%j.out
#
# Pull the migration set from rhea onto babel. Built to be preempted.
#
# preempt is sbatch-only and can kill the job at any moment, so nothing here
# is allowed to be a single long uninterruptible stream. Two mechanisms:
#
#   Part A  rsync --files-from with --partial --append-verify. Natively
#           resumable; a rerun moves only what is missing.
#   Part B  the 2.7M preprocessed PDBs, as 55 tar chunks of <=50k files.
#           One streamed tar of all of train would be ~18x faster to write
#           and catastrophic to lose at 80 GB, so it is chunked and each
#           completed chunk drops a marker. A preemption costs one chunk.
#
# rhea is behind a bastion, so every byte crosses one ssh -W relay. Hence
# tar+zstd rather than per-file rsync for Part B: the PDBs are text and
# compress 4.5x, and 2.7M per-file round trips through a relay is the thing
# to avoid.
#
# PREREQUISITE: mkdir -p /data/user_data/yixiz/logs
# Slurm resolves --output before the script runs and cannot create the
# directory itself, so without this the job dies instantly with no log to
# say why -- which reads exactly like a scheduler problem.
#
# Submit:   sbatch scripts/babel_pull.sbatch
# Resume:   sbatch scripts/babel_pull.sbatch        (same command; skips done work)
# Override: sbatch scripts/babel_pull.sbatch NEW=/somewhere/else
#
# Config comes from VAR=value positional arguments, never --export: an
# explicit export list makes slurmd try to rebuild the login environment on
# the compute node, which on some clusters requeues the job and holds it.

set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) echo "ignoring argument: $arg" >&2 ;;
  esac
done

NEW="${NEW:-/data/user_data/yixiz}"
NEWREPO="${NEWREPO:-/home/yixiz/Proteina-Complexa}"
REMOTE="${REMOTE:-rhea}"
RZFS="${RZFS:-/zfsauton/scratch/yixiz}"
MANIFEST_DIR="${MANIFEST_DIR:-$RZFS/migration_min_full}"
ZLEVEL="${ZLEVEL:-3}"

SRC="$RZFS/CPSea/CPSea_full/CPSea/preprocessed/processed"
DST="$NEW/CPSea/CPSea_full/CPSea/preprocessed/processed"
STATE="$NEW/.transfer_state"
CHUNKS="$NEW/.transfer_chunks"

mkdir -p "$NEW" "$NEWREPO" "$DST" "$STATE" "$CHUNKS" "$NEW/logs"

echo "=== $(date) job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} dest=$NEW"
df -h "$NEW" | tail -1

# A preempted job is requeued and lands back here, so check reachability
# every time rather than assuming the last run's network still holds.
if ! ssh -o BatchMode=yes "$REMOTE" true 2>/dev/null; then
  echo "FATAL: cannot ssh to $REMOTE from $(hostname)." >&2
  echo "If this works from the login node, compute nodes have no outbound" >&2
  echo "network and this transfer cannot run as a job -- use tmux on login." >&2
  exit 1
fi
echo "ssh to $REMOTE: ok"

# ---- Part A: everything except the preprocessed tensors -------------------
echo
echo "=== Part A: manifest-driven rsync ==="
rsync -avP "$REMOTE:$MANIFEST_DIR/tier0_home.files" /tmp/
rsync -avP "$REMOTE:$RZFS/migration_min/tier0.files" /tmp/tier0_nonproc.files

# Checkpoints dominate here and do not compress, so no -z. -H matters: 19
# frozen AEs are hardlinks to one inode and would otherwise expand into 19
# separate 4.1 GB copies.
rsync -avPH --partial --append-verify --files-from=/tmp/tier0_nonproc.files \
  "$REMOTE:$RZFS/" "$NEW/"

# The repo-tree results are CSV/JSON, so -z earns its keep here.
rsync -avPHz --partial --append-verify --files-from=/tmp/tier0_home.files \
  "$REMOTE:/zfsauton2/home/yixiz/Proteina-Complexa/" "$NEWREPO/"

echo "Part A done"

# ---- Part B: the 2.7M preprocessed PDBs, in chunks ------------------------
echo
echo "=== Part B: chunked tar stream ==="
rsync -avz --partial "$REMOTE:$MANIFEST_DIR/chunks/" "$CHUNKS/"

total=$(ls -1 "$CHUNKS"/*.list 2>/dev/null | wc -l)
[ "$total" -gt 0 ] || { echo "FATAL: no chunk lists in $CHUNKS" >&2; exit 1; }
echo "$total chunks to consider"

done_n=0; moved=0
# val and test first: ~135k files each, so they prove the pipe works before
# train's 2.46M commit anything substantial.
for list in $(ls -1 "$CHUNKS"/val_*.list "$CHUNKS"/test_*.list "$CHUNKS"/train_*.list 2>/dev/null); do
  name="$(basename "$list" .list)"
  if [ -f "$STATE/$name.done" ]; then
    done_n=$((done_n + 1)); continue
  fi

  n=$(wc -l < "$list")
  echo "--- $name ($n files) $(date +%H:%M:%S)"

  # The chunk list goes to the remote tar on stdin (-T -), and the compressed
  # archive comes back on stdout. pipefail means any stage failing leaves the
  # marker unwritten, so the chunk is retried on the next run.
  if ssh "$REMOTE" "tar -C '$SRC' --ignore-failed-read -cf - -T - | zstd -$ZLEVEL -T2" \
       < "$list" | zstd -d | tar -C "$DST" -xf -; then
    touch "$STATE/$name.done"
    moved=$((moved + 1))
  else
    echo "chunk $name FAILED or was interrupted; it will be retried" >&2
    exit 1
  fi
done

echo
echo "=== $(date) finished: $moved chunks this run, $done_n already done ==="
for s in train val test; do
  echo "$s: $(ls -1 "$DST/$s" 2>/dev/null | wc -l) files"
done
echo "expected: train 2455265, val 135125, test 136463"
