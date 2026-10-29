#!/bin/bash
# Shared defaults for the CPSea sequence-proposer (masked-LM) arms.
#
# Sourced by every job in the chain, so a knob is defined once rather than copied across
# job scripts. The submitter writes a per-submission env file that is sourced AFTER this,
# so a submission can override any default without editing a committed file.
#
# Deliberately self-sufficient: it must survive `env -i`, because a Slurm job on a compute
# node does not inherit the login environment. No `myfree`, no shell functions, no
# relative paths.

# --- where things live. Nothing is written to home: logs, env files, corpora, run dirs
# and checkpoints all land on scratch. A relative `#SBATCH --output=` would put logs in
# the repo on home, so every path here is absolute.
export CPSEQ_MLM_ROOT="${CPSEQ_MLM_ROOT:-/zfsauton/scratch/yixiz/cpseq_mlm}"
export CPSEQ_MLM_CORPUS_DIR="${CPSEQ_MLM_CORPUS_DIR:-$CPSEQ_MLM_ROOT/corpus}"
export CPSEQ_MLM_RUNS_DIR="${CPSEQ_MLM_RUNS_DIR:-$CPSEQ_MLM_ROOT/runs}"
export CPSEQ_MLM_LOG_DIR="${CPSEQ_MLM_LOG_DIR:-/zfsauton/scratch/yixiz/slurm_logs}"

# HuggingFace cache on scratch. The default is ~/.cache/huggingface, which would put
# 2.6 GB of weights on home.
export HF_HOME="${HF_HOME:-/zfsauton/scratch/yixiz/hf}"

# --- data
export CPSEA_METADATA_DIR="${CPSEA_METADATA_DIR:-/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea/preprocessed/metadata}"
export CPSEQ_MLM_POCKET_RADIUS="${CPSEQ_MLM_POCKET_RADIUS:-8.0}"
# Shards for the corpus array. This job is I/O bound, and the MEASURED scaling is the
# opposite of what per-shard rates suggest -- read the aggregate, not the per-shard number:
#
#   concurrent shards | per-shard | AGGREGATE
#   1 (idle node)     | 51 rows/s |  51 rows/s
#   2                 | 15.1      |  30
#   32                |  6.3      | 202
#
# So more concurrency wins on total throughput even though each shard slows ~8x; the
# per-shard collapse is contention, not waste. 24 shards is NOT a throughput choice, it is
# the QOS ceiling: every array task counts against MaxSubmitJobsPerUser (~33), so 24 shards
# + 2 trainers = 26 fits with headroom while 32 + 2 does not. At ~7.5 rows/s per shard that
# is 102k rows in ~3.8 h, inside the 8 h --time.
#
# Do NOT re-derive any of this from a login-node timing: the same code measures 0.019 s/row
# on warm page cache, which is 5x optimistic and would under-size --time.
export CPSEQ_MLM_NUM_SHARDS="${CPSEQ_MLM_NUM_SHARDS:-24}"
# Val is 134,394 rows against train's 2,444,209, so it does NOT need train's shard count:
# 4 shards is ~34k rows each (~51 min). Sharding it like train wasted 28 submission slots
# and is what tripped the limit below.
export CPSEQ_MLM_VAL_SHARDS="${CPSEQ_MLM_VAL_SHARDS:-4}"
# Refuse to submit past the QOS ceiling. Every ARRAY TASK counts as a submitted job here,
# so 32 val tasks + 32 train tasks + 2 trainers is 66 and sbatch rejects the overflow with
#   sbatch: error: QOSMaxSubmitJobPerUserLimit
# leaving a half-submitted chain. The exact limit is not queryable -- `sacctmgr`, like
# `sacct`, cannot reach slurmdbd on this cluster -- but 32 val tasks plus 1 other job
# submitted fine and the 33rd failed, so the ceiling is ~33. 30 leaves headroom for jobs
# this chain does not own.
export CPSEQ_MLM_MAX_SUBMIT="${CPSEQ_MLM_MAX_SUBMIT:-30}"
# Reader threads per shard. The job is I/O bound, not CPU bound: measured cold on 200 rows,
# 1 thread took 42.4 s (7.5 s of it in userspace) and 8 threads took 3.9 s -- a real 10.9x,
# both on cold files, with byte-identical output. Keep this at or below --cpus-per-task*4;
# the gain is from overlapping NFS waits, not from compute.
export CPSEQ_MLM_WORKERS="${CPSEQ_MLM_WORKERS:-8}"

# --- model / training
# PepMLM is ESM-2-650M already fine-tuned for target-conditioned binder infilling, which
# is this task's shape. `facebook/esm2_t33_650M_UR50D` is the control for "did the PepMLM
# warm start matter"; it is architecturally identical (EsmForMaskedLM, 33 layers, hidden
# 1280, vocab 33), so the two are interchangeable here.
export CPSEQ_MLM_MODEL="${CPSEQ_MLM_MODEL:-ChatterjeeLab/PepMLM-650M}"
export CPSEQ_MLM_MAX_STEPS="${CPSEQ_MLM_MAX_STEPS:-20000}"
export CPSEQ_MLM_BATCH_SIZE="${CPSEQ_MLM_BATCH_SIZE:-32}"
export CPSEQ_MLM_ACCUM="${CPSEQ_MLM_ACCUM:-1}"
# 1e-5 on a 650M pLM fine-tune. Project memory records an LP-generator run DIVERGING on
# an AF3-scale lr of 1.8e-3 at small effective batch, so this errs low on purpose.
export CPSEQ_MLM_LR="${CPSEQ_MLM_LR:-1e-5}"
export CPSEQ_MLM_WARMUP="${CPSEQ_MLM_WARMUP:-500}"
export CPSEQ_MLM_EVAL_EVERY="${CPSEQ_MLM_EVAL_EVERY:-500}"
export CPSEQ_MLM_SAVE_EVERY="${CPSEQ_MLM_SAVE_EVERY:-2000}"
export CPSEQ_MLM_EVAL_N="${CPSEQ_MLM_EVAL_N:-800}"
export CPSEQ_MLM_CYC_DROPOUT="${CPSEQ_MLM_CYC_DROPOUT:-0.15}"
export CPSEQ_MLM_SEED="${CPSEQ_MLM_SEED:-42}"
# Clusters withheld from TRAIN, one id per line. Empty means the parquet split is the only
# holdout; set it to the eval targets' clusters before any number is quoted as held out.
export CPSEQ_MLM_HOLDOUT_CLUSTERS="${CPSEQ_MLM_HOLDOUT_CLUSTERS:-}"
