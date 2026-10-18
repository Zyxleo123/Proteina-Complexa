# Moving this project off the autonlab cluster

Written 2026-10-18, ahead of the cluster going down. Steps 1 and 2 have to
happen **while rhea is still up**. Step 3 onwards can be done from the new
cluster at your own pace, as long as rhea is reachable.

---

## 0. What is already committed

Every working tree was dirty when this started. All of it is now committed
and **nothing is pushed** — the commits exist only on this cluster's disk.

| Branch | Worktree | Unpushed |
|---|---|---|
| `cpsea` | repo root | 1 commit (corruption-ceiling, provenance pairs, cp2lp backbone arms, lpgen hotspots) |
| `dflow-lp2cp` | `.claude/worktrees/dflow-lp2cp` | 1 commit — **the whole D-Flow arm**, which existed nowhere else |
| `cpsea-genkic` | `.claude/worktrees/genkic` | 6 commits (incl. new terminal-feasibility work) |
| `cp2lp-ca-internal` | `.claude/worktrees/ca-internal` | 10 commits |
| `flowedit-lp2cp` | `~/worktrees/flowedit` | 9 commits |

The last three have **no upstream configured at all** — they have never been
pushed anywhere. If this disk goes away before step 1, they are gone.

## 1. Push the code — do this first

```bash
cd /zfsauton2/home/yixiz/Proteina-Complexa
git push origin cpsea
git push -u origin cp2lp-ca-internal
git push -u origin cpsea-genkic
git push -u origin dflow-lp2cp
git push -u origin flowedit-lp2cp
```

Then confirm nothing is left behind:

```bash
git for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads/
git worktree list | while read -r w _; do
  echo "$w: $(git -C "$w" status --porcelain | grep -vE '\.venv|CPSea_data|env_files|evaluation_results' | wc -l) dirty"
done
```

Every branch should read `[gone]`-free and in sync; every worktree `0 dirty`.

## 2. Carry the secrets by hand

`.env` is gitignored and holds `GITLAB_TOKEN`, `WANDB_API_KEY` and
`HF_TOKEN`. It will not travel with the push. Copy it somewhere safe
(password manager, or `scp` it straight to the new cluster) — **not** into
the repo, and not into any file you later `git add`.

`env.sh` is also gitignored but contains no secrets, only paths; it is easier
to re-derive from `.env` on the new machine than to copy.

## 3. Work out what data to actually move

Everything real lives under `/zfsauton/scratch/yixiz`. The `store/` directory
inside the repo on home looks like a second copy but is a 512-byte symlink
farm pointing back into scratch — ignore it.

The raw numbers:

| Path | Size | Verdict |
|---|---|---|
| `Proteina-Complexa/training_runs/store` | ~700 GB, 71 runs | **thin it** — see below |
| `lpgen/` | 30 GB | thin it (11 GB is re-downloadable weights, 7.6 GB venvs, 4.2 GB uv cache) |
| `LPData/` | 12 GB | 3.9 GB preprocessed worth taking; the rest is re-downloadable archives |
| `CPSea/` | large, still measuring | raw + preprocessed CPSea; see step 5 |
| `*/.venv`, `uv_cache`, `envs` | ~20 GB | **skip** — rebuild from the frozen requirements |

The store is 700 GB because nothing ever prunes it: each run keeps every
periodic checkpoint *and* its EMA twin, and 23 runs additionally hold a full
4.1 GB copy of the AE they were frozen against.

Run the manifest builder to cut that down:

```bash
bash scripts/migration_manifest.sh
```

It writes `$ZFS/migration/tier0.files` — one highest-step EMA checkpoint per
real run, the config JSONs, every `frozen_ae_source.txt`, all JSON/JSONL/CSV
results, and the frozen AEs deduplicated by size. Measured output:

- **36,932 entries, 165 GB** (down from ~700 GB)
- 54 runs kept, 14 smoke/overfit runs dropped
- 19 of the 23 frozen-AE copies dropped as byte-identical duplicates

Pass `KEEP_SMOKE=1` if you want the smoke runs too.

### One irreplaceable file to not lose

`cpsea_peptide_exp_b_frozen_mean_dimnorm_aefix/frozen_ae.ckpt` is the **only
surviving copy** of its AE — its source run,
`cpsea_ae_finetune_full_BS32ACCUM4FIX`, has already been deleted from the
store. The manifest keeps it, but it is worth knowing it has no backup.

The pinned AE that most CPSea work is anchored to is
`finetune_full_128/checkpoints/chk_epoch=00000002_step=000000040000-EMA.ckpt`
(4.1 GB). Without it, cross-run loss curves stop being comparable.

## 4. Pull the data from the new cluster

Run the transfer **on the new cluster, pulling from rhea** — not as a job
here. The transfer then lives on the machine that is not going down, and
`rsync` resumes cleanly if the link drops.

```bash
# on the NEW cluster, inside tmux/screen
NEW=/path/to/new/scratch/yixiz
mkdir -p "$NEW"

# bring the file list over first
rsync -avP yixiz@rhea:/zfsauton/scratch/yixiz/migration/tier0.files /tmp/

# then the 165 GB, resumable
rsync -avP --partial --append-verify \
  --files-from=/tmp/tier0.files \
  yixiz@rhea:/zfsauton/scratch/yixiz/ "$NEW/"
```

If it dies partway, re-run the same command — `--partial --append-verify`
picks up where it stopped. Verify with:

```bash
rsync -avn --files-from=/tmp/tier0.files \
  yixiz@rhea:/zfsauton/scratch/yixiz/ "$NEW/" | tail -5
```

A clean second pass lists no files to transfer.

## 5. Datasets — decide before you copy

`CPSea/` and the bulky parts of `LPData/` are *re-derivable* but expensive.
Judgement call per dataset:

- `LPData/preprocessed` (3.9 GB) — take it; cheap and annoying to rebuild.
- `LPData/{PepMerge_release.zip, ProtFrag.tar.gz, PepPC-F_raw_data.tar.gz,
  La-Proteina_AFDB_IDs.zip}` — skip, these are public downloads.
- `CPSea/CPSea_PDB` — this is `CPSEA_DATA_PATH`, the raw + preprocessed CPSea
  set. Preprocessing it is a long job. If the new cluster has room, take the
  preprocessed half and leave the raw PDBs.
- `lpgen/weights/{rfd3_released, diffpepbuilder, unimomo}` — skip the weights,
  **take `weights_manifest.json`** (already in the manifest) so you can
  re-fetch the exact revisions.

## 6. Rebuild the environment

There is no `uv.lock` in this repo, so `env/requirements-frozen-20261018.txt`
is the only record of the versions the current results were produced with.
It was frozen from the working `.venv` on 2026-10-18 — 236 packages,
Python 3.12.13.

```bash
uv venv --python 3.12.13 .venv
uv pip install -r env/requirements-frozen-20261018.txt
```

Then rewrite `.env` for the new paths — at minimum `ZFS`,
`PROTEINA_ZFS_PATH`, `DATA_PATH`, `CPSEA_DATA_PATH`, `CPSEA_FULL_DATA_PATH`,
`CKPT_PATH` and `CPSEA_AE_CKPT_PATH` — and re-point the
`CLUSTER_*` block at the new scheduler. Then `source env.sh`.

Recreate the store symlink farm, or checkpoints will land on the home
filesystem instead of scratch:

```bash
mkdir -p "$PROTEINA_ZFS_PATH/training_runs/store"
ln -s "$PROTEINA_ZFS_PATH/training_runs/store" store
```

## 7. Before you trust the new setup

- `myfree` is an autonlab shell function; it will not exist on the new
  cluster. Every `#SBATCH` partition/GPU choice in `scripts/` is keyed to
  autonlab's node pool and needs revisiting against the new one.
- The `--export` workaround (submitter writes a timestamped env file, passes
  its path as a positional argument) exists because of an autonlab-specific
  slurmd failure. It is harmless elsewhere, but if the new cluster handles
  `--export` properly you may not need it.
- `sacct` is broken here and exits 0 with empty output. Check whether it
  works on the new cluster before assuming a job vanished.
- Re-run one short smoke job per arm before trusting any long chain, and read
  the first `.out` rather than relying on exit status.
