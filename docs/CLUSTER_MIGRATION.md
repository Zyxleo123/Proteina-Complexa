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

Note that `/zfsauton/scratch` is at **100% (36T used, ~317G free)**, so there
is no room to stage a tarball here. Transfer file-by-file over the network;
do not try to `tar` anything up first.

The raw numbers:

| Path | Size | Verdict |
|---|---|---|
| `Proteina-Complexa/training_runs/store` | ~725 GB, 71 runs | **thin it** — see below |
| `lpgen/` | 30 GB | thin it — but it holds two trained checkpoints, see below |
| `LPData/` | 12 GB | 3.9 GB preprocessed worth taking; the rest is re-downloadable archives |
| `CPSea/` | 360 GB | **re-download** from Zenodo; carry index/properties only (~2.3 GB) |
| `*/.venv`, `uv_cache`, `envs` | ~20 GB | **skip** — rebuild from the frozen requirements |

### CPSea: download it again, do not carry it

Measured: `CPSea_full` 290 GB, `CPBind` 68 GB, `CPSea_PDB` 1.6 GB,
`replay_buffers` 1.7 MB. That is ~360 GB, and it is the single biggest item
in the move — but none of it needs to cross the network.

CPSea is public. It is a NeurIPS 2025 Datasets and Benchmarks paper (Yang et
al., Tsinghua), the data is on [Zenodo record
16794716](https://zenodo.org/records/16794716) (~159 GB of archives, v2,
published 2025-08-15, including a `CPBind` subfolder), with scripts at
[github.com/YZY010418/CPSea](https://github.com/YZY010418/CPSea) and a Kaggle
mirror. Our local `CPSea_index.txt` has 2,714,414 rows, matching the paper's
2.71M complexes exactly.

Two caveats that decide what you still have to carry:

1. **The Zenodo record warns v2 has hydrogen-geometry and chirality issues
   and says a newer version is coming.** A fresh download may therefore not
   be the data these checkpoints were trained on. Pin v2 explicitly.
2. The `preprocessed/` halves are produced locally by
   `script_utils/preprocess_cpsea.py`, so they regenerate — but it is a long
   job over 2.71M structures.

So the manifest carries the *identity* of the dataset rather than its bulk:
`CPSea_index.txt` (109 MB) and `CPSea_properties` (508 MB) for the full set,
the same two for CPBind (8.5 MB + 104 MB), and all of `CPSea_PDB` (1.6 GB,
the eval root — too small to be worth re-deriving). That is ~2.3 GB instead
of 360 GB, and it pins exactly which structures were used.

The store is ~725 GB, and it is worth being precise about why, because the
obvious explanation is wrong. It is *not* mostly unpruned intermediates —
only three runs still carry periodic checkpoints. Measured breakdown:

| Category | Size |
|---|---|
| non-EMA checkpoints (optimizer state, = resumability) | 328 GB |
| EMA checkpoints (what evals and sampling load) | 309 GB |
| `frozen_ae.ckpt` copies, 23 runs | 88 GB |
| periodic `chk_epoch=*` non-best | 58 GB |

So most of the store is a legitimate two-checkpoint-per-run footprint. The
saving comes from carrying EMA weights everywhere and optimizer state only
where you actually intend to resume.

Run the manifest builder to cut that down:

```bash
bash scripts/migration_manifest.sh
```

It writes `$ZFS/migration/tier0.files` — one highest-step EMA checkpoint per
real run, the config JSONs and `wandb_run_id.txt`, every
`frozen_ae_source.txt`, all JSON/JSONL/CSV results, the frozen AEs
deduplicated, and `last.ckpt` for the arms listed in `LIVE_ARMS` so those can
be resumed rather than restarted. Measured output:

- **61,858 entries, 205 GB** (~31 GB live-arm optimizer state, ~7.8 GB lpgen, ~2.3 GB CPSea identity)
- 54 runs kept, 14 smoke/overfit runs dropped
- 19 of the 23 frozen-AE copies dropped as byte-identical duplicates

Override the resumable set with e.g. `LIVE_ARMS="cpsea_cyc_ringpe_bond_v4
shared_ae_lpcp_128" bash scripts/migration_manifest.sh`, or pass
`KEEP_SMOKE=1` to include the smoke runs.

### The prune that has already been applied

`scripts/migration_prune.sh` was run on 2026-10-18 and reclaimed
**148.7 GB** (scratch went from 100% to 99%, 310 GB → 453 GB
available). It is idempotent, so re-running it is harmless. What it did:

- Hardlinked 19 byte-identical `frozen_ae.ckpt` copies onto one inode
  (72.6 GB). Every path still resolves; no run was broken.
- Deleted 4 `ignore*.ckpt` files verified byte-identical to the `last*.ckpt`
  beside them. It *skipped* `from_scratch_full/ignore*.ckpt`, where the
  content genuinely differed — worth knowing those two are a real state.
- Deleted 16 periodic non-best `chk_epoch=*` checkpoints. One of them,
  `shared_ae_lpcp_128/.../step=000000050000.ckpt`, was a 1.1 GB truncated
  write (unreadable as a zip) from the job that got killed mid-save.
- Hardlinked `lpgen/weights/foundry/rfd3_latest.ckpt` onto the identical
  `rfd3_released/..._remapped.ckpt` (2.5 GB).

The script defaults to a dry run, re-verifies byte-identity at run time
rather than trusting an earlier scan, and refuses to touch anything listed in
`tier0.files`. After the run, all 36,995 manifest entries were confirmed
still present.

Because of the hardlinking, **use `rsync -H`** if you ever sync whole
directories instead of the manifest; without it the 19 links expand back into
19 separate 4.1 GB copies on the destination.

### lpgen trains outside the store

The LP generator writes to `$ZFS/lpgen/runs`, not to
`training_runs/store`, so the loop over the store misses it entirely. Two
checkpoints there are real trained output and nothing else reproduces them:

- `lpgen/runs/lp_finetune_v1/ckpt/epoch-0602.ckpt` (2.6 GB)
- `lpgen/runs/lp_hotspot_v1/ckpt/epoch-0580.ckpt` (2.6 GB)

Both are now in the manifest. Preflight and `_hsprobe` runs are skipped.

The base weights split two ways. The DiffPepBuilder weights (4.9 GB) have
URLs and md5s in `scripts/_lpgen_preamble.sh` and are listed as `verified` in
`weights_manifest.json`, so they are re-downloadable — skip them. The RFD3
weights are not: **`weights_manifest.json` covers only DiffPepBuilder**, no
URL for RFD3 appears anywhere in the repo, and foundry is a private registry.
So one copy of `rfd3_foundry_2025_12_01_remapped.ckpt` is carried.

`foundry/rfd3_latest.ckpt` turned out to be the same 2.6 GB file under a
second name; it is now a hardlink to the released copy.

The local edits in `lpgen/upstream/{foundry,diffpepbuilder}` need no action:
`script_utils/lpgen_patch_upstream.py` regenerates exactly those lazy-import
patches, and no upstream clone has unpushed commits. Only `foundry/.env`
differs by one line, which you will be rewriting on the new cluster anyway.

### One irreplaceable file to not lose

`cpsea_peptide_exp_b_frozen_mean_dimnorm_aefix/frozen_ae.ckpt` is the **only
surviving copy** of its AE — its source run,
`cpsea_ae_finetune_full_BS32ACCUM4FIX`, has already been deleted from the
store. The manifest keeps it, but it is worth knowing it has no backup.

The pinned AE that most CPSea work is anchored to is
`finetune_full_128/checkpoints/chk_epoch=00000002_step=000000040000-EMA.ckpt`
(4.1 GB). Without it, cross-run loss curves stop being comparable.

### Home and scratch are not equally forgiving

`/zfsauton2/home` has ZFS snapshots on a two-week retention, visible at
`/zfsauton2/home/yixiz/.zfs/snapshot/`. Anything deleted from the repo tree
is recoverable from there for two weeks — including the 27 GB of smoke and
overfit output removed on 2026-10-18, still readable at
`.zfs/snapshot/2026-10-16_15.00.00--2w/Proteina-Complexa/`. There is also a
snapshot named `migration_10_13`, so the admins appear to be snapshotting
ahead of the shutdown.

`/zfsauton/scratch` has **no snapshots at all**. Everything the prune removed
there is permanently gone, which is why that script verifies byte-identity
immediately before each delete rather than trusting a scan.

### The repo tree on home

46 GB of results accumulated under the repo despite the rule that artefacts
live on `$ZFS`. Almost all of it is bulk structures — 362k PDBs across
`inference/` and `evaluation_results/` — but the numbers behind every figure
and table are in the CSV/JSON/JSONL/npz beside them, and those come to only
**511 MB across 18,461 files**. The manifest builder writes them to
`$ZFS/migration/tier0_home.files`, with paths relative to the repo rather
than to `$ZFS`, so they need their own rsync:

```bash
rsync -avPH --partial --append-verify \
  --files-from=/tmp/tier0_home.files \
  yixiz@rhea:/zfsauton2/home/yixiz/Proteina-Complexa/ "$NEWREPO/"
```

The bulk PDBs were deliberately *not* deleted. They regenerate from the
checkpoints in principle, but sampling is stochastic and the seeds are not
recorded everywhere, so they are not reliably reproducible — they are the
actual designs behind the reported numbers.

One naming trap if you ever clean these by hand: the deleted `cpsea_peptide/`
directory is unrelated to the run
`cpsea_peptide_exp_b_frozen_mean_dimnorm_aefix`, which holds the orphaned AE.
A glob like `rm -rf *cpsea_peptide*` would take out the only copy of that AE.

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

# then the 205 GB, resumable
rsync -avPH --partial --append-verify \
  --files-from=/tmp/tier0.files \
  yixiz@rhea:/zfsauton/scratch/yixiz/ "$NEW/"
```

If it dies partway, re-run the same command — `--partial --append-verify`
picks up where it stopped. Verify with:

```bash
rsync -avnH --files-from=/tmp/tier0.files \
  yixiz@rhea:/zfsauton/scratch/yixiz/ "$NEW/" | tail -5
```

A clean second pass lists no files to transfer.

## 5. Datasets — decide before you copy

`CPSea/` and the bulky parts of `LPData/` are *re-derivable* but expensive.
Judgement call per dataset:

- `LPData/preprocessed` (3.9 GB) — take it; cheap and annoying to rebuild.
- `LPData/{PepMerge_release.zip, ProtFrag.tar.gz, PepPC-F_raw_data.tar.gz,
  La-Proteina_AFDB_IDs.zip}` — skip, these are public downloads.
- `CPSea/CPSea_PDB` — this is `CPSEA_DATA_PATH`, split into
  `CPSea_PDB_pdb` (raw), `preprocessed`, `CPSea_PDB_properties` and an index.
  Preprocessing is a long job, so take `preprocessed` + `CPSea_PDB_properties`
  + `CPSea_PDB_index.txt` and leave the raw PDBs if space is tight.
- `CPSea/replay_buffers` — **skip.** These were collected before the
  2026-08-04 fix for replay silently zeroing the receptor, so they have to be
  recollected regardless; copying them moves known-bad data.
- `CPSea/CPBind`, `CPSea/CPSea_full` — needed by the `full` data mode and the
  eval target set. Take them if you intend to re-run evals.
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
