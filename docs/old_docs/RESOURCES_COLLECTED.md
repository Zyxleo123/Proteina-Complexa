# Resources collected on babel after the rhea migration

Written 2026-10-19. Everything here was re-fetched from source, because the B2
bucket carried only our own trained output -- `LNR`, `LPData` and every
community weight were absent from it. Paths are on `/data/user_data/yixiz`,
which only compute nodes mount.

## Datasets

Zenodo record 13373108 (PepBench / ProtFrag / LNR), extracted to the layout
`script_utils/preprocess_lp.py` documents:

| Path | Contents | sha256 of archive |
|---|---|---|
| `LNR/` | 93 targets (`test.txt`: pdb_id, receptor chain, peptide chain), 96 PDBs | `3997e374fd2d46a4` |
| `LPData/PepBench/train_valid/` | 6,142 PDBs, plus `train.cluster` (4,157) / `valid.cluster` (114) | `adf82bd2e9870781` |
| `LPData/ProtFrag/` | 70,498 fragment PDBs | `61726669486be9b7` |

`LPData/preprocessed` (~3.9 GB) is **not** restored: it was never pushed to B2.
It regenerates from the three trees above -- see `preprocess_lp.py`, which also
holds LNR out of the training pool at 0.40 sequence identity and writes
`manifest/lnr_excluded.csv`.

The PepBench `*.cluster` files are a pre-existing clustering, usable for a
target-family split without inventing one.

## Model weights

| Model | Path | Source |
|---|---|---|
| Complexa protein/ligand/AME (6 ckpts, 18 GB) | `ckpts/` | NGC, via `complexa download --complexa-all` |
| AF2 params (5.3 GB) | `community_models/ckpts/AF2/` | `complexa download --all` |
| ESM2 650M (2.5 GB) | `community_models/ckpts/ESM2/` | HuggingFace `facebook/esm2_t33_650M_UR50D` |
| ProteinMPNN / LigandMPNN | `community_models/{ProteinMPNN,LigandMPNN}/` | `complexa download --all` |
| RF3 latest (3.0 GB) | `community_models/ckpts/RF3/` | `https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt` |
| AF3 (1.1 GB) | `community_models/ckpts/AF3/af3.bin` | `https://storage.googleapis.com/alphafold3/af3.bin.zst` -- see `AF3/TERMS.md` |
| Boltz-2 conf + aff + mols + ccd (6.2 GB) | `boltz_cache/` | HuggingFace `boltz-community/boltz-2` |

RF3 is **public** at files.ipd.uw.edu and needs no credential. Do not confuse it
with RFD3, lpgen's diffusion weights, which do come from a private foundry
registry -- that is the thing `docs/old_docs/CLUSTER_MIGRATION.md` calls unfetchable.

`GITLAB_TOKEN` is read only by `env/docker-ops.sh` for `docker login`, so it is
not needed on the uv runtime.

## Upstream source, pinned

Clones under `upstream/`, kept off `/home` and out of git:

- `boltz` b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc (2026-05-29) -- https://github.com/jwohlwend/boltz
- `highfold3` 2ff1ea31e59f3bac70e45aadde60a4a8fed6e199 (2025-09-24) -- https://github.com/hongliangduan/HighFold3
- `alphafold3` 131ab9e7420066523c62546199243f7a9b3e3e78 (2026-10-19) -- https://github.com/google-deepmind/alphafold3

## Two constraints found while collecting

**AF3's genetic databases do not fit.** `alphafold3/docs/installation.md` wants
up to 1 TB; the quota is 500 GB with ~225 GB free. AF3 accepts precomputed
alignments inline (`unpairedMsa` / `pairedMsaPath` in the input JSON), so the
route is to compute MSAs once per target context and cache them, skipping
`fetch_databases.sh` entirely. AF3 also asks for >=64 GB RAM.

**The frozen requirements are internally inconsistent.** `torchaudio==2.11.0`
cannot load against `torch==2.7.0` (`undefined symbol: torch_library_impl`),
while `torchvision==0.22.0` is the correct pair. Nothing here imports
torchaudio, but `transformers` probes for it, so a broken copy breaks any ESM
import -- which is what stopped the ESM2 download. Pinned to
`torchaudio==2.7.0+cu126` instead; this deviates from the freeze deliberately,
because the freeze cannot have worked as written.

`complexa download --status` reports a weight as installed when the file merely
*exists*. It said RF3 was installed at 92 MB of 3.0 GB. Check byte counts.
