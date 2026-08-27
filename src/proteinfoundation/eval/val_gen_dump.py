"""Persist a handful of val_generation complexes to disk for the Rosetta sidecar.

The in-loop binding metrics (`sampled_binder_metrics`) are pure geometry -- cheap enough to run
every validation. Rosetta interface dG is NOT: it needs a full-atom relax + InterfaceAnalyzer per
complex (seconds of CPU each), so it must never run on the GPU training path. Instead this dumps a
small, capped sample of val_gen complexes (binder + receptor, one PDB each) plus a manifest row per
complex, and a **separate CPU job** (`scripts/score_val_gen_rosetta.sbatch`) picks them up, scores
them, and appends the dG to a JSONL keyed by step. wandb then reads that JSONL back, lagging the
live curve by a checkpoint -- which is fine.

The dump is OFF unless a directory is configured, and capped (`max_samples`) so it costs a few PDB
writes, not a populationwide dump. The all-atom coords come straight from the AE decode, so no MPNN
sequence design is needed before Rosetta -- the decoded side chains ARE the sequence to score.
"""

from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import torch
from loguru import logger

from openfold.np.residue_constants import atom_order

from proteinfoundation.cyclization.constants import CYCLIZATION_TYPE_TO_NAME
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

CA_IDX = atom_order["CA"]
NM_TO_ANG = 10.0

# Chain-letter convention passed through to Rosetta: receptor is chain A (index 0), binder is the
# LAST chain (B, index 1) -- matching `rosetta_energy` / PepGLAD's "binder is the ligand chain".
_TARGET_CHAIN = "A"
_BINDER_CHAIN = "B"


def _residue_mask(m: torch.Tensor) -> torch.Tensor:
    """[.., 37] atom mask -> [..] residue mask; pass a residue mask through unchanged."""
    return m.any(dim=-1) if m.dim() >= 2 and m.shape[-1] == 37 else m


@torch.no_grad()
def dump_val_gen_complexes(
    out_dir: str,
    step: int,
    epoch: int,
    *,
    coors_nm: torch.Tensor,          # [B, L, 37, 3] sampled binder, nm
    aatype: torch.Tensor,            # [B, L] sampled residue types
    mask: torch.Tensor,              # [B, L] valid binder residues
    x_target: torch.Tensor | None,   # [B, T, 37, 3] or [B, T, 3] receptor, nm
    seq_target: torch.Tensor | None,  # [B, T] receptor residue types
    target_mask: torch.Tensor | None,  # [B, T, 37] or [B, T]
    cyc_i: torch.Tensor | None = None,   # [B] predicted closing-bond endpoints / type
    cyc_j: torch.Tensor | None = None,
    cyc_type: torch.Tensor | None = None,
    rank: int = 0,
    max_samples: int = 8,
) -> int:
    """Write up to ``max_samples`` complex PDBs for this (step, rank) and append manifest rows.

    Returns the number of complexes written. A no-op (returns 0) unless ``x_target`` and
    ``seq_target`` are present -- there is no interface to score without a receptor.
    """
    if x_target is None or seq_target is None or target_mask is None:
        return 0

    step_dir = os.path.join(out_dir, f"step_{step:08d}")
    os.makedirs(step_dir, exist_ok=True)
    manifest = os.path.join(step_dir, f"manifest_rank{rank}.jsonl")

    b = coors_nm.shape[0]
    n = min(b, max_samples)

    bca = coors_nm.detach().cpu().float().numpy() * NM_TO_ANG          # [B, L, 37, 3] Angstrom
    baa = aatype.detach().cpu().long().numpy()
    bm = mask.detach().cpu().bool().numpy()

    # Receptor coords may arrive as Ca-only [B, T, 3]; Rosetta needs a full residue, so we only dump
    # complexes when the receptor carries atom37 coordinates.
    if x_target.dim() != 4:
        return 0
    txyz = x_target.detach().cpu().float().numpy() * NM_TO_ANG          # [B, T, 37, 3]
    tseq = seq_target.detach().cpu().long().numpy()
    tm = _residue_mask(target_mask).detach().cpu().bool().numpy()       # [B, T]

    ci = cyc_i.detach().cpu().long().numpy() if cyc_i is not None else None
    cj = cyc_j.detach().cpu().long().numpy() if cyc_j is not None else None
    ct = cyc_type.detach().cpu().long().numpy() if cyc_type is not None else None

    written = 0
    with open(manifest, "a") as mf:
        for i in range(n):
            b_sel = bm[i]
            t_sel = tm[i]
            if b_sel.sum() == 0 or t_sel.sum() == 0:
                continue

            # Receptor first (chain A), binder last (chain B): one concatenated atom37 array.
            binder_xyz = bca[i][b_sel]                                          # [Nb, 37, 3]
            pos = np.concatenate([txyz[i][t_sel], binder_xyz], axis=0)          # [Nt+Nb, 37, 3]
            aa = np.concatenate([tseq[i][t_sel], baa[i][b_sel]], axis=0)         # [Nt+Nb]
            chain_index = np.concatenate(
                [np.zeros(int(t_sel.sum())), np.ones(int(b_sel.sum()))]
            ).astype(np.int64)

            name = f"epoch{epoch}_step{step}_rank{rank}_bid{i}.pdb"
            pdb_path = os.path.join(step_dir, name)
            write_prot_to_pdb(
                prot_pos=pos, file_path=pdb_path, aatype=aa,
                chain_index=chain_index, overwrite=True, no_indexing=True,
            )

            # Closing-bond endpoints as 0-based positions in the passed binder-row order. `to_pdb`
            # numbers the k-th passed row resSeq k+1 and OMITS any all-zero row; `rosetta_energy`
            # then looks up pdb2pose(chain, endpoint + 1) -- i.e. it re-adds the +1 -- so we pass the
            # 0-based position k, and pdb2pose finds row k iff it was written. The original NaNs came
            # from j landing on a DROPPED row (j=13 on an 11-mer -> pdb2pose('B',14) not found), so
            # the fix is to only record endpoints that point at WRITTEN rows. `written_mask[k]` flags
            # the rows to_pdb keeps.
            row: dict = {"pdb": name, "binder_chain": _BINDER_CHAIN, "target_chains": _TARGET_CHAIN,
                         "epoch": epoch, "step": step, "rank": rank, "bid": i}
            if ci is not None and cj is not None and ct is not None:
                written_mask = np.abs(binder_xyz).sum(axis=(1, 2)) > 1e-7        # [Nb] rows to_pdb keeps
                tname = CYCLIZATION_TYPE_TO_NAME.get(int(ct[i]))
                w = np.nonzero(written_mask)[0]
                if tname == "mainchain" and w.size >= 2:
                    # Head-to-tail by construction (CPSea macrocycles close first<->last): use the
                    # actual first/last WRITTEN residue, robust to the model's raw index landing on a
                    # dropped (empty) row -- which is exactly what produced j=13 on an 11-mer.
                    row["cyclization_type"] = tname
                    row["cyclization_i"] = int(w[0])
                    row["cyclization_j"] = int(w[-1])
                elif tname in ("disulfide", "isopeptide"):
                    # Side-chain-specific bond: map the model's full-grid endpoints through b_sel to
                    # kept-row positions, and only record if BOTH rows are written.
                    keep_idx = np.nonzero(b_sel)[0]
                    remap = {int(g): loc for loc, g in enumerate(keep_idx)}
                    li, lj = remap.get(int(ci[i])), remap.get(int(cj[i]))
                    if (li is not None and lj is not None
                            and 0 <= li < len(written_mask) and 0 <= lj < len(written_mask)
                            and written_mask[li] and written_mask[lj]):
                        row["cyclization_type"] = tname
                        row["cyclization_i"] = int(li)
                        row["cyclization_j"] = int(lj)
                # else: leave no cyclization -> Rosetta scores it linear (finite dG), never NaN.
            mf.write(json.dumps(row) + "\n")
            written += 1

    return written


# ---------------------------------------------------------------------------------------------
# Orchestration: log the sidecar's dG back into the SAME wandb run, and (re)submit it periodically.
# Both are called only from the training process, rank 0. The training process is the SOLE wandb
# writer -- the sidecar only ever writes JSONL -- so there are never two processes logging to one
# run. The dG lags the live curve by however long scoring takes, which is expected and fine.
# ---------------------------------------------------------------------------------------------


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A NUL-corrupted / half-written line (parallel appends): skip for reading, but it
                # stays in the file for a human to see. Never treat it as absence-of-data.
                continue
    return rows


def log_rosetta_dg_to_wandb(out_jsonl: str, wandb_run, logged_steps: set[int]) -> int:
    """Log per-step mean/median interface dG from the sidecar's JSONL into ``wandb_run``.

    Idempotent: only steps not already in ``logged_steps`` are emitted, and the CURRENT max step is
    skipped (its complexes may still be scoring) so each step is logged exactly once, when settled.
    dG is plotted against its true source step via a wandb step_metric, not the live training step.
    Returns the number of newly logged steps.
    """
    rows = _read_jsonl(out_jsonl)
    if not rows or wandb_run is None:
        return 0

    by_step: dict[int, list[dict]] = {}
    for r in rows:
        s = r.get("step")
        if s is None:
            continue
        by_step.setdefault(int(s), []).append(r)
    if not by_step:
        return 0

    if not getattr(wandb_run, "_valgen_rosetta_metric_defined", False):
        try:
            wandb_run.define_metric("val_gen_rosetta/scored_step")
            wandb_run.define_metric("val_gen_rosetta/*", step_metric="val_gen_rosetta/scored_step")
            wandb_run._valgen_rosetta_metric_defined = True
        except Exception:
            pass  # define_metric is best-effort; logging still works without it.

    max_step = max(by_step)
    n_logged = 0
    for s in sorted(by_step):
        if s in logged_steps or s >= max_step:
            continue
        dg = [
            float(r["binder_rosetta_dG_separated"])
            for r in by_step[s]
            if r.get("binder_rosetta_dG_separated") is not None
            and math.isfinite(float(r["binder_rosetta_dG_separated"]))
        ]
        if not dg:
            logged_steps.add(s)  # all-NaN step: mark done so we don't re-scan it forever.
            continue
        try:
            wandb_run.log(
                {
                    "val_gen_rosetta/dG_separated_mean": float(np.mean(dg)),
                    "val_gen_rosetta/dG_separated_median": float(np.median(dg)),
                    "val_gen_rosetta/dG_separated_min": float(np.min(dg)),
                    "val_gen_rosetta/n_scored": len(dg),
                    "val_gen_rosetta/scored_step": int(s),
                }
            )
            logged_steps.add(s)
            n_logged += 1
        except Exception as e:
            logger.warning(f"[val_gen_rosetta] wandb.log failed for step {s}: {e}")
            break  # try again next validation rather than skipping this step permanently.
    return n_logged


def maybe_submit_rosetta_sidecar(
    dump_dir: str,
    out_jsonl: str,
    *,
    repo_dir: str,
    submit_script: str = "scripts/submit_val_gen_rosetta.sh",
    job_name: str = "valgen_rosetta",
    max_per_step: int = 0,
) -> bool:
    """Submit the CPU dG sidecar, unless one is already queued/running (no pile-up).

    Called from inside the running training allocation, so ``SLURM_*`` is scrubbed from the child
    environment before sbatch -- otherwise sbatch sets SLURM_GET_USER_ENV=1 and the job is requeued
    and held with 'user env retrieval failed' (see the user's cluster notes). Returns True iff a new
    job was submitted.
    """
    user = os.environ.get("USER", "")
    try:
        q = subprocess.run(
            ["squeue", "-h", "-u", user, "--name", job_name, "-o", "%i"],
            capture_output=True, text=True, timeout=30,
        )
        if q.returncode == 0 and q.stdout.strip():
            return False  # one already in flight; it is resumable and will pick up new steps.
    except Exception:
        pass  # squeue unavailable -> fall through and attempt the submit anyway.

    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["DUMP_DIR"] = dump_dir
    env["OUT_JSONL"] = out_jsonl
    env["MAX_PER_STEP"] = str(max_per_step)
    env["JOB_NAME"] = job_name
    try:
        subprocess.run(
            ["bash", submit_script], env=env, cwd=repo_dir, check=True,
            timeout=180, capture_output=True, text=True,
        )
        logger.info(f"[val_gen_rosetta] submitted dG sidecar (dump={dump_dir}, out={out_jsonl}).")
        return True
    except Exception as e:
        out = getattr(e, "stderr", "") or ""
        logger.warning(f"[val_gen_rosetta] sidecar submit failed (non-fatal): {e} {out}")
        return False
