"""What to keep from a prediction, and what was only ever scaffolding.

The two-cache design in `evaluate.py` retains Boltz-2's whole working directory. That was
the right call at pilot scale -- it let the closure-constraint fix rescore ~926 rows
without re-predicting, saving ~3.6 GPU-hours -- and it is the wrong call at corpus scale,
because almost none of what it retains is a result.

Measured on 150 prediction directories from the PepBench valid run (mean 1.31 MB):

| component                    | MB/dir | share | what it actually is                     |
|------------------------------|--------|-------|-----------------------------------------|
| processed MSA npz            | 0.423  | 32.3% | an INPUT, per-receptor, regenerable     |
| pae/pde npz                  | 0.380  | 29.0% | matrices whose scalars are in the score |
| refined complex PDB          | 0.298  | 22.8% | 95.6% receptor (see below)              |
| raw complex PDB              | 0.135  | 10.3% | 95.6% receptor                          |
| template npz                 | 0.039  |  3.0% | an INPUT, per-receptor                  |
| record/score/small npz       | 0.017  |  1.3% | the result                              |

The structure files are the subtle one. A refined complex of a 16-mer on a 316-residue
receptor is 480 KB, of which the peptide -- the only thing that differs between the 32
candidates of a target -- is 19.2 KB, or 4.4 %. The receptor is the same protein every
time: 12.8 Å median pairwise backbone RMSD as written, but **0.90 Å after Kabsch
superposition**, because Boltz-2 emits each complex in its own arbitrary global frame.
So the bulk of every structure file is one receptor stored once per candidate in a
different pose each time.

The essential per-candidate payload -- peptide coordinates plus the score row -- measures
**5.0 KB against 1,310 KB retained**, a 263x overhead.

Tiers
-----
`full`         nothing is removed. The current behaviour and the default, so that
               enabling a tier is always a deliberate act.
`rescorable`   drop the inputs and the already-summarised confidence matrices, keep the
               raw complex gzipped (217 KB -> 47 KB). A new `SCORING_PROTOCOL_VERSION`
               can still re-relax and rescore WITHOUT re-predicting, which is the
               property the two-cache split exists to provide. ~50 KB/candidate, 26x
               smaller.
`deliverable`  additionally drop the raw complex. Keeps only the canonical-frame peptide
               coordinates and the score row. ~5 KB/candidate, 263x smaller. **A scoring
               protocol change then costs a re-prediction** (39.3 s GPU per candidate),
               so this tier trades a one-off compute risk for two orders of magnitude of
               storage. Choose it when the corpus matters more than the cache.

Every tier writes the design record, because that is the deliverable and it is the one
thing the retained tree never contained in a usable form: the peptides of a target all
expressed in ONE receptor frame, which is what makes them comparable to each other
without re-superimposing 32 complexes.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np

TIERS = ("full", "rescorable", "deliverable")

# Inputs and already-summarised matrices. Removing these loses nothing that is not either
# regenerable from the target context or present as a scalar on the score row.
_DROP_ALWAYS = (
    "pae_*.npz",       # scalars already extracted into the score row
    "pde_*.npz",       # ditto
    "plddt_*.npz",     # ditto
    "*_target.npz",    # target-only template: an INPUT, identical for every candidate
)
_DROP_DIRS_ALWAYS = ("processed", "lightning_logs")


def _iter_models(pred_dir: Path):
    """The raw/refined/isolated endpoint structures written for each sample."""
    base = pred_dir / "boltz"
    for pdb in sorted(base.rglob("*.pdb")):
        yield pdb


def _read_chain_atoms(pdb: Path, chains: set[str]) -> tuple[list[str], np.ndarray]:
    """ATOM lines for `chains`, as (full lines, Nx3 float32 coordinates)."""
    lines, xyz = [], []
    opener = gzip.open if pdb.suffix == ".gz" else open
    with opener(pdb, "rt") as fh:
        for line in fh:
            if line.startswith("ATOM") and line[21] in chains:
                lines.append(line.rstrip("\n"))
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return lines, np.asarray(xyz, dtype=np.float32).reshape(-1, 3)


def _atom_keys(lines: list[str]) -> list[tuple[str, str, str]]:
    return [(ln[21], ln[22:27].strip(), ln[12:16].strip()) for ln in lines]


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform taking `mobile` onto `target`, plus the fit RMSD.

    Returns (R, t, rmsd) with `mobile @ R.T + t` superimposed on `target`. Reflections are
    excluded via the determinant correction, so this cannot "fit" a mirror image.
    """
    mc, tc = mobile.mean(0), target.mean(0)
    P, Q = mobile - mc, target - tc
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    rot = P @ R.T
    rmsd = float(np.sqrt(((rot - Q) ** 2).sum(1).mean()))
    return R, tc - mc @ R.T, rmsd


def write_design_record(
    *,
    out_root: Path,
    parent_id: str,
    input_hash: str,
    sequence: str,
    original_sequence: str,
    binder_chain: str,
    receptor_chains: set[str],
    raw_pdb: Path,
    refined_pdb: Path | None,
) -> dict:
    """The deliverable: this candidate's peptide, in the target's canonical frame.

    The first candidate of a parent defines that frame by depositing its receptor as
    `designs/<parent>/receptor.pdb.gz`. Every later candidate is Kabsch-superimposed onto
    it on receptor atoms and stores only the transformed peptide, so all of a target's
    designs are directly overlayable without re-reading a single complex.

    `receptor_fit_rmsd_A` is recorded rather than assumed. The receptor is re-predicted
    for every candidate, so the fit is close but not exact (measured median 0.90 Å
    backbone, 1.33 Å all-atom over 45 pairs); a row whose fit is poor is a row whose
    peptide frame should not be trusted, and that is only detectable if the number is
    kept.
    """
    d = out_root / "designs" / parent_id
    d.mkdir(parents=True, exist_ok=True)
    canonical = d / "receptor.pdb.gz"

    rec_lines, rec_xyz = _read_chain_atoms(raw_pdb, receptor_chains)
    if not rec_lines:
        return {"design_record": None, "reason": "no receptor atoms in raw structure"}

    if not canonical.exists():
        with gzip.open(canonical, "wt", compresslevel=9) as fh:
            fh.write("\n".join(rec_lines) + "\n")
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        fit = 0.0
        n_fit = len(rec_xyz)
    else:
        can_lines, can_xyz = _read_chain_atoms(canonical, receptor_chains)
        # Superimpose on the atoms the two structures share, by (chain, resseq, name).
        can_idx = {k: i for i, k in enumerate(_atom_keys(can_lines))}
        pairs = [(i, can_idx[k]) for i, k in enumerate(_atom_keys(rec_lines))
                 if k in can_idx]
        if len(pairs) < 3:
            return {"design_record": None, "reason": "receptor atoms do not correspond"}
        mi = np.array([a for a, _ in pairs])
        ci = np.array([b for _, b in pairs])
        R, t, fit = kabsch(rec_xyz[mi], can_xyz[ci])
        n_fit = len(pairs)

    payload: dict = {
        "parent_id": parent_id,
        "input_hash": input_hash,
        "sequence": sequence,
        "original_sequence": original_sequence,
        "binder_chain": binder_chain,
        "receptor_fit_rmsd_A": round(float(fit), 4),
        "receptor_fit_atoms": int(n_fit),
    }
    arrays: dict[str, np.ndarray] = {}
    for tag, src in (("raw", raw_pdb), ("refined", refined_pdb)):
        if src is None or not Path(src).exists():
            continue
        lines, xyz = _read_chain_atoms(Path(src), {binder_chain})
        if not len(xyz):
            continue
        arrays[f"{tag}_xyz"] = (xyz @ R.T + t).astype(np.float32)
        payload[f"{tag}_atom_names"] = [ln[12:16].strip() for ln in lines]
        payload[f"{tag}_resseq"] = [ln[22:27].strip() for ln in lines]

    if not arrays:
        return {"design_record": None, "reason": "no peptide atoms found"}

    npz = d / f"{input_hash}.npz"
    np.savez_compressed(npz, meta=json.dumps(payload), **arrays)
    return {"design_record": str(npz), "receptor_fit_rmsd_A": payload["receptor_fit_rmsd_A"],
            "canonical_receptor": str(canonical)}


def compact_prediction_dir(pred_dir: Path, tier: str) -> dict:
    """Remove what the tier does not retain. Returns the bytes freed and what went.

    Idempotent: re-running on an already-compacted directory is a no-op, so a requeued
    task cannot double-count or fail on a missing file.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown retention tier {tier!r}; expected one of {TIERS}")
    pred_dir = Path(pred_dir)
    report = {"tier": tier, "freed_bytes": 0, "removed": [], "gzipped": []}
    if tier == "full" or not pred_dir.exists():
        return report

    def _rm(p: Path) -> None:
        try:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) \
                if p.is_dir() else p.stat().st_size
        except OSError:
            return
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        except OSError:
            return
        report["freed_bytes"] += size
        report["removed"].append(p.name)

    boltz = pred_dir / "boltz"
    if boltz.exists():
        for pattern in _DROP_ALWAYS:
            for f in boltz.rglob(pattern):
                _rm(f)
        for name in _DROP_DIRS_ALWAYS:
            for f in boltz.rglob(name):
                if f.is_dir():
                    _rm(f)

        # The refined and isolated structures are derived products: a rescore re-relaxes
        # from the raw complex and rewrites them, and the deliverable already holds the
        # refined peptide in canonical frame. The raw complex is the irreplaceable one,
        # because it is the predictor's output.
        for pdb in list(_iter_models(pred_dir)):
            if pdb.name.endswith(("_refined.pdb", "_isolated_relaxed.pdb")):
                _rm(pdb)

        for pdb in list(_iter_models(pred_dir)):
            if tier == "deliverable":
                _rm(pdb)
            elif tier == "rescorable":
                gz = pdb.with_suffix(pdb.suffix + ".gz")
                if gz.exists():
                    continue
                try:
                    raw = pdb.read_bytes()
                    gz.write_bytes(gzip.compress(raw, 9))
                except OSError:
                    continue
                report["freed_bytes"] += len(raw) - gz.stat().st_size
                report["gzipped"].append(pdb.name)
                try:
                    pdb.unlink()
                except OSError:
                    pass
    return report


def restore_raw_structures(pred_dir: Path) -> list[Path]:
    """Decompress any gzipped endpoint structures, for a rescore under a new protocol.

    This is what makes the `rescorable` tier honest: the scorer reads real PDB paths, so
    a protocol bump decompresses (milliseconds) instead of re-predicting (39.3 s GPU).
    """
    out = []
    for gz in sorted((Path(pred_dir) / "boltz").rglob("*.pdb.gz")):
        dest = gz.with_suffix("")
        if not dest.exists():
            dest.write_bytes(gzip.decompress(gz.read_bytes()))
        out.append(dest)
    return out
