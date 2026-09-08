"""Put an edited peptide back into its receptor, in the input PDB's own frame.

The CPSea loader hands the model a target-centred crop, so every coordinate the sampler
produces lives in a frame that exists only inside that process. Two loads of the SAME
example in two processes differ by a rigid transform of a couple of Angstroms (measured:
2.66 A on LNR_1jrr_A_P), because the receptor crop -- and therefore the centring -- is
drawn fresh each time. The consequence is sharp and easy to get wrong:

    an edited peptide written WITHOUT its receptor cannot be re-united with that receptor
    afterwards. The transform is not recoverable offline.

So the complex has to be written by the process that did the sampling, while the input
peptide is still in hand. That is what this module is for: `crystal_transform` recovers
the loader -> crystal transform EXACTLY by Kabsch-fitting the loader-frame INPUT peptide
onto the same peptide in the input PDB (the same structure on both sides, so the residual
is numerical noise and is asserted, not assumed), and `write_complex` applies it to the
edited coordinates and emits receptor + peptide as one PDB.

Shared by scripts/sdedit_cyclize.py (writes the complexes) and
scripts/score_sdedit_rosetta.py (scores them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_atom_lines(path) -> list[str]:
    return [l for l in Path(path).read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]


def chain_of(line: str) -> str:
    return line[21]


def xyz_of(line: str):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def set_xyz(line: str, xyz) -> str:
    return line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:]


def ca_trace(lines: list[str], chain: str) -> np.ndarray:
    return np.array([xyz_of(l) for l in lines
                     if chain_of(l) == chain and l[12:16].strip() == "CA"], dtype=np.float64)


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Rigid transform taking `mobile` onto `target`. Returns (R, t, rmsd_after)."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    fitted = (mobile - mc) @ r.T + tc
    return r, tc - mc @ r.T, float(np.sqrt(((fitted - target) ** 2).sum(-1).mean()))


def crystal_transform(loader_ca_A: np.ndarray, input_pdb, binder_chain: str):
    """Loader frame -> input-PDB frame, from the INPUT peptide's own CA trace.

    Returns (R, t, offset, residual_A, receptor_lines, target_chains).

    The loader can return fewer residues than the chain holds (its crop), so every
    contiguous offset is tried and the best residual wins; assuming offset 0 would fit two
    different subsegments onto each other and place the peptide plausibly but wrongly.
    """
    lines = read_atom_lines(input_pdb)
    receptor_lines = [l for l in lines if chain_of(l) != binder_chain]
    target_chains = sorted({chain_of(l) for l in receptor_lines})
    pdb_ca = ca_trace(lines, binder_chain)
    n, m = len(loader_ca_A), len(pdb_ca)
    if n == 0 or m == 0:
        raise RuntimeError(f"empty CA trace: loader {n} res, {input_pdb} chain {binder_chain} {m} res")
    if n > m:
        raise RuntimeError(f"loader peptide ({n} res) longer than {input_pdb} chain "
                           f"{binder_chain} ({m} res)")
    best = None
    for off in range(m - n + 1):
        r, t, rmsd = kabsch(loader_ca_A, pdb_ca[off:off + n])
        if best is None or rmsd < best[3]:
            best = (r, t, off, rmsd)
    return (*best, receptor_lines, target_chains)


def write_complex(out_path, receptor_lines: list[str], peptide_atom_lines: list[str],
                  binder_chain: str, r: np.ndarray, t: np.ndarray) -> None:
    """Receptor verbatim from the input PDB + the peptide rotated into its frame."""
    out, serial = [], 0
    for line in receptor_lines:
        serial += 1
        out.append(line[:6] + f"{serial:5d}" + line[11:])
    out.append("TER")
    for line in peptide_atom_lines:
        serial += 1
        xyz = np.asarray(xyz_of(line), dtype=np.float64) @ r.T + t
        new = set_xyz(line, xyz)
        out.append(new[:6] + f"{serial:5d}" + new[11:21] + binder_chain + new[22:])
    out.append("TER")
    out.append("END")
    Path(out_path).write_text("\n".join(out) + "\n")
