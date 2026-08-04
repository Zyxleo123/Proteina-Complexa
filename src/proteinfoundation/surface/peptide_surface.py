"""Peptide-interface molecular-surface extraction for CPSea complexes.

Ported from PepBridge's ``data/utils_pymol.py``:

    https://github.com/guanlueli/Pepbridge/blob/main/data/utils_pymol.py

    MIT License

    Copyright (c) 2026 Guanlue Li

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

What was ported
---------------
Only the surface-generation / mesh-loading / interface-filtering / sampling path:

* ``preprocess_surface_single`` -> :func:`generate_surface_obj`. Same PyMOL recipe
  (``show_as('surface')`` + an identity ``set_view``), same reason: the identity view
  matrix with a zero camera translation makes PyMOL's OBJ exporter write vertices in the
  model's own coordinate frame instead of camera space. Verified empirically on CPSea:
  the mesh bounding box is the atom bounding box grown by the probe radius, and the
  centroid offset is < 0.31 A.
* ``load_point_cloud_by_file_extension`` -> :func:`load_mesh`. Same trimesh
  ``force='mesh'`` load, which also de-duplicates the ~6x-redundant per-triangle
  vertices PyMOL emits (8292 -> 1384 on a 16-mer peptide).
* ``get_symmetric_interface_masks`` -> :func:`nearest_receptor_distance` +
  :func:`interface_mask`. See "Behavioural changes" below.
* ``subsample_points(method='fps')`` -> :func:`farthest_point_sample`. See below.

Deliberately NOT ported: the chemistry featurisers (``computer_property``,
hydrophobicity/charge, which pull in PepBridge's ``data.constants`` /
``data.computeCharges`` and the MaSIF-derived tables), Delaunay re-meshing
(``plot_surface``), DBSCAN prominent-point detection, Hungarian point matching, and
Gaussian smoothing. None of them are needed for an interface point cloud, and each
would drag in a dependency (pyvista, sklearn.cluster) for no gain here.

Behavioural changes vs. PepBridge
---------------------------------
1. **Asymmetric interface, not symmetric.** PepBridge's ``get_symmetric_interface_masks``
   iteratively trims the receptor mask until it has exactly as many points as the peptide
   mask, because its model consumes matched peptide/receptor point pairs. We only need the
   peptide-side patch, so we keep the plain criterion (peptide vertex within ``cutoff`` of
   any receptor vertex) and drop the O(iterations) rebalancing loop entirely. This makes
   the retained set a pure function of the geometry and the cutoff -- no tie-breaking on
   contact counts, hence reproducible.
2. **KD-tree instead of a dense ``torch.cdist``.** PepBridge materialises the full
   (N_pep x N_rec) distance matrix. A CPSea receptor surface is 20k-80k vertices, so that
   is a multi-GB allocation for a quantity we only need the row-minimum of. We use
   ``scipy.spatial.cKDTree`` and fall back to a chunked brute-force scan
   (:func:`_chunked_nearest_distance`) when SciPy is unavailable.
3. **Deterministic FPS.** PepBridge seeds farthest-point sampling with
   ``torch.randint`` off the global RNG, so two runs of the same complex return different
   point sets. We seed a local ``numpy.random.Generator`` from the caller's ``seed``, so a
   given (complex, cutoff, seed) always yields the same points. The greedy loop itself is
   unchanged. We also fix a latent bug in the original: it updates the running min-distance
   from ``points[fps_idx[i-1]]`` *before* using it but never seeds ``distances`` from the
   first point's own row in a way that survives ``target_size == 1``, and it recomputes
   from the previous index rather than the newly-added one -- equivalent here, but we write
   the standard formulation to keep the invariant obvious.
4. **Padding + validity mask.** PepBridge assumes enough points exist. Small or
   glancing CPSea interfaces routinely yield fewer than 96 vertices, so we pad to a fixed
   ``num_points`` and return ``sampled_valid_mask`` rather than returning a ragged array.
5. **Normals are explicitly renormalised.** PepBridge uses ``mesh.vertex_normals``
   as-is. trimesh's area-weighted averaging over merged vertices can return a slightly
   short vector (and a zero vector on degenerate faces); we normalise and replace
   degenerate normals with zeros, which the finite-ness checks then still pass.
6. **No global PyMOL state leak.** PepBridge calls ``cmd.reinitialize()`` on the module
   -level singleton and writes the ``.obj`` next to the input PDB. We keep every
   intermediate in a caller-scoped temporary directory and never touch the source PDB.

Units: Angstrom throughout, in the source PDB's own coordinate frame (no centering, no
rotation). Downstream transforms (``CoordsToNanometers``, ``GlobalRotationTransform``)
must be applied to the surface with :func:`transform_surface` if they are applied to the
structure -- see the rigid-transform test in ``script_utils/test_peptide_surface.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

# Must precede numpy/scipy import so ProcessPool children do not each spawn a full
# OpenMP team (that was collapsing SAS from ~50ms to multi-seconds at --workers 32).
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np

# Bump on any change that alters the numbers in a cache file. `extract_peptide_surface`
# stamps it into the metadata and `is_cache_valid` refuses caches from another version,
# so a stale cache is skipped-and-rebuilt rather than silently mixed into a dataset.
#
# 1.0.0 — PepBridge-style dual PyMOL SES (peptide + receptor meshes).
# 2.0.0 — default ``backend="sas"``: Shrake–Rupley peptide SAS points + receptor heavy
#         atoms (no PyMOL). ~50× faster; interface set is similar but not bit-identical.
EXTRACTOR_VERSION = "2.0.0"

DEFAULT_CUTOFF = 4.0
DEFAULT_NUM_POINTS = 96
DEFAULT_SEED = 0
DEFAULT_BACKEND = "sas"
DEFAULT_SAS_POINTS_PER_ATOM = 48

# ProtOr-style heavy-atom VdW radii (Å). Used by the SAS backend.
_VDW_RADIUS = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
}

# PyMOL view matrix that makes the OBJ exporter emit model-frame coordinates:
# identity rotation (0..8), zero camera translation (9..11), zero rotation origin
# (12..14), then near / far / orthoscopic. Taken verbatim from PepBridge.
_IDENTITY_VIEW = (
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 300.0, 1.0,
)

_ARRAY_KEYS = (
    "full_peptide_xyz",
    "full_peptide_normals",
    "interface_xyz",
    "interface_normals",
    "interface_receptor_distance",
    "sampled_xyz",
    "sampled_normals",
    "sampled_receptor_distance",
    "sampled_valid_mask",
)


class SurfaceExtractionError(RuntimeError):
    """Raised when a complex cannot be turned into a surface cache.

    Carries a short machine-readable ``reason`` so the batch driver can bucket failures
    (missing chain, empty mesh, PyMOL crash, ...) instead of grepping messages.
    """

    def __init__(self, message: str, reason: str = "error"):
        super().__init__(message)
        self.reason = reason


@dataclasses.dataclass
class PeptideSurface:
    """One complex's peptide-surface point cloud, in the source PDB's frame.

    ``full_*`` is the whole peptide molecular surface; ``interface_*`` is the subset
    within ``cutoff`` of the receptor surface; ``sampled_*`` is the fixed-size
    farthest-point subsample of the interface, padded with zeros where
    ``sampled_valid_mask`` is False.
    """

    full_peptide_xyz: np.ndarray  # (V, 3) float32
    full_peptide_normals: np.ndarray  # (V, 3) float32, unit norm
    interface_xyz: np.ndarray  # (I, 3) float32
    interface_normals: np.ndarray  # (I, 3) float32, unit norm
    interface_receptor_distance: np.ndarray  # (I,) float32, all <= cutoff
    sampled_xyz: np.ndarray  # (K, 3) float32
    sampled_normals: np.ndarray  # (K, 3) float32
    sampled_receptor_distance: np.ndarray  # (K,) float32
    sampled_valid_mask: np.ndarray  # (K,) bool
    metadata: dict

    @property
    def num_full(self) -> int:
        return int(self.full_peptide_xyz.shape[0])

    @property
    def num_interface(self) -> int:
        return int(self.interface_xyz.shape[0])

    @property
    def num_sampled(self) -> int:
        return int(self.sampled_valid_mask.sum())


# ---------------------------------------------------------------------------
# Chain splitting
# ---------------------------------------------------------------------------


def _open_maybe_gzip(path: str | os.PathLike):
    """Open a PDB, gzipped or not, translating a missing file into a typed failure.

    A dataset row pointing at a path that no longer exists is a routine data problem, not
    a bug: it must land in the batch driver's `missing_pdb` bucket rather than its
    `unexpected` bucket with a traceback.
    """
    path = str(path)
    if not os.path.exists(path):
        raise SurfaceExtractionError(f"missing PDB {path}", reason="missing_pdb")
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_chain_ids(pdb_path: str | os.PathLike) -> list[str]:
    """Return the chain IDs present in ATOM/HETATM records, in file order."""
    seen: list[str] = []
    with _open_maybe_gzip(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                chain = line[21]
                if chain not in seen:
                    seen.append(chain)
    return seen


def split_chains(
    pdb_path: str | os.PathLike,
    chains: Sequence[str],
    out_path: str | os.PathLike,
) -> int:
    """Write the ATOM/HETATM records of ``chains`` to ``out_path``; return atom count.

    Implemented as a record-level text filter rather than a PyMOL ``create``/``save``
    round-trip: PyMOL renumbers atoms and rewrites occupancies/element fields, which makes
    "did the split preserve the right atoms?" untestable against the source. The source
    file is opened read-only and never modified.

    TER records are re-emitted per kept chain so downstream parsers still see chain
    boundaries. CONECT records are dropped -- they reference the original atom serials,
    and the surface generator does not use connectivity.
    """
    wanted = set(chains)
    if not wanted:
        raise SurfaceExtractionError("no chains requested for split", reason="empty_chain_spec")

    kept = 0
    emitted_chains: set[str] = set()
    prev_chain: str | None = None
    with _open_maybe_gzip(pdb_path) as fh, open(out_path, "w") as out:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21]
            if chain not in wanted:
                continue
            if prev_chain is not None and chain != prev_chain:
                out.write("TER\n")
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            emitted_chains.add(chain)
            prev_chain = chain
        if kept:
            out.write("TER\n")
        out.write("END\n")

    missing = wanted - emitted_chains
    if missing:
        raise SurfaceExtractionError(
            f"chains {sorted(missing)} absent from {pdb_path}", reason="missing_chain"
        )
    if kept == 0:
        raise SurfaceExtractionError(f"no atoms for chains {sorted(wanted)}", reason="empty_chain")
    return kept


# ---------------------------------------------------------------------------
# Fast SAS backend (no PyMOL)
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Even unit-sphere directions (golden-section spiral), shape ``(n, 3)``."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


def _vdw_radii(elements: Sequence[str], probe: float) -> np.ndarray:
    return np.asarray(
        [_VDW_RADIUS.get(str(e).upper(), 1.70) + float(probe) for e in elements],
        dtype=np.float64,
    )


def _element_from_pdb_line(line: str) -> str:
    """PDB element (cols 77-78) or a best-effort guess from the atom name."""
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if not elem:
        name = line[12:16].strip()
        elem = "".join(c for c in name if c.isalpha())[:2]
        if elem and elem[0].isdigit():
            elem = elem[1:]
        elem = (elem[:1] + elem[1:].lower()) if elem else "C"
    return elem


def load_heavy_atoms(
    pdb_path: str | os.PathLike, chains: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Return ``(coords [N,3] float64, elements)`` for heavy atoms in ``chains``."""
    pep_coords, pep_elems, _, _, _ = load_heavy_atoms_split(pdb_path, chains, ())
    if pep_coords.shape[0] == 0:
        raise SurfaceExtractionError(
            f"no heavy atoms for chains {sorted(set(chains))} in {pdb_path}",
            reason="empty_chain",
        )
    return pep_coords, pep_elems


def load_heavy_atoms_split(
    pdb_path: str | os.PathLike,
    peptide_chains: Sequence[str],
    receptor_chains: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[str]]:
    """Single-pass heavy-atom load for peptide + receptor.

    If ``receptor_chains`` is ``None``, every non-peptide chain in the file becomes the
    receptor (same rule as :func:`resolve_chain_assignment`) — one PDB scan instead of
    chain-id discovery + two atom loads.
    """
    pep_want = set(peptide_chains)
    rec_want = None if receptor_chains is None else set(receptor_chains)
    pep_coords: list[list[float]] = []
    pep_elems: list[str] = []
    rec_coords: list[list[float]] = []
    rec_elems: list[str] = []
    seen_rec: set[str] = set()
    with _open_maybe_gzip(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            ch = line[21]
            if ch in pep_want:
                bucket_coords, bucket_elems = pep_coords, pep_elems
            elif rec_want is None:
                bucket_coords, bucket_elems = rec_coords, rec_elems
                seen_rec.add(ch)
            elif ch in rec_want:
                bucket_coords, bucket_elems = rec_coords, rec_elems
            else:
                continue
            elem = _element_from_pdb_line(line)
            if elem.upper() == "H":
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError as exc:
                raise SurfaceExtractionError(
                    f"bad coordinates in {pdb_path}: {line.strip()}", reason="bad_pdb"
                ) from exc
            bucket_coords.append(xyz)
            bucket_elems.append(elem)
    resolved_rec = sorted(seen_rec) if receptor_chains is None else list(receptor_chains)
    return (
        np.asarray(pep_coords, dtype=np.float64).reshape(-1, 3),
        pep_elems,
        np.asarray(rec_coords, dtype=np.float64).reshape(-1, 3),
        rec_elems,
        resolved_rec,
    )


def generate_sas_points(
    coords: np.ndarray,
    elements: Sequence[str],
    points_per_atom: int = DEFAULT_SAS_POINTS_PER_ATOM,
    probe: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrake–Rupley solvent-accessible points + outward normals.

    Returns float32 ``(P, 3)`` xyz and unit normals. Much faster than a PyMOL SES mesh
    for short peptides (tens of ms vs seconds).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise SurfaceExtractionError("coords must be (N, 3)", reason="bad_coords")
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        raise SurfaceExtractionError("no atoms for SAS", reason="empty_chain")

    radii = _vdw_radii(elements, probe)
    unit = _fibonacci_sphere(int(points_per_atom))
    pts = (coords[:, None, :] + radii[:, None, None] * unit[None, :, :]).reshape(-1, 3)
    parent = np.repeat(np.arange(n_atoms), int(points_per_atom))

    # Buried if inside any other atom's (vdw+probe) sphere. Peptides are tiny — dense check.
    dist = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=-1)  # (P, A)
    dist[np.arange(pts.shape[0]), parent] = np.inf
    accessible = (dist >= (radii[None, :] - 1e-3)).all(axis=1)
    pts = pts[accessible]
    parent = parent[accessible]
    if pts.shape[0] == 0:
        raise SurfaceExtractionError("SAS produced no accessible points", reason="empty_surface")

    normals = normalize_normals(pts - coords[parent])
    return pts.astype(np.float32), normals.astype(np.float32)


# ---------------------------------------------------------------------------
# PyMOL surface generation
# ---------------------------------------------------------------------------


_PYMOL_STARTED = False


def _ensure_pymol():
    """Start the headless PyMOL singleton once per process and return ``cmd``.

    ``-qck`` is quiet / no-GUI / no-pymolrc: the mesh is computed on the CPU, so no GL
    context and no display are needed, which is what lets this run on a compute node.
    """
    global _PYMOL_STARTED
    try:
        import pymol
        from pymol import cmd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SurfaceExtractionError(
            "pymol-open-source is required for surface extraction "
            "(`uv pip install pymol-open-source`)",
            reason="missing_pymol",
        ) from exc

    if not _PYMOL_STARTED:
        pymol.finish_launching(["pymol", "-qck"])
        _PYMOL_STARTED = True
    return cmd


def generate_surface_obj(
    pdb_path: str | os.PathLike,
    obj_path: str | os.PathLike,
    solvent_radius: float | None = None,
    surface_quality: int | None = None,
) -> Path:
    """Render ``pdb_path``'s molecular surface to a Wavefront OBJ at ``obj_path``.

    ``solvent_radius`` / ``surface_quality`` are exposed but default to PyMOL's own
    defaults (1.4 A probe, quality 0), which is what PepBridge used.
    """
    cmd = _ensure_pymol()
    obj_path = Path(obj_path)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "obj_src")
    if solvent_radius is not None:
        cmd.set("solvent_radius", solvent_radius)
    if surface_quality is not None:
        cmd.set("surface_quality", surface_quality)
    cmd.hide("everything")
    cmd.show_as("surface", "obj_src")
    # Identity view => OBJ vertices land in the model frame, not camera space.
    cmd.set_view(_IDENTITY_VIEW)
    cmd.save(str(obj_path), "obj_src")
    cmd.reinitialize()

    if not obj_path.exists() or obj_path.stat().st_size == 0:
        raise SurfaceExtractionError(
            f"PyMOL produced no surface for {pdb_path}", reason="empty_surface"
        )
    return obj_path


def normalize_normals(normals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each row to unit length; degenerate rows become exactly zero.

    A zero row is preferable to a NaN: it is finite, it fails a `norm ~= 1` check loudly
    instead of poisoning every downstream reduction, and it is what the validity checks
    in the test suite look for.
    """
    normals = np.asarray(normals, dtype=np.float64)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    out = np.zeros_like(normals)
    good = (norm[..., 0] > eps) & np.isfinite(norm[..., 0])
    out[good] = normals[good] / norm[good]
    return out.astype(np.float32)


def load_mesh(obj_path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices and unit vertex normals as float32 ``(V, 3)`` arrays."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SurfaceExtractionError(
            "trimesh is required for surface extraction (`uv pip install trimesh`)",
            reason="missing_trimesh",
        ) from exc

    mesh = trimesh.load(str(obj_path), force="mesh")
    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    if xyz.size == 0:
        raise SurfaceExtractionError(f"mesh {obj_path} has no vertices", reason="empty_surface")
    normals = normalize_normals(np.asarray(mesh.vertex_normals))
    return xyz, normals


# ---------------------------------------------------------------------------
# Interface filtering
# ---------------------------------------------------------------------------


def _chunked_nearest_distance(
    query: np.ndarray, reference: np.ndarray, chunk_size: int = 4096
) -> np.ndarray:
    """Brute-force nearest-neighbour distance in bounded memory (KD-tree fallback)."""
    out = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        block = query[start : start + chunk_size]
        d2 = ((block[:, None, :] - reference[None, :, :]) ** 2).sum(-1)
        out[start : start + block.shape[0]] = np.sqrt(d2.min(axis=1))
    return out


def nearest_receptor_distance(
    peptide_xyz: np.ndarray,
    receptor_xyz: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Distance from each peptide vertex to the nearest receptor-surface vertex.

    Uses ``scipy.spatial.cKDTree`` when available (O(N log M) and no large allocation)
    and otherwise falls back to a chunked brute-force scan. Both paths return the same
    numbers; the fallback exists so the module stays importable in a SciPy-free env.
    """
    peptide_xyz = np.asarray(peptide_xyz, dtype=np.float64)
    receptor_xyz = np.asarray(receptor_xyz, dtype=np.float64)
    if receptor_xyz.shape[0] == 0:
        raise SurfaceExtractionError("receptor surface is empty", reason="empty_surface")
    if peptide_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        # workers=1: under ProcessPool, workers=-1 oversubscribes OpenMP across
        # every child and turns a ~1ms query into multi-second thrashing.
        dist, _ = cKDTree(receptor_xyz).query(peptide_xyz, k=1, workers=1)
        dist = np.asarray(dist, dtype=np.float64)
    except ImportError:  # pragma: no cover - environment-dependent
        dist = _chunked_nearest_distance(peptide_xyz, receptor_xyz, chunk_size=chunk_size)
    return dist.astype(np.float32)


def interface_mask(distance: np.ndarray, cutoff: float) -> np.ndarray:
    """Boolean mask of vertices whose nearest-receptor distance is within ``cutoff``.

    Inclusive (``<=``) so that a vertex sitting exactly at the cutoff is retained, which
    is what makes the "every retained point is <= cutoff" invariant checkable without a
    floating-point slack term.
    """
    return np.asarray(distance) <= float(cutoff)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def farthest_point_sample(xyz: np.ndarray, num_points: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Greedy farthest-point sampling; returns indices into ``xyz``.

    Deterministic for a fixed ``seed``: the seed picks only the first index (via a local
    ``np.random.Generator``), and every subsequent pick is the argmax of the running
    min-distance, with ties broken by ``argmax``'s first-occurrence rule.

    Returns at most ``min(num_points, len(xyz))`` indices -- padding is the caller's job.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    k = int(min(num_points, n))
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = int(rng.integers(n))
    # Running squared distance from every point to the nearest already-selected point.
    min_d2 = ((xyz - xyz[idx[0]]) ** 2).sum(-1)
    for i in range(1, k):
        idx[i] = int(np.argmax(min_d2))
        d2 = ((xyz - xyz[idx[i]]) ** 2).sum(-1)
        min_d2 = np.minimum(min_d2, d2)
    return idx


def pad_to(array: np.ndarray, num_points: int) -> np.ndarray:
    """Zero-pad ``array`` along axis 0 up to ``num_points`` rows."""
    array = np.asarray(array)
    pad = num_points - array.shape[0]
    if pad <= 0:
        return array[:num_points]
    pad_shape = (pad,) + array.shape[1:]
    return np.concatenate([array, np.zeros(pad_shape, dtype=array.dtype)], axis=0)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def extract_peptide_surface(
    pdb_path: str | os.PathLike,
    receptor_chains: Sequence[str] | None,
    peptide_chains: Sequence[str],
    cutoff: float = DEFAULT_CUTOFF,
    num_points: int = DEFAULT_NUM_POINTS,
    seed: int = DEFAULT_SEED,
    work_dir: str | os.PathLike | None = None,
    keep_intermediates: bool = False,
    solvent_radius: float | None = None,
    surface_quality: int | None = None,
    backend: str = DEFAULT_BACKEND,
    sas_points_per_atom: int = DEFAULT_SAS_POINTS_PER_ATOM,
) -> PeptideSurface:
    """Extract the peptide interface surface of one complex.

    Backends
    --------
    ``sas`` (default, v2)
        Shrake–Rupley SAS points on peptide heavy atoms; interface = points within
        ``cutoff`` Å of a receptor heavy atom. No PyMOL, typically tens of ms.
        ``receptor_chains=None`` means "every non-peptide chain" (single PDB scan).
    ``pymol``
        PepBridge-style dual SES meshes (peptide + receptor) via PyMOL + trimesh.
        Slow (~seconds) but matches the original recipe.

    Raises :class:`SurfaceExtractionError` with a ``reason`` on any failure, including an
    empty interface (``reason='empty_interface'``) -- an empty patch is a data problem the
    batch driver should count, not a silently-cached array of zeros.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise SurfaceExtractionError(f"missing PDB {pdb_path}", reason="missing_pdb")

    peptide_chains = list(peptide_chains)
    if receptor_chains is not None:
        receptor_chains = list(receptor_chains)
        overlap = set(receptor_chains) & set(peptide_chains)
        if overlap:
            raise SurfaceExtractionError(
                f"chains {sorted(overlap)} assigned to both receptor and peptide",
                reason="chain_overlap",
            )

    backend = str(backend).lower().strip()
    if backend == "sas":
        pep_coords, pep_elems, rec_coords, _, receptor_chains = load_heavy_atoms_split(
            pdb_path, peptide_chains, receptor_chains
        )
        if pep_coords.shape[0] == 0:
            raise SurfaceExtractionError(
                f"no heavy atoms for peptide chains {peptide_chains} in {pdb_path}",
                reason="empty_chain",
            )
        if rec_coords.shape[0] == 0:
            raise SurfaceExtractionError(
                f"no heavy atoms for receptor chains {receptor_chains} in {pdb_path}",
                reason="empty_chain",
            )
        probe = 1.4 if solvent_radius is None else float(solvent_radius)
        pep_xyz, pep_normals = generate_sas_points(
            pep_coords,
            pep_elems,
            points_per_atom=int(sas_points_per_atom),
            probe=probe,
        )
        n_rec_atoms = int(rec_coords.shape[0])
        n_pep_atoms = int(pep_coords.shape[0])
        rec_xyz = rec_coords.astype(np.float32)
    elif backend == "pymol":
        if receptor_chains is None:
            receptor_chains, peptide_chains = resolve_chain_assignment(
                pdb_path, peptide_chains, None
            )
        with contextlib.ExitStack() as stack:
            if work_dir is None:
                work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pepsurf_")))
            else:
                work = Path(work_dir)
                work.mkdir(parents=True, exist_ok=True)
                if not keep_intermediates:
                    stack.callback(_maybe_cleanup, work)

            rec_pdb = work / "receptor.pdb"
            pep_pdb = work / "peptide.pdb"
            n_rec_atoms = split_chains(pdb_path, receptor_chains, rec_pdb)
            n_pep_atoms = split_chains(pdb_path, peptide_chains, pep_pdb)

            rec_obj = generate_surface_obj(
                rec_pdb, work / "receptor.obj", solvent_radius, surface_quality
            )
            pep_obj = generate_surface_obj(
                pep_pdb, work / "peptide.obj", solvent_radius, surface_quality
            )
            rec_xyz, _ = load_mesh(rec_obj)
            pep_xyz, pep_normals = load_mesh(pep_obj)
    else:
        raise SurfaceExtractionError(
            f"unknown surface backend {backend!r} (use 'sas' or 'pymol')",
            reason="bad_backend",
        )

    distance = nearest_receptor_distance(pep_xyz, rec_xyz)
    mask = interface_mask(distance, cutoff)

    interface_xyz = pep_xyz[mask]
    interface_normals = pep_normals[mask]
    interface_distance = distance[mask]

    if interface_xyz.shape[0] == 0:
        raise SurfaceExtractionError(
            f"no peptide surface vertex within {cutoff} A of the receptor",
            reason="empty_interface",
        )

    idx = farthest_point_sample(interface_xyz, num_points, seed=seed)
    n_valid = int(idx.shape[0])
    sampled_xyz = pad_to(interface_xyz[idx], num_points)
    sampled_normals = pad_to(interface_normals[idx], num_points)
    sampled_distance = pad_to(interface_distance[idx], num_points)
    sampled_valid_mask = np.zeros(num_points, dtype=bool)
    sampled_valid_mask[:n_valid] = True

    metadata = {
        "source_pdb": str(pdb_path.resolve()),
        "receptor_chains": receptor_chains,
        "peptide_chains": peptide_chains,
        "cutoff": float(cutoff),
        "sample_count": int(num_points),
        "seed": int(seed),
        "extractor_version": EXTRACTOR_VERSION,
        "backend": backend,
        # Provenance beyond the required keys -- cheap to store, and the only way to tell
        # a legitimately tiny interface from a botched chain assignment after the fact.
        "num_receptor_atoms": int(n_rec_atoms),
        "num_peptide_atoms": int(n_pep_atoms),
        "num_receptor_surface_vertices": int(rec_xyz.shape[0]),
        "num_full_peptide_vertices": int(pep_xyz.shape[0]),
        "num_interface_vertices": int(interface_xyz.shape[0]),
        "num_valid_sampled": n_valid,
    }
    if backend == "sas":
        metadata["sas_points_per_atom"] = int(sas_points_per_atom)

    return PeptideSurface(
        full_peptide_xyz=pep_xyz.astype(np.float32),
        full_peptide_normals=pep_normals.astype(np.float32),
        interface_xyz=interface_xyz.astype(np.float32),
        interface_normals=interface_normals.astype(np.float32),
        interface_receptor_distance=interface_distance.astype(np.float32),
        sampled_xyz=sampled_xyz.astype(np.float32),
        sampled_normals=sampled_normals.astype(np.float32),
        sampled_receptor_distance=sampled_distance.astype(np.float32),
        sampled_valid_mask=sampled_valid_mask,
        metadata=metadata,
    )


def _maybe_cleanup(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def save_surface_cache(path: str | os.PathLike, surface: PeptideSurface) -> Path:
    """Write one ``.npz`` per complex; metadata rides along as a JSON string.

    Written to a sibling ``.tmp`` and renamed, so an interrupted run leaves either the
    previous cache or nothing -- never a half-written file that ``--skip-existing`` would
    then accept.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    arrays = {key: getattr(surface, key) for key in _ARRAY_KEYS}
    # Write through an open handle: np.savez_compressed appends ".npz" to a *path* whose
    # name does not already end in it, which would leave the ".tmp" file unrenameable.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, metadata_json=np.array(json.dumps(surface.metadata)), **arrays)
    tmp.replace(path)
    return path


def load_surface_cache(path: str | os.PathLike) -> PeptideSurface:
    """Read a cache written by :func:`save_surface_cache`."""
    with np.load(str(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in _ARRAY_KEYS}
        metadata = json.loads(str(data["metadata_json"]))
    arrays["sampled_valid_mask"] = arrays["sampled_valid_mask"].astype(bool)
    return PeptideSurface(metadata=metadata, **arrays)


def _read_cache_metadata(path: str | os.PathLike) -> dict | None:
    """Read only ``metadata_json`` from an npz (no array decompress) — cheap prefilter."""
    import zipfile

    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("metadata_json.npy") as fh:
                meta_arr = np.load(fh, allow_pickle=False)
        return json.loads(str(meta_arr))
    except Exception:
        return None


def is_cache_valid(
    path: str | os.PathLike,
    cutoff: float | None = None,
    num_points: int | None = None,
    seed: int | None = None,
    version: str | None = EXTRACTOR_VERSION,
    backend: str | None = None,
    sas_points_per_atom: int | None = None,
) -> bool:
    """True if ``path`` is a readable cache matching the requested settings.

    A cache built with a different cutoff/sample count/seed/extractor version/
    extraction backend/backend resolution is *not* valid for the current request:
    reusing it would silently mix incompatible point sets into one dataset (e.g. a
    PyMOL-extracted surface accepted in place of an SAS-extracted one just because
    cutoff/num_points/seed/version happened to match -- `backend` is stored in
    every cache's metadata, see `extract_peptide_surface`, but was never checked
    here). Returning False here makes ``--skip-existing`` rebuild it.

    Only the embedded metadata blob is opened (zip entry), so validating millions of
    existing caches at job start stays I/O-bound on stats, not full decompress.

    Args:
        path: Cache file to check.
        cutoff, num_points, seed, version: As before.
        backend: Expected extraction backend (e.g. ``"pymol"`` or ``"sas"``).
            `None` skips this check (matches any backend).
        sas_points_per_atom: Expected Fibonacci-sphere density for the ``"sas"``
            backend. Only checked when the cache's own metadata records a
            `backend` of `"sas"` (a cache from a different backend has no such
            key and is already rejected by the `backend` check above whenever
            `backend` is also given; if `backend` is left `None`, a non-`"sas"`
            cache without this key is treated as not applicable, not a mismatch).
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    meta = _read_cache_metadata(path)
    if meta is None:
        return False
    checks = (
        (cutoff, "cutoff", float),
        (num_points, "sample_count", int),
        (seed, "seed", int),
        (version, "extractor_version", str),
        (backend, "backend", str),
    )
    for requested, key, caster in checks:
        if requested is None:
            continue
        if key not in meta or caster(meta[key]) != caster(requested):
            return False
    if sas_points_per_atom is not None and meta.get("backend") == "sas":
        if int(meta.get("sas_points_per_atom", -1)) != int(sas_points_per_atom):
            return False
    return True


# ---------------------------------------------------------------------------
# Geometry helpers used by consumers / tests
# ---------------------------------------------------------------------------


def transform_surface(
    surface: PeptideSurface, rotation: np.ndarray, translation: np.ndarray | None = None
) -> PeptideSurface:
    """Apply a rigid transform: points rotate *and* translate, normals only rotate.

    Normals are directions, not positions -- translating them would tilt every vector
    toward the translation and destroy the outward-pointing property the whole patch is
    for. Padded rows stay exactly zero (they are not real points), which keeps
    ``sampled_valid_mask`` the single source of truth about which rows mean anything.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {rotation.shape}")
    translation = (
        np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, float)
    )

    def rot_points(x):
        return (np.asarray(x, dtype=np.float64) @ rotation.T + translation).astype(np.float32)

    def rot_dirs(x):
        return (np.asarray(x, dtype=np.float64) @ rotation.T).astype(np.float32)

    valid = surface.sampled_valid_mask
    sampled_xyz = np.zeros_like(surface.sampled_xyz)
    sampled_normals = np.zeros_like(surface.sampled_normals)
    sampled_xyz[valid] = rot_points(surface.sampled_xyz[valid])
    sampled_normals[valid] = rot_dirs(surface.sampled_normals[valid])

    return dataclasses.replace(
        surface,
        full_peptide_xyz=rot_points(surface.full_peptide_xyz),
        full_peptide_normals=rot_dirs(surface.full_peptide_normals),
        interface_xyz=rot_points(surface.interface_xyz),
        interface_normals=rot_dirs(surface.interface_normals),
        sampled_xyz=sampled_xyz,
        sampled_normals=sampled_normals,
    )


def sampling_coverage(surface: PeptideSurface) -> tuple[float, float]:
    """(mean, max) distance from each retained interface vertex to its nearest sample.

    The diagnostic that catches the failure mode FPS is supposed to prevent: a sample set
    collapsed into one lobe leaves far-away retained vertices, so ``max`` blows up while
    the point count still looks fine.
    """
    valid = surface.sampled_valid_mask
    if surface.interface_xyz.shape[0] == 0 or not valid.any():
        return float("nan"), float("nan")
    dist = nearest_receptor_distance(surface.interface_xyz, surface.sampled_xyz[valid])
    return float(dist.mean()), float(dist.max())


def patch_components(surface: PeptideSurface, link_radius: float = 2.0) -> list[int]:
    """Sizes of the connected components of the retained patch, largest first.

    Two vertices are linked when they are within ``link_radius`` (PyMOL's default surface
    tessellation puts neighbouring vertices well under 1 A apart, so 2 A links a genuinely
    contiguous patch without bridging across the peptide). A second sizeable component
    usually means the peptide contacts two receptor lobes; many tiny ones mean the cutoff
    is slicing speckle out of the mesh.
    """
    xyz = surface.interface_xyz
    n = xyz.shape[0]
    if n == 0:
        return []
    try:
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        adjacency = cKDTree(xyz).sparse_distance_matrix(cKDTree(xyz), link_radius, output_type="coo_matrix")
        n_comp, labels = connected_components(adjacency, directed=False)
    except ImportError:  # pragma: no cover - environment-dependent
        return [n]
    return sorted(np.bincount(labels, minlength=n_comp).tolist(), reverse=True)


# ---------------------------------------------------------------------------
# Dataset-metadata helpers
# ---------------------------------------------------------------------------


def cache_shard_parts(example_id: str) -> tuple[str, str]:
    """Two-level hex prefix so 2.7M caches are not dumped into one ZFS directory."""
    import hashlib

    digest = hashlib.md5(example_id.encode("utf-8")).hexdigest()
    return digest[:2], digest[2:4]


def cache_path_for(
    output_dir: str | os.PathLike,
    example_id: str,
    *,
    shard: bool = True,
) -> Path:
    """Canonical write path for one complex.

    Default is sharded: ``<output_dir>/<hh>/<hh>/<example_id>.surface.npz``. Flat
    ``<output_dir>/<example_id>.surface.npz`` remains readable via
    :func:`resolve_cache_path` for older runs.
    """
    name = f"{example_id}.surface.npz"
    root = Path(output_dir)
    if not shard:
        return root / name
    a, b = cache_shard_parts(example_id)
    return root / a / b / name


def resolve_cache_path(output_dir: str | os.PathLike, example_id: str) -> Path:
    """Existing cache path (sharded or flat), else the sharded write location."""
    sharded = cache_path_for(output_dir, example_id, shard=True)
    if sharded.exists():
        return sharded
    flat = cache_path_for(output_dir, example_id, shard=False)
    if flat.exists():
        return flat
    return sharded


def resolve_chain_assignment(
    pdb_path: str | os.PathLike,
    peptide_chains: Iterable[str] | str,
    receptor_chains: Iterable[str] | str | None = None,
) -> tuple[list[str], list[str]]:
    """Turn our metadata's ``binder_chain_id`` into explicit (receptor, peptide) lists.

    Our CPSea metadata records only the binder chain; the receptor is "everything else",
    which has to be read off the file rather than assumed, because the preprocessed
    receptor may be a multi-chain pocket crop.
    """
    if isinstance(peptide_chains, str):
        peptide = [c for c in peptide_chains.split(",") if c]
    else:
        peptide = [c for c in peptide_chains if c]
    if not peptide:
        raise SurfaceExtractionError("no peptide chain given", reason="empty_chain_spec")

    if receptor_chains is None:
        present = read_chain_ids(pdb_path)
        receptor = [c for c in present if c not in set(peptide)]
    elif isinstance(receptor_chains, str):
        receptor = [c for c in receptor_chains.split(",") if c]
    else:
        receptor = [c for c in receptor_chains if c]

    if not receptor:
        raise SurfaceExtractionError(
            f"no receptor chains left after removing peptide {peptide} from {pdb_path}",
            reason="empty_chain_spec",
        )
    return receptor, peptide
