"""Tests for the PepBridge-derived peptide-interface surface extractor.

Most tests are pure geometry and run anywhere. The end-to-end test needs PyMOL and a real
CPSea complex, so it is skipped (never silently passed) when either is missing -- see
`test_end_to_end_on_a_real_cpsea_complex`.

    pytest script_utils/test_peptide_surface.py -v
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from proteinfoundation.surface.peptide_surface import (
    DEFAULT_CUTOFF,
    EXTRACTOR_VERSION,
    PeptideSurface,
    SurfaceExtractionError,
    extract_peptide_surface,
    farthest_point_sample,
    interface_mask,
    is_cache_valid,
    load_surface_cache,
    nearest_receptor_distance,
    normalize_normals,
    pad_to,
    read_chain_ids,
    resolve_chain_assignment,
    sampling_coverage,
    save_surface_cache,
    split_chains,
    transform_surface,
    _chunked_nearest_distance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CPSEA_SAMPLE_DIR = REPO_ROOT / "CPSea_data" / "preprocessed_sample100" / "processed" / "train"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _atom_line(serial, name, resname, chain, resseq, xyz, element):
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5d} {name:<4s}{resname:>4s} {chain}{resseq:>4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


@pytest.fixture
def two_chain_pdb(tmp_path):
    """A 3-atom receptor (chain A) and a 2-atom peptide (chain B), plus a stray chain C."""
    path = tmp_path / "complex.pdb"
    lines = [
        "REMARK   synthetic test complex\n",
        _atom_line(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0), "N"),
        _atom_line(2, "CA", "ALA", "A", 1, (1.5, 0.0, 0.0), "C"),
        _atom_line(3, "C", "ALA", "A", 2, (3.0, 0.0, 0.0), "C"),
        "TER\n",
        _atom_line(4, "N", "GLY", "B", 1, (0.0, 5.0, 0.0), "N"),
        _atom_line(5, "CA", "GLY", "B", 1, (1.5, 5.0, 0.0), "C"),
        "TER\n",
        _atom_line(6, "O", "HOH", "C", 1, (9.0, 9.0, 9.0), "O"),
        "CONECT    1    2\n",
        "END\n",
    ]
    path.write_text("".join(lines))
    return path


def _make_surface(n_interface=120, num_points=96, seed=0, cutoff=DEFAULT_CUTOFF):
    """A synthetic PeptideSurface with the same invariants a real one has."""
    rng = np.random.default_rng(1234)
    full = rng.normal(size=(400, 3)).astype(np.float32) * 5.0
    full_n = normalize_normals(rng.normal(size=(400, 3)))
    intf = full[:n_interface].copy()
    intf_n = full_n[:n_interface].copy()
    dist = rng.uniform(0.0, cutoff, size=n_interface).astype(np.float32)

    idx = farthest_point_sample(intf, num_points, seed=seed)
    n_valid = idx.shape[0]
    mask = np.zeros(num_points, dtype=bool)
    mask[:n_valid] = True
    return PeptideSurface(
        full_peptide_xyz=full,
        full_peptide_normals=full_n,
        interface_xyz=intf,
        interface_normals=intf_n,
        interface_receptor_distance=dist,
        sampled_xyz=pad_to(intf[idx], num_points),
        sampled_normals=pad_to(intf_n[idx], num_points),
        sampled_receptor_distance=pad_to(dist[idx], num_points),
        sampled_valid_mask=mask,
        metadata={
            "source_pdb": "/synthetic/complex.pdb",
            "receptor_chains": ["A"],
            "peptide_chains": ["B"],
            "cutoff": float(cutoff),
            "sample_count": int(num_points),
            "seed": int(seed),
            "extractor_version": EXTRACTOR_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Chain splitting
# ---------------------------------------------------------------------------


def test_chain_split_keeps_exactly_the_requested_atoms(two_chain_pdb, tmp_path):
    rec, pep = tmp_path / "rec.pdb", tmp_path / "pep.pdb"
    assert split_chains(two_chain_pdb, ["A"], rec) == 3
    assert split_chains(two_chain_pdb, ["B"], pep) == 2

    rec_lines = [line for line in rec.read_text().splitlines() if line.startswith("ATOM")]
    pep_lines = [line for line in pep.read_text().splitlines() if line.startswith("ATOM")]
    assert {line[21] for line in rec_lines} == {"A"}
    assert {line[21] for line in pep_lines} == {"B"}

    # The kept records are byte-identical to the source's, so coordinates and atom names
    # cannot have drifted through the split.
    source_a = [
        line
        for line in two_chain_pdb.read_text().splitlines()
        if line.startswith("ATOM") and line[21] == "A"
    ]
    assert rec_lines == source_a


def test_chain_split_does_not_modify_the_source(two_chain_pdb, tmp_path):
    before = two_chain_pdb.read_bytes()
    split_chains(two_chain_pdb, ["A"], tmp_path / "rec.pdb")
    split_chains(two_chain_pdb, ["B"], tmp_path / "pep.pdb")
    assert two_chain_pdb.read_bytes() == before


def test_chain_split_drops_connectivity_and_terminates_the_file(two_chain_pdb, tmp_path):
    out = tmp_path / "rec.pdb"
    split_chains(two_chain_pdb, ["A"], out)
    text = out.read_text()
    assert "CONECT" not in text
    assert "REMARK" not in text
    assert text.rstrip().endswith("END")


def test_chain_split_rejects_an_absent_chain(two_chain_pdb, tmp_path):
    with pytest.raises(SurfaceExtractionError) as exc:
        split_chains(two_chain_pdb, ["Z"], tmp_path / "z.pdb")
    assert exc.value.reason == "missing_chain"


def test_missing_pdb_is_a_typed_failure_not_a_traceback(tmp_path):
    # A dataset row whose file has vanished must be bucketed, not treated as a bug.
    with pytest.raises(SurfaceExtractionError) as exc:
        resolve_chain_assignment(tmp_path / "gone.pdb", "B")
    assert exc.value.reason == "missing_pdb"


def test_receptor_defaults_to_every_non_peptide_chain(two_chain_pdb):
    assert read_chain_ids(two_chain_pdb) == ["A", "B", "C"]
    receptor, peptide = resolve_chain_assignment(two_chain_pdb, "B")
    assert peptide == ["B"]
    assert receptor == ["A", "C"]


def test_explicit_receptor_chains_win(two_chain_pdb):
    receptor, peptide = resolve_chain_assignment(two_chain_pdb, "B", "A")
    assert (receptor, peptide) == (["A"], ["B"])


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------


def test_normalize_normals_gives_unit_length():
    raw = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [-1.0, -1.0, -1.0]])
    unit = normalize_normals(raw)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-6)
    # Direction is preserved, only the magnitude changes.
    assert np.allclose(unit[0], [0.6, 0.8, 0.0], atol=1e-6)


def test_normalize_normals_zeroes_degenerate_rows_instead_of_producing_nan():
    unit = normalize_normals(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert np.isfinite(unit).all()
    assert np.allclose(unit[0], 0.0)
    assert np.allclose(np.linalg.norm(unit[1]), 1.0)


# ---------------------------------------------------------------------------
# Nearest-neighbour distance and interface filtering
# ---------------------------------------------------------------------------


def test_kdtree_and_chunked_search_agree():
    rng = np.random.default_rng(0)
    query = rng.normal(size=(97, 3)) * 4.0
    reference = rng.normal(size=(233, 3)) * 4.0
    assert np.allclose(
        nearest_receptor_distance(query, reference),
        _chunked_nearest_distance(query, reference, chunk_size=16),
        atol=1e-5,
    )


def test_nearest_distance_is_the_true_minimum():
    receptor = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    peptide = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert np.allclose(nearest_receptor_distance(peptide, receptor), [1.0, 1.0, 5.0])


def test_interface_mask_keeps_inside_and_excludes_outside():
    dist = np.array([0.0, 2.0, 3.999, 4.0, 4.001, 12.0])
    mask = interface_mask(dist, 4.0)
    assert mask.tolist() == [True, True, True, True, False, False]
    assert (dist[mask] <= 4.0).all()
    assert (dist[~mask] > 4.0).all()


def test_points_beyond_the_cutoff_are_excluded_end_to_end():
    receptor = np.zeros((1, 3))
    peptide = np.array([[1.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [6.0, 0, 0], [20.0, 0, 0]])
    dist = nearest_receptor_distance(peptide, receptor)
    kept = peptide[interface_mask(dist, 4.0)]
    assert kept.shape[0] == 3
    assert 6.0 not in kept[:, 0]
    assert 20.0 not in kept[:, 0]


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------


def test_fps_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(7)
    xyz = rng.normal(size=(500, 3))
    a = farthest_point_sample(xyz, 96, seed=0)
    b = farthest_point_sample(xyz, 96, seed=0)
    assert np.array_equal(a, b)


def test_fps_differs_across_seeds_but_stays_reproducible():
    rng = np.random.default_rng(7)
    xyz = rng.normal(size=(500, 3))
    a = farthest_point_sample(xyz, 96, seed=0)
    c = farthest_point_sample(xyz, 96, seed=1)
    assert not np.array_equal(a, c)
    assert np.array_equal(c, farthest_point_sample(xyz, 96, seed=1))


def test_fps_returns_distinct_indices_and_caps_at_the_point_count():
    rng = np.random.default_rng(3)
    xyz = rng.normal(size=(40, 3))
    idx = farthest_point_sample(xyz, 96, seed=0)
    assert idx.shape[0] == 40
    assert len(set(idx.tolist())) == 40


def test_fps_spreads_over_separated_clusters():
    # Four tight clusters far apart: any sensible FPS hits all four within four picks.
    centres = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]], dtype=float)
    rng = np.random.default_rng(0)
    xyz = np.concatenate([c + rng.normal(scale=0.2, size=(50, 3)) for c in centres])
    idx = farthest_point_sample(xyz, 4, seed=0)
    assigned = {int(np.argmin(np.linalg.norm(centres - xyz[i], axis=1))) for i in idx}
    assert assigned == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Padding and validity mask
# ---------------------------------------------------------------------------


def test_padding_is_marked_by_the_validity_mask():
    surface = _make_surface(n_interface=30, num_points=96)
    assert surface.sampled_valid_mask.sum() == 30
    assert surface.sampled_valid_mask[:30].all()
    assert not surface.sampled_valid_mask[30:].any()
    # Padded rows are exactly zero, so a consumer that ignores the mask fails loudly
    # (zero-norm normals) rather than training on plausible-looking junk.
    assert np.allclose(surface.sampled_xyz[~surface.sampled_valid_mask], 0.0)
    assert np.allclose(surface.sampled_normals[~surface.sampled_valid_mask], 0.0)


def test_no_padding_when_enough_points_exist():
    surface = _make_surface(n_interface=200, num_points=96)
    assert surface.sampled_valid_mask.all()
    assert surface.sampled_xyz.shape == (96, 3)


def test_valid_sampled_rows_are_finite_and_unit_normalled():
    surface = _make_surface(n_interface=30, num_points=96)
    valid = surface.sampled_valid_mask
    assert np.isfinite(surface.sampled_xyz).all()
    assert np.isfinite(surface.sampled_normals).all()
    norms = np.linalg.norm(surface.sampled_normals[valid], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_round_trip_preserves_every_array_and_the_metadata(tmp_path):
    surface = _make_surface(n_interface=57, num_points=96)
    path = save_surface_cache(tmp_path / "x.surface.npz", surface)
    loaded = load_surface_cache(path)

    for field in (
        "full_peptide_xyz",
        "full_peptide_normals",
        "interface_xyz",
        "interface_normals",
        "interface_receptor_distance",
        "sampled_xyz",
        "sampled_normals",
        "sampled_receptor_distance",
    ):
        original, restored = getattr(surface, field), getattr(loaded, field)
        assert restored.dtype == original.dtype, field
        assert np.array_equal(restored, original), field
    assert loaded.sampled_valid_mask.dtype == bool
    assert np.array_equal(loaded.sampled_valid_mask, surface.sampled_valid_mask)
    assert loaded.metadata == surface.metadata

    for key in (
        "source_pdb",
        "receptor_chains",
        "peptide_chains",
        "cutoff",
        "sample_count",
        "seed",
        "extractor_version",
    ):
        assert key in loaded.metadata, key


def test_cache_validity_rejects_mismatched_settings(tmp_path):
    surface = _make_surface(num_points=96, seed=0)
    path = save_surface_cache(tmp_path / "x.surface.npz", surface)

    assert is_cache_valid(path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION)
    assert not is_cache_valid(path, 6.0, 96, 0, EXTRACTOR_VERSION)
    assert not is_cache_valid(path, DEFAULT_CUTOFF, 64, 0, EXTRACTOR_VERSION)
    assert not is_cache_valid(path, DEFAULT_CUTOFF, 96, 3, EXTRACTOR_VERSION)
    assert not is_cache_valid(path, DEFAULT_CUTOFF, 96, 0, "0.0.0-other")
    assert not is_cache_valid(tmp_path / "does_not_exist.npz")


def test_cache_validity_rejects_mismatched_backend_and_resolution(tmp_path):
    """Regression: `backend`/`sas_points_per_atom` are stored in every cache's
    metadata (see `extract_peptide_surface`) but were never checked by
    `is_cache_valid`, so a cache extracted with one backend (or SAS resolution)
    was silently accepted for a request specifying a different one whenever
    cutoff/num_points/seed/version happened to match."""
    surface = _make_surface(num_points=96, seed=0)
    surface.metadata["backend"] = "pymol"
    path = save_surface_cache(tmp_path / "pymol.surface.npz", surface)

    assert is_cache_valid(path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION, backend="pymol")
    assert not is_cache_valid(path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION, backend="sas")
    # No backend requested at all: unaffected, matches prior behavior.
    assert is_cache_valid(path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION)

    sas_surface = _make_surface(num_points=96, seed=0)
    sas_surface.metadata["backend"] = "sas"
    sas_surface.metadata["sas_points_per_atom"] = 20
    sas_path = save_surface_cache(tmp_path / "sas.surface.npz", sas_surface)

    assert is_cache_valid(sas_path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION, backend="sas", sas_points_per_atom=20)
    assert not is_cache_valid(
        sas_path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION, backend="sas", sas_points_per_atom=40
    )
    # A pymol cache has no sas_points_per_atom key at all; requesting one without
    # also pinning backend="sas" must not be treated as a mismatch for it.
    assert is_cache_valid(path, DEFAULT_CUTOFF, 96, 0, EXTRACTOR_VERSION, sas_points_per_atom=20)


def test_truncated_cache_is_not_valid(tmp_path):
    path = save_surface_cache(tmp_path / "x.surface.npz", _make_surface())
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 3])
    assert not is_cache_valid(path)


def test_cache_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    save_surface_cache(tmp_path / "x.surface.npz", _make_surface())
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def test_rigid_transform_rotates_and_translates_points_but_only_rotates_normals():
    surface = _make_surface(n_interface=40, num_points=96)
    R = _rotation([0.3, -0.7, 0.5], 0.9)
    t = np.array([12.0, -4.5, 3.25])
    moved = transform_surface(surface, R, t)

    assert np.allclose(moved.interface_xyz, surface.interface_xyz @ R.T + t, atol=1e-4)
    assert np.allclose(moved.full_peptide_xyz, surface.full_peptide_xyz @ R.T + t, atol=1e-4)
    # Normals rotate without picking up the translation.
    assert np.allclose(moved.interface_normals, surface.interface_normals @ R.T, atol=1e-5)
    assert np.allclose(moved.full_peptide_normals, surface.full_peptide_normals @ R.T, atol=1e-5)


def test_rigid_transform_preserves_norms_and_pairwise_geometry():
    surface = _make_surface(n_interface=40, num_points=96)
    moved = transform_surface(surface, _rotation([1.0, 1.0, 0.0], 2.1), [5.0, 5.0, 5.0])
    valid = surface.sampled_valid_mask

    assert np.allclose(np.linalg.norm(moved.interface_normals, axis=1), 1.0, atol=1e-5)
    before = np.linalg.norm(surface.interface_xyz[:20, None] - surface.interface_xyz[None, :20], axis=-1)
    after = np.linalg.norm(moved.interface_xyz[:20, None] - moved.interface_xyz[None, :20], axis=-1)
    assert np.allclose(before, after, atol=1e-3)
    # The distance-to-receptor column is a scalar invariant of a rigid motion.
    assert np.array_equal(moved.interface_receptor_distance, surface.interface_receptor_distance)
    assert np.array_equal(moved.sampled_valid_mask, valid)


def test_rigid_transform_leaves_padded_rows_at_zero():
    surface = _make_surface(n_interface=20, num_points=96)
    moved = transform_surface(surface, _rotation([0, 0, 1], 1.0), [7.0, 0.0, 0.0])
    pad = ~surface.sampled_valid_mask
    assert np.allclose(moved.sampled_xyz[pad], 0.0)
    assert np.allclose(moved.sampled_normals[pad], 0.0)


def test_transform_rejects_a_bad_rotation_shape():
    with pytest.raises(ValueError):
        transform_surface(_make_surface(), np.eye(4))


# ---------------------------------------------------------------------------
# Coverage diagnostic
# ---------------------------------------------------------------------------


def test_sampling_coverage_is_zero_when_every_point_is_sampled():
    surface = _make_surface(n_interface=40, num_points=96)
    mean, maximum = sampling_coverage(surface)
    assert mean == pytest.approx(0.0, abs=1e-5)
    assert maximum == pytest.approx(0.0, abs=1e-5)


def test_sampling_coverage_grows_when_samples_collapse():
    surface = _make_surface(n_interface=300, num_points=96)
    spread_mean, spread_max = sampling_coverage(surface)

    collapsed = _make_surface(n_interface=300, num_points=96)
    # Replace the FPS picks with the 96 mutually closest points (one lobe).
    order = np.argsort(np.linalg.norm(collapsed.interface_xyz - collapsed.interface_xyz[0], axis=1))
    collapsed.sampled_xyz = collapsed.interface_xyz[order[:96]]
    collapsed_mean, collapsed_max = sampling_coverage(collapsed)

    assert collapsed_max > spread_max
    assert collapsed_mean > spread_mean


# ---------------------------------------------------------------------------
# End-to-end on a real CPSea complex
# ---------------------------------------------------------------------------


def _real_complex() -> Path | None:
    if not CPSEA_SAMPLE_DIR.is_dir():
        return None
    candidates = sorted(CPSEA_SAMPLE_DIR.glob("*.pdb"))
    return candidates[0] if candidates else None


def _pymol_available() -> bool:
    # Use find_spec: a bare ``import pymol`` can hang for tens of seconds (or forever
    # without a display) during pytest collection, long before finish_launching("-qck").
    import importlib.util

    return (
        importlib.util.find_spec("pymol") is not None
        and importlib.util.find_spec("trimesh") is not None
    )


real_complex_test = pytest.mark.skipif(
    _real_complex() is None or not _pymol_available(),
    reason="needs pymol-open-source, trimesh and CPSea_data/preprocessed_sample100",
)


@pytest.fixture(scope="module")
def real_surface():
    pdb = _real_complex()
    if pdb is None or not _pymol_available():
        pytest.skip("needs pymol-open-source, trimesh and CPSea_data/preprocessed_sample100")
    receptor, peptide = resolve_chain_assignment(pdb, "B")
    return pdb, extract_peptide_surface(
        pdb,
        receptor_chains=receptor,
        peptide_chains=peptide,
        cutoff=4.0,
        num_points=96,
        seed=0,
        backend="pymol",
    )


@real_complex_test
def test_end_to_end_on_a_real_cpsea_complex(real_surface):
    pdb, surface = real_surface

    assert surface.num_full > 100, "peptide surface is implausibly coarse"
    assert surface.num_interface > 0, "no interface found on a real complex"
    assert surface.num_interface <= surface.num_full

    # Everything finite.
    for field in (
        "full_peptide_xyz",
        "full_peptide_normals",
        "interface_xyz",
        "interface_normals",
        "interface_receptor_distance",
        "sampled_xyz",
        "sampled_normals",
        "sampled_receptor_distance",
    ):
        assert np.isfinite(getattr(surface, field)).all(), field

    # Every retained vertex really is within the cutoff.
    assert (surface.interface_receptor_distance <= 4.0 + 1e-5).all()
    valid = surface.sampled_valid_mask
    assert (surface.sampled_receptor_distance[valid] <= 4.0 + 1e-5).all()

    # Non-padded normals are unit length.
    assert np.allclose(np.linalg.norm(surface.interface_normals, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(surface.sampled_normals[valid], axis=1), 1.0, atol=1e-4)

    assert surface.metadata["extractor_version"] == EXTRACTOR_VERSION
    assert surface.metadata["source_pdb"] == str(pdb.resolve())


@real_complex_test
def test_real_surface_stays_in_the_source_coordinate_frame(real_surface):
    """The mesh must be the peptide's atoms grown by a probe radius -- not recentred."""
    pdb, surface = real_surface
    coords = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[21] == "B":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    atoms = np.asarray(coords)

    # Centroid offset well under a probe+vdW radius, and the mesh strictly encloses the atoms.
    assert np.linalg.norm(surface.full_peptide_xyz.mean(0) - atoms.mean(0)) < 1.5
    assert (surface.full_peptide_xyz.min(0) < atoms.min(0)).all()
    assert (surface.full_peptide_xyz.max(0) > atoms.max(0)).all()


@real_complex_test
def test_real_extraction_is_reproducible(real_surface, tmp_path):
    """Re-running preprocessing must give byte-identical arrays for the same settings."""
    pdb, surface = real_surface
    receptor, peptide = resolve_chain_assignment(pdb, "B")
    again = extract_peptide_surface(
        pdb,
        receptor_chains=receptor,
        peptide_chains=peptide,
        cutoff=4.0,
        num_points=96,
        seed=0,
        backend="pymol",
    )
    for field in ("full_peptide_xyz", "interface_xyz", "sampled_xyz", "sampled_receptor_distance"):
        assert np.array_equal(getattr(again, field), getattr(surface, field)), field

    path = save_surface_cache(tmp_path / "real.surface.npz", surface)
    reloaded = load_surface_cache(path)
    assert np.array_equal(reloaded.sampled_xyz, surface.sampled_xyz)
    assert json.loads(json.dumps(reloaded.metadata)) == surface.metadata


@real_complex_test
def test_real_normals_point_outward(real_surface):
    """A correctly oriented closed surface has most normals on the far side of the centroid."""
    _, surface = real_surface
    centre = surface.full_peptide_xyz.mean(axis=0)
    radial = surface.full_peptide_xyz - centre
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert ((radial * surface.full_peptide_normals).sum(axis=1) > 0).mean() > 0.75


@real_complex_test
def test_real_extraction_rejects_a_receptorless_request(real_surface):
    pdb, _ = real_surface
    with pytest.raises(SurfaceExtractionError) as exc:
        extract_peptide_surface(pdb, receptor_chains=["A"], peptide_chains=["A"])
    assert exc.value.reason == "chain_overlap"


@pytest.mark.skipif(_real_complex() is None, reason="needs CPSea_data/preprocessed_sample100")
def test_sas_backend_is_fast_and_reproducible():
    """Default v2 path: no PyMOL, tens of ms, deterministic FPS."""
    pdb = _real_complex()
    receptor, peptide = resolve_chain_assignment(pdb, "B")
    t0 = time.perf_counter()
    surface = extract_peptide_surface(
        pdb,
        receptor_chains=receptor,
        peptide_chains=peptide,
        cutoff=4.0,
        num_points=96,
        seed=0,
        backend="sas",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"SAS path too slow: {elapsed:.3f}s"
    assert surface.num_interface > 0
    assert surface.num_sampled == 96 or surface.num_sampled == surface.num_interface
    assert surface.metadata["backend"] == "sas"
    again = extract_peptide_surface(
        pdb,
        receptor_chains=receptor,
        peptide_chains=peptide,
        cutoff=4.0,
        num_points=96,
        seed=0,
        backend="sas",
    )
    assert np.array_equal(again.sampled_xyz, surface.sampled_xyz)
