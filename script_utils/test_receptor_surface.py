"""Tests for `proteinfoundation.surface.peptide_surface.extract_receptor_surface`.

Pure-geometry / SAS-backend tests: no PyMOL needed (this function is SAS-only), just
numpy/scipy, so these run anywhere `test_peptide_surface.py`'s SAS tests do.

    pytest script_utils/test_receptor_surface.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from proteinfoundation.surface.peptide_surface import (
    DEFAULT_CUTOFF,
    EXTRACTOR_VERSION,
    RECEPTOR_SURFACE_EXTRACTOR_VERSION,
    SurfaceExtractionError,
    cache_path_for,
    extract_peptide_surface,
    extract_receptor_surface,
    is_cache_valid,
    load_surface_cache,
    resolve_cache_path,
    save_surface_cache,
)


def _atom_line(serial, name, resname, chain, resseq, xyz, element):
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5d} {name:<4s}{resname:>4s} {chain}{resseq:>4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


@pytest.fixture
def contacting_complex_pdb(tmp_path):
    """A small receptor (chain A) and peptide (chain B) close enough that their
    SAS surfaces genuinely interpenetrate within `DEFAULT_CUTOFF` -- unlike
    `test_peptide_surface.py`'s `two_chain_pdb` fixture (5 A apart, built for
    chain-splitting tests, not for a real interface)."""
    path = tmp_path / "complex.pdb"
    lines = ["REMARK   synthetic contacting complex\n"]
    serial = 1
    # Receptor: a small cluster around the origin.
    for i, xyz in enumerate([(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.0, 1.4, 0.0), (0.0, 0.0, 1.4)]):
        lines.append(_atom_line(serial, "C", "ALA", "A", i + 1, xyz, "C"))
        serial += 1
    lines.append("TER\n")
    # Peptide: a small cluster 3.0 A away in x -- well within a 4 A SAS-to-SAS cutoff
    # given ~1.7-3.1 A of vdW+probe radius sticking out from each side.
    for i, xyz in enumerate([(3.0, 0.0, 0.0), (4.4, 0.0, 0.0), (3.0, 1.4, 0.0)]):
        lines.append(_atom_line(serial, "C", "GLY", "B", i + 1, xyz, "C"))
        serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))
    return path


# ---------------------------------------------------------------------------
# A. Core contract
# ---------------------------------------------------------------------------
def test_extract_receptor_surface_returns_a_nonempty_pocket_patch(contacting_complex_pdb):
    surface = extract_receptor_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    assert surface.num_interface > 0
    assert surface.num_sampled > 0
    assert surface.metadata["role"] == "receptor"
    assert surface.metadata["extractor_version"] == RECEPTOR_SURFACE_EXTRACTOR_VERSION
    assert surface.metadata["backend"] == "sas"
    # Every retained point is within cutoff of the PEPTIDE (not the receptor's own atoms).
    assert (surface.interface_receptor_distance <= 4.0).all()


def test_extract_receptor_surface_points_lie_near_a_receptor_atom(contacting_complex_pdb):
    """The sampled points must sit on the RECEPTOR's own molecular surface (i.e. near a
    receptor heavy atom, within vdW(C) + probe radius), even though the distance stored
    alongside each point is measured to the peptide -- this is the property that makes
    the conditioning signal non-circular (it describes the receptor's shape, not the
    peptide's, unlike `extract_peptide_surface`'s output)."""
    surface = extract_receptor_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    receptor_atoms = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 1.4]])
    valid_pts = surface.sampled_xyz[surface.sampled_valid_mask]
    nearest_receptor_atom_dist = np.linalg.norm(
        valid_pts[:, None, :] - receptor_atoms[None, :, :], axis=-1
    ).min(axis=1)
    # vdW(C)=1.70 + probe=1.4 = 3.10 A is the exact SAS shell radius; small numerical margin.
    assert (nearest_receptor_atom_dist < 3.2).all()


def test_extract_receptor_surface_is_deterministic(contacting_complex_pdb):
    a = extract_receptor_surface(contacting_complex_pdb, ["A"], ["B"], cutoff=4.0, num_points=32, seed=0)
    b = extract_receptor_surface(contacting_complex_pdb, ["A"], ["B"], cutoff=4.0, num_points=32, seed=0)
    assert np.array_equal(a.sampled_xyz, b.sampled_xyz)


def test_extract_receptor_surface_rejects_chain_overlap(contacting_complex_pdb):
    with pytest.raises(SurfaceExtractionError) as exc:
        extract_receptor_surface(contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["A"])
    assert exc.value.reason == "chain_overlap"


def test_extract_receptor_surface_no_interface_beyond_cutoff(contacting_complex_pdb):
    """Two clusters far outside SAS-to-SAS range must raise `empty_interface`,
    the same typed failure `extract_peptide_surface` uses -- not a silent empty array."""
    with pytest.raises(SurfaceExtractionError) as exc:
        extract_receptor_surface(
            contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=1e-6, num_points=32
        )
    assert exc.value.reason == "empty_interface"


# ---------------------------------------------------------------------------
# B. Cache namespacing -- must never collide with a peptide-surface cache
# ---------------------------------------------------------------------------
def test_receptor_and_peptide_caches_do_not_collide_on_disk(tmp_path, contacting_complex_pdb):
    pep_surface = extract_peptide_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    rec_surface = extract_receptor_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    example_id = "complex_0"
    pep_path = cache_path_for(tmp_path, example_id, suffix="surface")
    rec_path = cache_path_for(tmp_path, example_id, suffix="receptor_surface")
    assert pep_path != rec_path

    save_surface_cache(pep_path, pep_surface)
    save_surface_cache(rec_path, rec_surface)
    assert pep_path.exists() and rec_path.exists()

    reloaded_pep = load_surface_cache(resolve_cache_path(tmp_path, example_id, suffix="surface"))
    reloaded_rec = load_surface_cache(resolve_cache_path(tmp_path, example_id, suffix="receptor_surface"))
    assert reloaded_pep.metadata.get("role") != "receptor"
    assert reloaded_rec.metadata["role"] == "receptor"
    assert not np.array_equal(reloaded_pep.sampled_xyz, reloaded_rec.sampled_xyz)


def test_is_cache_valid_rejects_receptor_cache_under_peptide_version(tmp_path, contacting_complex_pdb):
    """A receptor-surface cache must fail `is_cache_valid` under the peptide extractor's
    version (and vice versa) -- this is what stops `AttachPeptideSurfaceTransform`-style
    code from silently reading one role's cache as the other's."""
    rec_surface = extract_receptor_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    path = cache_path_for(tmp_path, "complex_0", suffix="receptor_surface")
    save_surface_cache(path, rec_surface)

    assert is_cache_valid(path, num_points=32, version=RECEPTOR_SURFACE_EXTRACTOR_VERSION, backend="sas")
    assert not is_cache_valid(path, num_points=32, version=EXTRACTOR_VERSION, backend="sas")


# ---------------------------------------------------------------------------
# C. AttachReceptorSurfaceTransform -- the dataset-pipeline wiring
# ---------------------------------------------------------------------------
def test_attach_receptor_surface_transform_loads_the_receptor_cache(tmp_path, contacting_complex_pdb):
    from proteinfoundation.datasets.transforms import AttachReceptorSurfaceTransform, Data

    rec_surface = extract_receptor_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    save_surface_cache(cache_path_for(tmp_path, "ex1", suffix="receptor_surface"), rec_surface)

    g = Data(id="ex1", example_id="ex1")
    g = AttachReceptorSurfaceTransform(str(tmp_path), num_points=32)(g)
    assert g.surface_xyz.shape == (32, 3)
    assert torch.equal(g.surface_mask, torch.from_numpy(rec_surface.sampled_valid_mask))


def test_peptide_and_receptor_transforms_never_cross_read_the_wrong_cache(tmp_path, contacting_complex_pdb):
    """Regression for the review's core complaint: pointing BOTH transforms at the SAME
    surface_dir must still load each role's own cache, not silently substitute the other."""
    from proteinfoundation.datasets.transforms import (
        AttachPeptideSurfaceTransform,
        AttachReceptorSurfaceTransform,
        Data,
    )

    pep_surface = extract_peptide_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    rec_surface = extract_receptor_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    save_surface_cache(cache_path_for(tmp_path, "ex1", suffix="surface"), pep_surface)
    save_surface_cache(cache_path_for(tmp_path, "ex1", suffix="receptor_surface"), rec_surface)

    g_pep = AttachPeptideSurfaceTransform(str(tmp_path), num_points=32)(Data(id="ex1", example_id="ex1"))
    g_rec = AttachReceptorSurfaceTransform(str(tmp_path), num_points=32)(Data(id="ex1", example_id="ex1"))

    assert not torch.equal(g_pep.surface_xyz, g_rec.surface_xyz)
    assert torch.allclose(g_pep.surface_xyz, torch.from_numpy(pep_surface.sampled_xyz))
    assert torch.allclose(g_rec.surface_xyz, torch.from_numpy(rec_surface.sampled_xyz))


def test_attach_receptor_surface_transform_rejects_a_peptide_cache_as_invalid(tmp_path, contacting_complex_pdb):
    """A directory that only has a PEPTIDE cache (e.g. an old run not yet migrated) must
    make the receptor transform treat it as missing, not silently attach peptide data
    under the receptor's semantics."""
    from proteinfoundation.datasets.transforms import AttachReceptorSurfaceTransform, Data

    pep_surface = extract_peptide_surface(
        contacting_complex_pdb, receptor_chains=["A"], peptide_chains=["B"], cutoff=4.0, num_points=32
    )
    save_surface_cache(cache_path_for(tmp_path, "ex1", suffix="surface"), pep_surface)

    with pytest.raises(FileNotFoundError):
        AttachReceptorSurfaceTransform(str(tmp_path), num_points=32)(Data(id="ex1", example_id="ex1"))


ALL_TESTS = [
    test_extract_receptor_surface_returns_a_nonempty_pocket_patch,
    test_extract_receptor_surface_points_lie_near_a_receptor_atom,
    test_extract_receptor_surface_is_deterministic,
    test_extract_receptor_surface_rejects_chain_overlap,
    test_extract_receptor_surface_no_interface_beyond_cutoff,
    test_receptor_and_peptide_caches_do_not_collide_on_disk,
    test_is_cache_valid_rejects_receptor_cache_under_peptide_version,
    test_attach_receptor_surface_transform_loads_the_receptor_cache,
    test_peptide_and_receptor_transforms_never_cross_read_the_wrong_cache,
    test_attach_receptor_surface_transform_rejects_a_peptide_cache_as_invalid,
]
