"""Integration tests for oracle peptide-surface conditioning (stages 1–4).

    pytest script_utils/test_surface_conditioning.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from proteinfoundation.surface.peptide_surface import (
    PeptideSurface,
    farthest_point_sample,
    normalize_normals,
    pad_to,
    save_surface_cache,
)

# Heavy imports (atomworks transforms / LocalLatentsTransformer) are deferred into the
# tests that need them so collection stays light on memory-constrained nodes.

REPO = Path(__file__).resolve().parents[1]
SAMPLE_SURFACES = REPO / "surfaces" / "cpsea_sample100"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_nn_kwargs(enable_surface: bool = True):
    # Keep tiny: full FeatureFactory+14-layer trunks OOMs on shared login nodes.
    return dict(
        name="local_latents_transformer",
        nlayers=4,
        token_dim=32,
        nheads=4,
        pair_repr_dim=16,
        dim_cond=16,
        idx_emb_dim=16,
        t_emb_dim=16,
        latent_dim=4,
        parallel_mha_transition=False,
        update_pair_repr=False,
        update_pair_repr_every_n=3,
        use_tri_mult=False,
        use_qkln=True,
        strict_feats=False,
        feats_seq=["xt_bb_ca", "xt_local_latents"],
        feats_cond_seq=["time_emb_bb_ca", "time_emb_local_latents"],
        feats_pair_repr=["rel_seq_sep", "xt_bb_ca_pair_dists"],
        feats_pair_cond=["time_emb_bb_ca"],
        xt_pair_dist_dim=4,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        x_sc_pair_dist_dim=4,
        x_sc_pair_dist_min=0.1,
        x_sc_pair_dist_max=3.0,
        seq_sep_dim=7,
        concat_features={
            "enable_motif": False,
            "enable_target": False,
            "enable_ligand": False,
            "motif_pair_features": False,
            "target_pair_features": False,
            "ligand_pair_features": False,
        },
        output_parameterization={"bb_ca": "v", "local_latents": "v"},
        enable_surface=enable_surface,
        surface={
            "num_points": 8,
            "gate_layers": [1, 2, 3],
            "rbf_n": 4,
            "pair_dim": 16,
        },
    )


def _fake_batch(b=1, n=6, m=8, enable_surface=True):
    torch.manual_seed(0)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, -1] = False
    batch = {
        "mask": mask,
        "x_t": {
            "bb_ca": torch.randn(b, n, 3),
            "local_latents": torch.randn(b, n, 4),
        },
        "t": {"bb_ca": torch.rand(b), "local_latents": torch.rand(b)},
        "residue_type": torch.zeros(b, n, dtype=torch.long),
        "chains": torch.zeros(b, n, dtype=torch.long),
        "residue_pdb_idx": torch.arange(n)[None, :].expand(b, -1),
    }
    if enable_surface:
        xyz = torch.randn(b, m, 3)
        normals = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        smask = torch.ones(b, m, dtype=torch.bool)
        smask[:, -2:] = False
        xyz = xyz * smask[..., None]
        normals = normals * smask[..., None]
        batch.update(
            {
                "surface_xyz": xyz,
                "surface_normals": normals,
                "surface_mask": smask,
                "surface_distance": torch.rand(b, m) * smask.float(),
            }
        )
    return batch


def _write_toy_cache(path: Path, example_id: str, seed: int = 0, m: int = 96):
    rng = np.random.default_rng(seed)
    full = rng.normal(size=(200, 3)).astype(np.float32)
    full_n = normalize_normals(rng.normal(size=(200, 3)))
    intf = full[:120]
    intf_n = full_n[:120]
    dist = rng.uniform(0, 4, size=120).astype(np.float32)
    idx = farthest_point_sample(intf, m, seed=seed)
    n_valid = idx.shape[0]
    mask = np.zeros(m, dtype=bool)
    mask[:n_valid] = True
    surface = PeptideSurface(
        full_peptide_xyz=full,
        full_peptide_normals=full_n,
        interface_xyz=intf,
        interface_normals=intf_n,
        interface_receptor_distance=dist,
        sampled_xyz=pad_to(intf[idx], m),
        sampled_normals=pad_to(intf_n[idx], m),
        sampled_receptor_distance=pad_to(dist[idx], m),
        sampled_valid_mask=mask,
        metadata={
            "source_pdb": f"/tmp/{example_id}.pdb",
            "receptor_chains": ["A"],
            "peptide_chains": ["B"],
            "cutoff": 4.0,
            "sample_count": m,
            "seed": seed,
            "extractor_version": "1.0.0",
            "example_id": example_id,
        },
    )
    save_surface_cache(path, surface)
    return surface


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def test_attach_and_rigid_transforms_align_surface(tmp_path):
    from proteinfoundation.datasets.transforms import (
        AttachPeptideSurfaceTransform,
        CenteringTransform,
        CoordsToNanometers,
        Data,
        GlobalRotationTransform,
    )

    _write_toy_cache(tmp_path / "ex1.surface.npz", "ex1", seed=1)
    g = Data(
        id="ex1",
        example_id="ex1",
        coords=torch.randn(10, 37, 3),
        coord_mask=torch.ones(10, 37, dtype=torch.bool),
        target_mask=torch.zeros(10, 37, dtype=torch.bool),
    )
    # Fake target residues for centering
    g.target_mask[:5] = True
    g.coords[:5] += 10.0

    g = AttachPeptideSurfaceTransform(str(tmp_path), num_points=96)(g)
    assert g.surface_xyz.shape == (96, 3)
    xyz0 = g.surface_xyz.clone()

    dist0 = g.surface_distance.clone()
    g = CoordsToNanometers()(g)
    assert torch.allclose(g.surface_xyz, xyz0 / 10.0)
    assert torch.allclose(g.surface_distance, dist0 / 10.0)

    g = GlobalRotationTransform()(g)
    assert hasattr(g, "global_rotation")
    # Normals remain unit length on valid rows
    valid = g.surface_mask
    norms = g.surface_normals[valid].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    g = CenteringTransform(center_mode="target", data_mode="all-atom")(g)
    assert hasattr(g, "center_offset")
    assert g.center_offset.shape == (3,)


def test_collate_keeps_fixed_m_surface():
    from proteinfoundation.datasets.structure_data import structure_collate_fn
    from proteinfoundation.datasets.transforms import Data

    samples = []
    for i in range(2):
        samples.append(
            Data(
                num_nodes=5 + i,
                coords=torch.zeros(5 + i, 37, 3),
                surface_xyz=torch.randn(96, 3),
                surface_normals=torch.randn(96, 3),
                surface_mask=torch.ones(96, dtype=torch.bool),
                surface_distance=torch.rand(96),
                global_rotation=torch.eye(3),
                center_offset=torch.zeros(3),
            )
        )
    batch = structure_collate_fn(samples)
    assert batch["surface_xyz"].shape == (2, 96, 3)
    assert batch["surface_mask"].shape == (2, 96)
    assert batch["global_rotation"].shape == (2, 3, 3)
    assert batch["mask"].shape == (2, 6)  # max binder len


# ---------------------------------------------------------------------------
# Pair-feature SE(3) invariance
# ---------------------------------------------------------------------------


def test_intra_surface_pair_feats_are_se3_invariant():
    from proteinfoundation.nn.surface.encoder import IntraSurfacePairFeatures

    torch.manual_seed(0)
    b, m = 2, 12
    xyz = torch.randn(b, m, 3)
    normals = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
    mask = torch.ones(b, m, dtype=torch.bool)
    mod = IntraSurfacePairFeatures(pair_dim=16, rbf_n=8)
    base = mod(xyz, normals, mask)

    R = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(R) < 0:
        R[:, 0] *= -1
    t = torch.tensor([1.5, -0.7, 2.2])
    xyz2 = xyz @ R + t
    normals2 = normals @ R
    moved = mod(xyz2, normals2, mask)
    assert torch.allclose(base, moved, atol=1e-4)


def test_binder_surface_pair_feats_are_se3_invariant():
    from proteinfoundation.nn.surface.encoder import BinderSurfacePairFeatures

    torch.manual_seed(1)
    b, n, m = 2, 6, 10
    bx = torch.randn(b, n, 3)
    sx = torch.randn(b, m, 3)
    sn = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
    bm = torch.ones(b, n, dtype=torch.bool)
    sm = torch.ones(b, m, dtype=torch.bool)
    tt = torch.rand(b)
    mod = BinderSurfacePairFeatures(pair_dim=16, rbf_n=8)
    base = mod(bx, sx, sn, bm, sm, tt)

    R = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(R) < 0:
        R[:, 0] *= -1
    t = torch.tensor([0.3, 0.4, -0.5])
    moved = mod(bx @ R + t, sx @ R + t, sn @ R, bm, sm, tt)
    assert torch.allclose(base, moved, atol=1e-4)


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------


def test_zero_gates_match_surface_disabled():
    from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

    batch = _fake_batch(enable_surface=True)
    nn_on = LocalLatentsTransformer(**_tiny_nn_kwargs(enable_surface=True))
    nn_off = LocalLatentsTransformer(**_tiny_nn_kwargs(enable_surface=False))
    off_sd = nn_off.state_dict()
    on_sd = nn_on.state_dict()
    off_sd.update({k: v for k, v in on_sd.items() if k in off_sd and off_sd[k].shape == v.shape})
    nn_off.load_state_dict(off_sd)
    for layer in nn_on.surface_cross_layers.values():
        assert float(layer.gate.detach()) == 0.0

    nn_on.eval()
    nn_off.eval()
    with torch.no_grad():
        out_on = nn_on(batch)
        out_off = nn_off(batch)
    assert torch.allclose(out_on["bb_ca"]["v"], out_off["bb_ca"]["v"], atol=1e-5)
    assert torch.allclose(
        out_on["local_latents"]["v"], out_off["local_latents"]["v"], atol=1e-5
    )


def test_surface_mask_makes_padded_points_inert():
    from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

    batch = _fake_batch(enable_surface=True)
    nn = LocalLatentsTransformer(**_tiny_nn_kwargs(enable_surface=True))
    with torch.no_grad():
        for layer in nn.surface_cross_layers.values():
            layer.gate.data.fill_(1.0)
    nn.eval()
    with torch.no_grad():
        base = nn(batch)
        batch2 = {**batch, "x_t": {k: v.clone() for k, v in batch["x_t"].items()}}
        batch2["surface_xyz"] = batch["surface_xyz"].clone()
        batch2["surface_normals"] = batch["surface_normals"].clone()
        invalid = ~batch["surface_mask"]
        batch2["surface_xyz"][invalid] = torch.randn_like(batch2["surface_xyz"][invalid]) * 50
        batch2["surface_normals"][invalid] = torch.randn_like(batch2["surface_normals"][invalid])
        out = nn(batch2)
    assert torch.allclose(base["bb_ca"]["v"], out["bb_ca"]["v"], atol=1e-5)


def test_surface_modules_receive_nonzero_gradients():
    from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

    batch = _fake_batch(enable_surface=True)
    nn = LocalLatentsTransformer(**_tiny_nn_kwargs(enable_surface=True))
    with torch.no_grad():
        for layer in nn.surface_cross_layers.values():
            layer.gate.data.fill_(0.5)
    nn.train()
    out = nn(batch)
    loss = out["bb_ca"]["v"].pow(2).mean() + out["local_latents"]["v"].pow(2).mean()
    loss.backward()
    assert nn.surface_encoder.type_emb.grad is not None
    assert nn.surface_encoder.type_emb.grad.abs().sum() > 0
    assert nn.binder_surface_pair_feats.proj[0].weight.grad is not None
    assert nn.binder_surface_pair_feats.proj[0].weight.grad.abs().sum() > 0
    gate_grads = [float(layer.gate.grad.abs()) for layer in nn.surface_cross_layers.values()]
    assert all(g > 0 for g in gate_grads)


def test_open_gates_change_predictions():
    from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

    batch = _fake_batch(enable_surface=True)
    nn = LocalLatentsTransformer(**_tiny_nn_kwargs(enable_surface=True))
    nn.eval()
    with torch.no_grad():
        closed = nn(batch)
        for layer in nn.surface_cross_layers.values():
            layer.gate.data.fill_(1.0)
        opened = nn(batch)
    assert not torch.allclose(closed["bb_ca"]["v"], opened["bb_ca"]["v"], atol=1e-5)


# ---------------------------------------------------------------------------
# Shuffle transform
# ---------------------------------------------------------------------------


def test_shuffle_surface_choice_is_reproducible_across_processes(tmp_path):
    """Regression: the shuffle pick used Python's built-in `hash()` on the example
    id, which is salted per-process (`PYTHONHASHSEED`) unless explicitly disabled --
    so the same id could pick a DIFFERENT "other" cache in a fresh process despite
    an identical `seed`. `hash()` cannot be re-salted from within one already-running
    process (`PYTHONHASHSEED` is read once at interpreter startup), so the only way
    to observe the actual bug is a fresh subprocess per seed -- but spawning
    subprocesses from inside this test file has caused hangs when run alongside the
    other GPU/multiprocessing-touching tests in this module (fork-safety with
    lingering torch/dataloader threads), so this instead directly checks that the
    transform's pick matches an independent `hashlib`-based reference computation.
    `hash()` is not merely unseeded but a DIFFERENT algorithm entirely, so matching
    this reference on every call (never ~1/N by chance) is only possible if the
    implementation really does use the deterministic `hashlib` path this test pins
    down, not `hash()`.
    """
    import hashlib

    from proteinfoundation.datasets.transforms import Data, ShufflePeptideSurfaceTransform

    n_caches = 6
    for i in range(n_caches):
        _write_toy_cache(tmp_path / f"ex{i}.surface.npz", f"ex{i}", seed=i)

    for seed in (0, 7):
        transform = ShufflePeptideSurfaceTransform(str(tmp_path), num_points=96, seed=seed)
        g = Data(id="ex0", example_id="ex0")
        g = transform(g)

        own_hash = int(hashlib.sha256(b"ex0").hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed + own_hash)
        choices = sorted(f"ex{i}" for i in range(n_caches) if f"ex{i}" != "ex0")
        expected = choices[int(rng.integers(len(choices)))]

        assert g.surface_shuffled_from == expected


def test_shuffle_surface_picks_a_different_cache(tmp_path):
    from proteinfoundation.datasets.transforms import (
        AttachPeptideSurfaceTransform,
        Data,
        ShufflePeptideSurfaceTransform,
    )

    _write_toy_cache(tmp_path / "a.surface.npz", "a", seed=1)
    _write_toy_cache(tmp_path / "b.surface.npz", "b", seed=2)
    g = Data(id="a", example_id="a")
    g = AttachPeptideSurfaceTransform(str(tmp_path), num_points=96)(g)
    own = g.surface_xyz.clone()
    g = ShufflePeptideSurfaceTransform(str(tmp_path), num_points=96, seed=0)(g)
    assert g.surface_shuffled_from == "b"
    assert not torch.allclose(own, g.surface_xyz)


# ---------------------------------------------------------------------------
# Config compose / real caches (optional)
# ---------------------------------------------------------------------------


def test_surface_dataset_config_files_wire_oracle_path():
    """Static check that oracle / shuffle / baseline configs point at the right pieces.

    Full Hydra composition is verified out-of-band via
    ``python -m proteinfoundation.train --config-name=example/training_cpsea_peptide_surface_oracle --cfg job``
    (spawning that from pytest after importing atomworks OOMs on shared login nodes).
    """
    dataset_yaml = (REPO / "configs/dataset/unified/cpsea_peptide_surface.yaml").read_text()
    assert "AttachPeptideSurfaceTransform" in dataset_yaml
    shuffle_yaml = (REPO / "configs/dataset/unified/cpsea_peptide_surface_shuffle.yaml").read_text()
    assert "ShufflePeptideSurfaceTransform" in shuffle_yaml

    oracle = (REPO / "configs/example/training_cpsea_peptide_surface_oracle.yaml").read_text()
    assert "cpsea_peptide_surface" in oracle
    assert "enable_surface: true" in oracle

    baseline = (REPO / "configs/example/training_cpsea_peptide_surface_baseline.yaml").read_text()
    assert "enable_surface: false" in baseline
    assert "/dataset/unified: cpsea_peptide\n" in baseline or "/dataset/unified: cpsea_peptide\r\n" in baseline
    assert "cpsea_peptide_surface" not in [
        line.split(":")[-1].strip()
        for line in baseline.splitlines()
        if "dataset/unified" in line
    ]

    shuffle = (REPO / "configs/example/training_cpsea_peptide_surface_shuffle.yaml").read_text()
    assert "cpsea_peptide_surface_shuffle" in shuffle
    assert "enable_surface: true" in shuffle


@pytest.mark.skipif(not SAMPLE_SURFACES.is_dir(), reason="sample100 surface caches missing")
def test_attach_loads_real_sample100_cache():
    from proteinfoundation.datasets.transforms import AttachPeptideSurfaceTransform, Data

    caches = sorted(SAMPLE_SURFACES.glob("*.surface.npz"))
    assert caches
    example_id = caches[0].name[: -len(".surface.npz")]
    g = Data(id=example_id)
    g = AttachPeptideSurfaceTransform(str(SAMPLE_SURFACES), num_points=96)(g)
    assert g.surface_xyz.shape == (96, 3)
    assert g.surface_mask.dtype == torch.bool
    assert int(g.surface_mask.sum()) == 96


def test_surface_agreement_metrics_self_is_near_zero():
    from proteinfoundation.eval.surface_metrics import surface_agreement_metrics

    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(40, 3))
    normals = normalize_normals(rng.normal(size=(40, 3)))
    m = surface_agreement_metrics(xyz, xyz, oracle_normals=normals, pred_normals=normals)
    assert m["surface_chamfer"] == pytest.approx(0.0, abs=1e-6)
    assert m["surface_coverage_mean"] == pytest.approx(0.0, abs=1e-6)
    assert m["surface_normal_consistency"] == pytest.approx(1.0, abs=1e-5)
