#!/usr/bin/env python3
"""Unit tests for sequence-conditioned generation ("give the sequence, generate the rest").

Pure-tensor tests: no GPU, no dataset, no checkpoint. Covers
`proteinfoundation.utils.seq_conditioning`,
`proteinfoundation.nn.feature_factory.seq_feats.OptionalResidueTypeSeqFeat`,
`proteinfoundation.eval.sequence_recovery_metrics.sequence_recovery_metrics`, and the
ordering contract in `proteinfoundation.utils.training_handlers`.

Usage:
    python script_utils/test_sequence_conditioning.py
    pytest script_utils/test_sequence_conditioning.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from proteinfoundation.eval.sequence_recovery_metrics import sequence_recovery_metrics
from proteinfoundation.nn.feature_factory.seq_feats import OptionalResidueTypeSeqFeat
from proteinfoundation.utils.seq_conditioning import (
    resolve_sequence_conditioning_mask,
    sample_sequence_conditioning_mask,
    sequence_conditioning_fraction,
)
from proteinfoundation.utils.training_handlers import handle_folding_n_inverse_folding

DEV = torch.device("cpu")


def _batch(b=2, n=5, pad_last=1):
    """A minimal batch the seq-mode feature factory can consume."""
    mask = torch.ones(b, n, dtype=torch.bool)
    if pad_last:
        mask[:, -pad_last:] = False
    return {
        "coords_nm": torch.zeros(b, n, 37, 3),
        "residue_type": torch.arange(b * n).reshape(b, n) % 20,
        "mask": mask,
    }


# --------------------------------------------------------------------------------------
# resolve_sequence_conditioning_mask
# --------------------------------------------------------------------------------------


def test_resolve_all_four_flag_forms():
    b, n = 3, 4
    assert resolve_sequence_conditioning_mask(False, b, n, DEV) is None
    assert resolve_sequence_conditioning_mask(None, b, n, DEV) is None

    full = resolve_sequence_conditioning_mask(True, b, n, DEV)
    assert full.shape == (b, n) and bool(full.all())

    per_ex = torch.tensor([True, False, True])
    m = resolve_sequence_conditioning_mask(per_ex, b, n, DEV)
    assert m.shape == (b, n)
    assert bool(m[0].all()) and not bool(m[1].any()) and bool(m[2].all())

    per_res = torch.zeros(b, n, dtype=torch.bool)
    per_res[1, 2] = True
    m = resolve_sequence_conditioning_mask(per_res, b, n, DEV)
    assert m.shape == (b, n) and int(m.sum()) == 1 and bool(m[1, 2])


def test_resolve_returns_none_when_nothing_revealed():
    """All-False is the unconditional case; callers must be able to short-circuit on it."""
    b, n = 2, 3
    assert resolve_sequence_conditioning_mask(torch.zeros(b, n, dtype=torch.bool), b, n, DEV) is None


def test_resolve_intersects_padding():
    b, n = 2, 4
    pad = torch.ones(b, n, dtype=torch.bool)
    pad[:, -1] = False
    m = resolve_sequence_conditioning_mask(True, b, n, DEV, pad_mask=pad)
    assert not bool(m[:, -1].any()), "padding must never count as conditioned"
    assert bool(m[:, :-1].all())


def test_resolve_rejects_wrong_shapes():
    for bad in (torch.ones(7, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool)):
        try:
            resolve_sequence_conditioning_mask(bad, 3, 4, DEV)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for shape {tuple(bad.shape)}")


def test_fraction_is_scalar():
    assert sequence_conditioning_fraction(False, 2, 4, DEV) == 0.0
    assert sequence_conditioning_fraction(True, 2, 4, DEV) == 1.0
    m = torch.zeros(2, 4, dtype=torch.bool)
    m[0, :2] = True
    assert abs(sequence_conditioning_fraction(m, 2, 4, DEV) - 0.25) < 1e-6


# --------------------------------------------------------------------------------------
# sample_sequence_conditioning_mask
# --------------------------------------------------------------------------------------


def test_sampler_respects_padding_and_rates():
    torch.manual_seed(0)
    b, n = 512, 10
    pad = torch.ones(b, n, dtype=torch.bool)
    pad[:, -2:] = False
    m = sample_sequence_conditioning_mask(pad, p=0.5, keep_frac_min=1.0, keep_frac_max=1.0, p_full=0.0)
    assert not bool(m[:, -2:].any()), "padding revealed"
    # keep_frac pinned to 1 => each conditioned example is fully revealed over real residues.
    per_ex = m[:, :-2]
    revealed_rows = per_ex.any(dim=-1)
    assert bool(per_ex[revealed_rows].all()), "keep_frac=1 must reveal every real residue"
    assert abs(revealed_rows.float().mean().item() - 0.5) < 0.06, "per-example rate p is off"


def test_sampler_p_zero_and_p_one():
    pad = torch.ones(64, 6, dtype=torch.bool)
    assert not bool(sample_sequence_conditioning_mask(pad, 0.0, 0.5, 1.0, 0.5).any())
    m = sample_sequence_conditioning_mask(pad, 1.0, 1.0, 1.0, 1.0)
    assert bool(m.all())


def test_sampler_p_full_produces_complete_rows():
    """The exact query a user types -- the WHOLE sequence -- must be a first-class case."""
    torch.manual_seed(1)
    pad = torch.ones(512, 8, dtype=torch.bool)
    m = sample_sequence_conditioning_mask(pad, p=1.0, keep_frac_min=0.1, keep_frac_max=0.4, p_full=0.5)
    complete = m.all(dim=-1).float().mean().item()
    assert abs(complete - 0.5) < 0.06, f"p_full not honoured (got {complete:.3f} complete rows)"


def test_sampler_validates_params():
    pad = torch.ones(2, 3, dtype=torch.bool)
    for kwargs in (
        dict(p=1.5, keep_frac_min=0.0, keep_frac_max=1.0, p_full=0.0),
        dict(p=0.5, keep_frac_min=0.8, keep_frac_max=0.2, p_full=0.0),
        dict(p=0.5, keep_frac_min=0.0, keep_frac_max=1.0, p_full=2.0),
    ):
        try:
            sample_sequence_conditioning_mask(pad, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------------------
# OptionalResidueTypeSeqFeat
# --------------------------------------------------------------------------------------


def test_feature_dim_is_unchanged():
    """Dim must stay 20 so a seq-conditioned arm can warm-start from an existing ckpt."""
    assert OptionalResidueTypeSeqFeat().get_dim() == 20


def test_feature_off_is_zeros():
    feat = OptionalResidueTypeSeqFeat()
    out = feat(_batch())
    assert out.shape == (2, 5, 20)
    assert float(out.abs().max()) == 0.0


def test_feature_full_reveal_is_onehot_of_the_sequence():
    batch = _batch()
    batch["use_residue_type_feature"] = True
    out = OptionalResidueTypeSeqFeat()(batch)
    real = batch["mask"]
    assert torch.equal(out[real].argmax(dim=-1), batch["residue_type"][real])
    assert bool((out[real].sum(dim=-1) == 1).all()), "revealed positions must be one-hot"
    assert float(out[~real].abs().max()) == 0.0, "padding must stay zero"


def test_feature_partial_reveal_zeroes_unrevealed_positions():
    batch = _batch(pad_last=0)
    cond = torch.zeros(2, 5, dtype=torch.bool)
    cond[0, 1] = True
    cond[1, :3] = True
    batch["use_residue_type_feature"] = cond
    out = OptionalResidueTypeSeqFeat()(batch)
    assert bool((out.sum(dim=-1) > 0).eq(cond).all()), "reveal mask not respected per residue"
    assert int(out[0, 1].argmax()) == int(batch["residue_type"][0, 1])
    # An unrevealed position is the all-zero vector -- that is what makes it distinguishable
    # from a revealed one without spending a 21st channel.
    assert float(out[0, 0].abs().max()) == 0.0


def test_feature_per_example_flag():
    batch = _batch(pad_last=0)
    batch["use_residue_type_feature"] = torch.tensor([True, False])
    out = OptionalResidueTypeSeqFeat()(batch)
    assert float(out[0].sum()) == 5.0
    assert float(out[1].abs().max()) == 0.0


# --------------------------------------------------------------------------------------
# handle_folding_n_inverse_folding
# --------------------------------------------------------------------------------------


def test_handler_legacy_path_untouched():
    """With no `sequence_conditioning` block, behaviour is exactly as before (bool flags)."""
    batch = _batch()
    cfg = SimpleNamespace(p_folding_n_inv_folding_iters=0.0)
    out = handle_folding_n_inverse_folding(dict(batch), cfg)
    assert out["use_residue_type_feature"] is False
    assert out["use_ca_coors_nm_feature"] is False


def test_handler_sequence_conditioning_yields_per_residue_mask():
    torch.manual_seed(0)
    batch = _batch(b=64, n=6, pad_last=1)
    cfg = {
        "p_folding_n_inv_folding_iters": 0.15,
        "sequence_conditioning": {
            "enabled": True,
            "p": 1.0,
            "keep_frac_min": 1.0,
            "keep_frac_max": 1.0,
            "p_full": 1.0,
        },
    }
    out = handle_folding_n_inverse_folding(dict(batch), cfg)
    m = out["use_residue_type_feature"]
    assert torch.is_tensor(m) and m.dtype == torch.bool and m.shape == (64, 6)
    assert torch.equal(m, batch["mask"]), "p=1, keep=1 must reveal exactly the real residues"
    # The inverse-folding (Ca-given) draw is disabled on this path, not left to chance:
    # revealing the sequence AND the Ca trace at once trains neither task.
    assert out["use_ca_coors_nm_feature"] is False


def test_handler_disabled_block_falls_back_to_legacy():
    batch = _batch()
    cfg = {"p_folding_n_inv_folding_iters": 0.0, "sequence_conditioning": {"enabled": False}}
    out = handle_folding_n_inverse_folding(dict(batch), cfg)
    assert out["use_residue_type_feature"] is False


def test_sequence_conditioning_decided_before_self_cond():
    """Ordering contract: the reveal must be fixed before the self-cond inner forward.

    If it is not, the inner pass that produces `x_sc` runs unconditioned while the outer
    pass runs conditioned -- a train/sample mismatch, since at sampling time every step's
    `x_sc` is produced with the conditioning present.
    """
    import inspect

    from proteinfoundation.utils import training_handlers

    src = inspect.getsource(training_handlers.handle_batch_conditioning)
    i_fold = src.index("handle_folding_n_inverse_folding(batch")
    i_sc = src.index("handle_self_cond(batch")
    assert i_fold < i_sc, "handle_folding_n_inverse_folding must run BEFORE handle_self_cond"


# --------------------------------------------------------------------------------------
# sequence_recovery_metrics
# --------------------------------------------------------------------------------------


def test_recovery_splits_conditioned_from_free():
    true = torch.tensor([[1, 2, 3, 4]])
    pred = torch.tensor([[1, 2, 9, 9]])  # conditioned half right, free half wrong
    mask = torch.ones(1, 4, dtype=torch.bool)
    cond = torch.tensor([[True, True, False, False]])
    m = sequence_recovery_metrics(pred, true, mask, cond, prefix="p")
    assert m["p/cond_recovery"] == 1.0
    assert m["p/free_recovery"] == 0.0
    assert m["p/all_recovery"] == 0.5
    assert m["p/n_cond_res"] == 2.0 and m["p/n_free_res"] == 2.0
    assert m["p/cond_frac"] == 0.5
    assert m["p/cond_exact_frac"] == 1.0


def test_recovery_ignores_padding():
    true = torch.tensor([[1, 2, 3]])
    pred = torch.tensor([[1, 2, 7]])
    mask = torch.tensor([[True, True, False]])  # last is padding
    cond = torch.ones(1, 3, dtype=torch.bool)  # deliberately over-broad
    m = sequence_recovery_metrics(pred, true, mask, cond, prefix="p")
    assert m["p/cond_recovery"] == 1.0, "padding leaked into the conditioned population"
    assert m["p/n_cond_res"] == 2.0


def test_recovery_empty_population_is_nan_not_zero():
    true = torch.tensor([[1, 2]])
    pred = torch.tensor([[1, 2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    cond = torch.ones(1, 2, dtype=torch.bool)  # nothing free
    m = sequence_recovery_metrics(pred, true, mask, cond, prefix="p")
    assert math.isnan(m["p/free_recovery"]), "0.0 would read as 'nothing matched'"
    assert m["p/n_free_res"] == 0.0


def test_recovery_exact_frac_is_per_peptide_not_per_residue():
    true = torch.tensor([[1, 2, 3], [4, 5, 6]])
    pred = torch.tensor([[1, 2, 3], [4, 5, 9]])  # one peptide perfect, one off by a residue
    mask = torch.ones(2, 3, dtype=torch.bool)
    cond = torch.ones(2, 3, dtype=torch.bool)
    m = sequence_recovery_metrics(pred, true, mask, cond, prefix="p")
    assert abs(m["p/cond_recovery"] - 5 / 6) < 1e-6
    assert m["p/cond_exact_frac"] == 0.5, "per-peptide pass/fail must not be the position mean"


# --------------------------------------------------------------------------------------
# PeptideSequenceFeatures (inference side)
# --------------------------------------------------------------------------------------


def test_peptide_feature_full_sequence():
    from openfold.np.residue_constants import restype_order

    from proteinfoundation.datasets.gen_dataset import PeptideSequenceFeatures

    f = PeptideSequenceFeatures(["ACDEF", "GHIKLM"])
    nres = []
    f.setup(nres)
    assert nres == [5, 6], "lengths must come from the sequences"
    r = f({}, 0)
    assert torch.equal(r["residue_type"], torch.tensor([restype_order[c] for c in "ACDEF"]))
    assert bool(r["use_residue_type_feature"].all())
    assert r["peptide_sequence"] == "ACDEF"


def test_peptide_feature_x_leaves_positions_free():
    from proteinfoundation.datasets.gen_dataset import PeptideSequenceFeatures

    f = PeptideSequenceFeatures(["AC--FGH--K"])
    f.setup([])
    r = f({}, 0)
    expected = torch.tensor([True, True, False, False, True, True, True, False, False, True])
    assert torch.equal(r["use_residue_type_feature"], expected), "'-' must leave the position free"


def test_peptide_feature_rejects_length_mismatch():
    """Silently padding/truncating would generate a peptide the user never asked for."""
    from proteinfoundation.datasets.gen_dataset import PeptideSequenceFeatures

    f = PeptideSequenceFeatures(["ACDEF"])
    try:
        f.setup([9])
    except ValueError:
        return
    raise AssertionError("expected ValueError on nres/sequence length mismatch")


def test_peptide_feature_fasta_with_per_sequence_type(tmp_path=None):
    import tempfile

    from proteinfoundation.cyclization.constants import NAME_TO_CYCLIZATION_TYPE
    from proteinfoundation.datasets.gen_dataset import PeptideSequenceFeatures

    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as fh:
        fh.write(">p1 cyclization_type=mainchain\nACDEF\n>p2\nGHIK\n")
        path = fh.name
    f = PeptideSequenceFeatures(path, cyclization_type="isopeptide")
    nres = []
    f.setup(nres)
    assert nres == [5, 4]
    assert int(f({}, 0)["cyclization_type_cond"]) == NAME_TO_CYCLIZATION_TYPE["mainchain"]
    assert int(f({}, 1)["cyclization_type_cond"]) == NAME_TO_CYCLIZATION_TYPE["isopeptide"], (
        "a sequence without its own header type must fall back to the default"
    )


def test_peptide_feature_collates_into_a_batchable_mask():
    """The per-residue flag must survive collation as a [b, n] bool tensor."""
    from proteinfoundation.datasets.gen_dataset import PeptideSequenceFeatures, collate_fn

    f = PeptideSequenceFeatures(["ACDEF", "GHIK"])
    nres = []
    f.setup(nres)
    samples = [
        f({"nres": n, "mask": torch.ones(n, dtype=torch.bool)}, i) for i, n in enumerate(nres)
    ]
    b = collate_fn(samples)
    flag = b["use_residue_type_feature"]
    assert flag.shape == (2, 5) and flag.dtype == torch.bool
    assert bool(flag[0].all())
    assert bool(flag[1, :4].all()) and not bool(flag[1, 4]), "padding must collate as un-revealed"
    # And the resolver accepts it verbatim, intersected with the batch mask.
    m = resolve_sequence_conditioning_mask(flag, 2, 5, DEV, pad_mask=b["mask"])
    assert int(m.sum()) == 9


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
