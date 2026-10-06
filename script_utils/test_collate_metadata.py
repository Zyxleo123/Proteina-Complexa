"""`example_id` must survive collation, because provenance is what the evaluation runs on.

Every field in `structure_collate_fn` routes through `_pad_and_stack`, which cannot stack
strings and returns None -- and a None result DROPS the key. So `example_id` silently
disappeared from every batch, and `cp2lp_bb_evaluate.py` fell back to the positional label its
own `cp.get("example_id", [f"row{i}" ...])` provides. That fallback is unique only WITHIN a
batch and carries no link to the panel, which is why the source contact map (and therefore
`contact_retention`, and therefore two of the report's five gates) could not be supplied at
all. The fallback in that caller is evidence the key was meant to survive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _sample(n: int, example_id: str, key: str = "id"):
    from torch_geometric.data import Data

    # `id` by default, because that is the attribute the real pipeline sets:
    # `atomarray_to_atom37` builds `Data(..., id=sample_id, ...)` and
    # `StructureDataset.__getitem__` skips `id_column` when propagating metadata, so
    # `example_id` is never an attribute in simple mode.
    # num_nodes is explicit: torch_geometric cannot infer it from this field set, and the
    # collate's binder-length maths gets None without it.
    kw = {key: example_id}
    return Data(coors_nm=torch.zeros(n, 3), num_nodes=n, **kw)


@pytest.mark.parametrize("key", ["id", "example_id"])
def test_the_provenance_key_survives_collation_as_a_list(key):
    """BOTH names must survive. The real sample carries `id` (simple mode); pipeline mode
    uses `example_id`. Allowlisting only `example_id` left the eval still falling back to
    positional labels, because the attribute it needed was never called that."""
    from proteinfoundation.datasets.structure_data import structure_collate_fn

    out = structure_collate_fn([_sample(5, "SRC_A", key), _sample(7, "SRC_B", key)])
    assert out[key] == ["SRC_A", "SRC_B"], out.get(key)


def test_the_passthrough_does_not_disturb_padding():
    """It must be a passthrough, not a participant: the ids are per-sample, so they must not
    be padded to the binder length or counted as residues."""
    from proteinfoundation.datasets.structure_data import structure_collate_fn

    out = structure_collate_fn([_sample(5, "SRC_A"), _sample(7, "SRC_B")])
    assert out["nsamples"] == 2
    assert out["nres"] == 7
    assert tuple(out["coors_nm"].shape) == (2, 7, 3)
    assert len(out["id"]) == out["nsamples"]


def test_slice_batch_keeps_ids_aligned_with_the_examples_it_selects():
    """The eval slices the batch down to the cyclic examples before reading ids, so a
    passthrough that `slice_batch` did not understand would hand back the WRONG id per row."""
    from proteinfoundation.cp2lp.data import slice_batch
    from proteinfoundation.datasets.structure_data import structure_collate_fn

    out = structure_collate_fn(
        [_sample(5, "SRC_A"), _sample(6, "SRC_B"), _sample(7, "SRC_C")]
    )
    keep = torch.tensor([True, False, True])
    sliced = slice_batch(out, keep, 3)
    assert sliced["id"] == ["SRC_A", "SRC_C"], sliced.get("id")
    assert tuple(sliced["coors_nm"].shape)[0] == 2


def test_metadata_allowlist_is_narrow():
    """An allowlist, not a blanket non-tensor passthrough: the collate drops unstackable
    fields on purpose, and widening that is how unrelated junk reaches every batch."""
    from proteinfoundation.datasets.structure_data import _SAMPLE_LEVEL_METADATA_KEYS

    assert "id" in _SAMPLE_LEVEL_METADATA_KEYS, "the key the real pipeline actually sets"
    assert "example_id" in _SAMPLE_LEVEL_METADATA_KEYS
    assert len(_SAMPLE_LEVEL_METADATA_KEYS) <= 3, _SAMPLE_LEVEL_METADATA_KEYS
