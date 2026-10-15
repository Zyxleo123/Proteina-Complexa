"""Tests for the RFD3 LP sampling weights.

The thing worth testing is that the weights actually *realise* the intended mixture when
drawn through the same sampler foundry uses. A per-row weight column can look plausible and
still integrate to the wrong share -- the LP pool is 92% ProtFrag by row count against a
24% PepBench target, so row counts and sampled shares differ by nearly 4x and only the
draw settles it.

The zero-weight guard is the other half: the manifest zeroes `sample_weight` outside the
train split, so a zero reaching the sampler means the dataset's filter and the weight
column disagree about which split is being trained on. Such a row is simply never sampled,
which is indistinguishable from it not being in the dataset -- silent, and exactly the kind
of mismatch that would otherwise be found by noticing the loss curve is odd.

Run:  .venv/bin/python script_utils/test_lpgen_rfd3_sampling.py
      .venv/bin/python -m pytest -q script_utils/test_lpgen_rfd3_sampling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.lpgen.rfd3_sampling import weights_from_column  # noqa: E402


def make_frame(n_pepbench=250, n_protfrag=3000, pep_frac=0.24):
    """A frame shaped like the real manifest: ProtFrag dominant by rows, weights by share."""
    rows = []
    for i in range(n_pepbench):
        rows.append({"example_id": f"pepbench__{i}", "dataset_source": "pepbench",
                     "sample_weight": pep_frac / n_pepbench})
    for i in range(n_protfrag):
        rows.append({"example_id": f"protfrag__{i}", "dataset_source": "protfrag",
                     "sample_weight": (1.0 - pep_frac) / n_protfrag})
    return pd.DataFrame(rows)


def test_weights_realise_the_configured_mixture_when_drawn():
    """Draw through WeightedRandomSampler and check the empirical mixture, not the column."""
    df = make_frame()
    # By ROWS the frame is 7.7% pepbench; by WEIGHT it should be 24%.
    row_share = (df.dataset_source == "pepbench").mean()
    assert row_share < 0.10, row_share

    w = weights_from_column(df)
    assert len(w) == len(df)

    g = torch.Generator().manual_seed(0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=w, num_samples=40_000, replacement=True, generator=g
    )
    drawn = df.dataset_source.to_numpy()[list(sampler)]
    pep = (drawn == "pepbench").mean()
    assert abs(pep - 0.24) < 0.02, f"realised pepbench share {pep:.3f}, wanted ~0.24"


def test_inverse_ratio_is_realised_too():
    df = make_frame(pep_frac=0.8)
    w = weights_from_column(df)
    g = torch.Generator().manual_seed(1)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=w, num_samples=40_000, replacement=True, generator=g
    )
    drawn = df.dataset_source.to_numpy()[list(sampler)]
    assert abs((drawn == "pepbench").mean() - 0.8) < 0.02


def test_zero_weight_rows_are_refused_by_default():
    """A zero-weight row is never sampled; that must not pass silently."""
    df = make_frame(n_pepbench=5, n_protfrag=5)
    df.loc[0, "sample_weight"] = 0.0
    try:
        weights_from_column(df)
    except ValueError as exc:
        assert "never" in str(exc) and "zero" in str(exc).lower()
    else:
        raise AssertionError("expected a ValueError for a zero-weight row")

    # Opt-in still works, for the case where it is genuinely intended.
    w = weights_from_column(df, allow_zero=True)
    assert float(w[0]) == 0.0


def test_bad_input_is_refused():
    df = make_frame(n_pepbench=3, n_protfrag=3)

    missing = df.drop(columns=["sample_weight"])
    try:
        weights_from_column(missing)
    except KeyError as exc:
        # The message must mention columns_to_load: that is the non-obvious reason the
        # column goes missing in the real config.
        assert "columns_to_load" in str(exc)
    else:
        raise AssertionError("expected KeyError for a missing column")

    neg = df.copy(); neg.loc[0, "sample_weight"] = -1.0
    try:
        weights_from_column(neg)
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("expected ValueError for a negative weight")

    nan = df.copy(); nan.loc[0, "sample_weight"] = float("nan")
    try:
        weights_from_column(nan)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("expected ValueError for a NaN weight")

    allzero = df.copy(); allzero["sample_weight"] = 0.0
    try:
        weights_from_column(allzero, allow_zero=True)
    except ValueError as exc:
        assert "sums to" in str(exc)
    else:
        raise AssertionError("expected ValueError when the weights sum to zero")


def test_real_manifest_weights_realise_24_76():
    """Against the actual manifest, if it is present on this machine."""
    man = Path("/zfsauton/scratch/yixiz/LPData/preprocessed/metadata_lpgen/lpgen_manifest.parquet")
    if not man.exists():
        print("    (skipped: manifest not on this machine)")
        return
    df = pd.read_parquet(man, columns=["split", "dataset_source", "sample_weight"])
    df = df[df.split == "train"].reset_index(drop=True)
    w = weights_from_column(df)
    g = torch.Generator().manual_seed(2)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=w, num_samples=50_000, replacement=True, generator=g
    )
    drawn = df.dataset_source.to_numpy()[list(sampler)]
    pep = (drawn == "pepbench").mean()
    rows = (df.dataset_source == "pepbench").mean()
    print(f"    manifest: pepbench {rows:.3%} of rows -> {pep:.3f} of draws")
    assert abs(pep - 0.24) < 0.02, pep


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
