#!/usr/bin/env python3
"""Unit tests for the uniform W&B validation metric schema.

Pure-dict tests: no GPU, no dataset, no model required. Covers
`proteinfoundation.logging.metric_schema`.

Usage (as a standalone script):
    python script_utils/test_metric_schema.py

Usage (via pytest, if installed):
    pytest script_utils/test_metric_schema.py -v
"""

from __future__ import annotations

from proteinfoundation.logging.metric_schema import (
    ALL_VALIDATION_METRIC_KEYS,
    RECON_PREFIXES,
    finalize_metrics,
    init_metric_dict,
)


def test_all_keys_unique():
    assert len(ALL_VALIDATION_METRIC_KEYS) == len(set(ALL_VALIDATION_METRIC_KEYS))


def test_init_metric_dict_covers_every_key_as_nan():
    d = init_metric_dict()
    assert set(d.keys()) == set(ALL_VALIDATION_METRIC_KEYS)
    assert all(v != v for v in d.values()), "every initial value should be NaN"


def test_finalize_metrics_empty_dict_fills_every_key_with_nan():
    out = finalize_metrics({}, warn_missing=False)
    assert set(out.keys()) == set(ALL_VALIDATION_METRIC_KEYS)
    assert all(v != v for v in out.values())


def test_finalize_metrics_preserves_provided_values():
    metrics = init_metric_dict()
    metrics["val/loss/latent"] = 1.23
    metrics["val/ae_recon/seq_acc"] = 0.87
    out = finalize_metrics(metrics, warn_missing=False)
    assert out["val/loss/latent"] == 1.23
    assert out["val/ae_recon/seq_acc"] == 0.87
    # Everything else that was never populated stays NaN, not dropped.
    assert out["val/loss/bb_ca"] != out["val/loss/bb_ca"]


def test_finalize_metrics_run_without_four_way_decode_still_has_those_keys_as_nan():
    # Simulates a run with `four_way_decode_eval_enabled=false`: the caller only populates
    # the ae_recon / loss / cyclization subset, never touching the four `decode_*` prefixes.
    metrics = init_metric_dict()
    metrics["val/ae_recon/seq_acc"] = 0.5
    metrics["val/loss/latent"] = 2.0
    metrics["val/loss/bb_ca"] = 0.1
    metrics["val/cyclization/acc"] = 0.9
    metrics["val/cyclization/ce"] = 0.2

    out = finalize_metrics(metrics, warn_missing=False)

    four_way_prefixes = [p for p in RECON_PREFIXES if p != "ae_recon"]
    assert len(four_way_prefixes) == 4
    for key in ALL_VALIDATION_METRIC_KEYS:
        if any(key.startswith(f"val/{p}/") for p in four_way_prefixes):
            assert out[key] != out[key], f"{key} should still be present as NaN"
        else:
            assert key in out


def test_finalize_metrics_extra_keys_are_preserved():
    metrics = init_metric_dict()
    metrics["some/run_specific_metric"] = 42.0
    out = finalize_metrics(metrics, warn_missing=False)
    assert out["some/run_specific_metric"] == 42.0


ALL_TESTS = [
    test_all_keys_unique,
    test_init_metric_dict_covers_every_key_as_nan,
    test_finalize_metrics_empty_dict_fills_every_key_with_nan,
    test_finalize_metrics_preserves_provided_values,
    test_finalize_metrics_run_without_four_way_decode_still_has_those_keys_as_nan,
    test_finalize_metrics_extra_keys_are_preserved,
]


if __name__ == "__main__":
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
