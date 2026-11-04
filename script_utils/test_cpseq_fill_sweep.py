#!/usr/bin/env python3
"""Checks for the filler sweep: fill dispatch, and shared masks across arms.

These two are what make a temperature sweep interpretable, and both fail SILENTLY rather
than loudly if they are wrong. A bad dispatch mislabels a whole arm in the journal, and
the label is permanent. Masks that differ between arms turn "which filler is better" into
"which arm got easier positions", which no later analysis can unpick.

No GPU, no folds on disk, no ProteinMPNN.

Run:  python script_utils/test_cpseq_fill_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_mask_mpnn_fill import (  # noqa: E402
    ARM_LLM_MASK_MPNN,
    ARM_LLM_MASK_RAND,
    ARM_RAND_MASK_MPNN,
    matched_random_masks,
    resolve_fill,
    sampled_mask_sizes,
)
from proteinfoundation.cpseq.search import budget_for  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


def test_dispatch_by_name_is_unchanged() -> None:
    """The default must reproduce the pilot exactly, or the sweep's arms are not
    comparable with the pilot's arms that share their directory."""
    print("by-name dispatch (the pilot's behaviour)")
    check("llm_mask_random_fill fills randomly",
          resolve_fill("by-name", ARM_LLM_MASK_RAND) == "random")
    for arm in (ARM_LLM_MASK_MPNN, ARM_RAND_MASK_MPNN):
        check(f"{arm} fills with mpnn", resolve_fill("by-name", arm) == "mpnn")
    # A custom label must not accidentally inherit the random filler by matching a
    # substring of the one arm whose name says `random`.
    check("a custom label containing 'random' still fills with mpnn",
          resolve_fill("by-name", "random_mask_mpnn_fill_t08") == "mpnn")


def test_dispatch_override_wins() -> None:
    """The override is the whole point: a sweep arm names its setting and still chooses
    its filler."""
    print("explicit dispatch (what a sweep arm needs)")
    check("--fill random overrides an mpnn-named arm",
          resolve_fill("random", ARM_LLM_MASK_MPNN) == "random")
    check("--fill mpnn overrides the random-named arm",
          resolve_fill("mpnn", ARM_LLM_MASK_RAND) == "mpnn")
    check("--fill mpnn on a custom sweep label",
          resolve_fill("mpnn", "mpnn_fill_t10") == "mpnn")
    check("--fill random on a custom sweep label",
          resolve_fill("random", "shared_mask_random_fill") == "random")
    try:
        resolve_fill("argmax", ARM_LLM_MASK_MPNN)
    except ValueError:
        check("an unknown fill mode raises rather than defaulting", True)
    else:
        check("an unknown fill mode raises rather than defaulting", False,
              "silently fell through to a filler")


def test_shared_masks_are_deterministic() -> None:
    """Same seed, same masks -- so a requeued or re-run arm fills what the others filled,
    and a resumed sweep stays paired."""
    print("shared masks are reproducible")
    L, n = 12, 12
    budget = budget_for(L)
    a_sizes = sampled_mask_sizes(n, budget, 4242)
    b_sizes = sampled_mask_sizes(n, budget, 4242)
    check("mask sizes are a pure function of the seed", a_sizes == b_sizes,
          f"{a_sizes} != {b_sizes}")
    a = matched_random_masks(a_sizes, L, seed=4242)
    b = matched_random_masks(b_sizes, L, seed=4242)
    check("masks are a pure function of the seed", a == b)
    c = matched_random_masks(sampled_mask_sizes(n, budget, 99), L, seed=99)
    check("a different seed gives different masks", a != c)


def test_shared_masks_are_legal() -> None:
    """A mask over budget would be rejected downstream and the arm would look like it
    proposed badly, rather than like it was fed an illegal mask."""
    print("shared masks respect the budget and the sequence")
    for L in (7, 9, 11, 12, 15, 16):
        budget = budget_for(L)
        sizes = sampled_mask_sizes(12, budget, seed=L * 7 + 1)
        masks = matched_random_masks(sizes, L, seed=L * 7 + 1)
        ok_budget = all(1 <= len(m) <= budget for m in masks)
        ok_range = all(all(0 <= p < L for p in m) for m in masks)
        ok_uniq = all(len(set(m)) == len(m) for m in masks)
        check(f"L={L:<3} budget={budget}: {len(masks)} masks in range, unique, in budget",
              ok_budget and ok_range and ok_uniq,
              f"budget={ok_budget} range={ok_range} uniq={ok_uniq}")


def main() -> int:
    test_dispatch_by_name_is_unchanged()
    test_dispatch_override_wins()
    test_shared_masks_are_deterministic()
    test_shared_masks_are_legal()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
