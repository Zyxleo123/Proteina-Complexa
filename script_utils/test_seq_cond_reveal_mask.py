#!/usr/bin/env python3
"""Checks for `choose_reveal_mask`: which positions a seq-conditioned sample reveals.

No model and no GPU. What is tested is the thing that decides WHICH residues the
generator is shown, because a bug there does not crash -- it reveals the wrong positions
and every metric downstream still looks reasonable.

Run:  python script_utils/test_seq_cond_reveal_mask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from proteinfoundation.utils.seq_conditioning import (  # noqa: E402
    choose_reveal_mask,
    resolve_sequence_conditioning_mask,
)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


def test_keep_frac_path() -> None:
    print("\nkeep_frac (the measurement path)")
    gen_mask = torch.ones(4, 10, dtype=torch.bool)
    gen_mask[2, 7:] = False  # a shorter peptide, padded

    m = choose_reveal_mask(None, 1.0, gen_mask)
    check("keep_frac >= 1.0 reveals every real residue", bool((m == gen_mask).all()))
    check("padding is never revealed", not bool(m[2, 7:].any()))

    torch.manual_seed(0)
    m = choose_reveal_mask(None, 0.5, gen_mask)
    check("keep_frac < 1.0 hides some positions", int(m.sum()) < int(gen_mask.sum()))
    check("and still reveals some", int(m.sum()) > 0)
    check("a keep_frac draw never reveals padding", not bool(m[2, 7:].any()))


def test_explicit_wins() -> None:
    print("\nan explicit mask overrides keep_frac")
    gen_mask = torch.ones(2, 6, dtype=torch.bool)
    explicit = torch.zeros(2, 6, dtype=torch.bool)
    explicit[:, [0, 3]] = True

    # keep_frac is deliberately 1.0 here: if it were consulted at all, everything would
    # be revealed and the test would fail loudly rather than subtly.
    m = choose_reveal_mask(explicit, 1.0, gen_mask)
    check("exactly the named positions are revealed",
          bool((m == explicit).all()), f"{m}")
    check("the hidden positions really are hidden",
          not bool(m[:, [1, 2, 4, 5]].any()))

    # The proposer's invariant: hidden == what the unmasking step must fill.
    free = gen_mask & ~m
    check("the free set is the complement within the real residues",
          int(free.sum()) == 2 * 4, f"{int(free.sum())}")


def test_explicit_is_masked_by_padding() -> None:
    print("\npadding still wins over an explicit reveal")
    gen_mask = torch.ones(2, 5, dtype=torch.bool)
    gen_mask[1, 3:] = False
    explicit = torch.ones(2, 5, dtype=torch.bool)
    m = choose_reveal_mask(explicit, 0.5, gen_mask)
    check("a mask that names padding does not reveal it",
          not bool(m[1, 3:].any()), f"{m}")
    check("real positions of that row are still revealed", bool(m[1, :3].all()))


def test_refusals() -> None:
    print("\nwhat it refuses")
    gen_mask = torch.ones(4, 10, dtype=torch.bool)

    try:
        choose_reveal_mask(torch.ones(2, 10, dtype=torch.bool), 1.0, gen_mask, n_repeat=2)
        check("a wrong-shaped mask is refused", False, "no error raised")
    except ValueError as e:
        check("a wrong-shaped mask is refused, not broadcast", True)
        check("the error says what shape was expected and why",
              "(4, 10)" in str(e) and "n_repeat=2" in str(e), str(e))

    try:
        choose_reveal_mask(torch.zeros(4, 10, dtype=torch.bool), 1.0, gen_mask)
        check("an all-hidden mask is refused", False, "no error raised")
    except ValueError as e:
        check("an all-hidden mask is refused as the unconditional pass",
              "seq_cond=False" in str(e), str(e))


def test_agrees_with_the_feature_resolver() -> None:
    """The mask this returns is handed to the feature as `use_residue_type_feature`.

    If `resolve_sequence_conditioning_mask` reinterpreted it, the positions revealed to
    the trunk would differ from the positions the recovery metric calls conditioned, and
    `cond_recovery` would be measured on the wrong set.
    """
    print("\nthe feature resolver round-trips it unchanged")
    gen_mask = torch.ones(3, 8, dtype=torch.bool)
    explicit = torch.zeros(3, 8, dtype=torch.bool)
    explicit[0, [1, 2]] = True
    explicit[1, [0]] = True
    explicit[2, [7]] = True

    chosen = choose_reveal_mask(explicit, 1.0, gen_mask)
    resolved = resolve_sequence_conditioning_mask(
        chosen, b=3, n=8, device=chosen.device, pad_mask=gen_mask
    )
    check("resolver returns the identical mask",
          resolved is not None and bool((resolved == chosen).all()), f"{resolved}")


def main() -> int:
    print("seq-conditioned reveal-mask checks")
    test_keep_frac_path()
    test_explicit_wins()
    test_explicit_is_masked_by_padding()
    test_refusals()
    test_agrees_with_the_feature_resolver()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
