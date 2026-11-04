#!/usr/bin/env python3
"""Checks for the collapsed trunk (README_CPSEQ_FOLD_PIPELINE_STANDARD sec. 2).

The lever folds a candidate once at `diffusion_samples = depth` instead of `depth` times
at one sample each, because the trunk is deterministic at inference and was being
recomputed identically every time. Everything here is a property whose failure would
produce a result that looks right and is not, so each is checked against the case that
would break it rather than a happy path:

- **The ~29k cached predictions keep their keys.** The new settings enter
  `Boltz2Config.to_dict()` only when set, so a default-valued run hashes exactly as every
  run before them did. Checked as an exact dict and an exact hash, because a single extra
  key would silently re-predict the whole corpus.
- **A collapsed fold and a legacy per-seed fold never satisfy each other.** This is the
  one that decides whether the lever is safe: without the fold slot in the journal key,
  the collapsed fold at `seeds[0]` keys identically to the legacy single fold at
  `seeds[0]`, resume calls it done, and the run writes ONE fold where the plan asked for
  `depth` -- a rung reported at full depth holding one draw of the noise.
- **The fold axis survives the collapse.** `single_fold_views` has to give each sample its
  own verdict and its own energy; an aggregate row would collapse a depth-10 rung to one
  fold and turn the ladder's reliability rate back into the coin flip it exists to
  replace.
- **The shared prediction cost is divided, not repeated.** Copying it onto every view
  would report the lever as costing exactly what it saves.
- **A depth the predictor will not deliver is refused before the GPU is touched.**

No GPU, no boltz, no PyRosetta: dict and string operations throughout.

Usage:
    python script_utils/test_cpseq_collapse.py
    pytest script_utils/test_cpseq_collapse.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_search import (  # noqa: E402
    STAGE_LADDER,
    Journal,
    collapsed_input_cost_s,
    ladder_plan,
    outcome_row,
    prefetch_batch_cap,
)
from proteinfoundation.cpseq.adapters.boltz2 import (  # noqa: E402
    Boltz2Config,
    _predict_cmd,
    _run_env,
    input_hash,
)
from proteinfoundation.cpseq.evaluate import Protocol, single_fold_views  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


# The settings dict every prediction before 2026-11-03 was keyed by. Written out in full
# rather than derived, so a change to the default config has to be made here too and
# cannot pass unnoticed.
LEGACY_KEYS = {
    "diffusion_samples", "recycling_steps", "sampling_steps",
    "use_potentials", "seed", "affinity",
}


def test_default_config_hashes_exactly_as_before() -> None:
    print("the cache key of a default run is untouched")
    base = Boltz2Config(diffusion_samples=1)
    check("to_dict carries the six legacy keys and nothing else",
          set(base.to_dict()) == LEGACY_KEYS,
          f"extra={set(base.to_dict()) - LEGACY_KEYS}")
    h0 = input_hash("yaml-text", base)
    # The shadow's PATH is a property of the filesystem, not of the prediction, so it
    # must not key. Two nodes with different scratch layouts have to share a cache.
    check("chunk_fix_root alone does not change the key",
          input_hash("yaml-text",
                     Boltz2Config(diffusion_samples=1, chunk_fix_root="/scratch/a")) == h0)
    check("an unset max_parallel_samples does not change the key",
          input_hash("yaml-text",
                     Boltz2Config(diffusion_samples=1, max_parallel_samples=None)) == h0)


def test_settings_that_change_numerics_do_change_the_key() -> None:
    """Sec. 3: a setting that moves the numerics and is absent from the key would mix two
    regimes under one hash, inside comparisons designed so that exactly one thing moves."""
    print("settings that move the numerics are in the key")
    h0 = input_hash("yaml-text", Boltz2Config(diffusion_samples=1))
    hp = input_hash("yaml-text",
                    Boltz2Config(diffusion_samples=1, chunk_fix=True,
                                 chunk_fix_root="/scratch/a"))
    hm = input_hash("yaml-text",
                    Boltz2Config(diffusion_samples=1, max_parallel_samples=15))
    check("chunk_fix keys apart from unpatched", hp != h0)
    check("max_parallel_samples keys apart", hm != h0)
    check("the two are not each other", hp != hm)
    # Different batch sizes are the same computation, so they must not be the same key.
    check("two batch sizes key apart",
          input_hash("y", Boltz2Config(max_parallel_samples=5))
          != input_hash("y", Boltz2Config(max_parallel_samples=30)))


def test_chunk_fix_without_a_root_refuses_to_run() -> None:
    """Running unpatched under a key that says patched is the exact failure sec. 3 is
    about, so it has to raise rather than fall back."""
    print("a patched key cannot be produced by unpatched code")
    try:
        _run_env(Boltz2Config(chunk_fix=True))
        check("chunk_fix with no root raises", False, "it returned instead")
    except ValueError:
        check("chunk_fix with no root raises", True)
    env = _run_env(Boltz2Config(chunk_fix=True, chunk_fix_root="/scratch/shadow"))
    check("the shadow goes FIRST on PYTHONPATH, ahead of the install",
          env["PYTHONPATH"].split(":")[0] == "/scratch/shadow")
    check("no shadow when chunk_fix is off",
          _run_env(Boltz2Config(chunk_fix=False)) is None)


def test_max_parallel_samples_reaches_the_command_line() -> None:
    print("the batch size is passed, not merely recorded")
    cfg = Boltz2Config(diffusion_samples=10, max_parallel_samples=15)
    cmd = _predict_cmd(target="in", out_dir="out", config=cfg, python_exec="py",
                       boltz_cache="bc", devices=1, accelerator="gpu")
    check("--max_parallel_samples is on the line", "--max_parallel_samples" in cmd)
    check("with the requested value",
          cmd[cmd.index("--max_parallel_samples") + 1] == "15")
    bare = _predict_cmd(target="in", out_dir="out", config=Boltz2Config(),
                        python_exec="py", boltz_cache="bc", devices=1, accelerator="gpu")
    check("and is absent when unset -- boltz's own default",
          "--max_parallel_samples" not in bare)


def test_collapsed_plan_is_one_entry_per_sequence_per_rung() -> None:
    print("the collapsed plan drops the seed axis and nothing else")
    seqs = ["REF"] + [f"D{i}" for i in range(8)]
    seeds = tuple(range(42, 52))          # depth 10
    budgets = (2, 4, 8)
    flat = ladder_plan(seqs, seeds, budgets)
    coll = ladder_plan(seqs, seeds, budgets, collapse_seeds=True)
    check("uncollapsed is depth times longer",
          len(flat) == len(coll) * len(seeds),
          f"{len(flat)} vs {len(coll)}x{len(seeds)}")
    check("every collapsed entry is at seeds[0]",
          all(sd == seeds[0] for _r, _k, _s, sd in coll))
    check("every collapsed entry is at rank 0",
          all(k == 0 for _r, k, _s, _sd in coll))
    # The ordering guarantee the ladder is built on: a rung is a PREFIX of the draws, and
    # the reference is paid for inside the first rung.
    first_rung = [s for r, _k, s, _sd in coll if r == budgets[0]]
    check("the reference is in the first rung", "REF" in first_rung)
    check("the first rung holds the reference plus the first B draws",
          first_rung == ["REF", "D0", "D1"], str(first_rung))
    # Each sequence appears exactly once overall: its depth now lives inside the one
    # prediction, so a second entry would be a second prediction of the same thing.
    seen = [s for _r, _k, s, _sd in coll]
    check("no sequence is planned twice", len(seen) == len(set(seen)))
    check("the rungs cover the same sequences as uncollapsed",
          set(seen) == {s for _r, _k, s, _sd in flat})


def _sample(idx: int, *, valid: bool, dg: float | None, viol: float | None) -> dict:
    return {
        "sample_index": idx,
        "valid": valid,
        "prerelax_valid": valid,
        "refined_dG_separated": dg,
        "violation": viol,
    }


def _rec(samples: list[dict]) -> dict:
    return {
        "sequence": "ACDEF",
        "prediction_seed": 42,
        "status": "scored",
        "samples": samples,
        "n_samples": len(samples),
        "prediction_runtime_s": 140.0,
        "wall_time_s": 600.0,
    }


def test_each_sample_becomes_its_own_fold() -> None:
    """The ladder counts `n_pass` out of `n_folds` and applies a k-of-n rule, so the row
    axis has to stay per-fold. One aggregate row would make depth 10 read as depth 1."""
    print("one sample, one fold row")
    samples = [
        _sample(0, valid=True, dg=-9.0, viol=0.1),
        _sample(1, valid=False, dg=-30.0, viol=0.9),   # attractive AND invalid
        _sample(2, valid=True, dg=-7.0, viol=0.2),
    ]
    views = single_fold_views(_rec(samples))
    check("one view per sample", len(views) == 3)
    check("each view holds exactly one sample",
          all(v["n_samples"] == 1 and len(v["samples"]) == 1 for v in views))
    check("fold_index is the sample index",
          [v["fold_index"] for v in views] == [0, 1, 2])
    check("verdicts are per sample, not the aggregate",
          [v["valid"] for v in views] == [True, False, True])
    # The screening objective is the best dG among VALID samples. An invalid structure
    # contributes no energy however attractive, so the invalid view must carry None --
    # the -30.0 must not leak into any row.
    check("an invalid sample carries no energy",
          [v["best_valid_dG_separated"] for v in views] == [-9.0, None, -7.0])
    check("the attractive invalid energy reaches no row",
          all(v["best_valid_dG_separated"] != -30.0 for v in views))
    check("violation is the sample's own",
          [v["min_violation"] for v in views] == [0.1, 0.9, 0.2])


def test_the_shared_prediction_cost_is_divided_not_repeated() -> None:
    print("the shared cost is attributed once")
    views = single_fold_views(_rec([
        _sample(i, valid=True, dg=-8.0, viol=0.1) for i in range(10)
    ]))
    total = sum(v["prediction_runtime_s"] for v in views)
    check("the per-fold shares sum back to the one invocation",
          abs(total - 140.0) < 1e-6, f"summed to {total}")
    check("the undivided number is kept beside the share",
          all(v["prediction_runtime_shared_s"] == 140.0 for v in views))
    check("each view says how many folds shared it",
          all(v["n_folds_sharing_prediction"] == 10 for v in views))


def test_a_crash_is_one_row_not_depth_rows() -> None:
    """`depth` copies of one technical failure would inflate every failure denominator by
    the collapse factor, and a crash is not evidence about `depth` folds."""
    print("a record with no samples is itself, once")
    crashed = {
        "sequence": "ACDEF", "prediction_seed": 42,
        "status": "prediction_failed", "technical_failure": True, "samples": [],
    }
    check("a crash yields one view", len(single_fold_views(crashed)) == 1)
    check("and is returned unchanged", single_fold_views(crashed)[0] is crashed)
    one = _rec([_sample(0, valid=True, dg=-8.0, viol=0.1)])
    check("a genuinely single-sample record is not rewritten",
          single_fold_views(one)[0] is one)


def _row(seq: str, seed: int, arm: str, *, fold_index=None, rank=0) -> dict:
    rec = {
        "sequence": seq, "prediction_seed": seed, "status": "scored",
        "samples": [_sample(fold_index or 0, valid=True, dg=-8.0, viol=0.1)],
        "n_samples": 1, "valid": True, "fold_index": fold_index,
    }
    return outcome_row(rec=rec, proposal=None, parent_id="P", arm=arm,
                       stage=STAGE_LADDER,
                       extra={"seed_rank": rank if fold_index is None else fold_index})


def test_collapsed_and_legacy_folds_never_satisfy_each_other() -> None:
    """The property the lever's safety rests on. Both directions, because either one
    silently under-folds: a legacy row read as covering `depth` collapsed folds writes one
    fold where the plan asked for ten, and a collapsed row read as covering the legacy
    fold leaves a seed unfolded while the report counts it."""
    print("the two regimes key apart in the journal")
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td), "P")
        # A legacy run: one fold per seed, no fold slot.
        for sd in (42, 43):
            j.add_outcome(_row("ACDEF", sd, "random", rank=0))
        check("a legacy fold satisfies the legacy check",
              j.has("ACDEF", 42, STAGE_LADDER, "random"))
        check("a legacy fold does NOT satisfy collapsed fold 0",
              not j.has("ACDEF", 42, STAGE_LADDER, "random", 0))
        check("nor any other collapsed fold",
              not any(j.has("ACDEF", 42, STAGE_LADDER, "random", k)
                      for k in range(10)))

    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td), "P")
        # A collapsed run: ten folds from one prediction at seeds[0].
        for k in range(10):
            j.add_outcome(_row("ACDEF", 42, "random", fold_index=k))
        check("every collapsed fold satisfies its own slot",
              all(j.has("ACDEF", 42, STAGE_LADDER, "random", k) for k in range(10)))
        check("collapsed folds do NOT satisfy the legacy check",
              not j.has("ACDEF", 42, STAGE_LADDER, "random"))


def test_a_half_written_collapsed_sequence_resumes() -> None:
    print("a task killed mid-sequence resumes on the missing folds only")
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td), "P")
        for k in range(4):                      # killed after 4 of 10
            j.add_outcome(_row("ACDEF", 42, "random", fold_index=k))
        done = [k for k in range(10) if j.has("ACDEF", 42, STAGE_LADDER, "random", k)]
        check("the four written folds are done", done == [0, 1, 2, 3])
        check("the sequence is NOT complete",
              not all(j.has("ACDEF", 42, STAGE_LADDER, "random", k) for k in range(10)))
        # Reopening the journal from disk must see the same thing: the resume decision is
        # made by a fresh process after a requeue, not by the one that wrote the rows.
        j2 = Journal(Path(td), "P")
        again = [k for k in range(10) if j2.has("ACDEF", 42, STAGE_LADDER, "random", k)]
        check("and a reopened journal agrees", again == [0, 1, 2, 3], str(again))


def test_seed_rank_is_the_fold_axis_the_report_reads() -> None:
    """`cpseq_ladder_report.py` keys folds by `seed_rank`, counts `n_folds(depth)` out of
    it and splits it in half. Collapsed, the sample index has to take that slot or the
    report sees one fold per draw and reports a depth-10 rung as depth 1."""
    print("the report's rank axis is filled by the sample index")
    rows = [_row("ACDEF", 42, "random", fold_index=k) for k in range(10)]
    ranks = sorted(r["seed_rank"] for r in rows)
    check("ten distinct ranks from one prediction", ranks == list(range(10)))
    check("fold_index is recorded alongside, for the journal key",
          sorted(r["fold_index"] for r in rows) == list(range(10)))
    legacy = _row("ACDEF", 42, "random", rank=7)
    check("a legacy row still carries its seed rank", legacy["seed_rank"] == 7)
    check("and no fold_index", legacy["fold_index"] is None)


def test_a_depth_the_predictor_will_not_deliver_is_refused() -> None:
    """Checked by `run_parent_ladder` before any GPU time. The failure it prevents is a
    rung reported at a depth it never reached, which no later reader could detect."""
    print("a mismatched depth is refused up front")
    import cpseq_search as cs

    seeds = tuple(range(42, 52))
    kw = dict(
        parent_id="P", target=None, out_dir=Path(tempfile.mkdtemp()),
        store=None, n_draws=1, seeds=seeds, budgets=(1,), collapse_seeds=True,
    )
    # ds=4 against depth 10: the prediction would return 4 structures for a plan entry
    # that owes 10 folds.
    try:
        cs.run_parent_ladder(protocol=Protocol(diffusion_samples=4), **kw)
        check("ds != depth raises", False, "it ran")
    except ValueError as e:
        check("ds != depth raises", "diffusion_samples" in str(e))
    # ds=10 is the right depth but needs the chunk fix, or the batch size is whatever
    # `30 % max_parallel_samples + 1` happens to be.
    try:
        cs.run_parent_ladder(protocol=Protocol(diffusion_samples=10), **kw)
        check("a deep collapse without chunk_fix raises", False, "it ran")
    except ValueError as e:
        check("a deep collapse without chunk_fix raises", "chunk_fix" in str(e))


def test_the_prefetch_batch_is_capped_to_fit_the_timeout() -> None:
    """The trap the collapse introduces. Each input costs `K + depth*s` instead of
    `K + s`, so the 65-input group that always fitted the 7200 s timeout at ds=1 is
    ~9000 s at depth 10 -- and a timeout returns NOTHING, losing the batch and leaving
    every one of its folds to be predicted singly. That throws away the entire saving."""
    print("a collapsed invocation is sized to the timeout")
    # Pinned to the measured per-input marginal cost, not to a fitted intercept: job
    # 95978's A7 folded 9 inputs at ds=10 in 850.5 s, i.e. (850.5 - F)/9 = 91.9 s each.
    check("the cost model reproduces the measured per-input cost",
          abs(collapsed_input_cost_s(10) - 91.9) < 0.1,
          f"{collapsed_input_cost_s(10)} against the measured 91.9")
    check("and is marginal -- it excludes the per-invocation F",
          collapsed_input_cost_s(0) < 20.0,
          f"ds=0 costs {collapsed_input_cost_s(0)}, which looks like it includes F")
    cap = prefetch_batch_cap(10, 7200)
    check("depth 10 on a 7200 s timeout caps the batch", cap == 38, f"cap={cap}")
    # The property, not the number: a full batch must be planned to finish well inside
    # the timeout.
    for depth in (1, 2, 4, 10, 20, 40):
        c = prefetch_batch_cap(depth, 7200)
        planned = 23.2 + c * collapsed_input_cost_s(depth)
        check(f"depth={depth:<2} cap={c:<4} plans {planned:.0f}s of a 7200s timeout",
              planned <= 7200 * 0.5 + collapsed_input_cost_s(depth),
              f"planned {planned:.0f}s")
    check("a deeper collapse gets a smaller cap",
          prefetch_batch_cap(40, 7200) < prefetch_batch_cap(10, 7200)
          < prefetch_batch_cap(1, 7200))
    check("a longer timeout allows a bigger batch",
          prefetch_batch_cap(10, 14400) > prefetch_batch_cap(10, 7200))
    # A cap of 0 would read as "no batching" at the call site, and an input too big for
    # the timeout still has to be attempted rather than silently skipped.
    check("the cap is never 0, however tight the timeout",
          prefetch_batch_cap(40, 1) == 1)


def main() -> int:
    test_default_config_hashes_exactly_as_before()
    test_settings_that_change_numerics_do_change_the_key()
    test_chunk_fix_without_a_root_refuses_to_run()
    test_max_parallel_samples_reaches_the_command_line()
    test_collapsed_plan_is_one_entry_per_sequence_per_rung()
    test_each_sample_becomes_its_own_fold()
    test_the_shared_prediction_cost_is_divided_not_repeated()
    test_a_crash_is_one_row_not_depth_rows()
    test_collapsed_and_legacy_folds_never_satisfy_each_other()
    test_a_half_written_collapsed_sequence_resumes()
    test_seed_rank_is_the_fold_axis_the_report_reads()
    test_a_depth_the_predictor_will_not_deliver_is_refused()
    test_the_prefetch_batch_is_capped_to_fit_the_timeout()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
