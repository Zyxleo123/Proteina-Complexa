#!/usr/bin/env python3
"""Measure RFD3's hotspot conditioning on OUR LP data, before spending GPU time on it.

WHY THIS RUNS FIRST. Enabling hotspots on the LP arm is a config change, so there is nothing
to unit-test -- but there is a great deal to MEASURE, because the stock dosage was chosen for
RFD3's own PPI data (whole-protein binders against whole-protein targets) and our binder is a
5-16 residue peptide. Two specific ways it can be quietly wrong here:

ZERO-HOTSPOT EXAMPLES. `AddPPIHotspotFeature` (ppi_transforms.py:279-302) provides

    n_provided = np.random.randint(0, ceil(n_gt * max_hotspots_frac_to_provide))

which is uniform on ``[0, ceil(n_gt*frac) - 1]``, EXCLUSIVE of the top. So the requested
fraction is never actually reached, and -- the part that matters -- when
``ceil(n_gt*frac) == 1`` the draw is ``randint(0, 1)``, which is ALWAYS 0. Such an example
gets no hotspots at all while ``add_ppi_hotspots`` was true, and nothing logs it: the
`"No hotspots found in the input data"` warning fires only when ``n_gt`` itself rounds to
zero, which is a different case. At frac 0.2 the silent corner is every example with
``n_gt <= 5``.

This is not hypothetical for this project. The same family of bug was measured on the CPSea
side of the house, where the analogous construction (``int(n_contacts * uniform(0, 0.2))``
followed by a 0.5 dropout) left **65-78% of examples carrying zero hotspots** -- a channel
that is wired, passes every assertion, and is almost never populated. The arithmetic here is
different, so the rate has to be measured rather than carried over.

THE CHANNEL MUST REACH THE NETWORK. `is_atom_level_hotspot` is a 1-dim atom-level input
feature (model/components/rfd3_net.yaml:43), gathered into ``data["feats"]`` by
`AddAdditional1dFeaturesToFeats` -- which is constructed with
``autofill_zeros_if_not_present_in_atomarray=True`` (pipelines.py:521). That autofill is why
hotspots-off is not a crash, and it is also why hotspots-on could silently be a no-op: if the
annotation never lands on the atom array, the feature is manufactured as zeros and the
example looks perfectly normal. So the probe reads the FEATURE TENSOR, not just the
annotation.

WHAT IS CHECKED (each one reported PASS/FAIL, and each one a way to waste a tranche):
  1. provided hotspots are a SUBSET of the ground-truth interface  -- else we condition on noise
  2. provided hotspots are on the RECEPTOR only, never the peptide -- else we leak the answer
  3. the feats tensor agrees with the annotation, atom for atom -- else the channel is dead
  4. hotspots survive the spatial crop                            -- see note below
  5. the conditioning cutoff really is the one we configured      -- recomputed independently

ON THE CROP. `AddPPIHotspotFeature` sits AFTER `PadTokensWithVirtualAtoms`, which the pipeline
comments call the "Last transform which modulates atom count", so hotspots are computed on the
already-cropped array and cannot be cropped away. Check 4 verifies that ordering holds rather
than trusting the comment, because the reverse order would compute hotspots on a 1,339-residue
receptor and then discard most of them -- which would read as a low provided-count, not as a
bug.

TWO PASSES, because one number cannot answer both questions:
  `production`  the config exactly as the arm will train. Gives the REALISED rate -- what
                fraction of examples the model actually sees a hotspot request on, including
                the condition mixture and `add_ppi_hotspots: 0.75`.
  `forced`      `add_ppi_hotspots: 1.0` and the ppi condition at frequency 1.0 with the others
                at 0. Gives the CONDITIONAL distribution of hotspot counts with usable
                statistics, which is what the dosage decision needs.
Reporting only the first would hide the dosage; reporting only the second would overstate how
often the channel is live.

Runs on CPU in the foundry arm's env. No GPU, no model -- the dataset and its transform stack
only.

Run:
    ./scripts/submit_lpgen.sh --submit hsprobe
    ./scripts/submit_lpgen.sh --submit hsprobe --set HSPROBE_N=400
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config-dir", required=True,
                   help="models/rfd3/configs in the foundry tree (hydra's primary config dir).")
    p.add_argument("--searchpath", required=True,
                   help="Comma-separated hydra.searchpath entries, as the train job passes them.")
    p.add_argument("--experiment", default="lp_finetune_hotspot",
                   help="Experiment to compose. The point is to probe the REAL arm config, so "
                        "this should be the experiment the training job will run.")
    p.add_argument("--n", type=int, default=200,
                   help="Examples per pass. 200 gives a +-3%% band on a rate near 0.5, which "
                        "is enough to decide a dosage.")
    p.add_argument("--passes", default="production,forced")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


# --------------------------------------------------------------------------- pure helpers
# Kept free of hydra/atomworks so they are unit-testable in the project venv, which does not
# have the arm's environment. See script_utils/test_lpgen_hotspots.py.

def degenerate_draw_ceiling(n_gt: int, frac: float) -> int:
    """``ceil(n_gt * frac)``, the exclusive upper bound upstream draws against.

    A return of 1 means ``np.random.randint(0, 1)`` and therefore ALWAYS zero hotspots
    provided; 0 means upstream takes its early-return branch and warns.
    """
    return int(math.ceil(n_gt * frac))


def expected_provided(n_gt: int, frac: float) -> float:
    """Mean hotspots provided under upstream's exclusive-bound draw.

    Uniform on ``[0, c-1]`` has mean ``(c-1)/2``, NOT ``c/2`` and not ``n_gt*frac``. Stated
    explicitly because the config key is named ``max_..._frac_to_provide``, which invites
    reading the realised mean as ``n_gt*frac/2`` and the realised max as ``n_gt*frac``.
    """
    c = degenerate_draw_ceiling(n_gt, frac)
    return 0.0 if c <= 1 else (c - 1) / 2.0


def summarize(values: list) -> dict:
    """Count/mean/median/min/max plus the quantiles worth seeing, or an empty dict."""
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
        "frac_zero": float((a == 0).mean()),
    }


# --------------------------------------------------------------------------- the probe
def build_cfg(args, pass_name: str):
    """Compose the arm's config, with the `forced` pass's overrides applied."""
    from hydra import compose, initialize_config_dir

    overrides = [
        f"experiment={args.experiment}",
        "hydra.searchpath=[" + ",".join(args.searchpath.split(",")) + "]",
        # The TRAIN pipeline ships `return_atom_array: False`
        # (datasets/train/pdb/base_transform_args.yaml:10), and the final `SubsetToKeys`
        # then drops `atom_array`, `specification` AND `conditions` -- so without this the
        # probe can see neither the hotspot annotation nor whether the example was even
        # asked for hotspots.
        #
        # This is safe as a probe-only change, and the reason is worth stating: the flag is
        # read at pipelines.py:568 purely to EXTEND `keys_to_keep`. It adds no transform,
        # removes none, and changes no sampling, so every example is built exactly as
        # training builds it; only more of the result survives to be inspected. Upstream's
        # own comment on the `conditions` branch is "For debugging & tests".
        "++datasets.train.lpgen.dataset.transform.return_atom_array=True",
    ]
    if pass_name == "forced":
        # Make every example a ppi example that requests hotspots, so the conditional
        # distribution is measured on n examples rather than on ~60% of them.
        overrides += [
            "++datasets.global_transform_args.meta_conditioning_probabilities.add_ppi_hotspots=1.0",
            "++datasets.global_transform_args.train_conditions.ppi.frequency=1.0",
            "++datasets.global_transform_args.train_conditions.unconditional.frequency=0.0",
            "++datasets.global_transform_args.train_conditions.island.frequency=0.0",
            "++datasets.global_transform_args.train_conditions.sequence_design.frequency=0.0",
            "++datasets.global_transform_args.train_conditions.tipatom.frequency=0.0",
        ]

    with initialize_config_dir(config_dir=str(args.config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
    return cfg


def recompute_gt_hotspots(atom_array, is_binder: np.ndarray, cutoff: float) -> np.ndarray:
    """Independent recomputation of the interchain contact set, by brute force.

    Deliberately NOT a call to ``get_hotspot_atoms``: checking upstream's output against
    upstream's own function would only prove it is deterministic. This is O(n_pep * n_rec)
    pairwise distances, which is trivial for a peptide, and it is what makes check 5
    ("the cutoff is the one we configured") mean anything.

    Takes the binder MASK rather than a pn_unit id so it cannot disagree with the mask the
    rest of the probe uses.
    """
    is_binder = np.asarray(is_binder).astype(bool)
    coord = np.asarray(atom_array.coord, dtype=float)
    finite = np.isfinite(coord).all(axis=-1)

    out = np.zeros(len(is_binder), dtype=bool)
    bidx = np.where(is_binder & finite)[0]
    ridx = np.where((~is_binder) & finite)[0]
    if bidx.size == 0 or ridx.size == 0:
        return out
    d = np.linalg.norm(coord[ridx][:, None, :] - coord[bidx][None, :, :], axis=-1)
    out[ridx] = d.min(axis=1) <= cutoff
    return out


def probe_pass(cfg, pass_name: str, n: int, seed: int, frac: float, cutoff: float) -> tuple:
    """Iterate ``n`` real examples, returning (rows, failures)."""
    import hydra

    ds = hydra.utils.instantiate(cfg.datasets.train.lpgen.dataset)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))[:n]

    rows, failures = [], []
    for i in order:
        try:
            ex = ds[int(i)]
        except Exception as e:  # noqa: BLE001 - a failing example is data, not a crash
            failures.append({"index": int(i), "error": f"{type(e).__name__}: {e}"})
            continue

        aa = ex["atom_array"]
        annots = set(aa.get_annotation_categories())
        cond = str(ex.get("sampled_condition_name", "?"))
        # THE ACTUAL GATE IS A CONJUNCTION, and reading only the meta-probability
        # overstates the request rate. `ConditionalRoute` fires on
        #     data["sampled_condition_name"] == "ppi" AND data["conditions"]["add_ppi_hotspots"]
        # (pipelines.py:457-459), while `add_ppi_hotspots` is sampled for EVERY example
        # regardless of which condition was drawn. Measured on the first run: 26 rows had the
        # meta-condition set while the transform never ran, and all 26 were non-ppi
        # conditions. Both are reported, because the gap between them is the dosage that a
        # change to the condition mixture would move.
        meta = bool(ex.get("conditions", {}).get("add_ppi_hotspots", False))
        asked = bool(meta and cond == "ppi")

        # `binder_pn_unit` is a `data` key, and the pipeline's final SubsetToKeys does not
        # keep it -- so read the binder off the ATOM ARRAY instead, via the annotation
        # `LPBinderCondition.is_valid_for_example` writes alongside it
        # (proteinfoundation/lpgen/rfd3_conditions.py:220). That is also the more honest
        # source: it is the per-atom mask the rest of the pipeline acts on, rather than a
        # scalar we would be re-deriving.
        pn = np.asarray(aa.pn_unit_iid, dtype=str)
        if "is_binder_pn_unit" in annots:
            is_binder = np.asarray(aa.get_annotation("is_binder_pn_unit")).astype(bool)
            uniq = sorted(set(pn[is_binder].tolist()))
            binder = uniq[0] if len(uniq) == 1 else ""
        else:
            binder = str(ex.get("binder_pn_unit", ""))
            is_binder = pn == binder if binder else np.zeros(len(pn), dtype=bool)

        provided = (
            np.asarray(aa.get_annotation("is_atom_level_hotspot")).astype(bool)
            if "is_atom_level_hotspot" in annots
            else np.zeros(len(pn), dtype=bool)
        )
        gt_up = (
            np.asarray(aa.get_annotation("is_hotspot_gt")).astype(bool)
            if "is_hotspot_gt" in annots
            else np.zeros(len(pn), dtype=bool)
        )
        gt_mine = recompute_gt_hotspots(aa, is_binder, cutoff) if is_binder.any() else gt_up

        # The feature tensor, which is what the network is actually handed. A mismatch here
        # means the annotation never reached `feats` and autofill manufactured zeros.
        feats = ex.get("feats", {})
        ft = feats.get("is_atom_level_hotspot", None)
        if ft is None:
            feat_sum, feat_agrees, feat_present = -1.0, False, False
        else:
            arr = np.asarray(ft).reshape(-1)
            feat_present = True
            feat_sum = float(arr.sum())
            feat_agrees = bool(
                arr.size == provided.size
                and np.array_equal(arr.astype(bool), provided)
            )

        c = degenerate_draw_ceiling(int(gt_up.sum()), frac)
        rows.append({
            "pass": pass_name,
            "index": int(i),
            "example_id": str(ex.get("example_id", "") or ex.get("extra_info", {}).get("example_id", "")),
            "condition": cond,
            "asked_for_hotspots": asked,
            "meta_add_ppi_hotspots": meta,
            "n_atoms": int(len(pn)),
            "n_binder_atoms": int(is_binder.sum()),
            "n_gt_upstream": int(gt_up.sum()),
            "n_gt_recomputed": int(gt_mine.sum()),
            "n_provided": int(provided.sum()),
            "draw_ceiling": c,
            "degenerate_always_zero": bool(c == 1),
            "no_gt_at_all": bool(c == 0),
            # check 1: conditioning on real interface atoms, not noise
            "provided_subset_of_gt": bool(provided.sum() == 0 or (provided & ~gt_up).sum() == 0),
            # check 2: never leak the answer by marking the peptide
            "n_provided_on_binder": int((provided & is_binder).sum()),
            # check 3: the channel reaches the network
            "feat_present": feat_present,
            "feat_sum": feat_sum,
            "feat_agrees_with_annotation": feat_agrees,
            # check 5: the cutoff is the configured one
            "gt_matches_recompute": bool(int(gt_up.sum()) == int(gt_mine.sum())),
            "annot_present": "is_atom_level_hotspot" in annots,
        })
    return rows, failures


def report(rows: list, failures: list, frac: float, cutoff: float) -> dict:
    """Build the summary and print the tables. Returns the summary dict."""
    out: dict = {
        "config": {"max_ppi_hotspots_frac_to_provide": frac, "ppi_hotspot_max_distance": cutoff},
        "n_rows": len(rows),
        "n_failures": len(failures),
        "passes": {},
        "checks": {},
    }

    for pass_name in sorted({r["pass"] for r in rows}):
        pr = [r for r in rows if r["pass"] == pass_name]
        asked = [r for r in pr if r["asked_for_hotspots"]]
        live = [r for r in asked if r["n_provided"] > 0]
        out["passes"][pass_name] = {
            "n_examples": len(pr),
            "condition_mix": dict(Counter(r["condition"] for r in pr)),
            # The realised rate: of every example the model sees, how many carry a request
            # that actually populated the channel. This is the number the dosage decision
            # turns on, and it is NOT `add_ppi_hotspots`.
            "frac_asked": len(asked) / len(pr) if pr else 0.0,
            # The meta-probability on its own, and the gap to `frac_asked`, which is entirely
            # examples whose sampled condition was not `ppi`.
            "frac_meta_set": sum(r["meta_add_ppi_hotspots"] for r in pr) / len(pr) if pr else 0.0,
            "n_meta_set_but_not_ppi": sum(
                1 for r in pr if r["meta_add_ppi_hotspots"] and r["condition"] != "ppi"
            ),
            "n_gt_rows_checked_against_recompute": sum(1 for r in pr if r["annot_present"]),
            "frac_live_of_all": len(live) / len(pr) if pr else 0.0,
            "frac_live_of_asked": len(live) / len(asked) if asked else 0.0,
            "frac_zero_despite_asking": (
                1.0 - len(live) / len(asked) if asked else 0.0
            ),
            "frac_asked_but_degenerate": (
                sum(r["degenerate_always_zero"] for r in asked) / len(asked) if asked else 0.0
            ),
            "frac_asked_but_no_gt": (
                sum(r["no_gt_at_all"] for r in asked) / len(asked) if asked else 0.0
            ),
            "n_gt": summarize([r["n_gt_upstream"] for r in asked]),
            "n_provided_given_asked": summarize([r["n_provided"] for r in asked]),
            "n_provided_given_live": summarize([r["n_provided"] for r in live]),
            "expected_provided_from_formula": summarize(
                [expected_provided(r["n_gt_upstream"], frac) for r in asked]
            ),
        }

    checks = {
        "1_provided_subset_of_gt": all(r["provided_subset_of_gt"] for r in rows),
        "2_no_hotspot_on_binder": all(r["n_provided_on_binder"] == 0 for r in rows),
        "3_feats_agree_with_annotation": all(
            r["feat_agrees_with_annotation"] for r in rows if r["n_provided"] > 0
        ),
        "3b_feats_channel_present": all(r["feat_present"] for r in rows),
        # ONLY where the transform ran. `is_hotspot_gt` is absent on every example the
        # ConditionalRoute skipped, and the probe defaults it to zero -- so comparing that
        # default against a real brute-force recompute reported 79 false failures on the
        # first run, every one of them an example that was never conditioned at all.
        "5_gt_matches_independent_recompute": all(
            r["gt_matches_recompute"] for r in rows if r["annot_present"]
        ),
    }
    # check 4: hotspots are computed post-crop, so a cropped example still carries them.
    cropped = [r for r in rows if r["asked_for_hotspots"] and r["n_atoms"] < 3840]
    live_cropped = [r for r in cropped if r["n_provided"] > 0]
    checks["4_hotspots_survive_crop"] = bool(not cropped or live_cropped)
    out["checks"] = checks
    out["crop_evidence"] = {
        "n_asked_and_cropped": len(cropped),
        "n_of_those_with_hotspots": len(live_cropped),
    }

    print("=" * 78)
    print(f"HOTSPOT PROBE  frac={frac}  cutoff={cutoff} A  rows={len(rows)}  failures={len(failures)}")
    print("=" * 78)
    for pass_name, p in out["passes"].items():
        print(f"\n--- pass: {pass_name}  ({p['n_examples']} examples) ---")
        print(f"  condition mix            : {p['condition_mix']}")
        print(f"  meta add_ppi_hotspots    : {p['frac_meta_set']:.1%}"
              f"   ({p['n_meta_set_but_not_ppi']} of those were NOT ppi, so no transform ran)")
        print(f"  ASKED (ppi AND meta)     : {p['frac_asked']:.1%}")
        print(f"  CHANNEL LIVE (all)       : {p['frac_live_of_all']:.1%}")
        print(f"  CHANNEL LIVE (of asked)  : {p['frac_live_of_asked']:.1%}")
        print(f"  zero DESPITE asking      : {p['frac_zero_despite_asking']:.1%}"
              f"   (of which degenerate ceiling==1: {p['frac_asked_but_degenerate']:.1%},"
              f" no gt at all: {p['frac_asked_but_no_gt']:.1%})")
        g, pv = p["n_gt"], p["n_provided_given_asked"]
        if g.get("n"):
            print(f"  ground-truth hotspot atoms: median {g['median']:.0f}  "
                  f"[{g['min']:.0f}, {g['max']:.0f}]  p05 {g['p05']:.0f}  p95 {g['p95']:.0f}")
        if pv.get("n"):
            print(f"  provided | asked          : median {pv['median']:.0f}  "
                  f"mean {pv['mean']:.1f}  max {pv['max']:.0f}  zero {pv['frac_zero']:.1%}")

    print("\n--- invariants ---")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  crop evidence: {out['crop_evidence']}")

    if failures:
        print(f"\n--- {len(failures)} example(s) failed to load ---")
        for f in failures[:5]:
            print(f"    {f['index']}: {f['error']}")
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, all_fail = [], []
    frac = cutoff = None
    for pass_name in [p.strip() for p in args.passes.split(",") if p.strip()]:
        cfg = build_cfg(args, pass_name)
        gta = cfg.datasets.global_transform_args
        frac = float(gta.max_ppi_hotspots_frac_to_provide)
        cutoff = float(gta.ppi_hotspot_max_distance)
        if frac <= 0.0:
            print(f"GATE: max_ppi_hotspots_frac_to_provide is {frac} in experiment "
                  f"'{args.experiment}' -- hotspots are OFF, so there is nothing to probe. "
                  f"Use --experiment lp_finetune_hotspot.", file=sys.stderr)
            return 1
        print(f"[probe] pass={pass_name} frac={frac} cutoff={cutoff} "
              f"add_ppi_hotspots="
              f"{gta.meta_conditioning_probabilities.add_ppi_hotspots}")
        rows, fails = probe_pass(cfg, pass_name, args.n, args.seed, frac, cutoff)
        all_rows += rows
        all_fail += fails

    with (out_dir / "hotspot_probe_rows.jsonl").open("w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")

    summary = report(all_rows, all_fail, frac, cutoff)
    (out_dir / "hotspot_probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nrows    -> {out_dir / 'hotspot_probe_rows.jsonl'}")
    print(f"summary -> {out_dir / 'hotspot_probe_summary.json'}")

    if not all_rows:
        print("FATAL: no examples were probed", file=sys.stderr)
        return 1
    # A failed invariant means the arm would train on wrong conditioning, so fail the job.
    # A low live-rate is NOT a failure here -- it is the measurement this job exists to
    # produce, and the dosage decision belongs to whoever reads it.
    return 0 if all(summary["checks"].values()) else 2


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    raise SystemExit(main())
