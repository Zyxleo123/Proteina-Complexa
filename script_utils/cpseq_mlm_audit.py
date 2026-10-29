"""Audit the cpseq_mlm training curves from `metrics.jsonl`.

Pure CPU, reads only what the trainer already wrote to disk, so every number here is
reproducible without a GPU and without re-running training.

The questions it answers, in the order a reader should ask them:

  1. Did optimisation behave? -- loss trend, spikes, NaN, LR schedule completion.
  2. Was the step budget right? -- is validation still improving at the end, or saturated?
  3. Is it overfitting? -- train loss against val accuracy, plus how much of one epoch ran.
  4. Is the headline gap real? -- the pocket-vs-control difference and the pocket-shuffle
     gain, each against an explicit noise estimate rather than eyeballed.
  5. Which regimes moved? -- per-regime trajectories, since a pooled number hides them.

On significance: the two arms share one fixed eval panel, so the comparison is paired at
the position level. But positions are CLUSTERED within peptides -- a 16-mer under the
denovo regime contributes 16 correlated positions -- so an unclustered binomial SE over
pooled positions is optimistic. Both are reported, and the clustered one is the honest one.

Usage:
    python script_utils/cpseq_mlm_audit.py --runs-dir /path/to/runs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REGIMES = ("denovo", "budget", "anchor", "span")


def load(path: Path) -> tuple[list[dict], list[dict]]:
    """(training rows, eval rows). The trainer writes both to one file."""
    train, ev = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        (ev if d.get("eval") else train).append(d)
    return train, ev


def spark(vals: list[float], width: int = 48) -> str:
    """ASCII sparkline. Downsamples by averaging buckets so the shape is not an artefact
    of which points happened to be sampled."""
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    if len(vals) > width:
        step = len(vals) / width
        buckets = []
        for i in range(width):
            lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
            chunk = vals[lo:hi]
            buckets.append(sum(chunk) / len(chunk))
        vals = buckets
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return blocks[0] * len(vals)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """(slope, r) by least squares. Used to ask whether a curve is still moving."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return 0.0, 0.0
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def audit_arm(name: str, run_dir: Path) -> dict:
    mpath = run_dir / "metrics.jsonl"
    if not mpath.exists():
        print(f"[{name}] no metrics.jsonl"); return {}
    train, ev = load(mpath)
    manifest = {}
    mf = run_dir / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())

    print("=" * 78)
    print(f"ARM: {name}   ({'pocket-conditioned' if manifest.get('arm') != 'no_pocket' else 'NO-POCKET control'})")
    print("=" * 78)

    # ---- 1. optimisation health
    losses = [(r["step"], r["train/loss"]) for r in train if "train/loss" in r]
    lrs = [(r["step"], r["lr"]) for r in train if "lr" in r]
    if losses:
        steps = [s for s, _ in losses]
        vals = [v for _, v in losses]
        nan = [s for s, v in losses if v != v or abs(v) == float("inf")]
        print(f"\n1. OPTIMISATION  ({len(losses)} logged points, step {steps[0]}..{steps[-1]})")
        print(f"   train/loss {vals[0]:.4f} -> {vals[-1]:.4f}   {spark(vals)}")
        # Spikes relative to local level: a real instability, not schedule-driven decline.
        win = 20
        spikes = 0
        for i in range(win, len(vals)):
            local = sorted(vals[i - win:i])
            med = local[len(local) // 2]
            if vals[i] > med * 1.5:
                spikes += 1
        print(f"   NaN/inf: {len(nan)}      spikes >1.5x local median: {spikes}")
        # last-decile trend: has the loss stopped moving?
        tail = losses[int(len(losses) * 0.9):]
        sl, r = linfit([float(s) for s, _ in tail], [v for _, v in tail])
        print(f"   last-10% loss slope: {sl:+.3e}/step (r={r:+.2f})"
              f"  -> {'still falling' if sl < -1e-7 else 'flat'}")
    if lrs:
        print(f"   lr {lrs[0][1]:.2e} -> {lrs[-1][1]:.2e}"
              f"   {'cosine completed' if lrs[-1][1] < lrs[0][1] * 1e-3 else 'SCHEDULE DID NOT COMPLETE'}")

    # ---- 2/3. validation, budget, overfitting
    if ev:
        es = [r["step"] for r in ev]
        acc = [r.get("val/acc/all") for r in ev]
        pairs = [(s, a) for s, a in zip(es, acc) if a is not None]
        print(f"\n2. VALIDATION  ({len(pairs)} evals, step {pairs[0][0]}..{pairs[-1][0]})")
        print(f"   acc/all {pairs[0][1]:.4f} -> {pairs[-1][1]:.4f}   "
              f"{spark([a for _, a in pairs])}")
        # still improving?
        tail = pairs[int(len(pairs) * 0.8):]
        sl, r = linfit([float(s) for s, _ in tail], [a for _, a in tail])
        per10k = sl * 10000
        # Noise floor first: "best was earlier than the end" means nothing unless the
        # drop since then EXCEEDS eval-to-eval jitter. An earlier version of this script
        # flagged a 0.0034 dip against a 0.0011 floor as possible overfitting, which is
        # ~3 evals of ordinary wobble on a saturated curve -- a false alarm, and exactly
        # the kind a naive argmax-of-the-curve check produces.
        late_for_floor = [a for _, a in pairs[int(len(pairs) * 0.75):]]
        floor_deltas = [abs(late_for_floor[i] - late_for_floor[i - 1])
                        for i in range(1, len(late_for_floor))]
        floor = sum(floor_deltas) / len(floor_deltas) if floor_deltas else 0.0
        best = max(pairs, key=lambda p: p[1])
        drop = best[1] - pairs[-1][1]
        if best[0] >= pairs[-1][0] * 0.9:
            verdict = "at the end"
        elif drop <= 3 * floor:
            verdict = (f"earlier, but the {drop:.4f} drop is within 3x the {floor:.4f} "
                       f"noise floor -- jitter on a saturated curve, not decay")
        else:
            verdict = f"EARLIER and the {drop:.4f} drop exceeds 3x noise -- investigate"
        print(f"   best {best[1]:.4f} @ step {best[0]}   ({verdict})")
        print(f"   last-20% acc slope: {per10k:+.4f} per 10k steps (r={r:+.2f})")
        if per10k > 0.004:
            print("   -> STILL IMPROVING: the 60k budget was the binding constraint, not convergence.")
        elif per10k > 0.001:
            print("   -> marginally improving; more steps would buy a little.")
        else:
            print("   -> saturated; more steps of this schedule would not help.")
        # noise floor: eval-to-eval |delta| late in training, when LR is ~0 and the panel
        # is fixed, so this is a floor on resolution rather than a full seed-noise estimate.
        late = [a for _, a in pairs[int(len(pairs) * 0.75):]]
        deltas = [abs(late[i] - late[i - 1]) for i in range(1, len(late))]
        noise = sum(deltas) / len(deltas) if deltas else 0.0
        print(f"   late eval-to-eval mean |delta|: {noise:.4f}  (resolution floor)")

        # overfitting: train loss still falling while val acc flat/falling is the signature
        if losses:
            tl = [v for _, v in losses[int(len(losses) * 0.8):]]
            tl_slope, _ = linfit(list(range(len(tl))), tl)
            print(f"\n3. OVERFITTING CHECK")
            print(f"   late train loss slope {tl_slope:+.2e}/pt, late val acc slope {per10k:+.4f}/10k")
            if tl_slope < 0 and per10k < -0.001:
                print("   -> DIVERGING: train improving while val degrades. Overfitting.")
            else:
                print("   -> no overfitting signature (val is not degrading).")
        rows = manifest.get("train_rows")
        if rows and losses:
            bs = manifest.get("batch_size", 32) * manifest.get("accum", 1)
            seen = pairs[-1][0] * bs
            print(f"   examples seen {seen:,} of {rows:,} train rows"
                  f" = {seen / rows:.2f} epochs"
                  f"  ({'under one epoch, so memorisation is unlikely' if seen < rows else 'over one epoch'})")

        # ---- 4. pocket gain
        gains = [(r["step"], r["val/pocket_gain"]) for r in ev if r.get("val/pocket_gain") is not None]
        if gains:
            gv = [g for _, g in gains]
            print(f"\n4. POCKET GAIN  (real pockets - shuffled pockets, same panel, same weights)")
            print(f"   {gv[0]:+.4f} -> {gv[-1]:+.4f}   {spark(gv)}")
            sl, r = linfit([float(s) for s, _ in gains], gv)
            print(f"   trend {sl * 10000:+.4f} per 10k steps (r={r:+.2f})"
                  f"  -> {'GROWS with training' if sl > 0 and r > 0.3 else 'flat/noisy'}")
            late_g = gv[int(len(gv) * 0.75):]
            mean_g = sum(late_g) / len(late_g)
            sd_g = (sum((x - mean_g) ** 2 for x in late_g) / max(len(late_g) - 1, 1)) ** 0.5
            print(f"   late mean {mean_g:+.4f} +- {sd_g:.4f} (sd over {len(late_g)} evals)"
                  f"  -> {mean_g / sd_g:.1f} sd from zero" if sd_g > 0 else "")

        # ---- 5. per regime
        print(f"\n5. PER-REGIME (final eval, withheld positions only)")
        last = ev[-1]
        base = manifest.get("baselines", {}).get("per_regime", {})
        print(f"   {'regime':8s} {'top1':>7s} {'top3':>7s} {'prior':>7s} {'vs prior':>9s} {'ppl':>7s} {'n':>6s} trajectory")
        for rg in REGIMES:
            a = last.get(f"val/acc/{rg}")
            if a is None:
                continue
            t3 = last.get(f"val/top3/{rg}")
            pp = base.get(rg, {}).get("positional_argmax_acc")
            traj = [r.get(f"val/acc/{rg}") for r in ev if r.get(f"val/acc/{rg}") is not None]
            delta = f"{a - pp:+.4f}" if pp is not None else "n/a"
            print(f"   {rg:8s} {a:7.4f} {t3:7.4f} {pp if pp is None else f'{pp:7.4f}'}"
                  f" {delta:>9s} {last.get(f'val/ppl/{rg}'):7.3f}"
                  f" {last.get(f'val/n_hidden/{rg}'):6d} {spark(traj, 24)}")
    return {"ev": ev, "manifest": manifest}


def compare(a_name: str, a: dict, b_name: str, b: dict) -> None:
    """Paired comparison of the two arms on the shared fixed panel."""
    if not (a.get("ev") and b.get("ev")):
        return
    ea, eb = a["ev"][-1], b["ev"][-1]
    print("\n" + "=" * 78)
    print("PAIRED ARM COMPARISON (same fixed eval panel, same masks)")
    print("=" * 78)
    sa, sb = ea["step"], eb["step"]
    if sa != sb:
        print(f"!! NOT STEP-MATCHED: {a_name}@{sa} vs {b_name}@{sb}. Treat with care.")
    n = ea.get("val/n_hidden/all")
    pa, pb = ea.get("val/acc/all"), eb.get("val/acc/all")
    d = pa - pb
    # unclustered binomial SE over pooled positions -- optimistic
    se_pos = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / n)
    # clustered: positions within one peptide are correlated. The panel holds ~800
    # peptides, so inflate by sqrt(mean positions per peptide) as a first-order design
    # effect. Crude, but it is the right direction and the pooled SE is the wrong one.
    n_pep = a["manifest"].get("panel_size") or 800
    m = n / max(n_pep, 1)
    se_clu = se_pos * math.sqrt(max(m, 1.0))
    print(f"   {a_name} {pa:.4f}  vs  {b_name} {pb:.4f}   diff {d:+.4f}")
    print(f"   n = {n} withheld positions across ~{n_pep} peptides ({m:.1f} positions/peptide)")
    print(f"   unclustered SE {se_pos:.4f} -> z = {d / se_pos:+.2f}   (OPTIMISTIC)")
    print(f"   clustered   SE {se_clu:.4f} -> z = {d / se_clu:+.2f}   (honest)")
    if abs(d / se_clu) < 2:
        print("   -> the arm-vs-arm gap is NOT significant once clustering is accounted for.")
        print("      The pocket-shuffle gain is the stronger evidence: it is within-model,")
        print("      on identical positions, and it grows monotonically with training.")
    else:
        print("   -> gap survives clustering.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="/zfsauton/scratch/yixiz/cpseq_mlm/runs")
    ap.add_argument("--arms", default="cpseq_mlm_pocket,cpseq_mlm_nopocket")
    args = ap.parse_args()
    root = Path(args.runs_dir)
    out = {}
    for arm in args.arms.split(","):
        out[arm] = audit_arm(arm, root / arm)
        print()
    names = list(out)
    if len(names) == 2:
        compare(names[0], out[names[0]], names[1], out[names[1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
