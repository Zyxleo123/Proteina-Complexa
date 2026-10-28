"""Measure where a Boltz-2 fold's wall time actually goes, and test one way to cut it.

Two questions, one job, because they need the same GPU and the same input.

COST. Every fold today pays a trunk pass (template -> MSA -> pairformer, recycling+1
rounds) and then `diffusion_samples` draws from it. The trunk is deterministic at
inference -- `--subsample_msa` is an unpassed flag so the `torch.randperm` in
`trunkv2.py` never runs, and the compiled modules are reverted to `_orig_mod` when not
training -- so the ten seeds we fold each candidate at are recomputing a bit-identical
trunk ten times. If the trunk is a large share of the 26 s/structure, one invocation at
`diffusion_samples=40` replaces ten at 4 and the saving is nine trunk passes per
candidate. If the trunk is a small share, it does not, and we should not touch it. That
split has never been measured here, so everything below is a measurement, not a claim.

The arithmetic the configurations are designed to solve, with F the per-invocation fixed
cost (spawn, torch import, 2.2 GB checkpoint), K the trunk and s the marginal sample:

    T(n_inputs, ds) = F + n_inputs * (K + ds * s)

A1/A2/A3 vary `ds` at one input, which gives s and F+K. A6 adds inputs at fixed `ds`,
which separates F from K. A7 is the proposed configuration measured directly, against
10x A6 which is today's per-target cost.

CHUNKING. `diffusionv2.py:383` computes the number of denoiser chunks as
`multiplicity % max_parallel_samples + 1` where it means
`ceil(multiplicity / max_parallel_samples)`. With potentials on, multiplicity is
`3 * diffusion_samples`. At ds=4 that is `12 % 5 + 1 = 3` chunks of 4, which is
survivable by accident; at ds=40 it is `120 % 5 + 1 = 1` chunk of 120, which is not a
batch size anyone chose. So `ds` cannot be raised without addressing it, and the probe
runs the ds>4 configurations against a PATCHED copy of the package.

The patch is a chunk-count correction and nothing else. It cannot change a result: the
initial noise, the per-step noise and the random augmentation are all drawn for the full
multiplicity OUTSIDE the chunk loop, and the denoiser has no cross-sample term, so the
partition decides only how the work is batched. Floating point still means a different
batch size is not bitwise identical, which is why EQUIVALENCE below is measured rather
than asserted. The patched copy is a separate directory shadowed in via PYTHONPATH; the
installed package is never written to, because production tasks are reading it.

EQUIVALENCE. The scientific check, and the one that decides whether any of this is
allowed. Forty structures the current way (ten invocations, seeds 42..51, ds=4,
unpatched) against forty the proposed way (one invocation, seed 42, ds=40, patched).
These are not the same forty structures and are not meant to be -- the claim is that
they are draws from the same distribution, so the comparison is of the confidence
DISTRIBUTIONS, by mean, spread and a two-sample test. A shift in the mean, or a
collapse in the spread, is the collapse understating fold noise and is a reason to stop.

Results go to JSONL so the summary and any figure can be redrawn without refolding.

Run as a job, never in a node-hold allocation: boltz's ~11 GiB working set is above the
extern step's ceiling there and the failure is a bare SIGKILL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Taken from the adapter rather than retyped, so the probe measures the command line
# production actually runs. The adapter imports nothing from boltz.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proteinfoundation.cpseq.adapters.boltz2 import (  # noqa: E402
    BOLTZ_REV,
    Boltz2Config,
    _predict_cmd,
)

# The one line of upstream this probe changes, and what it becomes.
CHUNK_BUG = "sample_ids.chunk(\n                    multiplicity % max_parallel_samples + 1\n                )"
CHUNK_FIX = "sample_ids.chunk(\n                    max(1, -(-multiplicity // max_parallel_samples))\n                )"


def build_patched_boltz(site_boltz: Path, dest: Path) -> dict:
    """Copy the installed package and correct the chunk count. Never writes to the original.

    Returns a record of what was changed, including the sha of the file before and after,
    so a run's log says exactly which code produced its numbers.
    """
    import hashlib

    pkg = dest / "boltz"
    if pkg.exists():
        shutil.rmtree(pkg)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(site_boltz, pkg)

    target = pkg / "model" / "modules" / "diffusionv2.py"
    before = target.read_text()
    if CHUNK_BUG not in before:
        raise SystemExit(
            f"FATAL: the chunk-count expression is not where this probe expects it in "
            f"{target}. Upstream changed; re-read it before trusting any timing here."
        )
    after = before.replace(CHUNK_BUG, CHUNK_FIX)
    if after.count("max(1, -(-multiplicity // max_parallel_samples))") != 1:
        raise SystemExit("FATAL: chunk patch did not apply exactly once")
    target.write_text(after)
    return {
        "patched_file": str(target),
        "sha_before": hashlib.sha256(before.encode()).hexdigest()[:16],
        "sha_after": hashlib.sha256(after.encode()).hexdigest()[:16],
    }


def mutate_peptide(yaml_text: str, new_seq: str) -> str:
    """Swap the peptide chain's sequence, leaving every other line byte-identical.

    The probe needs several inputs that cost what a real candidate costs: same receptor,
    same MSA, same template, same pocket, different peptide. Editing the one line is
    safer than rebuilding the YAML, which could silently drop a constraint and make the
    multi-input timings measure a cheaper job than production runs.
    """
    lines = yaml_text.split("\n")
    # The peptide is the LAST `sequence:` in the file (receptor chains come first).
    idx = max(i for i, ln in enumerate(lines) if ln.strip().startswith("sequence:"))
    indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
    lines[idx] = f"{indent}sequence: {new_seq}"
    return "\n".join(lines)


def peptide_of(yaml_text: str) -> str:
    lines = [ln for ln in yaml_text.split("\n") if ln.strip().startswith("sequence:")]
    return lines[-1].split("sequence:", 1)[1].strip()


# A fixed alphabet and a fixed rotation: the probe's variants must be the same on every
# rerun, or two runs of the cost model are not comparable.
_AA = "ACDEFGHIKLMNPQRSTVWY"


def variants(seq: str, n: int) -> list[str]:
    """`n` distinct same-length peptides derived deterministically from `seq`."""
    out = [seq]
    k = 0
    while len(out) < n:
        k += 1
        pos = k % len(seq)
        sub = list(seq)
        sub[pos] = _AA[(_AA.index(sub[pos]) + k) % len(_AA)] if sub[pos] in _AA else "A"
        cand = "".join(sub)
        if cand not in out:
            out.append(cand)
    return out


def harvest_confidence(out_dir: Path) -> list[dict]:
    """Every per-sample confidence json boltz wrote under `out_dir`, with its stem."""
    rows = []
    for p in sorted(out_dir.rglob("confidence_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        m = re.match(r"confidence_(.+)_model_(\d+)$", p.stem)
        rows.append(
            {
                "stem": m.group(1) if m else p.stem,
                "model": int(m.group(2)) if m else -1,
                **{
                    k: float(d[k])
                    for k in ("confidence_score", "iptm", "ptm", "complex_plddt")
                    if isinstance(d.get(k), (int, float))
                },
            }
        )
    return rows


def run_config(
    *,
    name: str,
    yaml_texts: dict[str, str],
    diffusion_samples: int,
    max_parallel_samples: int,
    seed: int,
    patched: bool,
    args,
    patched_root: Path,
    kernels: bool = False,
    preprocessing_threads: int = 1,
) -> dict:
    """One boltz invocation, timed, with its confidence values harvested.

    `kernels=True` drops `--no_kernels`, which the adapter currently always passes. That
    flag's stated reason -- cuequivariance 0.13.0 needing torch>=2.11 against a pinned
    torch 2.7.0+cu126 -- no longer describes this venv: it holds torch 2.14.1+cu130 and
    an importable cuequivariance_torch 0.13.0, both installed 2026-10-19. Whether the
    fused triangle-multiplication kernels actually run, and what they are worth, is a
    measurement, which is why this is an arm and not an edit to the adapter.
    """
    stage = Path(args.work) / "stage" / name
    if stage.exists():
        shutil.rmtree(stage)
    # Inputs in their own directory, output a SIBLING of it -- never nested inside.
    # Pointed at a directory, boltz scans it for inputs, so an output directory within
    # it gets scanned too and the run fails: measured by the batching work as 0 of 6
    # structures returned with `out` as a child against 8 of 8 as a sibling.
    inputs_dir = stage / "inputs"
    out_dir = stage / "out"
    inputs_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    for key, text in yaml_texts.items():
        (inputs_dir / f"{key}.yaml").write_text(text)

    # Match production's two shapes exactly: `predict` points boltz at ONE yaml file and
    # `predict_many` at a directory. A single-input arm is the per-fold baseline every
    # other number is measured against, so it has to be the command production runs, not
    # a one-element batch that also pays for a directory scan.
    target = inputs_dir / f"{next(iter(yaml_texts))}.yaml" if len(yaml_texts) == 1 else inputs_dir

    cfg = Boltz2Config(
        diffusion_samples=diffusion_samples,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        use_potentials=not args.no_potentials,
        seed=seed,
    )
    cmd = _predict_cmd(
        target=target,
        out_dir=out_dir,
        config=cfg,
        python_exec=args.boltz_python,
        boltz_cache=args.boltz_cache,
        devices=1,
        accelerator="gpu",
        preprocessing_threads=preprocessing_threads,
    )
    cmd += ["--max_parallel_samples", str(max_parallel_samples)]
    if kernels:
        cmd = [c for c in cmd if c != "--no_kernels"]

    env = dict(os.environ)
    if patched:
        env["PYTHONPATH"] = str(patched_root) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

    multiplicity = diffusion_samples * (3 if cfg.use_potentials else 1)
    chunks = (
        max(1, -(-multiplicity // max_parallel_samples))
        if patched
        else multiplicity % max_parallel_samples + 1
    )
    print(
        f"[{name}] n_inputs={len(yaml_texts)} ds={diffusion_samples} "
        f"mps={max_parallel_samples} patched={patched} kernels={kernels} seed={seed} "
        f"-> multiplicity={multiplicity}, {chunks} chunk(s) of ~{-(-multiplicity // chunks)}",
        flush=True,
    )

    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=args.timeout_s, env=env
    )
    elapsed = time.time() - t0

    conf = harvest_confidence(out_dir)
    produced = len({(c["stem"]) for c in conf})
    rec = {
        "name": name,
        "n_inputs": len(yaml_texts),
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": max_parallel_samples,
        "multiplicity": multiplicity,
        "n_chunks": chunks,
        "patched": patched,
        "kernels": kernels,
        "preprocessing_threads": preprocessing_threads,
        "seed": seed,
        "use_potentials": cfg.use_potentials,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "predictor_error",
        "elapsed_s": round(elapsed, 2),
        "n_inputs_with_output": produced,
        "n_samples": len(conf),
        "confidence": conf,
        "boltz_rev": BOLTZ_REV,
        "stderr_tail": proc.stderr[-3000:] if proc.returncode != 0 else "",
    }
    if proc.returncode == 0 and len(conf):
        rec["s_per_sample_naive"] = round(elapsed / len(conf), 2)
    print(
        f"[{name}] {rec['status']} in {elapsed:.0f}s, "
        f"{len(conf)} sample(s) from {produced}/{len(yaml_texts)} input(s)",
        flush=True,
    )
    if proc.returncode != 0:
        # Not fatal. An OOM at a large batch is a RESULT -- it is the ceiling on
        # max_parallel_samples -- so the probe records it and carries on to the
        # smaller configurations rather than losing the whole job to it.
        print(f"[{name}] stderr tail:\n{proc.stderr[-1500:]}", flush=True)
    return rec


def summarise(key: str, rows: list[dict]) -> dict:
    vals = [c[key] for c in rows if key in c]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def mann_whitney_u(a: list[float], b: list[float]) -> dict:
    """Two-sided Mann-Whitney U with a normal approximation, ties corrected.

    Written out rather than imported because the repo venv is not on this job's path and
    the probe shells out to a boltz venv that has no scipy. n=40 per side is comfortably
    inside the normal approximation's range.
    """
    import math

    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {}
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    i = 0
    tie_term = 0.0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t**3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma_sq = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma_sq <= 0:
        return {"u": u1, "p_two_sided": 1.0}
    z = (u1 - mu) / math.sqrt(sigma_sq)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"u": round(u1, 1), "z": round(z, 3), "p_two_sided": round(p, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-yaml", required=True, help="a real production input.yaml")
    ap.add_argument("--work", required=True, help="scratch/work dir for stages")
    ap.add_argument("--out", required=True, help="results JSONL")
    ap.add_argument(
        "--boltz-python", default="/data/user_data/yixiz/envs/boltz2/.venv/bin/python"
    )
    ap.add_argument("--boltz-cache", default="/data/user_data/yixiz/boltz_cache")
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--no-potentials", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=5400)
    ap.add_argument("--group-size", type=int, default=9,
                    help="inputs per batched invocation; production folds 8 candidates "
                         "plus the unchanged peptide")
    ap.add_argument("--equiv-seeds", type=int, default=10,
                    help="seeds in the current-scheme arm of the equivalence test")
    ap.add_argument("--production-ds", type=int, default=1,
                    help="diffusion_samples per fold as production actually runs it. "
                         "Verified as 1 from the live corpus records on 2026-10-25, "
                         "NOT the 4 in Boltz2Config's default -- which matters, because "
                         "with potentials on it makes the internal multiplicity 3 and "
                         "the buggy chunk count 3%%5+1=4, so torch.chunk returns three "
                         "chunks of ONE and the denoiser runs at batch size 1.")
    ap.add_argument("--wall-budget-s", type=int, default=6000,
                    help="stop starting phase-3 configurations once this much of the "
                         "job's wall clock is gone; 0 disables the guard")
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    args = ap.parse_args()

    base_yaml = Path(args.input_yaml).read_text()
    pep = peptide_of(base_yaml)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    site = subprocess.run(
        [args.boltz_python, "-c", "import boltz,os;print(os.path.dirname(boltz.__file__))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    patched_root = work / "boltz_patched"
    patch_rec = build_patched_boltz(Path(site), patched_root)

    print(f"peptide:      {pep} (len {len(pep)})", flush=True)
    print(f"installed:    {site}", flush=True)
    print(f"patched copy: {patched_root}  {patch_rec['sha_before']} -> {patch_rec['sha_after']}", flush=True)

    seqs = variants(pep, args.group_size)
    one = {"c0": base_yaml}
    group = {f"c{i}": mutate_peptide(base_yaml, s) for i, s in enumerate(seqs)}

    results: list[dict] = []

    def record(rec: dict) -> None:
        results.append(rec)
        with out_path.open("a") as fh:
            fh.write(json.dumps({**rec, "patch": patch_rec}) + "\n")

    t_start = time.time()

    def budget_left(reserve_s: int) -> bool:
        """Whether `reserve_s` of work still fits in the job's wall clock."""
        if args.wall_budget_s <= 0:
            return True
        left = args.wall_budget_s - (time.time() - t_start)
        if left < reserve_s:
            print(f"  (skipping: {left / 60:.0f} min left, needs ~{reserve_s / 60:.0f})",
                  flush=True)
            return False
        return True

    # ------------------------------------------------- phase 1: cost model, singles
    # Cheapest and safest first. An OOM at a large batch is itself a result -- the
    # ceiling on max_parallel_samples -- and must not cost the configurations after it.
    pd = args.production_ds
    coll = pd * args.equiv_seeds          # the collapsed fold: all ten seeds at once
    if not args.skip_cost:
        print(f"\n--- phase 1: cost model (single input, ds {pd} -> {coll}) ---", flush=True)
        for name, ds, mps, patched, kern in [
            # A1 is today, bit for bit: unpatched, so multiplicity 3 in chunks of one.
            (f"A1_prod_ds{pd}_mps5", pd, 5, False, False),
            # A2 is A1 with ONLY the chunk count corrected -- same samples, same noise,
            # one batch of 3 instead of three of 1. The chunk fix's value, isolated.
            (f"A2_prod_ds{pd}_mps5_fix", pd, 5, True, False),
            (f"A3_ds{coll}_mps5_fix", coll, 5, True, False),
            (f"A4_ds{coll}_mps15_fix", coll, 15, True, False),
            (f"A5_ds{coll}_mps30_fix", coll, 30, True, False),
            (f"K1_prod_ds{pd}_mps5_kern", pd, 5, True, True),
            (f"K2_ds{coll}_mps15_kern", coll, 15, True, True),
        ]:
            record(run_config(
                name=name, yaml_texts=one, diffusion_samples=ds,
                max_parallel_samples=mps, seed=42, patched=patched, kernels=kern,
                args=args, patched_root=patched_root,
            ))

    # ------------------------------------------------------- phase 2: equivalence
    # The decision-critical phase, so it runs before the 9-input confirmations, which
    # the cost model can predict anyway. Every arm draws the same NUMBER of structures
    # (4 * equiv_seeds); they are not the same structures and are not meant to be.
    if not args.skip_equivalence:
        print("\n--- phase 2: equivalence ---", flush=True)
        old: list[dict] = []
        for k in range(args.equiv_seeds):
            r = run_config(
                name=f"E_old_seed{42 + k}", yaml_texts=one, diffusion_samples=pd,
                max_parallel_samples=5, seed=42 + k, patched=False,
                args=args, patched_root=patched_root,
            )
            record(r)
            old += r["confidence"]

        arms = {}
        for label, kern in [(f"new_ds{coll}", False), (f"new_ds{coll}_kern", True)]:
            r = run_config(
                name=f"E_{label}", yaml_texts=one, diffusion_samples=coll,
                max_parallel_samples=15, seed=42, patched=True, kernels=kern,
                args=args, patched_root=patched_root,
            )
            record(r)
            arms[label] = r["confidence"]

        comparison = {
            "name": "EQUIVALENCE",
            "baseline": f"{args.equiv_seeds} invocations x ds={pd}, seeds "
                        f"42..{41 + args.equiv_seeds}, unpatched, no kernels",
            "n_old": len(old),
        }
        for label, new in arms.items():
            block = {"n_new": len(new)}
            for key in ("confidence_score", "iptm", "ptm", "complex_plddt"):
                a = [c[key] for c in old if key in c]
                b = [c[key] for c in new if key in c]
                block[key] = {
                    "old": summarise(key, old),
                    "new": summarise(key, new),
                    "test": mann_whitney_u(a, b),
                }
            comparison[label] = block
        record(comparison)
        print(json.dumps(comparison, indent=2), flush=True)

    # --------------------------------------- phase 3: 9-input confirmations (last)
    # Predictable from phase 1 via T = F + n*(K + ds*s); measured anyway because the
    # prediction is the thing being tested. Guarded by the wall clock.
    if not args.skip_cost:
        print("\n--- phase 3: batched invocations ---", flush=True)
        # P1/P2/P4 isolate what serialised preprocessing costs. Boltz preprocesses a
        # directory of inputs in parallel and each thread holds its own working set, so
        # 9 threads in a 32 GB task was SIGKILLed; production now pins it to 1. The
        # batch verification then came back at ~40 s/structure against the 31 s the
        # 44 s + 26 s model predicts, which points at the serialisation rather than the
        # model. If 2 or 4 threads fit, some of that is recoverable; if they die, the
        # OOM is the answer and the probe records it and moves on.
        for name, ds, mps, patched, kern, thr, reserve in [
            (f"A6_9in_ds{pd}_mps5_thr1", pd, 5, False, False, 1, 600),  # today's batch
            (f"P2_9in_ds{pd}_mps5_thr2", pd, 5, False, False, 2, 600),
            (f"P4_9in_ds{pd}_mps5_thr4", pd, 5, False, False, 4, 600),
            (f"A7_9in_ds{coll}_mps15_fix", coll, 15, True, False, 1, 1500),  # proposal
            (f"K3_9in_ds{coll}_mps15_kern", coll, 15, True, True, 1, 1500),
        ]:
            if not budget_left(reserve):
                continue
            record(run_config(
                name=name, yaml_texts=group, diffusion_samples=ds,
                max_parallel_samples=mps, seed=42, patched=patched, kernels=kern,
                preprocessing_threads=thr, args=args, patched_root=patched_root,
            ))

    print(f"\nwrote {out_path}\n", flush=True)
    print(f"  {'config':26s} {'elapsed':>9s} {'samples':>8s} {'s/sample':>9s}", flush=True)
    for r in results:
        if r.get("name", "").startswith(("A", "K", "E_")) and r.get("status") == "ok":
            print(f"  {r['name']:26s} {r['elapsed_s']:8.1f}s {r['n_samples']:8d} "
                  f"{r.get('s_per_sample_naive', 0):9.2f}", flush=True)


if __name__ == "__main__":
    main()
