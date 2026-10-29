"""Fine-tune ESM-2 / PepMLM into the CPSea cyclic-peptide unmasking proposer.

The arm: given a partially-masked cyclic-peptide sequence, the receptor pocket and the
requested cyclization chemistry, fill the masked positions. One forward per query, which
is what makes it usable as the proposal step of `cpseq` search -- the alternative, running
the flow co-generator's ODE, is 200-400 network evaluations per proposal.

WARM START. `--model` defaults to `ChatterjeeLab/PepMLM-650M` rather than raw
`facebook/esm2_t33_650M_UR50D`. PepMLM is ESM-2-650M already fine-tuned for exactly this
shape of task -- target sequence first, masked binder region at the C-terminus,
reconstruct the binder -- so it starts from a model that expects "context, then fill",
and this run only has to teach it cyclic-peptide statistics, the chemistry token and the
partial-mask regime. Raw ESM-2 is a supported fallback and the honest control for "did
the PepMLM warm start help".

WHAT IS MEASURED, and why each number is here. Hidden-position accuracy alone is not a
result; this project has twice produced a sequence metric that looked like a capability
and was not:
  * scoring positions that were fed in gives ~1.0 and measures COPYING. So `labels` is
    -100 on every revealed position (see `cpseq_mlm_data.EsmMlmEncoder.encode`) and the
    accuracy reported here is over withheld positions only.
  * a hidden-position accuracy can land exactly on the "always guess the commonest
    residue" rate and still read as learning. So every eval reports
    `marginal_argmax_acc` and `positional_argmax_acc` on the SAME hidden positions. The
    positional one is the real threat: CPSea cyclises at the termini, so a model that
    learns "anchor residue at the ends" scores well while knowing nothing about a pocket.
  * accuracy that does not MOVE with training is a dosage or objective problem, not a
    "needs more steps" problem, so the metrics JSONL carries every eval and the step it
    came from.

THE POCKET-SHUFFLE CONTROL. Reported at every eval: the same peptides and masks, with
pockets permuted across examples. A pocket-conditioned model must lose accuracy; one that
does not has learned a CPSea amino-acid prior and is target-blind. Two arms in this
project shipped a conditioning channel whose gates never opened, so this is not a
formality. The `--no-pocket` twin is the other half of the same question.

Metrics are appended to `metrics.jsonl` in the run directory and never only logged, so
every figure is redrawable from disk without a GPU.

Usage:
    python script_utils/cpseq_train_mlm.py \
        --corpus-dir $SCRATCH/cpseq_mlm_corpus --run-dir $SCRATCH/runs/cpseq_mlm_v1 \
        --max-steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "script_utils") not in sys.path:
    sys.path.insert(0, str(_REPO / "script_utils"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cpseq_mlm_data import (  # noqa: E402
    AA20,
    REGIMES,
    CorpusRow,
    EsmMlmEncoder,
    MaskScheduleConfig,
    add_task_tokens,
    amino_acid_marginal,
    baseline_accuracy,
    collate,
    draw_mask,
    load_corpus,
    positional_marginal,
)
from proteinfoundation.cyclization.constants import UNSPECIFIED  # noqa: E402


def build_eval_panel(
    rows: list[CorpusRow], cfg: MaskScheduleConfig, seed: int, n: int
) -> list[tuple[CorpusRow, list[int], str]]:
    """A FIXED set of (row, hidden positions, regime), drawn once.

    Fixed because every comparison this arm makes is paired: checkpoint against
    checkpoint, pocket against no-pocket, real pocket against shuffled. Redrawing the
    masks per eval would put sampling noise on exactly the axis being measured, and with
    peptides of 5-16 residues that noise is large.

    Stratified by regime rather than drawn from the mixture, so each regime gets a usable
    count even though `p_anchor` is 0.15 of the training mixture.
    """
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    per = max(1, n // len(REGIMES))
    panel: list[tuple[CorpusRow, list[int], str]] = []
    idx = 0
    for regime in REGIMES:
        forced = MaskScheduleConfig(
            p_denovo=1.0 if regime == "denovo" else 0.0,
            p_budget=1.0 if regime == "budget" else 0.0,
            p_anchor=1.0 if regime == "anchor" else 0.0,
            p_span=1.0 if regime == "span" else 0.0,
            anchor_extra_min=cfg.anchor_extra_min,
            anchor_extra_max=cfg.anchor_extra_max,
            span_frac_min=cfg.span_frac_min,
            span_frac_max=cfg.span_frac_max,
            cyc_dropout=0.0,
        )
        for _ in range(per):
            if idx >= len(pool):
                break
            row = pool[idx]
            idx += 1
            hidden, got = draw_mask(row, forced, rng)
            panel.append((row, hidden, got))
    return panel


@torch.no_grad()
def evaluate(
    model,
    encoder: EsmMlmEncoder,
    panel: list[tuple[CorpusRow, list[int], str]],
    *,
    device: str,
    batch_size: int,
    residue_ids: torch.Tensor,
    shuffle_pockets: bool = False,
    shuffle_seed: int = 0,
) -> dict:
    """Per-regime hidden-position accuracy, top-3 and perplexity.

    The softmax is restricted to the 20 canonical residues before anything is scored. An
    ESM-2 head can emit `X`, `B`, `<mask>` or a special token; leaving those in the
    denominator would report a perplexity over a vocabulary the task does not use and
    would let an "X" prediction count as a near-miss rather than as no answer at all.
    """
    model.eval()
    rows = [p[0] for p in panel]
    if shuffle_pockets:
        perm = list(range(len(rows)))
        random.Random(shuffle_seed).shuffle(perm)
        # A row paired with its own pocket by chance is left alone rather than forced to
        # move: forcing a derangement would bias which examples get an easy pocket.
        rows = [
            CorpusRow(
                example_id=r.example_id, pep_seq=r.pep_seq,
                pocket_seq=rows[perm[i]].pocket_seq, cyc_type=r.cyc_type,
                cyc_anchor_i=r.cyc_anchor_i, cyc_anchor_j=r.cyc_anchor_j,
                cluster_id=r.cluster_id, split=r.split,
            )
            for i, r in enumerate(rows)
        ]

    per_regime: dict[str, dict] = {
        r: {"n": 0, "hit": 0, "hit3": 0, "nll": 0.0} for r in REGIMES
    }
    encoded = []
    for (row_orig, hidden, regime), row in zip(panel, rows):
        enc = encoder.encode(row, hidden)
        if enc["too_long"]:
            continue
        encoded.append((enc, regime))

    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        batch = collate([c[0] for c in chunk], encoder.pad_id)
        regimes = [c[1] for c in chunk]
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[..., residue_ids]  # [b, t, 20]
        labels = batch["labels"]
        # Map vocab-id labels onto 0..19 positions in `residue_ids`.
        lut = torch.full((int(residue_ids.max()) + 1,), -100, dtype=torch.long, device=device)
        lut[residue_ids] = torch.arange(len(residue_ids), device=device)
        scored = labels != -100
        safe = labels.clamp(min=0)
        tgt = torch.where(scored, lut[safe], torch.full_like(labels, -100))

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for b in range(logits.shape[0]):
            m = scored[b] & (tgt[b] >= 0)
            if not bool(m.any()):
                continue
            lp = logprobs[b][m]            # [k, 20]
            t = tgt[b][m]                  # [k]
            top1 = lp.argmax(-1)
            top3 = lp.topk(min(3, lp.shape[-1]), dim=-1).indices
            d = per_regime[regimes[b]]
            d["n"] += int(m.sum())
            d["hit"] += int((top1 == t).sum())
            d["hit3"] += int((top3 == t[:, None]).any(-1).sum())
            d["nll"] += float(-lp.gather(-1, t[:, None]).sum())

    out_metrics: dict = {}
    tot_n = tot_hit = tot_hit3 = 0
    tot_nll = 0.0
    for regime, d in per_regime.items():
        n = d["n"]
        if n == 0:
            continue
        out_metrics[f"acc/{regime}"] = d["hit"] / n
        out_metrics[f"top3/{regime}"] = d["hit3"] / n
        out_metrics[f"ppl/{regime}"] = math.exp(d["nll"] / n)
        out_metrics[f"n_hidden/{regime}"] = n
        tot_n += n
        tot_hit += d["hit"]
        tot_hit3 += d["hit3"]
        tot_nll += d["nll"]
    if tot_n:
        out_metrics["acc/all"] = tot_hit / tot_n
        out_metrics["top3/all"] = tot_hit3 / tot_n
        out_metrics["ppl/all"] = math.exp(tot_nll / tot_n)
        out_metrics["n_hidden/all"] = tot_n
    model.train()
    return out_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="ChatterjeeLab/PepMLM-650M")
    ap.add_argument("--train-glob", default="train_*.jsonl")
    ap.add_argument("--val-glob", default="val_*.jsonl")
    ap.add_argument("--no-pocket", action="store_true",
                    help="the target-free control arm: identical in every other respect")
    ap.add_argument("--holdout-clusters", default="",
                    help="file of cluster_ids, one per line, dropped from TRAIN")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=800)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-limit", type=int, default=0, help="smoke only")
    ap.add_argument("--cyc-dropout", type=float, default=0.15)
    ap.add_argument(
        "--keep-last-n-hf", type=int, default=2,
        help="HF-format checkpoint directories to retain. Each is ~2.6 GB, so a 20k-step "
             "run at the default save cadence would leave 26 GB behind; project memory "
             "records that exact cost from a trainer that never pruned. 0 keeps all.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== cpseq_train_mlm ===")
    print(f"host={os.uname().nodename} device={device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name} mem={props.total_memory / 1e9:.0f}GB")
    print(f"arm={'NO-POCKET (control)' if args.no_pocket else 'pocket-conditioned'}")

    # ---- data
    corpus = Path(args.corpus_dir)
    train_files = sorted(corpus.glob(args.train_glob))
    val_files = sorted(corpus.glob(args.val_glob))
    if not train_files:
        raise SystemExit(f"no train shards matching {args.train_glob} in {corpus}")
    holdout: set[str] = set()
    if args.holdout_clusters:
        holdout = {
            ln.strip() for ln in Path(args.holdout_clusters).read_text().splitlines()
            if ln.strip()
        }
        print(f"holdout clusters: {len(holdout)}")
    train_rows = load_corpus(train_files, exclude_clusters=holdout)
    val_rows = load_corpus(val_files) if val_files else []
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if not val_rows:
        # Never silently validate on training data: carve a disjoint slice by cluster.
        print("[warn] no val shards; holding out 2% of train BY CLUSTER for validation")
        clusters = sorted({r.cluster_id or r.example_id for r in train_rows})
        random.Random(args.seed).shuffle(clusters)
        val_cl = set(clusters[: max(1, len(clusters) // 50)])
        val_rows = [r for r in train_rows if (r.cluster_id or r.example_id) in val_cl]
        train_rows = [r for r in train_rows if (r.cluster_id or r.example_id) not in val_cl]
    print(f"train rows={len(train_rows)} val rows={len(val_rows)}")

    # ---- model
    os.environ.setdefault("HF_HOME", str(Path(os.environ.get("SCRATCH", "/tmp")) / "hf"))
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"loading {args.model} (HF_HOME={os.environ['HF_HOME']})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, dtype=torch.float32)
    n_added = add_task_tokens(tok, model)
    print(f"added {n_added} task tokens; vocab now {len(tok)}")
    model.to(device)
    model.train()

    cfg = MaskScheduleConfig(cyc_dropout=args.cyc_dropout)
    encoder = EsmMlmEncoder(tok, max_length=args.max_length,
                            pocket_enabled=not args.no_pocket)
    residue_ids = torch.tensor(encoder.residue_ids(), device=device)

    # ---- baselines, computed ONCE from train and reported beside every eval
    panel = build_eval_panel(val_rows, cfg, seed=args.seed, n=args.eval_n)
    marg = amino_acid_marginal(train_rows)
    posm = positional_marginal(train_rows)
    base_all = baseline_accuracy(
        [p[0] for p in panel], [p[1] for p in panel], marg, posm
    )
    per_regime_base = {}
    for regime in REGIMES:
        sub = [(r, h) for (r, h, g) in panel if g == regime]
        if sub:
            per_regime_base[regime] = baseline_accuracy(
                [s[0] for s in sub], [s[1] for s in sub], marg, posm
            )
    manifest = {
        "model": args.model, "arm": "no_pocket" if args.no_pocket else "pocket",
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "panel_size": len(panel), "seed": args.seed,
        "lr": args.lr, "batch_size": args.batch_size, "accum": args.accum,
        "mask_schedule": cfg.__dict__, "holdout_clusters": len(holdout),
        "baselines": {"all": base_all, "per_regime": per_regime_base},
        "amino_acid_marginal": marg,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("baselines (hidden positions only):")
    print(f"  marginal argmax ('{base_all['marginal_argmax_residue']}'): "
          f"{base_all['marginal_argmax_acc']:.4f}")
    print(f"  positional argmax:                {base_all['positional_argmax_acc']:.4f}")
    print("  >>> the trained model must beat the POSITIONAL number, not the marginal one.")

    # ---- optimiser
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (no_decay if (p_.ndim == 1 or n_.endswith(".bias")) else decay).append(p_)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    metrics_path = run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- resume
    start_step = 0
    last = ckpt_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = int(state["step"])
        print(f"resumed from {last} at step {start_step}")

    rng = random.Random(args.seed + start_step)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"bf16 autocast: {use_bf16}")

    t0 = time.time()
    step = start_step
    running = []
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            encs = []
            while len(encs) < args.batch_size:
                row = train_rows[rng.randrange(len(train_rows))]
                hidden, _regime = draw_mask(row, cfg, rng)
                cyc = UNSPECIFIED if rng.random() < cfg.cyc_dropout else row.cyc_type
                enc = encoder.encode(row, hidden, cyc_type=cyc)
                if not enc["too_long"]:
                    encs.append(enc)
            batch = collate(encs, encoder.pad_id)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
                loss = F.cross_entropy(
                    out.logits.view(-1, out.logits.shape[-1]).float(),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
            (loss / args.accum).backward()
            running.append(float(loss))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        step += 1

        if step % 50 == 0:
            row = {"step": step, "train/loss": sum(running) / len(running),
                   "lr": lr_at(step), "seconds": round(time.time() - t0, 1)}
            running = []
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"step {step}/{args.max_steps} loss={row['train/loss']:.4f} "
                  f"lr={row['lr']:.2e} {row['seconds']:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.max_steps:
            m = evaluate(model, encoder, panel, device=device,
                         batch_size=args.eval_batch_size, residue_ids=residue_ids)
            m_shuf = evaluate(model, encoder, panel, device=device,
                              batch_size=args.eval_batch_size, residue_ids=residue_ids,
                              shuffle_pockets=True, shuffle_seed=args.seed)
            row = {"step": step, "eval": True}
            row.update({f"val/{k}": v for k, v in m.items()})
            row.update({f"val_pocketshuf/{k}": v for k, v in m_shuf.items()})
            row["val/baseline_marginal_acc"] = base_all["marginal_argmax_acc"]
            row["val/baseline_positional_acc"] = base_all["positional_argmax_acc"]
            # The headline: how much of the accuracy survives losing the right pocket.
            if "acc/all" in m and "acc/all" in m_shuf:
                row["val/pocket_gain"] = m["acc/all"] - m_shuf["acc/all"]
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  [eval {step}] acc/all={m.get('acc/all', float('nan')):.4f} "
                  f"(pos-prior {base_all['positional_argmax_acc']:.4f}) "
                  f"shuffled={m_shuf.get('acc/all', float('nan')):.4f} "
                  f"gain={row.get('val/pocket_gain', float('nan')):+.4f}", flush=True)
            for regime in REGIMES:
                if f"acc/{regime}" in m:
                    b = per_regime_base.get(regime, {})
                    print(f"      {regime:7s} acc={m[f'acc/{regime}']:.4f} "
                          f"ppl={m[f'ppl/{regime}']:.3f} "
                          f"pos-prior={b.get('positional_argmax_acc', float('nan')):.4f} "
                          f"n={m[f'n_hidden/{regime}']}")

        if step % args.save_every == 0 or step == args.max_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)}, ckpt_dir / "last.pt.tmp")
            os.replace(ckpt_dir / "last.pt.tmp", ckpt_dir / "last.pt")
            # A HF-format copy too, so the proposer loads it with `from_pretrained`
            # instead of needing this script's state-dict layout.
            hf_dir = ckpt_dir / f"hf_step{step}"
            model.save_pretrained(hf_dir)
            tok.save_pretrained(hf_dir)
            print(f"  saved step {step} -> {hf_dir}", flush=True)

            if args.keep_last_n_hf > 0:
                # Sorted by the step in the NAME, not by mtime: a resumed run rewrites
                # older directories' mtimes and an mtime sort would then prune the
                # newest checkpoint. Keyed numerically, not lexically, so step 10000
                # does not sort before step 2000.
                existing = sorted(
                    (d for d in ckpt_dir.glob("hf_step*") if d.is_dir()),
                    key=lambda d: int(d.name.removeprefix("hf_step")),
                )
                for stale in existing[: -args.keep_last_n_hf]:
                    import shutil

                    shutil.rmtree(stale, ignore_errors=True)
                    print(f"  pruned {stale.name}", flush=True)

    print(f"done in {(time.time() - t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
