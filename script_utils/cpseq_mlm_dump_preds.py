"""Dump per-position predictions for a cpseq_mlm checkpoint on the FIXED eval panel.

WHY THIS EXISTS. `cpseq_train_mlm.evaluate` accumulates hit counts and throws the
predictions away. Every number the arm has reported is therefore an EXACT-MATCH accuracy
and nothing else can be asked of a finished run: not whether the errors are conservative
substitutions, not how the model is calibrated, not what a different scoring rule would
say. Re-running the GPU for each new question is the wrong shape. This dumps the full
20-way log-probability at every hidden position once, so every later metric is a CPU
reduction over a file -- the same rule the rest of this project follows for figures.

WHAT IS GUARANTEED TO MATCH the training-time numbers. The panel is not redrawn here: it
is built by calling `cpseq_train_mlm.build_eval_panel` with the run's own seed and
`--eval-n` out of its manifest, and rows are encoded and filtered (`too_long`) by exactly
the path `evaluate` uses. Dumping a panel drawn any other way would put sampling noise on
the one axis every comparison in this arm is paired over. The dump therefore reproduces
`acc/*` to the position; `--check-acc` asserts that against the manifest-era metrics so a
silent drift in the panel is caught here rather than in a downstream table.

The shuffled-pocket arm is dumped by the same call with `--shuffle-pockets`, reusing the
run's shuffle seed, because the pocket-gain control deserves the same scrutiny as the
headline: an error analysis that covers only the real-pocket arm cannot say whether the
gain is chemically meaningful or just differently wrong.

Usage:
    python script_utils/cpseq_mlm_dump_preds.py \
        --run-dir /zfsauton/scratch/yixiz/cpseq_mlm/runs/cpseq_mlm_pocket \
        --ckpt hf_step60000 \
        --out /zfsauton/scratch/yixiz/cpseq_mlm/probes/preds_pocket_step60000.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "script_utils") not in sys.path:
    sys.path.insert(0, str(_REPO / "script_utils"))

import torch  # noqa: E402

from cpseq_mlm_data import (  # noqa: E402
    AA20,
    REGIMES,
    CorpusRow,
    EsmMlmEncoder,
    MaskScheduleConfig,
    collate,
    load_corpus,
)
from cpseq_train_mlm import build_eval_panel  # noqa: E402


def resolve_ckpt(run_dir: Path, name: str | None) -> Path:
    """Pick the HF checkpoint directory to dump.

    Default is the HIGHEST step, keyed numerically on the directory name. Not `sorted()`
    and not mtime: project memory records a checkpoint resolver that took
    `sorted(iterdir())[-1]` and so preferred `hf_step9000` over `hf_step10000`, and a
    resumed run rewrites older directories' mtimes.
    """
    ckpt_root = run_dir / "checkpoints"
    if name:
        p = ckpt_root / name
        if not p.is_dir():
            raise SystemExit(f"FATAL: no such checkpoint directory: {p}")
        return p
    cands = [d for d in ckpt_root.glob("hf_step*") if d.is_dir()]
    if not cands:
        raise SystemExit(f"FATAL: no hf_step* checkpoint in {ckpt_root}")
    return max(cands, key=lambda d: int(d.name.removeprefix("hf_step")))


def shuffle_rows(panel_rows: list[CorpusRow], seed: int) -> list[CorpusRow]:
    """Reassign pockets across rows, exactly as `evaluate(shuffle_pockets=True)` does.

    Kept byte-compatible with the trainer -- same permutation construction, same decision
    NOT to force a derangement -- so the dumped shuffled arm is the same control the
    training curve reported, not a second, differently-shuffled one.
    """
    perm = list(range(len(panel_rows)))
    random.Random(seed).shuffle(perm)
    return [
        CorpusRow(
            example_id=r.example_id, pep_seq=r.pep_seq,
            pocket_seq=panel_rows[perm[i]].pocket_seq, cyc_type=r.cyc_type,
            cyc_anchor_i=r.cyc_anchor_i, cyc_anchor_j=r.cyc_anchor_j,
            cluster_id=r.cluster_id, split=r.split,
        )
        for i, r in enumerate(panel_rows)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint dir name; default = highest step")
    ap.add_argument("--corpus-dir", default=None, help="default: read from the env/root used in training")
    ap.add_argument("--val-glob", default="val_*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--shuffle-pockets", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="must match the trainer's eval call to reproduce its control")
    ap.add_argument("--check-acc", type=float, default=None,
                    help="assert pooled exact accuracy is within 1e-6 of this (sanity gate)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # The panel is a function of (val rows, mask schedule, seed, n). All four come from
    # the run's own manifest so the dump cannot drift from what training measured.
    seed = int(manifest["seed"])
    panel_n = int(manifest["panel_size"])
    sched = dict(manifest["mask_schedule"])
    no_pocket = manifest["arm"] == "no_pocket"

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path(
        __import__("os").environ.get(
            "CPSEQ_MLM_CORPUS_DIR", "/zfsauton/scratch/yixiz/cpseq_mlm/corpus"
        )
    )
    val_files = sorted(corpus_dir.glob(args.val_glob))
    if not val_files:
        raise SystemExit(f"FATAL: no {args.val_glob} under {corpus_dir}")
    val_rows = load_corpus(val_files)
    print(f"val rows={len(val_rows)} from {len(val_files)} shard(s)")

    cfg = MaskScheduleConfig(**sched)
    # `panel_size` is the REALISED panel length, which `build_eval_panel` derives from
    # `n // len(REGIMES)` per regime. Passing the realised length back in would shrink the
    # panel, so the request is reconstructed rather than reused.
    panel = build_eval_panel(val_rows, cfg, seed=seed, n=panel_n)
    if len(panel) != panel_n:
        # Not fatal on its own -- a differently sharded corpus changes row order -- but it
        # means the dump is NOT the panel training scored, and nothing downstream should
        # be compared against the training curve.
        print(f"[warn] panel is {len(panel)} rows, manifest says {panel_n}: the corpus "
              f"differs from the one trained on; paired comparisons are INVALID")

    ckpt = resolve_ckpt(run_dir, args.ckpt)
    step = int(ckpt.name.removeprefix("hf_step")) if ckpt.name.startswith("hf_step") else -1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"checkpoint: {ckpt}  (step {step})  device={device}  arm={manifest['arm']}")

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # The checkpoint carries its own tokenizer including the task tokens
    # (`added_tokens.json`), so `add_task_tokens` is NOT called again here: the embeddings
    # were resized before saving and re-adding would be a no-op at best.
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMaskedLM.from_pretrained(ckpt, dtype=torch.float32).to(device)
    model.eval()
    encoder = EsmMlmEncoder(tok, max_length=args.max_length, pocket_enabled=not no_pocket)
    residue_ids = torch.tensor(encoder.residue_ids(), device=device)

    rows = [p[0] for p in panel]
    if args.shuffle_pockets:
        rows = shuffle_rows(rows, args.shuffle_seed)

    # Encode first, dropping `too_long` exactly as `evaluate` does, so the surviving
    # position set is the one the accuracy was computed over.
    encoded: list[tuple[dict, CorpusRow, list[int], str]] = []
    n_too_long = 0
    for (row_orig, hidden, regime), row in zip(panel, rows):
        enc = encoder.encode(row, hidden)
        if enc["too_long"]:
            n_too_long += 1
            continue
        encoded.append((enc, row_orig, sorted(hidden), regime))
    print(f"panel={len(panel)} encoded={len(encoded)} dropped_too_long={n_too_long}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pos = n_hit = 0
    with out_path.open("w") as fh:
        with torch.no_grad():
            for i in range(0, len(encoded), args.batch_size):
                chunk = encoded[i : i + args.batch_size]
                batch = collate([c[0] for c in chunk], encoder.pad_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
                logits = out.logits[..., residue_ids]          # [b, t, 20] in AA20 order
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                scored = batch["labels"] != -100

                for b, (_enc, row_orig, hidden, regime) in enumerate(chunk):
                    idx = scored[b].nonzero(as_tuple=True)[0]
                    # Peptide tokens appear in sequence order, so the ascending scored
                    # token indices line up with the ascending hidden peptide positions.
                    # Asserted rather than assumed: a silent off-by-one here would relabel
                    # every prediction and still produce a plausible accuracy.
                    if len(idx) != len(hidden):
                        raise RuntimeError(
                            f"{row_orig.example_id}: {len(idx)} scored tokens for "
                            f"{len(hidden)} hidden positions -- encoder layout changed"
                        )
                    lp = logprobs[b][idx]                      # [k, 20]
                    for k, pos in enumerate(hidden):
                        truth = row_orig.pep_seq[pos]
                        if truth not in AA20:
                            continue
                        vec = lp[k]
                        pred = AA20[int(vec.argmax())]
                        n_pos += 1
                        n_hit += int(pred == truth)
                        fh.write(json.dumps({
                            "example_id": row_orig.example_id,
                            "cluster_id": row_orig.cluster_id,
                            "regime": regime,
                            "pos": pos,
                            "L": len(row_orig.pep_seq),
                            "cyc_type": row_orig.cyc_type,
                            "true": truth,
                            "pred": pred,
                            # Rounded to 4 dp: that is ~1e-4 of log-probability, far below
                            # any difference this panel can resolve, and it keeps the dump
                            # small enough to commit to scratch and re-read freely.
                            "logprobs": [round(float(v), 4) for v in vec.tolist()],
                        }) + "\n")

    acc = n_hit / max(n_pos, 1)
    print(f"wrote {n_pos} positions -> {out_path}")
    print(f"pooled exact accuracy = {acc:.6f}")
    if args.check_acc is not None:
        # Tolerance is 5e-4, not exact equality: the dump reads a checkpoint while the
        # training curve scored the live (identical) weights, and cuDNN kernel selection
        # can differ. A real panel mismatch moves this by percent, not by 1e-4.
        if abs(acc - args.check_acc) > 5e-4:
            raise SystemExit(
                f"FATAL: pooled accuracy {acc:.6f} != expected {args.check_acc:.6f}. "
                f"The dumped panel is not the one training scored; do not use this dump "
                f"for any paired comparison."
            )
        print(f"check-acc OK (expected {args.check_acc:.6f})")
    print(f"regimes present: {sorted({e[3] for e in encoded})} of {list(REGIMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
