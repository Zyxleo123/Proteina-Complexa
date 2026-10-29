"""Corpus reading, tokenisation and the mask schedule for the CPSea sequence proposer.

Shared by training, evaluation and the proposer itself, so the masking regime a number
was measured under cannot drift from the one the model was trained on.

THE INPUT LAYOUT, and why it is PepMLM's and not something new:

    <cls> <pocket residues ...> <eos> <cyc:TYPE> <peptide ...> <eos>

PepMLM (Nat Biotechnol 2025) puts the target sequence first and the binder region at the
C-terminus, and fine-tunes ESM-2 to reconstruct the binder. Keeping that layout is what
makes `ChatterjeeLab/PepMLM-650M` a usable warm start rather than just a weight donor:
the model already expects "context, then the region I must fill".

Two additions CPSea needs and PepMLM has no concept of:

  <cyc:TYPE>  The requested chemistry, as a token immediately before the peptide. A ring
              is not a property of the sequence -- the same 8-mer is a different molecule
              head-to-tail than disulfide-closed -- so the chemistry has to be an input,
              not something the model infers. `<cyc:unspecified>` is the classifier-free
              null and is what the dropout rewrites a row to, matching how the flow stack
              treats UNSPECIFIED.
  <gap>       Between pocket residues that are not consecutive in the receptor. A pocket
              is a handful of fragments, not a contiguous chain, and project memory
              measured the difference between a contiguous receptor and a fragmented
              pocket at +22 to +34 points of closure. Without a gap marker the model is
              told two residues 40 apart in the chain are neighbours.

THE MASK SCHEDULE IS A MIXTURE, NOT A RATE. This is the part most likely to be got wrong,
and the failure is silent. A model trained at one mask rate is being asked, at inference,
a question it never saw: the search draws single substitutions, random bands at fixed
mutation fractions, and whole-peptide redesigns, and those are different queries. So the
schedule samples a REGIME per example and records which one, and every metric is reported
per regime. A pooled accuracy over regimes of unequal difficulty is uninterpretable -- the
same averaging mistake that, elsewhere in this project, made a length-matched gate read
backwards when it was pooled.

Regimes:
  denovo   every peptide position masked. PepMLM's own regime, and the search's "propose
           from scratch" round.
  budget   k ~ U{1..floor(L/2)} positions, uniformly placed. The search's edit regime;
           floor(L/2) is exactly `search.budget_for`, so a proposal drawn this way cannot
           exceed the budget the `Proposal` dataclass enforces.
  anchor   the cyclization anchors, plus optionally a few more positions. This is the
           LP->CP query: what a linear peptide lacks is the anchor identities, and
           project memory records isopeptide closure failure as ~50% anchor IDENTITY
           rather than geometry.
  span     one contiguous run. PepMLM masks spans, and a span is what a region being
           redesigned as a unit looks like.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from proteinfoundation.cyclization.constants import (
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
    UNSPECIFIED,
)

GAP_CHAR = "/"
GAP_TOKEN = "<gap>"

# One token per conditioning chemistry. LINEAR is deliberately absent: this corpus is
# cyclic peptides, and a token that never appears in training would be an untrained
# embedding handed out at inference as though it meant something.
CYC_TOKENS = {
    MAINCHAIN: "<cyc:mainchain>",
    DISULFIDE: "<cyc:disulfide>",
    ISOPEPTIDE: "<cyc:isopeptide>",
    UNSPECIFIED: "<cyc:unspecified>",
}

ADDED_TOKENS = [GAP_TOKEN, *CYC_TOKENS.values()]

AA20 = "ACDEFGHIKLMNPQRSTVWY"

REGIMES = ("denovo", "budget", "anchor", "span")


@dataclass
class MaskScheduleConfig:
    """Mixture weights over regimes, and the knobs inside each one.

    Defaults weight `budget` highest because that is the query the search actually makes;
    `denovo` is kept well represented because it is the round-0 query and because a model
    that has only ever filled 2 of 10 positions has not learned to write a peptide.
    """

    p_denovo: float = 0.25
    p_budget: float = 0.45
    p_anchor: float = 0.15
    p_span: float = 0.15
    # Extra non-anchor positions opened in the `anchor` regime, inclusive range. The LP->CP
    # edit is not only the anchors -- the residues beside them move too -- so a pure
    # two-position mask would train an unrealistically narrow query.
    anchor_extra_min: int = 0
    anchor_extra_max: int = 2
    # Span length as a fraction of L, inclusive range.
    span_frac_min: float = 0.2
    span_frac_max: float = 0.6
    # Probability a row's chemistry token is rewritten to `<cyc:unspecified>`. The
    # classifier-free null: without it the model cannot be queried for "any ring", and
    # `<cyc:unspecified>` rows in the corpus (unparsable CONECT) would be the only
    # examples it ever saw, which would make the null token mean "bad data".
    cyc_dropout: float = 0.15

    def weights(self) -> list[float]:
        w = [self.p_denovo, self.p_budget, self.p_anchor, self.p_span]
        total = sum(w)
        if total <= 0:
            raise ValueError("mask schedule weights sum to zero")
        return [x / total for x in w]


@dataclass
class CorpusRow:
    example_id: str
    pep_seq: str
    pocket_seq: str
    cyc_type: int
    cyc_anchor_i: int
    cyc_anchor_j: int
    cluster_id: str | None = None
    split: str | None = None
    extra: dict = field(default_factory=dict)


def assert_one_shard_grouping(paths: list[str | Path]) -> None:
    """Refuse a corpus directory holding two different shard counts for one split.

    Shard files are named `<split>_shard<i>of<N>.jsonl`, so re-running a split at a
    different N leaves BOTH groupings on disk under different names. They cover the same
    rows, and a glob would then load that split twice -- silently doubling it and, for a
    validation split, making every metric a measurement of a corpus that does not exist.
    Nothing about the duplicate is visible in a row, so this is checked by filename.
    """
    import re
    from collections import defaultdict

    groups: dict[str, set[str]] = defaultdict(set)
    for p in paths:
        m = re.match(r"(?P<split>.+)_shard\d+of(?P<n>\d+)\.jsonl$", Path(p).name)
        if m:
            groups[m.group("split")].add(m.group("n"))
    bad = {s: sorted(ns) for s, ns in groups.items() if len(ns) > 1}
    if bad:
        raise ValueError(
            f"corpus directory holds more than one shard grouping: {bad}. These cover the "
            f"same rows under different filenames, so loading them together would double "
            f"that split. Delete the grouping you do not want before training."
        )


def load_corpus(
    paths: list[str | Path],
    *,
    exclude_clusters: set[str] | None = None,
    min_len: int = 2,
) -> list[CorpusRow]:
    """Read JSONL shards into rows.

    A malformed line RAISES. It is not skipped: concurrent appends on this filesystem
    were measured to NUL-corrupt rows, and a reader that skips corruption turns "a third
    of my data vanished" into a silent accuracy change.
    """
    assert_one_shard_grouping(paths)
    rows: list[CorpusRow] = []
    dropped_cluster = 0
    for p in paths:
        with open(p) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{lineno} is not valid JSON ({e}). Refusing to "
                                     f"skip it -- see the corpus builder on shard output "
                                     f"corruption.") from e
                if exclude_clusters and d.get("cluster_id") in exclude_clusters:
                    dropped_cluster += 1
                    continue
                if len(d["pep_seq"]) < min_len:
                    continue
                rows.append(CorpusRow(
                    example_id=d["example_id"],
                    pep_seq=d["pep_seq"],
                    pocket_seq=d["pocket_seq"],
                    cyc_type=int(d["cyc_type"]),
                    cyc_anchor_i=int(d.get("cyc_anchor_i", -1)),
                    cyc_anchor_j=int(d.get("cyc_anchor_j", -1)),
                    cluster_id=d.get("cluster_id"),
                    split=d.get("split"),
                ))
    if dropped_cluster:
        print(f"[corpus] excluded {dropped_cluster} rows by cluster holdout")
    return rows


def budget_for(length: int) -> int:
    """`floor(L/2)`, identical to `cpseq.search.budget_for`.

    Duplicated rather than imported so this module stays importable without the search
    stack, but it must not diverge -- `test_cpseq_mlm.py` asserts they agree.
    """
    return length // 2


def draw_mask(
    row: CorpusRow,
    cfg: MaskScheduleConfig,
    rng: random.Random,
) -> tuple[list[int], str]:
    """Positions to hide, and the regime name that produced them.

    Always returns at least one position: an example with nothing hidden contributes no
    gradient and no metric, and in a sibling project half the training examples turned out
    to hide zero positions, which is how a sequence head came to be measured at chance
    without anyone noticing it was barely being trained.
    """
    L = len(row.pep_seq)
    regime = rng.choices(REGIMES, weights=cfg.weights(), k=1)[0]
    all_pos = list(range(L))

    if regime == "denovo":
        return all_pos, regime

    if regime == "budget":
        k = rng.randint(1, max(1, budget_for(L)))
        return sorted(rng.sample(all_pos, k)), regime

    if regime == "anchor":
        anchors = [p for p in (row.cyc_anchor_i, row.cyc_anchor_j) if 0 <= p < L]
        if not anchors:
            # No parsed anchor: fall back to the termini, which is where CPSea's
            # cyclization is -- the dataset is terminal-only -- rather than silently
            # degrading to a uniform draw and mislabelling the regime.
            anchors = sorted({0, L - 1})
        extra_budget = max(0, budget_for(L) - len(anchors))
        n_extra = min(
            extra_budget,
            rng.randint(cfg.anchor_extra_min, cfg.anchor_extra_max),
        )
        pool = [p for p in all_pos if p not in set(anchors)]
        extra = rng.sample(pool, min(n_extra, len(pool))) if pool else []
        return sorted(set(anchors) | set(extra)), regime

    # span
    frac = rng.uniform(cfg.span_frac_min, cfg.span_frac_max)
    span_len = max(1, min(L, round(frac * L)))
    start = rng.randint(0, L - span_len)
    return list(range(start, start + span_len)), regime


class EsmMlmEncoder:
    """Turns a `CorpusRow` plus a mask into token ids, labels and bookkeeping.

    Holds the tokenizer and the ids of the tokens this task adds, so nothing downstream
    re-derives a special-token id and gets a different answer.
    """

    def __init__(self, tokenizer, max_length: int = 1024, pocket_enabled: bool = True):
        self.tok = tokenizer
        self.max_length = max_length
        self.pocket_enabled = pocket_enabled
        self.mask_id = tokenizer.mask_token_id
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        missing = [t for t in ADDED_TOKENS if t not in tokenizer.get_vocab()]
        if missing:
            raise ValueError(
                f"tokenizer is missing {missing}; call `add_task_tokens(tokenizer, model)` "
                f"before building an encoder, and resize the model's embeddings with it."
            )
        self.gap_id = tokenizer.convert_tokens_to_ids(GAP_TOKEN)
        self.cyc_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in CYC_TOKENS.items()
        }
        self.aa_ids = {a: tokenizer.convert_tokens_to_ids(a) for a in AA20}
        unknown = [a for a, i in self.aa_ids.items() if i == tokenizer.unk_token_id]
        if unknown:
            raise ValueError(f"tokenizer maps {unknown} to <unk>; wrong tokenizer?")

    def residue_ids(self) -> list[int]:
        """Vocab ids of the 20 canonical residues, in `AA20` order.

        The proposer restricts its softmax to these: an ESM-2 head can emit `X`, `B`,
        `<mask>` or a special token, and a proposal containing one is not a peptide.
        """
        return [self.aa_ids[a] for a in AA20]

    def encode(
        self,
        row: CorpusRow,
        hidden: list[int],
        *,
        cyc_type: int | None = None,
    ) -> dict:
        """One example.

        `labels` is -100 everywhere except the hidden peptide positions, so the loss and
        every metric see ONLY positions whose answer was withheld. This is the single most
        important line in the file: scoring positions that were fed in as input produces
        a recovery near 1.0 that measures copying, which has twice been mistaken in this
        project for a model that worked.
        """
        cyc = row.cyc_type if cyc_type is None else cyc_type
        cyc_id = self.cyc_ids.get(cyc, self.cyc_ids[UNSPECIFIED])

        ids: list[int] = [self.cls_id]
        if self.pocket_enabled and row.pocket_seq:
            for ch in row.pocket_seq:
                if ch == GAP_CHAR:
                    ids.append(self.gap_id)
                else:
                    ids.append(self.aa_ids.get(ch, self.tok.unk_token_id))
            ids.append(self.eos_id)
        ids.append(cyc_id)

        pep_start = len(ids)
        hidden_set = set(hidden)
        labels = [-100] * pep_start
        for pos, ch in enumerate(row.pep_seq):
            true_id = self.aa_ids.get(ch, self.tok.unk_token_id)
            if pos in hidden_set:
                ids.append(self.mask_id)
                labels.append(true_id)
            else:
                ids.append(true_id)
                labels.append(-100)
        pep_end = len(ids)
        ids.append(self.eos_id)
        labels.append(-100)

        if len(ids) > self.max_length:
            # Truncating the pocket would change what the model was conditioned on
            # without saying so, so an over-long row is reported and dropped by the
            # caller instead.
            return {"too_long": True, "length": len(ids)}

        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": [1] * len(ids),
            "pep_start": pep_start,
            "pep_end": pep_end,
            "hidden": sorted(hidden_set),
            "too_long": False,
        }


def add_task_tokens(tokenizer, model=None) -> int:
    """Add `<gap>` and the `<cyc:*>` tokens, resizing the model's embeddings to match.

    Returns the number of tokens added. New rows are randomly initialised, which is
    correct here: there is no pretrained embedding that means "the receptor skips" or
    "close this head-to-tail", and seeding them from an existing residue would import a
    meaning the token does not have.

    ESM NEEDS ITS OUTPUT BIAS RESIZED BY HAND. `EsmLMHead` computes
    `self.decoder(x) + self.bias`, where `bias` is a standalone `nn.Parameter` of size
    vocab, NOT the decoder Linear's own bias. `resize_token_embeddings` only resizes the
    input embedding and the output embedding it can find via `get_output_embeddings()`,
    so that separate parameter is left at the old width and the first forward dies with

        RuntimeError: The size of tensor a (38) must match the size of tensor b (33)

    Measured on `facebook/esm2_t12_35M_UR50D`. The new entries are zero-initialised,
    which is the neutral prior for a token the model has never emitted -- and it never
    should emit these, since they are input-only markers.
    """
    import torch

    n = tokenizer.add_special_tokens({"additional_special_tokens": ADDED_TOKENS})
    if model is not None and n > 0:
        new_vocab = len(tokenizer)
        model.resize_token_embeddings(new_vocab)
        head = getattr(model, "lm_head", None)
        for owner, attr in ((head, "bias"), (getattr(head, "decoder", None), "bias")):
            if owner is None:
                continue
            b = getattr(owner, attr, None)
            if b is None or b.dim() != 1 or b.numel() == new_vocab:
                continue
            grown = torch.zeros(new_vocab, dtype=b.dtype, device=b.device)
            grown[: b.numel()] = b.data
            setattr(owner, attr, torch.nn.Parameter(grown))
    return n


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad to the batch max. `labels` pads with -100 so padding is never scored."""
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    out = {
        "input_ids": [], "attention_mask": [], "labels": [],
        "pep_start": [], "pep_end": [],
    }
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * k)
        out["attention_mask"].append(b["attention_mask"] + [0] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["pep_start"].append(b["pep_start"])
        out["pep_end"].append(b["pep_end"])
    return {
        "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(out["labels"], dtype=torch.long),
        "pep_start": torch.tensor(out["pep_start"], dtype=torch.long),
        "pep_end": torch.tensor(out["pep_end"], dtype=torch.long),
    }


# --------------------------------------------------------------------------------------
# Baselines. A model's hidden-position accuracy means nothing on its own.
# --------------------------------------------------------------------------------------


def amino_acid_marginal(rows: list[CorpusRow]) -> dict[str, float]:
    """P(residue) over every peptide position in the corpus."""
    counts: dict[str, int] = {a: 0 for a in AA20}
    total = 0
    for r in rows:
        for ch in r.pep_seq:
            if ch in counts:
                counts[ch] += 1
                total += 1
    return {a: c / max(total, 1) for a, c in counts.items()}


def positional_marginal(rows: list[CorpusRow]) -> dict[tuple[int, int], dict[str, float]]:
    """P(residue | position, length).

    The baseline that actually threatens a conditioned model. CPSea cyclises at the
    termini, so position 0 and L-1 are anchor-enriched -- a model that has learned only
    "CYS at the ends, hydrophobic in the middle" will score well above the flat marginal
    while knowing nothing about any pocket. If the trained model does not beat THIS, it
    has not learned the task.
    """
    counts: dict[tuple[int, int], dict[str, int]] = {}
    for r in rows:
        L = len(r.pep_seq)
        for pos, ch in enumerate(r.pep_seq):
            if ch not in AA20:
                continue
            d = counts.setdefault((pos, L), {})
            d[ch] = d.get(ch, 0) + 1
    out: dict[tuple[int, int], dict[str, float]] = {}
    for key, d in counts.items():
        tot = sum(d.values())
        out[key] = {a: c / tot for a, c in d.items()}
    return out


def baseline_accuracy(
    rows: list[CorpusRow],
    masks: list[list[int]],
    marginal: dict[str, float],
    positional: dict[tuple[int, int], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Accuracy of the always-argmax baselines on exactly the hidden positions given."""
    best_global = max(marginal, key=marginal.get) if marginal else "L"
    n = 0
    hit_global = 0
    hit_pos = 0
    for row, hidden in zip(rows, masks):
        L = len(row.pep_seq)
        for p in hidden:
            truth = row.pep_seq[p]
            n += 1
            if truth == best_global:
                hit_global += 1
            if positional is not None:
                d = positional.get((p, L))
                if d:
                    if truth == max(d, key=d.get):
                        hit_pos += 1
                elif truth == best_global:
                    hit_pos += 1
    return {
        "n_hidden": n,
        "marginal_argmax_acc": hit_global / max(n, 1),
        "positional_argmax_acc": hit_pos / max(n, 1),
        "marginal_argmax_residue": best_global,
    }
