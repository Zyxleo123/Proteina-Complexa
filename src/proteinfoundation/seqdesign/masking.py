"""Which peptide positions are hidden, for every arm of the sequence-design ladder.

ONE masking module for all of them. The ladder compares a PepMLM-based proposer, a
Proteina-architecture sequence head, and the flow co-generator; those three were
previously trained and scored against *different* hidden-position distributions, which
makes their recovery numbers incomparable in a way that no amount of eval care fixes
afterwards:

  * `script_utils/cpseq_mlm_data.draw_mask` draws one of four structured regimes
    (denovo / budget / anchor / span) -- the queries the search actually makes.
  * `proteinfoundation.utils.seq_conditioning.sample_sequence_conditioning_mask` draws a
    per-example keep fraction and then an independent Bernoulli per residue.

A Bernoulli reveal NEVER produces a contiguous hidden run, so the `span` regime -- the
hardest one, and the one the pLM contributes most on (+0.056, versus +0.029 on `anchor`)
-- simply does not occur in the co-generator's training distribution. Scoring the two on
a shared panel while they were optimised for different queries measures the mismatch, not
the models. So this module is the single source of hidden positions for TRAINING and for
EVALUATION, across every arm.

The regimes and their defaults mirror `cpseq_mlm_data` exactly, so the already-trained
pLM arm stays comparable; `test_seqdesign_masking.py` asserts the two agree on the
structural invariants rather than trusting the duplication. The difference here is that
this implementation is batched over a padded `[b, n]` tensor instead of per-row Python,
because it runs inside the training step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REGIMES = ("denovo", "budget", "anchor", "span")
REGIME_INDEX = {name: i for i, name in enumerate(REGIMES)}


@dataclass
class MaskScheduleConfig:
    """Mixture weights over regimes, and the knobs inside each one.

    Defaults are `cpseq_mlm_data.MaskScheduleConfig`'s, deliberately: changing them here
    would silently break comparability with the trained pLM arm. `budget` is weighted
    highest because it is the query the search makes; `denovo` stays well represented
    because it is the round-0 query, and a model that has only ever filled 2 of 10
    positions has not learned to write a peptide.
    """

    p_denovo: float = 0.25
    p_budget: float = 0.45
    p_anchor: float = 0.15
    p_span: float = 0.15
    # Extra non-anchor positions opened in the `anchor` regime, inclusive range. The
    # LP->CP edit is not only the anchors -- the residues beside them move too -- so a
    # pure two-position mask would train an unrealistically narrow query.
    anchor_extra_min: int = 0
    anchor_extra_max: int = 2
    # Span length as a fraction of L, inclusive range.
    span_frac_min: float = 0.2
    span_frac_max: float = 0.6

    def weights(self) -> list[float]:
        w = [self.p_denovo, self.p_budget, self.p_anchor, self.p_span]
        total = sum(w)
        if total <= 0:
            raise ValueError("mask schedule weights sum to zero")
        return [x / total for x in w]


def budget_for(length: torch.Tensor) -> torch.Tensor:
    """`floor(L/2)`, matching `cpseq.search.budget_for` and `cpseq_mlm_data.budget_for`.

    Kept as a function (rather than inlined) so the three definitions can be asserted
    equal in tests; they must not drift.
    """
    return torch.div(length, 2, rounding_mode="floor")


def _rank_within_row(scores: torch.Tensor) -> torch.Tensor:
    """Rank of each column within its row, 0 = smallest. Used to pick k-of-n without
    replacement in a batched way: `rank < k` selects exactly `k` positions per row."""
    return scores.argsort(dim=1).argsort(dim=1)


def draw_hidden_mask(
    pad_mask: torch.Tensor,
    cfg: MaskScheduleConfig,
    anchors: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw which positions to HIDE.

    Args:
        pad_mask: [b, n] bool, True on real residues. Peptide only -- target tokens are
            concatenated downstream and are never masked.
        cfg: mixture weights and per-regime knobs.
        anchors: optional [b, 2] long, the cyclization bond endpoints, -1 where unknown.
            CPSea is terminal-only (measured: 27k raw structures all cyclize at
            `(0, L-1)`), so the fallback below is almost always the true anchor pair --
            but falling back silently to a UNIFORM draw would mislabel the regime, which
            is the failure `cpseq_mlm_data.draw_mask` calls out.
        generator: optional RNG for reproducibility; the frozen eval panel uses one.

    Returns:
        hidden: [b, n] bool, True where the residue identity is withheld.
        regime: [b] long, index into `REGIMES`, for per-regime metrics. Pooling accuracy
            over regimes of unequal difficulty is uninterpretable -- measured spread under
            one set of weights is perplexity 4.14 (anchor) vs 14.14 (span) -- so every
            consumer of this function must keep the regime label with the prediction.

    Guarantees at least one hidden position per non-empty row. An example with nothing
    hidden contributes no gradient and no metric; in a sibling project half the training
    examples hid zero positions, and a sequence head was measured at chance for weeks
    before anyone noticed it was barely being trained.
    """
    pad_mask = pad_mask.bool()
    b, n = pad_mask.shape
    device = pad_mask.device

    def _rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    lengths = pad_mask.sum(dim=1)  # [b]
    idx = torch.arange(n, device=device).expand(b, n)  # [b, n]

    w = torch.tensor(cfg.weights(), device=device, dtype=torch.float32)
    regime = torch.multinomial(w.expand(b, len(REGIMES)), 1, generator=generator).squeeze(1)

    # Random ranks over real positions only; padding is pushed to the end so it is never
    # selected by a `rank < k` test.
    scores = torch.where(pad_mask, _rand(b, n), torch.full((b, n), float("inf"), device=device))
    ranks = _rank_within_row(scores)

    # --- denovo: everything real.
    hid_denovo = pad_mask

    # --- budget: k ~ U{1..floor(L/2)} positions, uniformly placed.
    max_k = torch.clamp(budget_for(lengths), min=1)
    k = (torch.floor(_rand(b) * max_k.float()) + 1).long()  # U{1..max_k}
    k = torch.minimum(k, torch.clamp(lengths, min=1))
    hid_budget = ranks < k[:, None]

    # --- anchor: the cyclization endpoints, plus a few neighbours.
    if anchors is None:
        anchor_mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    else:
        anchor_mask = torch.zeros(b, n, dtype=torch.bool, device=device)
        a = anchors.to(device=device, dtype=torch.long)
        for col in range(a.shape[1]):
            v = a[:, col]
            ok = (v >= 0) & (v < n)
            anchor_mask |= ok[:, None] & (idx == v.clamp(min=0)[:, None])
        anchor_mask &= pad_mask
    # Rows with no usable anchor fall back to the termini, which is where CPSea's
    # cyclization actually is.
    no_anchor = ~anchor_mask.any(dim=1)
    termini = (idx == 0) | (idx == (lengths - 1).clamp(min=0)[:, None])
    anchor_mask = torch.where(no_anchor[:, None], termini & pad_mask, anchor_mask)

    n_anchor = anchor_mask.sum(dim=1)
    extra_budget = torch.clamp(budget_for(lengths) - n_anchor, min=0)
    span_extra = cfg.anchor_extra_max - cfg.anchor_extra_min + 1
    n_extra = cfg.anchor_extra_min + torch.floor(_rand(b) * span_extra).long()
    n_extra = torch.minimum(n_extra, extra_budget)
    # Re-rank with anchors excluded, so `extra` never lands on a position already hidden
    # and the requested count is actually delivered.
    scores_na = torch.where(
        pad_mask & ~anchor_mask, _rand(b, n), torch.full((b, n), float("inf"), device=device)
    )
    hid_anchor = anchor_mask | (_rank_within_row(scores_na) < n_extra[:, None])

    # --- span: one contiguous run, length a fraction of L.
    frac = cfg.span_frac_min + (cfg.span_frac_max - cfg.span_frac_min) * _rand(b)
    span_len = torch.round(frac * lengths.float()).long().clamp(min=1)
    span_len = torch.minimum(span_len, torch.clamp(lengths, min=1))
    start_hi = (lengths - span_len).clamp(min=0) + 1  # U{0 .. L-span_len}
    start = torch.floor(_rand(b) * start_hi.float()).long()
    hid_span = (idx >= start[:, None]) & (idx < (start + span_len)[:, None]) & pad_mask

    stacked = torch.stack([hid_denovo, hid_budget, hid_anchor, hid_span], dim=0)  # [4, b, n]
    hidden = stacked.gather(0, regime[None, :, None].expand(1, b, n)).squeeze(0)
    hidden = hidden & pad_mask

    # Guarantee: never return an all-visible row. Hide the first real position instead.
    empty = (~hidden.any(dim=1)) & pad_mask.any(dim=1)
    if bool(empty.any()):
        first_real = pad_mask.float().argmax(dim=1)
        hidden = hidden | (empty[:, None] & (idx == first_real[:, None]))

    return hidden, regime


# ---------------------------------------------------------------------------
# Frozen evaluation panel
#
# Every arm must be scored on the SAME hidden positions. Drawing the mask inside each
# arm's eval -- even from the same config and seed -- does not achieve that: batch order,
# world size and dataloader workers all perturb the RNG stream, so two arms silently score
# different positions and the difference is read as a model difference. The panel is
# therefore drawn once, written to disk, and loaded by every arm including the flow
# co-generator reference.
# ---------------------------------------------------------------------------


@dataclass
class PanelRow:
    example_id: str
    length: int
    hidden: list[int]
    regime: str
    cyc_type: str | None = None


def freeze_panel(
    examples: list[dict],
    cfg: MaskScheduleConfig,
    seed: int,
    device: torch.device | str = "cpu",
) -> list[PanelRow]:
    """Draw the hidden positions once, for a fixed list of examples.

    `examples` entries need `example_id` and `length`, and may carry `anchors` (a pair of
    ints) and `cyc_type`. Returns one row per example, in input order.
    """
    if not examples:
        return []
    device = torch.device(device)
    n = max(int(e["length"]) for e in examples)
    b = len(examples)

    pad = torch.zeros(b, n, dtype=torch.bool, device=device)
    anchors = torch.full((b, 2), -1, dtype=torch.long, device=device)
    for i, e in enumerate(examples):
        L = int(e["length"])
        if L <= 0:
            raise ValueError(f"example {e.get('example_id')!r} has length {L}")
        pad[i, :L] = True
        a = e.get("anchors")
        if a is not None:
            anchors[i, 0], anchors[i, 1] = int(a[0]), int(a[1])

    g = torch.Generator(device=device).manual_seed(seed)
    hidden, regime = draw_hidden_mask(pad, cfg, anchors=anchors, generator=g)

    rows = []
    for i, e in enumerate(examples):
        rows.append(
            PanelRow(
                example_id=str(e["example_id"]),
                length=int(e["length"]),
                hidden=torch.nonzero(hidden[i], as_tuple=False).flatten().tolist(),
                regime=REGIMES[int(regime[i])],
                cyc_type=e.get("cyc_type"),
            )
        )
    return rows


def write_panel(rows: list[PanelRow], path: str | Path, cfg: MaskScheduleConfig, seed: int) -> None:
    """Write the panel as JSONL, with the draw's provenance on the first line.

    The header is not decoration: a panel whose schedule or seed is unknown cannot be
    regenerated, and a silently regenerated panel is exactly how two arms end up scored on
    different positions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"_header": {"schedule": asdict(cfg), "seed": seed, "n_rows": len(rows)}}) + "\n")
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def read_panel(path: str | Path) -> tuple[list[PanelRow], dict]:
    """Read a panel written by `write_panel`. Returns (rows, header)."""
    rows: list[PanelRow] = []
    header: dict = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_header" in obj:
                header = obj["_header"]
                continue
            rows.append(PanelRow(**obj))
    if not header:
        raise ValueError(f"{path}: no _header line; cannot verify the panel's provenance")
    if header.get("n_rows") != len(rows):
        raise ValueError(f"{path}: header says {header.get('n_rows')} rows, found {len(rows)}")
    return rows, header


def panel_to_tensors(
    rows: list[PanelRow], n: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a panel slice as `(hidden [b, n] bool, regime [b] long)`."""
    b = len(rows)
    device = torch.device(device)
    hidden = torch.zeros(b, n, dtype=torch.bool, device=device)
    regime = torch.zeros(b, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        if r.length > n:
            raise ValueError(f"panel row {r.example_id} has length {r.length} > padded width {n}")
        for p in r.hidden:
            hidden[i, p] = True
        regime[i] = REGIME_INDEX[r.regime]
    return hidden, regime
