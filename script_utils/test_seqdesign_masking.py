"""Tests for `proteinfoundation.seqdesign.masking`.

Two jobs. First, the structural invariants of each regime -- a regime that quietly
degrades to a uniform draw still produces a plausible-looking mask, so the label has to be
checked against the shape of what it produced. Second, AGREEMENT WITH THE pLM ARM: this
module duplicates `script_utils/cpseq_mlm_data`'s schedule so the batched training step
does not have to call per-row Python, and the whole comparability claim of the ladder
rests on that duplication not drifting. The defaults and `budget_for` are asserted equal
rather than assumed.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proteinfoundation.seqdesign.masking import (  # noqa: E402
    REGIMES,
    MaskScheduleConfig,
    budget_for,
    draw_hidden_mask,
    freeze_panel,
    panel_to_tensors,
    read_panel,
    write_panel,
)


def _only(regime: str) -> MaskScheduleConfig:
    """A schedule that draws `regime` with probability 1, so invariants are testable."""
    kw = {f"p_{r}": (1.0 if r == regime else 0.0) for r in REGIMES}
    return MaskScheduleConfig(**kw)


def _pad(lengths: list[int], n: int | None = None) -> torch.Tensor:
    n = n or max(lengths)
    pad = torch.zeros(len(lengths), n, dtype=torch.bool)
    for i, L in enumerate(lengths):
        pad[i, :L] = True
    return pad


LENGTHS = [5, 8, 12, 16, 7, 3]


# --------------------------------------------------------------------------- regimes


def test_denovo_hides_every_real_position():
    pad = _pad(LENGTHS)
    hidden, regime = draw_hidden_mask(pad, _only("denovo"))
    assert torch.equal(hidden, pad)
    assert (regime == REGIMES.index("denovo")).all()


def test_budget_respects_floor_half_l():
    pad = _pad(LENGTHS)
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        hidden, _ = draw_hidden_mask(pad, _only("budget"), generator=g)
        k = hidden.sum(dim=1)
        lengths = pad.sum(dim=1)
        assert (k >= 1).all(), "budget must hide at least one position"
        assert (k <= torch.clamp(lengths // 2, min=1)).all(), f"budget exceeded floor(L/2): {k}"


def test_anchor_always_contains_the_anchor_pair():
    pad = _pad(LENGTHS)
    anchors = torch.tensor([[0, L - 1] for L in LENGTHS], dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        hidden, _ = draw_hidden_mask(pad, _only("anchor"), anchors=anchors, generator=g)
        for i, L in enumerate(LENGTHS):
            assert hidden[i, 0], f"row {i}: anchor 0 not hidden"
            assert hidden[i, L - 1], f"row {i}: anchor {L - 1} not hidden"


def test_anchor_falls_back_to_termini_not_to_a_uniform_draw():
    """A row with no parsed anchor must still produce an ANCHOR-shaped mask."""
    pad = _pad(LENGTHS)
    anchors = torch.full((len(LENGTHS), 2), -1, dtype=torch.long)
    g = torch.Generator().manual_seed(2)
    for _ in range(30):
        hidden, _ = draw_hidden_mask(pad, _only("anchor"), anchors=anchors, generator=g)
        for i, L in enumerate(LENGTHS):
            assert hidden[i, 0] and hidden[i, L - 1], "fallback must hide both termini"


def test_anchor_extra_stays_within_budget():
    pad = _pad(LENGTHS)
    anchors = torch.tensor([[0, L - 1] for L in LENGTHS], dtype=torch.long)
    cfg = _only("anchor")
    g = torch.Generator().manual_seed(3)
    for _ in range(50):
        hidden, _ = draw_hidden_mask(pad, cfg, anchors=anchors, generator=g)
        for i, L in enumerate(LENGTHS):
            n_hidden = int(hidden[i].sum())
            # 2 anchors (or 1 when L == 1) plus at most anchor_extra_max extras, and the
            # extras are additionally capped by floor(L/2) - n_anchors.
            assert n_hidden <= 2 + cfg.anchor_extra_max


def test_span_is_contiguous():
    pad = _pad(LENGTHS)
    g = torch.Generator().manual_seed(4)
    for _ in range(50):
        hidden, _ = draw_hidden_mask(pad, _only("span"), generator=g)
        for i, L in enumerate(LENGTHS):
            pos = torch.nonzero(hidden[i], as_tuple=False).flatten().tolist()
            assert pos, "span hid nothing"
            assert pos == list(range(pos[0], pos[-1] + 1)), f"span not contiguous: {pos}"
            assert pos[-1] < L, "span ran into padding"


def test_span_length_tracks_the_configured_fraction():
    lengths = [20] * 64
    pad = _pad(lengths)
    cfg = _only("span")
    g = torch.Generator().manual_seed(5)
    hidden, _ = draw_hidden_mask(pad, cfg, generator=g)
    n_hidden = hidden.sum(dim=1).float()
    lo = round(cfg.span_frac_min * 20)
    hi = round(cfg.span_frac_max * 20)
    assert (n_hidden >= lo).all() and (n_hidden <= hi).all(), f"span lengths {n_hidden.unique()}"


# --------------------------------------------------------------------------- guarantees


def test_padding_is_never_hidden():
    pad = _pad(LENGTHS, n=24)
    g = torch.Generator().manual_seed(6)
    for _ in range(40):
        hidden, _ = draw_hidden_mask(pad, MaskScheduleConfig(), generator=g)
        assert not bool((hidden & ~pad).any()), "hid a padded position"


def test_never_returns_an_all_visible_row():
    """The lesson this guarantee exists for: a row hiding nothing contributes no gradient
    and no metric, and a head trained mostly on such rows measures at chance."""
    pad = _pad([1, 2, 3, 4, 5, 6])
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        hidden, _ = draw_hidden_mask(pad, MaskScheduleConfig(), generator=g)
        assert (hidden.sum(dim=1) >= 1).all(), "produced an all-visible row"


def test_single_residue_rows_are_handled():
    pad = _pad([1, 1, 1])
    for r in REGIMES:
        hidden, _ = draw_hidden_mask(pad, _only(r))
        assert (hidden.sum(dim=1) == 1).all(), f"{r} failed on L=1"


def test_generator_makes_the_draw_reproducible():
    pad = _pad(LENGTHS)
    a, ra = draw_hidden_mask(pad, MaskScheduleConfig(), generator=torch.Generator().manual_seed(11))
    b, rb = draw_hidden_mask(pad, MaskScheduleConfig(), generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b) and torch.equal(ra, rb)


def test_all_four_regimes_are_reachable_under_default_weights():
    pad = _pad(LENGTHS)
    g = torch.Generator().manual_seed(12)
    seen = set()
    for _ in range(200):
        _, regime = draw_hidden_mask(pad, MaskScheduleConfig(), generator=g)
        seen.update(regime.tolist())
    assert seen == set(range(len(REGIMES))), f"unreachable regimes: {set(range(4)) - seen}"


# ------------------------------------------------------- agreement with the pLM arm


def test_budget_for_agrees_with_the_plm_arm():
    import cpseq_mlm_data

    for L in range(1, 40):
        assert int(budget_for(torch.tensor(L))) == cpseq_mlm_data.budget_for(L), f"L={L}"


def test_schedule_defaults_match_the_plm_arm():
    """If these drift, the pLM arm and the Proteina arms are answering different queries
    and their recovery numbers stop being comparable."""
    import cpseq_mlm_data

    ours = asdict(MaskScheduleConfig())
    theirs = asdict(cpseq_mlm_data.MaskScheduleConfig())
    shared = set(ours) & set(theirs)
    assert "p_denovo" in shared and "span_frac_max" in shared, "schedules lost their shared keys"
    for k in sorted(shared):
        assert ours[k] == theirs[k], f"{k}: ours={ours[k]} plm={theirs[k]}"


def test_regime_names_match_the_plm_arm():
    import cpseq_mlm_data

    assert REGIMES == cpseq_mlm_data.REGIMES


# --------------------------------------------------------------------------- panel


def _examples():
    return [
        {"example_id": f"ex{i}", "length": L, "anchors": (0, L - 1), "cyc_type": "mainchain"}
        for i, L in enumerate(LENGTHS)
    ]


def test_panel_is_deterministic_for_a_seed():
    cfg = MaskScheduleConfig()
    a = freeze_panel(_examples(), cfg, seed=99)
    b = freeze_panel(_examples(), cfg, seed=99)
    assert [r.hidden for r in a] == [r.hidden for r in b]
    assert [r.regime for r in a] == [r.regime for r in b]


def test_panel_round_trips_through_disk(tmp_path):
    cfg = MaskScheduleConfig()
    rows = freeze_panel(_examples(), cfg, seed=5)
    p = tmp_path / "panel.jsonl"
    write_panel(rows, p, cfg, seed=5)
    back, header = read_panel(p)
    assert [asdict(r) for r in back] == [asdict(r) for r in rows]
    assert header["seed"] == 5
    assert header["schedule"]["p_budget"] == cfg.p_budget


def test_read_panel_rejects_a_panel_with_no_provenance(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"example_id": "x", "length": 4, "hidden": [0], "regime": "budget"}\n')
    with pytest.raises(ValueError, match="no _header"):
        read_panel(p)


def test_read_panel_rejects_a_truncated_panel(tmp_path):
    cfg = MaskScheduleConfig()
    rows = freeze_panel(_examples(), cfg, seed=5)
    p = tmp_path / "panel.jsonl"
    write_panel(rows, p, cfg, seed=5)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="header says"):
        read_panel(p)


def test_panel_to_tensors_restores_the_exact_positions():
    cfg = MaskScheduleConfig()
    rows = freeze_panel(_examples(), cfg, seed=7)
    n = max(LENGTHS)
    hidden, regime = panel_to_tensors(rows, n)
    for i, r in enumerate(rows):
        assert torch.nonzero(hidden[i], as_tuple=False).flatten().tolist() == r.hidden
        assert REGIMES[int(regime[i])] == r.regime


def test_panel_to_tensors_refuses_a_row_wider_than_the_batch():
    cfg = MaskScheduleConfig()
    rows = freeze_panel(_examples(), cfg, seed=7)
    with pytest.raises(ValueError, match="padded width"):
        panel_to_tensors(rows, n=2)


def test_panel_hidden_positions_are_within_length():
    cfg = MaskScheduleConfig()
    rows = freeze_panel(_examples(), cfg, seed=13)
    for r in rows:
        assert r.hidden, f"{r.example_id} hid nothing"
        assert max(r.hidden) < r.length, f"{r.example_id} hid past its length"
