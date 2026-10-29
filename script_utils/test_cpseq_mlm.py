"""Tests for the CPSea sequence-proposer data path.

The properties checked here are the ones whose failure is SILENT -- a model would still
train, the loss would still fall, and the reported accuracy would be wrong. In particular:
a revealed position must never be scored (that is the copy-metric trap this project has
hit twice), and the budget regime must never exceed `cpseq.search.budget_for`, because
the `Proposal` dataclass rejects an over-budget sequence at the far end of the pipeline
where the cause is no longer visible.

Run:
    .venv/bin/python -m pytest script_utils/test_cpseq_mlm.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "script_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cpseq_mlm_data import (  # noqa: E402
    AA20,
    GAP_CHAR,
    REGIMES,
    CorpusRow,
    MaskScheduleConfig,
    amino_acid_marginal,
    baseline_accuracy,
    budget_for,
    draw_mask,
    positional_marginal,
)
from proteinfoundation.cyclization.constants import (  # noqa: E402
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)


def _row(pep="ACDEFGHIKL", pocket="MNPQ/RST", cyc=MAINCHAIN, i=0, j=None):
    return CorpusRow(
        example_id="x", pep_seq=pep, pocket_seq=pocket, cyc_type=cyc,
        cyc_anchor_i=i, cyc_anchor_j=(len(pep) - 1 if j is None else j),
    )


# --------------------------------------------------------------------------------------
# Mask schedule
# --------------------------------------------------------------------------------------


def test_budget_matches_search_definition():
    """Divergence here is how an over-budget proposal reaches `Proposal.__post_init__`."""
    from proteinfoundation.cpseq.search import budget_for as search_budget

    for L in range(2, 40):
        assert budget_for(L) == search_budget(L), L


def test_budget_regime_never_exceeds_budget():
    cfg = MaskScheduleConfig(p_denovo=0, p_budget=1, p_anchor=0, p_span=0)
    rng = random.Random(0)
    for L in range(4, 17):
        row = _row(pep="A" * L)
        for _ in range(200):
            hidden, regime = draw_mask(row, cfg, rng)
            assert regime == "budget"
            assert 1 <= len(hidden) <= budget_for(L), (L, hidden)
            assert len(set(hidden)) == len(hidden)
            assert all(0 <= p < L for p in hidden)


def test_every_regime_hides_at_least_one_position():
    """An example with nothing hidden contributes no gradient and no metric. In a sibling
    project half the training examples hid zero positions and the sequence head was then
    measured at chance without anyone noticing it was barely supervised."""
    rng = random.Random(1)
    for regime in REGIMES:
        cfg = MaskScheduleConfig(
            p_denovo=1.0 if regime == "denovo" else 0.0,
            p_budget=1.0 if regime == "budget" else 0.0,
            p_anchor=1.0 if regime == "anchor" else 0.0,
            p_span=1.0 if regime == "span" else 0.0,
        )
        for L in range(2, 17):
            row = _row(pep="A" * L)
            for _ in range(50):
                hidden, got = draw_mask(row, cfg, rng)
                assert got == regime
                assert len(hidden) >= 1, (regime, L)


def test_denovo_masks_everything():
    cfg = MaskScheduleConfig(p_denovo=1, p_budget=0, p_anchor=0, p_span=0)
    row = _row(pep="ACDEFG")
    hidden, _ = draw_mask(row, cfg, random.Random(0))
    assert hidden == [0, 1, 2, 3, 4, 5]


def test_anchor_regime_includes_the_anchors():
    cfg = MaskScheduleConfig(p_denovo=0, p_budget=0, p_anchor=1, p_span=0,
                             anchor_extra_min=0, anchor_extra_max=0)
    row = _row(pep="ACDEFGHIKL", cyc=ISOPEPTIDE, i=2, j=7)
    for seed in range(20):
        hidden, regime = draw_mask(row, cfg, random.Random(seed))
        assert regime == "anchor"
        assert 2 in hidden and 7 in hidden, hidden


def test_anchor_regime_falls_back_to_termini_when_unparsed():
    """A row whose CONECT could not be classified still has to produce the LP->CP query;
    CPSea cyclises terminally, so the termini are the right fallback."""
    cfg = MaskScheduleConfig(p_denovo=0, p_budget=0, p_anchor=1, p_span=0,
                             anchor_extra_min=0, anchor_extra_max=0)
    row = _row(pep="ACDEFG", i=-1, j=-1)
    hidden, _ = draw_mask(row, cfg, random.Random(0))
    assert 0 in hidden and 5 in hidden


def test_span_regime_is_contiguous():
    cfg = MaskScheduleConfig(p_denovo=0, p_budget=0, p_anchor=0, p_span=1)
    rng = random.Random(3)
    for L in range(4, 17):
        row = _row(pep="A" * L)
        for _ in range(50):
            hidden, _ = draw_mask(row, cfg, rng)
            assert hidden == list(range(hidden[0], hidden[-1] + 1)), hidden
            assert hidden[-1] < L


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


def test_mixed_shard_groupings_are_refused():
    """Re-running a split at a different shard count leaves both groupings on disk under
    different names, covering the same rows. A glob would load that split twice, which for
    a validation split makes every metric describe a corpus that does not exist."""
    from cpseq_mlm_data import assert_one_shard_grouping

    assert_one_shard_grouping(["val_shard0000of0004.jsonl", "val_shard0001of0004.jsonl"])
    assert_one_shard_grouping(
        ["val_shard0000of0004.jsonl", "train_shard0000of0016.jsonl"]
    )  # different splits may differ
    with pytest.raises(ValueError, match="more than one shard grouping"):
        assert_one_shard_grouping(
            ["val_shard0000of0004.jsonl", "val_shard0000of0032.jsonl"]
        )


def test_chemistry_aware_prior_is_at_least_as_strong_as_the_positional_one():
    """The model is HANDED the chemistry token, so a prior without it is handicapped and
    the margin measured against it is inflated. Measured on the real panel, quoting the
    chemistry-free prior overstated this arm's margin by 60% (+0.0633 vs +0.0393).

    Constructed so chemistry is the ONLY thing that disambiguates: at position 0 a
    mainchain peptide starts with A and a disulfide one with C, so P(r|pos,L) is at chance
    between them while P(r|pos,L,cyc) is exact."""
    from cpseq_mlm_data import positional_chemistry_marginal

    rows = ([_row(pep="AAAA", cyc=MAINCHAIN) for _ in range(50)]
            + [_row(pep="CAAA", cyc=DISULFIDE) for _ in range(50)])
    masks = [[0] for _ in rows]
    marg = amino_acid_marginal(rows)
    posm = positional_marginal(rows)
    poscm = positional_chemistry_marginal(rows)
    b = baseline_accuracy(rows, masks, marg, posm, poscm)
    assert b["positional_chem_argmax_acc"] == pytest.approx(1.0)
    assert b["positional_argmax_acc"] == pytest.approx(0.5)
    assert b["positional_chem_argmax_acc"] >= b["positional_argmax_acc"]


def test_chemistry_prior_backs_off_on_unseen_cells():
    """A (pos, L, cyc) cell absent from train must fall back to (pos, L), not score a free
    miss -- otherwise the fair baseline is understated on rare length/chemistry pairs and
    the model's margin gets re-inflated by the very fix meant to remove the bias."""
    from cpseq_mlm_data import positional_chemistry_marginal

    train = [_row(pep="AAAA", cyc=MAINCHAIN) for _ in range(20)]
    poscm = positional_chemistry_marginal(train)
    posm = positional_marginal(train)
    marg = amino_acid_marginal(train)
    # scored row carries a chemistry the prior never saw
    unseen = [_row(pep="AAAA", cyc=ISOPEPTIDE)]
    b = baseline_accuracy(unseen, [[0]], marg, posm, poscm)
    assert b["positional_chem_argmax_acc"] == pytest.approx(1.0)


def test_marginal_sums_to_one_and_counts_only_residues():
    rows = [_row(pep="AAAC"), _row(pep="AAAA")]
    m = amino_acid_marginal(rows)
    assert pytest.approx(sum(m.values()), abs=1e-9) == 1.0
    assert m["A"] == pytest.approx(7 / 8)
    assert m["C"] == pytest.approx(1 / 8)


def test_positional_prior_beats_flat_marginal_on_terminal_anchors():
    """The point of carrying a positional baseline: a corpus that always puts C at both
    ends is perfectly predicted by position alone, so a model that only learned that
    would score high against the flat marginal and have learned nothing about a pocket."""
    rows = [_row(pep="C" + "AAAA"[: 4] + "C") for _ in range(50)]
    masks = [[0, len(r.pep_seq) - 1] for r in rows]
    marg = amino_acid_marginal(rows)
    posm = positional_marginal(rows)
    b = baseline_accuracy(rows, masks, marg, posm)
    assert b["positional_argmax_acc"] == pytest.approx(1.0)
    assert b["marginal_argmax_acc"] < 0.5
    assert b["n_hidden"] == 100


# --------------------------------------------------------------------------------------
# Encoder. These need the tokenizer, so they are skipped without transformers.
# --------------------------------------------------------------------------------------


def _encoder(pocket_enabled=True):
    transformers = pytest.importorskip("transformers")
    from cpseq_mlm_data import add_task_tokens

    try:
        tok = transformers.AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    except Exception as e:  # offline / no cache
        pytest.skip(f"tokenizer unavailable: {e}")
    add_task_tokens(tok, None)
    from cpseq_mlm_data import EsmMlmEncoder

    return EsmMlmEncoder(tok, pocket_enabled=pocket_enabled), tok


def test_only_hidden_positions_are_labelled():
    """THE load-bearing assertion. A revealed position carrying a label is the copy
    metric: recovery goes to ~1.0 and measures nothing."""
    enc_obj, tok = _encoder()
    row = _row(pep="ACDEFGHIKL")
    hidden = [1, 4, 9]
    out = enc_obj.encode(row, hidden)
    assert not out["too_long"]
    labels = out["labels"]
    scored = [i for i, v in enumerate(labels) if v != -100]
    assert scored == [out["pep_start"] + p for p in hidden]
    # and the scored labels are the TRUE residues, not the mask token
    for p, idx in zip(hidden, scored):
        assert labels[idx] == tok.convert_tokens_to_ids(row.pep_seq[p])
    # the hidden input positions really are masked
    for p in hidden:
        assert out["input_ids"][out["pep_start"] + p] == enc_obj.mask_id
    # and the revealed ones really are not
    for p in range(len(row.pep_seq)):
        if p not in hidden:
            assert out["input_ids"][out["pep_start"] + p] != enc_obj.mask_id


def test_gap_char_becomes_the_gap_token_not_unk():
    """A pocket is a handful of fragments; if `/` fell through to <unk> the model would
    be told every fragment boundary is an unknown residue."""
    enc_obj, tok = _encoder()
    row = _row(pep="ACDE", pocket="MN" + GAP_CHAR + "PQ")
    out = enc_obj.encode(row, [0])
    ids = out["input_ids"]
    assert enc_obj.gap_id in ids
    assert tok.unk_token_id not in ids


def test_no_pocket_arm_drops_the_pocket_entirely():
    """The control arm must differ in exactly one way: no pocket tokens. If the pocket
    merely became <mask> or <pad> the two arms would not be a clean single variable."""
    with_p, _ = _encoder(pocket_enabled=True)
    without_p, _ = _encoder(pocket_enabled=False)
    row = _row(pep="ACDE", pocket="MNPQRST")
    a = with_p.encode(row, [0])
    b = without_p.encode(row, [0])
    assert b["pep_start"] < a["pep_start"]
    # the peptide region is identical in content, only shifted
    assert (a["input_ids"][a["pep_start"]:a["pep_end"]]
            == b["input_ids"][b["pep_start"]:b["pep_end"]])
    # and the chemistry token survives in both
    assert a["input_ids"][a["pep_start"] - 1] == b["input_ids"][b["pep_start"] - 1]


def test_chemistry_token_differs_by_type():
    enc_obj, _ = _encoder()
    row = _row(pep="ACDE", cyc=MAINCHAIN)
    a = enc_obj.encode(row, [0])
    b = enc_obj.encode(row, [0], cyc_type=ISOPEPTIDE)
    assert a["input_ids"][a["pep_start"] - 1] != b["input_ids"][b["pep_start"] - 1]


def test_resized_model_survives_a_forward_pass():
    """Catches the ESM output-bias trap: `resize_token_embeddings` grows the input
    embedding and the decoder weight but NOT `EsmLMHead.bias`, a standalone parameter, so
    the first forward dies on a 38-vs-33 shape mismatch. Only a real forward finds it --
    the resize itself raises nothing."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    from cpseq_mlm_data import EsmMlmEncoder, add_task_tokens, collate

    try:
        tok = transformers.AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            "facebook/esm2_t12_35M_UR50D"
        )
    except Exception as e:  # offline / no cache
        pytest.skip(f"model unavailable: {e}")

    n = add_task_tokens(tok, model)
    assert n == 5
    enc_obj = EsmMlmEncoder(tok)
    batch = collate([enc_obj.encode(_row(), [0, 2])], enc_obj.pad_id)
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"])
    assert out.logits.shape[-1] == len(tok)
    # and the added tokens really are addressable in the output space
    assert out.logits.shape[-1] > 33


def test_residue_ids_are_the_twenty_canonical_and_not_unk():
    enc_obj, tok = _encoder()
    ids = enc_obj.residue_ids()
    assert len(ids) == 20 == len(set(ids))
    assert tok.unk_token_id not in ids
    assert [tok.convert_ids_to_tokens(i) for i in ids] == list(AA20)


# --------------------------------------------------------------------------------------
# Proposer request gate. No weights needed -- that is the point of splitting it out.
# --------------------------------------------------------------------------------------


def test_proposer_refuses_over_budget_mask():
    """The mask IS the budget. Refusing here, where the caller sees why, beats letting
    `Proposal.__post_init__` reject every sequence later with only a count to go on."""
    from proteinfoundation.cpseq.mlm_propose import validate_request

    original = "ACDEFGHIKL"  # L=10, budget 5
    assert validate_request(original, [0, 1, 2, 3, 4]) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="budget"):
        validate_request(original, [0, 1, 2, 3, 4, 5])


def test_proposer_budget_agrees_with_search_gate():
    from proteinfoundation.cpseq.mlm_propose import validate_request
    from proteinfoundation.cpseq.search import budget_for as search_budget

    for L in range(2, 20):
        original = "A" * L
        k = search_budget(L)
        if k >= 1:
            validate_request(original, list(range(k)))  # at the ceiling: allowed
        with pytest.raises(ValueError):
            validate_request(original, list(range(k + 1)))  # one over: refused


def test_proposer_rejects_bad_mask_and_order():
    from proteinfoundation.cpseq.mlm_propose import validate_request

    with pytest.raises(ValueError, match="empty"):
        validate_request("ACDEFG", [])
    with pytest.raises(ValueError, match="out of range"):
        validate_request("ACDEFG", [99])
    with pytest.raises(ValueError, match="out of range"):
        validate_request("ACDEFG", [-1])
    with pytest.raises(ValueError, match="order"):
        validate_request("ACDEFG", [0], order="cleverest")


def test_proposer_mask_is_deduplicated_and_sorted():
    from proteinfoundation.cpseq.mlm_propose import validate_request

    assert validate_request("ACDEFGHIKL", [4, 1, 1, 0]) == [0, 1, 4]


def test_proposer_alphabet_matches_the_data_module():
    """A mismatch would misalign the logit columns the proposer slices, so a proposal
    would silently be built from the wrong residues."""
    from cpseq_mlm_data import AA20 as DATA_AA20
    from proteinfoundation.cpseq.mlm_propose import AA20 as PROP_AA20

    assert DATA_AA20 == PROP_AA20


def test_proposer_excludes_cys_by_default():
    """Same reason as `search.default_alphabet`: a free cysteine can pair and close a
    ring nobody asked for, making the molecule differ from its own chemistry label."""
    from proteinfoundation.cpseq.mlm_propose import DEFAULT_EXCLUDED

    assert "C" in DEFAULT_EXCLUDED


def test_collate_pads_labels_with_ignore_index():
    import torch

    from cpseq_mlm_data import collate

    enc_obj, _ = _encoder()
    a = enc_obj.encode(_row(pep="ACDE"), [0])
    b = enc_obj.encode(_row(pep="ACDEFGHIKL"), [0, 3])
    batch = collate([a, b], enc_obj.pad_id)
    assert batch["input_ids"].shape == batch["labels"].shape
    # every padded slot is unscored and unattended
    pad_slots = batch["attention_mask"] == 0
    assert bool((batch["labels"][pad_slots] == -100).all())
    assert bool((batch["input_ids"][pad_slots] == enc_obj.pad_id).all())
    assert int((batch["labels"] != -100).sum()) == 3  # 1 + 2 hidden positions
    assert isinstance(batch["input_ids"], torch.Tensor)
