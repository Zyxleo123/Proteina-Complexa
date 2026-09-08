"""CPU unit tests for loading a checkpoint trained before a cyclization type was added.

Run:  .venv/bin/python scripts/test_cyc_cond_compat.py

Why this exists
---------------
`NUM_CYCLIZATION_COND_TYPES` sizes `CyclizationTypeSeqFeat.embedding`, so appending a type
(LINEAR, taking the table 4 -> 5) makes every checkpoint trained before it fail to load:

    size mismatch for nn.cond_factory.feat_creators.2.embedding.weight:
      copying a param with shape torch.Size([4, 256]), the shape in current model is [5, 256]

That killed all 12 shards of an SDEdit sweep at load time (job 44557, 2026-09-07), and it
would equally kill design, eval and every other consumer of a pre-existing checkpoint --
`strict=False` does not help, because a size mismatch is fatal regardless.

The two properties that matter are opposite in sign, so both are tested:
  * an OLD checkpoint must LOAD (pad the missing rows), and
  * the padded rows must never be USED (they are untrained zeros; embedding one returns a
    zero vector that is indistinguishable from a real conditioning signal, so the run would
    look like it honoured the request while ignoring it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from proteinfoundation.utils.ema_callback import EMAOptimizer
from proteinfoundation.cyclization.constants import (  # noqa: E402
    DISULFIDE,
    ISOPEPTIDE,
    LINEAR,
    MAINCHAIN,
    NUM_CYCLIZATION_COND_TYPES,
    UNSPECIFIED,
)
from proteinfoundation.nn.feature_factory.seq_cond_feats import (  # noqa: E402
    CyclizationTypeSeqFeat,
)

DIM = 8


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"  ok  {name}")


def batch(types, n=5):
    """`extract_bs_and_n` reads the shape off a coordinate/latent tensor, not off the mask."""
    t = torch.tensor(types, dtype=torch.long)
    return {"cyclization_type_cond": t,
            "coords_nm": torch.zeros(len(types), n, 37, 3),
            "mask": torch.ones(len(types), n, dtype=torch.bool)}


def old_state_dict(n_types, dim=DIM):
    """What a checkpoint trained with `n_types` conditioning types looks like."""
    return {"embedding.weight": torch.arange(1, n_types * dim + 1,
                                             dtype=torch.float32).reshape(n_types, dim)}


def test_old_checkpoint_loads():
    n_old = NUM_CYCLIZATION_COND_TYPES - 1
    feat = CyclizationTypeSeqFeat(cyc_emb_dim=DIM)
    sd = old_state_dict(n_old)
    missing, unexpected = feat.load_state_dict(sd, strict=False)
    check("an older checkpoint loads at all", True)
    check("no unexpected keys", not unexpected, unexpected)
    check("embedding kept the current width",
          feat.embedding.weight.shape[0] == NUM_CYCLIZATION_COND_TYPES,
          feat.embedding.weight.shape)
    check("trained rows are the checkpoint's, unchanged",
          torch.equal(feat.embedding.weight[:n_old], sd["embedding.weight"][:n_old]))
    check("added rows are zero, not random",
          float(feat.embedding.weight[n_old:].abs().max()) == 0.0)
    check("the pad is recorded", feat._trained_num_types == n_old, feat._trained_num_types)


def test_a_current_checkpoint_is_untouched():
    """No padding, no guard: a checkpoint that matches the table must behave exactly as before."""
    feat = CyclizationTypeSeqFeat(cyc_emb_dim=DIM)
    sd = old_state_dict(NUM_CYCLIZATION_COND_TYPES)
    feat.load_state_dict(sd, strict=False)
    check("nothing was padded", feat._trained_num_types is None, feat._trained_num_types)
    check("weights are the checkpoint's exactly",
          torch.equal(feat.embedding.weight, sd["embedding.weight"]))
    out = feat(batch([LINEAR]))
    check("every type embeds, including the newest",
          float(out.abs().sum()) > 0.0)


def test_padded_checkpoint_refuses_the_untrained_type():
    """The half that matters. A zero embedding row is not a fallback, it is a silent lie."""
    n_old = NUM_CYCLIZATION_COND_TYPES - 1
    feat = CyclizationTypeSeqFeat(cyc_emb_dim=DIM)
    feat.load_state_dict(old_state_dict(n_old), strict=False)
    try:
        feat(batch([LINEAR]))
    except RuntimeError as e:
        check("requesting the untrained type is a hard error", "untrained" in str(e), str(e))
        check("the error names the type", "linear" in str(e), str(e))
    else:
        raise AssertionError("FAIL an untrained cyclization type was silently embedded")


def test_padded_checkpoint_still_serves_the_types_it_knows():
    """The sweep this fix unblocks only ever requests disulfide and isopeptide."""
    n_old = NUM_CYCLIZATION_COND_TYPES - 1
    feat = CyclizationTypeSeqFeat(cyc_emb_dim=DIM)
    sd = old_state_dict(n_old)
    feat.load_state_dict(sd, strict=False)
    for t, name in ((MAINCHAIN, "mainchain"), (DISULFIDE, "disulfide"),
                    (ISOPEPTIDE, "isopeptide"), (UNSPECIFIED, "unspecified")):
        out = feat(batch([t]))
        check(f"{name} embeds from the trained row",
              torch.allclose(out[0, 0], sd["embedding.weight"][t]))
    out = feat(batch([DISULFIDE, ISOPEPTIDE]))
    check("a mixed batch is fine", out.shape == (2, 5, DIM), out.shape)
    check("and it is broadcast over residues",
          torch.equal(out[:, 0, :], out[:, -1, :]))


def test_one_bad_entry_in_a_batch_is_caught():
    """The guard is on the max over the batch, so a single bad sample cannot slip through."""
    n_old = NUM_CYCLIZATION_COND_TYPES - 1
    feat = CyclizationTypeSeqFeat(cyc_emb_dim=DIM)
    feat.load_state_dict(old_state_dict(n_old), strict=False)
    try:
        feat(batch([DISULFIDE, LINEAR, ISOPEPTIDE]))
    except RuntimeError:
        check("one untrained sample fails the whole batch", True)
    else:
        raise AssertionError("FAIL an untrained type hid behind valid ones in the same batch")


def test_newer_checkpoint_under_older_code_still_fails():
    """Padding is one-directional. Silently DROPPING a type the checkpoint trained would
    change the model's behaviour, so that case must keep failing loudly -- including under
    `strict=False`, which does not soften a size mismatch."""
    feat = CyclizationTypeSeqFeat(cyc_emb_dim=DIM)
    try:
        feat.load_state_dict(old_state_dict(NUM_CYCLIZATION_COND_TYPES + 1), strict=False)
    except RuntimeError as e:
        check("a wider checkpoint still fails, not truncated", "size mismatch" in str(e), str(e))
        check("and nothing was recorded as padded", feat._trained_num_types is None)
    else:
        raise AssertionError("FAIL a wider checkpoint was silently accepted")


# ---------------------------------------------------------------------------
# The EMA / optimizer half of the same skew.
#
# The hook above only reaches `checkpoint["state_dict"]`. The EMA shadow copy of every
# parameter, and the inner optimizer's `exp_avg` / `exp_avg_sq`, live in
# `checkpoint["optimizer_states"]` and are stored POSITIONALLY -- there are no names to
# match on. Padding the model alone therefore produced a run that resumed, trained one
# batch, and died at the first checkpoint save with
#   RuntimeError: The size of tensor a (5) must match the size of tensor b (4)
# from `EMAOptimizer.switch_main_parameter_weights` (job 44925, 2026-09-07), with a second
# failure of the same kind waiting at the first AdamW step.
# ---------------------------------------------------------------------------

def _skewed_ema_state(p_old_rows=4):
    """An EMAOptimizer state_dict whose parameter 1 still has the pre-LINEAR row count."""
    p0 = torch.nn.Parameter(torch.randn(8, 8))
    p1 = torch.nn.Parameter(torch.zeros(NUM_CYCLIZATION_COND_TYPES, DIM))
    inner = torch.optim.AdamW([p0, p1], lr=1e-4)
    for p in (p0, p1):
        p.grad = torch.randn_like(p)
    inner.step()

    opt_sd = inner.state_dict()
    for k in ("exp_avg", "exp_avg_sq"):
        opt_sd["state"][1][k] = opt_sd["state"][1][k][:p_old_rows].clone()
    ema = [p0.data.clone(), torch.randn(p_old_rows, DIM)]
    state = {"opt": opt_sd, "ema": ema, "current_step": 10, "decay": 0.999,
             "every_n_steps": 1}
    return (p0, p1), state


def test_ema_and_optimizer_state_are_padded():
    (p0, p1), state = _skewed_ema_state()
    opt = EMAOptimizer(torch.optim.AdamW([p0, p1], lr=1e-4), device=torch.device("cpu"))
    opt.load_state_dict(state)

    check("EMA shadow was padded to the current width",
          tuple(opt.ema_params[1].shape) == (NUM_CYCLIZATION_COND_TYPES, DIM),
          str(tuple(opt.ema_params[1].shape)))
    check("the appended EMA row is zero, matching the model-side pad",
          bool((opt.ema_params[1][-1] == 0).all()))
    moments = opt.optimizer.state_dict()["state"][1]
    check("optimizer moments were padded too",
          all(tuple(moments[k].shape) == (NUM_CYCLIZATION_COND_TYPES, DIM)
              for k in ("exp_avg", "exp_avg_sq")),
          str({k: tuple(moments[k].shape) for k in ("exp_avg", "exp_avg_sq")}))
    check("the appended moment rows are zero, not inherited momentum",
          all(bool((moments[k][-1] == 0).all()) for k in ("exp_avg", "exp_avg_sq")))


def test_the_two_operations_that_actually_crashed():
    """Save-time EMA swap, then an optimizer step -- the two real failure points."""
    (p0, p1), state = _skewed_ema_state()
    opt = EMAOptimizer(torch.optim.AdamW([p0, p1], lr=1e-4), device=torch.device("cpu"))
    opt.load_state_dict(state)

    opt.switch_main_parameter_weights(saving_ema_model=True)
    opt.switch_main_parameter_weights(saving_ema_model=False)
    check("the save-time EMA weight swap no longer raises", True)

    for p in (p0, p1):
        p.grad = torch.randn_like(p)
    opt.optimizer.step()
    check("the first optimizer step after resume no longer raises", True)


def test_a_genuinely_different_shape_is_refused():
    """Padding must not paper over an architecture change -- only appended rows."""
    p0 = torch.nn.Parameter(torch.randn(8, 8))
    p1 = torch.nn.Parameter(torch.zeros(NUM_CYCLIZATION_COND_TYPES, DIM))
    inner = torch.optim.AdamW([p0, p1], lr=1e-4)
    for p in (p0, p1):
        p.grad = torch.randn_like(p)
    inner.step()
    state = {"opt": inner.state_dict(),
             "ema": [p0.data.clone(), torch.randn(NUM_CYCLIZATION_COND_TYPES, DIM // 2)],
             "current_step": 10, "decay": 0.999, "every_n_steps": 1}

    opt = EMAOptimizer(torch.optim.AdamW([p0, p1], lr=1e-4), device=torch.device("cpu"))
    try:
        opt.load_state_dict(state)
    except RuntimeError as e:
        check("a changed embedding WIDTH is refused, not padded",
              "Only appending rows" in str(e), str(e))
    else:
        raise AssertionError("FAIL a different embedding width was silently accepted")


def main():
    for fn in (test_old_checkpoint_loads,
               test_a_current_checkpoint_is_untouched,
               test_padded_checkpoint_refuses_the_untrained_type,
               test_padded_checkpoint_still_serves_the_types_it_knows,
               test_one_bad_entry_in_a_batch_is_caught,
               test_newer_checkpoint_under_older_code_still_fails,
               test_ema_and_optimizer_state_are_padded,
               test_the_two_operations_that_actually_crashed,
               test_a_genuinely_different_shape_is_refused):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL CYCLIZATION-COND COMPAT TESTS PASSED")


if __name__ == "__main__":
    main()
