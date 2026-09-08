"""Sequence recovery for sequence-conditioned generation.

Two very different questions share the word "recovery" here, and collapsing them into one
number destroys both:

``cond_recovery``
    On positions whose residue type was HANDED to the model, does the decoded structure
    carry that residue? This is **compliance**, not prediction -- a correct model is near
    1.0 and a model that ignores the conditioning channel sits near chance (~0.05-0.1,
    or higher wherever the marginal is skewed toward GLY/ALA). It is the single number
    that says whether "give me this peptide's structure" was honoured at all, and no
    geometry metric can substitute for it: a model that silently designs its own sequence
    can still close its rings and pass every clash check.

``free_recovery``
    On positions left unspecified, does the model reproduce the native residue? This is a
    genuine (hard) prediction number -- "generate the rest" -- and is expected to be well
    below 1.0. Reported only when some positions are actually free.

Both are computed on the residue types DECODED from the sampled local latents, i.e. what
the model actually emitted, never on the conditioning input.
"""

from __future__ import annotations

import torch


def sequence_recovery_metrics(
    pred_aatype: torch.Tensor,
    true_aatype: torch.Tensor,
    mask: torch.Tensor,
    cond_mask: torch.Tensor,
    prefix: str = "val_gen/seq",
) -> dict[str, float]:
    """Per-position sequence agreement, split by whether the position was conditioned.

    Args:
        pred_aatype: ``[b, n]`` residue types decoded from the sample.
        true_aatype: ``[b, n]`` requested / native residue types.
        mask: ``[b, n]`` bool, real (non-padding) residues.
        cond_mask: ``[b, n]`` bool, positions whose type was revealed to the model.
        prefix: metric key prefix.

    Returns:
        Dict of scalars. Rates over an empty population are NaN, never 0.0 -- 0.0 would read
        as "nothing matched", a different claim from "not measurable here" (`log_nan_safe`
        skips NaNs when aggregating). Counts are tallies and stay 0.
    """
    mask = mask.bool()
    cond_mask = cond_mask.bool() & mask
    free_mask = mask & ~cond_mask
    correct = pred_aatype == true_aatype

    def _rate(sel: torch.Tensor) -> float:
        n = int(sel.sum().item())
        if n == 0:
            return float("nan")
        return float((correct & sel).sum().item()) / n

    n_cond = int(cond_mask.sum().item())
    n_free = int(free_mask.sum().item())
    out = {
        f"{prefix}/cond_recovery": _rate(cond_mask),
        f"{prefix}/free_recovery": _rate(free_mask),
        f"{prefix}/all_recovery": _rate(mask),
        f"{prefix}/n_cond_res": float(n_cond),
        f"{prefix}/n_free_res": float(n_free),
        f"{prefix}/cond_frac": (float(n_cond) / float(n_cond + n_free)) if (n_cond + n_free) else float("nan"),
    }

    # Per-example "the whole revealed sequence came back exactly" -- the user-facing pass/fail
    # for "fold this peptide". A mean over positions can sit at 0.95 while no single peptide is
    # fully correct, which is the case that matters and the one the mean hides.
    per_ex_n = cond_mask.sum(dim=-1)  # [b]
    per_ex_ok = (correct | ~cond_mask).all(dim=-1) & (per_ex_n > 0)  # [b]
    n_ex = int((per_ex_n > 0).sum().item())
    out[f"{prefix}/cond_exact_frac"] = (float(per_ex_ok.sum().item()) / n_ex) if n_ex else float("nan")
    return out
