#!/usr/bin/env python3
"""A learned mean-dG scorer for cyclic-peptide candidates, and what it is worth.

This trains the thing `cpseq_meanscore_baseline.py` leaves a gap for. The baseline asks
what hand-built features can do; this asks whether a small network over the *edit itself*
does better, under the same metric, the same held-out protocol and the same ceiling.

The architecture follows the shape of the data rather than fashion. Every candidate in
this table is a substitution variant of its target's single reference peptide (verified:
2031/2031 rows equal-length to their reference, 1-7 substitutions, one reference per
target), so a candidate is fully described by its **set of substitutions**. The model
encodes each substitution independently -- what it replaces, with what, where, and in
what neighbourhood -- and sums the encodings. That makes it an additive mutation-effect
model with a learned, shared effect function, which is the standard inductive bias for
~100 variants per target and the one most likely to transfer to a target never seen.

What it may and may not look at
-------------------------------
* **No target identity.** There is no target-id embedding, because the headline protocol
  holds out whole targets and an id embedding is undefined for a target never seen. The
  target enters only through its reference peptide (composition, length, properties).
  Scores on a held-out target are therefore honest about what a new target would get, and
  are also the reason they are lower than the within-target numbers below.
* **No post-fold readouts.** iptm, plddt, dSASA, sc_value and the rest all require the
  Boltz-2 fold and Rosetta refinement this scorer exists to avoid paying for, so they are
  not features. They are reported by `cpseq_meanscore_baseline.py` as a separate,
  explicitly-labelled reference block, not as a competitor.

Two protocols, because they answer different questions
------------------------------------------------------
1. **`loto` -- leave one target out.** Train on 19 targets, rank the 20th. This is "can
   it triage candidates for a target we have never scored". It is the hard question and
   the one a general scorer has to answer.
2. **`within` -- held-out sequences inside a target.** Train on every other target plus
   80 % of this target's own scored sequences, rank the remaining 20 %. This is the
   setting a search actually runs in: by the time triage matters, the journal already
   holds scored sequences for this exact target. It is the easier question and the more
   operationally relevant one.

Both are scored by within-target Spearman averaged over targets, imported from
`cpseq_meanscore_baseline` so there is exactly one definition of the metric in the repo.

Reading the number
------------------
Raw rho is uninterpretable here. The label is a ~10-fold estimate whose own rank
reliability is ~0.79, so a model that predicted the latent truth *perfectly* would score
**sqrt(0.79) ~ 0.89** against it, not 1.0 -- the label carries error the model is right
not to reproduce. Every rho is therefore printed next to `ceiling` and as a fraction of
it. (`cpseq_meanscore_baseline.py` compares against the reliability 0.79 directly; that
is the conservative-looking choice that actually *over*-credits a model, and the two
conventions are printed side by side here so the difference is visible rather than
inherited.)

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_scorer.py \
        --dataset /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --json    /zfsauton/scratch/yixiz/cpseq/meanscore/scorer_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

# One definition of the metric and the ceiling, shared with the feature baselines.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import (  # noqa: E402
    AA,
    CHARGE,
    KD,
    label_ceiling,
    within_target_rho,
)

AA_IDX = {a: i for i, a in enumerate(AA)}

# Per-residue scalars. A compact, standard set: enough for the encoder to express
# "hydrophobe -> charge at the junction" without 20 free parameters per context.
VOLUME = dict(zip(AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6,
                       166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0,
                       227.8, 193.6]))
AROMATIC = {a: 1.0 if a in "FWYH" else 0.0 for a in AA}
POLAR = {a: 1.0 if a in "STNQCYHKRDE" else 0.0 for a in AA}
# Backbone flexibility proxy: glycine permits, proline forbids.
FLEX = {a: (1.0 if a == "G" else (-1.0 if a == "P" else 0.0)) for a in AA}
# Side-chain hydrogen-bond donors + acceptors.
HBOND = dict(zip(AA, [0, 0, 2, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 2, 3, 1, 1, 0, 1, 1]))
PROPS = (KD, CHARGE, VOLUME, AROMATIC, POLAR, FLEX, HBOND)
N_PROP = len(PROPS)


def _props(aa: str) -> list[float]:
    return [float(p.get(aa, 0.0)) for p in PROPS]


def _onehot(aa: str) -> list[float]:
    v = [0.0] * len(AA)
    if aa in AA_IDX:
        v[AA_IDX[aa]] = 1.0
    return v


def mutation_features(seq: str, ref: str) -> list[list[float]]:
    """One feature vector per substitution. Empty list if the candidate is the reference.

    Position is encoded relative to length and relative to the closing junction rather
    than as an absolute index, because absolute index 7 means nothing in common between a
    5-mer and a 16-mer target. The junction term exists because these peptides close
    head-to-tail at (0, L-1), so the two termini are chemically adjacent, and the
    neighbourhood features wrap for the same reason.
    """
    L = len(seq)
    out = []
    for p, (a, b) in enumerate(zip(ref, seq)):
        if a == b:
            continue
        pa, pb = _props(a), _props(b)
        rel = p / (L - 1) if L > 1 else 0.0
        half = max(1.0, (L - 1) / 2)
        junction = min(p, L - 1 - p) / half
        prev_aa = ref[(p - 1) % L]
        next_aa = ref[(p + 1) % L]
        out.append(
            _onehot(a)
            + _onehot(b)
            + pa
            + pb
            + [y - x for x, y in zip(pa, pb)]
            + [rel, junction, 1.0 if p in (0, L - 1) else 0.0]
            + _props(prev_aa)
            + _props(next_aa)
        )
    return out


MUT_DIM = 2 * len(AA) + 5 * N_PROP + 3


def context_features(ref: str) -> list[float]:
    """The target, as seen through its reference peptide only. No target identity."""
    n = len(ref)
    comp = [ref.count(a) / n for a in AA]
    mean_props = [st.mean(_props(c)[i] for c in ref) for i in range(N_PROP)]
    return comp + mean_props + [
        n / 16.0,
        sum(CHARGE.get(c, 0.0) for c in ref),
        st.mean(KD.get(c, 0.0) for c in ref),
    ]


CTX_DIM = len(AA) + N_PROP + 3


# --------------------------------------------------------------------------------------
# Aggregators. Kept in sync with cpseq_meanscore_reliability.py: the label the scorer
# trains on has to be the statistic that measurement chose.
# --------------------------------------------------------------------------------------
def aggregate_label(fold_dG: list[float], agg: str, clip_at: float = 0.0) -> float:
    v = [min(x, clip_at) for x in fold_dG]
    if agg == "mean_clip0":
        return st.mean(v)
    if agg == "median_clip0":
        return st.median(v)
    if agg == "trimmed20":
        s = sorted(v)
        cut = int(len(s) * 0.20)
        kept = s[cut : len(s) - cut] if len(s) - 2 * cut >= 1 else s
        return st.mean(kept)
    if agg == "hodges_lehmann":
        pw = [(v[i] + v[j]) / 2 for i in range(len(v)) for j in range(i, len(v))]
        return st.median(pw)
    raise SystemExit(f"unknown --agg {agg}")


def load_extra_features(paths, rows):
    """Merge per-candidate scalar features, keyed `<parent_id>|<sequence>`.

    Used for the pretrained-ProteinMPNN features from `cpseq_meanscore_mpnn_feats.py`.
    Features that are entirely missing or all-NaN across the kept rows are dropped and
    named in the return, so a half-finished feature job degrades to "fewer features"
    rather than a column of zeros that silently reads as a real value.
    """
    merged: dict[str, dict] = {}
    for p in paths:
        blob = json.loads(Path(p).read_text())
        for k, v in (blob.get("features") or {}).items():
            merged.setdefault(k, {}).update(v)
    keys = [f"{r['parent_id']}|{r['sequence']}" for r in rows]
    have = [k in merged for k in keys]
    names = sorted({n for k in keys if k in merged for n in merged[k]})
    usable, dropped = [], []
    for n in names:
        vals = [merged[k].get(n) for k in keys if k in merged]
        good = [v for v in vals if isinstance(v, (int, float)) and v == v]
        (usable if len(good) >= 0.5 * max(1, len(vals)) else dropped).append(n)
    return merged, usable, dropped, have


def extra_matrix(merged, names, rows, parents):
    """(N, F) features, centred within each target.

    Per-target centring uses only the features, never the label, so it leaks nothing --
    and it is the right frame for a within-target metric: an absolute MPNN likelihood is
    dominated by peptide length and by how designable that backbone is, both constant
    inside a target and therefore pure nuisance.
    """
    X = np.zeros((len(rows), len(names)), dtype=float)
    for i, r in enumerate(rows):
        d = merged.get(f"{r['parent_id']}|{r['sequence']}", {})
        for j, n in enumerate(names):
            v = d.get(n)
            X[i, j] = v if isinstance(v, (int, float)) and v == v else np.nan
    # Column means per target, then global fallback for a target with no values at all.
    for j in range(X.shape[1]):
        col = X[:, j]
        for t in set(parents):
            m = np.array([p == t for p in parents])
            vals = col[m & ~np.isnan(col)]
            if len(vals):
                col[m] = np.where(np.isnan(col[m]), vals.mean(), col[m]) - vals.mean()
            else:
                col[m] = 0.0
        X[:, j] = np.nan_to_num(col, nan=0.0)
    return X


class MutationEffectScorer(nn.Module):
    """Sum of learned per-substitution effects, conditioned on the reference peptide."""

    def __init__(self, d_mut: int = 48, d_ctx: int = 24, d_hidden: int = 64,
                 dropout: float = 0.1, n_extra: int = 0):
        super().__init__()
        self.n_extra = n_extra
        self.mut = nn.Sequential(
            nn.Linear(MUT_DIM, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_mut), nn.GELU(),
        )
        self.ctx = nn.Sequential(
            nn.Linear(CTX_DIM, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_ctx), nn.GELU(),
        )
        # sum + mean + max pooling over the substitution set, plus the context vector,
        # the substitution count, and any pretrained structure-aware scalars.
        self.head = nn.Sequential(
            nn.Linear(3 * d_mut + d_ctx + 1 + n_extra, d_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, mut, mask, ctx, extra=None):
        # mut: (B, M, MUT_DIM)  mask: (B, M)  ctx: (B, CTX_DIM)
        h = self.mut(mut) * mask.unsqueeze(-1)
        n = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = torch.cat(
            [
                h.sum(1),
                h.sum(1) / n,
                h.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values,
            ],
            dim=-1,
        )
        parts = [pooled, self.ctx(ctx), mask.sum(1, keepdim=True)]
        if self.n_extra:
            parts.append(extra)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def build_tensors(rows, agg, device, extra_np=None):
    """Pad the per-row substitution sets into one batch, and centre labels per target."""
    muts = [mutation_features(r["sequence"], r["original_sequence"]) for r in rows]
    M = max(1, max(len(m) for m in muts))
    mut = torch.zeros(len(rows), M, MUT_DIM)
    mask = torch.zeros(len(rows), M)
    for i, m in enumerate(muts):
        for j, f in enumerate(m):
            mut[i, j] = torch.tensor(f)
            mask[i, j] = 1.0
    ctx = torch.tensor([context_features(r["original_sequence"]) for r in rows],
                       dtype=torch.float32)
    y = torch.tensor([aggregate_label(r["fold_dG"], agg) for r in rows],
                     dtype=torch.float32)
    se = torch.tensor([r.get("label_se") or 1.0 for r in rows], dtype=torch.float32)
    parents = [r["parent_id"] for r in rows]
    if extra_np is None:
        extra_np = np.zeros((len(rows), 0))
    extra = torch.tensor(extra_np, dtype=torch.float32)
    return (mut.to(device), mask.to(device), ctx.to(device), y.to(device),
            se.to(device), parents, extra.to(device))


def centre_by_target(y: torch.Tensor, parents: list[str], means: dict | None = None):
    """Subtract each target's own mean.

    The pooled label sd is dominated by between-target spread (12.3 REU against a
    within-target 3.76), so an uncentred squared error is mostly a fit to the target mean
    -- a model that predicts only the target mean scores well on the pooled correlation
    and ranks nothing within a target. Centring removes that nuisance from the loss, and
    costs nothing at test time because within-target Spearman is invariant to it.
    """
    if means is None:
        acc = collections.defaultdict(list)
        for p, v in zip(parents, y.tolist()):
            acc[p].append(v)
        means = {p: st.mean(v) for p, v in acc.items()}
    mu = torch.tensor([means.get(p, 0.0) for p in parents], dtype=y.dtype,
                      device=y.device)
    return y - mu, means


def pairwise_rank_loss(pred, y, parents, margin_scale: float = 1.0):
    """Within-target pairwise margin loss on the label gap. Aligned with the metric."""
    idx = collections.defaultdict(list)
    for i, p in enumerate(parents):
        idx[p].append(i)
    total, n = pred.new_zeros(()), 0
    for ids in idx.values():
        if len(ids) < 2:
            continue
        ii = torch.tensor(ids, device=pred.device)
        dp = pred[ii].unsqueeze(0) - pred[ii].unsqueeze(1)
        dy = y[ii].unsqueeze(0) - y[ii].unsqueeze(1)
        keep = dy.abs() > 1e-6
        if not keep.any():
            continue
        # Hinge on the sign of the true gap, with the margin scaled by the gap itself so
        # near-ties are not forced apart harder than the label can justify.
        loss = torch.relu(margin_scale * dy.sign() * dy.abs().clamp(max=5.0) - dp * dy.sign())
        total = total + loss[keep].mean()
        n += 1
    return total / max(n, 1)


def train_once(tr, va, cfg, device, seed):
    torch.manual_seed(seed)
    mut, mask, ctx, y, se, parents, extra = tr
    yv_pack = va
    model = MutationEffectScorer(dropout=cfg["dropout"],
                                 n_extra=extra.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    yc, means = centre_by_target(y, parents)
    w = (1.0 / se.clamp(min=0.2) ** 2)
    w = (w / w.mean()).clamp(max=5.0)

    best = (-2.0, None, 0)
    n = mut.shape[0]
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg["batch"]):
            b = perm[s : s + cfg["batch"]]
            pred = model(mut[b], mask[b], ctx[b], extra[b])
            bp = [parents[i] for i in b.tolist()]
            loss = pred.new_zeros(())
            if cfg["loss"] in ("mse", "both"):
                loss = loss + (w[b] * (pred - yc[b]) ** 2).mean()
            if cfg["loss"] in ("rank", "both"):
                loss = loss + cfg["rank_weight"] * pairwise_rank_loss(pred, yc[b], bp)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (epoch + 1) % cfg["eval_every"] == 0 and yv_pack is not None:
            vmut, vmask, vctx, vy, _vse, vpar, vextra = yv_pack
            model.eval()
            with torch.no_grad():
                vp = model(vmut, vmask, vctx, vextra).cpu().numpy()
            rho, _ = within_target_rho(vpar, vy.cpu().numpy(), vp)
            if rho == rho and rho > best[0]:
                best = (rho, {k: v.detach().clone() for k, v in model.state_dict().items()},
                        epoch + 1)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0], best[2], means


def predict(model, pack, device):
    mut, mask, ctx, extra = pack[0], pack[1], pack[2], pack[6]
    model.eval()
    with torch.no_grad():
        return model(mut, mask, ctx, extra).cpu().numpy()


def subset(pack, ids):
    mut, mask, ctx, y, se, parents, extra = pack
    # dtype is explicit: an empty `ids` would otherwise give a float tensor and indexing
    # fails with a confusing error far from the cause.
    t = torch.tensor(list(ids), device=mut.device, dtype=torch.long)
    return (mut[t], mask[t], ctx[t], y[t], se[t], [parents[i] for i in ids], extra[t])


def ridge_loto_baselines(rows, agg, targets, flat_feature_fns):
    """The same LOTO loop for linear baselines, so the comparison is apples to apples."""
    out = {}
    y = np.array([aggregate_label(r["fold_dG"], agg) for r in rows])
    parents = np.array([r["parent_id"] for r in rows])
    for name, fn in flat_feature_fns.items():
        X = np.array([fn(r) for r in rows], dtype=float)
        pred = np.zeros_like(y)
        for t in targets:
            te = parents == t
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            # Centre the training labels per target, exactly as the network does.
            ytr = y[tr].copy()
            for p in set(parents[tr]):
                m = parents[tr] == p
                ytr[m] -= ytr[m].mean()
            model = RidgeCV(alphas=np.logspace(-2, 4, 25))
            model.fit((X[tr] - mu) / sd, ytr)
            pred[te] = model.predict((X[te] - mu) / sd)
        rho, used = within_target_rho(parents, y, pred, 4)
        out[name] = {"rho": rho, "n_targets": used}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--agg", default="trimmed20",
                    choices=("trimmed20", "mean_clip0", "median_clip0", "hodges_lehmann"),
                    help="target statistic over the folds; pick what the reliability "
                         "sweep recommended")
    ap.add_argument("--loss", default="both", choices=("mse", "rank", "both"))
    ap.add_argument("--protocol", default="both", choices=("loto", "within", "both"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--n-val-targets", type=int, default=3,
                    help="training targets held back to pick the epoch (never the test target)")
    ap.add_argument("--within-folds", type=int, default=5)
    ap.add_argument("--include-reference", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--extra-features", nargs="+", default=[], type=Path,
                    help="per-candidate scalar feature JSONs from "
                         "cpseq_meanscore_mpnn_feats.py (pretrained ProteinMPNN). "
                         "Rows without features are dropped, and the count is reported.")
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    rows = payload["rows"]
    if not args.include_reference:
        rows = [r for r in rows if r.get("arm") != "reference"]
    rows = [r for r in rows if r.get("original_sequence")
            and len(r["sequence"]) == len(r["original_sequence"])
            and any(a != b for a, b in zip(r["sequence"], r["original_sequence"]))]

    device = torch.device(args.device)
    cfg = {k: getattr(args, k) for k in
           ("loss", "epochs", "batch", "lr", "weight_decay", "dropout", "rank_weight",
            "eval_every")}

    extra_np, extra_names, extra_dropped = None, [], []
    if args.extra_features:
        merged, extra_names, extra_dropped, have = load_extra_features(
            args.extra_features, rows)
        n_before = len(rows)
        rows = [r for r, h in zip(rows, have) if h]
        print(f"extra     {len(extra_names)} features from "
              f"{len(args.extra_features)} file(s); kept {len(rows)}/{n_before} rows")
        if extra_dropped:
            print(f"          dropped (mostly missing/NaN): {extra_dropped}")
        if not rows:
            raise SystemExit("no rows carry the requested extra features")

    targets = sorted({r["parent_id"] for r in rows})
    if extra_names:
        extra_np = extra_matrix(merged, extra_names, rows,
                                [r["parent_id"] for r in rows])
    print(f"dataset   {args.dataset}")
    print(f"label     {args.agg} of fold_dG clipped at 0 REU")
    print(f"rows      {len(rows)} substitution variants over {len(targets)} targets")
    print(f"model     additive mutation-effect net, mut_dim={MUT_DIM} ctx_dim={CTX_DIM} "
          f"n_extra={len(extra_names)}, "
          f"{sum(p.numel() for p in MutationEffectScorer(n_extra=len(extra_names)).parameters())} params")
    print(f"loss      {args.loss}   seeds {args.seeds}   device {args.device}")

    half, full, nceil = label_ceiling(rows, 4)
    ceiling = full**0.5 if full == full and full > 0 else float("nan")
    print(f"\nCEILING")
    print(f"  split-half rho (5 folds vs 5)          : {half:.3f}  ({nceil} targets)")
    print(f"  Spearman-Brown reliability of 10 folds : {full:.3f}")
    print(f"  sqrt(reliability) = max rho for a PERFECT model : {ceiling:.3f}  <-- compare to this")

    pack = build_tensors(rows, args.agg, device, extra_np)
    parents_all = pack[5]
    results = {"extra_features": extra_names, "extra_dropped": extra_dropped}

    # ---- linear baselines under the identical LOTO loop ------------------------------
    def f_nmut(r):
        return [float(r.get("n_mutations") or 0), float(r.get("identity_to_original") or 1)]

    def f_comp(r):
        s = r["sequence"]
        return [s.count(a) / len(s) for a in AA]

    def f_delta(r):
        s, o = r["sequence"], r["original_sequence"]
        d = [(s.count(a) - o.count(a)) / len(s) for a in AA]
        return d + [
            st.mean(KD.get(c, 0.0) for c in s) - st.mean(KD.get(c, 0.0) for c in o),
            sum(CHARGE.get(c, 0.0) for c in s) - sum(CHARGE.get(c, 0.0) for c in o),
            float(r.get("n_mutations") or 0),
        ]

    def f_mutbag(r):
        """Bag of (wt->mut) substitution types, position-blind. The positional control:
        if the network does not beat this, it is not using position."""
        v = [0.0] * (len(AA) * len(AA))
        for a, b in zip(r["original_sequence"], r["sequence"]):
            if a != b and a in AA_IDX and b in AA_IDX:
                v[AA_IDX[a] * len(AA) + AA_IDX[b]] += 1.0
        return v

    feature_fns = {
        "n_mutations+identity": f_nmut,
        "composition": f_comp,
        "composition_delta": f_delta,
        "substitution_bag (position-blind)": f_mutbag,
    }
    if extra_names:
        # The MPNN features arrive already centred per target, so the row lookup has to
        # go through the same matrix the network sees -- not the raw JSON -- or the ridge
        # would be fitting a different quantity and the comparison would be void.
        pos = {f"{r['parent_id']}|{r['sequence']}": i for i, r in enumerate(rows)}

        def f_mpnn(r, _X=extra_np, _pos=pos):
            return list(_X[_pos[f"{r['parent_id']}|{r['sequence']}"]])

        def f_comp_mpnn(r):
            return f_delta(r) + f_mpnn(r)

        feature_fns["mpnn (pretrained, structure-aware)"] = f_mpnn
        feature_fns["composition_delta + mpnn"] = f_comp_mpnn

    print("\nLINEAR BASELINES (leave-one-target-out, identical loop)")
    base = ridge_loto_baselines(rows, args.agg, targets, feature_fns)
    for name, v in sorted(base.items(), key=lambda kv: -kv[1]["rho"]):
        print(f"  {name:36s} rho {v['rho']:6.3f}   {v['rho']/ceiling*100:5.1f}% of ceiling")
    results["linear_baselines"] = base

    # ---- the network, leave-one-target-out --------------------------------------------
    if args.protocol in ("loto", "both"):
        print(f"\nNETWORK -- leave-one-target-out ({len(targets)} folds x {args.seeds} seeds)")
        per_target, per_seed_rho = {}, collections.defaultdict(list)
        for ti, t in enumerate(targets):
            te_ids = [i for i, p in enumerate(parents_all) if p == t]
            tr_pool = [i for i, p in enumerate(parents_all) if p != t]
            other = [x for x in targets if x != t]
            preds = []
            for s in range(args.seeds):
                rs = np.random.RandomState(1000 + s)
                # At least one training target must survive the validation split, or
                # `tr_ids` is empty and the failure surfaces as an indexing error.
                n_val = max(1, min(args.n_val_targets, len(other) - 1))
                val_t = set(rs.choice(other, size=n_val, replace=False).tolist())
                tr_ids = [i for i in tr_pool if parents_all[i] not in val_t]
                va_ids = [i for i in tr_pool if parents_all[i] in val_t]
                model, vrho, bep, _ = train_once(
                    subset(pack, tr_ids), subset(pack, va_ids), cfg, device, 1234 + s)
                preds.append(predict(model, subset(pack, te_ids), device))
            yt = pack[3][torch.tensor(te_ids, device=device)].cpu().numpy()
            for s, pr in enumerate(preds):
                r = spearmanr(yt, pr).statistic
                if r == r:
                    per_seed_rho[s].append(r)
            ens = np.mean(preds, axis=0)
            rho = spearmanr(yt, ens).statistic
            per_target[t] = {"n_seq": len(te_ids), "rho_ensemble": float(rho),
                             "rho_per_seed": [float(spearmanr(yt, p).statistic) for p in preds]}
            print(f"  [{ti+1:2d}/{len(targets)}] {t:24s} n={len(te_ids):4d} "
                  f"rho={rho:6.3f}")
        rhos = [v["rho_ensemble"] for v in per_target.values() if v["rho_ensemble"] == v["rho_ensemble"]]
        mean_rho = st.mean(rhos)
        seed_means = [st.mean(v) for v in per_seed_rho.values() if v]
        print(f"\n  LOTO within-target rho (mean over {len(rhos)} targets)")
        print(f"    5-seed ensemble : {mean_rho:.3f}   "
              f"{mean_rho/ceiling*100:.1f}% of ceiling {ceiling:.3f}")
        print(f"    single seed     : {st.mean(seed_means):.3f} "
              f"(spread over seeds {min(seed_means):.3f}-{max(seed_means):.3f})")
        print(f"    median target   : {st.median(rhos):.3f}   "
              f"worst {min(rhos):.3f}  best {max(rhos):.3f}")
        results["loto"] = {
            "per_target": per_target,
            "mean_rho_ensemble": mean_rho,
            "median_rho": st.median(rhos),
            "mean_rho_single_seed": st.mean(seed_means),
            "frac_of_ceiling": mean_rho / ceiling,
        }

    # ---- the network, held-out sequences inside a target -----------------------------
    if args.protocol in ("within", "both"):
        print(f"\nNETWORK -- held-out sequences within target "
              f"({args.within_folds}-fold, {args.seeds} seeds)")
        rs = np.random.RandomState(7)
        by_t = collections.defaultdict(list)
        for i, p in enumerate(parents_all):
            by_t[p].append(i)
        assign = {}
        for t, ids in by_t.items():
            f = rs.permutation(len(ids)) % args.within_folds
            for i, fi in zip(ids, f):
                assign[i] = int(fi)
        pred_all = np.zeros(len(parents_all))
        for fold in range(args.within_folds):
            te_ids = [i for i in range(len(parents_all)) if assign[i] == fold]
            tr_ids = [i for i in range(len(parents_all)) if assign[i] != fold]
            va_n = max(1, len(tr_ids) // 10)
            va_ids = list(rs.choice(tr_ids, size=va_n, replace=False))
            tr_ids2 = [i for i in tr_ids if i not in set(va_ids)]
            preds = []
            for s in range(args.seeds):
                model, vrho, bep, _ = train_once(
                    subset(pack, tr_ids2), subset(pack, va_ids), cfg, device, 777 + s)
                preds.append(predict(model, subset(pack, te_ids), device))
            pred_all[te_ids] = np.mean(preds, axis=0)
            print(f"  fold {fold+1}/{args.within_folds}  n_test={len(te_ids)}")
        y_np = pack[3].cpu().numpy()
        rho, used = within_target_rho(parents_all, y_np, pred_all, 4)
        print(f"\n  WITHIN-target rho (mean over {used} targets): {rho:.3f}   "
              f"{rho/ceiling*100:.1f}% of ceiling {ceiling:.3f}")
        results["within_target"] = {"rho": rho, "n_targets": used,
                                    "frac_of_ceiling": rho / ceiling}

    out = {
        "dataset": str(args.dataset),
        "label_agg": args.agg,
        "label_note": "Rosetta ref2015 REU; NOT a measured binding free energy",
        "n_rows": len(rows),
        "n_targets": len(targets),
        "config": {**cfg, "seeds": args.seeds, "n_val_targets": args.n_val_targets,
                   "within_folds": args.within_folds},
        "ceiling": {"split_half_rho": half, "reliability_10fold": full,
                    "sqrt_reliability": ceiling,
                    "note": "compare models to sqrt_reliability"},
        **results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote     {args.json}")


if __name__ == "__main__":
    main()
