"""Sequence proposals from the CPSea unmasking model.

The proposal step of `cpseq` search, as a learned alternative to the random mutation
bands, the single-substitution scan, the MPNN redesigns and the Llama proposer. It emits
rows in the same shape every other arm emits, so the measurement -- a fresh Boltz-2 refold
through the frozen gate -- is identical and the arms differ only in where their sequences
came from.

THE MASK IS THE BUDGET. The search's one gate is `n_mutations(original, proposal) <=
budget_for(L) = L // 2`, enforced by `Proposal.__post_init__`. A masked-LM proposal
satisfies it by construction: open k <= L//2 positions and the revealed positions are
copied verbatim, so an over-budget sequence cannot be produced. Nothing is rejected and
resampled, which also means the proposal distribution is not quietly reshaped by a
rejection filter.

WHY NOT ONE-SHOT ARGMAX OVER THE MASKED POSITIONS. A masked LM gives per-position
marginals, not a joint (Brooks et al., NeurIPS 2024). Filling every masked position at
once from one forward composes marginals into a sequence that no position actually voted
for -- in this project's own notes, "leave-one-out masking gives marginals that do NOT
compose into a joint sequence; mask jointly to design". So positions are filled ONE AT A
TIME with a re-forward after each, which is the ancestral sampler for a masked diffusion
model and is what the EvoDiff order-agnostic and MDLM lines of work actually do. The cost
is k forwards for k masked positions, and k <= 8 on a 5-16-mer.

UNMASKING ORDER IS A REAL LEVER, not an implementation detail: Path Planning (ICLR 2025)
reports +22% relative protein foldability from changing nothing but the order. Two orders
are offered here --
    confidence  fill the position the model is currently most sure about first. Greedy,
                low diversity, the one to use when a single best proposal is wanted.
    random      fill in a random order. Higher diversity at the same temperature, which
                is what a search wants from `n` draws off one parent.
-- and the choice is recorded on every row, because a proposal set is not reproducible
without it.

CYS IS EXCLUDED BY DEFAULT, matching `search.default_alphabet`: a free cysteine
introduced at random can pair with another and close a ring nobody asked for, making the
molecule differ from the chemistry its own label names. The chemistry's own anchors are
placed by the caller, not drawn here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AA20 = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_EXCLUDED = ("C",)


def validate_request(
    original: str,
    edit_mask: list[int] | tuple[int, ...],
    order: str = "random",
    budget: int | None = None,
) -> list[int]:
    """Check a proposal request and return the normalised mask.

    Separate from `MlmProposer.propose` so it costs nothing and can be exercised without
    weights. The budget check is the important one: it refuses here, where the caller can
    see why, rather than letting `Proposal.__post_init__` reject every sequence much
    later with only the count to go on.
    """
    L = len(original)
    if L == 0:
        raise ValueError("original sequence is empty")
    mask = sorted({int(p) for p in edit_mask})
    if not mask:
        raise ValueError("edit_mask is empty: nothing to propose")
    if any(p < 0 or p >= L for p in mask):
        raise ValueError(f"edit_mask {mask} out of range for length {L}")
    cap = L // 2 if budget is None else int(budget)
    if len(mask) > cap:
        raise ValueError(
            f"edit_mask opens {len(mask)} positions but the budget is {cap} "
            f"(floor(L/2) for L={L}); the search gate would reject every proposal."
        )
    if order not in ("random", "confidence"):
        raise ValueError(f"order must be 'random' or 'confidence', got {order!r}")
    return mask


@dataclass
class MlmProposal:
    """One proposal, plus everything needed to attribute and reproduce it."""

    sequence: str
    edited_positions: tuple[int, ...]
    edit_mask: tuple[int, ...]
    order: str
    temperature: float
    seed: int
    logprob: float
    per_position: list[dict]

    def to_row(self) -> dict:
        return {
            "sequence": self.sequence,
            "edit_mask": ",".join(str(i) for i in self.edit_mask),
            "edited_positions": ",".join(str(i) for i in self.edited_positions),
            "unmask_order": self.order,
            "temperature": self.temperature,
            "proposal_seed": self.seed,
            "mlm_logprob": round(self.logprob, 6),
            "generator": "cpsea_mlm",
        }


class MlmProposer:
    """Loads a fine-tuned checkpoint once and serves proposals for many parents.

    The model is ~650M parameters (2.6 GB in fp32), so it is loaded once per process and
    the propose stage is a job of its own rather than something a fold loop does between
    evaluations -- the same arrangement the Llama proposer uses, and for the same reason.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float32",
        exclude: tuple[str, ...] = DEFAULT_EXCLUDED,
        max_length: int = 1024,
        pocket_enabled: bool | None = None,
    ):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"model path does not exist: {self.model_path}. This stage never "
                f"downloads weights; point it at a local fine-tuned checkpoint."
            )
        self.device = device
        self.dtype = dtype
        self.exclude = tuple(exclude)
        self.max_length = max_length
        self._pocket_override = pocket_enabled
        self._model = None
        self._tok = None
        self._encoder = None
        self._residue_ids = None
        self._allowed = None

    # -- loading ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        import sys

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        repo = Path(__file__).resolve().parents[3]
        if str(repo / "script_utils") not in sys.path:
            sys.path.insert(0, str(repo / "script_utils"))
        from cpseq_mlm_data import AA20 as DATA_AA20
        from cpseq_mlm_data import EsmMlmEncoder

        if DATA_AA20 != AA20:
            raise RuntimeError(
                f"residue alphabet disagrees between the proposer ({AA20}) and the data "
                f"module ({DATA_AA20}); logit column indices would be misaligned."
            )

        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.model_path, dtype=getattr(torch, self.dtype)
        )
        self._model.to(self.device).eval()

        pocket_enabled = self._pocket_override
        if pocket_enabled is None:
            # The arm is recorded in the training run's manifest. Guessing it would be a
            # silent error: feeding a pocket to a model trained without one puts the
            # peptide at an offset the model never saw.
            manifest = Path(self.model_path).parent.parent / "manifest.json"
            if manifest.exists():
                pocket_enabled = json.loads(manifest.read_text()).get("arm") != "no_pocket"
            else:
                pocket_enabled = True
        self.pocket_enabled = bool(pocket_enabled)

        self._encoder = EsmMlmEncoder(
            self._tok, max_length=self.max_length, pocket_enabled=self.pocket_enabled
        )
        self._residue_ids = torch.tensor(self._encoder.residue_ids(), device=self.device)
        keep = [i for i, a in enumerate(AA20) if a not in set(self.exclude)]
        self._allowed = torch.tensor(keep, device=self.device)

    @property
    def config(self) -> dict:
        """Everything needed to reproduce a draw, recorded on every round."""
        return {
            "model_path": self.model_path,
            "dtype": self.dtype,
            "excluded_residues": list(self.exclude),
            "pocket_enabled": getattr(self, "pocket_enabled", None),
        }

    # -- proposing ----------------------------------------------------------------

    def propose(
        self,
        *,
        original: str,
        edit_mask: list[int] | tuple[int, ...],
        pocket_seq: str = "",
        cyc_type: int = 3,
        n: int = 1,
        temperature: float = 1.0,
        order: str = "random",
        seed: int = 0,
        budget: int | None = None,
    ) -> list[MlmProposal]:
        """`n` proposals for one parent, rewriting only `edit_mask` positions.

        `budget` defaults to `len(original) // 2`, the search's own ceiling, and an
        `edit_mask` larger than it is refused here rather than producing sequences the
        `Proposal` dataclass will reject later, where the cause is no longer visible.
        """
        # Validate BEFORE loading: a bad call should fail in milliseconds, not after a
        # 2.6 GB weight load, and it makes the gate testable without weights on disk.
        mask = validate_request(original, edit_mask, order, budget)

        import torch

        self.load()
        L = len(original)

        from cpseq_mlm_data import CorpusRow

        out: list[MlmProposal] = []
        for draw in range(n):
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + draw)
            current = list(original)
            remaining = list(mask)
            if order == "random":
                perm = torch.randperm(len(remaining), generator=g).tolist()
                remaining = [remaining[i] for i in perm]
            total_lp = 0.0
            per_position: list[dict] = []

            while remaining:
                row = CorpusRow(
                    example_id="propose", pep_seq="".join(current),
                    pocket_seq=pocket_seq, cyc_type=cyc_type,
                    cyc_anchor_i=-1, cyc_anchor_j=-1,
                )
                enc = self._encoder.encode(row, remaining, cyc_type=cyc_type)
                if enc["too_long"]:
                    raise ValueError(
                        f"encoded length {enc['length']} exceeds max_length "
                        f"{self.max_length}; the pocket is too large for this model "
                        f"(ESM-2 allows 1026 positions)."
                    )
                ids = torch.tensor([enc["input_ids"]], device=self.device)
                attn = torch.tensor([enc["attention_mask"]], device=self.device)
                with torch.no_grad():
                    logits = self._model(input_ids=ids, attention_mask=attn).logits[0]
                # Restrict to the 20 canonical residues, then to the allowed subset. An
                # ESM-2 head can emit `X`, `B` or a special token, and a proposal
                # containing one is not a peptide.
                res_logits = logits[:, self._residue_ids][:, self._allowed].float()

                if order == "confidence":
                    # Fill the position the model is currently surest about. Measured on
                    # the max probability, not the entropy: it is the quantity the
                    # High-Confidence heuristic in the masked-diffusion literature uses.
                    probs = torch.softmax(res_logits, dim=-1)
                    conf = torch.tensor(
                        [float(probs[enc["pep_start"] + p].max()) for p in remaining]
                    )
                    pick_idx = int(conf.argmax())
                else:
                    pick_idx = 0
                pos = remaining.pop(pick_idx)

                slot = res_logits[enc["pep_start"] + pos]
                if temperature <= 0:
                    choice = int(slot.argmax())
                else:
                    probs = torch.softmax(slot / temperature, dim=-1)
                    choice = int(torch.multinomial(probs.cpu(), 1, generator=g))
                lp = float(torch.log_softmax(slot, dim=-1)[choice])
                aa = AA20[int(self._allowed[choice])]
                current[pos] = aa
                total_lp += lp
                per_position.append({"position": pos, "residue": aa,
                                     "logprob": round(lp, 6)})

            seq = "".join(current)
            edited = tuple(i for i in range(L) if seq[i] != original[i])
            out.append(MlmProposal(
                sequence=seq, edited_positions=edited, edit_mask=tuple(mask),
                order=order, temperature=temperature, seed=seed + draw,
                logprob=total_lp, per_position=per_position,
            ))
        return out

    # -- filling masks somebody else chose -----------------------------------------

    def fill(
        self,
        *,
        masks: list,
        pocket_seq: str,
        cyc_type: int,
        original: str,
        draws_per_mask: int = 1,
        temperature: float = 1.0,
        order: str = "random",
        seed: int = 0,
        budget: int | None = None,
    ) -> list[dict]:
        """Fill each of `masks`, returning one row per draw with its provenance.

        The division of labour this exists for: something else -- an LLM reading the
        interface, or a uniform draw -- decides WHICH positions to open, and this model
        decides what goes in them. `masks` is a list of objects carrying `.mask`,
        `.start_from` and optionally `.reason`/`.seed` (`llm_propose.MaskProposal`), or
        bare position tuples, in which case the fill starts from `original`.

        THE START IS NOT ALWAYS THE ORIGINAL. Filling a mask on an elite rewrites only
        the masked positions of THAT sequence, so the result keeps the elite's other
        substitutions. The budget is therefore still counted against `original`, and
        this function does not enforce it -- `llm_propose.screen_candidates` does, after
        the fact, because only the finished sequence says how many positions actually
        differ. A mask whose worst case already breaks the budget should have been
        refused by `screen_masks` before any of this ran.

        NO-OP FILLS ARE RETURNED, NOT DROPPED. The model may re-emit the residue that
        was already there, and at a 20-way softmax it sometimes will. Silently discarding
        those would overstate the proposal rate and hide a temperature that is too low;
        they are flagged `noop` and the caller decides.
        """
        rows: list[dict] = []
        for k, item in enumerate(masks):
            mask = tuple(getattr(item, "mask", item))
            start_from = getattr(item, "start_from", original)
            reason = getattr(item, "reason", "")
            mask_seed = int(getattr(item, "seed", 0))
            props = self.propose(
                original=start_from,
                edit_mask=list(mask),
                pocket_seq=pocket_seq,
                cyc_type=cyc_type,
                n=draws_per_mask,
                temperature=temperature,
                order=order,
                seed=seed + 1000 * k,
                # Against `start_from`, whose length equals the original's. The real
                # gate is `screen_candidates` against `original`.
                budget=len(start_from) if budget is None else budget,
            )
            for p in props:
                changed = tuple(
                    i for i in range(len(original))
                    if i < len(p.sequence) and p.sequence[i] != original[i]
                )
                rows.append({
                    "sequence": p.sequence,
                    "mask": ",".join(str(i) for i in mask),
                    "mask_size": len(mask),
                    "start_from": start_from,
                    "edited_vs_start": ",".join(str(i) for i in p.edited_positions),
                    "edited_vs_original": ",".join(str(i) for i in changed),
                    "n_mutations_vs_original": len(changed),
                    "noop": p.sequence == start_from,
                    "mlm_logprob": round(p.logprob, 6),
                    "unmask_order": p.order,
                    "temperature": p.temperature,
                    "fill_seed": p.seed,
                    "mask_seed": mask_seed,
                    "mask_reason": reason[:200],
                    "per_position": p.per_position,
                    "generator": "cpsea_mlm",
                })
        return rows

    # -- scoring ------------------------------------------------------------------

    def score(
        self,
        *,
        sequence: str,
        pocket_seq: str = "",
        cyc_type: int = 3,
        positions: list[int] | None = None,
    ) -> dict:
        """Pseudo-log-likelihood of `sequence` at `positions` (default: all).

        One forward per scored position, each with that position alone masked -- the
        standard masked-LM pseudo-perplexity, and the quantity PepMLM reports. It is NOT
        a joint likelihood and must not be compared against an autoregressive model's
        perplexity as though it were.
        """
        import torch

        self.load()
        from cpseq_mlm_data import CorpusRow

        pos_list = list(range(len(sequence))) if positions is None else list(positions)
        total = 0.0
        rows = []
        for p in pos_list:
            row = CorpusRow(
                example_id="score", pep_seq=sequence, pocket_seq=pocket_seq,
                cyc_type=cyc_type, cyc_anchor_i=-1, cyc_anchor_j=-1,
            )
            enc = self._encoder.encode(row, [p], cyc_type=cyc_type)
            if enc["too_long"]:
                raise ValueError(
                    f"encoded length {enc['length']} exceeds max_length "
                    f"{self.max_length}; the pocket is too large for this model."
                )
            ids = torch.tensor([enc["input_ids"]], device=self.device)
            attn = torch.tensor([enc["attention_mask"]], device=self.device)
            with torch.no_grad():
                logits = self._model(input_ids=ids, attention_mask=attn).logits[0]
            slot = logits[enc["pep_start"] + p, self._residue_ids].float()
            lp = torch.log_softmax(slot, dim=-1)
            true_idx = AA20.index(sequence[p]) if sequence[p] in AA20 else None
            if true_idx is None:
                continue
            total += float(lp[true_idx])
            rows.append({"position": p, "residue": sequence[p],
                         "logprob": round(float(lp[true_idx]), 6)})
        n = max(len(rows), 1)
        import math

        return {
            "pseudo_logprob": total,
            "pseudo_perplexity": math.exp(-total / n),
            "n_scored": len(rows),
            "per_position": rows,
        }
