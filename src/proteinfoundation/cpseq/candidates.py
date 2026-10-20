"""Candidate sequence construction: equal-length substitution variants of an LP reference.

Task shape
----------
Given a linear-peptide reference sequence, produce candidates at roughly 12.5 / 25 /
37.5 / 50 % mutation, two distinct variants per nonzero band, plus the unchanged
sequence: **nine candidates per reference** where the length allows it. Substitutions
only, equal length -- insertions and deletions are out of scope for this version.

Four things this module is careful about, each of which would otherwise corrupt the
mutation-fraction label that a future scorer would be trained against:

1. **Nominal band != actual fraction.** 12.5 % of a 13-mer is 1.625 residues. The
   candidate carries the *actual* integer count and the *actual* fraction `k / L`, and
   the nominal band is kept only as a grouping tag. A row that said "0.125" when 2 of 13
   residues changed would be a 0.154 example mislabelled.

2. **Chemistry anchors are mutations too.** A terminal disulfide needs CYS at both ends.
   If the reference does not already have them, satisfying the chemistry *is* a
   substitution, and it is placed and counted FIRST, before any free position is
   sampled. A band whose budget is smaller than its anchor requirement is reported as
   infeasible rather than silently overspent.

3. **An identical replacement is not a mutation.** Sampling the residue that is already
   there would inflate the count while changing nothing, so replacements are drawn from
   the alphabet minus the current residue.

4. **Short sequences make bands collide.** On a 5-mer, 12.5 % and 25 % both round to 1
   substitution. Dedup is on the realised *sequence*, so two bands that coincide
   contribute one candidate carrying both band tags, instead of two rows that look like
   independent samples of the same string.

Determinism
-----------
Every candidate's seed is derived from `(parent_id, chemistry, nominal_band,
variant_index)` through a stable hash, so a candidate is reproducible from its recorded
seed alone and adding a parent does not renumber anyone else's draws.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from proteinfoundation.cpseq.chemistry import CHEMISTRIES, ChemistrySpec

# Canonical 20, one-letter.
AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

# Nominal mutation bands. 0.0 is the unchanged reference.
DEFAULT_BANDS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5)
DEFAULT_VARIANTS_PER_BAND = 2

# Cysteine is excluded from *random* substitution by default. A stray free cysteine can
# pair with another one and close a ring we did not ask for, which would make the
# molecule's chemistry differ from the chemistry recorded in its own label. Anchor
# placement still introduces CYS where a chemistry requires it -- that is requested, not
# adventitious. Flip `allow_cys` to include it.
DEFAULT_EXCLUDED_FROM_RANDOM = ("C",)


def _seed_for(parent_id: str, chemistry: str, band: float, variant_index: int) -> int:
    """Stable 63-bit seed from the candidate's identity, not from a global counter."""
    key = f"{parent_id}|{chemistry}|{band:.4f}|{variant_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


@dataclass
class Candidate:
    """One candidate sequence, with everything needed to reproduce and audit it."""

    candidate_id: str
    parent_id: str
    chemistry: str
    sequence: str
    reference_sequence: str
    # 0-based positions whose residue differs from the reference, sorted.
    mutated_positions: tuple[int, ...]
    # Of those, the ones forced by the chemistry's anchor requirements.
    anchor_positions: tuple[int, ...]
    n_mutations: int
    mutation_fraction: float
    nominal_bands: tuple[float, ...]
    variant_index: int
    seed: int
    is_reference: bool
    # Set when the request could not be met exactly; the candidate is still emitted with
    # what was achievable, and this says what gave way.
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.sequence) != len(self.reference_sequence):
            raise ValueError(
                f"{self.candidate_id}: substitutions only, but length changed "
                f"{len(self.reference_sequence)} -> {len(self.sequence)}"
            )
        observed = tuple(
            i for i, (a, b) in enumerate(zip(self.reference_sequence, self.sequence)) if a != b
        )
        if observed != tuple(sorted(self.mutated_positions)):
            raise ValueError(
                f"{self.candidate_id}: recorded mutated_positions "
                f"{sorted(self.mutated_positions)} disagree with the sequence diff "
                f"{list(observed)} -- the label would not describe the string"
            )
        if self.n_mutations != len(observed):
            raise ValueError(
                f"{self.candidate_id}: n_mutations={self.n_mutations} but {len(observed)} "
                "positions differ"
            )

    def to_row(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "chemistry": self.chemistry,
            "sequence": self.sequence,
            "reference_sequence": self.reference_sequence,
            "peptide_length": len(self.sequence),
            "mutated_positions": ",".join(str(p) for p in self.mutated_positions),
            "anchor_positions": ",".join(str(p) for p in self.anchor_positions),
            "n_mutations": self.n_mutations,
            "mutation_fraction": round(self.mutation_fraction, 6),
            "nominal_bands": ",".join(f"{b:.3f}" for b in self.nominal_bands),
            "variant_index": self.variant_index,
            "generation_seed": self.seed,
            "is_reference": self.is_reference,
            "note": self.note,
        }


def _anchor_edits(
    reference: str, spec: ChemistrySpec
) -> tuple[dict[int, str], list[int]]:
    """{position -> required residue} and the subset that actually changes the sequence.

    The first element covers every anchor position (changed or not), because those
    positions are **protected** from random substitution either way: a disulfide anchor
    that is already CYS must stay CYS.
    """
    required = spec.anchor_requirements(len(reference))
    fixed: dict[int, str] = {}
    changed: list[int] = []
    for pos, allowed3 in required.items():
        allowed1 = [THREE_TO_ONE[a] for a in allowed3 if a in THREE_TO_ONE]
        if not allowed1:
            raise ValueError(f"chemistry {spec.name} requires non-canonical residue at {pos}")
        if reference[pos] in allowed1:
            fixed[pos] = reference[pos]  # already satisfied; still protected
        else:
            # Deterministic choice: the first allowed residue, so an isopeptide acid
            # anchor does not vary between runs.
            fixed[pos] = allowed1[0]
            changed.append(pos)
    return fixed, sorted(changed)


def build_candidates(
    parent_id: str,
    reference: str,
    chemistry: str,
    *,
    bands: Sequence[float] = DEFAULT_BANDS,
    variants_per_band: int = DEFAULT_VARIANTS_PER_BAND,
    allow_cys: bool = False,
    alphabet: Sequence[str] = AA20,
) -> list[Candidate]:
    """Nine-ish candidates for one (parent, chemistry): the reference plus variants.

    Returns candidates in a stable order: the reference first, then by nominal band and
    variant index. Sequences are deduplicated -- a repeat carries its extra band tag on
    the surviving row instead of producing a second row.
    """
    reference = reference.upper()
    bad = sorted(set(reference) - set(AA20))
    if bad:
        raise ValueError(
            f"{parent_id}: reference has non-canonical residues {bad}; this version is "
            "canonical-residue only"
        )
    spec = CHEMISTRIES[chemistry]
    length = len(reference)

    fixed, anchor_changed = _anchor_edits(reference, spec)
    protected = set(fixed)
    free_positions = [p for p in range(length) if p not in protected]

    pool = [a for a in alphabet if allow_cys or a not in DEFAULT_EXCLUDED_FROM_RANDOM]

    def apply_anchors(seq: list[str]) -> list[str]:
        for pos, res in fixed.items():
            seq[pos] = res
        return seq

    out: list[Candidate] = []
    # sequence -> index into `out`, for the dedup-with-band-merge.
    seen: dict[str, int] = {}

    for band in bands:
        # The reference arm: anchors still apply, because the molecule must be able to
        # close. An anchored "reference" is therefore not always identical to the LP
        # sequence, and its n_mutations says so.
        if band == 0.0:
            seq = "".join(apply_anchors(list(reference)))
            cand = Candidate(
                candidate_id=f"{parent_id}__{chemistry}__ref",
                parent_id=parent_id,
                chemistry=chemistry,
                sequence=seq,
                reference_sequence=reference,
                mutated_positions=tuple(anchor_changed),
                anchor_positions=tuple(anchor_changed),
                n_mutations=len(anchor_changed),
                mutation_fraction=len(anchor_changed) / length,
                nominal_bands=(0.0,),
                variant_index=0,
                seed=_seed_for(parent_id, chemistry, 0.0, 0),
                is_reference=True,
                note=(
                    "anchor substitutions required by the chemistry are present in the "
                    "'unchanged' arm and counted"
                    if anchor_changed
                    else ""
                ),
            )
            seen[seq] = len(out)
            out.append(cand)
            continue

        # Target integer count. Round half away from zero so 12.5 % of an 8-mer is 1, not
        # 0 -- a band that rounded to zero would silently duplicate the reference arm.
        target = int(np.floor(band * length + 0.5))
        target = max(target, 1)
        # Hard cap at floor(L / 2), because the task's retention floor is ~50 % identity
        # and rounding up at an odd length would breach it: 50 % of a 13-mer rounds to 7
        # substitutions, which is 46 % identity -- outside the constraint the dataset is
        # supposed to respect. The cap binds only at the top band and odd lengths, and
        # `note` records it when it does.
        identity_cap = length // 2
        if target > identity_cap:
            target = identity_cap

        for variant_index in range(variants_per_band):
            seed = _seed_for(parent_id, chemistry, band, variant_index)
            rng = np.random.default_rng(seed)

            note = ""
            if target < int(np.floor(band * length + 0.5)):
                note = (
                    f"band {band:.3f} capped at {target} substitutions (floor(L/2)) to "
                    "keep identity to the reference at or above 50%"
                )
            n_free_needed = target - len(anchor_changed)
            if n_free_needed < 0:
                # The chemistry alone already costs more than this band allows.
                note = (
                    f"band {band:.3f} allows {target} substitutions but the chemistry "
                    f"requires {len(anchor_changed)} anchor substitutions; emitted at the "
                    "anchor-only count"
                )
                n_free_needed = 0
            if n_free_needed > len(free_positions):
                note = (
                    f"band {band:.3f} wants {n_free_needed} free substitutions but only "
                    f"{len(free_positions)} unprotected positions exist; emitted at the "
                    "maximum"
                )
                n_free_needed = len(free_positions)

            chosen = (
                rng.choice(free_positions, size=n_free_needed, replace=False)
                if n_free_needed
                else np.array([], dtype=int)
            )
            seq_list = apply_anchors(list(reference))
            for pos in sorted(int(p) for p in chosen):
                # Draw from the alphabet minus whatever is already there, so every draw
                # is a real change and the count cannot be inflated by a no-op.
                options = [a for a in pool if a != reference[pos]]
                seq_list[pos] = str(rng.choice(options))
            seq = "".join(seq_list)

            if seq in seen:
                # Bands collided (short sequence) or the same draw recurred. Tag the
                # surviving row with this band rather than emitting a duplicate.
                prev = out[seen[seq]]
                if band not in prev.nominal_bands:
                    prev.nominal_bands = tuple(sorted(set(prev.nominal_bands) | {band}))
                    prev.note = (prev.note + "; " if prev.note else "") + (
                        f"also realises nominal band {band:.3f} (bands coincide at this "
                        "length)"
                    )
                continue

            mutated = tuple(
                i for i, (a, b) in enumerate(zip(reference, seq)) if a != b
            )
            cand = Candidate(
                candidate_id=f"{parent_id}__{chemistry}__b{int(round(band * 1000)):03d}_v{variant_index}",
                parent_id=parent_id,
                chemistry=chemistry,
                sequence=seq,
                reference_sequence=reference,
                mutated_positions=mutated,
                anchor_positions=tuple(anchor_changed),
                n_mutations=len(mutated),
                mutation_fraction=len(mutated) / length,
                nominal_bands=(band,),
                variant_index=variant_index,
                seed=seed,
                is_reference=False,
                note=note,
            )
            seen[seq] = len(out)
            out.append(cand)

    return out


def identity_to_reference(candidate: Candidate) -> float:
    """Fraction of positions still equal to the reference.

    The task's retention floor is ~50 % identity; at equal length this is exactly
    `1 - mutation_fraction`, and it is computed from the strings rather than from the
    recorded fraction so the two can be cross-checked.
    """
    ref, seq = candidate.reference_sequence, candidate.sequence
    same = sum(1 for a, b in zip(ref, seq) if a == b)
    return same / len(ref)
