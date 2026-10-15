"""RFdiffusion3 training condition for the LP arm: peptide binder, sequence revealed.

RFD3's stock ``PPICondition`` is de-novo binder design -- it fixes the target and generates
the binder's structure *and* sequence. The LP milestone needs something different: the
peptide sequence is an input, and only its geometry (plus, in one of the two tasks, a few
identities) is generated. Two specific things in the stock condition are wrong for us, and
one piece of plumbing constrains how we may fix them.

WHY THIS SUBCLASSES ``PPICondition`` AND KEEPS ``name = "ppi"``. The literal string
``"ppi"`` is load-bearing in five places in rfd3, not one:

  * ``transforms/pipelines.py:227``  gates ``PPIFullBinderCropSpatial`` -- the crop that
    keeps the binder whole. The plan requires retaining full peptide chains, and this is
    the mechanism that does it.
  * ``transforms/pipelines.py:458``  a second ppi-only route
  * ``transforms/design_transforms.py:589`` and ``:637``
  * ``transforms/hbonds.py:304``

A condition registered under any other name silently skips all five, including the
full-binder crop, and nothing errors -- the peptide just gets sliced by a generic spatial
crop sometimes. So we keep the name and override only what must change.

WHAT CHANGES.

1.  **The binder is pinned, not flipped.** Stock does
    ``binder_pn_unit = np.random.choice(self.query_pn_unit_iids)``, so half the time the
    *receptor* becomes the thing to generate and the peptide becomes the fixed motif --
    exactly backwards. Measured on our manifest, the peptide is the smaller protein
    ``pn_unit`` in 6/6 examples (``A_1`` = receptor at 74-417 residues, ``B_1`` = peptide
    at 5-14), and those counts match the manifest's own length columns, so "smaller unit"
    is an exact handle. A chain-id selector is available as a cross-check.

2.  **The peptide sequence is revealed, its geometry is not.** Stock reveals nothing about
    the binder. The native per-probability knobs cannot express this: both
    ``sample_is_motif_atom_with_fixed_seq`` and ``sample_fix_motif_coordinates`` are
    all-or-nothing *per-example* coin flips over the whole motif set, so they cannot give
    the receptor and the peptide different treatment. ``PPICondition`` itself sets the
    annotations directly instead of calling ``sample_conditioning_strategy``, and that is
    the pattern followed here.

THE TWO TASKS, as the plan specifies them:

  * with probability ``p_full_sequence`` (default 0.5): reveal every peptide identity and
    generate only geometry;
  * otherwise: reveal a uniform random fraction in ``[reveal_frac_min, reveal_frac_max]``
    (default 0.80-0.95) of peptide identities, generating the remaining identities *and*
    all of the geometry.

The revealed fraction is realised by choosing which peptide tokens are motif, which is the
native token-level mechanism, rather than by patching the per-atom sequence sampler.

The resulting annotations:

    receptor tokens          motif, sequence fixed, coordinates fixed, indexed
    revealed peptide tokens  motif, sequence fixed, coordinates NOT fixed, indexed
    hidden peptide tokens    not motif at all

``is_motif_atom`` stays receptor-only: we supply the peptide's *identity*, never its atom
positions, so its geometry is always generated. The training target is always the original
LP, so this teaches conditional completion -- it is not supervised improvement from one
sequence to a better one.
"""

from __future__ import annotations

import logging

import numpy as np

# stdlib logging, NOT loguru: this module is imported inside the foundry environment, which
# does not ship loguru. An import this class cannot satisfy there turns a condition lookup
# into a training-time crash.
logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only inside the foundry environment
    from rfd3.transforms.training_conditions import PPICondition
except ImportError:  # pragma: no cover
    PPICondition = None


#: Selector strategies for deciding which ``pn_unit`` is the peptide.
SELECTORS = ("smaller", "chain")


def _token_counts_by_pn_unit(atom_array, pn_units) -> dict:
    """Residue (token) count per ``pn_unit_iid``.

    Counted on residues rather than atoms: the receptor can be smaller than the peptide in
    atom terms only in pathological cases, but residue counts are what the manifest records
    and what "peptide length" means everywhere else in this project.
    """
    counts = {}
    for unit in pn_units:
        sub = atom_array[atom_array.pn_unit_iid == unit]
        if sub.array_length() == 0:
            counts[unit] = 0
            continue
        # A token is a residue: (res_id, ins_code) within the unit.
        ins = getattr(sub, "ins_code", None)
        keys = (
            list(zip(sub.res_id, ins)) if ins is not None else list(zip(sub.res_id))
        )
        counts[unit] = len(set(keys))
    return counts


def _residue_keys(atom_array) -> np.ndarray:
    """Per-atom residue identity, as an integer group index.

    Built from ``(chain_iid, res_id, ins_code)`` rather than ``res_id`` alone: a bare
    residue number repeats across chains and collides across insertion codes, which would
    merge two residues into one token and silently reveal or hide the wrong identity.
    """
    chain = np.asarray(
        getattr(atom_array, "chain_iid", getattr(atom_array, "chain_id")), dtype=str
    )
    res_id = np.asarray(atom_array.res_id, dtype=str)
    ins = getattr(atom_array, "ins_code", None)
    ins = np.asarray(ins, dtype=str) if ins is not None else np.zeros(len(res_id), dtype=str)
    composite = np.char.add(np.char.add(np.char.add(chain, "|"), res_id), ins)
    _, inverse = np.unique(composite, return_inverse=True)
    return inverse


class LPBinderCondition(PPICondition if PPICondition is not None else object):
    """Peptide-binder condition with a pinned binder and a revealed peptide sequence."""

    #: MUST remain "ppi" -- five places in rfd3 branch on this literal. See the module
    #: docstring; renaming it silently disables the full-binder crop among other things.
    name = "ppi"

    def __init__(
        self,
        frequency: float = 1.0,
        *,
        selector: str = "smaller",
        peptide_chain_id: str = "B",
        p_full_sequence: float = 0.5,
        reveal_frac_min: float = 0.80,
        reveal_frac_max: float = 0.95,
        warn_on_selector_disagreement: bool = True,
    ):
        if PPICondition is not None:
            super().__init__(frequency)
        else:  # pragma: no cover - only when rfd3 is absent (unit tests)
            self.frequency = frequency
        if selector not in SELECTORS:
            raise ValueError(f"selector must be one of {SELECTORS}, got {selector!r}")
        if not 0.0 <= reveal_frac_min <= reveal_frac_max <= 1.0:
            raise ValueError(
                f"need 0 <= reveal_frac_min <= reveal_frac_max <= 1, got "
                f"{reveal_frac_min} / {reveal_frac_max}"
            )
        self.selector = selector
        self.peptide_chain_id = peptide_chain_id
        self.p_full_sequence = p_full_sequence
        self.reveal_frac_min = reveal_frac_min
        self.reveal_frac_max = reveal_frac_max
        self.warn_on_selector_disagreement = warn_on_selector_disagreement
        #: Set for the caller/tests to inspect what the last example actually did.
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ binder choice
    def pick_binder(self, atom_array, pn_units) -> str:
        """Choose the peptide ``pn_unit_iid`` deterministically.

        Deterministic is the whole point: stock ``PPICondition`` flips a coin here, which
        would make the receptor the generated chain on half of all examples.
        """
        pn_units = [str(u) for u in pn_units]
        counts = _token_counts_by_pn_unit(atom_array, pn_units)
        by_size = min(pn_units, key=lambda u: (counts[u], u))

        by_chain = None
        chain_attr = getattr(atom_array, "chain_id", None)
        if chain_attr is not None:
            for unit in pn_units:
                sub = atom_array[atom_array.pn_unit_iid == unit]
                if sub.array_length() and self.peptide_chain_id in set(
                    np.asarray(sub.chain_id, dtype=str)
                ):
                    by_chain = unit
                    break

        chosen = by_size if self.selector == "smaller" else (by_chain or by_size)
        if (
            self.warn_on_selector_disagreement
            and by_chain is not None
            and by_chain != by_size
        ):
            # Not fatal, but it means one of the two handles is wrong for this example and
            # the binder assignment deserves a look rather than a silent coin-flip's worth
            # of doubt.
            logger.warning(
                "LPBinderCondition: selectors disagree -- smallest unit %r (%d tokens) vs "
                "chain %r unit %r; using %r (selector=%s)",
                by_size, counts[by_size], self.peptide_chain_id, by_chain,
                chosen, self.selector,
            )
        return chosen

    def is_valid_for_example(self, data) -> bool:
        """Delegate validation upstream, then pin the binder.

        Upstream's checks are reused verbatim (exactly two distinct protein
        ``query_pn_unit_iids``, both protein) so this stays correct as they evolve; only
        its random binder pick is replaced afterwards.

        Note those checks require ``query_pn_unit_iids`` to be populated, which happens
        only when the dataset's ``GenericDFParser`` is given ``pn_unit_iid_colnames``.
        Measured: with the columns supplied, upstream accepts 6/6 of our examples; with the
        distillation template's ``null``, 0/3.
        """
        if not super().is_valid_for_example(data):
            return False

        atom_array = data["atom_array"]
        binder = self.pick_binder(atom_array, self.query_pn_unit_iids)
        data["binder_pn_unit"] = binder
        atom_array.set_annotation(
            "is_binder_pn_unit", np.asarray(atom_array.pn_unit_iid, dtype=str) == binder
        )
        return True

    # ------------------------------------------------------------------ mask sampling
    def _sample_revealed_tokens(self, atom_array, is_binder: np.ndarray) -> np.ndarray:
        """Per-atom mask of peptide atoms whose residue identity is revealed."""
        groups = _residue_keys(atom_array)
        binder_groups = np.unique(groups[is_binder])
        n = len(binder_groups)

        if np.random.random() < self.p_full_sequence:
            frac, n_reveal, task = 1.0, n, "full_sequence"
        else:
            frac = float(np.random.uniform(self.reveal_frac_min, self.reveal_frac_max))
            # Round rather than floor, and never hide every residue: a 5-mer at frac 0.80
            # floors to 4, which is fine, but guard the degenerate ends explicitly.
            n_reveal = int(round(frac * n))
            n_reveal = max(0, min(n, n_reveal))
            task = "partial_sequence"

        revealed_groups = (
            binder_groups
            if n_reveal >= n
            else np.random.choice(binder_groups, size=n_reveal, replace=False)
        )
        revealed = is_binder & np.isin(groups, revealed_groups)

        self.last_stats = {
            "task": task,
            "requested_frac": frac,
            "n_binder_tokens": int(n),
            "n_revealed_tokens": int(len(revealed_groups)),
            "realised_frac": float(len(revealed_groups) / n) if n else 0.0,
        }
        return revealed

    def sample(self, data):
        """Set the motif annotations directly, per token.

        Direct assignment rather than ``sample_conditioning_strategy`` because that helper's
        probabilities are per-example all-or-nothing over the entire motif set and so
        cannot treat the receptor and the peptide differently. ``PPICondition`` sets these
        annotations directly for the same reason.
        """
        atom_array = data["atom_array"]
        pn_unit = np.asarray(atom_array.pn_unit_iid, dtype=str)
        is_binder = pn_unit == str(data["binder_pn_unit"])
        is_target = ~is_binder

        revealed = self._sample_revealed_tokens(atom_array, is_binder)

        # Motif = the target, plus the peptide residues whose identity we hand over.
        atom_array.set_annotation("is_motif_token", is_target | revealed)
        # Identity is given for the target and for the revealed peptide residues.
        atom_array.set_annotation("is_motif_atom_with_fixed_seq", is_target | revealed)
        # Atoms are given for the target ONLY: the peptide's geometry is always generated,
        # which is what makes this pose prediction from a sequence rather than copying.
        atom_array.set_annotation("is_motif_atom", is_target)
        atom_array.set_annotation("is_motif_atom_with_fixed_coord", is_target)
        # Indices are fixed throughout; nothing is unindexed and there are no breakpoints.
        zeros = np.zeros_like(is_binder)
        atom_array.set_annotation("is_motif_atom_unindexed", zeros)
        atom_array.set_annotation("is_motif_atom_unindexed_motif_breakpoint", zeros)

        self.last_stats.update(
            {
                "n_target_atoms": int(is_target.sum()),
                "n_binder_atoms": int(is_binder.sum()),
                "n_revealed_atoms": int(revealed.sum()),
            }
        )
        return atom_array
