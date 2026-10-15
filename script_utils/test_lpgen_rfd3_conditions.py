"""Tests for the RFD3 LP binder condition.

These run in the PROJECT venv, where rfd3 is not installed, so they exercise everything
that does not need upstream: the binder pick, the revealed-token sampling and the
annotations written by ``sample()``. ``is_valid_for_example`` delegates to upstream and is
covered by the in-env preflight instead.

What is worth testing here is what would be invisible in a log line and wrong in training:

the binder pick must be deterministic
    Stock ``PPICondition`` flips a coin between the two chains, which makes the receptor
    the generated chain half the time. A pinned pick is the entire point, so it is checked
    over many calls rather than once.

whole residues, never partial ones
    Identity is revealed per token. If the atom->residue grouping is wrong, a residue can
    end up half-revealed, which still trains and is silently incoherent. Residue identity
    is ``(chain, res_id, ins_code)``: a bare ``res_id`` repeats across chains and collides
    across insertion codes.

the four annotations must differ between receptor and peptide
    That asymmetry is the thing the native per-example probabilities cannot express, and
    it is the reason this class sets annotations directly. In particular
    ``is_motif_atom``/``is_motif_atom_with_fixed_coord`` must be receptor-only, or the
    peptide's coordinates get handed to the model and the task collapses into copying.

Run:  .venv/bin/python script_utils/test_lpgen_rfd3_conditions.py
      .venv/bin/python -m pytest -q script_utils/test_lpgen_rfd3_conditions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.lpgen.rfd3_conditions import (  # noqa: E402
    LPBinderCondition,
    _residue_keys,
    _token_counts_by_pn_unit,
)


class FakeAtomArray:
    """Duck-typed stand-in for a biotite AtomArray with the annotations we read."""

    def __init__(self, pn_unit_iid, chain_id, res_id, ins_code=None):
        self.pn_unit_iid = np.asarray(pn_unit_iid, dtype=object)
        self.chain_id = np.asarray(chain_id, dtype=object)
        self.chain_iid = np.asarray(chain_id, dtype=object)
        self.res_id = np.asarray(res_id)
        self.ins_code = (
            np.asarray(ins_code, dtype=object)
            if ins_code is not None
            else np.array([""] * len(self.res_id), dtype=object)
        )
        self.annotations: dict = {}

    def array_length(self):
        return len(self.res_id)

    def set_annotation(self, name, value):
        self.annotations[name] = np.asarray(value)
        setattr(self, name, self.annotations[name])

    def __getitem__(self, mask):
        mask = np.asarray(mask)
        return FakeAtomArray(
            self.pn_unit_iid[mask], self.chain_id[mask], self.res_id[mask],
            self.ins_code[mask],
        )


def make_complex(n_rec_res=20, n_pep_res=8, atoms_per_res=4):
    """Receptor chain A (pn_unit A_1) plus peptide chain B (pn_unit B_1)."""
    pn, ch, res = [], [], []
    for i in range(n_rec_res):
        pn += ["A_1"] * atoms_per_res
        ch += ["A"] * atoms_per_res
        res += [i] * atoms_per_res
    for i in range(n_pep_res):
        pn += ["B_1"] * atoms_per_res
        ch += ["B"] * atoms_per_res
        res += [i] * atoms_per_res
    return FakeAtomArray(pn, ch, res)


def _cond(**kw):
    return LPBinderCondition(**kw)


# -------------------------------------------------------------------------- basics
def test_name_must_stay_ppi():
    """Five places in rfd3 branch on this literal; renaming disables the binder crop."""
    assert LPBinderCondition.name == "ppi"


def test_constructor_validates():
    for bad in ({"selector": "nope"},
                {"reveal_frac_min": 0.9, "reveal_frac_max": 0.5},
                {"reveal_frac_min": -0.1},
                {"reveal_frac_max": 1.5}):
        try:
            _cond(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_token_counts_are_residues_not_atoms():
    aa = make_complex(n_rec_res=20, n_pep_res=8, atoms_per_res=4)
    counts = _token_counts_by_pn_unit(aa, ["A_1", "B_1"])
    assert counts == {"A_1": 20, "B_1": 8}, counts


def test_residue_keys_do_not_merge_across_chains_or_insertion_codes():
    # Same res_id 5 in two chains, and 5 vs 5A in one chain: three distinct residues.
    aa = FakeAtomArray(
        pn_unit_iid=["A_1", "A_1", "B_1", "B_1", "B_1", "B_1"],
        chain_id=["A", "A", "B", "B", "B", "B"],
        res_id=[5, 5, 5, 5, 5, 5],
        ins_code=["", "", "", "", "A", "A"],
    )
    groups = _residue_keys(aa)
    assert len(np.unique(groups)) == 3, np.unique(groups)
    # The two chain-A atoms are one residue, distinct from both chain-B residues.
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[4] == groups[5]
    assert len({groups[0], groups[2], groups[4]}) == 3


# ------------------------------------------------------------------- binder pick
def test_binder_pick_is_the_smaller_unit_and_deterministic():
    aa = make_complex(n_rec_res=50, n_pep_res=9)
    c = _cond()
    picks = {c.pick_binder(aa, ["A_1", "B_1"]) for _ in range(200)}
    assert picks == {"B_1"}, picks
    # Order of the query list must not matter either.
    assert c.pick_binder(aa, ["B_1", "A_1"]) == "B_1"


def test_binder_pick_by_chain_selector():
    aa = make_complex(n_rec_res=50, n_pep_res=9)
    assert _cond(selector="chain", peptide_chain_id="B").pick_binder(aa, ["A_1", "B_1"]) == "B_1"
    # Asking for the receptor's chain picks the receptor -- the selector is obeyed, which
    # is what makes a misconfiguration visible rather than silently corrected.
    assert _cond(selector="chain", peptide_chain_id="A").pick_binder(aa, ["A_1", "B_1"]) == "A_1"


def test_selector_disagreement_still_returns_the_configured_choice():
    # Pathological: chain B is the LARGER unit. "smaller" must still pick A_1.
    aa = make_complex(n_rec_res=5, n_pep_res=30)
    c = _cond(selector="smaller", peptide_chain_id="B")
    assert c.pick_binder(aa, ["A_1", "B_1"]) == "A_1"


# ------------------------------------------------------------ annotations / tasks
def _run_sample(c, aa, binder="B_1"):
    data = {"atom_array": aa, "binder_pn_unit": binder}
    c.sample(data)
    return aa.annotations


def test_annotations_split_receptor_and_peptide():
    aa = make_complex(n_rec_res=20, n_pep_res=8)
    c = _cond(p_full_sequence=1.0)  # reveal everything, the simplest case
    ann = _run_sample(c, aa)

    is_pep = np.asarray(aa.pn_unit_iid, dtype=str) == "B_1"
    is_rec = ~is_pep

    # Identity: given for both (the whole sequence is revealed in this task).
    assert ann["is_motif_atom_with_fixed_seq"].all()
    assert ann["is_motif_token"].all()
    # Coordinates and atoms: receptor ONLY. This is the load-bearing asymmetry -- if the
    # peptide's coordinates were given, the task would be copying, not pose prediction.
    assert (ann["is_motif_atom"] == is_rec).all()
    assert (ann["is_motif_atom_with_fixed_coord"] == is_rec).all()
    assert not ann["is_motif_atom"][is_pep].any()
    assert not ann["is_motif_atom_with_fixed_coord"][is_pep].any()
    # Nothing unindexed.
    assert not ann["is_motif_atom_unindexed"].any()
    assert not ann["is_motif_atom_unindexed_motif_breakpoint"].any()


def test_full_sequence_task_reveals_every_peptide_token():
    aa = make_complex(n_pep_res=11)
    c = _cond(p_full_sequence=1.0)
    _run_sample(c, aa)
    assert c.last_stats["task"] == "full_sequence"
    assert c.last_stats["n_revealed_tokens"] == 11
    assert c.last_stats["realised_frac"] == 1.0


def test_partial_task_reveals_a_fraction_and_hides_the_rest():
    np.random.seed(0)
    aa_template = dict(n_rec_res=20, n_pep_res=20)
    c = _cond(p_full_sequence=0.0, reveal_frac_min=0.80, reveal_frac_max=0.95)
    fracs = []
    for _ in range(60):
        aa = make_complex(**aa_template)
        ann = _run_sample(c, aa)
        assert c.last_stats["task"] == "partial_sequence"
        fracs.append(c.last_stats["realised_frac"])

        is_pep = np.asarray(aa.pn_unit_iid, dtype=str) == "B_1"
        # Some peptide identity is hidden, and hidden peptide tokens are not motif at all.
        hidden = is_pep & ~ann["is_motif_atom_with_fixed_seq"]
        assert hidden.any(), "partial task must hide at least one residue"
        assert not ann["is_motif_token"][hidden].any()
        # Receptor is never hidden.
        assert ann["is_motif_atom_with_fixed_seq"][~is_pep].all()

    assert 0.78 <= min(fracs) <= max(fracs) <= 0.96, (min(fracs), max(fracs))
    assert len(set(fracs)) > 1, "the revealed fraction must actually vary"


def test_reveal_is_whole_residues_never_partial():
    """A revealed residue has ALL its atoms revealed; a hidden one none."""
    np.random.seed(1)
    c = _cond(p_full_sequence=0.0, reveal_frac_min=0.5, reveal_frac_max=0.5)
    for _ in range(30):
        aa = make_complex(n_rec_res=6, n_pep_res=10, atoms_per_res=5)
        ann = _run_sample(c, aa)
        groups = _residue_keys(aa)
        is_pep = np.asarray(aa.pn_unit_iid, dtype=str) == "B_1"
        revealed = ann["is_motif_atom_with_fixed_seq"]
        for g in np.unique(groups[is_pep]):
            vals = set(revealed[groups == g].tolist())
            assert len(vals) == 1, f"residue {g} is half revealed: {vals}"


def test_task_mixture_is_roughly_the_requested_split():
    np.random.seed(2)
    c = _cond(p_full_sequence=0.5)
    tasks = []
    for _ in range(400):
        aa = make_complex(n_pep_res=12)
        _run_sample(c, aa)
        tasks.append(c.last_stats["task"])
    full = tasks.count("full_sequence") / len(tasks)
    assert 0.40 < full < 0.60, full


def test_short_peptide_does_not_degenerate():
    """A 5-mer at 0.80-0.95 must still reveal most of itself and hide at most a little."""
    np.random.seed(3)
    c = _cond(p_full_sequence=0.0, reveal_frac_min=0.80, reveal_frac_max=0.95)
    for _ in range(40):
        aa = make_complex(n_rec_res=10, n_pep_res=5)
        _run_sample(c, aa)
        n = c.last_stats["n_revealed_tokens"]
        assert 4 <= n <= 5, n


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
