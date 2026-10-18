"""Hotspot-retention metrics for the LP arm.

WHY THIS EXISTS RATHER THAN A CONFIG ENTRY FOR ``rfd3.metrics.design_metrics.PPIMetrics``.
Upstream's `PPIMetrics` returns a hard ``0.0`` on every spec of a sequence-given panel, and
it does so silently. The arithmetic, read from `get_hotspot_contacts`
(rfd3/metrics/metrics_utils.py:94-118):

    diffused_interchain_contacts_mask = interchain_contacts_mask[
        :, ~get_motif_features(atom_array)["is_motif_atom"]
    ]
    contacted_hotspots_mask = np.any(diffused_interchain_contacts_mask, axis=1)
    return float(contacted_hotspots_mask.mean())

and `get_motif_features` (transforms/conditioning_base.py:55-66) defines

    is_motif_atom = is_fixed | is_sequence_fixed | is_unindexed

On the LP validation panel the peptide's SEQUENCE is given (it sits in the contig) while its
COORDINATES are released, so peptide atoms carry ``is_motif_atom_with_fixed_seq = True`` and
``is_motif_atom_with_fixed_coord = False``. Both receptor and peptide are therefore
``is_motif_atom = True``, ``~is_motif_atom`` selects NO atoms, the mask has shape
``(n_hotspots, 0)``, and ``np.any(..., axis=1)`` over an empty axis is all-False -- so the
"fraction of hotspots contacted" is 0.0 for every example, every epoch. Verified:
``np.any(np.zeros((7, 0), bool), axis=1).mean() == 0.0``.

That number is worse than a missing one. A flat 0.0 curve is indistinguishable from a model
that never obeys the hotspot channel, which is exactly the conclusion this arm exists to
test -- so the arm would have been judged a failure by its own broken instrument.
(`redesign_motif_sidechains` does not rescue it: it defaults to False, input_parsing.py:186,
and the panel does not set it.)

THE FIX IS ONE WORD OF SEMANTICS. "Did the generated thing reach the hotspot" is a question
about atoms whose COORDINATES were generated, so the diffused set is
``~is_motif_atom_with_fixed_coord`` alone. That is independent of whether the sequence was
supplied, which is what makes it correct for every arm of the panel and for the base
no-hotspot arm too.

WHAT ELSE IS REPORTED, AND WHY IT IS NOT OPTIONAL. A binary contact fraction is useless on
the decoy arm: if the peptide ignores a decoy request entirely, the fraction is 0.0 and so is
the fraction for a peptide that moved 15 A toward the decoy and stopped short. Both read as
"did not obey", and the difference between them is the whole signal. So the continuous
distances are reported alongside -- ``com_to_hotspot_com`` and ``min_dist_to_hotspot`` give
the decoy arm a gradient instead of a floor.
"""

from __future__ import annotations

import logging

import numpy as np

# stdlib logging, NOT loguru: this module is imported inside the foundry environment, which
# does not ship loguru.
logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only inside the foundry environment
    from foundry.metrics.metric import Metric
except ImportError:  # pragma: no cover
    Metric = object


def hotspot_retention(
    hotspot_coord: np.ndarray,
    generated_coord: np.ndarray,
    distance_cutoff: float = 4.5,
) -> dict:
    """Retention of a requested hotspot set by a set of generated atoms.

    Pure geometry, so it is unit-testable without the arm's environment.

    Args:
        hotspot_coord: ``(n_hotspots, 3)`` coordinates of the REQUESTED hotspot atoms.
        generated_coord: ``(n_generated, 3)`` coordinates of atoms whose positions were
            generated -- NOT merely "non-motif". See the module docstring.
        distance_cutoff: a hotspot counts as contacted when some generated atom is within
            this distance. Must match the conditioning cutoff, or an obedient model is
            graded short.

    Returns a dict with:
        fraction_hotspots_contacted  the headline retention score, in [0, 1]
        n_hotspots_requested         so an empty request is visible rather than inferred
        n_generated_atoms            a spec that released nothing would score 0.0 otherwise
        com_to_hotspot_com           continuous, and the only thing that separates "ignored
                                     the request" from "moved toward it and fell short"
        min_dist_to_hotspot          closest approach of any generated atom to any hotspot
    """
    hs = np.asarray(hotspot_coord, dtype=float).reshape(-1, 3)
    gen = np.asarray(generated_coord, dtype=float).reshape(-1, 3)
    hs = hs[np.isfinite(hs).all(axis=1)]
    gen = gen[np.isfinite(gen).all(axis=1)]

    out = {
        "n_hotspots_requested": float(hs.shape[0]),
        "n_generated_atoms": float(gen.shape[0]),
    }
    if hs.shape[0] == 0 or gen.shape[0] == 0:
        # No request, or nothing was generated. Deliberately NOT reported as 0.0 retention:
        # that would be a measurement of a model that was never asked anything.
        return out

    d = np.linalg.norm(hs[:, None, :] - gen[None, :, :], axis=-1)  # (n_hs, n_gen)
    out["fraction_hotspots_contacted"] = float((d.min(axis=1) <= distance_cutoff).mean())
    out["min_dist_to_hotspot"] = float(d.min())
    out["com_to_hotspot_com"] = float(np.linalg.norm(gen.mean(0) - hs.mean(0)))
    return out


class LPHotspotMetrics(Metric):
    """`PPIMetrics` with the diffused set defined by generated COORDINATES.

    Registered by configs/rfd3_lp/trainer/metrics/lp_design_metrics.yaml. Reports nothing at
    all for a spec that carries no hotspot request, so the base no-hotspot panel is unaffected
    and the metric cannot be diluted by arms that were never asked anything.
    """

    def __init__(self, distance_cutoff: float = 4.5, **kwargs):
        super().__init__(**kwargs)
        self.distance_cutoff = distance_cutoff

    @property
    def kwargs_to_compute_args(self) -> dict:
        # The bare-string form is what upstream's own registered `AtomArrayMetrics` uses
        # (design_metrics.py:190), and `nested_dict.getitem` resolves it as a single key.
        return {"atom_array_stack": "predicted_atom_array_stack"}

    def compute(self, atom_array_stack) -> dict:
        per_example: dict = {}
        for atom_array in atom_array_stack:
            cats = atom_array.get_annotation_categories()
            if "is_atom_level_hotspot" not in cats:
                continue
            hotspot = np.asarray(
                atom_array.get_annotation("is_atom_level_hotspot")
            ).astype(bool)
            if hotspot.sum() == 0:
                continue

            # THE CORRECTION: generated == coordinates not fixed. Not `is_motif_atom`,
            # which is also true for every atom whose SEQUENCE we supplied -- i.e. the whole
            # peptide on this panel.
            if "is_motif_atom_with_fixed_coord" not in cats:
                logger.warning(
                    "is_motif_atom_with_fixed_coord absent; cannot identify generated atoms"
                )
                continue
            generated = ~np.asarray(
                atom_array.get_annotation("is_motif_atom_with_fixed_coord")
            ).astype(bool)

            # Interchain only, matching upstream's intent: a hotspot must be reached by the
            # OTHER chain, not by a generated atom of its own.
            if "pn_unit_iid" in cats:
                pn = np.asarray(atom_array.pn_unit_iid, dtype=str)
                hs_units = set(pn[hotspot].tolist())
                generated = generated & ~np.isin(pn, list(hs_units))

            res = hotspot_retention(
                atom_array.coord[hotspot],
                atom_array.coord[generated],
                distance_cutoff=self.distance_cutoff,
            )
            for k, v in res.items():
                per_example.setdefault(k, []).append(v)

        if not per_example:
            # No spec in this batch carried a request. An ABSENT column means "nothing was
            # asked"; it must never be reported as zero retention.
            return {}
        return {k: float(np.mean(v)) for k, v in per_example.items()}
