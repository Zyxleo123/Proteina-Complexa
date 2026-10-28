"""One sequence-evaluation interface: sequence in, scored cyclic complex out.

    evaluate(original_sequence, candidate_sequence, target, prediction_seed, protocol)

Every arm of the search goes through this function and nothing else, so a difference
between arms cannot come from a difference in how they were predicted or scored. The
target representation (receptor sequences, cached MSA, conditioning pocket, target-only
template) is built ONCE per parent in `TargetContext` and reused byte-for-byte by every
candidate of that parent, in every arm.

What it does, in order:

1. **Checks the sequence before spending a prediction.** Length, alphabet support and the
   exact mutation count against the original. A rejected sequence returns a record with
   the reason and costs nothing.
2. **Predicts the complex with Boltz-2**, head-to-tail `cyclic: true`, the peptide given
   `msa: empty` and **no template of its own** -- the peptide structure is predicted
   afresh from its sequence every time. Arm C proposes sequences from generator
   backbones, but those coordinates never enter this validation step; if they did, the
   arm would be scoring its own proposal rather than an independent refold.
3. **Scores with the corrected, closure-constrained protocol** and returns the component
   metrics with an explicit status.
4. **Saves raw and refined endpoint structures** plus a compact record. No trajectories.

Caching, and why the two caches are separate
--------------------------------------------
`predictions/<parent>/<input_hash>/` is keyed by `boltz2.input_hash`, which already
covers the candidate sequence, the full target representation, the topology, the seed,
the predictor revision and the resolved inference config -- everything the plan asks for.

`scores/<scoring_version>/<input_hash>.json` is keyed by the prediction identity AND the
scoring protocol version. That split is the point: the closure-constraint fix invalidated
every score in this project while leaving the predictions perfectly good, and a single
combined key would have thrown away ~3.6 GPU-hours of structures to fix a CPU-side bug.
Bumping `SCORING_PROTOCOL_VERSION` invalidates scores and nothing else.

Energy units are Rosetta REU. They are not measured binding free energies, and no
function here converts them into one.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from proteinfoundation.cpseq import retention as retmod
from proteinfoundation.cpseq import tolerances as tolmod
from proteinfoundation.cpseq.adapters.boltz2 import (
    BOLTZ_REV,
    Boltz2Config,
    build_input_yaml,
    chemistry_supported,
    confidence_for_pdb,
    input_hash,
    interface_pae_for_pdb,
)
from proteinfoundation.cpseq.adapters.boltz2 import predict as boltz_predict
from proteinfoundation.cpseq.adapters.boltz2 import predict_many as boltz_predict_many
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure
from proteinfoundation.cpseq.geometry import validity_report
from proteinfoundation.cpseq.scorer import score_structure
from proteinfoundation.cpseq.search import (
    budget_for,
    default_alphabet,
    mutated_positions,
)

BINDER_CHAIN = "B"

# One Boltz-2 process per task, enforced here rather than by convention.
#
# Folding is GPU work in a subprocess; scoring is single-threaded PyRosetta in this
# process. That makes them overlappable -- a driver can prefetch the next seed's folds
# on a background thread while the main thread relaxes the current seed's -- and the
# measured prize is large: the live corpus run spends 51.5 % of its wall time in boltz
# and the rest with the GPU idle.
#
# What the overlap must not do is put two boltz processes on one GPU. The prefetch
# thread and `evaluate`'s own fallback prediction can otherwise collide: the fallback
# fires exactly when a prefetch skipped a candidate, which is the moment the prefetch
# thread has already moved on to the next seed. Both paths therefore take this lock, so
# serialisation is a property of the module rather than of the caller remembering.
#
# It is uncontended, and costs nothing, in the single-threaded driver.
GPU_LOCK = threading.Lock()

# The scoring protocol this module writes. v2 is the closure-constrained relax: the
# refinement carries a harmonic AtomPairConstraint on the closing bond, on BOTH the
# complex and the isolated peptide, so relax can no longer open the ring it was meant to
# preserve. v1 (unconstrained) rows live under `scores_unconstrained_relax/` and are
# retained for provenance only -- see the module docstring.
# v3 changes the VALIDITY criteria, not the relax: the closing amide is judged against
# the nearest arrangement its chemistry permits (cis as well as trans, within 30 deg)
# rather than against trans alone, and the closure windows are MuCO's published ones
# (amide 1.23-1.43 A, disulfide 1.95-2.15 A). Provenance for both is in `tolerances`.
#
# The version is bumped because every validity label in the project changes. Predictions
# are untouched and remain valid, which is the whole reason the prediction and score
# caches are keyed separately -- v2 rows stay on disk under their own directory and a
# v3 rescore costs CPU only.
#
# v4 changes WHICH STRUCTURE validity is judged on: the relaxed complex, the one the
# reported `refined_dG_separated` was computed on, rather than the raw prediction. v3
# paired a label from the predicted structure with an energy from the relaxed one. The
# pre-relax verdict survives on every row under `prerelax_*`, so nothing is lost and the
# difference between the two labels is measurable per arm.
#
# The version is bumped again because every validity label changes again. As with v3,
# predictions are untouched: a v4 rescore is CPU only, and v3 rows stay on disk under
# their own directory.
#
# NOT yet changed in v3/v4: the steric criterion is still the element-blind zero-tolerance
# 2.2 A heavy-atom count. It is the one criterion with no calibration provenance and it
# rejects 2 of 4 mainchain crystal controls (1SFI on a native internal disulfide, 7K2F
# on an O-O hydrogen bond). Replacing it with Probe/MolProbity contact analysis is a
# separate change, blocked on that toolchain; until then this version's structural
# validity is known to be conservative on sterics.
# v5 restrains the whole closure JUNCTION during relax -- both angles and the torsion,
# alongside the bond length that v2 introduced -- because an AtomPairConstraint alone
# holds a distance and lets the minimiser bend the ring open to pay for its repulsive
# term. Measured on 24 folds under v4: closure angles were over tolerance in 0 of 48
# measurements before relax and 37 of 48 after, while the restrained bond length stayed
# perfect. The refined coordinates and every refined energy change, so this is a new
# version rather than a new validity rule on old structures.
SCORING_PROTOCOL_VERSION = "v5-junction-restrained-relax"

# Rosetta energies are REU. Recorded on every row so no reader has to assume.
ENERGY_UNITS = "Rosetta ref2015 REU (NOT a measured binding free energy)"


@dataclass
class Protocol:
    """Everything that has to be identical across candidates and arms."""

    chemistry: str = "mainchain"
    # One structure per screening prediction, by design: the pilot buys breadth over
    # samples, and the confirmation stage is where repeats are spent.
    diffusion_samples: int = 1
    recycling_steps: int = 3
    sampling_steps: int = 200
    use_potentials: bool = False
    scoring_version: str = SCORING_PROTOCOL_VERSION
    boltz_python: str = "/data/user_data/yixiz/envs/boltz2/.venv/bin/python"
    boltz_cache: str = "/data/user_data/yixiz/boltz_cache"
    timeout_s: int = 7200
    allow_cys: bool = False

    def boltz_config(self, seed: int) -> Boltz2Config:
        return Boltz2Config(
            diffusion_samples=self.diffusion_samples,
            recycling_steps=self.recycling_steps,
            sampling_steps=self.sampling_steps,
            use_potentials=self.use_potentials,
            seed=seed,
        )

    def to_dict(self) -> dict:
        return {
            "chemistry": self.chemistry,
            "diffusion_samples": self.diffusion_samples,
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "use_potentials": self.use_potentials,
            "scoring_version": self.scoring_version,
            "predictor": "boltz2",
            "predictor_revision": BOLTZ_REV,
            "energy_units": ENERGY_UNITS,
            "allow_cys": self.allow_cys,
        }


@dataclass
class TargetContext:
    """One parent's frozen target representation. Built once, reused by every candidate."""

    parent_id: str
    pdb_id: str
    reference_sequence: str
    family_id: str
    split: str
    # Keyed by the chain ids the PREDICTOR sees (the template's own ids).
    receptor_sequences: dict[str, str]
    receptor_chains: list[str]
    receptor_msa_path: str | None
    pocket_contacts: list[tuple[str, int]]
    target_template_cif: str
    target_template_pdb: str
    chain_map: dict[str, str]

    @property
    def peptide_length(self) -> int:
        return len(self.reference_sequence)

    @property
    def budget(self) -> int:
        return budget_for(self.peptide_length)


def load_target_contexts(
    dataset_dir: Path, parent_ids: list[str] | None = None
) -> dict[str, TargetContext]:
    """Build every parent's target context from the tranche manifest.

    Receptor sequences come from the target-only TEMPLATE, not from a parent context --
    the same decision `cpseq_predict.run_boltz2` documents. The template is what the
    predictor is actually handed, so reading the sequences from it keeps the YAML and the
    template coordinates consistent by construction, and works for controls as well as
    LNR parents.
    """
    targets = pd.read_csv(dataset_dir / "manifest" / "targets.csv")
    out: dict[str, TargetContext] = {}
    tmpl_cache: dict[str, dict[str, str]] = {}

    for _, t in targets.iterrows():
        pid = str(t["parent_id"])
        if parent_ids is not None and pid not in parent_ids:
            continue
        tmpl = str(t["target_template_path"])
        if tmpl not in tmpl_cache:
            parsed = parse_structure(Path(tmpl))
            tmpl_cache[tmpl] = {
                ch: chain_sequence(parsed, ch) for ch in sorted(parsed["atoms"])
            }
        receptor_sequences = tmpl_cache[tmpl]

        msa_paths = json.loads(t["receptor_msa_paths"])
        chain_map = json.loads(t["chain_map"])
        first_msa = next(
            (msa_paths[c] for c in sorted(receptor_sequences) if msa_paths.get(c)), None
        )
        pocket = [
            (chain_map.get(ch, ch), int(idx))
            for ch, idx in json.loads(t["pocket_contacts"])
        ]
        out[pid] = TargetContext(
            parent_id=pid,
            pdb_id=str(t["pdb_id"]),
            reference_sequence=str(t["reference_sequence"]),
            family_id=str(t["family_id"]),
            split=str(t["split"]),
            receptor_sequences=receptor_sequences,
            # Same convention as cpseq_score.py: the scorer's receptor chains are the
            # mapped (predictor-side) ids.
            receptor_chains=sorted(set(chain_map.values())),
            receptor_msa_path=first_msa,
            pocket_contacts=pocket,
            target_template_cif=str(t["target_template_cif"]),
            target_template_pdb=tmpl,
            chain_map=chain_map,
        )
    return out


# --------------------------------------------------------------------------------------
# Sequence admissibility -- checked before any compute is spent
# --------------------------------------------------------------------------------------


def check_sequence(
    original: str, candidate: str, *, allow_cys: bool = False
) -> tuple[bool, list[str], dict]:
    """Length, alphabet and the exact mutation count against the original."""
    reasons: list[str] = []
    alpha = set(default_alphabet(allow_cys=allow_cys))

    if len(candidate) != len(original):
        # Length is fixed for this task, so this is a rejection and not a renumbering.
        return (
            False,
            [f"length {len(candidate)} != original {len(original)} (substitutions only)"],
            {"n_mutations": None, "mutation_fraction": None},
        )

    # A residue is admissible if the alphabet offers it OR it is what the original has at
    # that position. The two are not the same thing, and conflating them breaks a real
    # parent: `lnr_2v8x` is RIIYDRKFLMECRN, whose native CYS is outside the substitution
    # alphabet, so a flat membership test rejects the parent's OWN reference sequence and
    # every candidate derived from it -- the parent would report zero evaluations and look
    # like a search failure.
    #
    # Keeping it positional is what preserves the reason CYS is excluded in the first
    # place: a native cysteine may stay where evolution put it, and may be substituted
    # away, but no substitution may introduce one anywhere else, where it could pair into
    # a ring the `mainchain` label does not name.
    unsupported = sorted(
        {c for i, c in enumerate(candidate) if c not in alpha and c != original[i]}
    )
    if unsupported:
        reasons.append(
            f"unsupported residue(s) {unsupported} introduced at a position where the "
            f"original does not have them; alphabet is "
            f"canonical{'' if allow_cys else ' minus CYS'}"
        )

    positions = mutated_positions(original, candidate)
    n_mut = len(positions)
    budget = budget_for(len(original))
    if n_mut > budget:
        reasons.append(f"{n_mut} substitutions exceeds the ceiling floor(L/2)={budget}")

    info = {
        "n_mutations": n_mut,
        "mutation_fraction": n_mut / len(original),
        "mutated_positions": positions,
        "budget": budget,
        "identity_to_original": 1.0 - n_mut / len(original),
    }
    return (not reasons), reasons, info


# --------------------------------------------------------------------------------------
# Continuous geometry violation -- for exploration only, never for promotion
# --------------------------------------------------------------------------------------


def violation_magnitude(validity: dict, tol) -> float | None:
    """How far a structure is from passing, as one continuous number.

    Used ONLY to let the archive move toward feasibility while nothing is valid yet
    (plan 4.4). It never promotes a structure: an open or clashing pose with a good
    energy stays unaccepted no matter how small this number is. Each term is scaled by
    its own tolerance so an angle in degrees and a bond in angstroms are comparable.

    `None` when the ring could not be measured at all -- an unmeasurable closure is not
    a small violation, and returning 0 would make it look like the best candidate.
    """
    closure = validity.get("closure", {})
    if closure.get("status") != "ok":
        return None

    terms: list[float] = []
    d = closure.get("bond_length_A")
    if d is None:
        return None
    if d < tol.bond_length_min_A:
        terms.append((tol.bond_length_min_A - d) / max(tol.bond_length_min_A, 1e-6))
    elif d > tol.bond_length_max_A:
        terms.append((d - tol.bond_length_max_A) / max(tol.bond_length_max_A, 1e-6))
    else:
        terms.append(0.0)

    # `dihedral_dev_deg` is geometry.py's own name for the closure torsion deviation;
    # `tolerances.verdict` reports the same check as `torsion_in_tol`. Using the verdict's
    # name here would read None on every row and make every structure look unmeasurable.
    for key, tolerance in (
        ("angle_i_dev_deg", tol.angle_tol_deg),
        ("angle_j_dev_deg", tol.angle_tol_deg),
        ("dihedral_dev_deg", tol.torsion_tol_deg),
    ):
        dev = closure.get(key)
        if dev is None:
            return None
        terms.append(max(0.0, (dev - tolerance) / max(tolerance, 1e-6)))

    cl = validity.get("clashes", {})
    n_intra = cl.get("n_intra_clashes", 0) or 0
    n_inter = cl.get("n_inter_clashes", 0) or 0
    terms.append(float(max(0, n_intra - tol.max_intra_clashes)))
    terms.append(float(max(0, n_inter - tol.max_inter_clashes)))

    return round(float(sum(terms)), 6)


# --------------------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------------------


@dataclass
class Store:
    """Where predictions, scores and structures live. One per search run."""

    root: Path
    tolerances_path: Path
    # What to keep from each prediction once it is scored. `full` retains Boltz-2's whole
    # working directory, which is the behaviour every row in this project was produced
    # under, so it stays the default and a tier is only ever entered deliberately. See
    # `retention` for the measured breakdown and what each tier gives up.
    retention: str = "full"
    # Write the canonical-frame design records. Off by default so that enabling it is a
    # deliberate act and a Slurm array cannot start producing them mid-run; any non-full
    # retention tier implies it, since the records must exist before the structures they
    # derive from are removed.
    write_designs: bool = False
    _tol: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "predictions").mkdir(parents=True, exist_ok=True)
        (self.root / "scores").mkdir(parents=True, exist_ok=True)
        if not self._tol:
            self._tol = tolmod.load(Path(self.tolerances_path))

    def tolerance(self, chemistry: str):
        return self._tol.get(chemistry)

    def prediction_dir(self, parent_id: str, ihash: str) -> Path:
        return self.root / "predictions" / parent_id / ihash

    def score_path(self, scoring_version: str, ihash: str) -> Path:
        d = self.root / "scores" / scoring_version
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{ihash}.json"


def prefetch_predictions(
    *,
    candidate_sequences: list[str],
    target: TargetContext,
    prediction_seed: int,
    protocol: Protocol,
    store: Store,
) -> dict:
    """Predict several candidates of one target, at one seed, in a single boltz run.

    This is a pure speed optimisation and deliberately changes nothing else. It fills the
    SAME prediction cache that `evaluate` reads, writing each candidate's structures and
    `record.json` exactly where a one-at-a-time run would have put them -- so `evaluate`
    is untouched, every candidate it is later asked for is a cache hit, and if anything
    here fails or skips a candidate, `evaluate` simply predicts that one itself. There is
    no path where batching changes a result; it only changes who paid for it.

    Why it is worth the machinery: measured on an A6000 with this project's settings,
    boltz costs 44 s per INVOCATION plus 26 s per structure. One structure per invocation
    spends 63 % of its time on process spawn, the torch import, the 2.2 GB checkpoint
    read and the model build. Nine per invocation spends 16 %.

    Candidates already cached are skipped, so a resumed task batches only what it still
    owes. Returns a record of what was attempted, for the caller's log.
    """
    cfg = protocol.boltz_config(prediction_seed)
    wanted: dict[str, tuple[str, str]] = {}   # ihash -> (sequence, yaml_text)
    for seq in dict.fromkeys(candidate_sequences):      # de-duplicate, keep order
        yaml_text = build_input_yaml(
            peptide_sequence=seq,
            receptor_sequences=target.receptor_sequences,
            chemistry=protocol.chemistry,
            pocket_contacts=target.pocket_contacts,
            receptor_msa_path=target.receptor_msa_path,
            target_template_path=target.target_template_cif,
        )
        ihash = input_hash(yaml_text, cfg)
        pred_dir = store.prediction_dir(target.parent_id, ihash)
        rec_path = pred_dir / "record.json"
        if rec_path.exists():
            try:
                if json.loads(rec_path.read_text()).get("status") == "ok":
                    continue          # already paid for
            except Exception:  # noqa: BLE001
                pass
        wanted[ihash] = (seq, yaml_text)

    out = {"requested": len(wanted), "produced": 0, "status": "nothing_to_do"}
    if not wanted:
        return out

    stage = store.root / "predictions" / target.parent_id / f"_batch_seed{prediction_seed}"
    with GPU_LOCK:
        res = boltz_predict_many(
            yaml_texts={k: v[1] for k, v in wanted.items()},
            stage_dir=stage,
            config=cfg,
            python_exec=protocol.boltz_python,
            boltz_cache=protocol.boltz_cache,
            timeout_s=protocol.timeout_s,
        )
    per_key = res.get("per_key") or {}

    for ihash, (_seq, yaml_text) in wanted.items():
        src = per_key.get(ihash)
        if not src:
            # No structures for this candidate. Nothing is recorded: a missing record is
            # a cache MISS, so `evaluate` will predict it on its own and a batch-level
            # problem never becomes a per-candidate failure row.
            continue
        pred_dir = store.prediction_dir(target.parent_id, ihash)
        dest = pred_dir / "boltz" / "predictions" / ihash
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        (pred_dir / "input.yaml").write_text(yaml_text)
        samples = []
        for n, pdb in enumerate(sorted(dest.glob("*.pdb"))):
            samples.append({"sample_index": n, "pdb_path": str(pdb)})
        (pred_dir / "record.json").write_text(json.dumps({
            "input_hash": ihash,
            "predictor": "boltz2",
            "predictor_revision": BOLTZ_REV,
            "config": cfg.to_dict(),
            "samples": samples,
            "status": "ok" if samples else "no_structure",
            "returncode": res.get("returncode"),
            # The invocation's total, not this candidate's share. Recorded under its own
            # name so no reader mistakes it for a per-candidate cost: the per-fold
            # runtime of a batched prediction is not a measurable quantity.
            "batch_runtime_s": res.get("runtime_s"),
            "batch_size": len(wanted),
        }, indent=2))
        out["produced"] += 1

    # The staging directory holds the shared processed MSA and the batch wrapper, neither
    # of which is a result; the structures have been moved out of it.
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    out["status"] = res.get("status")
    out["batch_runtime_s"] = res.get("runtime_s")
    # On a batch that produced nothing, carry the predictor's own words out. Without
    # this the caller sees "0 of 9 structures" and no reason, which is how a misplaced
    # output directory cost two verification rounds.
    if out["produced"] < out["requested"]:
        out["stderr_tail"] = (res.get("stderr_tail") or "")[-600:]
        out["returncode"] = res.get("returncode")
    return out


def evaluate(
    *,
    original_sequence: str,
    candidate_sequence: str,
    target: TargetContext,
    prediction_seed: int,
    protocol: Protocol,
    store: Store,
    force_predict: bool = False,
    force_score: bool = False,
) -> dict:
    """Predict and score one candidate sequence. Never raises on a model failure.

    The returned record always carries `status`, and a failure says which KIND it was:

    - `sequence_rejected`   the sequence is not admissible; nothing was run
    - `chemistry_unsupported` the predictor cannot express this topology
    - `prediction_failed`   the predictor errored or timed out  -- TECHNICAL
    - `no_structure`        the predictor returned nothing       -- TECHNICAL
    - `scored`              a structure exists and was scored

    A technical failure is not a nonbinding label, and `valid: False` on a scored row is
    a structural verdict, not a crash. The driver keeps the two in separate counters.
    """
    t_start = time.time()
    rec: dict = {
        "parent_id": target.parent_id,
        "pdb_id": target.pdb_id,
        "family_id": target.family_id,
        "split": target.split,
        "chemistry": protocol.chemistry,
        "original_sequence": original_sequence,
        "sequence": candidate_sequence,
        "peptide_length": len(candidate_sequence),
        "prediction_seed": prediction_seed,
        "protocol": protocol.to_dict(),
        "scoring_version": protocol.scoring_version,
        "status": "unevaluated",
        "cache": {"prediction": "miss", "score": "miss"},
    }

    # --- 1. admissibility ------------------------------------------------------------
    ok, reasons, info = check_sequence(
        original_sequence, candidate_sequence, allow_cys=protocol.allow_cys
    )
    rec |= {
        "n_mutations": info["n_mutations"],
        "mutation_fraction": info["mutation_fraction"],
        "mutated_positions": ",".join(str(p) for p in info.get("mutated_positions", ())),
        "substitution_budget": info.get("budget"),
        "identity_to_original": info.get("identity_to_original"),
    }
    if not ok:
        rec |= {"status": "sequence_rejected", "reasons": reasons, "valid": False}
        rec["wall_time_s"] = round(time.time() - t_start, 2)
        return rec

    if not chemistry_supported(protocol.chemistry):
        rec |= {
            "status": "chemistry_unsupported",
            "reasons": [f"boltz2 cannot express {protocol.chemistry}"],
            "valid": False,
        }
        rec["wall_time_s"] = round(time.time() - t_start, 2)
        return rec

    # --- 2. prediction (cached) ------------------------------------------------------
    cfg = protocol.boltz_config(prediction_seed)
    yaml_text = build_input_yaml(
        peptide_sequence=candidate_sequence,
        receptor_sequences=target.receptor_sequences,
        chemistry=protocol.chemistry,
        pocket_contacts=target.pocket_contacts,
        receptor_msa_path=target.receptor_msa_path,
        # Target-only. The peptide has no template, so its structure is predicted
        # afresh from sequence -- see the module docstring.
        target_template_path=target.target_template_cif,
    )
    ihash = input_hash(yaml_text, cfg)
    rec["input_hash"] = ihash
    pred_dir = store.prediction_dir(target.parent_id, ihash)
    pred_record_path = pred_dir / "record.json"

    # A cached prediction counts as a hit only if it actually produced a structure.
    #
    # The record is written whatever the predictor did, failures included, so a crashed
    # prediction used to be cached AS a crash: every later attempt found `record.json`,
    # called it a hit, read `status != "ok"` and re-recorded the same technical failure
    # in 0.03 s. The failure was therefore permanent and invisible to any retry.
    # Measured 2026-10-23: one task on `pepbench_C_D_pdb4wjq` lost its GPU and wrote 72
    # failed records; a requeue "re-ran" all 72 in 12 s, reporting 72 cache hits and
    # recovering nothing. On a preemptible queue, where a prediction can be killed
    # mid-flight at any moment, caching a crash poisons that fold forever.
    #
    # Only `status == "ok"` is cacheable. A sequence the predictor legitimately cannot
    # express is rejected earlier, by `check_sequence` and the chemistry gate, so what
    # reaches here as a failure is infrastructure -- a crash, a timeout, a revoked GPU --
    # which is exactly what should be retried rather than remembered.
    cached: dict | None = None
    if pred_record_path.exists() and not force_predict:
        cached = json.loads(pred_record_path.read_text())
        if cached.get("status") != "ok":
            rec["cache"]["prediction_retry_over_failed_record"] = cached.get("status")
            cached = None
    if cached is not None:
        pred = cached
        rec["cache"]["prediction"] = "hit"
    else:
        pred_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = pred_dir / "input.yaml"
        yaml_path.write_text(yaml_text)
        with GPU_LOCK:
            result = boltz_predict(
                yaml_path=yaml_path,
                out_dir=pred_dir / "boltz",
                config=cfg,
                python_exec=protocol.boltz_python,
                boltz_cache=protocol.boltz_cache,
                timeout_s=protocol.timeout_s,
            )
        samples = []
        for n, pdb in enumerate(sorted((pred_dir / "boltz").rglob("*.pdb"))):
            # Skip anything this module itself wrote next to the prediction.
            if pdb.name.endswith(("_refined.pdb", "_isolated_relaxed.pdb")):
                continue
            samples.append({"sample_index": n, "pdb_path": str(pdb)})
        pred = {
            "input_hash": ihash,
            "predictor": "boltz2",
            "predictor_revision": BOLTZ_REV,
            "config": cfg.to_dict(),
            "samples": samples,
            **{k: v for k, v in result.items() if k not in ("stdout_tail",)},
        }
        pred_record_path.write_text(json.dumps(pred, indent=2))

    rec["prediction_status"] = pred.get("status")
    rec["prediction_runtime_s"] = pred.get("runtime_s")
    samples = pred.get("samples", [])

    # A `rescorable`-tier prediction keeps its raw complex gzipped, so a rescore under a
    # new protocol version decompresses rather than re-predicting. This is the line that
    # makes that tier's claim true; without it a cache hit would hand the scorer a path
    # that no longer exists.
    if samples and any(not Path(s["pdb_path"]).exists() for s in samples):
        retmod.restore_raw_structures(pred_dir)

    if pred.get("status") != "ok":
        rec |= {
            "status": "prediction_failed",
            "technical_failure": True,
            "reasons": [f"boltz2 status {pred.get('status')}: {str(pred.get('stderr_tail'))[-400:]}"],
            "valid": False,
        }
        rec["wall_time_s"] = round(time.time() - t_start, 2)
        return rec
    if not samples:
        rec |= {
            "status": "no_structure",
            "technical_failure": True,
            "reasons": ["predictor returned ok but wrote no pdb"],
            "valid": False,
        }
        rec["wall_time_s"] = round(time.time() - t_start, 2)
        return rec

    # --- 3. scoring (cached separately, by protocol version) -------------------------
    score_path = store.score_path(protocol.scoring_version, ihash)
    if score_path.exists() and not force_score:
        scored = json.loads(score_path.read_text())
        rec["cache"]["score"] = "hit"
    else:
        tol = store.tolerance(protocol.chemistry)
        per_sample = []
        for s in samples:
            pdb = Path(s["pdb_path"])
            stem = pdb.with_suffix("")
            row: dict = {"sample_index": s["sample_index"], "raw_pdb_path": str(pdb)}
            try:
                val = validity_report(
                    pdb,
                    binder_chain=BINDER_CHAIN,
                    receptor_chains=target.receptor_chains,
                    chemistry=protocol.chemistry,
                    requested_sequence=candidate_sequence,
                )
                verd = (
                    tolmod.verdict(val, tol)
                    if tol
                    else {"valid": False, "reasons": [f"no tolerance for {protocol.chemistry}"], "checks": {}}
                )
                sc = score_structure(
                    pdb,
                    binder_chain=BINDER_CHAIN,
                    receptor_chains=target.receptor_chains,
                    chemistry=protocol.chemistry,
                    requested_sequence=candidate_sequence,
                    refined_out_path=f"{stem}_refined.pdb",
                    isolated_out_path=f"{stem}_isolated_relaxed.pdb",
                )
                # Validity is judged on the RELAXED complex, because that is the
                # structure `refined_dG_separated` was computed on. Judging the raw
                # prediction and then reporting the relaxed energy -- what v3 did --
                # pairs a label from one structure with a number from another: relax
                # moves atoms, so it can close a ring the prediction left open and it
                # can open one the prediction had closed. The energy only means
                # something for a complex that is still the molecule we asked for.
                #
                # The pre-relax verdict is kept under `prerelax_*` rather than dropped:
                # it is the only record of what the predictor itself produced, and the
                # gap between the two labels is how much of each arm's validity the
                # relax is responsible for.
                refined_pdb = Path(f"{stem}_refined.pdb")
                rval = rverd = None
                if refined_pdb.exists():
                    rval = validity_report(
                        refined_pdb,
                        binder_chain=BINDER_CHAIN,
                        receptor_chains=target.receptor_chains,
                        chemistry=protocol.chemistry,
                        requested_sequence=candidate_sequence,
                    )
                    rverd = (
                        tolmod.verdict(rval, tol)
                        if tol
                        else {"valid": False, "reasons": [f"no tolerance for {protocol.chemistry}"], "checks": {}}
                    )
                geom = rval if rval is not None else val
                row |= {
                    "scoring_status": sc.get("status"),
                    # No relaxed structure means no post-relax verdict and no refined
                    # energy either, so there is nothing to call valid. Reported as its
                    # own reason rather than silently inheriting the pre-relax label.
                    "valid": bool(rverd["valid"]) if rverd else False,
                    "validity_reasons": (
                        rverd["reasons"] if rverd
                        else [f"no relaxed structure to judge (scorer: {sc.get('status')})"]
                    ),
                    "validity_checks": rverd["checks"] if rverd else {},
                    "validity_stage": "postrelax" if rverd else "unavailable",
                    "violation": (
                        violation_magnitude(rval, tol) if (rval is not None and tol) else None
                    ),
                    "sequence_matches": val.get("sequence_matches"),
                    "sequence_observed": val.get("sequence_observed"),
                    "closure": geom.get("closure"),
                    "clashes": {
                        k: v for k, v in (geom.get("clashes") or {}).items()
                        if not isinstance(v, (list, dict))
                    },
                    "prerelax_valid": bool(verd["valid"]),
                    "prerelax_validity_reasons": verd["reasons"],
                    "prerelax_validity_checks": verd["checks"],
                    "prerelax_violation": violation_magnitude(val, tol) if tol else None,
                    "prerelax_closure": val.get("closure"),
                    "prerelax_clashes": {
                        k: v for k, v in (val.get("clashes") or {}).items()
                        if not isinstance(v, (list, dict))
                    },
                    "refined_pdb_path": f"{stem}_refined.pdb",
                    # Recorded for every fold, never part of the objective: these are
                    # the predictor's own confidence in the pose, and optimising them
                    # would be optimising Boltz-2's self-assessment rather than the
                    # interface. Harvested here because retention deletes the matrices
                    # they come from.
                    "confidence": confidence_for_pdb(pdb)
                    | interface_pae_for_pdb(
                        pdb,
                        binder_chain=BINDER_CHAIN,
                        receptor_chains=target.receptor_chains,
                    ),
                }
                # Interface energetics and quality, raw and refined, under their own
                # names so a refined number can never be read as a raw one.
                for k in (
                    "dG_separated", "dSASA_int", "dG_separated_per_1000A2",
                    "delta_unsatHbonds", "hbonds_int", "sc_value", "packstat",
                    "nres_int", "total_score",
                ):
                    row[f"raw_{k}"] = (sc.get("raw") or {}).get(k)
                    row[f"refined_{k}"] = (sc.get("refined") or {}).get(k)
                disp = sc.get("refinement_displacement") or {}
                row["refine_binder_ca_rmsd_A"] = disp.get("binder_ca_rmsd_A")
                row["refine_receptor_ca_rmsd_A"] = disp.get("receptor_ca_rmsd_A")
                row["refine_total_score_delta"] = disp.get("total_score_delta")
                row["ring_declared_complex"] = (
                    sc.get("ring_declared_complex") or {}
                ).get("declared")
                # Strain: reported when the proxy computed, explicitly unavailable
                # otherwise. Never defaulted to a number that would read as "passed".
                defo = sc.get("deformation") or {}
                row["deformation_status"] = defo.get("status", "unavailable")
                row["local_deformation_energy_proxy"] = defo.get("deformation_energy")
                row["strain_available"] = defo.get("status") == "ok"
                row["strain_is_thermodynamic"] = defo.get(
                    "is_thermodynamic_free_energy", False
                )
                row["scorer_protocol"] = sc.get("protocol")
            except Exception as exc:  # noqa: BLE001
                row |= {
                    "scoring_status": "scorer_error",
                    "technical_failure": True,
                    "valid": False,
                    "validity_reasons": [f"{type(exc).__name__}: {exc}"],
                }
            per_sample.append(row)

        scored = {
            "input_hash": ihash,
            "scoring_version": protocol.scoring_version,
            "energy_units": ENERGY_UNITS,
            "samples": per_sample,
        }
        score_path.write_text(json.dumps(scored, indent=2))

    rec["samples"] = scored.get("samples", [])
    rec["status"] = "scored"
    rec["n_samples"] = len(rec["samples"])
    rec["n_valid"] = sum(1 for s in rec["samples"] if s.get("valid"))
    rec["valid"] = rec["n_valid"] > 0
    # The same count under the pre-relax label, so a run can report how many structures
    # the predictor itself closed versus how many the relax closed or opened.
    rec["n_valid_prerelax"] = sum(1 for s in rec["samples"] if s.get("prerelax_valid"))
    rec["valid_prerelax"] = rec["n_valid_prerelax"] > 0

    # The screening objective: the best refined dG among VALID samples only. An invalid
    # structure contributes no energy, however attractive -- plan 4.4.
    valid_energies = [
        s.get("refined_dG_separated")
        for s in rec["samples"]
        if s.get("valid") and s.get("refined_dG_separated") is not None
    ]
    rec["best_valid_dG_separated"] = min(valid_energies) if valid_energies else None
    violations = [
        s.get("violation") for s in rec["samples"] if s.get("violation") is not None
    ]
    rec["min_violation"] = min(violations) if violations else None

    # --- 4. the deliverable, then compaction -----------------------------------------
    # The design record is the one artefact the retained tree never held in a usable
    # form: this target's peptides all expressed in ONE receptor frame. The retained
    # complexes each sit in their own arbitrary global frame (measured 12.8 A median
    # pairwise receptor RMSD as written, 0.90 A after superposition), so comparing
    # designs from them means re-superimposing every complex.
    #
    # It is nonetheless OFF by default, and a non-`full` retention tier turns it on
    # because it must be written before the structures it derives from are deleted.
    # The reason for the default is operational rather than scientific: a Slurm array
    # with pending tasks picks up an edit that its running tasks never see, so anything
    # unconditional here would make the corpus half-and-half mid-run. This project has
    # already paid for that lesson once (README sec. 5, the budget cap written at
    # 22:53:38 against a process that loaded the module at 22:44:03).
    if store.write_designs or store.retention != "full":
        _write_design_records(rec, store, target, ihash, candidate_sequence,
                              original_sequence)

    if store.retention != "full":
        try:
            rec["retention"] = retmod.compact_prediction_dir(pred_dir, store.retention)
        except Exception as exc:  # noqa: BLE001
            rec["retention"] = {"tier": store.retention,
                                "error": f"{type(exc).__name__}: {exc}"}

    rec["wall_time_s"] = round(time.time() - t_start, 2)
    return rec


def _write_design_records(
    rec: dict,
    store: "Store",
    target: TargetContext,
    ihash: str,
    candidate_sequence: str,
    original_sequence: str,
) -> None:
    """Deposit each scored sample's peptide in the target's canonical receptor frame."""
    for s in rec["samples"]:
        raw = Path(s.get("raw_pdb_path", ""))
        if not raw.exists():
            continue
        try:
            dr = retmod.write_design_record(
                out_root=store.root,
                parent_id=target.parent_id,
                input_hash=ihash,
                sequence=candidate_sequence,
                original_sequence=original_sequence,
                binder_chain=BINDER_CHAIN,
                receptor_chains=set(target.receptor_chains),
                raw_pdb=raw,
                refined_pdb=Path(s["refined_pdb_path"])
                if s.get("refined_pdb_path") else None,
            )
            s |= {k: v for k, v in dr.items() if v is not None}
        except Exception as exc:  # noqa: BLE001
            # A deliverable that cannot be written must not fail an evaluation that
            # already succeeded -- the score row is the result and it is already on disk.
            s["design_record_error"] = f"{type(exc).__name__}: {exc}"
