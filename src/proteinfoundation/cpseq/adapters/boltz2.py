"""Boltz-2 adapter: fixed-sequence cyclic-peptide/target complex prediction.

Runs in its own interpreter (`scripts/cpseq_env_boltz2.sbatch` builds it), because
Boltz-2 pins numpy<2 against the complexa venv's numpy>=2. This module therefore only
*writes inputs and shells out*; it imports nothing from boltz.

Schema source: `upstream/boltz/docs/prediction.md` at the pinned revision. The fields
used here, and why each one:

  `cyclic: true`         head-to-tail polymer flag on the PEPTIDE chain only. This is a
                         backbone ring. It is NOT a disulfide and is never set for one.
  `bond` constraint      the disulfide, as an explicit atom pair
                         `[chain, res_idx, SG] <-> [chain, res_idx, SG]`. The doc states
                         bonds are supported for canonical residues, which CYS is.
  `pocket` constraint    binder = peptide chain, contacts = the parent's receptor pocket
                         residues, `force: false`. Localisation only: it says where on
                         the receptor to look, not where any individual peptide residue
                         must sit. `force: true` would add a potential pulling the
                         peptide in, which would be steering a score rather than
                         conditioning a prediction.
  `templates`            the TARGET-ONLY structure. The peptide is never templated and
                         its known bound coordinates are never supplied, or the task
                         would be solved by the input.
  `msa`                  cached `.a3m` per parent for the receptor; `empty`
                         (single-sequence) for the peptide. A candidate peptide is a
                         synthetic sequence with no homologs, and routing it to an MSA
                         server would spend a search per candidate for nothing -- which
                         is exactly the repeated expensive alignment the task forbids.

Sequence fidelity is verified after the fact, from the output structure, by
`verify_output_sequence`: the requested sequence is a parameter of the input, but
checking it on the way out is what catches a chain-order or index mistake.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from proteinfoundation.cpseq.chemistry import (
    CHEMISTRIES,
    DISULFIDE,
    MAINCHAIN,
    SUPPORTED,
    capability,
)

TOOL = "boltz2"

# Pinned upstream revision; recorded on every row so a prediction is traceable.
BOLTZ_REV = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"

# Chain ids in the generated YAML. The receptor keeps chain A and the peptide chain B,
# matching this repo's convention everywhere else (CPSea: A = receptor, B = binder), so
# the evaluator does not need a per-predictor chain map.
RECEPTOR_CHAIN = "A"
PEPTIDE_CHAIN = "B"

# Pocket localisation distance. 6 A is boltz's own default and the middle of its
# supported 4-20 A range; frozen for the tranche and recorded in the input hash.
POCKET_MAX_DISTANCE_A = 6.0


@dataclass
class Boltz2Config:
    """Inference settings. Frozen per tranche and hashed into every task id."""

    diffusion_samples: int = 4
    recycling_steps: int = 3
    sampling_steps: int = 200
    # Inference-time potentials improve physical plausibility. Left OFF by default so
    # the raw prediction is what the shared evaluator sees; turning them on is a
    # predictor-specific steering choice and is recorded as such.
    use_potentials: bool = False
    seed: int = 42
    # The affinity module is deliberately NOT requested. It is designed for small-molecule
    # ligands (boltz's own doc caps it at ~56-128 atoms and says protein/RNA/DNA targets
    # give unreliable output), so it is not a peptide-affinity teacher.
    affinity: bool = False

    def to_dict(self) -> dict:
        return {
            "diffusion_samples": self.diffusion_samples,
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "use_potentials": self.use_potentials,
            "seed": self.seed,
            "affinity": self.affinity,
        }


def chemistry_supported(chemistry: str) -> bool:
    return capability(TOOL, chemistry).status == SUPPORTED


def build_input_yaml(
    *,
    peptide_sequence: str,
    receptor_sequences: dict[str, str],
    chemistry: str,
    pocket_contacts: list[tuple[str, int]] | None,
    receptor_msa_path: str | Path | None,
    target_template_path: str | Path | None,
) -> str:
    """The YAML text for one candidate. Pure function of its arguments, so it hashes.

    `pocket_contacts` are (original receptor chain id, 1-based residue index **in the
    YAML's own receptor numbering**). Receptor chains are concatenated into one `A` entry
    only when there is a single chain; multi-chain receptors keep separate ids, because
    merging them would invent a covalent link between two chains that are not bonded.
    """
    if not chemistry_supported(chemistry):
        raise ValueError(
            f"{TOOL} does not support {chemistry!r} "
            f"(status: {capability(TOOL, chemistry).status})"
        )
    spec = CHEMISTRIES[chemistry]
    L = len(peptide_sequence)

    lines = ["version: 1", "sequences:"]

    # Receptor chains, in sorted order so the YAML is deterministic.
    chain_ids = sorted(receptor_sequences)
    yaml_chain_of: dict[str, str] = {}
    for n, orig in enumerate(chain_ids):
        # A, C, D, ... keeping B for the peptide.
        letters = [c for c in "ACDEFGHIJKLMNOPQRSTUVWXYZ"]
        yaml_id = letters[n]
        yaml_chain_of[orig] = yaml_id
        lines += [
            "  - protein:",
            f"      id: {yaml_id}",
            f"      sequence: {receptor_sequences[orig]}",
        ]
        if receptor_msa_path is not None:
            lines.append(f"      msa: {receptor_msa_path}")
        else:
            # Explicit, not defaulted: without --use_msa_server boltz REQUIRES an msa
            # field, and letting it fall through would fail at parse time rather than
            # here where the reason is visible.
            lines.append("      msa: empty")

    # The peptide.
    lines += [
        "  - protein:",
        f"      id: {PEPTIDE_CHAIN}",
        f"      sequence: {peptide_sequence}",
        # Single-sequence on purpose: a candidate peptide has no homologs, and an MSA
        # search per candidate is the repeated expensive alignment we are avoiding.
        "      msa: empty",
    ]
    if chemistry == MAINCHAIN:
        lines.append("      cyclic: true")

    # Constraints.
    constraints: list[str] = []
    if chemistry == DISULFIDE:
        # Side-chain ring: the backbone is NOT cyclic, so `cyclic` stays unset and the
        # bond is named atom-for-atom instead.
        i1, j1 = spec.as_one_based(0, L - 1)
        constraints += [
            "  - bond:",
            f"      atom1: [{PEPTIDE_CHAIN}, {i1}, {spec.bond.atom_i}]",
            f"      atom2: [{PEPTIDE_CHAIN}, {j1}, {spec.bond.atom_j}]",
        ]
    if pocket_contacts:
        contacts = ", ".join(
            f"[{yaml_chain_of[ch]}, {idx}]" for ch, idx in pocket_contacts
        )
        constraints += [
            "  - pocket:",
            f"      binder: {PEPTIDE_CHAIN}",
            f"      contacts: [{contacts}]",
            f"      max_distance: {POCKET_MAX_DISTANCE_A}",
            "      force: false",
        ]
    if constraints:
        lines.append("constraints:")
        lines += constraints

    if target_template_path is not None:
        # Target-only template. `chain_id` lists the receptor chains ONLY; the peptide is
        # absent from both the template file and this list.
        tmpl_chains = ", ".join(yaml_chain_of[c] for c in chain_ids)
        # CIF, not PDB: the entity sequence boltz aligns the template against has to
        # match the template's own coordinates, and that only survives the trip in a CIF
        # built by `targets.write_target_template_cif` (see its docstring).
        lines += [
            "templates:",
            f"  - cif: {target_template_path}",
            f"    chain_id: [{tmpl_chains}]",
        ]

    return "\n".join(lines) + "\n"


def input_hash(yaml_text: str, config: Boltz2Config) -> str:
    """Hash over the exact input text and the inference settings."""
    payload = json.dumps(
        {"yaml": yaml_text, "config": config.to_dict(), "rev": BOLTZ_REV}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def predict(
    *,
    yaml_path: Path,
    out_dir: Path,
    config: Boltz2Config,
    python_exec: str,
    boltz_cache: str | Path,
    devices: int = 1,
    accelerator: str = "gpu",
    timeout_s: int = 7200,
) -> dict:
    """Run one boltz prediction. Returns a status record; never raises on model failure.

    `--diffusion_samples` is how the four samples per candidate are produced: boltz runs
    the trunk once and draws that many structures from it, which is the shared-preprocessing
    multi-sample inference the task asks for rather than four independent invocations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exec, "-m", "boltz.main", "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--cache", str(boltz_cache),
        "--diffusion_samples", str(config.diffusion_samples),
        "--recycling_steps", str(config.recycling_steps),
        "--sampling_steps", str(config.sampling_steps),
        "--seed", str(config.seed),
        "--output_format", "pdb",
        "--accelerator", accelerator,
        "--devices", str(devices),
        "--write_full_pae",
        # The cuequivariance triangular-multiplication kernels are a speed
        # optimisation, not part of the model, and we cannot have them here: the
        # current `cuequivariance_ops_torch_cu12` (0.13.0) requires torch>=2.11 and
        # pulls CUDA 13, which would replace this env's pinned torch 2.7.0+cu126 and
        # the boltz install built against it. Without the flag boltz still *tries* the
        # kernels and dies with `ModuleNotFoundError: cuequivariance_torch` inside the
        # pairformer -- measured, on both preflight tasks.
        "--no_kernels",
    ]
    if config.use_potentials:
        cmd.append("--use_potentials")

    import time

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        status = "ok" if proc.returncode == 0 else "predictor_error"
        return {
            "status": status,
            "returncode": proc.returncode,
            "runtime_s": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            # Tails only: a full boltz log per task would dwarf the dataset.
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "returncode": None,
            "runtime_s": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            "stdout_tail": "",
            "stderr_tail": f"timed out after {timeout_s}s",
        }
