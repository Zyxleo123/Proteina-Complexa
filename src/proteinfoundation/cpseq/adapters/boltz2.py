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

import numpy as np

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


def _predict_cmd(
    *,
    target: Path,
    out_dir: Path,
    config: Boltz2Config,
    python_exec: str,
    boltz_cache: str | Path,
    devices: int,
    accelerator: str,
    preprocessing_threads: int = 1,
) -> list[str]:
    """The boltz command line. `target` is one yaml, or a directory of them.

    Shared by the single and batched paths deliberately: the inference settings are part
    of the prediction cache key, so a flag present on one path and absent from the other
    would produce structures that the cache treats as interchangeable and are not.

    `--preprocessing-threads 1` is why a batch fits in memory. Pointed at a directory,
    boltz preprocesses the inputs in parallel -- "Processing 9 inputs with 9 threads" --
    and each thread holds its own copy of the working set, so peak memory scales with the
    batch while a single-input run never shows the problem. Measured: 9 inputs in a 32 GB
    task died with SIGKILL (rc -9) in 5-7 s, having produced nothing. Serialising
    preprocessing costs a few seconds per batch and leaves the saving intact, because
    what the batch is amortising is the model load, not the preprocessing.
    """
    cmd = [
        python_exec, "-m", "boltz.main", "predict", str(target),
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
        "--preprocessing-threads", str(preprocessing_threads),
    ]
    if config.use_potentials:
        cmd.append("--use_potentials")
    return cmd


def predict_many(
    *,
    yaml_texts: dict[str, str],
    stage_dir: Path,
    config: Boltz2Config,
    python_exec: str,
    boltz_cache: str | Path,
    devices: int = 1,
    accelerator: str = "gpu",
    timeout_s: int = 7200,
    preprocessing_threads: int = 1,
) -> dict:
    """One boltz invocation for several inputs, which is where the wall time is.

    Measured on an A6000 with potentials on, this environment's settings, and inputs of
    the size this project folds: **44 s fixed + 26 s per structure** (70 s for 1 input,
    145 s for 4, 253 s for 8). The fixed part is process spawn, the torch import, the
    2.2 GB checkpoint read and the model build on the GPU, and it is paid once per
    INVOCATION. Folding one structure per invocation therefore spends 44 of every 70 s
    on setup; nine per invocation spends 44 of every 278.

    Nine is the natural group here -- one target's candidates plus its unchanged peptide,
    at one seed -- and it is already near the asymptote: 31 s/structure at 9 against
    28 s at 18.

    `yaml_texts` is keyed by each input's content hash, which becomes its yaml stem and
    therefore the name of its own output directory. Boltz names every artefact after that
    stem (`<stem>_model_0.pdb`, `confidence_<stem>_model_0.json`), and every reader in
    this project finds those by glob rather than by a fixed name, so the files can be
    filed into per-candidate directories afterwards without renaming.

    Returns the shared invocation record plus `per_key`, mapping each key to the
    directory holding its structures -- absent for any key boltz produced nothing for,
    which the caller must treat as that candidate's own failure rather than the batch's.
    """
    import time

    # The inputs get a directory of their own, holding nothing but the yamls, and the
    # output goes to a SIBLING. Boltz is pointed at a directory here rather than a single
    # file, and it scans that directory for inputs -- so an output directory nested
    # inside it is scanned too, and the run fails. Measured: with `out` as a child, 0 of
    # 6 structures came back; as a sibling, 8 of 8.
    inputs_dir = stage_dir / "inputs"
    out_dir = stage_dir / "out"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, text in yaml_texts.items():
        (inputs_dir / f"{key}.yaml").write_text(text)

    cmd = _predict_cmd(
        target=inputs_dir, out_dir=out_dir, config=config, python_exec=python_exec,
        boltz_cache=boltz_cache, devices=devices, accelerator=accelerator,
        preprocessing_threads=preprocessing_threads,
    )
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        status = "ok" if proc.returncode == 0 else "predictor_error"
        rec = {
            "status": status,
            "returncode": proc.returncode,
            "runtime_s": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        rec = {
            "status": "timeout",
            "returncode": None,
            "runtime_s": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            "stdout_tail": "",
            "stderr_tail": f"timed out after {timeout_s}s",
        }

    # Boltz puts each input's results under `boltz_results_<stage>/predictions/<stem>/`.
    # Locate by stem rather than assuming the wrapper directory's name.
    per_key: dict[str, str] = {}
    for key in yaml_texts:
        hits = [d for d in out_dir.rglob(f"predictions/{key}") if d.is_dir()]
        if hits and any(hits[0].glob("*.pdb")):
            per_key[key] = str(hits[0])
    rec["per_key"] = per_key
    rec["n_requested"] = len(yaml_texts)
    rec["n_produced"] = len(per_key)
    return rec


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
    cmd = _predict_cmd(
        target=yaml_path, out_dir=out_dir, config=config, python_exec=python_exec,
        boltz_cache=boltz_cache, devices=devices, accelerator=accelerator,
    )

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


# Confidence keys boltz writes per sample. Harvested from the sibling
# `confidence_<name>_model_<k>.json` rather than re-running a prediction, and kept under
# boltz's own names so they are never confused with AF2's pLDDT/ptm conventions -- the
# two are different quantities on different scales and must not be pooled.
BOLTZ_CONFIDENCE_KEYS = (
    "confidence_score",
    "ptm",
    "iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


def interface_pae_for_pdb(
    pdb_path: str | Path,
    *,
    binder_chain: str,
    receptor_chains: list[str],
) -> dict:
    """The peptide-target block of boltz's PAE matrix, as two scalars.

    Boltz writes `pae_<stem>.npz` beside `<stem>.pdb`: a token x token matrix of expected
    aligned error in angstroms. For an all-protein complex there is one token per
    residue, in the order the chains appear in the structure file, so the binder's block
    can be located from the PDB itself. Verified on a 285-token matrix against a 275 + 10
    residue complex (`pepbench_A_C_pdb5t70`).

    Two numbers come back:

    - `interface_pae`, the mean over BOTH off-diagonal blocks. PAE is asymmetric -- the
      error in placing i given j's frame is not the error in placing j given i's -- and
      which of the two directions matters depends on what is being asked, so neither is
      privileged and the mean of the pair is reported.
    - `binder_pae`, the mean over the binder's own block, which is the peptide's internal
      confidence rather than a statement about the interface. Carried alongside so a
      confident peptide placed badly and a vague peptide placed well can be told apart.

    This is harvested at scoring time because `retention.py` deletes the npz: its scalars
    are supposed to be on the score row by then, and until this function existed the
    interface block was the one scalar that was not. It is a recorded confidence metric
    and never part of the objective, so a missing matrix returns `{}` -- reported as
    absent, never as zero, the same contract as `confidence_for_pdb`.
    """
    pdb_path = Path(pdb_path)
    sidecar = pdb_path.parent / f"pae_{pdb_path.stem}.npz"
    if not sidecar.exists():
        return {}
    try:
        with np.load(sidecar) as data:
            pae = np.asarray(data["pae"], dtype=np.float32)
    except Exception:  # noqa: BLE001
        return {}
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return {}

    # Chain label per residue, in file order -- the order the tokens are in.
    chains: list[str] = []
    seen: set[tuple[str, str]] = set()
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                key = (line[21], line[22:27].strip())
                if key in seen:
                    continue
                seen.add(key)
                chains.append(key[0])
    except OSError:
        return {}

    # A token count that does not match the residue count means something in this
    # complex is not one-token-per-residue -- a ligand, a modified residue, an atomised
    # chain. The block boundaries would then be wrong, and a wrong interface PAE is worse
    # than none, so this says so instead of slicing anyway.
    if len(chains) != pae.shape[0]:
        return {
            "interface_pae_status": (
                f"token count {pae.shape[0]} != {len(chains)} residues; "
                f"blocks not located"
            )
        }

    labels = np.asarray(chains)
    b = labels == binder_chain
    r = np.isin(labels, list(receptor_chains))
    if not b.any() or not r.any():
        return {"interface_pae_status": "binder or receptor chain absent from structure"}

    cross = np.concatenate([pae[np.ix_(b, r)].ravel(), pae[np.ix_(r, b)].ravel()])
    return {
        "interface_pae": float(cross.mean()),
        "binder_pae": float(pae[np.ix_(b, b)].mean()),
        "interface_pae_status": "ok",
        "n_binder_tokens": int(b.sum()),
        "n_receptor_tokens": int(r.sum()),
    }


def confidence_for_pdb(pdb_path: str | Path) -> dict:
    """Boltz's confidence values for the sample whose structure is at `pdb_path`.

    Boltz writes `<stem>.pdb` and `confidence_<stem>.json` side by side, so the mapping
    is by filename. Returns {} when the file is absent -- a missing confidence is
    reported as absent, never as zero.
    """
    pdb_path = Path(pdb_path)
    sidecar = pdb_path.parent / f"confidence_{pdb_path.stem}.json"
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001
        return {}
    return {
        k: float(data[k])
        for k in BOLTZ_CONFIDENCE_KEYS
        if k in data and isinstance(data[k], (int, float))
    }
