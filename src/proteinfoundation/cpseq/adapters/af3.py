"""AlphaFold 3 as the sequence folder, with the cyclic peptide as a SMILES ligand.

The arm this serves
-------------------
`random_mask_mpnn_fill` (`README_CPSEQ_MPNN_FILL_SETTINGS.md`) is measured with Boltz-2.
This adapter swaps the MEASUREMENT predictor for AF3 and changes nothing else -- the
proposal stage still threads onto the Boltz-2 reference fold, because changing the
proposer and the folder at once would leave the result unattributable.

How the ring is expressed
-------------------------
Not as a restraint and not as a patched position encoding, but as chemistry: the peptide
is a ligand whose SMILES contains the closing amide bond (`cpseq.smiles`). An open ring
is then not a conformation of the molecule AF3 was given, so nothing has to hold it shut.

What that buys and what it costs:

- **Buys:** the constraint is exact, needs no AF3 patch, and the closing bond is subject
  to the same geometry the rest of the molecule is.
- **Costs tokens.** AF3 tokenises a ligand per ATOM where a polymer is one token per
  residue. Measured with `cpseq.smiles`: 8-9 heavy atoms per residue, so a 14-mer is
  ~128 tokens instead of 14. `token_estimate` reports this before a run.
- **Costs distribution.** AF3's ligands are overwhelmingly small molecules, so a
  peptide-sized SMILES ligand is in-vocabulary but off-distribution. This is why the arm
  is calibrated on the six deposited crystal macrocycles before any design is folded,
  rather than being trusted because the input validates.

Two things that must not be assumed
-----------------------------------
**Seeds.** AF3 does not choose seeds for you. It runs exactly the `modelSeeds` it is
given, times `num_diffusion_samples` structures each. So fold depth stays an explicit
decision, and it is the decision that withdrew four earlier results in this project --
`README_CPSEQ_MPNN_FILL_SETTINGS.md` section 2 fixes it at ten folds for every draw AND
for the reference.

**The target moves.** A template in AF3 is an input feature, not a restraint, so the
receptor is free to move and the complex is jointly predicted. That is the intended
behaviour here and it is why the binding-site displacement from the original structure is
worth reporting -- but it also means an AF3 pose is not comparable fold-for-fold with a
Boltz-2 pose, and the reference must be re-folded under whichever predictor scores the
designs.

Genetic databases are not installed (AF3 wants up to 1 TB). Alignments are supplied
inline from the same cached per-target ColabFold MSAs the Boltz-2 arm uses, so the two
predictors see the same evolutionary input and differ as folders rather than as
pipelines.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from proteinfoundation.cpseq.smiles import CyclicPeptide, cyclic_peptide

TOOL = "af3"

# Pinned upstream revision, matching `docs/RESOURCES_COLLECTED.md` and the env built by
# `scripts/cpseq_env_af3.sbatch`. Recorded on every prediction record: a prediction is
# only comparable to another made by the same weights and the same code.
AF3_REV = "131ab9e7420066523c62546199243f7a9b3e3e78"

# Chain ids the shared evaluator expects downstream. The receptor keeps 'A' and the
# peptide becomes 'B' when the ligand is rebuilt as a peptide chain, so the scorer and
# the geometry module need no AF3-specific branch.
RECEPTOR_CHAIN = "A"
PEPTIDE_CHAIN = "B"

# AF3's own id for the ligand entity in the input JSON. Distinct from PEPTIDE_CHAIN
# because in the AF3 input the peptide is NOT a peptide -- conflating the two is how the
# output parser ends up looking for a polymer that was never requested.
LIGAND_CHAIN_ID = "L"

# AF3 declines to predict beyond its largest compilation bucket, and a ligand-tokenised
# peptide pushes the token count up hard. A ceiling here turns that into a refusal with
# a number attached rather than a crash deep in the model.
MAX_TOKENS = 5120

SUPPORTED_CHEMISTRIES = ("mainchain",)


class AF3InputError(ValueError):
    """The requested prediction cannot be expressed as a valid AF3 job."""


@dataclass(frozen=True)
class AF3Config:
    """Inference settings. Frozen per run and hashed into every prediction id."""

    # Ten seeds, matching the locked Boltz-2 measurement depth so the two predictors are
    # read at the same depth. AF3 produces num_diffusion_samples structures PER seed, so
    # the fold count is len(model_seeds) * num_diffusion_samples.
    model_seeds: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 111)
    num_diffusion_samples: int = 1
    num_recycles: int = 10
    # AF3's default compilation buckets are tuned for protein complexes. Left at the
    # model's own default rather than guessed at; set explicitly only once a measurement
    # says the default wastes compile time on this token profile.
    buckets: tuple[int, ...] | None = None
    # Persist jax's compiled kernels across invocations. This arm shells out once per
    # job, so without a shared cache every fold pays AF3's compilation again -- and
    # since each candidate is folded at ten seeds over twenty complexes, that is the
    # difference between compiling once per bucket and compiling thousands of times.
    # NOT part of the prediction's identity: it changes how long a fold takes and not
    # what it returns, so `to_dict` leaves it out and the input hash does not see it.
    jax_compilation_cache_dir: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_seeds": list(self.model_seeds),
            "num_diffusion_samples": self.num_diffusion_samples,
            "num_recycles": self.num_recycles,
            "buckets": list(self.buckets) if self.buckets else None,
        }

    @property
    def folds_per_prediction(self) -> int:
        return len(self.model_seeds) * self.num_diffusion_samples


@dataclass(frozen=True)
class ReceptorChain:
    """One receptor polymer chain, with the alignment and template AF3 should use."""

    chain_id: str
    sequence: str
    # Cached ColabFold alignment, as A3M text. Empty string is a DELIBERATE
    # single-sequence prediction and is recorded as such; None means "not supplied",
    # which AF3 would answer by running its own data pipeline against databases this
    # install does not have.
    unpaired_msa: str | None = None
    # mmCIF text of the original structure, used as a template. A template is a feature
    # and not a restraint, so supplying it does not pin the receptor.
    template_mmcif: str | None = None
    template_query_indices: tuple[int, ...] | None = None
    template_template_indices: tuple[int, ...] | None = None


def chemistry_supported(chemistry: str) -> bool:
    return chemistry in SUPPORTED_CHEMISTRIES


def token_estimate(receptor_chains: tuple[ReceptorChain, ...], peptide: CyclicPeptide) -> dict:
    """Token budget for this job, before it is submitted.

    Reported rather than inferred from a wall clock afterwards, because the per-atom
    ligand tokenisation is the main cost difference against Boltz-2 and it scales with
    peptide length in a way the residue count hides.
    """
    receptor = sum(len(c.sequence) for c in receptor_chains)
    ligand = peptide.n_heavy_atoms
    return {
        "receptor_tokens": receptor,
        "ligand_tokens": ligand,
        "total_tokens": receptor + ligand,
        "peptide_length": peptide.length,
        "tokens_per_residue_as_ligand": round(ligand / peptide.length, 2),
        "tokens_if_polymer": peptide.length,
        "ligand_tokenisation_multiplier": round(ligand / peptide.length, 2),
    }


def build_input_json(
    *,
    name: str,
    receptor_chains: tuple[ReceptorChain, ...],
    sequence: str,
    chemistry: str = "mainchain",
    config: AF3Config | None = None,
) -> tuple[str, CyclicPeptide, dict]:
    """The AF3 input JSON for one candidate. Returns (json_text, peptide, token_budget).

    Raises rather than degrading: a job that silently loses its template, its alignment
    or its ring would still predict something, and that something would be scored and
    reported as this arm's output.
    """
    config = config or AF3Config()
    if not chemistry_supported(chemistry):
        raise AF3InputError(
            f"chemistry {chemistry!r} is not expressible here (have "
            f"{SUPPORTED_CHEMISTRIES}). A disulfide or isopeptide ring is a different "
            f"molecule, so build its SMILES explicitly rather than folding this one."
        )
    if not receptor_chains:
        raise AF3InputError("no receptor chain: this arm predicts a complex, not a peptide")

    peptide = cyclic_peptide(sequence, chemistry=chemistry)
    budget = token_estimate(receptor_chains, peptide)
    if budget["total_tokens"] > MAX_TOKENS:
        raise AF3InputError(
            f"{budget['total_tokens']} tokens exceeds the {MAX_TOKENS} ceiling "
            f"({budget['receptor_tokens']} receptor + {budget['ligand_tokens']} ligand; "
            f"the peptide costs {budget['ligand_tokenisation_multiplier']}x what it "
            f"would as a polymer chain)"
        )

    sequences: list[dict] = []
    for chain in receptor_chains:
        entry: dict = {"id": chain.chain_id, "sequence": chain.sequence}
        if chain.unpaired_msa is None:
            raise AF3InputError(
                f"receptor chain {chain.chain_id} has no alignment. Pass the cached "
                f"ColabFold A3M, or pass '' to request a single-sequence prediction "
                f"deliberately. Leaving it unset makes AF3 run its own data pipeline "
                f"against genetic databases this install does not have."
            )
        # AF3 reads an empty string as "no alignment, do not go looking", which is what
        # makes an intentional single-sequence run possible without the databases.
        entry["unpairedMsa"] = chain.unpaired_msa
        entry["pairedMsa"] = ""
        if chain.template_mmcif is None:
            entry["templates"] = []
        else:
            if (chain.template_query_indices is None
                    or chain.template_template_indices is None):
                raise AF3InputError(
                    f"chain {chain.chain_id} has template mmCIF but no index mapping; "
                    f"AF3 needs queryIndices/templateIndices to align the two"
                )
            entry["templates"] = [{
                "mmcif": chain.template_mmcif,
                "queryIndices": list(chain.template_query_indices),
                "templateIndices": list(chain.template_template_indices),
            }]
        sequences.append({"protein": entry})

    # The peptide, as a molecule. `smiles` carries the closing bond, so no bondedAtomPairs
    # entry is needed and none is emitted -- the ring is internal to this one entity.
    sequences.append({"ligand": {"id": LIGAND_CHAIN_ID, "smiles": peptide.smiles}})

    job = {
        "name": name,
        "modelSeeds": list(config.model_seeds),
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
    }
    return json.dumps(job, indent=2, sort_keys=True), peptide, budget


def input_hash(json_text: str, config: AF3Config) -> str:
    """Identity of a prediction: the whole input plus the resolved inference settings.

    Covers the candidate sequence, the receptor representation, the alignments, the
    template, the topology (via the SMILES), the seeds and the predictor revision -- so
    a cache hit is a hit on the same prediction and not merely on the same sequence.
    """
    payload = json.dumps(
        {
            "input": json_text,
            "config": config.to_dict(),
            "tool": TOOL,
            "revision": AF3_REV,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _predict_cmd(
    *,
    input_json: Path,
    out_dir: Path,
    python_exec: str,
    run_script: str | Path,
    model_dir: str | Path,
    config: AF3Config,
) -> list[str]:
    cmd = [
        str(python_exec), str(run_script),
        f"--json_path={input_json}",
        f"--output_dir={out_dir}",
        f"--model_dir={model_dir}",
        f"--num_diffusion_samples={config.num_diffusion_samples}",
        f"--num_recycles={config.num_recycles}",
        # The databases are not installed, so the data pipeline must not run. The
        # alignments and templates travel inline in the JSON instead.
        "--norun_data_pipeline",
        "--run_inference=true",
    ]
    if config.buckets:
        cmd.append(f"--buckets={','.join(str(b) for b in config.buckets)}")
    if config.jax_compilation_cache_dir:
        Path(config.jax_compilation_cache_dir).mkdir(parents=True, exist_ok=True)
        cmd.append(f"--jax_compilation_cache_dir={config.jax_compilation_cache_dir}")
    return cmd


def predict(
    *,
    input_json: Path,
    out_dir: Path,
    config: AF3Config,
    python_exec: str,
    run_script: str | Path,
    model_dir: str | Path,
    timeout_s: int = 14400,
) -> dict:
    """Run one AF3 job. Returns a status record; never raises on model failure.

    Shelled out to AF3's own venv for the same reason Boltz-2 is: AF3 pins jax and numpy
    against the complexa environment's torch stack, and resolving them together removes
    one of the two.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _predict_cmd(
        input_json=input_json, out_dir=out_dir, python_exec=python_exec,
        run_script=run_script, model_dir=model_dir, config=config,
    )
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return {
            "status": "ok" if proc.returncode == 0 else "predictor_error",
            "returncode": proc.returncode,
            "runtime_s": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            "predictor": TOOL,
            "predictor_revision": AF3_REV,
            # Tails only: a full AF3 log per task would dwarf the results.
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "returncode": None,
            "runtime_s": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            "predictor": TOOL,
            "predictor_revision": AF3_REV,
            "stdout_tail": "",
            "stderr_tail": f"timed out after {timeout_s}s",
        }


@dataclass(frozen=True)
class CifAtom:
    group: str
    element: str
    atom_name: str
    comp_id: str
    asym_id: str
    seq_id: str
    x: float
    y: float
    z: float
    b_factor: float
    auth_seq_id: str
    auth_asym_id: str


def read_model_cif(path: str | Path) -> list[CifAtom]:
    """Atom records from an AF3 model mmCIF, in file order.

    A hand-rolled reader for exactly one loop, rather than a dependency: AF3 writes a
    fixed `_atom_site` loop and the column ORDER is read from the header rather than
    assumed, so a reordering upstream cannot silently shift fields.
    """
    path = Path(path)
    columns: list[str] = []
    atoms: list[CifAtom] = []
    in_loop = False
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            columns.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop and (stripped.startswith("ATOM") or stripped.startswith("HETATM")):
            parts = stripped.split()
            if len(parts) < len(columns):
                continue
            row = dict(zip(columns, parts))
            atoms.append(CifAtom(
                group=row["group_PDB"],
                element=row["type_symbol"],
                atom_name=row["label_atom_id"],
                comp_id=row["label_comp_id"],
                asym_id=row["label_asym_id"],
                seq_id=row.get("label_seq_id", "."),
                x=float(row["Cartn_x"]), y=float(row["Cartn_y"]), z=float(row["Cartn_z"]),
                b_factor=float(row.get("B_iso_or_equiv", 0.0)),
                auth_seq_id=row.get("auth_seq_id", row.get("label_seq_id", ".")),
                auth_asym_id=row.get("auth_asym_id", row["label_asym_id"]),
            ))
            continue
        if in_loop and stripped in ("#", "") and atoms:
            break
    if not columns:
        raise AF3InputError(f"{path} has no _atom_site loop")
    return atoms


class AF3MappingError(ValueError):
    """AF3's output could not be mapped back onto the requested peptide."""


def map_ligand_atoms(
    ligand: list[CifAtom], peptide: CyclicPeptide, *, bond_tolerance_A: tuple[float, float] = (1.0, 2.0)
) -> list[tuple[CifAtom, int]]:
    """Pair each AF3 ligand atom with the molecule atom it is, and PROVE the pairing.

    AF3 returns ligand coordinates in the order of the SMILES STRING, which is not the
    order of the molecule RDKit built -- measured on INPYLYP, SMILES position 0 is
    molecule atom 7 (ILE1 CD1) and not atom 0 (ILE1 N). Mapping by position without the
    permutation assigns every coordinate to the wrong atom and still yields something
    that looks like a peptide, scores, and produces an energy. So the mapping is
    asserted on two independent grounds before it is used:

    1. **Names.** AF3 names a SMILES ligand's atoms `<element><running index>` in SMILES
       order. The expected names are generated from the molecule and must match the file
       atom for atom.
    2. **Bonds.** Every bond of the molecule must be a plausible bond length in AF3's
       coordinates under the mapping. A wrong permutation scrambles this immediately.

    Both held on the 5TU6 preflight (62/62 names, 66/66 bonds). Either failing raises.
    """
    if peptide.mol is None:
        raise AF3MappingError("peptide has no molecule to map against")
    if len(ligand) != peptide.n_heavy_atoms:
        raise AF3MappingError(
            f"AF3 returned {len(ligand)} ligand atoms, the peptide has "
            f"{peptide.n_heavy_atoms}. These are not the same molecule."
        )

    expected = peptide.predicted_atom_names()
    observed = tuple(a.atom_name for a in ligand)
    if observed != expected:
        first = next((i for i, (o, e) in enumerate(zip(observed, expected)) if o != e), 0)
        raise AF3MappingError(
            f"AF3's ligand atom names do not match the expected SMILES-order naming; "
            f"first difference at position {first}: file {observed[first]!r} vs expected "
            f"{expected[first]!r}. The naming convention has changed, so the coordinates "
            f"must NOT be attached to residues by position -- that would mis-assign every "
            f"atom while still producing a plausible-looking peptide."
        )

    pairs = [(atom, peptide.smiles_atom_order[k]) for k, atom in enumerate(ligand)]
    position = {mol_idx: k for k, mol_idx in enumerate(peptide.smiles_atom_order)}
    lo, hi = bond_tolerance_A
    offenders = []
    for bond in peptide.mol.GetBonds():
        i, j = position[bond.GetBeginAtomIdx()], position[bond.GetEndAtomIdx()]
        a, b = ligand[i], ligand[j]
        d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        if not (lo <= d <= hi):
            offenders.append((a.atom_name, b.atom_name, round(d, 3)))
    if offenders:
        raise AF3MappingError(
            f"{len(offenders)} of {peptide.mol.GetNumBonds()} molecular bonds are not "
            f"bond-length distances in AF3's coordinates under this mapping "
            f"(e.g. {offenders[:5]}). Either the permutation is wrong or the returned "
            f"ligand is not the molecule that was requested."
        )
    return pairs


def _pdb_atom_line(
    serial: int, atom_name: str, res_name: str, chain: str, res_seq: int,
    x: float, y: float, z: float, b_factor: float, element: str,
) -> str:
    # The atom-name field is column-sensitive: a name shorter than four characters whose
    # element is one letter starts at column 14, not 13. Getting this wrong makes a CB
    # read as a different atom to every downstream parser, including Rosetta's.
    name = atom_name if len(atom_name) >= 4 else (
        f" {atom_name:<3s}" if len(element) == 1 else f"{atom_name:<4s}"
    )
    return (
        f"ATOM  {serial:5d} {name}{'':1s}{res_name:>3s} {chain:1s}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b_factor:6.2f}          {element:>2s}"
    )


def write_complex_pdb(
    *,
    cif_path: str | Path,
    peptide: CyclicPeptide,
    out_path: str | Path,
    peptide_chain: str = PEPTIDE_CHAIN,
    ligand_comp_prefix: str = "LIG_",
) -> dict:
    """Rewrite an AF3 model as a PDB the shared evaluator can read unchanged.

    The point of this function: downstream, `scorer.score_structure`,
    `geometry.closure_geometry` and `validity_report` all expect a PEPTIDE CHAIN with
    residue names, residue numbers and PDB atom names. AF3 returned a bag of ligand
    atoms called C1, N1, O1 in one residue. This restores the peptide identity that the
    SMILES representation gave up, using the mapping proved in `map_ligand_atoms`, so no
    part of the evaluator needs an AF3-specific branch and the AF3 and Boltz-2 arms are
    scored by identical code.

    Receptor atoms are passed through with their own chain ids, residue numbering and
    atom names, so binding-site displacement against the original structure stays
    computable per residue.
    """
    atoms = read_model_cif(cif_path)
    ligand = [a for a in atoms if a.comp_id.startswith(ligand_comp_prefix)]
    receptor = [a for a in atoms if not a.comp_id.startswith(ligand_comp_prefix)]
    if not ligand:
        raise AF3MappingError(
            f"no ligand atoms (comp_id starting {ligand_comp_prefix!r}) in {cif_path}; "
            f"comp_ids present: {sorted({a.comp_id for a in atoms})[:12]}"
        )

    receptor_chain_ids = sorted({a.auth_asym_id for a in receptor})
    if peptide_chain in receptor_chain_ids:
        raise AF3MappingError(
            f"receptor already uses chain {peptide_chain!r} ({receptor_chain_ids}), "
            f"which is where the peptide has to go. Rename the receptor chains in the "
            f"AF3 input rather than letting two chains collide in the output."
        )

    pairs = map_ligand_atoms(ligand, peptide)

    lines: list[str] = []
    serial = 1
    for atom in receptor:
        try:
            res_seq = int(atom.auth_seq_id)
        except ValueError:
            continue
        lines.append(_pdb_atom_line(
            serial, atom.atom_name, atom.comp_id, atom.auth_asym_id, res_seq,
            atom.x, atom.y, atom.z, atom.b_factor, atom.element,
        ))
        serial += 1
    lines.append("TER")

    # The peptide, in residue order then atom order within a residue, so the file reads
    # like a polymer rather than like the SMILES it arrived as.
    by_residue: dict[int, list[tuple[CifAtom, int]]] = {}
    for atom, mol_idx in pairs:
        by_residue.setdefault(peptide.atom_records[mol_idx].residue_number, []).append(
            (atom, mol_idx)
        )
    for res_num in sorted(by_residue):
        for atom, mol_idx in by_residue[res_num]:
            record = peptide.atom_records[mol_idx]
            lines.append(_pdb_atom_line(
                serial, record.atom_name, record.residue_name, peptide_chain,
                record.residue_number, atom.x, atom.y, atom.z, atom.b_factor,
                atom.element,
            ))
            serial += 1
    lines.append("TER")

    # CONECT for the closing bond. The ring is the whole point of this arm, and a PDB
    # with no CONECT asserts a linear peptide to anything that reads bonds from the file.
    position = {mol_idx: k for k, (_, mol_idx) in enumerate(pairs)}
    c_idx, n_idx = peptide.closure_atom_indices()
    serial_of: dict[int, int] = {}
    s = len([a for a in receptor if a.auth_seq_id.lstrip("-").isdigit()]) + 1
    for res_num in sorted(by_residue):
        for _, mol_idx in by_residue[res_num]:
            serial_of[mol_idx] = s
            s += 1
    lines.append(f"CONECT{serial_of[c_idx]:5d}{serial_of[n_idx]:5d}")
    lines.append("END")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    a, b = ligand[position[c_idx]], ligand[position[n_idx]]
    closure_distance = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
    return {
        "pdb_path": str(out_path),
        "receptor_chains": receptor_chain_ids,
        "peptide_chain": peptide_chain,
        "n_receptor_atoms": sum(1 for a in receptor if a.auth_seq_id.lstrip("-").isdigit()),
        "n_peptide_atoms": len(pairs),
        "peptide_sequence": peptide.sequence,
        # Measured on AF3's OWN coordinates, before any relax. This is the predictor's
        # closure, which under v3 is also the structure validity is judged on.
        "af3_closure_bond_A": round(closure_distance, 3),
        "mapping_verified": "atom names (all) and bond lengths (all) under the "
                            "SMILES output-order permutation",
    }


def describe_output(out_dir: Path) -> dict:
    """Inventory whatever AF3 wrote, without assuming its layout.

    This exists because the output converter has to be written against AF3's real output
    and not against a remembered schema. The preflight calls it and prints the result, so
    the first GPU run establishes the format rather than testing a guess about it.
    """
    out_dir = Path(out_dir)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    by_suffix: dict[str, int] = {}
    for p in files:
        by_suffix[p.suffix or "<none>"] = by_suffix.get(p.suffix or "<none>", 0) + 1

    inventory: dict = {
        "root": str(out_dir),
        "n_files": len(files),
        "by_suffix": by_suffix,
        "files": [
            {"path": str(p.relative_to(out_dir)), "bytes": p.stat().st_size}
            for p in files[:200]
        ],
    }

    # Confidence JSONs are small and their KEYS are what the scorer needs to know about,
    # so record the keys rather than the values.
    for p in files:
        if p.suffix == ".json" and p.stat().st_size < 2_000_000:
            try:
                doc = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if isinstance(doc, dict):
                inventory.setdefault("json_keys", {})[str(p.relative_to(out_dir))] = \
                    sorted(doc.keys())[:40]

    # First lines of each mmCIF, enough to see how the ligand entity is named and
    # whether atom ordering is recoverable.
    for p in files:
        if p.suffix in (".cif", ".mmcif"):
            head = p.read_text(errors="replace").splitlines()[:40]
            inventory.setdefault("cif_head", {})[str(p.relative_to(out_dir))] = head
            break
    return inventory
