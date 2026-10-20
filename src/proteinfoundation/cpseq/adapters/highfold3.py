"""HighFold3 adapter: cyclic-aware AF3, with its topology controls verified by probing.

HighFold3 is AF3 with two additions -- a Cyclic Position Offset Encoding Matrix
(CycPOEM) and a "cyclization switch" -- exposed as `--head_to_tail` and
`--disulfide_chain_res`. Its install instruction is "refer to the conda version of
AlphaFold3 installation, and replace the feature files by highfold3/model".

What had to be established by experiment
----------------------------------------
The module that implements all of this, `model/cyclic_peptide_matrix`, ships **only** as
`model/__pycache__/cyclic_peptide_matrix.cpython-311.pyc`. There is no `.py` in the
repository (`git ls-files model` shows the `.pyc` tracked and the source absent), so what
the flags do cannot be read -- only exercised. The bytecode does load under CPython
3.11, and its own traceback reveals where it was built:
`/home/admin/.conda/envs/af3-cs/lib/python3.11/site-packages/alphafold3/model/
cyclic_peptide_matrix.py`.

Probing `calc_cyclic_matrix_af3(residue_index, head_to_tail, list_cid_ss, token_wise,
signal_bug_fix)` directly, on both the residue-wise and token-wise paths that
`model/features.py:975,982` call, gives four facts:

**1. The ring is applied to the FIRST chain, always.** The returned matrix is sized from
where `residue_index` returns to 1 -- i.e. the start of the second chain. With a 5-mer
peptide first and an 8-residue receptor second, the matrix is 5x5 and correctly rings the
peptide. With the receptor first, the matrix is 8x8 and the ring lands on the
**receptor**, leaving the peptide linear.

**This is the trap.** AF3's own examples list the protein first, and `--head_to_tail`
takes no chain argument, so the natural input ordering silently applies a cyclic
positional encoding to the target protein and produces confident output answering the
wrong question. The adapter therefore writes the **peptide chain first**, and that is not
a style choice.

**2. `--disulfide_chain_res`'s chain field is ignored.** Documented as
`[[chain, Cys_i, Cys_j]]`, but `[[1, 1, 6]]` and `[[2, 1, 6]]` return matrices that are
bit-identical, on both the residue-wise and token-wise paths. The crosslink is applied to
the first chain's block regardless of what the first element says. So the chain field
cannot be used to target a chain, and peptide-first ordering is required for the
disulfide control too.

**3. Residue indices are 1-based.** `[[1, 2, 5]]` puts the +/-1 offset between 0-based
rows 1 and 4, i.e. residues 2 and 5.

**4. With no topology requested the function returns `[[0]]`** -- a 1x1 placeholder, not
a full linear offset matrix. Worth knowing when comparing a "linear" run against a cyclic
one: a naive elementwise diff against it broadcasts rather than compares.

The encoding itself is correct where it is applied: the 5x5 head-to-tail matrix is the
wrapped shortest-path offset, with residues 0 and L-1 reading as +/-1 apart -- the same
construction as `evaluation.cyclic_offset.cyclic_offset_block`. And unlike the AF2 offset
wrap, `list_cid_ss` encodes a **side-chain** crosslink as a graph shortest path, which is
why HighFold3 can express a disulfide ring that AfCycDesign cannot.

Execution status
----------------
This adapter builds inputs and the command line. Running it needs the AF3 runtime, which
is a separate build; `scripts/cpseq_env_highfold3.sbatch` attempts it and
`docs/README_CPSEQ_DATASET.md` records how far it got. The version constraint is real and
narrow: HighFold3's bytecode requires CPython 3.11, while current AF3 (131ab9e7) sets
`requires-python = ">=3.12"`. The AF3 revision contemporaneous with HighFold3's snapshot,
`a8ecdb2d` (2025-09-08), requires `>=3.11` with `jax==0.4.34` and `dm-haiku==0.0.13`, so
that is the revision to build against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from proteinfoundation.cpseq.chemistry import (
    DISULFIDE,
    MAINCHAIN,
    SUPPORTED,
    capability,
)

TOOL = "highfold3"

HIGHFOLD3_REV = "2ff1ea31e59f3bac70e45aadde60a4a8fed6e199"
# The AF3 revision to build HighFold3 against: the last one before HighFold3's snapshot
# that still allows CPython 3.11, which its bytecode requires.
AF3_REV_FOR_HIGHFOLD3 = "a8ecdb2d7a433c5e9f8510a0f52434a0c55018c4"
AF3_PYTHON = "3.11"

# Chain ids in the generated input. The PEPTIDE IS FIRST -- see point 1 in the module
# docstring. Reversing these two silently rings the receptor.
PEPTIDE_CHAIN = "A"
RECEPTOR_CHAIN_START = "B"


@dataclass
class HighFold3Config:
    num_seeds: int = 4
    num_recycles: int = 10
    seed: int = 42
    # Flash attention etc. are AF3 runtime flags; recorded for provenance.
    flash_attention: bool = True

    def to_dict(self) -> dict:
        return {
            "num_seeds": self.num_seeds,
            "num_recycles": self.num_recycles,
            "seed": self.seed,
            "flash_attention": self.flash_attention,
            "highfold3_rev": HIGHFOLD3_REV,
            "af3_rev": AF3_REV_FOR_HIGHFOLD3,
            "python": AF3_PYTHON,
        }


def chemistry_supported(chemistry: str) -> bool:
    return capability(TOOL, chemistry).status == SUPPORTED


def build_input_json(
    *,
    name: str,
    peptide_sequence: str,
    receptor_sequences: dict[str, str],
    receptor_msas: dict[str, str] | None,
    chemistry: str,
    config: HighFold3Config,
) -> dict:
    """AF3-format input with the PEPTIDE FIRST, which HighFold3's encoding requires.

    MSAs are supplied inline/by path because AF3's genetic databases do not fit the
    quota (see `docs/RESOURCES_COLLECTED.md`); AF3 accepts `unpairedMsaPath`, so the
    alignments computed once per target are reused here exactly as the other arms reuse
    them. The peptide gets an explicitly empty MSA rather than a search.
    """
    if not chemistry_supported(chemistry):
        raise ValueError(
            f"{TOOL} does not support {chemistry!r} (status: {capability(TOOL, chemistry).status})"
        )

    sequences = [
        {
            "protein": {
                "id": PEPTIDE_CHAIN,
                "sequence": peptide_sequence,
                # A candidate peptide has no homologs; an empty paired/unpaired MSA keeps
                # it in single-sequence mode instead of triggering a search per candidate.
                "unpairedMsa": "",
                "pairedMsa": "",
                "templates": [],
            }
        }
    ]
    letters = [c for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ"]
    chain_map = {PEPTIDE_CHAIN: "peptide"}
    for n, orig in enumerate(sorted(receptor_sequences)):
        cid = letters[n]
        chain_map[cid] = orig
        entry: dict = {"id": cid, "sequence": receptor_sequences[orig]}
        if receptor_msas and receptor_msas.get(orig):
            entry["unpairedMsaPath"] = receptor_msas[orig]
            entry["pairedMsa"] = ""
        else:
            entry["unpairedMsa"] = ""
            entry["pairedMsa"] = ""
        entry["templates"] = []
        sequences.append({"protein": entry})

    return {
        "name": name,
        "modelSeeds": [config.seed + k for k in range(config.num_seeds)],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
        # Not part of AF3's schema -- carried alongside so the record says which chain is
        # which without re-deriving it from the ordering.
        "_cpseq_chain_map": chain_map,
        "_cpseq_peptide_chain": PEPTIDE_CHAIN,
        "_cpseq_peptide_is_first": True,
    }


def build_command(
    *,
    json_path: Path,
    out_dir: Path,
    model_dir: str,
    chemistry: str,
    peptide_length: int,
    python_exec: str,
    run_script: str,
    db_dir: str | None = None,
) -> list[str]:
    """The HighFold3 command line, with the topology flag for this chemistry.

    `--head_to_tail` is a global bool and `--disulfide_chain_res`'s chain field is
    ignored, so both rely on the peptide being chain 1 -- which `build_input_json`
    guarantees. The residue indices passed here are 1-based, as probed.
    """
    cmd = [
        python_exec, run_script,
        f"--json_path={json_path}",
        f"--output_dir={out_dir}",
        f"--model_dir={model_dir}",
    ]
    if db_dir:
        cmd.append(f"--db_dir={db_dir}")

    if chemistry == MAINCHAIN:
        cmd.append("--head_to_tail=1")
    elif chemistry == DISULFIDE:
        cmd.append("--head_to_tail=0")
        # [[chain, Cys_i, Cys_j]], 1-based. The chain element is ignored by the
        # implementation (probed), so it is set to 1 and correctness comes from the
        # peptide being the first chain.
        cmd.append(f"--disulfide_chain_res=[[1,1,{peptide_length}]]")
    else:
        raise ValueError(f"{TOOL}: no topology control for {chemistry!r}")
    return cmd


def input_hash(payload: dict, config: HighFold3Config) -> str:
    body = {k: v for k, v in payload.items() if not k.startswith("_cpseq_")}
    return hashlib.sha256(
        json.dumps({"input": body, "config": config.to_dict()}, sort_keys=True).encode()
    ).hexdigest()[:16]


def probe_cyclic_matrix(pyc_path: Path, python311: str) -> dict:
    """Re-run the verification in the module docstring, as a check rather than a claim.

    Kept callable so the facts the adapter depends on can be re-established against a
    different HighFold3 revision instead of being trusted from this docstring.
    """
    import subprocess

    script = (
        "import importlib.util, json\n"
        "import numpy as np\n"
        "import sys\n"
        "spec = importlib.util.spec_from_file_location('cpm', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "pep_first = np.array([1,2,3,4,5, 1,2,3,4,5,6,7,8])\n"
        "rec_first = np.array([1,2,3,4,5,6,7,8, 1,2,3,4,5])\n"
        "a = np.array(m.calc_cyclic_matrix_af3(pep_first, head_to_tail=True, token_wise=False))\n"
        "b = np.array(m.calc_cyclic_matrix_af3(rec_first, head_to_tail=True, token_wise=False))\n"
        "ri = np.array([1,2,3,4,5,6, 1,2,3,4,5,6,7,8])\n"
        "c1 = np.array(m.calc_cyclic_matrix_af3(ri, list_cid_ss=[[1,1,6]], token_wise=False))\n"
        "c2 = np.array(m.calc_cyclic_matrix_af3(ri, list_cid_ss=[[2,1,6]], token_wise=False))\n"
        "print(json.dumps({\n"
        "  'peptide_first_matrix_shape': list(a.shape),\n"
        "  'receptor_first_matrix_shape': list(b.shape),\n"
        "  'ring_follows_first_chain': list(a.shape) == [5,5] and list(b.shape) == [8,8],\n"
        "  'disulfide_chain_field_ignored': bool(np.array_equal(c1, c2)),\n"
        "}))\n"
    )
    proc = subprocess.run(
        [python311, "-I", "-c", script, str(pyc_path)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {"status": "probe_failed", "stderr_tail": proc.stderr[-2000:]}
    return {"status": "ok", **json.loads(proc.stdout.strip().splitlines()[-1])}
