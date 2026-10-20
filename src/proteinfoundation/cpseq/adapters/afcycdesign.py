"""AfCycDesign adapter: cyclic-aware ColabDesign AF2, as fixed-sequence complex prediction.

AfCycDesign's contribution is a positional encoding, not a model: wrap the pairwise
sequence offset around the ring so AF2 reads the binder's termini as backbone
neighbours. This repo already implements that wrap, including the part that is easy to
get wrong -- `evaluation.cyclic_offset.build_complex_cyclic_offset` patches **only** the
binder's diagonal block and rebuilds nothing, so the large residue-index gap that
ColabDesign puts at the target/binder boundary survives. Rebuilding the whole matrix
from `arange(total_len)` would collapse that gap and tell AF2 the receptor and peptide
are sequence-adjacent -- the same instrument problem, moved to a different chain
boundary.

What this adapter adds is the *fixed-sequence complex prediction* path:

  - design and hallucination are **off**. The sequence enters as one-hot logits scaled
    by 1e9, which is how this repo's AF2 reward model already pins a sequence, and
    nothing optimises it: `run(backprop=False)` with no design steps.
  - `use_binder_template=False` and `rm_template_ic=False`, so the peptide contributes
    **no** template and its bound coordinates never enter the input.
  - `rm_target`, `rm_target_seq`, `rm_target_sc` all False, so the target-only template
    is given in full -- the same target information the other arms get.
  - the output sequence is read back off the predicted structure and compared to the
    request. The sequence is an input, but checking it on the way out is what catches a
    chain-order or offset mistake.

Two conditioning differences from the other arms, both unavoidable and both recorded
rather than papered over:

  **No pocket constraint.** Boltz and HighFold3 accept an explicit pocket/contact
  constraint that enters the forward pass. ColabDesign's `hotspot` argument enters the
  *design loss*, not the structure prediction, so in a fixed-sequence prediction it
  changes nothing. There is no mechanism to hand AF2 a pocket, so this arm gets target
  chains plus target template and no pocket, and `pocket_conditioning` says `none`.

  **Monomer models, not multimer.** The cyclic wrap patches `inputs["offset"]`, which is
  the monomer relative-encoding path; AF-Multimer builds its relative encoding
  differently. AfCycDesign itself uses the monomer _ptm params. Recorded as
  `use_multimer=False`.

Sampling: this arm gives TWO samples, not four
----------------------------------------------
Both halves of this were measured, not assumed.

ColabDesign loads `model_{1,2}_ptm` only when `use_templates=True`, and
`[1,2,3,4,5]` only when it is False (`af/model.py:110-117`) -- because those are the
monomer-ptm parameter sets with template weights. The binder protocol here *is*
templated (the target structure), so exactly 2 parameter sets exist for it. Asking for
models 2 or 3 raised `IndexError`, and no `num_models` argument changes it.

Re-seeding does not substitute. With a fixed sequence, no MSA and dropout off, the
forward pass is deterministic: the same parameter set at seed 42 and seed 777 produced
peptide CA coordinates identical to 0.000000 A. So seeds are not a sample control for
this arm.

`SAMPLE_CEILING = 2` therefore, and a caller asking for 4 gets 2 with
`sample_ceiling_reason` recorded. Best-of-2 and best-of-4 are different statistics and
the summaries have to keep them apart rather than quietly comparing them.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from proteinfoundation.cpseq.chemistry import (
    MAINCHAIN,
    SUPPORTED,
    capability,
)

TOOL = "afcycdesign"

# ColabDesign snapshot: the published release this adapter targets.
COLABDESIGN_ZENODO_DOI = "10.5281/zenodo.13309081"

# One-hot logit scale that pins the sequence through ColabDesign's softmax. Matches
# `rewards.alphafold2_reward`, which already fixes sequences this way.
FIXED_SEQ_LOGIT_SCALE = 1e9

# AF2's canonical 20, in ColabDesign/AF2 order (alphabetical one-letter).
AF2_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


# Templated monomer-ptm AF2 has exactly two parameter sets. See the module docstring --
# this is a property of AF2, not a configuration choice.
SAMPLE_CEILING = 2


@dataclass
class AfCycConfig:
    num_recycles: int = 3
    # One sample per available parameter set. Not (0,1,2,3): models 2+ do not exist on
    # the templated path and asking for them raised IndexError.
    model_nums: tuple[int, ...] = (0, 1)
    use_multimer: bool = False
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "num_recycles": self.num_recycles,
            "model_nums": list(self.model_nums),
            "use_multimer": self.use_multimer,
            "seed": self.seed,
            "fixed_seq_logit_scale": FIXED_SEQ_LOGIT_SCALE,
            "colabdesign_zenodo_doi": COLABDESIGN_ZENODO_DOI,
        }


def chemistry_supported(chemistry: str) -> bool:
    return capability(TOOL, chemistry).status == SUPPORTED


def one_hot_logits(sequence: str) -> np.ndarray:
    """[1, L, 20] logits that softmax to an exact one-hot of `sequence`."""
    idx = [AF2_ALPHABET.index(c) for c in sequence]
    seq = np.zeros((1, len(sequence), 20), dtype=np.float32)
    seq[0, np.arange(len(sequence)), idx] = FIXED_SEQ_LOGIT_SCALE
    return seq


def input_hash(
    *,
    sequence: str,
    template_path: str,
    target_chain: str,
    chemistry: str,
    config: AfCycConfig,
) -> str:
    payload = json.dumps(
        {
            "sequence": sequence,
            "template": str(template_path),
            "target_chain": target_chain,
            "chemistry": chemistry,
            "config": config.to_dict(),
            # The template file's own content, so editing it invalidates the hash.
            "template_sha": hashlib.sha256(Path(template_path).read_bytes()).hexdigest()[:16],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def sequence_from_pdb(pdb_path: Path, chain: str) -> str:
    """One-letter sequence of `chain` in a predicted PDB, in file order."""
    from proteinfoundation.cpseq.candidates import THREE_TO_ONE

    seen: dict[tuple[int, str], str] = {}
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        resname = line[17:20].strip()
        if resname not in THREE_TO_ONE:
            continue
        seen.setdefault((int(line[22:26]), line[26]), THREE_TO_ONE[resname])
    return "".join(seen[k] for k in sorted(seen))


def sequence_from_pdb_when_ready(
    pdb_path: Path, chain: str, *, attempts: int = 10, delay_s: float = 0.2
) -> str:
    """`sequence_from_pdb`, but only once the file is actually finished being written.

    Reading immediately after `save_pdb` intermittently returned an empty sequence for
    the first sample of a run while the identical file parsed correctly moments later --
    a partially written file, not a wrong prediction. Left alone that shows up in the
    dataset as `sequence_matches: false`, i.e. a compliance failure invented by a race.
    Waits for a terminating record before parsing, and reports empty only if it never
    arrives.
    """
    import time

    for _ in range(attempts):
        try:
            text = Path(pdb_path).read_text()
        except OSError:
            text = ""
        if text.rstrip().endswith(("END", "ENDMDL", "TER")):
            return sequence_from_pdb(pdb_path, chain)
        time.sleep(delay_s)
    return sequence_from_pdb(pdb_path, chain) if Path(pdb_path).exists() else ""


def predict(
    *,
    sequence: str,
    template_path: Path,
    target_chain: str,
    chemistry: str,
    out_dir: Path,
    config: AfCycConfig,
    af2_params_dir: str,
    binder_chain_out: str = "B",
) -> dict:
    """Predict one candidate as a complex, once per model index. Returns a record.

    Never raises on a model failure: the per-sample status carries the reason, because a
    prediction failure is a data point about the arm, not an excuse to drop the row.
    """
    if not chemistry_supported(chemistry):
        cap = capability(TOOL, chemistry)
        return {
            "status": "chemistry_unsupported",
            "reason": f"{TOOL}/{chemistry} is {cap.status}: {cap.note}",
            "samples": [],
        }

    from colabdesign import mk_afdesign_model

    from proteinfoundation.evaluation.cyclic_offset import apply_cyclic_offset

    out_dir.mkdir(parents=True, exist_ok=True)
    L = len(sequence)
    seq_logits = one_hot_logits(sequence)

    samples: list[dict] = []
    model = None
    try:
        # No num_models here: it is accepted into **kwargs and changes nothing. The
        # loaded set is fixed at model_{1,2}_ptm by use_templates (af/model.py:110-117).
        model = mk_afdesign_model(
            protocol="binder",
            use_multimer=config.use_multimer,
            num_recycles=config.num_recycles,
            data_dir=af2_params_dir,
        )
        for sample_idx, model_num in enumerate(config.model_nums):
            t0 = time.time()
            try:
                model.prep_inputs(
                    pdb_filename=str(template_path),
                    target_chain=target_chain,
                    binder_len=L,
                    seq=seq_logits,
                    # No peptide template, and no peptide coordinates of any kind.
                    use_binder_template=False,
                    rm_template_ic=False,
                    # The target is given in full: coordinates, sequence, side chains.
                    rm_target=False,
                    rm_target_seq=False,
                    rm_target_sc=False,
                    # AF2 has no forward-pass pocket mechanism; see module docstring.
                    hotspot=None,
                    seed=config.seed + sample_idx,
                )
                cyclic_applied = apply_cyclic_offset(
                    af_model=model, binder_len=L, linkage_type=chemistry
                )
                # Fixed sequence: soft softmax over 1e9-scaled one-hot logits is exact,
                # and no design step runs.
                model.set_opt(hard=False, soft=True, temp=1.0, dropout=False, pssm_hard=False)
                model.run(
                    num_recycles=config.num_recycles,
                    models=[model_num],
                    sample_models=False,
                    backprop=False,
                )
                pdb_out = out_dir / f"sample_{sample_idx}_model{model_num}.pdb"
                model.save_pdb(str(pdb_out))

                got = sequence_from_pdb_when_ready(pdb_out, binder_chain_out)
                samples.append(
                    {
                        "sample_index": sample_idx,
                        "af2_model_num": model_num,
                        "status": "ok",
                        "pdb_path": str(pdb_out),
                        "runtime_s": round(time.time() - t0, 2),
                        "cyclic_offset_applied": bool(cyclic_applied),
                        "sequence_requested": sequence,
                        "sequence_predicted": got,
                        "sequence_matches": got == sequence,
                        "plddt": _scalar(model.aux, "plddt"),
                        "ptm": _scalar(model.aux, "ptm"),
                        "i_ptm": _scalar(model.aux, "i_ptm"),
                        "i_pae": _scalar(model.aux, "i_pae"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                samples.append(
                    {
                        "sample_index": sample_idx,
                        "af2_model_num": model_num,
                        "status": "predictor_error",
                        "pdb_path": None,
                        "runtime_s": round(time.time() - t0, 2),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "setup_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "samples": samples,
        }
    finally:
        if model is not None:
            try:
                model.aux = None
            except Exception:  # noqa: BLE001, S110
                pass

    n_ok = sum(1 for s in samples if s["status"] == "ok")
    return {
        "status": "ok" if n_ok else "all_samples_failed",
        "n_samples_ok": n_ok,
        "sample_ceiling": SAMPLE_CEILING,
        "sample_ceiling_reason": (
            "templated monomer-ptm AF2 has 2 parameter sets, and re-seeding is "
            "bit-identical (measured: 0.000000 A CA deviation), so this arm yields 2 "
            "independent samples regardless of the requested count"
        ),
        "pocket_conditioning": "none",
        "conditioning_note": (
            "AF2 has no forward-pass pocket mechanism; ColabDesign's hotspot enters the "
            "design loss, which a fixed-sequence prediction does not use. Target chains "
            "and target-only template are supplied; pocket is not."
        ),
        "samples": samples,
    }


def _scalar(aux, key: str):
    """Pull a scalar metric out of ColabDesign's aux dict, mean-reducing if per-residue."""
    if aux is None:
        return None
    val = aux.get(key) if hasattr(aux, "get") else None
    if val is None and hasattr(aux, "get"):
        val = (aux.get("log") or {}).get(key)
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64)
    return float(arr.mean()) if arr.size else None
