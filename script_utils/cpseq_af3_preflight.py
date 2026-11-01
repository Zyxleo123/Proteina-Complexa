#!/usr/bin/env python3
"""Does the AF3 install predict a cyclic-peptide complex on a GPU here, and in what form?

Two questions, in order, both of which have to be answered before any design is folded:

1. **Does it run?** jax has to see a CUDA device, the weights have to load, and one
   prediction has to finish. The env build ran on a CPU node, so `jax_devices` in its
   provenance says `cpu:0` and proves nothing about the GPU path.
2. **What does it write?** The converter that turns AF3's ligand coordinates back into a
   peptide chain -- so the existing scorer and geometry module can read it -- must be
   written against AF3's real output. This dumps an inventory of that output instead of
   assuming a schema, so the format is established by this run.

Deliberately NOT a scientific result. One control, one seed, single-sequence (no
alignment, no template), so a failure is attributable to the install rather than to the
inputs. Pose quality is the next job's question, with the real MSAs and templates.

Run as a JOB, never in an interactive allocation: processes adopted into a node-hold
job's extern step get a memory ceiling below what a folder needs, and the failure is a
bare SIGKILL with no traceback (`README_CPSEQ_SEARCH_PILOT.md` section 11).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.adapters.af3 import (  # noqa: E402
    AF3_REV,
    AF3Config,
    ReceptorChain,
    build_input_json,
    describe_output,
    predict,
)
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", type=Path, required=True,
                    help="controls.json from cpseq_fetch_controls.py")
    ap.add_argument("--pdb-id", default="5TU6",
                    help="which control to fold (default 5TU6: L=7, the smallest "
                         "mainchain control, so the token count is the lowest available)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--af3-python", required=True)
    ap.add_argument("--af3-run-script", required=True)
    ap.add_argument("--af3-model-dir", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--timeout-s", type=int, default=10800)
    ap.add_argument("--jax-cache-dir", default=None,
                    help="persist jax's compiled kernels here so later folds skip "
                         "AF3's compilation")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    controls = json.loads(args.controls.read_text())
    control = next((c for c in controls
                    if c["pdb_id"].upper() == args.pdb_id.upper()), None)
    if control is None:
        print(f"FATAL: {args.pdb_id} not in {args.controls} "
              f"(have {[c['pdb_id'] for c in controls]})")
        return 1
    if control["chemistry"] != "mainchain":
        print(f"FATAL: {args.pdb_id} is {control['chemistry']}; this arm folds only "
              f"head-to-tail mainchain rings as a SMILES ligand")
        return 1

    structure = Path(control["structure_path"])
    parsed = parse_structure(structure)
    receptor_ids = list(control["receptor_chains"])
    receptors = tuple(
        # Single-sequence on purpose: '' is "no alignment, do not go looking", which is
        # what lets this run without the 1 TB of genetic databases. It is recorded on the
        # record so no later reader mistakes this for an aligned prediction.
        ReceptorChain(chain_id=cid, sequence=chain_sequence(parsed, cid), unpaired_msa="")
        for cid in receptor_ids
    )

    config = AF3Config(model_seeds=(args.seed,), num_diffusion_samples=1,
                       jax_compilation_cache_dir=args.jax_cache_dir)
    try:
        json_text, peptide, budget = build_input_json(
            name=f"preflight_{args.pdb_id.lower()}",
            receptor_chains=receptors,
            sequence=control["sequence"],
            chemistry="mainchain",
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: could not build the AF3 input: {type(exc).__name__}: {exc}")
        return 1

    print(f"=== AF3 preflight: {args.pdb_id} ===")
    print(f"af3_rev          {AF3_REV}")
    print(f"peptide          {control['sequence']} (L={peptide.length}, "
          f"{control['chemistry']})")
    print(f"receptor chains  {receptor_ids} "
          f"({[len(r.sequence) for r in receptors]} residues)")
    print(f"alignment        single-sequence (deliberate; databases not installed)")
    print(f"template         none (this is an install check, not a pose measurement)")
    print(f"macrocycle       ring of {peptide.macrocycle_ring_size} atoms "
          f"(3 x {peptide.length}), closing bond in the SMILES")
    print(f"smiles           {peptide.smiles}")
    print("\ntoken budget")
    for k, v in budget.items():
        print(f"  {k:34s} {v}")
    print(f"\n  NOTE: the peptide costs {budget['ligand_tokenisation_multiplier']}x "
          f"what it would as a polymer chain. That multiplier, not the receptor, is the "
          f"cost difference against Boltz-2.")

    input_json = args.out_dir / "af3_input.json"
    input_json.write_text(json_text)
    print(f"\n[preflight] input -> {input_json}")

    # Is there a GPU, and does jax see it? Asked in AF3's own venv, because that is the
    # jax that will run the model.
    import subprocess
    probe = subprocess.run(
        [args.af3_python, "-c",
         "import jax, json; print(json.dumps({'devices': [str(d) for d in jax.devices()], "
         "'backend': jax.default_backend()}))"],
        capture_output=True, text=True, timeout=600,
    )
    print(f"[preflight] jax: {probe.stdout.strip() or probe.stderr.strip()[-500:]}")
    gpu_visible = '"backend": "gpu"' in probe.stdout or "cuda" in probe.stdout.lower()
    if not gpu_visible:
        print("[preflight] WARNING: jax does not report a GPU backend. Continuing so the "
              "failure is recorded with its own error rather than guessed at, but a CPU "
              "AF3 run is not a usable folder for this arm.")

    run_dir = args.out_dir / "af3_out"
    print(f"[preflight] running AF3 (timeout {args.timeout_s}s) -> {run_dir}")
    t0 = time.time()
    result = predict(
        input_json=input_json,
        out_dir=run_dir,
        config=config,
        python_exec=args.af3_python,
        run_script=args.af3_run_script,
        model_dir=args.af3_model_dir,
        timeout_s=args.timeout_s,
    )
    print(f"[preflight] status={result['status']} returncode={result['returncode']} "
          f"runtime={result['runtime_s']}s (wall {round(time.time() - t0, 1)}s)")

    if result["status"] != "ok":
        print("\n--- stderr tail ---")
        print(result["stderr_tail"])
        print("\n--- stdout tail ---")
        print(result["stdout_tail"][-2000:])

    inventory = describe_output(run_dir)
    record = {
        "pdb_id": args.pdb_id.upper(),
        "af3_rev": AF3_REV,
        "config": config.to_dict(),
        "token_budget": budget,
        "peptide": {
            "sequence": peptide.sequence,
            "smiles": peptide.smiles,
            "n_heavy_atoms": peptide.n_heavy_atoms,
            "macrocycle_ring_size": peptide.macrocycle_ring_size,
        },
        "alignment": "single-sequence (empty unpairedMsa, deliberate)",
        "template": None,
        "jax_probe": probe.stdout.strip() or probe.stderr.strip()[-2000:],
        "prediction": {k: v for k, v in result.items()
                       if k not in ("stdout_tail", "stderr_tail")},
        "stderr_tail": result["stderr_tail"],
        "output_inventory": inventory,
    }
    out_json = args.out_dir / "preflight.json"
    out_json.write_text(json.dumps(record, indent=2))

    print(f"\n[preflight] AF3 wrote {inventory['n_files']} files: "
          f"{inventory['by_suffix']}")
    for f in inventory["files"][:40]:
        print(f"    {f['bytes']:>12,d}  {f['path']}")
    if inventory.get("json_keys"):
        print("\n[preflight] confidence/metric JSON keys (what the scorer can read):")
        for path, keys in inventory["json_keys"].items():
            print(f"    {path}: {keys}")
    if inventory.get("cif_head"):
        for path, head in inventory["cif_head"].items():
            print(f"\n[preflight] head of {path} -- how the ligand entity is represented:")
            for line in head:
                print(f"    {line}")

    print(f"\n[preflight] record -> {out_json}")
    print("[preflight] the output converter (AF3 ligand atoms -> peptide chain B) is "
          "written against THIS inventory, not against a remembered schema.")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
