"""Stage the verified experimental controls as parent contexts, so the same pipeline runs them.

The controls are experimental CP–target complexes (see `cpseq_fetch_controls.py`, which
accepts a structure only if its ring actually closes). To get **receptor-aligned peptide
pose recovery** out of them, they have to go through exactly the same path as the LNR
candidates: target-only template, cached receptor MSA, same conditioning pocket, same
predictors, same evaluator. Anything else and a pose-recovery number would be measuring
a different pipeline than the one the dataset was built with.

Each control contributes **one candidate: its own native sequence**, unmutated. The point
is not to explore sequence space on a crystal structure, it is to ask whether the
pipeline puts a known peptide back where the crystal has it.

The crystal pose is kept as `reference_complex_path` so
`scorer.receptor_aligned_peptide_rmsd` can compare against it. The peptide is **excluded
from the template**, exactly as for the LNR parents, so the answer is not supplied in the
input.

These are **controls, not a prospective benchmark**: every one predates the predictors'
training data and may sit inside it. They check that the implementation and the
capabilities work, nothing more.

Usage:
    python script_utils/cpseq_build_control_contexts.py \
        --controls /data/user_data/yixiz/cpseq/controls/controls.json \
        --out-dir /data/user_data/yixiz/cpseq/controls_tranche \
        --boltz-python /data/user_data/yixiz/envs/boltz2/.venv/bin/python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import chemistry as chem  # noqa: E402
from proteinfoundation.cpseq.contexts import (  # noqa: E402
    POCKET_RADIUS_A,
    ParentContext,
    chain_sequence,
    parse_structure,
    pocket_for,
)
from proteinfoundation.cpseq.targets import (  # noqa: E402
    POCKET_CONDITIONING_MAX_CONTACTS,
    POCKET_CONDITIONING_MAX_DIST_A,
    fetch_receptor_msa,
    pocket_contacts_for,
    receptor_chain_map,
    sequence_key,
    write_target_only_template,
    write_target_template_cif,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--boltz-python", required=True)
    ap.add_argument("--skip-msa", action="store_true")
    args = ap.parse_args()

    man = args.out_dir / "manifest"
    man.mkdir(parents=True, exist_ok=True)
    tmpl_dir = args.out_dir / "targets"
    msa_dir = args.out_dir / "msa_cache"

    controls = json.loads(args.controls.read_text())
    parent_rows, target_rows, cand_rows, msa_records = [], [], [], {}

    for c in controls:
        path = Path(c["structure_path"])
        parsed = parse_structure(path)
        pep_chain = c["peptide_chain"]
        rec_chains = tuple(c["receptor_chains"])
        local, resseq, mindist = pocket_for(parsed, rec_chains, pep_chain)

        ctx = ParentContext(
            parent_id=f"ctrl_{c['pdb_id'].lower()}",
            source=f"RCSB:{c['pdb_id']}",
            pdb_id=c["pdb_id"],
            structure_path=str(path),
            receptor_chains=rec_chains,
            peptide_chain=pep_chain,
            reference_sequence=c["sequence"],
            peptide_length=c["peptide_length"],
            pocket_residues=local,
            pocket_resseq=resseq,
            pocket_min_dist=mindist,
            receptor_sequences={ch: chain_sequence(parsed, ch) for ch in rec_chains},
            # Controls are their own family: they are not LNR targets and must not be
            # pooled into an LNR family by accident.
            family_id=f"fam_ctrl_{c['pdb_id'].lower()}",
            split="control",
            is_control=True,
            note="experimental CP-target complex; ring verified by measurement",
        )

        rec = write_target_only_template(ctx, tmpl_dir / f"{ctx.parent_id}_target.pdb")
        cif = write_target_template_cif(
            Path(rec["template_path"]), tmpl_dir / f"{ctx.parent_id}_target.cif", args.boltz_python
        )
        chain_map = receptor_chain_map(ctx)
        pocket = pocket_contacts_for(ctx)

        msa_paths = {}
        for orig, seq in sorted(ctx.receptor_sequences.items()):
            key = sequence_key(seq)
            if args.skip_msa:
                msa_paths[chain_map[orig]] = None
                continue
            if key not in msa_records:
                print(f"[ctrl] {ctx.parent_id} chain {orig} (L={len(seq)}) -> MSA {key}")
                msa_records[key] = fetch_receptor_msa(seq, msa_dir, args.boltz_python) | {
                    "sequence_length": len(seq)
                }
            msa_paths[chain_map[orig]] = msa_records[key].get("msa_path")

        parent_rows.append(dict(ctx.to_row(), selected_for_tranche=True))
        target_rows.append({
            "parent_id": ctx.parent_id,
            "source": ctx.source,
            "pdb_id": ctx.pdb_id,
            "reference_sequence": ctx.reference_sequence,
            "peptide_length": ctx.peptide_length,
            "receptor_chains_original": ",".join(rec_chains),
            "chain_map": json.dumps(chain_map),
            "target_chain_spec": ",".join(chain_map[ch] for ch in sorted(rec_chains)),
            "target_template_path": rec["template_path"],
            "target_template_cif": cif["cif_path"],
            "target_template_cif_entity_lengths": json.dumps(cif["written"]),
            "template_n_atoms": rec["n_atoms"],
            "template_n_residues": json.dumps(rec["n_residues_per_chain"]),
            "peptide_chain_excluded_from_template": rec["peptide_chain_excluded"],
            "n_pocket_residues_descriptive": sum(len(v) for v in local.values()),
            "pocket_radius_descriptive_A": POCKET_RADIUS_A,
            "n_pocket_contacts_conditioning": len(pocket),
            "pocket_conditioning_max_contacts": POCKET_CONDITIONING_MAX_CONTACTS,
            "pocket_conditioning_max_dist_A": POCKET_CONDITIONING_MAX_DIST_A,
            "pocket_contacts": json.dumps([[a, b] for a, b in pocket]),
            "receptor_msa_paths": json.dumps(msa_paths),
            "family_id": ctx.family_id,
            "split": "control",
            # What pose recovery is measured against. Never fed to a predictor.
            "reference_complex_path": str(path),
            "reference_peptide_chain": pep_chain,
            "note": ctx.note,
        })

        # One candidate: the native sequence, unmutated.
        cand_rows.append({
            "candidate_id": f"{ctx.parent_id}__{c['chemistry']}__native",
            "parent_id": ctx.parent_id,
            "chemistry": c["chemistry"],
            "sequence": c["sequence"],
            "reference_sequence": c["sequence"],
            "peptide_length": c["peptide_length"],
            "mutated_positions": "",
            "anchor_positions": "",
            "n_mutations": 0,
            "mutation_fraction": 0.0,
            "nominal_bands": "0.000",
            "variant_index": 0,
            "generation_seed": 0,
            "is_reference": True,
            "note": "experimental control, native sequence",
            "identity_to_reference": 1.0,
            "family_id": ctx.family_id,
            "split": "control",
        })

    pd.DataFrame(parent_rows).to_csv(man / "parents.csv", index=False)
    pd.DataFrame(target_rows).to_csv(man / "targets.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(man / "candidates.csv", index=False)
    pd.DataFrame(chem.capability_table_rows()).to_csv(man / "capability_table.csv", index=False)
    (man / "msa_cache.json").write_text(json.dumps(msa_records, indent=2))

    cfg = {
        "source": str(args.controls),
        "n_controls": len(controls),
        "chemistries": sorted({c["chemistry"] for c in controls}),
        "pocket_radius_A": POCKET_RADIUS_A,
        "note": "controls check implementation and capability, not prospective accuracy; "
                "all predate the predictors' training data and may sit inside it",
    }
    cfg["config_hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    (man / "build_config.json").write_text(json.dumps(cfg, indent=2))

    # The frozen tolerances are shared with the main tranche, not re-derived here --
    # deriving them from the controls and then judging the controls by them would be
    # circular.
    print(f"\n[ctrl] {len(cand_rows)} control candidates over {len(parent_rows)} parents "
          f"-> {man}")
    print(f"[ctrl] config_hash={cfg['config_hash']}")
    print(f"[ctrl] chemistries: {cfg['chemistries']}")
    print("[ctrl] copy manifest/geometry_tolerances.json from the main tranche before scoring "
          "(shared, not re-derived -- judging controls by thresholds derived from them "
          "would be circular)")


if __name__ == "__main__":
    main()
