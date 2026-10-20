"""Prepare and cache the shared target conditioning for every selected parent.

Per parent: one target-only template PDB, and one receptor MSA per unique receptor
sequence. Both are cached, so re-running is cheap and a candidate never triggers an
alignment search -- the MSA is keyed by receptor sequence hash, not by candidate.

Writes `manifest/targets.csv` and `manifest/msa_cache.json` under `--out-dir`.

Usage:
    python script_utils/cpseq_prepare_targets.py \
        --out-dir /data/user_data/yixiz/cpseq/tranche1 \
        --boltz-python /data/user_data/yixiz/envs/boltz2/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.contexts import load_lnr_contexts  # noqa: E402
from proteinfoundation.cpseq.contexts import POCKET_RADIUS_A  # noqa: E402
from proteinfoundation.cpseq.targets import (  # noqa: E402
    POCKET_CONDITIONING_MAX_CONTACTS,
    POCKET_CONDITIONING_MAX_DIST_A,
    fetch_receptor_msa,
    pocket_contacts_for,
    receptor_chain_map,
    receptor_msa_path,
    sequence_key,
    write_target_only_template,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lnr-root", type=Path, default=Path("/data/user_data/yixiz/LNR"))
    ap.add_argument("--boltz-python", required=True)
    ap.add_argument("--skip-msa", action="store_true", help="templates only")
    args = ap.parse_args()

    man = args.out_dir / "manifest"
    man.mkdir(parents=True, exist_ok=True)
    tmpl_dir = args.out_dir / "targets"
    msa_dir = args.out_dir / "msa_cache"

    parents = pd.read_csv(man / "parents.csv")
    selected = set(parents.loc[parents["selected_for_tranche"], "parent_id"])
    print(f"[targets] {len(selected)} parents selected for the tranche")

    contexts, _ = load_lnr_contexts(args.lnr_root)
    contexts = [c for c in contexts if c.parent_id in selected]

    rows, msa_records = [], {}
    for ctx in contexts:
        rec = write_target_only_template(ctx, tmpl_dir / f"{ctx.parent_id}_target.pdb")
        chain_map = receptor_chain_map(ctx)
        pocket = pocket_contacts_for(ctx)

        msa_paths = {}
        for orig, seq in sorted(ctx.receptor_sequences.items()):
            key = sequence_key(seq)
            if args.skip_msa:
                msa_paths[chain_map[orig]] = None
                continue
            if key in msa_records:
                msa_paths[chain_map[orig]] = msa_records[key].get("msa_path")
                continue
            print(f"[targets] {ctx.parent_id} chain {orig} (L={len(seq)}) -> MSA {key}")
            out = fetch_receptor_msa(seq, msa_dir, args.boltz_python)
            msa_records[key] = out | {"sequence_length": len(seq)}
            msa_paths[chain_map[orig]] = out.get("msa_path")
            print(f"           {out['status']}"
                  + (f", {out.get('n_sequences')} sequences" if out.get("n_sequences") else "")
                  + (f", {out.get('runtime_s')}s" if out.get("runtime_s") else ""))

        rows.append(
            {
                "parent_id": ctx.parent_id,
                "source": ctx.source,
                "pdb_id": ctx.pdb_id,
                "reference_sequence": ctx.reference_sequence,
                "peptide_length": ctx.peptide_length,
                "receptor_chains_original": ",".join(ctx.receptor_chains),
                "chain_map": json.dumps(chain_map),
                # Chain spec in the predictors' own ids, comma separated.
                "target_chain_spec": ",".join(chain_map[c] for c in sorted(ctx.receptor_chains)),
                "target_template_path": rec["template_path"],
                "template_n_atoms": rec["n_atoms"],
                "template_n_residues": json.dumps(rec["n_residues_per_chain"]),
                "peptide_chain_excluded_from_template": rec["peptide_chain_excluded"],
                # Two pockets, kept apart on purpose. The descriptive one is the full
                # 8 A shell and is what the report characterises the site with; the
                # conditioning one is the small satisfiable set actually handed to the
                # predictors (see `targets.pocket_contacts_for` for why it must be small).
                "n_pocket_residues_descriptive": sum(
                    len(v) for v in ctx.pocket_residues.values()
                ),
                "pocket_radius_descriptive_A": POCKET_RADIUS_A,
                "n_pocket_contacts_conditioning": len(pocket),
                "pocket_conditioning_max_contacts": POCKET_CONDITIONING_MAX_CONTACTS,
                "pocket_conditioning_max_dist_A": POCKET_CONDITIONING_MAX_DIST_A,
                "pocket_contacts": json.dumps([[c, i] for c, i in pocket]),
                "receptor_msa_paths": json.dumps(msa_paths),
                "family_id": ctx.family_id,
                "split": ctx.split,
                "note": ctx.note,
            }
        )

    pd.DataFrame(rows).to_csv(man / "targets.csv", index=False)
    (man / "msa_cache.json").write_text(json.dumps(msa_records, indent=2))

    n_msa = sum(1 for r in msa_records.values() if r.get("msa_path"))
    print(f"\n[targets] wrote {len(rows)} target records -> {man / 'targets.csv'}")
    print(f"[targets] {len(msa_records)} unique receptor sequences, {n_msa} with an MSA")
    failed = {k: v["status"] for k, v in msa_records.items() if not v.get("msa_path")}
    if failed:
        print(f"[targets] MSA failures (recorded, not fatal): {failed}")


if __name__ == "__main__":
    main()
