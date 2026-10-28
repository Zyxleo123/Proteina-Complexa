"""Prepare and cache the shared target conditioning for every selected parent.

Per parent: one target-only template PDB, and one receptor MSA per unique receptor
sequence. Both are cached, so re-running is cheap and a candidate never triggers an
alignment search -- the MSA is keyed by receptor sequence hash, not by candidate.

Writes `manifest/targets.csv` and `manifest/msa_cache.json` under `--out-dir`.

Usage:
    # LNR (the original tranche)
    python script_utils/cpseq_prepare_targets.py \
        --out-dir /data/user_data/yixiz/cpseq/tranche1 \
        --boltz-python /data/user_data/yixiz/envs/boltz2/.venv/bin/python

    # PepBench valid split -- same parser, the layout is identical
    python script_utils/cpseq_prepare_targets.py \
        --out-dir /data/user_data/yixiz/cpseq/pepbench_valid \
        --lnr-root /data/user_data/yixiz/LPData/PepBench/train_valid \
        --list-file valid.txt \
        --boltz-python /data/user_data/yixiz/envs/boltz2/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.contexts import load_lnr_contexts  # noqa: E402
from proteinfoundation.cpseq.contexts import (  # noqa: E402
    PEPTIDE_MAX_LEN,
    PEPTIDE_MIN_LEN,
    POCKET_RADIUS_A,
)
from proteinfoundation.cpseq.targets import (  # noqa: E402
    POCKET_CONDITIONING_MAX_CONTACTS,
    POCKET_CONDITIONING_MAX_DIST_A,
    fetch_receptor_msa,
    pocket_contacts_for,
    receptor_chain_map,
    receptor_msa_path,
    sequence_key,
    write_target_only_template,
    write_target_template_cif,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lnr-root", type=Path, default=Path("/data/user_data/yixiz/LNR"),
                    help="dataset root: a dir with the list file and pdbs/. Named for LNR "
                         "because that was the first corpus; PepBench has the identical "
                         "layout (id, receptor_chain, peptide_chain + pdbs/<id>.pdb) so it "
                         "loads through the same parser.")
    ap.add_argument("--list-file", default="test.txt",
                    help="split list inside the dataset root (LNR: test.txt; "
                         "PepBench: valid.txt / train.txt / all.txt)")
    ap.add_argument("--source", default="LNR",
                    help="corpus label recorded on every parent (LNR, PepBench, ...). "
                         "Also supplies the parent_id prefix, lowercased.")
    ap.add_argument("--min-len", type=int, default=PEPTIDE_MIN_LEN,
                    help=f"peptide length window, inclusive (default {PEPTIDE_MIN_LEN}). "
                         "The default is the project standard; changing it makes results "
                         "incomparable with tranches prepared under the old window, so it "
                         "is recorded in the manifest.")
    ap.add_argument("--max-len", type=int, default=PEPTIDE_MAX_LEN,
                    help=f"peptide length window, inclusive (default {PEPTIDE_MAX_LEN})")
    ap.add_argument("--boltz-python", required=True)
    ap.add_argument("--skip-msa", action="store_true", help="templates only")
    ap.add_argument("--msa-delay-s", type=float, default=2.0,
                    help="pause between MSA fetches that actually hit the network. "
                         "api.colabfold.com is a free shared service and this corpus has "
                         "~89%% unique receptors, so the fetch loop must not run flat out. "
                         "Cache hits are not delayed.")
    args = ap.parse_args()

    man = args.out_dir / "manifest"
    man.mkdir(parents=True, exist_ok=True)
    tmpl_dir = args.out_dir / "targets"
    msa_dir = args.out_dir / "msa_cache"

    contexts, rejected = load_lnr_contexts(
        args.lnr_root, args.list_file,
        min_len=args.min_len, max_len=args.max_len,
        source=args.source, id_prefix=args.source.lower(),
    )

    # `parents.csv` is the LNR tranche's hand-curated selection. A corpus prepared from
    # scratch (PepBench) has no such file, and absent it the split list IS the selection
    # -- inventing one here would silently drop targets the caller asked for.
    pcsv = man / "parents.csv"
    if pcsv.exists():
        parents = pd.read_csv(pcsv)
        selected = set(parents.loc[parents["selected_for_tranche"], "parent_id"])
        contexts = [c for c in contexts if c.parent_id in selected]
        print(f"[targets] {len(selected)} parents selected by {pcsv.name}")
    else:
        print(f"[targets] no parents.csv; taking all {len(contexts)} parents from "
              f"{args.list_file}")
    if rejected:
        print(f"[targets] {len(rejected)} entries rejected at parse "
              f"(length window / missing chain); see manifest/rejected.csv")
        pd.DataFrame(rejected).to_csv(man / "rejected.csv", index=False)

    rows, msa_records = [], {}
    for ctx in contexts:
        rec = write_target_only_template(ctx, tmpl_dir / f"{ctx.parent_id}_target.pdb")
        cif = write_target_template_cif(
            Path(rec["template_path"]),
            tmpl_dir / f"{ctx.parent_id}_target.cif",
            args.boltz_python,
        )
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
            # Checkpoint after every fetch, not at the end: a run interrupted at target
            # 90 of 114 must not re-query the server for the 89 it already has. The .a3m
            # files are the real cache (fetch_receptor_msa short-circuits on them), and
            # this keeps the manifest consistent with them.
            (man / "msa_cache.json").write_text(json.dumps(msa_records, indent=2))
            if out["status"] == "fetched" and args.msa_delay_s > 0:
                time.sleep(args.msa_delay_s)

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
                "target_template_cif": cif["cif_path"],
                "target_template_cif_entity_lengths": json.dumps(cif["written"]),
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

    (man / "prepare_params.json").write_text(json.dumps({
        "source": args.source, "dataset_root": str(args.lnr_root),
        "list_file": args.list_file,
        "peptide_length_window": [args.min_len, args.max_len],
        "n_parsed": len(contexts), "n_rejected": len(rejected),
        "msa_delay_s": args.msa_delay_s, "skip_msa": bool(args.skip_msa),
    }, indent=2))
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
