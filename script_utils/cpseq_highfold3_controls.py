#!/usr/bin/env python3
"""Run HighFold3 on the six crystal macrocycle controls, and measure pose recovery.

This is the GATE a second predictor must pass before its opinion of a designed sequence
means anything. The precedent is afcycdesign: it produced confident output on every
control and sat 34.67 A from the deposited peptide, so its disagreement with Boltz-2
(32.9 A median) measured its own docking failure rather than any real ambiguity. Boltz-2
passes the same gate at 1.42 A mainchain / 1.18 A disulfide, peptide never templated.

So: same six complexes, same measure (receptor-aligned peptide CA RMSD against the
deposited structure), peptide never templated. Nothing here touches a designed sequence.

The peptide is written as chain A -- FIRST -- because HighFold3's ring is applied to the
first chain regardless of what `--head_to_tail` is given (probed; see the adapter). With
the receptor first the cyclic encoding lands on the TARGET and the run answers a
different question, confidently.

Usage:
    python script_utils/cpseq_highfold3_controls.py \
        --dataset-dir /data/user_data/yixiz/cpseq/controls_tranche \
        --out-dir     /data/user_data/yixiz/cpseq/hf3_controls \
        --model-dir   /data/user_data/yixiz/models/af3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.adapters import highfold3 as hf3  # noqa: E402
from proteinfoundation.cpseq.contexts import chain_sequence, parse_structure  # noqa: E402
from proteinfoundation.cpseq.geometry import read_chain  # noqa: E402

HF3_PY = "/data/user_data/yixiz/envs/highfold3/.venv/bin/python"
HF3_RUN = "/data/user_data/yixiz/upstream/highfold3/run_alphafold.py"


def ca_from_cif(path: Path, chains: list[str]) -> np.ndarray:
    """CA coordinates per chain, in residue order, from an mmCIF `_atom_site` loop.

    AF3 writes mmCIF, and this must NOT go through the PDB reader. An mmCIF atom row
    also begins with "ATOM", so a fixed-column PDB parser does not reject the file --
    it reads whitespace-delimited fields at PDB column offsets and yields nonsense,
    raising only when a slice happens not to parse as an int. That is a silent-wrong
    failure mode, so the format is dispatched on explicitly rather than sniffed.

    Chains are matched on `auth_asym_id`, which carries the ids set in the input JSON.
    """
    text = Path(path).read_text().splitlines()
    cols: dict[str, int] = {}
    rows: list[list[str]] = []
    in_loop = False
    for line in text:
        t = line.strip()
        if t.startswith("_atom_site."):
            cols[t.split(".", 1)[1].split()[0]] = len(cols)
            in_loop = True
            continue
        if in_loop:
            if not t or t.startswith(("#", "loop_", "_", "data_")):
                if rows:
                    break
                in_loop = bool(cols)
                continue
            f = t.split()
            if len(f) >= len(cols):
                rows.append(f)
    need = ("group_PDB", "label_atom_id", "label_alt_id", "Cartn_x", "Cartn_y",
            "Cartn_z", "auth_asym_id", "auth_seq_id", "pdbx_PDB_model_num")
    missing = [k for k in need if k not in cols]
    if missing:
        raise ValueError(f"{path}: _atom_site lacks {missing}")

    per_chain: dict[str, list[tuple[int, np.ndarray]]] = {c: [] for c in chains}
    for f in rows:
        if f[cols["group_PDB"]] != "ATOM":
            continue
        if f[cols["pdbx_PDB_model_num"]] != "1":
            continue
        if f[cols["label_atom_id"]].strip('"') != "CA":
            continue
        if f[cols["label_alt_id"]] not in (".", "?", "A"):
            continue
        ch = f[cols["auth_asym_id"]]
        if ch not in per_chain:
            continue
        per_chain[ch].append((
            int(f[cols["auth_seq_id"]]),
            np.array([float(f[cols["Cartn_x"]]), float(f[cols["Cartn_y"]]),
                      float(f[cols["Cartn_z"]])], dtype=np.float64),
        ))
    out = []
    for ch in chains:
        for _, xyz in sorted(per_chain[ch], key=lambda kv: kv[0]):
            out.append(xyz)
    return np.asarray(out, dtype=np.float64)


def ca(path: Path, chains: list[str], *, reference: bool) -> np.ndarray:
    """CA coordinates in residue order. `reference` uses the template's own parser."""
    out = []
    if reference:
        parsed = parse_structure(Path(path))
        for ch in chains:
            for _, atoms in parsed["atoms"].get(ch, {}).items():
                if "CA" in atoms:
                    out.append(np.asarray(atoms["CA"], dtype=np.float64))
        return np.asarray(out, dtype=np.float64)
    if str(path).endswith((".cif", ".mmcif")):
        return ca_from_cif(Path(path), chains)
    for ch in chains:
        for _, atoms in read_chain(path, ch):
            if "CA" in atoms:
                out.append(atoms["CA"])
    return np.asarray(out, dtype=np.float64)


def receptor_aligned_rmsd(pred: Path, ref: Path, *, pred_rec, pred_pep, ref_rec, ref_pep) -> dict:
    """Kabsch on the receptor only; the transform is then applied to the peptide."""
    ra, rb = ca(pred, pred_rec, reference=False), ca(ref, ref_rec, reference=True)
    pa, pb = ca(pred, [pred_pep], reference=False), ca(ref, [ref_pep], reference=True)
    if ra.size == 0 or rb.size == 0:
        return {"status": "no_receptor_ca"}
    if ra.shape != rb.shape:
        return {"status": "receptor_length_mismatch", "n_pred": len(ra), "n_ref": len(rb)}
    if pa.shape != pb.shape or pa.size == 0:
        return {"status": "peptide_length_mismatch", "n_pred": len(pa), "n_ref": len(pb)}
    ca_, cb_ = ra.mean(0), rb.mean(0)
    u, _, vt = np.linalg.svd((ra - ca_).T @ (rb - cb_))
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    rb_m = (rb - cb_) @ rot.T + ca_
    pb_m = (pb - cb_) @ rot.T + ca_
    return {
        "status": "ok",
        "peptide_ca_rmsd_receptor_aligned_A": round(
            float(np.sqrt(((pa - pb_m) ** 2).sum(1).mean())), 3),
        "receptor_ca_rmsd_after_alignment_A": round(
            float(np.sqrt(((ra - rb_m) ** 2).sum(1).mean())), 3),
        "n_receptor_ca": int(len(ra)), "n_peptide_ca": int(len(pa)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--num-seeds", type=int, default=1)
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument("--parents", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-predict even when a structure is already on disk")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.dataset_dir / "manifest" / "targets.csv")
    chem = {}
    for rec in (args.dataset_dir / "predictions").rglob("record.json"):
        r = json.loads(rec.read_text())
        if r.get("parent_id") and r.get("chemistry"):
            chem[r["parent_id"]] = r["chemistry"]

    wanted = {p.strip() for p in args.parents.split(",") if p.strip()}
    cfg = hf3.HighFold3Config(num_seeds=args.num_seeds, num_recycles=args.num_recycles)
    rows = []

    for t in targets.to_dict("records"):
        pid = str(t["parent_id"])
        if wanted and pid not in wanted:
            continue
        chemistry = chem.get(pid)
        if chemistry is None:
            print(f"[hf3] {pid}: no chemistry recorded; skipped")
            continue

        cmap = json.loads(t["chain_map"])                 # original -> predictor id
        rec_seqs, msas = {}, {}
        parsed = parse_structure(Path(t["reference_complex_path"]))
        for orig in cmap:
            rec_seqs[orig] = chain_sequence(parsed, orig)
        mp = json.loads(t["receptor_msa_paths"])
        for orig in cmap:
            p = mp.get(cmap[orig]) or mp.get(orig)
            if p:
                msas[orig] = p

        # An a3m's first record is its query. If that disagrees with the sequence being
        # folded, AF3 is handed an alignment for a different protein and will still run.
        for orig, a3m in msas.items():
            lines = Path(a3m).read_text().splitlines()
            query = next((ln.strip() for ln in lines[1:] if ln and not ln.startswith(">")), "")
            if query and query != rec_seqs[orig]:
                print(f"[hf3] {pid}: MSA query != receptor chain {orig} "
                      f"({len(query)} vs {len(rec_seqs[orig])} aa); refusing")
                rows.append({"parent_id": pid, "status": "msa_sequence_mismatch",
                             "msa_query_len": len(query), "receptor_len": len(rec_seqs[orig])})
                rec_seqs = None
                break
        if rec_seqs is None:
            continue
        if any(not v for v in rec_seqs.values()):
            print(f"[hf3] {pid}: empty receptor sequence; refusing")
            rows.append({"parent_id": pid, "status": "empty_receptor_sequence"})
            continue

        payload = hf3.build_input_json(
            name=pid, peptide_sequence=str(t["reference_sequence"]),
            receptor_sequences=rec_seqs, receptor_msas=msas,
            chemistry=chemistry, config=cfg,
        )
        h = hf3.input_hash(payload, cfg)
        wdir = args.out_dir / pid / h
        wdir.mkdir(parents=True, exist_ok=True)
        jp = wdir / "input.json"
        jp.write_text(json.dumps({k: v for k, v in payload.items()
                                  if not k.startswith("_cpseq_")}, indent=2))
        (wdir / "cpseq_meta.json").write_text(json.dumps(
            {k: v for k, v in payload.items() if k.startswith("_cpseq_")}
            | {"chemistry": chemistry, "input_hash": h}, indent=2))

        cmd = hf3.build_command(
            json_path=jp, out_dir=wdir / "hf3", model_dir=args.model_dir,
            chemistry=chemistry, peptide_length=int(t["peptide_length"]),
            python_exec=HF3_PY, run_script=HF3_RUN,
        )
        # MSAs are supplied, so the genetic-search pipeline must not run -- AF3's
        # databases are deliberately not fetched (1 TB against the quota).
        cmd.append("--norun_data_pipeline")

        print(f"[hf3] {pid} ({chemistry}, L={t['peptide_length']}) -> {wdir}")
        if args.dry_run:
            print("      " + " ".join(cmd))
            continue

        # Resumable: a prediction already on disk is reused. preempt REQUEUEs, and each
        # control costs minutes of GPU; the output dir is keyed by input hash, so a
        # structure sitting there was produced by this exact input and config.
        existing = (sorted((wdir / "hf3").rglob("*_model.cif"))
                    or sorted((wdir / "hf3").rglob("model.cif")))
        if existing and not args.force:
            print(f"      cached: {existing[0].name}")
            proc, rt = None, 0.0
        else:
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            rt = round(time.time() - t0, 1)
        row = {"parent_id": pid, "pdb_id": t["pdb_id"], "chemistry": chemistry,
               "peptide_length": int(t["peptide_length"]), "input_hash": h,
               "runtime_s": rt, "returncode": 0 if proc is None else proc.returncode,
               "from_cache": proc is None}
        if proc is not None and proc.returncode != 0:
            row["status"] = "predictor_error"
            row["stderr_tail"] = proc.stderr[-2500:]
            print(f"      FAILED rc={proc.returncode} in {rt}s")
            rows.append(row)
            (wdir / "record.json").write_text(json.dumps(row, indent=2))
            continue

        # AF3 writes <name>_model.cif for the top-ranked sample plus per-sample
        # seed-*/model.cif; take the ranked one when present.
        cif = (sorted((wdir / "hf3").rglob("*_model.cif"))
               or sorted((wdir / "hf3").rglob("model.cif"))
               or sorted((wdir / "hf3").rglob("*.pdb")))
        if not cif:
            row["status"] = "no_structure"
            print(f"      no structure written ({rt}s)")
        else:
            row["status"] = "ok"
            row["structure"] = str(cif[0])
            # Chain ids must come from the HF3 payload, NOT from the Boltz chain_map.
            # HighFold3 inputs put the PEPTIDE at chain A and receptors from B onward,
            # so the Boltz map (original -> boltz id) names chain A for the receptor and
            # would superpose the receptor onto the peptide. `_cpseq_chain_map` is
            # predictor-id -> original, written by the adapter for exactly this reason.
            hf3_map = payload["_cpseq_chain_map"]
            pred_rec = [cid for cid in hf3_map if cid != hf3.PEPTIDE_CHAIN]
            ref_rec = [hf3_map[cid] for cid in pred_rec]
            try:
                m = receptor_aligned_rmsd(
                    Path(cif[0]), Path(t["reference_complex_path"]),
                    pred_rec=pred_rec, pred_pep=hf3.PEPTIDE_CHAIN,
                    ref_rec=ref_rec, ref_pep=str(t["reference_peptide_chain"]),
                )
            except Exception as exc:  # noqa: BLE001
                # One unmeasurable control must not discard the other five predictions,
                # which cost minutes of GPU each and are already on disk.
                m = {"status": f"comparison_failed: {type(exc).__name__}: {exc}"}
            row |= {f"recovery_{k}": v for k, v in m.items()}
            print(f"      ok in {rt}s  peptide CA RMSD "
                  f"{m.get('peptide_ca_rmsd_receptor_aligned_A')} A "
                  f"(receptor fit {m.get('receptor_ca_rmsd_after_alignment_A')})")
        rows.append(row)
        (wdir / "record.json").write_text(json.dumps(row, indent=2))

    if rows and not args.dry_run:
        df = pd.DataFrame(rows)
        (args.out_dir / "controls.csv").parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_dir / "controls.csv", index=False)
        print(f"\n[hf3] {len(df)} controls -> {args.out_dir / 'controls.csv'}")
        ok = df[df.status == "ok"] if "status" in df else df.iloc[0:0]
        col = "recovery_peptide_ca_rmsd_receptor_aligned_A"
        if len(ok) and col in ok:
            for c, g in ok.groupby("chemistry"):
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                if len(v):
                    print(f"  {c:10s} n={len(v)} median {v.median():.2f} A  best {v.min():.2f} A")
            print("\n  Boltz-2 on the same controls: mainchain 1.42 A median (best 0.40), "
                  "disulfide 1.18 A; afcycdesign 34.67 A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
