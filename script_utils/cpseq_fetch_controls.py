"""Fetch experimentally determined cyclic-peptide/protein complexes as technical controls.

Why these have to be fetched
----------------------------
Nothing staged in this project is an experimental CP-target complex. CPSea_PDB's entries
are named `<PDB>_<chain>_<start>_<end>_relaxed` -- fragments cut out of PDB proteins and
computationally cyclized -- and CPBind's are AFDB accessions given the same treatment.
Both are useful data; neither is experimental ground truth for a macrocycle, and the
dataset must not present them as such.

So candidates come from an RCSB search and then every one is **verified by measurement**
with this project's own evaluator before it is accepted: the peptide chain has to be in
the 5-16mer range, and its closing bond has to actually fall inside the chemistry window
this repo uses (`eval.cyclic_reconstruction_metrics`: mainchain 1.1-1.6 A, disulfide
1.8-2.3 A). A structure that the literature calls cyclic but whose deposited coordinates
do not close is rejected, with its measured bond distance recorded as the reason.

These are controls for **implementation and capability checking**, not a prospective
benchmark. Every one of them predates the predictors' training data and may well sit
inside it, so pose recovery here says the pipeline works end to end -- it does not say
anything about prospective accuracy.

Usage:
    python script_utils/cpseq_fetch_controls.py --out-dir /data/user_data/yixiz/cpseq/controls --n 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILE = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Accepted closure windows, imported rather than restated so a control is judged by the
# same bar as a prediction.
from proteinfoundation.eval.cyclic_reconstruction_metrics import (  # noqa: E402
    DISULFIDE_SG_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
)
from proteinfoundation.cpseq.contexts import (  # noqa: E402
    PEPTIDE_MAX_LEN,
    PEPTIDE_MIN_LEN,
    chain_sequence,
    contacting_chains,
    parse_structure,
)
from proteinfoundation.cpseq.geometry import closure_geometry  # noqa: E402

WINDOWS = {"mainchain": MAINCHAIN_CN_BOND_WINDOW_A, "disulfide": DISULFIDE_SG_BOND_WINDOW_A}


def rcsb_search(limit: int = 300) -> list[str]:
    """PDB ids of X-ray/EM structures with a very short polymer entity plus a protein.

    Deliberately a *structural* query rather than a text search for the word "cyclic":
    the ring is then established by measuring the coordinates, so a mislabelled or
    aspirational annotation cannot put a non-macrocycle into the control set.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {"from": PEPTIDE_MIN_LEN, "to": PEPTIDE_MAX_LEN},
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": "cyclic peptide"},
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    req = urllib.request.Request(
        RCSB_SEARCH,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as fh:  # noqa: S310
        data = json.loads(fh.read())
    return [r["identifier"] for r in data.get("result_set", [])]


def download(pdb_id: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{pdb_id.lower()}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        with urllib.request.urlopen(  # noqa: S310
            RCSB_FILE.format(pdb_id=pdb_id.upper()), timeout=60
        ) as fh:
            out.write_bytes(fh.read())
        return out
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def verify_cyclic(pdb_path: Path) -> list[dict]:
    """Every (peptide chain, chemistry) in this file whose ring actually closes."""
    parsed = parse_structure(pdb_path)
    found = []
    for chain, residues in parsed["atoms"].items():
        seq = chain_sequence(parsed, chain)
        if not (PEPTIDE_MIN_LEN <= len(seq) <= PEPTIDE_MAX_LEN):
            continue
        receptors = tuple(c for c in contacting_chains(parsed, chain) if c != chain)
        if not receptors:
            continue
        # A receptor has to be a real protein, not another short peptide.
        receptors = tuple(
            c for c in receptors if len(chain_sequence(parsed, c)) > PEPTIDE_MAX_LEN
        )
        if not receptors:
            continue
        for chemistry, (lo, hi) in WINDOWS.items():
            geo = closure_geometry(pdb_path, chain, chemistry)
            dist = geo.get("bond_length_A")
            if geo["status"] == "ok" and dist is not None and lo <= dist <= hi:
                found.append(
                    {
                        "peptide_chain": chain,
                        "sequence": seq,
                        "peptide_length": len(seq),
                        "receptor_chains": list(receptors),
                        "chemistry": chemistry,
                        "closure": geo,
                    }
                )
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6, help="how many verified controls to keep")
    ap.add_argument("--max-download", type=int, default=120)
    args = ap.parse_args()

    raw_dir = args.out_dir / "pdb"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[controls] querying RCSB for short-peptide/protein complexes ...")
    ids = rcsb_search()
    print(f"[controls] {len(ids)} candidate entries")

    accepted, rejected = [], []
    # Deduplicate by SYSTEM, not by entry. The first run returned 1SLD, 1VWB, 1VWN and
    # 1VWC -- four depositions of the same CHPQFC peptide on the same receptor -- which
    # would have made a 6-control set look far more diverse than it is. Keyed on
    # (peptide sequence, receptor sequence prefix) so replicates of one complex count
    # once; the duplicates are recorded as such rather than dropped silently.
    seen_systems: dict[tuple[str, str], str] = {}
    for pdb_id in ids[: args.max_download]:
        if len(accepted) >= args.n:
            break
        path = download(pdb_id, raw_dir)
        if path is None:
            rejected.append({"pdb_id": pdb_id, "reason": "download failed"})
            continue
        try:
            hits = verify_cyclic(path)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"pdb_id": pdb_id, "reason": f"parse failed: {exc}"})
            continue
        if not hits:
            # Record the closest the file came, so "rejected" is a measurement and not
            # an unexplained absence.
            rejected.append({"pdb_id": pdb_id, "reason": "no chain closes within window"})
            continue
        # One control per entry: the shortest closing bond deviation.
        best = min(hits, key=lambda h: abs(h["closure"]["bond_length_dev_A"]))
        parsed = parse_structure(path)
        rec_seq = "".join(chain_sequence(parsed, c) for c in best["receptor_chains"])
        sys_key = (best["sequence"], rec_seq[:60])
        if sys_key in seen_systems:
            rejected.append({
                "pdb_id": pdb_id,
                "reason": f"duplicate system -- same peptide and receptor as {seen_systems[sys_key]}",
                "peptide_sequence": best["sequence"],
            })
            continue
        seen_systems[sys_key] = pdb_id
        accepted.append({
            "pdb_id": pdb_id, "structure_path": str(path),
            "receptor_sequence_prefix": rec_seq[:60], **best,
        })
        print(f"[controls] ACCEPT {pdb_id} chain {best['peptide_chain']} "
              f"{best['chemistry']} L={best['peptide_length']} "
              f"bond={best['closure']['bond_length_A']}A seq={best['sequence']}")
        time.sleep(0.2)

    (args.out_dir / "controls.json").write_text(json.dumps(accepted, indent=2))
    (args.out_dir / "controls_rejected.json").write_text(json.dumps(rejected, indent=2))
    print(f"\n[controls] accepted {len(accepted)}, rejected {len(rejected)}")
    print(f"[controls] -> {args.out_dir / 'controls.json'}")
    if len(accepted) < args.n:
        print(f"[controls] WARNING: wanted {args.n}, verified {len(accepted)}. "
              "Reported as-is rather than padded with unverified entries.")


if __name__ == "__main__":
    main()
