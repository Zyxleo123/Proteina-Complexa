"""Target preparation, computed once per parent and shared by every candidate and arm.

Two artefacts per parent, both cached:

  **target-only template** -- the receptor chains and nothing else. The peptide is
  absent from the file, so no arm can be handed the bound peptide coordinates it is
  supposed to predict. Residues are renumbered 1..N per chain in sequence order, which
  is what makes a pocket index mean the same thing in the template, in the YAML
  sequence, and in the evaluator.

  **receptor MSA** -- one `.a3m` per unique receptor *sequence*, keyed by the hash of
  that sequence rather than by parent id, so two parents sharing a receptor share one
  search. Candidate peptides never get an MSA: they are synthetic sequences with no
  homologs, and a search per candidate would be the repeated expensive alignment the
  task rules out. 165 candidates against 12 parents means 12 searches, not 165.

Why renumbering is safe here
----------------------------
Only residue *numbers* and chain *ids* are rewritten. Coordinates are copied verbatim,
which `build_lnr_metadata.py` already established as the convention for this project --
the bound receptor pose has to stay bit-for-bit what the crystal said, or the template
stops being the structure it claims to be.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from proteinfoundation.cpseq.contexts import ParentContext, parse_structure

# Chain ids handed to the predictors. The peptide takes B, matching this repo's CPSea
# convention (A = receptor, B = binder), so the evaluator needs no per-predictor map.
PEPTIDE_CHAIN = "B"
_RECEPTOR_LETTERS = "ACDEFGHIJKLMNOPQRSTUVWXYZ"

# ColabFold's public MSA server, which is what boltz's --use_msa_server targets.
DEFAULT_MSA_SERVER_URL = "https://api.colabfold.com"


def receptor_chain_map(ctx: ParentContext) -> dict[str, str]:
    """{original chain id -> predictor chain id}, deterministic in sorted order."""
    return {
        orig: _RECEPTOR_LETTERS[n] for n, orig in enumerate(sorted(ctx.receptor_chains))
    }


def write_target_only_template(ctx: ParentContext, out_path: Path) -> dict:
    """Write the receptor-only template PDB. Returns a record of what went in.

    Raises if the peptide chain would end up in the file -- that check is cheap and the
    failure it guards against (templating the answer) is not recoverable after the fact.
    """
    parsed = parse_structure(Path(ctx.structure_path))
    chain_map = receptor_chain_map(ctx)
    if PEPTIDE_CHAIN in chain_map.values() and ctx.peptide_chain in chain_map:
        raise AssertionError("peptide chain collided with a receptor chain id")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    serial = 0
    n_res = {}
    with open(out_path, "w") as fh:
        fh.write(f"REMARK   1 TARGET-ONLY TEMPLATE FOR {ctx.parent_id}\n")
        fh.write(f"REMARK   1 SOURCE {Path(ctx.structure_path).name}\n")
        fh.write(
            f"REMARK   1 PEPTIDE CHAIN {ctx.peptide_chain} DELIBERATELY EXCLUDED\n"
        )
        for orig in sorted(ctx.receptor_chains):
            if orig == ctx.peptide_chain:
                raise AssertionError(
                    f"{ctx.parent_id}: peptide chain {orig} is in the receptor set"
                )
            new_chain = chain_map[orig]
            residues = parsed["atoms"][orig]
            resnames = parsed["resnames"][orig]
            for new_resseq, key in enumerate(residues, start=1):
                resname = resnames[key]
                for atom, (x, y, z) in residues[key].items():
                    serial += 1
                    element = atom[0]
                    fh.write(
                        f"ATOM  {serial:5d} {atom:<4s}{resname:>4s}{new_chain:>2s}"
                        f"{new_resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                        f"  1.00  0.00          {element:>2s}\n"
                    )
            fh.write(f"TER   {serial + 1:5d}      {resname:>3s}{new_chain:>2s}{new_resseq:4d}\n")
            serial += 1
            n_res[new_chain] = new_resseq
        fh.write("END\n")

    # The guard that matters: read the file back and confirm the peptide is not in it.
    written_chains = {
        line[21] for line in out_path.read_text().splitlines() if line.startswith("ATOM")
    }
    if ctx.peptide_chain in written_chains and ctx.peptide_chain not in chain_map.values():
        raise AssertionError(
            f"{ctx.parent_id}: peptide chain {ctx.peptide_chain} leaked into the template"
        )

    return {
        "template_path": str(out_path),
        "chain_map": chain_map,
        "n_residues_per_chain": n_res,
        "n_atoms": serial,
        "peptide_chain_excluded": ctx.peptide_chain,
    }


# The CONDITIONING pocket is deliberately much smaller than the descriptive one.
#
# Boltz's `pocket` constraint applies `max_distance` between the binder and **each** of
# the listed contacts, so handing it the full 8 A shell -- 27 to 48 residues on these
# parents -- would demand that every one of them sit within 6 A of a 5-16mer peptide at
# once. No pose satisfies that, and an unsatisfiable constraint does not localise a
# prediction, it distorts it. HighFold3 and the rest get the same set for the same
# reason.
#
# So the conditioning set is the K receptor residues in genuine contact with the parent's
# bound peptide, nearest first. Being real contacts in the crystal, they are jointly
# satisfiable by construction, and K is small enough that the constraint says "this site"
# rather than "this exact pose".
POCKET_CONDITIONING_MAX_CONTACTS = 6
POCKET_CONDITIONING_MAX_DIST_A = 5.0


def pocket_contacts_for(
    ctx: ParentContext,
    *,
    max_contacts: int = POCKET_CONDITIONING_MAX_CONTACTS,
    max_dist_A: float = POCKET_CONDITIONING_MAX_DIST_A,
) -> list[tuple[str, int]]:
    """Conditioning pocket as [(original chain id, 1-based index in that chain's sequence)].

    Indices are 0-based positions in the chain's residue list plus one, which is the same
    ordering the template renumbering and the predictors' sequences use, so one index
    means one residue in all three.

    Residues are ranked by min heavy-atom distance to the parent's bound peptide, filtered
    at `max_dist_A`, and truncated to `max_contacts`. Ties break on (chain, index) so the
    set is deterministic. Defined once per parent and reused unchanged by every mutant of
    that parent -- it localises the site and restrains no individual peptide residue.
    """
    ranked: list[tuple[float, str, int]] = []
    for chain in sorted(ctx.pocket_residues):
        locals_ = ctx.pocket_residues[chain]
        dists = ctx.pocket_min_dist.get(chain, ())
        if len(dists) != len(locals_):
            # Without distances there is no defensible way to pick a small subset, and
            # passing the full shell would be the unsatisfiable constraint above.
            raise ValueError(
                f"{ctx.parent_id}: pocket_min_dist missing or misaligned for chain "
                f"{chain} ({len(dists)} distances for {len(locals_)} residues)"
            )
        for local, dist in zip(locals_, dists):
            if dist <= max_dist_A:
                ranked.append((dist, chain, local + 1))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]))
    return [(chain, idx) for _, chain, idx in ranked[:max_contacts]]


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def receptor_msa_path(cache_dir: Path, sequence: str) -> Path:
    """Where this receptor sequence's MSA lives. Keyed by sequence, not by parent."""
    return Path(cache_dir) / f"{sequence_key(sequence)}.a3m"


def fetch_receptor_msa(
    sequence: str,
    cache_dir: Path,
    python_exec: str,
    *,
    msa_server_url: str = DEFAULT_MSA_SERVER_URL,
    timeout_s: int = 3600,
) -> dict:
    """Fetch and cache one receptor MSA. A cache hit does no network work at all.

    Shells out to boltz's own `run_mmseqs2`, in the boltz interpreter, so the alignment
    is produced by exactly the code path `--use_msa_server` would have used -- the cache
    changes when the search happens, not what it computes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = receptor_msa_path(cache_dir, sequence)
    meta = out.with_suffix(".json")
    if out.exists() and out.stat().st_size > 0:
        return {
            "status": "cache_hit",
            "msa_path": str(out),
            "n_sequences": sum(1 for ln in out.read_text().splitlines() if ln.startswith(">")),
        }

    script = (
        "import sys, json, tempfile, pathlib\n"
        "from boltz.data.msa.mmseqs2 import run_mmseqs2\n"
        "seq = sys.argv[1]; out = pathlib.Path(sys.argv[2]); url = sys.argv[3]\n"
        "with tempfile.TemporaryDirectory() as td:\n"
        "    res = run_mmseqs2([seq], td, use_env=True, use_pairing=False, host_url=url)\n"
        "out.write_text(res[0])\n"
        "print(json.dumps({'n_sequences': res[0].count('>')}))\n"
    )
    import time

    t0 = time.time()
    proc = subprocess.run(
        [python_exec, "-I", "-c", script, sequence, str(out), msa_server_url],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0 or not out.exists():
        return {
            "status": "msa_failed",
            "msa_path": None,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "runtime_s": round(time.time() - t0, 2),
        }
    record = {
        "status": "fetched",
        "msa_path": str(out),
        "sequence_key": sequence_key(sequence),
        "sequence_length": len(sequence),
        "msa_server_url": msa_server_url,
        "runtime_s": round(time.time() - t0, 2),
        "n_sequences": sum(1 for ln in out.read_text().splitlines() if ln.startswith(">")),
    }
    meta.write_text(json.dumps(record, indent=2))
    return record
