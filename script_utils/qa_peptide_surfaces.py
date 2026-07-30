"""QA pass over a directory of peptide-surface caches: pick a stratified sample, render
each one, and check the things a human would otherwise have to eyeball.

    python script_utils/qa_peptide_surfaces.py \
        --surface-dir surfaces/cpsea_sample100 \
        --output-dir surfaces/qa --num 20

Picks `--num` caches spread across cyclization types and peptide-length terciles (falling
back to a uniform random draw when the metadata lacks those fields), runs
`scripts/visualize_peptide_surface.py` on each, and writes an `index.html` linking them
all with the per-complex numbers.

Each of the manual completion criteria is also computed, so "I looked at 20 pictures" is
backed by a number on every one of them:

* *patch is on the receptor-facing side* -> `facing_cos`: cosine between
  (patch centroid - peptide centroid) and (receptor centroid - peptide centroid). Positive
  means the patch is on the receptor's side; we flag <= 0.
* *points do not come from the receptor or an unrelated chain* -> `max_d_to_peptide_atom`:
  every retained vertex must sit on the peptide's own solvent-excluded surface, i.e. about
  one vdW radius off a peptide heavy atom. (Contamination is structurally impossible --
  the peptide mesh is generated from a PDB holding only the peptide chains -- so what this
  really tests is the chain assignment.) `frac_closer_to_peptide` is *reported but not
  failed on*: measured over the CPSea sample it is 0.918-1.000, because at a tight
  interface a receptor atom is routinely nearer to a peptide surface vertex (1.1-2.3 A)
  than the peptide atom that generated it (1.5-2.5 A). Only the median is guarded, which
  is what would collapse if the two chain sets were swapped.
* *normals point outward* -> `outward_frac`.
* *sampling covers the whole interface* -> `coverage_max` (worst retained vertex's
  distance to its nearest sample) and `sample_spread_ratio` (sample spread / patch spread);
  a collapsed sample set has a ratio well under 1.
* *re-running is identical* -> `--check-reproducibility` re-extracts each picked complex
  and compares every array bit-for-bit.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from proteinfoundation.surface.peptide_surface import (  # noqa: E402
    extract_peptide_surface,
    load_surface_cache,
    nearest_receptor_distance,
    patch_components,
    sampling_coverage,
)

logger = logging.getLogger("qa_peptide_surfaces")

# A solvent-excluded surface vertex sits roughly one vdW radius (1.5-2.0 A) off its
# nearest heavy atom. 4.0 A is generous headroom; anything past it is not on this
# peptide's surface at all.
MAX_VERTEX_TO_PEPTIDE_ATOM = 4.0


def read_atoms(pdb_path: Path, chains: set[str]) -> np.ndarray:
    opener = gzip.open if str(pdb_path).endswith(".gz") else open
    xyz = []
    with opener(str(pdb_path), "rt") as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")) and line[21] in chains:
                if line[76:78].strip() == "H":
                    continue
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def qa_metrics(surface, pdb_path: Path) -> dict:
    peptide_atoms = read_atoms(pdb_path, set(surface.metadata.get("peptide_chains", ["B"])))
    receptor_atoms = read_atoms(pdb_path, set(surface.metadata.get("receptor_chains", [])))
    intf = surface.interface_xyz.astype(np.float64)

    d_pep = nearest_receptor_distance(intf, peptide_atoms)
    d_rec = nearest_receptor_distance(intf, receptor_atoms)

    pep_centroid = peptide_atoms.mean(0)
    to_patch = intf.mean(0) - pep_centroid
    to_receptor = receptor_atoms.mean(0) - pep_centroid
    denom = np.linalg.norm(to_patch) * np.linalg.norm(to_receptor)
    facing_cos = float(to_patch @ to_receptor / denom) if denom > 0 else float("nan")

    centre = surface.full_peptide_xyz.mean(0)
    radial = surface.interface_xyz - centre
    radial = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-8)
    outward_frac = float(((radial * surface.interface_normals).sum(1) > 0).mean())

    cov_mean, cov_max = sampling_coverage(surface)
    valid = surface.sampled_valid_mask
    patch_spread = float(np.linalg.norm(intf - intf.mean(0), axis=1).mean())
    sampled = surface.sampled_xyz[valid].astype(np.float64)
    sample_spread = float(np.linalg.norm(sampled - sampled.mean(0), axis=1).mean())

    components = patch_components(surface)
    cutoff = float(surface.metadata.get("cutoff", 4.0))

    metrics = {
        "example_id": surface.metadata.get("example_id", pdb_path.stem),
        "peptide_length": surface.metadata.get("peptide_length"),
        "cyclization_type": surface.metadata.get("cyclization_type"),
        "n_full": surface.num_full,
        "n_interface": surface.num_interface,
        "retained_fraction": round(surface.num_interface / max(surface.num_full, 1), 4),
        "n_sampled": surface.num_sampled,
        "max_receptor_distance": round(float(surface.interface_receptor_distance.max()), 3),
        "max_d_to_peptide_atom": round(float(d_pep.max()), 3),
        "frac_closer_to_peptide": round(float((d_pep < d_rec).mean()), 4),
        "facing_cos": round(facing_cos, 4),
        "outward_frac": round(outward_frac, 4),
        "coverage_mean": round(cov_mean, 3),
        "coverage_max": round(cov_max, 3),
        "sample_spread_ratio": round(sample_spread / patch_spread, 3) if patch_spread else None,
        "n_components": len(components),
        "largest_component_frac": round(components[0] / max(surface.num_interface, 1), 3)
        if components
        else 0.0,
        "all_finite": bool(
            np.isfinite(surface.interface_xyz).all()
            and np.isfinite(surface.interface_normals).all()
            and np.isfinite(surface.sampled_xyz).all()
        ),
        "normals_unit": bool(
            np.allclose(np.linalg.norm(surface.interface_normals, axis=1), 1.0, atol=1e-4)
        ),
        "within_cutoff": bool((surface.interface_receptor_distance <= cutoff + 1e-5).all()),
    }

    fails = []
    if not metrics["within_cutoff"]:
        fails.append("a retained vertex exceeds the cutoff")
    if not metrics["all_finite"]:
        fails.append("non-finite values")
    if not metrics["normals_unit"]:
        fails.append("normals not unit length")
    if metrics["max_d_to_peptide_atom"] > MAX_VERTEX_TO_PEPTIDE_ATOM:
        fails.append(
            f"a retained vertex is {metrics['max_d_to_peptide_atom']:.1f} A from any peptide atom"
        )
    # Not `< 0.99`: see the module docstring. A swapped chain assignment would put this
    # well under a half, which is what the threshold is set to catch.
    if metrics["frac_closer_to_peptide"] < 0.5:
        fails.append(
            f"only {metrics['frac_closer_to_peptide']:.1%} of vertices are nearer the peptide "
            "than the receptor -- check the chain assignment"
        )
    if not (metrics["facing_cos"] > 0):
        fails.append(f"patch is not on the receptor-facing side (cos={metrics['facing_cos']})")
    if metrics["outward_frac"] < 0.75:
        fails.append(f"only {metrics['outward_frac']:.0%} of normals point outward")
    if metrics["sample_spread_ratio"] is not None and metrics["sample_spread_ratio"] < 0.8:
        fails.append(f"samples are collapsed (spread ratio {metrics['sample_spread_ratio']})")
    metrics["fails"] = fails
    return metrics


def stratified_pick(caches: list[Path], num: int, seed: int) -> list[Path]:
    """Spread the pick over cyclization types and peptide-length terciles."""
    rng = np.random.default_rng(seed)
    rows = []
    for path in caches:
        try:
            meta = load_surface_cache(path).metadata
        except Exception:
            continue
        rows.append((path, meta.get("cyclization_type", "unknown"), meta.get("peptide_length")))
    if not rows:
        return []

    lengths = [r[2] for r in rows if isinstance(r[2], int)]
    if lengths:
        lo, hi = np.percentile(lengths, [33.3, 66.7])

        def bucket(length):
            if not isinstance(length, int):
                return "len:?"
            return "len:short" if length <= lo else ("len:mid" if length <= hi else "len:long")
    else:

        def bucket(length):
            return "len:?"

    strata: dict[tuple, list[Path]] = {}
    for path, cyc, length in rows:
        strata.setdefault((cyc, bucket(length)), []).append(path)

    # Round-robin across strata so no single (type, length) bucket dominates the sample.
    for key in strata:
        order = rng.permutation(len(strata[key]))
        strata[key] = [strata[key][i] for i in order]
    keys = sorted(strata, key=lambda k: (-len(strata[k]), str(k)))
    picked: list[Path] = []
    while len(picked) < num and any(strata[k] for k in keys):
        for key in keys:
            if strata[key] and len(picked) < num:
                picked.append(strata[key].pop())
    return picked


def _index_html(records: list[dict], title: str) -> str:
    def cell(record, key, fmt="{}"):
        value = record.get(key)
        return "-" if value is None else fmt.format(value)

    rows = []
    for record in records:
        bad = bool(record["fails"])
        rows.append(
            "<tr class='{cls}'>"
            "<td><a href='{href}'>{name}</a></td><td>{cyc}</td><td>{plen}</td>"
            "<td>{nfull}</td><td>{nintf}</td><td>{frac}</td><td>{nsamp}</td>"
            "<td>{maxd}</td><td>{facing}</td><td>{outward}</td><td>{cov}</td>"
            "<td>{spread}</td><td>{comps}</td><td>{status}</td></tr>".format(
                cls="bad" if bad else "good",
                href=html.escape(record["html"]),
                name=html.escape(str(record["example_id"])),
                cyc=html.escape(str(record.get("cyclization_type"))),
                plen=cell(record, "peptide_length"),
                nfull=record["n_full"],
                nintf=record["n_interface"],
                frac=f"{record['retained_fraction']:.1%}",
                nsamp=record["n_sampled"],
                maxd=f"{record['max_receptor_distance']:.2f}",
                facing=f"{record['facing_cos']:.2f}",
                outward=f"{record['outward_frac']:.0%}",
                cov=f"{record['coverage_max']:.2f}",
                spread=cell(record, "sample_spread_ratio", "{:.2f}"),
                comps=record["n_components"],
                status="FAIL: " + "; ".join(record["fails"]) if bad else "ok",
            )
        )

    n_bad = sum(1 for r in records if r["fails"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root {{ color-scheme: light dark; }}
body {{ font: 14px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
        margin: 0; padding: 24px; background:#fbfbfd; color:#1f2328; }}
@media (prefers-color-scheme: dark) {{ body {{ background:#14161a; color:#e6e6e6; }} }}
h1 {{ font-size: 19px; margin: 0 0 4px; }}
p.sub {{ opacity:.7; margin:0 0 18px; }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 1100px; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,.25); text-align: right;
          white-space: nowrap; }}
th {{ text-align: right; font-size: 11px; letter-spacing:.05em; text-transform: uppercase;
      opacity:.65; position: sticky; top: 0; background: inherit; }}
td:first-child, th:first-child, td:last-child, th:last-child {{ text-align: left; }}
tr.bad td:last-child {{ color:#d93025; font-weight:600; }}
tr.good td:last-child {{ color:#1e8e3e; }}
a {{ color: inherit; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(records)} complexes &middot; {n_bad} flagged &middot;
click a name for the interactive 3D view.</p>
<div class="wrap"><table>
<tr><th>complex</th><th>cyc type</th><th>len</th><th>full</th><th>retained</th><th>frac</th>
<th>sampled</th><th>max d</th><th>facing cos</th><th>outward</th><th>cov max</th>
<th>spread</th><th>comps</th><th>status</th></tr>
{''.join(rows)}
</table></div></body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-reproducibility", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="metrics only, no HTML views")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    caches = sorted(args.surface_dir.glob("*.surface.npz"))
    if not caches:
        raise SystemExit(f"no caches in {args.surface_dir}")
    picked = stratified_pick(caches, args.num, args.seed)
    logger.info("picked %d/%d caches for QA", len(picked), len(caches))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for n, cache in enumerate(picked, start=1):
        surface = load_surface_cache(cache)
        pdb = Path(surface.metadata["source_pdb"])
        record = qa_metrics(surface, pdb)
        record["cache"] = str(cache)

        if args.check_reproducibility:
            again = extract_peptide_surface(
                pdb,
                receptor_chains=surface.metadata["receptor_chains"],
                peptide_chains=surface.metadata["peptide_chains"],
                cutoff=surface.metadata["cutoff"],
                num_points=surface.metadata["sample_count"],
                seed=surface.metadata["seed"],
            )
            identical = all(
                np.array_equal(getattr(again, f), getattr(surface, f))
                for f in (
                    "full_peptide_xyz",
                    "full_peptide_normals",
                    "interface_xyz",
                    "interface_normals",
                    "interface_receptor_distance",
                    "sampled_xyz",
                    "sampled_normals",
                    "sampled_receptor_distance",
                    "sampled_valid_mask",
                )
            )
            record["reproducible"] = bool(identical)
            if not identical:
                record["fails"].append("re-extraction did not reproduce the cache")

        out_html = args.output_dir / f"{record['example_id']}.html"
        record["html"] = out_html.name
        if not args.no_render:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "visualize_peptide_surface.py"),
                    "--pdb", str(pdb),
                    "--surface-cache", str(cache),
                    "--output", str(out_html),
                ],
                check=True,
                capture_output=True,
            )
        logger.info(
            "[%d/%d] %s  retained=%d (%.0f%%) facing=%.2f outward=%.0f%% cov_max=%.2f  %s",
            n, len(picked), record["example_id"], record["n_interface"],
            100 * record["retained_fraction"], record["facing_cos"],
            100 * record["outward_frac"], record["coverage_max"],
            "FAIL " + "; ".join(record["fails"]) if record["fails"] else "ok",
        )
        records.append(record)

    index = args.output_dir / "index.html"
    index.write_text(_index_html(records, f"Peptide surface QA - {args.surface_dir.name}"))
    (args.output_dir / "qa_metrics.json").write_text(json.dumps(records, indent=2))

    n_bad = sum(1 for r in records if r["fails"])
    logger.info("=" * 60)
    logger.info("%d/%d complexes pass every automated QA check", len(records) - n_bad, len(records))
    for key in ("retained_fraction", "facing_cos", "outward_frac", "coverage_max"):
        values = np.array([r[key] for r in records], dtype=float)
        logger.info("%-20s min=%.3f median=%.3f max=%.3f", key, values.min(),
                    float(np.median(values)), values.max())
    logger.info("index: %s", index)
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
