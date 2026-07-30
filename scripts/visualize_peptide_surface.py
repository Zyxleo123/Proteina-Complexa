"""Render one peptide-surface cache as a self-contained interactive HTML inspection page.

    python scripts/visualize_peptide_surface.py \
        --pdb complex.pdb \
        --surface-cache surfaces/cpsea/<example_id>.surface.npz \
        --output inspection.html

The page shows, in the source PDB's coordinate frame:

* receptor atoms in gray,
* the native peptide's atoms in green,
* the full peptide molecular surface as a transparent mesh,
* the retained interface vertices coloured by nearest-receptor distance,
* the sampled points as larger markers,
* a subsample of outward normals as cones,

with the QA diagnostics panel (counts, retained fraction, distance quantiles, sampling
coverage, warnings) beside the viewer.

Plotly is used rather than py3Dmol because ``include_plotlyjs='inline'`` produces a file
that opens with no network access -- these get reviewed on cluster nodes and copied around,
and a CDN-dependent page renders blank there. The surface *faces* are not in the cache (the
cache format is fixed and stores point clouds), so the mesh is regenerated from the PDB
through the extractor's own PyMOL path -- same recipe, same frame. If PyMOL is unavailable
the page degrades to a point-cloud rendering of the full surface and says so.
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from proteinfoundation.surface.peptide_surface import (  # noqa: E402
    PeptideSurface,
    load_surface_cache,
    nearest_receptor_distance,
    patch_components,
    sampling_coverage,
)

logger = logging.getLogger("visualize_peptide_surface")

# A retained patch below either of these is worth a human look: CPSea peptides are 5-16
# residues, so a genuine interface is hundreds of vertices and tens of percent of the
# peptide surface. Well under that usually means a glancing pose or a chain mix-up.
MIN_REASONABLE_VERTICES = 50
MIN_REASONABLE_FRACTION = 0.05
# A second component this large relative to the biggest one is a real second contact
# patch (bidentate binding), not mesh speckle.
COMPONENT_REPORT_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Structure reading
# ---------------------------------------------------------------------------


def read_atoms(pdb_path: Path, chains: set[str]) -> tuple[np.ndarray, list[str]]:
    """Read ATOM/HETATM coordinates and element symbols for the given chains."""
    import gzip

    opener = gzip.open if str(pdb_path).endswith(".gz") else open
    xyz, elements = [], []
    with opener(str(pdb_path), "rt") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21] not in chains:
                continue
            try:
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            elements.append(line[76:78].strip() or line[12:16].strip()[:1])
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3), elements


def peptide_mesh_faces(
    pdb_path: Path, peptide_chains: list[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Regenerate the peptide surface mesh; return (faces, vertices) or None if unavailable.

    Vertices are deliberately discarded: the cache's ``full_peptide_xyz`` is the ground
    truth for what was extracted, and trimesh's vertex merge is deterministic, so the
    regenerated face indices address exactly the cached vertex array. The shapes are
    checked before use so a mismatch degrades to points instead of drawing nonsense.
    """
    try:
        from proteinfoundation.surface.peptide_surface import (
            generate_surface_obj,
            split_chains,
        )
        import trimesh
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("cannot regenerate mesh faces (%s); falling back to points", exc)
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="pepsurf_viz_") as tmp:
            tmp = Path(tmp)
            split_chains(pdb_path, peptide_chains, tmp / "peptide.pdb")
            generate_surface_obj(tmp / "peptide.pdb", tmp / "peptide.obj")
            mesh = trimesh.load(str(tmp / "peptide.obj"), force="mesh")
            return np.asarray(mesh.faces, dtype=np.int64), np.asarray(mesh.vertices)
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("mesh regeneration failed (%s); falling back to points", exc)
        return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def compute_diagnostics(surface: PeptideSurface) -> dict:
    dist = surface.interface_receptor_distance
    n_full = surface.num_full
    n_intf = surface.num_interface
    fraction = n_intf / n_full if n_full else 0.0
    cov_mean, cov_max = sampling_coverage(surface)
    components = patch_components(surface)

    quantiles = {}
    if dist.size:
        q = np.percentile(dist.astype(np.float64), [0, 25, 50, 75, 90, 100])
        quantiles = {
            "min": float(q[0]),
            "p25": float(q[1]),
            "median": float(q[2]),
            "p75": float(q[3]),
            "p90": float(q[4]),
            "max": float(q[5]),
        }

    warnings: list[str] = []
    cutoff = float(surface.metadata.get("cutoff", float("nan")))
    if n_intf == 0:
        warnings.append("EMPTY interface: no peptide vertex within the cutoff of the receptor.")
    else:
        if n_intf < MIN_REASONABLE_VERTICES:
            warnings.append(
                f"SMALL patch: only {n_intf} retained vertices "
                f"(< {MIN_REASONABLE_VERTICES}); check the chain assignment and the pose."
            )
        if fraction < MIN_REASONABLE_FRACTION:
            warnings.append(
                f"SMALL patch: {fraction:.1%} of the peptide surface retained "
                f"(< {MIN_REASONABLE_FRACTION:.0%}); likely a glancing contact."
            )
        if len(components) > 1:
            big = components[0]
            secondary = [c for c in components[1:] if c >= COMPONENT_REPORT_FRACTION * big]
            if secondary:
                warnings.append(
                    f"DISCONNECTED patch: {len(components)} components "
                    f"(sizes {components[:5]}); the peptide may straddle two receptor lobes."
                )
            elif len(components) > 4:
                warnings.append(
                    f"SPECKLED patch: {len(components)} components, "
                    f"{len(components) - 1} of them tiny; the mesh may be fragmenting."
                )
    if np.isfinite(dist).all() is np.False_ or not np.isfinite(dist).all():
        warnings.append("NON-FINITE distances present in the cache.")
    if surface.num_sampled < surface.sampled_valid_mask.size:
        warnings.append(
            f"PADDED: only {surface.num_sampled}/{surface.sampled_valid_mask.size} "
            "sampled points are real; the rest are zero-padding."
        )
    norms = np.linalg.norm(surface.interface_normals, axis=1)
    if norms.size and not np.allclose(norms, 1.0, atol=1e-3):
        warnings.append("NORMALS are not unit length in the retained set.")

    return {
        "cutoff": cutoff,
        "n_full": n_full,
        "n_interface": n_intf,
        "retained_fraction": fraction,
        "n_sampled_valid": surface.num_sampled,
        "n_sampled_slots": int(surface.sampled_valid_mask.size),
        "distance_quantiles": quantiles,
        "coverage_mean": cov_mean,
        "coverage_max": cov_max,
        "components": components,
        "warnings": warnings,
    }


def outwardness(surface: PeptideSurface) -> float:
    """Fraction of retained normals pointing away from the peptide surface centroid.

    A crude but effective sanity check: a correctly oriented closed surface has almost
    every normal on the far side of the centroid from itself. A value near 0.5 means the
    orientation is scrambled; near 0 means it is inverted.
    """
    xyz, normals = surface.interface_xyz, surface.interface_normals
    if xyz.shape[0] == 0:
        return float("nan")
    centre = surface.full_peptide_xyz.mean(axis=0)
    radial = xyz - centre
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return float((((radial / norm) * normals).sum(axis=1) > 0).mean())


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(
    surface: PeptideSurface,
    receptor_xyz: np.ndarray,
    peptide_atom_xyz: np.ndarray,
    faces: np.ndarray | None,
    normal_stride: int,
    normal_length: float,
    receptor_shell: float,
):
    import plotly.graph_objects as go

    traces = []

    # Receptor atoms, thinned to a shell around the peptide: drawing 10k+ spheres both
    # kills the browser and hides the interface behind the bulk of the protein.
    if receptor_xyz.size:
        if receptor_shell > 0 and surface.full_peptide_xyz.size:
            d = nearest_receptor_distance(receptor_xyz, surface.full_peptide_xyz)
            shown = receptor_xyz[d <= receptor_shell]
        else:
            shown = receptor_xyz
        traces.append(
            go.Scatter3d(
                x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                mode="markers",
                marker=dict(size=2.2, color="#9aa0a6", opacity=0.55),
                name=f"receptor atoms (within {receptor_shell:g} A)",
                hoverinfo="skip",
            )
        )

    if peptide_atom_xyz.size:
        traces.append(
            go.Scatter3d(
                x=peptide_atom_xyz[:, 0], y=peptide_atom_xyz[:, 1], z=peptide_atom_xyz[:, 2],
                mode="markers",
                marker=dict(size=3.6, color="#1e8e3e"),
                name="native peptide atoms",
                hoverinfo="skip",
            )
        )

    full = surface.full_peptide_xyz
    if faces is not None:
        traces.append(
            go.Mesh3d(
                x=full[:, 0], y=full[:, 1], z=full[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#c7d2fe", opacity=0.25, flatshading=False,
                name="full peptide surface", showlegend=True, hoverinfo="skip",
            )
        )
    else:
        traces.append(
            go.Scatter3d(
                x=full[:, 0], y=full[:, 1], z=full[:, 2],
                mode="markers",
                marker=dict(size=1.6, color="#c7d2fe", opacity=0.35),
                name="full peptide surface (points; mesh unavailable)",
                hoverinfo="skip",
            )
        )

    intf = surface.interface_xyz
    if intf.size:
        traces.append(
            go.Scatter3d(
                x=intf[:, 0], y=intf[:, 1], z=intf[:, 2],
                mode="markers",
                marker=dict(
                    size=3.2,
                    color=surface.interface_receptor_distance,
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=float(surface.metadata.get("cutoff", 4.0)),
                    colorbar=dict(title="d to receptor (A)", len=0.55, x=1.02),
                    showscale=True,
                ),
                name=f"retained interface vertices ({intf.shape[0]})",
                text=[f"d = {d:.2f} A" for d in surface.interface_receptor_distance],
                hoverinfo="text",
            )
        )

    valid = surface.sampled_valid_mask
    sampled = surface.sampled_xyz[valid]
    if sampled.size:
        traces.append(
            go.Scatter3d(
                x=sampled[:, 0], y=sampled[:, 1], z=sampled[:, 2],
                mode="markers",
                marker=dict(
                    size=7.5, color="#d93025", opacity=0.95,
                    line=dict(color="#3c0000", width=1),
                ),
                name=f"sampled points ({sampled.shape[0]})",
                text=[f"d = {d:.2f} A" for d in surface.sampled_receptor_distance[valid]],
                hoverinfo="text",
            )
        )

    if intf.size and normal_stride > 0:
        sel = slice(None, None, max(1, normal_stride))
        p, n = intf[sel], surface.interface_normals[sel]
        traces.append(
            go.Cone(
                x=p[:, 0], y=p[:, 1], z=p[:, 2],
                u=n[:, 0] * normal_length, v=n[:, 1] * normal_length, w=n[:, 2] * normal_length,
                sizemode="absolute", sizeref=normal_length, anchor="tail",
                showscale=False, colorscale=[[0, "#111111"], [1, "#111111"]],
                name=f"normals (every {max(1, normal_stride)}th)", showlegend=True,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, x=0),
        scene=dict(
            aspectmode="data",
            xaxis_title="x (A)", yaxis_title="y (A)", zaxis_title="z (A)",
        ),
        height=760,
    )
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       background: #fbfbfd; color: #1f2328; }
@media (prefers-color-scheme: dark) { body { background: #14161a; color: #e6e6e6; } }
header { padding: 14px 20px; border-bottom: 1px solid rgba(128,128,128,.28); }
h1 { margin: 0; font-size: 17px; font-weight: 620; }
h1 small { font-weight: 400; opacity: .65; margin-left: 8px; }
.wrap { display: flex; gap: 18px; padding: 16px 20px; align-items: flex-start; flex-wrap: wrap; }
.viewer { flex: 1 1 620px; min-width: 380px; }
.panel { flex: 0 1 360px; min-width: 300px; }
.card { border: 1px solid rgba(128,128,128,.28); border-radius: 9px; padding: 12px 14px;
        margin-bottom: 14px; background: rgba(127,127,127,.05); }
.card h2 { margin: 0 0 9px; font-size: 12px; letter-spacing: .07em; text-transform: uppercase;
           opacity: .62; font-weight: 650; }
table { width: 100%; border-collapse: collapse; }
td { padding: 3px 0; vertical-align: top; }
td:first-child { opacity: .7; padding-right: 12px; }
td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.warn { border-left: 3px solid #d93025; padding: 7px 11px; margin: 7px 0; border-radius: 4px;
        background: rgba(217,48,37,.09); }
.ok { border-left: 3px solid #1e8e3e; padding: 7px 11px; border-radius: 4px;
      background: rgba(30,142,62,.09); }
.legend span { display: inline-flex; align-items: center; margin-right: 12px; }
.legend i { width: 11px; height: 11px; border-radius: 50%; margin-right: 5px; display: inline-block; }
code { font-size: 12px; opacity: .8; word-break: break-all; }
"""


def _rows(pairs) -> str:
    return "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in pairs
    )


def build_page(surface: PeptideSurface, diag: dict, plot_html: str, outward: float) -> str:
    meta = surface.metadata
    example_id = meta.get("example_id") or Path(meta.get("source_pdb", "?")).stem

    q = diag["distance_quantiles"]
    dist_rows = (
        _rows(
            [
                ("min", f"{q['min']:.2f} A"),
                ("p25", f"{q['p25']:.2f} A"),
                ("median", f"{q['median']:.2f} A"),
                ("p75", f"{q['p75']:.2f} A"),
                ("p90", f"{q['p90']:.2f} A"),
                ("max", f"{q['max']:.2f} A"),
            ]
        )
        if q
        else "<tr><td colspan=2>no retained vertices</td></tr>"
    )

    warn_html = (
        "".join(f"<div class='warn'>{html.escape(w)}</div>" for w in diag["warnings"])
        if diag["warnings"]
        else "<div class='ok'>No warnings: patch size, connectivity, normals and padding all look sane.</div>"
    )

    comps = diag["components"]
    comp_text = (
        f"{len(comps)} ({', '.join(str(c) for c in comps[:6])}{'...' if len(comps) > 6 else ''})"
        if comps
        else "0"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Peptide surface QA - {html.escape(example_id)}</title>
<style>{_PAGE_CSS}</style></head><body>
<header>
  <h1>Peptide interface surface
    <small>{html.escape(example_id)} &middot; cutoff {diag['cutoff']:g} A &middot;
    {diag['n_sampled_slots']} requested points &middot; seed {meta.get('seed')}</small>
  </h1>
</header>
<div class="wrap">
  <div class="viewer">{plot_html}</div>
  <div class="panel">
    <div class="card">
      <h2>Vertex counts</h2>
      <table>{_rows([
        ("full peptide surface", f"{diag['n_full']:,}"),
        ("retained (interface)", f"{diag['n_interface']:,}"),
        ("retained fraction", f"{diag['retained_fraction']:.1%}"),
        ("sampled (valid / slots)", f"{diag['n_sampled_valid']} / {diag['n_sampled_slots']}"),
        ("receptor surface vertices", f"{meta.get('num_receptor_surface_vertices', '?'):,}"
            if isinstance(meta.get('num_receptor_surface_vertices'), int) else "?"),
        ("patch components", comp_text),
      ])}</table>
    </div>
    <div class="card">
      <h2>Nearest-receptor distance</h2>
      <table>{dist_rows}</table>
    </div>
    <div class="card">
      <h2>Sampling coverage</h2>
      <table>{_rows([
        ("mean retained &rarr; nearest sample", f"{diag['coverage_mean']:.2f} A"),
        ("max retained &rarr; nearest sample", f"{diag['coverage_max']:.2f} A"),
        ("normals pointing outward", f"{outward:.1%}"),
      ])}</table>
    </div>
    <div class="card">
      <h2>Checks</h2>
      {warn_html}
    </div>
    <div class="card">
      <h2>Provenance</h2>
      <table>{_rows([
        ("receptor chains", ",".join(meta.get("receptor_chains", []))),
        ("peptide chains", ",".join(meta.get("peptide_chains", []))),
        ("peptide length", meta.get("peptide_length", "?")),
        ("cyclization type", meta.get("cyclization_type", "?")),
        ("extractor version", meta.get("extractor_version", "?")),
      ])}</table>
      <p><code>{html.escape(str(meta.get("source_pdb", "")))}</code></p>
    </div>
    <div class="card legend">
      <h2>Legend</h2>
      <span><i style="background:#9aa0a6"></i>receptor</span>
      <span><i style="background:#1e8e3e"></i>peptide atoms</span>
      <span><i style="background:#c7d2fe"></i>full surface</span>
      <span><i style="background:#d93025"></i>sampled</span>
      <p style="margin:8px 0 0;opacity:.7">Interface vertices are coloured by
      nearest-receptor distance (0 &rarr; {diag['cutoff']:g} A); black cones are surface normals.</p>
    </div>
  </div>
</div></body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--surface-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normal-stride", type=int, default=12, help="draw every Nth normal (0 disables)"
    )
    parser.add_argument("--normal-length", type=float, default=1.2, help="arrow length, A")
    parser.add_argument(
        "--receptor-shell",
        type=float,
        default=12.0,
        help="only draw receptor atoms within this distance of the peptide surface (0 = all)",
    )
    parser.add_argument("--no-mesh", action="store_true", help="skip mesh regeneration")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    surface = load_surface_cache(args.surface_cache)
    meta = surface.metadata
    peptide_chains = list(meta.get("peptide_chains", ["B"]))
    receptor_chains = list(meta.get("receptor_chains", []))

    receptor_xyz, _ = read_atoms(args.pdb, set(receptor_chains))
    peptide_xyz, _ = read_atoms(args.pdb, set(peptide_chains))
    if peptide_xyz.size == 0:
        logger.warning("no peptide atoms found in %s for chains %s", args.pdb, peptide_chains)

    faces = None
    if not args.no_mesh:
        result = peptide_mesh_faces(args.pdb, peptide_chains)
        if result is not None:
            candidate_faces, candidate_vertices = result
            if candidate_vertices.shape[0] == surface.full_peptide_xyz.shape[0]:
                faces = candidate_faces
            else:
                logger.warning(
                    "regenerated mesh has %d vertices but the cache has %d; drawing points instead",
                    candidate_vertices.shape[0],
                    surface.full_peptide_xyz.shape[0],
                )

    diag = compute_diagnostics(surface)
    fig = build_figure(
        surface,
        receptor_xyz,
        peptide_xyz,
        faces,
        args.normal_stride,
        args.normal_length,
        args.receptor_shell,
    )
    plot_html = fig.to_html(full_html=False, include_plotlyjs="inline", default_height="760px")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(surface, diag, plot_html, outwardness(surface)))
    logger.info(
        "wrote %s  (full=%d retained=%d [%.1f%%] sampled=%d, %d warning(s))",
        args.output,
        diag["n_full"],
        diag["n_interface"],
        100 * diag["retained_fraction"],
        diag["n_sampled_valid"],
        len(diag["warnings"]),
    )
    for warning in diag["warnings"]:
        logger.warning("%s", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
