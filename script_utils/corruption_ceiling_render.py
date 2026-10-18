#!/usr/bin/env python
"""Render the CP -> LP corruption trajectory: closed macrocycle, intermediates, pulled apart.

Reads only what is already on disk -- the per-rung complex PDBs the ladder dumped and the ladder
JSONL -- so every figure re-renders without re-running a single minimization.

Why a shared camera is legitimate here, and why it matters: the receptor is frozen during the
pull and `write_projected_pdb` copies its lines through byte-for-byte, so all ten rungs of an
example are in ONE frame (verified: identical md5 over the chain-A lines). So no superposition is
done, and every panel uses the same `set_view`. Anything that appears to move in the filmstrip is
the peptide actually moving, not a per-panel realignment.

The camera is oriented on the UNION of all rungs' peptides, not on the native one -- the termini
travel up to ~22 A, and a camera framed on the closed ring would push the open states out of shot
and make the pull look smaller than it is.

Outputs, per selected example:
  filmstrip_<cyc>_<example>.png   the rungs side by side, captioned with gap / retention / dG
  overlay_<cyc>_<example>.png     every rung's peptide superimposed in one view
and across examples:
  filmstrip_combined.png          one row per chemistry -- the side-by-side comparison

Examples are picked as the MEDIAN final separation within each chemistry, so the figure is
representative rather than the prettiest ladder available.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# Blue (closed ring) -> red (pulled apart). Read as a progression left to right.
RAMP = [(0.16, 0.40, 0.75), (0.20, 0.55, 0.72), (0.27, 0.66, 0.60), (0.55, 0.72, 0.36),
        (0.80, 0.72, 0.25), (0.90, 0.60, 0.22), (0.90, 0.44, 0.22), (0.85, 0.27, 0.22),
        (0.78, 0.16, 0.25), (0.65, 0.10, 0.28)]


def read_rows(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    raise SystemExit(f"FATAL: corrupt ladder row {path}:{lineno}.")
    return [r for r in rows if r.get("status") != "failed"]


def attach_delta_dg(rows, patterns):
    """Add `delta_dG` per row, referenced to that example's BASELINE rung.

    The same reference the report uses. Against the unminimized `native` rung every delta would
    carry the constant amber-vs-Rosetta relaxation offset instead of the cost of opening.
    """
    dg = {}
    for pat in patterns or []:
        for path in sorted(glob.glob(pat)):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                dg[r["pdb_tag"]] = r.get("binder_rosetta_dG_separated")
    if not dg:
        return
    base = {r["example_id"]: dg.get(r["pdb_tag"])
            for r in rows if r.get("is_baseline_rung")}
    for r in rows:
        mine, ref = dg.get(r["pdb_tag"]), base.get(r["example_id"])
        r["delta_dG"] = (mine - ref) if (mine is not None and ref is not None) else None


def pick_examples(rows, per_chem):
    """Median final separation within each chemistry -- representative, not cherry-picked."""
    by_ex = {}
    for r in rows:
        by_ex.setdefault(r["example_id"], []).append(r)
    finals = {}
    for eid, rs in by_ex.items():
        rs.sort(key=lambda r: r["rung_index"])
        if not rs[-1].get("ladder_complete"):
            continue
        finals[eid] = (rs[0]["cyc_type"], rs[-1]["terminal_ca_A"])
    picked = []
    for cyc in ("mainchain", "disulfide", "isopeptide"):
        cand = sorted(((v, k) for k, (c, v) in finals.items() if c == cyc))
        if not cand:
            continue
        # walk outward from the median so per_chem>1 stays near-typical
        mid = len(cand) // 2
        order = sorted(range(len(cand)), key=lambda i: abs(i - mid))
        picked += [cand[i][1] for i in order[:per_chem]]
    return picked, by_ex


def rung_label(r):
    k = r.get("rung_kind")
    if k == "native":
        return "native\n(as built)"
    if k == "baseline":
        return "baseline\n(minimized only)"
    return f"pull {r['target_gap_A']:g} $\\mathrm{{\\AA}}$"


def setup_scene(cmd, objs, rows, args):
    """One style pass over every loaded rung, then a camera framed on all of them."""
    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("surface_quality", 0 if args.fast else 1)
    cmd.set("transparency", args.receptor_transparency)
    cmd.set("dash_width", 3.0)
    cmd.set("dash_gap", 0.35)
    cmd.set("label_size", 20)
    cmd.set("label_color", "black")
    cmd.set("stick_radius", 0.16)
    cmd.set("sphere_scale", 0.42)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.22)

    for i, (obj, r) in enumerate(zip(objs, rows)):
        col = RAMP[min(i, len(RAMP) - 1)]
        cmd.set_color(f"rung{i}", list(col))
        cmd.show("surface", f"{obj} and chain A")
        cmd.color("grey80", f"{obj} and chain A")
        cmd.show("sticks", f"{obj} and chain B")
        cmd.color(f"rung{i}", f"{obj} and chain B and elem C")
        # The termini are what the experiment moves, so they get a fixed, non-ramped colour
        # in every panel: the eye tracks the same two atoms across the whole strip.
        first, last = terminal_resis(cmd, obj)
        nsel = f"{obj} and chain B and resi {first} and name CA"
        csel = f"{obj} and chain B and resi {last} and name CA"
        cmd.show("spheres", f"{nsel} or {csel}")
        cmd.color("marine", nsel)
        cmd.color("firebrick", csel)
        cmd.distance(f"term_{i}", nsel, csel)
        cmd.color("black", f"term_{i}")
        cmd.hide("labels", f"term_{i}")          # the caption carries the number instead

    # Frame on every rung's peptide at once; a camera set on the closed ring alone would crop
    # the open states and understate the pull.
    cmd.orient(" or ".join(f"({o} and chain B)" for o in objs))
    cmd.zoom(" or ".join(f"({o} and chain B)" for o in objs), buffer=args.zoom_buffer)
    return cmd.get_view()


def terminal_resis(cmd, obj):
    resis = sorted({int(a.resi) for a in cmd.get_model(f"{obj} and chain B and name CA").atom})
    return resis[0], resis[-1]


def render_example(cmd, eid, rows, args, out_dir):
    rows = sorted(rows, key=lambda r: r["rung_index"])
    objs, kept = [], []
    for i, r in enumerate(rows):
        pdb = Path(r["complex_pdb"])
        if not pdb.is_file() and args.complex_dir:
            pdb = Path(args.complex_dir) / f"{r['pdb_tag']}__complex.pdb"
        if not pdb.is_file():
            print(f"  MISSING {pdb}")
            continue
        obj = f"r{i:02d}"
        cmd.load(str(pdb), obj)
        objs.append(obj)
        kept.append(r)
    if not objs:
        return None, None

    view = setup_scene(cmd, objs, kept, args)
    panel_paths = []
    for i, obj in enumerate(objs):
        for o in objs:
            cmd.disable(o)
            cmd.disable(f"term_{objs.index(o)}")
        cmd.enable(obj)
        cmd.enable(f"term_{i}")
        cmd.set_view(view)
        p = out_dir / f"_panel_{eid}_{i:02d}.png"
        cmd.png(str(p), width=args.panel_px, height=args.panel_px, dpi=150,
                ray=0 if args.fast else 1)
        panel_paths.append(p)

    # Overlay: the whole trajectory at once. Receptor surface from one object only, otherwise
    # ten superimposed surfaces render as mud.
    for i, o in enumerate(objs):
        cmd.enable(o)
        cmd.enable(f"term_{i}")
        if i > 0:
            cmd.hide("surface", f"{o} and chain A")
    cmd.set_view(view)
    overlay = out_dir / f"overlay_{kept[0]['cyc_type']}_{eid}.png"
    cmd.png(str(overlay), width=args.panel_px * 2, height=args.panel_px * 2, dpi=150,
            ray=0 if args.fast else 1)
    cmd.delete("all")
    return panel_paths, kept


def strip_figure(panels, rows, eid, out_path, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 3.0))
    axes = np.atleast_1d(axes)
    for ax, p, r in zip(axes, panels, rows):
        ax.imshow(mpimg.imread(p))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#d4d4d8")
        ax.set_title(rung_label(r), fontsize=8.5, pad=4)
        cap = f"{r['terminal_ca_A']:.1f} $\\mathrm{{\\AA}}$\nret {r['contact_retention']:.2f}"
        dg = r.get("delta_dG")
        if dg is not None and not (isinstance(dg, float) and np.isnan(dg)):
            cap += f"\n$\\Delta$dG {dg:+.1f}*"
        ax.set_xlabel(cap, fontsize=8)
    fig.suptitle(f"{rows[0]['cyc_type']}  ·  {eid}{title_extra}\n"
                 f"terminal CA–CA pulled {rows[0]['terminal_ca_A']:.1f} → "
                 f"{rows[-1]['terminal_ca_A']:.1f} $\\mathrm{{\\AA}}$ "
                 f"(receptor frozen, one shared camera, no superposition)",
                 fontsize=10)
    # A single structure's dG scatters by ~+-10 REU against a ~19 REU population SD, so a
    # per-panel value is noise. The interpretable quantity is the n=60 median in the report.
    fig.text(0.5, 0.015, "* per-panel $\\Delta$dG is noise-dominated (single structure, "
                         "population SD ~19 REU) — read the n=60 median in the report, not a "
                         "panel. native→baseline is the minimization offset, not opening.",
             ha="center", fontsize=7, color="#52525b")
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.savefig(out_path, dpi=args_dpi)
    plt.close(fig)


def rung_key(r):
    """What makes two rungs 'the same column' across examples."""
    k = r.get("rung_kind")
    return k if k in ("native", "baseline") else float(r["target_gap_A"])


def combined_figure(per_ex, out_path):
    """One row per example, columns ALIGNED BY RUNG -- not by position.

    Examples whose starting separation already exceeds a ladder rung never run it (the pull
    only ever goes outward), so they have fewer panels. Packing those left would put a panel
    under a header describing a different rung: the isopeptide example starts at 9.5 A, skips
    the 8 A rung, and every one of its panels then sat one column left of its own label.
    Aligning by rung keeps the headers true for every row, and the blank it leaves is itself
    the information that this example started past that rung.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Union of rungs across examples, in ladder order.
    keys, seen = [], set()
    for _, rows in per_ex.values():
        for r in rows:
            k = rung_key(r)
            if k not in seen:
                seen.add(k); keys.append(k)
    gaps = sorted(k for k in keys if not isinstance(k, str))
    columns = ["native", "baseline"] + gaps
    header_row = {}
    for _, rows in per_ex.values():
        for r in rows:
            header_row.setdefault(rung_key(r), r)

    ncol, nrow = len(columns), len(per_ex)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.55 * nrow), squeeze=False)
    for ri, (eid, (panels, rows)) in enumerate(per_ex.items()):
        by_key = {rung_key(r): (p, r) for p, r in zip(panels, rows)}
        for ci, key in enumerate(columns):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(rung_label(header_row[key]), fontsize=8.5, pad=4)
            if key not in by_key:
                # Not a failure: this example started past this rung.
                ax.axis("off")
                ax.set_xlabel("")
                continue
            path, r = by_key[key]
            ax.imshow(mpimg.imread(path))
            for sp in ax.spines.values():
                sp.set_edgecolor("#d4d4d8")
            ax.set_xlabel(f"{r['terminal_ca_A']:.1f} $\\mathrm{{\\AA}}$ · "
                          f"ret {r['contact_retention']:.2f}", fontsize=7.5)
        # The row label belongs to column 0 only -- setting it on the first panel AFTER a
        # blank stamped a second copy mid-row.
        axes[ri][0].set_ylabel(rows[0]["cyc_type"], fontsize=10)
        axes[ri][0].axis("on")
        axes[ri][0].set_xticks([]); axes[ri][0].set_yticks([])

    fig.suptitle("Cyclic peptide pulled open into a linear one, in a frozen pocket\n"
                 "one row per chemistry · columns aligned by rung · blue→red = progression · "
                 "blue/red spheres are the two termini being separated", fontsize=11)
    fig.text(0.5, 0.008, "Receptor is frozen and byte-identical across panels, so one camera "
                         "serves all of them and nothing is superposed: apparent motion is real. "
                         "'baseline' is minimized but NOT pulled — it is the reference the dG "
                         "deltas are read against. A blank cell means that example's termini "
                         "already started beyond that rung.",
             ha="center", fontsize=7.5, color="#52525b")
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    fig.savefig(out_path, dpi=args_dpi)
    plt.close(fig)


args_dpi = 160


def main() -> int:
    global args_dpi
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder-glob", nargs="+", required=True)
    ap.add_argument("--rosetta-glob", nargs="+", default=[],
                    help="dg_shard*.jsonl; adds a delta-dG caption referenced to the baseline.")
    ap.add_argument("--complex-dir", default=None,
                    help="Fallback for complex_pdb if the run directory moved.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--examples", nargs="+", default=None,
                    help="example_ids to render. Default: the median ladder per chemistry.")
    ap.add_argument("--per-chemistry", type=int, default=1)
    ap.add_argument("--panel-px", type=int, default=700)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--zoom-buffer", type=float, default=4.0)
    ap.add_argument("--receptor-transparency", type=float, default=0.45)
    ap.add_argument("--fast", action="store_true",
                    help="Skip ray tracing and surface refinement (draft quality, ~10x faster).")
    ap.add_argument("--keep-panels", action="store_true",
                    help="Keep the per-rung PNGs instead of deleting them after assembly.")
    args = ap.parse_args()
    args_dpi = args.dpi

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.ladder_glob)
    if not rows:
        print("GATE: no usable ladder rows on disk; nothing to render. Exiting 0.")
        return 0

    attach_delta_dg(rows, args.rosetta_glob)
    chosen, by_ex = pick_examples(rows, args.per_chemistry)
    if args.examples:
        chosen = [e for e in args.examples if e in by_ex]
        missing = [e for e in args.examples if e not in by_ex]
        if missing:
            print(f"WARNING: not in the ladder rows: {missing}")
    if not chosen:
        print("GATE: no complete ladder to render. Exiting 0.")
        return 0
    print(f"rendering {len(chosen)} examples: {chosen}")

    import pymol
    pymol.finish_launching(["pymol", "-qck"])
    from pymol import cmd

    per_ex = {}
    for eid in chosen:
        print(f"  {eid} ...", flush=True)
        panels, kept = render_example(cmd, eid, by_ex[eid], args, out_dir)
        if not panels:
            continue
        cyc = kept[0]["cyc_type"]
        strip_figure(panels, kept, eid, out_dir / f"filmstrip_{cyc}_{eid}.png")
        per_ex[eid] = (panels, kept)

    if len(per_ex) > 1:
        combined_figure(per_ex, out_dir / "filmstrip_combined.png")

    if not args.keep_panels:
        for p in out_dir.glob("_panel_*.png"):
            p.unlink()

    print(f"\nwrote {len(list(out_dir.glob('filmstrip_*.png')))} filmstrips and "
          f"{len(list(out_dir.glob('overlay_*.png')))} overlays -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
