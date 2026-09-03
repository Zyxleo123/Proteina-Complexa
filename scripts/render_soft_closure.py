"""Figures for the soft-closure projection: a before/after PNG and an animated GIF.

Reads only saved arrays -- the projection's per-rung .npz files (scripts/soft_closure_project.py)
and, when present, the SDEdit trajectory .npz files (scripts/sdedit_trajectory.py). No OpenMM,
no model, no GPU, so the styling can be re-run as often as it needs changing.

The story a figure has to tell
------------------------------
These inputs failed for a geometric reason: their head-to-tail gap was too wide for the flow
model to span, and closure in the sweep was all-or-nothing rather than near-miss. So both
figures are built around ONE number -- the distance between the two atoms the chosen chemistry
would bond -- shown at every stage of the pull and then through the SDEdit integration.

    before_after.png   INPUT (crystal LP) | PROJECTED (after the pull) | SDEDIT (cyclized)
    softclose.gif      the same sequence animated, rung by rung and then step by step

Frames from the two sources live in different frames of reference: the projection works in the
crystal frame, while the CPSea loader hands SDEdit a target-centred crop. They are reconciled
with a Kabsch fit of the SDEdit run's OWN INPUT onto the projected coordinates -- the same
peptide, so the fit is rigid and exact. The residual is printed and, above --align-tol-A,
the SDEdit panel is dropped rather than drawn in the wrong place.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

# atom37 index of the backbone atoms, taken from the project's own constants rather than
# hardcoded, so a reordering upstream cannot silently mislabel a panel.
from openfold.np import residue_constants as rc

N37, CA37, C37 = rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]

THREE_TO_ONE = {v: k for k, v in rc.restype_1to3.items()}
CLASS_OF = {}
for _letters, _name in (("GAVLIPMFW", "nonpolar"), ("STCYNQ", "polar"),
                        ("DE", "acidic"), ("KRH", "basic")):
    for _l in _letters:
        CLASS_OF[_l] = _name
CLASS_COLOR = {"nonpolar": "#4C6EF5", "polar": "#12B886",
               "acidic": "#E8590C", "basic": "#AE3EC9", "unknown": "#868E96"}
BOND_WINDOW = {"mainchain": (1.1, 1.6), "disulfide": (1.8, 2.3), "isopeptide": (1.1, 1.6)}


# ---------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------
def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Rigid transform taking `mobile` onto `target`. Returns (R, t, rmsd_after)."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    fitted = (mobile - mc) @ r.T + tc
    return r, tc - mc @ r.T, float(np.sqrt(((fitted - target) ** 2).sum(-1).mean()))


def principal_frame(pts: np.ndarray) -> np.ndarray:
    """Rotation putting the largest-variance axis on x, second on y. One frame for every
    panel and every animation frame, so apparent motion is real motion."""
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1.0
    return vt


class Peptide:
    """One peptide pose in a common representation: heavy atoms with names and residue index."""

    def __init__(self, xyz, names, resid, resnames):
        self.xyz, self.names, self.resid = np.asarray(xyz), list(names), np.asarray(resid)
        self.resnames = list(resnames)
        self.n_res = len(self.resnames)
        self._by = {}
        for i, (n, r) in enumerate(zip(self.names, self.resid)):
            self._by[(int(r), n)] = i

    def atom(self, res_i, name):
        i = self._by.get((int(res_i), name))
        return None if i is None else self.xyz[i]

    def backbone(self):
        pts = []
        for r in range(self.n_res):
            for n in ("N", "CA", "C"):
                p = self.atom(r, n)
                if p is not None:
                    pts.append(p)
        return np.asarray(pts)

    def ca(self):
        return np.asarray([self.atom(r, "CA") for r in range(self.n_res)])

    def letters(self):
        return [THREE_TO_ONE.get(rn, "X") for rn in self.resnames]

    def transformed(self, r, t):
        return Peptide(self.xyz @ r.T + t, self.names, self.resid, self.resnames)


def from_atom37(coords37, aatype):
    """A Peptide from an SDEdit frame. Only N/CA/C are carried: they are what the backbone
    trace and the closing-distance annotation need, and they exist for every residue."""
    xyz, names, resid = [], [], []
    for i in range(coords37.shape[0]):
        for nm, k in (("N", N37), ("CA", CA37), ("C", C37)):
            xyz.append(coords37[i, k]); names.append(nm); resid.append(i)
    resnames = [rc.restype_1to3.get(rc.restypes[int(a)], "UNK") if 0 <= int(a) < len(rc.restypes)
                else "UNK" for a in aatype]
    return Peptide(np.asarray(xyz), names, resid, resnames)


def closing_pair(pep: Peptide, cyc_type: str):
    """The two atoms the chemistry would bond, at the conditioned endpoints (0, L-1)."""
    a_name, b_name = {"mainchain": ("N", "C"), "disulfide": ("SG", "SG"),
                      "isopeptide": ("NZ", "CG")}[cyc_type]
    a, b = pep.atom(0, a_name), pep.atom(pep.n_res - 1, b_name)
    if a is None or b is None:
        return None, None, float("nan")
    return a, b, float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------
def draw_pose(ax, pep: Peptide, receptor, lims, box, cyc_type, title, sub, window):
    ax.clear()
    ax.set_box_aspect(box, zoom=2.15)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_axis_off()
    ax.view_init(elev=88, azim=-90)

    if receptor is not None and len(receptor):
        ax.scatter(*receptor.T, s=6, c="#CED4DA", alpha=0.55, linewidths=0, depthshade=False)

    bb = pep.backbone()
    ax.plot(*bb.T, color="#212529", lw=1.9, alpha=0.9, solid_capstyle="round")

    letters = pep.letters()
    for r in range(pep.n_res):
        lt = letters[r]
        col = CLASS_COLOR[CLASS_OF.get(lt, "unknown")]
        pts = pep.xyz[pep.resid == r]
        ax.scatter(*pts.T, s=20, c=col, alpha=0.95, linewidths=0, depthshade=False)
        ca = pep.atom(r, "CA")
        if ca is not None:
            ax.text(ca[0], ca[1], ca[2], lt, fontsize=9.0, color=col,
                    weight="bold", ha="center", va="center", zorder=6)

    a, b, d = closing_pair(pep, cyc_type)
    if a is not None:
        ok = window[0] <= d <= window[1]
        dcol = "#2F9E44" if ok else ("#F08C00" if d < 10.0 else "#E03131")
        ax.plot(*np.stack([a, b]).T, color=dcol, lw=2.1, ls=(0, (3, 2)), alpha=0.95)
        ax.scatter(*np.stack([a, b]).T, s=110, facecolors="none",
                   edgecolors=dcol, linewidths=2.0, depthshade=False)
        mid = (a + b) / 2.0
        ax.text(mid[0], mid[1], mid[2], f"{d:.2f} A", fontsize=11, color=dcol,
                weight="bold", ha="center", va="bottom", zorder=7)

    # Header sits ABOVE the axes box, and the subtitle grows downward from a fixed anchor, so
    # a two-line metrics block cannot ride up into the panel title or the figure suptitle.
    ax.text2D(0.5, 1.105, title, transform=ax.transAxes, fontsize=10.5, weight="bold",
              color="#212529", ha="center", va="bottom", clip_on=False)
    ax.text2D(0.5, 1.082, sub, transform=ax.transAxes, fontsize=8.4, linespacing=1.5,
              color="#495057", ha="center", va="top", clip_on=False)
    return d


def view_box(all_pts):
    pts = all_pts[np.isfinite(all_pts).all(1)]
    c0 = (pts.max(0) + pts.min(0)) / 2.0
    ext = pts.max(0) - pts.min(0)
    hxy = max(ext[0], ext[1]) / 2.0 * 1.12          # isotropic in the viewing plane
    hz = max(ext[2] / 2.0 * 1.12, hxy * 0.18)       # floor keeps the slab non-degenerate
    halves = np.array([hxy, hxy, hz])
    lims = [(c0[k] - halves[k], c0[k] + halves[k]) for k in range(3)]
    return lims, (1.0, 1.0, float(hz / hxy))


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------
def load_projection(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    stages = json.loads(str(z["stages_json"]))
    # Backfill identity from the filename ("<example_id>__proj<k>.npz") for npz written before
    # the writer stamped example_id into meta_json. Every other field the panels read is
    # produced inside project_once and has always been present; only the identity added later
    # in the main loop could be missing.
    stem = npz_path.stem
    if "example_id" not in meta:
        meta["example_id"] = stem.split("__proj")[0]
    if "replica" not in meta and "__proj" in stem:
        try:
            meta["replica"] = int(stem.split("__proj")[1])
        except ValueError:
            meta["replica"] = 0
    meta.setdefault("input_nc_gap_A", float("nan"))
    names = [str(x) for x in z["pep_heavy_names"]]
    resid = z["pep_heavy_resid"]
    resnames = [str(x) for x in z["pep_resnames"]]
    frames = [Peptide(f, names, resid, resnames) for f in z["pep_heavy_xyz"]]
    crystal = Peptide(z["pep_heavy_ref"], names, resid, resnames)
    return {"meta": meta, "stages": stages, "frames": frames, "crystal": crystal,
            "rec_ca": z["rec_ca"], "tag": npz_path.stem}


def load_sdedit(traj_dir: Path, tag: str, state: str):
    """SDEdit frames for this projected input, if the GPU arm has run. Matched on the
    projection tag, which is the example_id the projected metadata parquet carries."""
    hits = sorted(traj_dir.glob(f"{tag}_*.npz"))
    if not hits:
        return None
    z = np.load(hits[0], allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return {
        "meta": meta, "path": hits[0],
        "input": from_atom37(z["input_coords"], z["input_aatype"]),
        "frames_raw": z[f"{state}_coords"], "aatype": z[f"{state}_aatype"],
        "receptor_ca": z["receptor_ca"] if "receptor_ca" in z.files else None,
    }


def align_sdedit(proj_last: Peptide, sd, tol_A: float):
    """Kabsch the SDEdit run's own input onto the projected pose it was built from.

    Same peptide in two frames of reference, so a rigid fit is exact and the residual is a
    CHECK, not a fitting error: anything above `tol_A` means the tag matched the wrong case
    and the panel must not be drawn.
    """
    tgt = proj_last.ca()
    mob = sd["input"].ca()
    if mob.shape != tgt.shape:
        return None, float("inf")
    r, t, rmsd = kabsch(mob, tgt)
    if rmsd > tol_A:
        return None, rmsd
    n = sd["frames_raw"].shape[0]
    frames = [from_atom37(sd["frames_raw"][i], sd["aatype"][i]).transformed(r, t)
              for i in range(n)]
    rec = sd["receptor_ca"] @ r.T + t if sd["receptor_ca"] is not None else None
    return {"frames": frames, "receptor_ca": rec, "meta": sd["meta"]}, rmsd


# ---------------------------------------------------------------------------------------
# The two figures
# ---------------------------------------------------------------------------------------
def render_case(proj, sd_aligned, out_dir: Path, args) -> dict:
    tag = proj["tag"]
    meta, stages = proj["meta"], proj["stages"]
    cyc = meta["cyc_type"]
    window = BOND_WINDOW[cyc]

    crystal, projected = proj["crystal"], proj["frames"][-1]
    panels = [("INPUT  (linear crystal pose)", crystal, proj["rec_ca"]),
              ("PROJECTED  (soft closure, no bond formed)", projected, proj["rec_ca"])]
    if sd_aligned is not None:
        panels.append(("SDEDIT  (cyclized)", sd_aligned["frames"][-1], proj["rec_ca"]))

    # One fixed camera for every panel AND every animation frame.
    pool = [crystal.xyz, projected.xyz] + [f.xyz for f in proj["frames"]]
    if sd_aligned is not None:
        pool += [f.xyz for f in sd_aligned["frames"]]
    ref_pts = np.concatenate(pool)
    rot = principal_frame(ref_pts)
    ctr = ref_pts.mean(0)
    xf = lambda p: (p - ctr) @ rot.T  # noqa: E731

    def place(pep):
        return Peptide(xf(pep.xyz), pep.names, pep.resid, pep.resnames)

    lims, box = view_box(xf(ref_pts))
    rec_r = None
    if proj["rec_ca"] is not None and len(proj["rec_ca"]):
        rr = xf(proj["rec_ca"])
        inbox = np.ones(len(rr), dtype=bool)
        for k in range(3):
            inbox &= (rr[:, k] >= lims[k][0]) & (rr[:, k] <= lims[k][1])
        rec_r = rr[inbox]

    # ---- before/after PNG -------------------------------------------------------------
    ncol = len(panels)
    fig = plt.figure(figsize=(4.9 * ncol, 5.4), dpi=args.dpi)
    subs = [
        f"terminal {cyc} pair = {meta['pull_dist_start_A']:.2f} A   "
        f"CB-CB = {meta['terminal_cb_start_A']:.2f} A",
        f"terminal {cyc} pair = {meta['pull_dist_final_A']:.2f} A   "
        f"CB-CB = {meta['terminal_cb_final_A']:.2f} A\n"
        f"contact retention {meta['contact_retention']:.2f}   "
        f"CA RMSD {meta['ca_rmsd_to_input_A']:.2f} A   "
        f"closest peptide-receptor heavy {meta['min_pep_rec_heavy_A']:.2f} A",
    ]
    if sd_aligned is not None:
        fin = sd_aligned["meta"].get("final_metrics", {})
        ok = fin.get(f"cyc/{'mainchain_cn' if cyc == 'mainchain' else cyc}_bond_success")
        subs.append(
            f"t_ca={sd_aligned['meta'].get('t_ca')}  t_lat={sd_aligned['meta'].get('t_lat')}   "
            f"{'CLOSED' if ok == 1.0 else 'did not close'}\n"
            f"contact retention {fin.get('contact_retention', float('nan')):.2f}   "
            f"substitutions {fin.get('n_substitutions', float('nan')):.0f}")

    dists = []
    for k, ((title, pep, _), sub) in enumerate(zip(panels, subs)):
        ax = fig.add_subplot(1, ncol, k + 1, projection="3d")
        dists.append(draw_pose(ax, place(pep), rec_r, lims, box, cyc, title, sub, window))
    handles = [Line2D([], [], marker="o", ls="", color=c, label=n)
               for n, c in CLASS_COLOR.items() if n != "unknown"]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        f"{meta['example_id']}   replica {meta['replica']}   {cyc}   "
        f"input head-to-tail gap {meta.get('input_nc_gap_A', float('nan')):.1f} A",
        fontsize=12.5, weight="bold", y=0.975)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.78, wspace=0.02)
    png = out_dir / f"{tag}_before_after.png"
    fig.savefig(png, facecolor="white")
    plt.close(fig)

    # ---- GIF --------------------------------------------------------------------------
    figg = plt.figure(figsize=(5.6, 5.4), dpi=args.gif_dpi)
    axg = figg.add_subplot(111, projection="3d")
    figg.subplots_adjust(left=0.0, right=1.0, bottom=0.04, top=0.84)
    title = f"{meta['example_id']}  |  {cyc}  |  replica {meta['replica']}"

    def snap():
        buf = io.BytesIO()
        figg.savefig(buf, format="png", dpi=args.gif_dpi, facecolor="white")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    images = []
    draw_pose(axg, place(crystal), rec_r, lims, box, cyc, title,
              "INPUT: linear crystal pose, no restraints applied", window)
    images += [snap()] * args.hold

    # phase 1: the pull ladder. frames[0] is the starting pose (perturbed for replica > 0),
    # so it is labelled separately from the crystal panel above.
    for i, f in enumerate(proj["frames"]):
        if i == 0:
            sub = ("PULL stage 0: starting pose"
                   + (" (backbone torsions perturbed)" if meta["replica"] > 0 else ""))
        else:
            st = stages[i - 1]
            sub = (f"PULL stage {i}/{len(stages)}:  r0 = {st['r0_A']:.1f} A"
                   f"   -> minimize   (E = {st['energy_kj']:,.0f} kJ/mol)")
        draw_pose(axg, place(f), rec_r, lims, box, cyc, title, sub, window)
        images += [snap()] * args.stage_frames
    images += [snap()] * args.hold

    n_sd = 0
    if sd_aligned is not None:
        sdf = sd_aligned["frames"]
        idx = np.unique(np.linspace(0, len(sdf) - 1, min(args.max_sdedit_frames, len(sdf)))
                        .round().astype(int))
        n_sd = len(idx)
        for k, i in enumerate(idx):
            draw_pose(axg, place(sdf[i]), rec_r, lims, box, cyc, title,
                      f"SDEDIT step {i + 1}/{len(sdf)}   "
                      f"(t_ca={sd_aligned['meta'].get('t_ca')}, "
                      f"t_lat={sd_aligned['meta'].get('t_lat')})", window)
            images.append(snap())
        images += [snap()] * (args.hold * 2)
    plt.close(figg)

    gif = out_dir / f"{tag}_softclose.gif"
    q = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in images]
    q[0].save(gif, save_all=True, append_images=q[1:], duration=int(1000 / args.fps),
              loop=0, optimize=True, disposal=2)

    return {"tag": tag, "example_id": meta["example_id"], "replica": meta["replica"],
            "cyc_type": cyc, "png": str(png), "gif": str(gif),
            "gif_kb": round(gif.stat().st_size / 1024, 1), "n_frames": len(images),
            "n_sdedit_frames": n_sd, "accepted": meta.get("accepted"),
            "input_pair_A": round(dists[0], 2), "projected_pair_A": round(dists[1], 2),
            "sdedit_pair_A": round(dists[2], 2) if len(dists) > 2 else None,
            "contact_retention": meta.get("contact_retention"),
            "input_nc_gap_A": meta.get("input_nc_gap_A")}


def overview_figure(rows, out_dir: Path, dpi: int) -> Path:
    """One panel per input peptide: how far the closing pair travelled at each handoff.

    Drawn from the per-case rows, so it stays correct whether or not the SDEdit arm has run.
    """
    by_ex = {}
    for r in rows:
        by_ex.setdefault(r["example_id"], []).append(r)
    order = sorted(by_ex, key=lambda e: -by_ex[e][0]["input_nc_gap_A"])

    fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(order) + 3.0), 5.0), dpi=dpi)
    x = np.arange(len(order))
    inp = [by_ex[e][0]["input_pair_A"] for e in order]
    # Best replica per peptide: the oracle's question is whether ANY projection works.
    prj = [min(r["projected_pair_A"] for r in by_ex[e]) for e in order]
    sde = [min([r["sdedit_pair_A"] for r in by_ex[e] if r["sdedit_pair_A"] is not None],
               default=np.nan) for e in order]

    w = 0.27
    ax.bar(x - w, inp, w, label="input (crystal LP)", color="#868E96")
    ax.bar(x, prj, w, label="after soft-closure projection", color="#4C6EF5")
    if not all(np.isnan(sde)):
        ax.bar(x + w, sde, w, label="after SDEdit", color="#2F9E44")
    ax.axhline(1.33, color="#E03131", ls="--", lw=1.4)
    ax.text(-0.55, 1.38, "ideal C-N bond 1.33 A", color="#E03131",
            fontsize=8.5, ha="left", va="bottom")
    # Log scale because the input gaps (15-46 A) and a formed bond (1.3 A) differ by more than
    # an order of magnitude; a linear axis would flatten every bar that matters into the floor.
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(1.0, max(inp) * 1.35)
    ax.set_ylabel("terminal closing-pair distance (A, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("LNR_", "") for e in order], rotation=35, ha="right", fontsize=9)
    ax.set_title("Soft-closure projection on the SDEdit sweep's mainchain failures\n"
                 "(best replica per peptide)", fontsize=12, weight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / "soft_closure_overview.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-dir", required=True, help="<projection out-dir>/stages")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sdedit-traj-dir", default=None,
                    help="Frames from scripts/sdedit_trajectory.py run on the projected "
                         "metadata. Omit and the figures show input vs projection only.")
    ap.add_argument("--state", choices=["x1", "xt"], default="x1")
    ap.add_argument("--accepted-only", action="store_true",
                    help="Render only replicas that passed every acceptance gate.")
    ap.add_argument("--per-example", type=int, default=1,
                    help="Replicas to render per input peptide, best closure first.")
    ap.add_argument("--align-tol-A", type=float, default=0.5,
                    help="Max Kabsch residual between an SDEdit run's input and the projection "
                         "it claims to come from. Above this the SDEdit panel is dropped.")
    ap.add_argument("--max-sdedit-frames", type=int, default=40)
    ap.add_argument("--stage-frames", type=int, default=4, help="GIF frames held per pull rung.")
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--gif-dpi", type=int, default=78)
    args = ap.parse_args()

    stage_dir = Path(args.stage_dir)
    files = sorted(stage_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"FATAL: no projection .npz files in {stage_dir}")

    projections = [load_projection(f) for f in files]
    # Enrich any npz whose meta_json predates the example_id/gap stamp from the sibling
    # projections.jsonl (which was always written with the full row). One read, keyed by tag,
    # so re-rendering an older run recovers real gaps instead of the nan filename fallback.
    jsonl = stage_dir.parent / "projections.jsonl"
    if jsonl.is_file():
        by_tag = {}
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("tag"):
                by_tag[r["tag"]] = r
        for p in projections:
            src = by_tag.get(p["tag"])
            if src:
                for key in ("example_id", "replica", "input_nc_gap_A", "accepted"):
                    if key in src:
                        p["meta"][key] = src[key]
    if args.accepted_only:
        projections = [p for p in projections if p["meta"].get("accepted")]
        if not projections:
            raise SystemExit(
                "FATAL: --accepted-only, but no replica passed every gate. That is the "
                "oracle's verdict, not a rendering bug -- rerun without the flag to see why.")
    chosen, by_ex = [], {}
    for p in projections:
        by_ex.setdefault(p["meta"]["example_id"], []).append(p)
    for ex in sorted(by_ex):
        chosen += sorted(by_ex[ex],
                         key=lambda p: p["meta"]["pull_dist_final_A"])[: args.per_example]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(args.sdedit_traj_dir) if args.sdedit_traj_dir else None
    if traj_dir is not None and not traj_dir.is_dir():
        raise SystemExit(f"FATAL: --sdedit-traj-dir does not exist: {traj_dir}")

    rows, n_with_sdedit = [], 0
    for proj in chosen:
        sd_aligned = None
        if traj_dir is not None:
            sd = load_sdedit(traj_dir, proj["tag"], args.state)
            if sd is not None:
                sd_aligned, rmsd = align_sdedit(proj["frames"][-1], sd, args.align_tol_A)
                if sd_aligned is None:
                    print(f"  WARNING {proj['tag']}: SDEdit input does not match this "
                          f"projection (Kabsch residual {rmsd:.2f} A > {args.align_tol_A}); "
                          f"drawing without the SDEdit panel", flush=True)
                else:
                    n_with_sdedit += 1
        r = render_case(proj, sd_aligned, out_dir, args)
        rows.append(r)
        sd_txt = f" -> {r['sdedit_pair_A']:6.2f}" if r["sdedit_pair_A"] is not None else ""
        print(f"  {r['tag']:28s} {r['input_pair_A']:7.2f} -> {r['projected_pair_A']:6.2f}{sd_txt} A"
              f"   ret {r['contact_retention']:.2f}   {r['n_frames']:3d} frames"
              f"   {r['gif_kb']:7.1f} KB", flush=True)

    ov = overview_figure(rows, out_dir, args.dpi)
    (out_dir / "render_index.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} case(s), {n_with_sdedit} with an SDEdit panel", flush=True)
    print(f"overview: {ov}", flush=True)
    print(f"figures:  {out_dir}", flush=True)


if __name__ == "__main__":
    main()
