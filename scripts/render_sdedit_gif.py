"""Render SDEdit trajectories (from `sdedit_trajectory.py`) as animated GIFs.

Reads only the saved .npz files, so figures are redrawable without touching a GPU or
re-sampling -- rerun this as often as the styling needs changing.

What a frame shows
------------------
  * every atom the SAMPLED residue actually has (atom37 masked by aatype), so a missing
    CYS SG is visibly missing rather than silently imputed
  * the one-letter residue code at each CA, coloured by chemical class
  * the two atoms the model was conditioned to bond -- N of residue 0 and C of residue
    L-1, the ring-PE fallback endpoints -- ringed, with a dashed connector carrying the
    live distance, green inside the [1.1, 1.6] A acceptance window and red outside
  * the receptor CAs near the binder, in grey, so pose drift is visible

Frames are the model's running clean prediction (`x_1_pred`), i.e. its evolving belief
about the finished peptide. `--state xt` animates the actual noisy integration state
instead. Orientation is fixed across frames: coordinates are rotated onto the principal
axes of the whole trajectory and viewed down PC3, so the ring plane faces the camera and
apparent motion is real motion, never a moving camera.
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
import torch
from matplotlib.lines import Line2D
from openfold.np import residue_constants as rc
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from proteinfoundation.eval.sampled_binder_metrics import atom37_mask_from_aatype  # noqa: E402

N_IDX, CA_IDX_, C_IDX = rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]
WINDOW = (1.1, 1.6)

CLASS_OF = {}
for _letters, _name in (("GAVLIPMFW", "nonpolar"), ("STCYNQ", "polar"),
                        ("DE", "acidic"), ("KRH", "basic")):
    for _l in _letters:
        CLASS_OF[_l] = _name
CLASS_COLOR = {"nonpolar": "#4C6EF5", "polar": "#12B886",
               "acidic": "#E8590C", "basic": "#AE3EC9", "unknown": "#868E96"}


def nc_dist(coords: np.ndarray, aatype: np.ndarray) -> float:
    """Terminal N->C for the frame being drawn, measured off that frame's own atoms.

    Taken from the coordinates rather than a precomputed series so it stays correct for
    whichever `--state` is animated, and so a residue lacking the atom reads NaN instead of
    borrowing a distance from the other state's trajectory.
    """
    valid = atom37_mask_from_aatype(torch.from_numpy(aatype)).numpy()
    if not (valid[0, N_IDX] and valid[-1, C_IDX]):
        return float("nan")
    return float(np.linalg.norm(coords[0, N_IDX] - coords[-1, C_IDX]))


def letter(aa_idx: int) -> str:
    return rc.restypes[aa_idx] if 0 <= aa_idx < len(rc.restypes) else "X"


def principal_frame(pts: np.ndarray) -> np.ndarray:
    """Rotation putting the largest-variance axis on x, second on y. Fixed for all frames."""
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    r = vt
    if np.linalg.det(r) < 0:
        r[2] *= -1.0
    return r


def draw(ax, coords, aatype, receptor, lims, box, title, sub, dist, closed_ok):
    ax.clear()
    # zoom fills the frame: an axis-off 3D cube otherwise wastes most of the canvas
    # A slab, not a cube: viewed top-down, a cube's depth axis is pure wasted canvas.
    # x and y share a scale so in-plane distances stay honest; z gets its real extent.
    ax.set_box_aspect(box, zoom=1.9)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_axis_off()
    ax.view_init(elev=88, azim=-90)

    if receptor is not None and len(receptor):
        ax.scatter(*receptor.T, s=6, c="#CED4DA", alpha=0.55, linewidths=0, depthshade=False)

    valid = atom37_mask_from_aatype(torch.from_numpy(aatype)).numpy()
    L = coords.shape[0]

    # backbone trace N-CA-C, residue by residue
    bb = np.concatenate([coords[i, [N_IDX, CA_IDX_, C_IDX]] for i in range(L)])
    ax.plot(*bb.T, color="#212529", lw=1.9, alpha=0.9, solid_capstyle="round")

    for i in range(L):
        lt = letter(int(aatype[i]))
        col = CLASS_COLOR[CLASS_OF.get(lt, "unknown")]
        pts = coords[i][valid[i]]
        ax.scatter(*pts.T, s=21, c=col, alpha=0.95, linewidths=0, depthshade=False)
        ca = coords[i, CA_IDX_]
        ax.text(ca[0], ca[1], ca[2], lt, fontsize=9.5, color=col,
                weight="bold", ha="center", va="center", zorder=6)

    # the conditioned closing pair
    n0, cl = coords[0, N_IDX], coords[-1, C_IDX]
    dcol = "#2F9E44" if closed_ok else "#E03131"
    ax.plot(*np.stack([n0, cl]).T, color=dcol, lw=2.1, ls=(0, (3, 2)), alpha=0.95)
    ax.scatter(*np.stack([n0, cl]).T, s=115, facecolors="none",
               edgecolors=dcol, linewidths=2.0, depthshade=False)
    mid = (n0 + cl) / 2.0
    ax.text(mid[0], mid[1], mid[2], f"{dist:.2f} A", fontsize=11, color=dcol,
            weight="bold", ha="center", va="bottom", zorder=7)

    # text2D (not fig.text / set_title): ax.clear() removes it each frame, so labels
    # cannot accumulate on top of one another over the animation.
    ax.text2D(0.5, 1.055, title, transform=ax.transAxes, fontsize=10.5, weight="bold",
              color="#212529", ha="center", va="bottom", clip_on=False)
    ax.text2D(0.5, 1.012, sub, transform=ax.transAxes, fontsize=9.0,
              color="#495057", ha="center", va="bottom", clip_on=False)


def render_case(npz_path: Path, out_dir: Path, state: str, max_frames: int,
                fps: int, hold: int, dpi: int) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    coords_all = z[f"{state}_coords"]
    aatype_all = z[f"{state}_aatype"]
    inp_c, inp_a = z["input_coords"], z["input_aatype"]
    receptor = z["receptor_ca"] if "receptor_ca" in z.files else None

    n = coords_all.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(max_frames, n)).round().astype(int))

    # one fixed frame for the whole animation
    ref = np.concatenate([inp_c.reshape(-1, 3), coords_all[idx].reshape(-1, 3)])
    ref = ref[np.isfinite(ref).all(1)]
    R = principal_frame(ref)
    ctr = ref.mean(0)
    rot = lambda p: (p - ctr) @ R.T  # noqa: E731

    inp_r = rot(inp_c.reshape(-1, 3)).reshape(inp_c.shape)
    frames_r = [rot(coords_all[i].reshape(-1, 3)).reshape(coords_all[i].shape) for i in idx]

    pep = np.concatenate([inp_r.reshape(-1, 3)] + [f.reshape(-1, 3) for f in frames_r])
    pep = pep[np.isfinite(pep).all(1)]
    c0 = (pep.max(0) + pep.min(0)) / 2.0
    ext = pep.max(0) - pep.min(0)
    hxy = max(ext[0], ext[1]) / 2.0 * 1.10          # isotropic in the viewing plane
    hz = max(ext[2] / 2.0 * 1.10, hxy * 0.18)       # floor keeps the slab non-degenerate
    halves = np.array([hxy, hxy, hz])
    lims = [(c0[k] - halves[k], c0[k] + halves[k]) for k in range(3)]
    box = (1.0, 1.0, float(hz / hxy))

    rec_r = None
    if receptor is not None and len(receptor):
        rr = rot(receptor)
        inbox = np.ones(len(rr), dtype=bool)
        for k in range(3):
            inbox &= (rr[:, k] >= lims[k][0]) & (rr[:, k] <= lims[k][1])
        rec_r = rr[inbox]

    fig = plt.figure(figsize=(5.6, 5.4), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.90)

    chem, tca, tlat = meta["chem"], meta["t_ca"], meta["t_lat"]
    fin = meta["final_metrics"]
    ok_final = fin.get("cyc/mainchain_cn_bond_success")
    verdict = ("CLOSED" if ok_final == 1.0 else "did not close") if ok_final is not None else "not scorable"
    title = f"{meta['example_id']}  |  {chem}  |  t_ca={tca}, t_lat={tlat}"

    def snap():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    images = []
    d_in = meta["input_nc_A"]
    draw(ax, inp_r, inp_a, rec_r, lims, box, title,
         f"INPUT (linear crystal pose)   terminal N-C = {d_in:.2f} A",
         d_in, WINDOW[0] <= d_in <= WINDOW[1])
    for _ in range(hold):
        images.append(snap())

    series = []
    for k, i in enumerate(idx):
        di = nc_dist(coords_all[i], aatype_all[i])
        series.append(di)
        ok = WINDOW[0] <= di <= WINDOW[1]
        draw(ax, frames_r[k], aatype_all[i], rec_r, lims, box, title,
             f"step {i + 1}/{n}   terminal N-C = {di:.2f} A"
             f"   (target {WINDOW[0]}-{WINDOW[1]} A)", di, ok)
        images.append(snap())
    for _ in range(hold * 2):
        images.append(images[-1])

    # legend strip, drawn once onto the final still
    handles = [Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in CLASS_COLOR.items() if k != "unknown"]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    still = snap()
    plt.close(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['example_id']}_{chem}"
    gif = out_dir / f"{stem}.gif"
    q = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in images]
    q[0].save(gif, save_all=True, append_images=q[1:], duration=int(1000 / fps),
              loop=0, optimize=True, disposal=2)
    still_p = out_dir / f"{stem}_final.png"
    still.save(still_p, optimize=True)

    return {"tag": meta["tag"], "example_id": meta["example_id"], "chem": chem,
            "verdict": verdict, "gif": str(gif), "gif_kb": round(gif.stat().st_size / 1024, 1),
            "still": str(still_p), "n_frames": len(images),
            "input_nc_A": d_in, "final_nc_A": series[-1] if series else float("nan"),
            "retention": fin.get("contact_retention"),
            "ca_rmsd_A": fin.get("ca_rmsd_to_input_A"),
            "n_substitutions": fin.get("n_substitutions"),
            "peptide_length": meta["n_residues"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--state", choices=["x1", "xt"], default="x1",
                    help="x1 = running clean prediction (legible); xt = raw noisy state")
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=int, default=6, help="frames to hold on input and final")
    ap.add_argument("--dpi", type=int, default=78)
    args = ap.parse_args()

    files = sorted(Path(args.traj_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"FATAL: no .npz trajectories in {args.traj_dir}")
    out_dir = Path(args.out_dir)
    rows = []
    for f in files:
        r = render_case(f, out_dir, args.state, args.max_frames, args.fps, args.hold, args.dpi)
        rows.append(r)
        print(f"  {r['example_id']:16} {r['chem']:11} {r['verdict']:14} "
              f"{r['input_nc_A']:6.2f} -> {r['final_nc_A']:6.2f} A   "
              f"{r['n_frames']:3} frames  {r['gif_kb']:7.1f} KB", flush=True)
    (out_dir / "render_index.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} GIF(s) + render_index.json -> {out_dir}", flush=True)
    print(f"total GIF size: {sum(r['gif_kb'] for r in rows) / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
