"""Render SDEdit trajectories as scrubable SPRITE SHEETS instead of GIFs.

Why not a GIF
-------------
A GIF plays at a fixed rate, which is exactly wrong for this data: the interesting motion
is crushed into the first handful of steps and the remaining hundreds are near-stationary.
A slider lets you sit on step 1, where the whole edit happens.

Two states are emitted per case, and the contrast between them is the point:

    xt   the actual integration state -- the thing that genuinely evolves over 400 steps
    x1   the model's running clean-sample prediction, which is a finished peptide from
         the first forward pass and therefore barely moves after step 1

Steps are sampled densely early and geometrically later, because that is where the motion
is; the emitted index carries the TRUE step number for every frame, so the slider label
never implies a uniform schedule that does not exist (`bb_ca` is front-loaded, and
`local_latents` back-loaded, over different intervals).

One vertical sprite sheet per (case, state) rather than N separate images: a shared palette
compresses far better, and scrubbing is a background-position change with nothing to fetch.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from render_sdedit_gif import (  # noqa: E402
    WINDOW,
    draw,
    nc_dist,
    principal_frame,
)


def sample_steps(n: int, n_early: int = 11, n_late: int = 23) -> list[int]:
    """Every one of the first `n_early` steps, then geometric out to the last.

    The edit lands in the first few steps, so uniform sampling would spend most of the
    slider on a plateau and skip the only interesting frames.
    """
    early = list(range(min(n_early, n)))
    if n <= n_early:
        return early
    late = np.unique(np.geomspace(n_early, n - 1, num=n_late).round().astype(int))
    return sorted(set(early) | set(int(x) for x in late))


def render(npz_path: Path, out_dir: Path, state: str, dpi: int) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    coords_all, aatype_all = z[f"{state}_coords"], z[f"{state}_aatype"]
    inp_c, inp_a = z["input_coords"], z["input_aatype"]
    receptor = z["receptor_ca"] if "receptor_ca" in z.files else None

    n = coords_all.shape[0]
    idx = sample_steps(n)

    ref = np.concatenate([inp_c.reshape(-1, 3), coords_all[idx].reshape(-1, 3)])
    ref = ref[np.isfinite(ref).all(1)]
    R, ctr = principal_frame(ref), ref.mean(0)
    rot = lambda p: (p - ctr) @ R.T  # noqa: E731

    inp_r = rot(inp_c.reshape(-1, 3)).reshape(inp_c.shape)
    frames_r = [rot(coords_all[i].reshape(-1, 3)).reshape(coords_all[i].shape) for i in idx]

    pep = np.concatenate([inp_r.reshape(-1, 3)] + [f.reshape(-1, 3) for f in frames_r])
    pep = pep[np.isfinite(pep).all(1)]
    c0 = (pep.max(0) + pep.min(0)) / 2.0
    ext = pep.max(0) - pep.min(0)
    hxy = max(ext[0], ext[1]) / 2.0 * 1.10
    hz = max(ext[2] / 2.0 * 1.10, hxy * 0.18)
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

    fig = plt.figure(figsize=(4.9, 4.6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=1.0)

    tiles, dists = [], []
    for k, i in enumerate(idx):
        di = nc_dist(coords_all[i], aatype_all[i])
        dists.append(None if not np.isfinite(di) else round(float(di), 3))
        ok = WINDOW[0] <= di <= WINDOW[1]
        # Title/subtitle live in the HTML, not the bitmap: they would otherwise be baked
        # into every tile and could not be restyled or themed.
        draw(ax, frames_r[k], aatype_all[i], rec_r, lims, box, "", "", di, ok)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        tiles.append(Image.open(buf).convert("RGB"))
    plt.close(fig)

    w, h = tiles[0].size
    sheet = Image.new("RGB", (w, h * len(tiles)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (0, k * h))
    sheet = sheet.quantize(colors=200, method=Image.Quantize.MEDIANCUT)

    out_dir.mkdir(parents=True, exist_ok=True)
    # The stem MUST carry the grid point and seed: several trajectories now share an
    # (example, chem) -- the whole t_ca x t_lat x seed grid for one peptide -- so a stem of
    # just (example, chem, state) would collide and every grid point would overwrite the last.
    tca_s, tlat_s, seed = meta.get("t_ca"), meta.get("t_lat"), meta.get("seed")
    stem = f"{meta['example_id']}_{meta['chem']}_tca{tca_s}_tlat{tlat_s}_s{seed}_{state}"
    png = out_dir / f"{stem}.png"
    sheet.save(png, optimize=True)

    ts_ca, ts_lat = z["ts_bb_ca"], z["ts_local_latents"]
    return {
        "example_id": meta["example_id"], "chem": meta["chem"], "state": state,
        # Grid-point identity, so the page can offer t_ca / t_lat / pass selectors. `t_ca`
        # (below) is the per-FRAME schedule for the scrubber; `t_ca_start` is the grid knob.
        "t_ca_start": None if tca_s is None else round(float(tca_s), 3),
        "t_lat_start": None if tlat_s is None else round(float(tlat_s), 3),
        "seed": None if seed is None else int(seed),
        # The scored row for THIS exact edit, straight from the sweep (score_edit wrote it into
        # the npz meta), so the page's numbers are identical to the sweep's -- closure, CB,
        # retention, substitutions, seq identity, requested-type, all keyed to this grid point.
        "metrics": meta.get("final_metrics", {}),
        "sprite": png.name, "frame_w": w, "frame_h": h, "n_frames": len(idx),
        "nsteps": n, "steps": [int(i) + 1 for i in idx], "nc_A": dists,
        "t_ca": [round(float(ts_ca[min(i, len(ts_ca) - 1)]), 5) for i in idx],
        "t_lat": [round(float(ts_lat[min(i, len(ts_lat) - 1)]), 5) for i in idx],
        "input_nc_A": meta["input_nc_A"], "kb": round(png.stat().st_size / 1024, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--states", nargs="+", default=["xt"],
                    help="xt = the reverse-ODE state, the actual trajectory. x1 (the "
                         "clean-sample prediction) is near-final from step 1 and is not "
                         "a trajectory -- available, but not the default for a reason.")
    ap.add_argument("--dpi", type=int, default=70)
    args = ap.parse_args()

    files = sorted(Path(args.traj_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"FATAL: no .npz trajectories in {args.traj_dir}")
    out_dir, rows = Path(args.out_dir), []
    for f in files:
        for st in args.states:
            r = render(f, out_dir, st, args.dpi)
            rows.append(r)
            print(f"  {r['example_id']:16} {r['chem']:11} {st:3} "
                  f"{r['n_frames']:3} frames of {r['nsteps']}  {r['kb']:7.1f} KB", flush=True)
    (out_dir / "slider_index.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} sprite sheet(s) -> {out_dir}", flush=True)
    print(f"total {sum(r['kb'] for r in rows) / 1024:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
