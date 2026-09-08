"""The scrubable SDEdit slideshow page, with the Rosetta interface energy on both ends.

Fuses two artefacts that until now lived apart:

    scripts/render_sdedit_slider.py   sprite sheets + slider_index.json -- the structure at
                                      each ODE step, scrubable, with the ring gap per frame
    scripts/score_sdedit_rosetta.py   rosetta_shard*.jsonl -- dG_separated of the input
                                      complex and of the edited complex

The slideshow shows the geometry moving; it never said whether the thing being built still
binds. So every case gets the paired energy around it: the input's dG on the left of the
scrubber, the edited peptide's on the right, and the move between them stated once. Rosetta
is not run per frame -- a FastRelax per ODE step would cost more than the sampling did --
so the two ends are measured and the middle is the ring gap curve, which IS per frame.

Pairing is by (example, chemistry) and, when the grid point is recorded on both sides, by
(t_ca, t_lat, seed) exactly. A case with no matching score says so rather than borrowing a
number from a different attempt.

Sprites are embedded as data URIs: the page is one self-contained file with no network
dependency beyond its webfonts, and scrubbing is a background-position change with nothing
to fetch.
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

DG = "binder_rosetta_dG_separated"


def start_of(v):
    """The index stores t_ca / t_lat as the per-frame schedule; the grid point is its start."""
    if isinstance(v, (list, tuple)):
        return num(v[0]) if v else None
    return num(v)


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 3)


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"FATAL: corrupt row {p}:{lineno}. Do not trust this run.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slider-index", required=True, help="slider_index.json from the renderer.")
    ap.add_argument("--rosetta", nargs="*", default=[], help="rosetta_shard*.jsonl (optional).")
    ap.add_argument("--out", required=True, help="HTML file to write.")
    ap.add_argument("--run-label", default="", help="Run id, for the provenance strip.")
    ap.add_argument("--checkpoint", default="", help="Flow checkpoint pin, for provenance.")
    ap.add_argument("--state", default="xt", choices=["xt", "x1", "both"],
                    help="Which trajectory state to show. `xt` is the integration state that "
                         "actually evolves; `x1` is the running clean prediction, which barely "
                         "moves after step 1.")
    args = ap.parse_args()

    idx_path = Path(args.slider_index)
    cases_raw = json.loads(idx_path.read_text())
    if args.state != "both":
        cases_raw = [c for c in cases_raw if c.get("state") == args.state]
    if not cases_raw:
        raise SystemExit(f"FATAL: no cases with state={args.state} in {idx_path}")

    ros = load_jsonl([Path(p) for p in args.rosetta])
    before = {r["example_id"]: r for r in ros if r.get("kind") == "before"}
    after = [r for r in ros if r.get("kind") == "after"]

    def score_for(case):
        """(before, after, how) for this case: exact grid point if both sides record it."""
        ex, chem = case["example_id"], case["chem"]
        pool = [r for r in after if r.get("example_id") == ex and r.get("cyc_type") == chem]
        if not pool:
            return None, None, "none"
        exact = [r for r in pool
                 if num(r.get("t_ca_start")) == start_of(case.get("t_ca"))
                 and num(r.get("t_lat_start")) == start_of(case.get("t_lat"))]
        if len(exact) == 1:
            return before.get(ex), exact[0], "exact"
        if exact:
            # Same grid point, several seeds: the slideshow is one of them but the index does
            # not record which, so take the median attempt rather than pretending to know.
            exact.sort(key=lambda r: (num(r.get(DG)) is None, num(r.get(DG)) or 0))
            return before.get(ex), exact[len(exact) // 2], "median of %d seeds" % len(exact)
        pool.sort(key=lambda r: (num(r.get(DG)) is None, num(r.get(DG)) or 0))
        return before.get(ex), pool[len(pool) // 2], "median of %d attempts" % len(pool)

    cases = []
    for c in cases_raw:
        sprite = idx_path.parent / c["sprite"]
        if not sprite.is_file():
            print(f"  WARNING: sprite missing, case dropped: {sprite}", flush=True)
            continue
        b, a, how = score_for(c)
        dg_b = num(b.get(DG)) if b else None
        dg_a = num(a.get(DG)) if a else None
        cases.append({
            "id": f'{c["example_id"]}_{c["chem"]}_{c.get("state", "xt")}',
            "peptide": c["example_id"].replace("LNR_", ""),
            "chem": c["chem"],
            "state": c.get("state", "xt"),
            "t_ca": start_of(c.get("t_ca")), "t_lat": start_of(c.get("t_lat")),
            "nsteps": c.get("nsteps"),
            "w": c["frame_w"], "h": c["frame_h"], "n": c["n_frames"],
            "steps": c.get("steps", []),
            "nc": [num(v) for v in c.get("nc_A", [])],
            "input_nc": num(c.get("input_nc_A")),
            "img": "data:image/png;base64," + base64.b64encode(sprite.read_bytes()).decode(),
            "dg_before": dg_b, "dg_after": dg_a,
            "ddg": None if (dg_b is None or dg_a is None) else round(dg_a - dg_b, 3),
            "match": how,
            "outcome": (None if a is None else
                        ("closed" if a.get("closed") else
                         ("open" if a.get("requested_type_satisfied") else "abstained"))),
            "ring_A": num(a.get("ring_dist_A")) if a else None,
            "retention": num(a.get("contact_retention")) if a else None,
            "norelax": num(a.get("binder_rosetta_dG_separated_norelax")) if a else None,
        })
    cases.sort(key=lambda c: (c["peptide"], c["chem"]))

    meta = {
        "run": args.run_label or idx_path.parent.parent.name,
        "checkpoint": args.checkpoint,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_cases": len(cases),
        "n_scored": sum(1 for c in cases if c["ddg"] is not None),
    }
    html = (TEMPLATE.replace("__DATA__", json.dumps(cases, separators=(",", ":")))
                    .replace("__META__", json.dumps(meta, separators=(",", ":"))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({meta['n_cases']} cases, {meta['n_scored']} with paired dG, {mb:.1f} MB)",
          flush=True)


TEMPLATE = r"""<title>Cyclization Step Viewer</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>
:root {
  --bg: #eef1f0; --surface: #ffffff; --surface-2: #f6f8f7;
  --ink: #101a1d; --ink-2: #3c4a4e; --muted: #6a787c; --line: #d8e0de;
  --accent: #155e75; --better: #0f766e; --worse: #9f1239;
  --mainchain: #2a78d6; --disulfide: #d9541f; --isopeptide: #12876a;
  --shadow: 0 1px 2px rgba(16,26,29,.06), 0 8px 24px -18px rgba(16,26,29,.5);
  --serif: "IBM Plex Serif", Georgia, serif;
  --sans: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1415; --surface: #151d1f; --surface-2: #1a2325;
    --ink: #e9f0ee; --ink-2: #b8c6c6; --muted: #8b9a9c; --line: #26332f;
    --accent: #67c2d8; --better: #2dd4bf; --worse: #fb7185;
    --mainchain: #4b95e8; --disulfide: #ef7c48; --isopeptide: #2bb98c;
    --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 30px -20px rgba(0,0,0,.9);
  }
}
:root[data-theme="dark"] {
  --bg: #0e1415; --surface: #151d1f; --surface-2: #1a2325;
  --ink: #e9f0ee; --ink-2: #b8c6c6; --muted: #8b9a9c; --line: #26332f;
  --accent: #67c2d8; --better: #2dd4bf; --worse: #fb7185;
  --mainchain: #4b95e8; --disulfide: #ef7c48; --isopeptide: #2bb98c;
  --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 30px -20px rgba(0,0,0,.9);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
       font-size: 15px; line-height: 1.55; -webkit-font-smoothing: antialiased; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 34px 24px 76px;
        display: flex; flex-direction: column; gap: 24px; }
h1 { font-family: var(--serif); font-weight: 600; font-size: 34px; line-height: 1.15;
     margin: 0; letter-spacing: -.01em; text-wrap: balance; }
.lede { color: var(--ink-2); max-width: 66ch; margin: 0; }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .13em;
           text-transform: uppercase; color: var(--muted); }
header { border-bottom: 2px solid var(--ink); padding-bottom: 20px;
         display: flex; flex-direction: column; gap: 12px; }
.prov { display: flex; flex-wrap: wrap; gap: 8px 26px; font-family: var(--mono);
        font-size: 12px; color: var(--muted); }
.prov b { color: var(--ink-2); font-weight: 500; }

.case { background: var(--surface); border: 1px solid var(--line); border-radius: 3px;
        box-shadow: var(--shadow); padding: 18px 20px 20px;
        display: flex; flex-direction: column; gap: 14px; }
.case > .top { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 14px; }
.case h2 { font-family: var(--mono); font-size: 17px; font-weight: 600; margin: 0;
           letter-spacing: -.01em; }
.chip { font-family: var(--mono); font-size: 11px; padding: 2px 8px; border-radius: 2px;
        border: 1px solid currentColor; }
.grid3 { display: grid; grid-template-columns: minmax(150px, 1fr) auto minmax(150px, 1fr);
         gap: 18px; align-items: center; }
@media (max-width: 780px) { .grid3 { grid-template-columns: 1fr; } }

.end { display: flex; flex-direction: column; gap: 3px; padding: 12px 14px;
       background: var(--surface-2); border: 1px solid var(--line); border-radius: 3px; }
.end.after { border-left: 3px solid var(--accent); }
.end .lab { font-family: var(--mono); font-size: 10.5px; letter-spacing: .12em;
            text-transform: uppercase; color: var(--muted); }
.end .v { font-family: var(--mono); font-size: 25px; font-weight: 500;
          font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.end .u { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.end .sub { font-size: 12.5px; color: var(--ink-2); }
.up { color: var(--worse); } .down { color: var(--better); }
.unscored .v { color: var(--muted); font-size: 19px; }

.viewer { display: flex; flex-direction: column; gap: 9px; align-items: center; }
.frame { border: 1px solid var(--line); border-radius: 2px; background: var(--surface-2);
         background-repeat: no-repeat; image-rendering: -webkit-optimize-contrast; }
.controls { display: flex; align-items: center; gap: 10px; width: 100%; }
input[type=range] { flex: 1; accent-color: var(--accent); }
button.play { font: inherit; font-size: 12px; font-family: var(--mono); padding: 3px 10px;
              border: 1px solid var(--line); background: var(--surface-2); color: var(--ink-2);
              border-radius: 2px; cursor: pointer; min-width: 62px; }
button.play:hover { border-color: var(--muted); }
button.play:focus-visible, input[type=range]:focus-visible { outline: 2px solid var(--accent);
              outline-offset: 2px; }
.readout { font-family: var(--mono); font-size: 12px; color: var(--ink-2);
           font-variant-numeric: tabular-nums; display: flex; gap: 14px; flex-wrap: wrap;
           justify-content: center; }
.readout b { font-weight: 600; color: var(--ink); }
.gapchart { width: 100%; }
svg text { font-family: var(--mono); font-size: 10.5px; fill: var(--muted); }
.foot { font-size: 12.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 18px; }

.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px;
         margin-top: 6px; }
.note { border-left: 2px solid var(--accent); padding-left: 14px; }
.note h3 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
.note p { margin: 0; font-size: 13.5px; color: var(--ink-2); }
code { font-family: var(--mono); font-size: .92em; background: var(--surface-2);
       padding: 1px 5px; border-radius: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Proteina&#8209;Complexa &middot; track&#8209;asymmetric SDEdit</div>
    <h1>Cyclization Step Viewer</h1>
    <p class="lede">Scrub the flow ODE and watch a bound linear peptide become a macrocycle.
      Each trajectory is flanked by the Rosetta interface energy of what went in and what came
      out &mdash; the geometry moving is only half the question; the other half is whether the
      thing still binds.</p>
    <div class="prov" id="prov"></div>
  </header>
  <main id="cases"></main>

  <section class="notes">
    <div class="note">
      <h3>Energy at the ends, gap in the middle</h3>
      <p>Rosetta is not run per frame &mdash; a FastRelax per ODE step would cost more than the
        sampling did. The two ends are measured; the curve under each slider is the ring gap,
        which is computed for every frame.</p>
    </div>
    <div class="note">
      <h3>The edit happens immediately</h3>
      <p>Frames are sampled densely over the first steps and geometrically after, because that is
        where the motion is. The step label is the true step number, never a uniform schedule.</p>
    </div>
    <div class="note">
      <h3>&ldquo;Before&rdquo; is the model&rsquo;s input</h3>
      <p>The staged crystal complex for a control arm, the projected pose for a soft-closure
        <em>projected</em> arm. Both ends are scored on the same receptor, which is what makes the
        difference readable &mdash; <code>dG_separated</code> does not compare across targets.</p>
    </div>
  </section>
</div>

<script>
const CASES = __DATA__;
const META = __META__;
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 1) => (v === null || v === undefined) ? "—" : v.toFixed(d);
const sgn = v => (v === null || v === undefined) ? "—" : (v > 0 ? "+" : "") + v.toFixed(1);
const NS = "http://www.w3.org/2000/svg";
const mk = (n, a) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

document.getElementById("prov").innerHTML = [
  ["run", META.run], ["checkpoint", META.checkpoint || "—"],
  ["trajectories", META.n_cases], ["with paired dG", META.n_scored + " / " + META.n_cases],
  ["built", META.built],
].map(([k, v]) => `<span><b>${k}</b> ${v}</span>`).join("");

const host = document.getElementById("cases");
CASES.forEach(c => {
  const col = cssv("--" + c.chem) || cssv("--accent");
  const el = document.createElement("section");
  el.className = "case";
  const outcomeTxt = c.outcome === null ? "not scored"
    : (c.outcome === "open" ? "ring not closed" : (c.outcome === "closed" ? "ring closed" : "abstained"));
  el.innerHTML = `
    <div class="top">
      <h2>${c.peptide}</h2>
      <span class="chip" style="color:${col}">${c.chem}</span>
      <span class="chip" style="color:var(--muted)">t_ca ${fmt(c.t_ca, 1)} · t_lat ${fmt(c.t_lat, 1)}</span>
      <span class="chip" style="color:var(--muted)">${c.nsteps} ODE steps</span>
      <span class="chip" style="color:${c.outcome === "closed" ? cssv("--better") : cssv("--muted")}">${outcomeTxt}</span>
    </div>
    <div class="grid3">
      <div class="end ${c.dg_before === null ? "unscored" : ""}">
        <span class="lab">before &mdash; linear input</span>
        <span class="v">${c.dg_before === null ? "not scored" : fmt(c.dg_before, 1)}</span>
        <span class="u">${c.dg_before === null ? "" : "REU · Rosetta dG_separated"}</span>
        <span class="sub">head-to-tail gap ${fmt(c.input_nc, 1)} Å</span>
      </div>
      <div class="viewer">
        <div class="frame" style="width:${c.w}px;height:${c.h}px;background-image:url('${c.img}')"></div>
        <div class="controls">
          <button class="play" aria-label="Play the trajectory">▶ play</button>
          <input type="range" min="0" max="${c.n - 1}" value="0"
                 aria-label="ODE step for ${c.peptide} ${c.chem}">
        </div>
        <div class="readout">
          <span>step <b class="st">${c.steps[0]}</b> / ${c.nsteps}</span>
          <span>ring gap <b class="gp">${fmt(c.nc[0], 1)}</b> Å</span>
        </div>
        <div class="gapchart"></div>
      </div>
      <div class="end after ${c.dg_after === null ? "unscored" : ""}">
        <span class="lab">after &mdash; edited peptide</span>
        <span class="v">${c.dg_after === null ? "not scored" : fmt(c.dg_after, 1)}</span>
        <span class="u">${c.dg_after === null ? "" : "REU · Rosetta dG_separated"}</span>
        <span class="sub ${c.ddg === null ? "" : (c.ddg > 0 ? "up" : "down")}">
          ${c.ddg === null ? "no paired score for this attempt" :
            sgn(c.ddg) + " REU vs the input" + (c.ddg > 0 ? " · weaker" : " · stronger")}</span>
      </div>
    </div>
    <div class="foot">
      <span>final gap ${fmt(c.nc[c.n - 1], 1)} Å</span>
      ${c.retention === null ? "" : `<span>contact retention ${fmt(c.retention, 2)}</span>`}
      ${c.norelax === null ? "" : `<span>pre-relax dG ${fmt(c.norelax, 0)} REU</span>`}
      ${c.match === "none" ? "<span>energy: no matching scored attempt in this run</span>"
        : (c.match === "exact" ? "<span>energy: this exact grid point</span>"
           : `<span>energy: ${c.match} at this grid point</span>`)}
    </div>`;
  host.appendChild(el);

  const frame = el.querySelector(".frame");
  const range = el.querySelector("input[type=range]");
  const stEl = el.querySelector(".st"), gpEl = el.querySelector(".gp");
  const chart = el.querySelector(".gapchart");
  const play = el.querySelector("button.play");

  /* ring gap per frame, with the current step marked */
  const W = 560, H = 84, pl = 40, pr = 14, pt = 12, pb = 20;
  const vals = c.nc.filter(v => v !== null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = (hi - lo) || 1;
  const xf = i => pl + i / Math.max(1, c.n - 1) * (W - pl - pr);
  const yf = v => pt + (hi - v) / span * (H - pt - pb);
  const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H,
    preserveAspectRatio: "none", role: "img",
    "aria-label": "Ring gap over the integration, in Angstrom" });
  svg.appendChild(mk("line", { x1: pl, x2: W - pr, y1: yf(lo), y2: yf(lo),
    stroke: cssv("--line"), "stroke-width": 1 }));
  const path = c.nc.map((v, i) => (i ? "L" : "M") + xf(i) + " " + yf(v)).join(" ");
  svg.appendChild(mk("path", { d: path, fill: "none", stroke: col, "stroke-width": 2,
    "stroke-linejoin": "round" }));
  const hiT = mk("text", { x: pl - 8, y: yf(hi) + 4, "text-anchor": "end" });
  hiT.textContent = hi.toFixed(0); svg.appendChild(hiT);
  const loT = mk("text", { x: pl - 8, y: yf(lo) + 4, "text-anchor": "end" });
  loT.textContent = lo.toFixed(0); svg.appendChild(loT);
  const unit = mk("text", { x: pl - 8, y: pt - 2, "text-anchor": "end" });
  unit.textContent = "Å"; svg.appendChild(unit);
  const cur = mk("circle", { cx: xf(0), cy: yf(c.nc[0]), r: 4, fill: col,
    stroke: cssv("--surface"), "stroke-width": 1.5 });
  const rule = mk("line", { x1: xf(0), x2: xf(0), y1: pt - 4, y2: H - pb + 2,
    stroke: cssv("--muted"), "stroke-width": 1, "stroke-dasharray": "3 3" });
  svg.appendChild(rule); svg.appendChild(cur);
  const cap = mk("text", { x: pl, y: H - 4, "text-anchor": "start" });
  cap.textContent = "ring gap across the integration →"; svg.appendChild(cap);
  chart.appendChild(svg);

  function show(i) {
    frame.style.backgroundPosition = `0px ${-i * c.h}px`;
    stEl.textContent = c.steps[i];
    gpEl.textContent = fmt(c.nc[i], 1);
    cur.setAttribute("cx", xf(i)); cur.setAttribute("cy", yf(c.nc[i]));
    rule.setAttribute("x1", xf(i)); rule.setAttribute("x2", xf(i));
  }
  range.addEventListener("input", () => show(+range.value));
  show(0);

  let timer = null;
  play.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; }
    play.textContent = "❚❚ pause";
    timer = setInterval(() => {
      const i = (+range.value + 1) % c.n;
      range.value = i; show(i);
      if (i === c.n - 1) { clearInterval(timer); timer = null; play.textContent = "▶ play"; }
    }, 140);
  });
});
</script>
"""


if __name__ == "__main__":
    main()
