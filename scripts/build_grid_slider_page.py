"""The (t_ca, t_lat, pass)-selectable SDEdit step viewer.

Same visual language as scripts/build_slider_page.py -- a card per case, a step slider that
scrubs the flow ODE, the ring-gap curve under it -- but with the whole NOISE grid navigable
inside one card:

    * a case is one (example, chemistry)
    * within it, dropdowns pick t_ca, t_lat and the pass (seed)
    * each pass is tagged in its dropdown label with whether its ring CLOSED (checkmark) or
      not, for the currently selected (t_ca, t_lat) -- the mark tracks the selection, because
      whether a seed closes depends on the noise it was given

Every shown combination carries its full score panel: contact retention, sequence retention
(identity and substitution count), Rosetta interface dG before/after, CB endpoint distance
(and whether it is in the closure window), CA displacement, and whether the requested bond
formed. The numbers are the sweep's own -- score_edit wrote each edit's metrics row into the
trajectory npz, so the page cannot drift from the table it visualizes.

Sprites are embedded as data URIs: one self-contained file, scrubbing and switching are
background-position / background-image changes with nothing to fetch.

CPU only; reads the rendered sprites + rosetta rows, so it re-renders without the model.
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

DG = "binder_rosetta_dG_separated"

# The requested ring's bond-success key, per chemistry -- identical to the summarizer's, so a
# case is graded on the chemistry it asked for and the page agrees with sdedit_grid.csv.
CLOSURE_KEY = {
    "mainchain": "cyc/mainchain_cn_bond_success",
    "disulfide": "cyc/disulfide_bond_success",
    "isopeptide": "cyc/isopeptide_bond_success",
}
CB_WINDOW = (3.0, 8.0)


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 3)


def load_jsonl(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            if "\x00" in line:
                raise SystemExit(f"FATAL: NUL byte in {p}:{lineno} -- corrupt.")
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
    ap.add_argument("--state", default="xt", choices=["xt", "x1"],
                    help="Which trajectory state to show. xt is the integration state that "
                         "evolves; x1 is the running clean prediction (near-final from step 1).")
    args = ap.parse_args()

    idx_path = Path(args.slider_index)
    entries = [e for e in json.loads(idx_path.read_text()) if e.get("state") == args.state]
    if not entries:
        raise SystemExit(f"FATAL: no entries with state={args.state} in {idx_path}")

    ros = load_jsonl(args.rosetta)
    before = {r["example_id"]: r for r in ros if r.get("kind") == "before"}
    after = [r for r in ros if r.get("kind") == "after"]

    def after_dg(ex, chem, tca, tlat, seed):
        """dG of the edited complex at this EXACT grid point and seed, or None."""
        for r in after:
            if (r.get("example_id") == ex and r.get("cyc_type") == chem
                    and num(r.get("t_ca_start")) == num(tca)
                    and num(r.get("t_lat_start")) == num(tlat)
                    and (r.get("seed") is None or int(r.get("seed")) == int(seed))):
                return num(r.get(DG))
        return None

    # Group entries into (example, chem) cases; each variant keyed by (t_ca, t_lat, seed).
    cases = {}
    for e in entries:
        sprite = idx_path.parent / e["sprite"]
        if not sprite.is_file():
            print(f"  WARNING: sprite missing, variant dropped: {sprite}", flush=True)
            continue
        ex, chem = e["example_id"], e["chem"]
        m = e.get("metrics", {}) or {}
        tca, tlat, seed = e.get("t_ca_start"), e.get("t_lat_start"), e.get("seed")

        closed_bond = m.get(CLOSURE_KEY.get(chem))
        closed_bond = None if closed_bond is None else int(float(closed_bond) >= 0.5)
        cb = num(m.get("term_cb_dist_A"))
        closed_cb = None if cb is None else int(CB_WINDOW[0] <= cb <= CB_WINDOW[1])
        reqsat = m.get("requested_type_satisfied")
        reqsat = None if reqsat is None else int(float(reqsat) >= 0.5)
        outcome = ("closed" if closed_bond == 1
                   else ("abstained" if reqsat == 0 else "open"))

        dg_b = num((before.get(ex) or {}).get(DG))
        dg_a = after_dg(ex, chem, tca, tlat, seed)

        variant = {
            "t_ca": num(tca), "t_lat": num(tlat), "seed": seed,
            "img": "data:image/png;base64," + base64.b64encode(sprite.read_bytes()).decode(),
            "w": e["frame_w"], "h": e["frame_h"], "n": e["n_frames"],
            "nsteps": e.get("nsteps"), "steps": e.get("steps", []),
            "nc": [num(v) for v in e.get("nc_A", [])],
            # scores, all keyed to this exact edit
            "closed": closed_bond, "closed_cb": closed_cb, "cb_dist": cb, "outcome": outcome,
            "retention": num(m.get("contact_retention")),
            "seq_identity": num(m.get("seq_identity")),
            "n_subs": num(m.get("n_substitutions")),
            "ca_rmsd": num(m.get("ca_rmsd_to_input_A")),
            "reqsat": reqsat,
            "dg_before": dg_b, "dg_after": dg_a,
            "ddg": None if (dg_b is None or dg_a is None) else round(dg_a - dg_b, 3),
        }
        key = (ex, chem)
        c = cases.setdefault(key, {
            "example_id": ex, "peptide": ex.replace("LNR_", ""), "chem": chem,
            "input_nc_A": num(e.get("input_nc_A")), "variants": {},
        })
        c["variants"][f'{variant["t_ca"]}|{variant["t_lat"]}|{seed}'] = variant

    case_list = []
    for c in cases.values():
        vs = c["variants"]
        tcas = sorted({v["t_ca"] for v in vs.values() if v["t_ca"] is not None})
        tlats = sorted({v["t_lat"] for v in vs.values() if v["t_lat"] is not None})
        seeds = sorted({v["seed"] for v in vs.values() if v["seed"] is not None})
        c["tcas"], c["tlats"], c["seeds"] = tcas, tlats, seeds
        c["n_closed"] = sum(1 for v in vs.values() if v["closed"] == 1)
        c["n_variants"] = len(vs)
        case_list.append(c)
    case_list.sort(key=lambda c: (c["peptide"], c["chem"]))
    if not case_list:
        raise SystemExit("FATAL: no cases assembled -- every sprite was missing?")

    meta = {
        "run": args.run_label or idx_path.parent.parent.name,
        "checkpoint": args.checkpoint,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_cases": len(case_list),
        "n_variants": sum(c["n_variants"] for c in case_list),
        "n_dg": sum(1 for c in case_list for v in c["variants"].values()
                    if v["ddg"] is not None),
    }
    # allow_nan=False: a stray NaN would serialize as the token `NaN`, which is invalid JSON and
    # would make the browser's JSON.parse throw and the page render blank. Every numeric field
    # above is routed through num()/int() (NaN -> None), so this should never fire -- it is here
    # to fail the build LOUDLY if a new field ever skips that, rather than ship a broken page.
    html = (TEMPLATE.replace("__DATA__", json.dumps(case_list, separators=(",", ":"), allow_nan=False))
                    .replace("__META__", json.dumps(meta, separators=(",", ":"), allow_nan=False)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({meta['n_cases']} cases, {meta['n_variants']} variants, "
          f"{meta['n_dg']} with dG, {mb:.1f} MB)", flush=True)


TEMPLATE = r"""<title>Cyclization Grid Viewer</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>
:root {
  --bg: #eef1f0; --surface: #ffffff; --surface-2: #f6f8f7;
  --ink: #101a1d; --ink-2: #3c4a4e; --muted: #6a787c; --line: #d8e0de;
  --accent: #155e75; --better: #0f766e; --worse: #9f1239; --ok: #0f766e; --no: #9f1239;
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
    --accent: #67c2d8; --better: #2dd4bf; --worse: #fb7185; --ok: #2dd4bf; --no: #fb7185;
    --mainchain: #4b95e8; --disulfide: #ef7c48; --isopeptide: #2bb98c;
    --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 30px -20px rgba(0,0,0,.9);
  }
}
:root[data-theme="dark"] {
  --bg: #0e1415; --surface: #151d1f; --surface-2: #1a2325;
  --ink: #e9f0ee; --ink-2: #b8c6c6; --muted: #8b9a9c; --line: #26332f;
  --accent: #67c2d8; --better: #2dd4bf; --worse: #fb7185; --ok: #2dd4bf; --no: #fb7185;
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
.lede { color: var(--ink-2); max-width: 68ch; margin: 0; }
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

.selrow { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: flex-end; }
.self { display: flex; flex-direction: column; gap: 3px; }
.self label { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
              text-transform: uppercase; color: var(--muted); }
.self select { font-family: var(--mono); font-size: 13px; padding: 5px 8px; color: var(--ink);
               background: var(--surface-2); border: 1px solid var(--line); border-radius: 2px;
               min-width: 118px; }
.outc { margin-left: auto; font-family: var(--mono); font-size: 12px; padding: 4px 10px;
        border-radius: 2px; border: 1px solid currentColor; align-self: center; }

.grid3 { display: grid; grid-template-columns: minmax(150px, 1fr) auto minmax(150px, 1fr);
         gap: 18px; align-items: start; }
@media (max-width: 820px) { .grid3 { grid-template-columns: 1fr; } }

.panel { display: flex; flex-direction: column; gap: 8px; padding: 12px 14px;
         background: var(--surface-2); border: 1px solid var(--line); border-radius: 3px; }
.panel.after { border-left: 3px solid var(--accent); }
.panel .lab { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
              text-transform: uppercase; color: var(--muted); }
.metric { display: flex; justify-content: space-between; gap: 10px; font-family: var(--mono);
          font-size: 12.5px; font-variant-numeric: tabular-nums; border-bottom: 1px dotted var(--line);
          padding-bottom: 3px; }
.metric:last-child { border-bottom: none; }
.metric .k { color: var(--ink-2); } .metric .v { color: var(--ink); font-weight: 500; }
.metric .v.up { color: var(--worse); } .metric .v.down { color: var(--better); }
.metric .v.yes { color: var(--ok); } .metric .v.no { color: var(--no); }
.bigdg { font-family: var(--mono); font-size: 24px; font-weight: 500; letter-spacing: -.02em;
         font-variant-numeric: tabular-nums; }

.viewer { display: flex; flex-direction: column; gap: 9px; align-items: center; }
.frame { border: 1px solid var(--line); border-radius: 2px; background: var(--surface-2);
         background-repeat: no-repeat; image-rendering: -webkit-optimize-contrast; }
.controls { display: flex; align-items: center; gap: 10px; width: 100%; }
input[type=range] { flex: 1; accent-color: var(--accent); }
button.play { font: inherit; font-size: 12px; font-family: var(--mono); padding: 3px 10px;
              border: 1px solid var(--line); background: var(--surface-2); color: var(--ink-2);
              border-radius: 2px; cursor: pointer; min-width: 62px; }
button.play:hover { border-color: var(--muted); }
button.play:focus-visible, input[type=range]:focus-visible, select:focus-visible {
              outline: 2px solid var(--accent); outline-offset: 2px; }
.readout { font-family: var(--mono); font-size: 12px; color: var(--ink-2);
           font-variant-numeric: tabular-nums; display: flex; gap: 14px; flex-wrap: wrap;
           justify-content: center; }
.readout b { font-weight: 600; color: var(--ink); }
.gapchart { width: 100%; }
svg text { font-family: var(--mono); font-size: 10.5px; fill: var(--muted); }

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
    <h1>Cyclization Grid Viewer</h1>
    <p class="lede">Pick a noise level (<code>t_ca</code>, <code>t_lat</code>) and a pass, then
      scrub the flow ODE and watch a bound linear peptide become a macrocycle. Each pass is
      tagged with whether its ring closed, and every combination carries its full score panel
      &mdash; retention, sequence cost, Rosetta interface energy, CB and CA geometry, and
      whether the requested bond actually formed.</p>
    <div class="prov" id="prov"></div>
  </header>
  <main id="cases"></main>

  <section class="notes">
    <div class="note">
      <h3>The mark tracks the selection</h3>
      <p>A pass&rsquo;s &check; / &times; is whether <em>that seed</em> closed the ring at the
        <em>currently selected</em> <code>t_ca</code>/<code>t_lat</code> &mdash; a seed that
        closes at low noise may abstain at high noise, so the mark moves with the knobs.</p>
    </div>
    <div class="note">
      <h3>Two closure columns</h3>
      <p><b>bond</b> is the anchor-atom distance in its chemistry window (C&ndash;N / SG&ndash;SG /
        NZ&ndash;CG); it abstains when the sampled sequence carries no anchor pair. <b>CB</b> is
        the anchor-free endpoint bracket, defined even then &mdash; the gap between them is the
        wrong-residue population.</p>
    </div>
    <div class="note">
      <h3>Energy is measured, not per frame</h3>
      <p>Rosetta <code>dG_separated</code> is scored on the input and on the finished edit, on the
        same receptor. The curve under the slider is the ring gap, which IS per frame; the
        FastRelax at the ends would cost more than the sampling if run per step.</p>
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
const key = (tca, tlat, seed) => `${tca}|${tlat}|${seed}`;

document.getElementById("prov").innerHTML = [
  ["run", META.run], ["checkpoint", META.checkpoint || "—"],
  ["cases", META.n_cases], ["variants", META.n_variants],
  ["with dG", META.n_dg + " / " + META.n_variants], ["built", META.built],
].map(([k, v]) => `<span><b>${k}</b> ${v}</span>`).join("");

const host = document.getElementById("cases");
CASES.forEach((c, ci) => {
  const col = cssv("--" + c.chem) || cssv("--accent");
  const el = document.createElement("section");
  el.className = "case";
  const optList = (arr, sel) => arr.map(v =>
    `<option value="${v}"${v === sel ? " selected" : ""}>${v.toFixed(1)}</option>`).join("");
  el.innerHTML = `
    <div class="top">
      <h2>${c.peptide}</h2>
      <span class="chip" style="color:${col}">${c.chem}</span>
      <span class="chip" style="color:var(--muted)">input gap ${fmt(c.input_nc_A, 1)} Å</span>
      <span class="chip" style="color:var(--muted)">${c.n_closed}/${c.n_variants} closed</span>
    </div>
    <div class="selrow">
      <div class="self"><label>t_ca (backbone noise)</label>
        <select class="ss-tca">${optList(c.tcas, c.tcas[0])}</select></div>
      <div class="self"><label>t_lat (sequence noise)</label>
        <select class="ss-tlat">${optList(c.tlats, c.tlats[0])}</select></div>
      <div class="self"><label>pass (seed)</label>
        <select class="ss-seed"></select></div>
      <span class="outc"></span>
    </div>
    <div class="grid3">
      <div class="panel before">
        <span class="lab">before &mdash; linear input</span>
        <span class="bigdg dgb">—</span>
        <div class="metric"><span class="k">Rosetta dG</span><span class="v ub">—</span></div>
        <div class="metric"><span class="k">head&ndash;tail gap</span><span class="v">${fmt(c.input_nc_A, 1)} Å</span></div>
      </div>
      <div class="viewer">
        <div class="frame"></div>
        <div class="controls">
          <button class="play" aria-label="Play the trajectory">▶ play</button>
          <input type="range" min="0" value="0" aria-label="ODE step for ${c.peptide} ${c.chem}">
        </div>
        <div class="readout">
          <span>step <b class="st">—</b> / <b class="ns">—</b></span>
          <span>ring gap <b class="gp">—</b> Å</span>
        </div>
        <div class="gapchart"></div>
      </div>
      <div class="panel after">
        <span class="lab">after &mdash; edited peptide</span>
        <span class="bigdg dga">—</span>
        <div class="metric"><span class="k">Δ dG vs input</span><span class="v ddg">—</span></div>
        <div class="metric"><span class="k">bond closed</span><span class="v cb">—</span></div>
        <div class="metric"><span class="k">CB in window</span><span class="v ccb">—</span></div>
        <div class="metric"><span class="k">CB distance</span><span class="v cbd">—</span></div>
        <div class="metric"><span class="k">CA displacement</span><span class="v car">—</span></div>
        <div class="metric"><span class="k">contact retention</span><span class="v ret">—</span></div>
        <div class="metric"><span class="k">sequence identity</span><span class="v sid">—</span></div>
        <div class="metric"><span class="k">substitutions</span><span class="v sub">—</span></div>
      </div>
    </div>`;
  host.appendChild(el);

  const q = s => el.querySelector(s);
  const selTca = q(".ss-tca"), selTlat = q(".ss-tlat"), selSeed = q(".ss-seed");
  const frame = q(".frame"), range = q("input[type=range]"), play = q("button.play");
  const stEl = q(".st"), nsEl = q(".ns"), gpEl = q(".gp"), chart = q(".gapchart");
  const outc = q(".outc");

  let cur = null, svgState = null, timer = null;

  function variantFor() {
    return c.variants[key(+selTca.value, +selTlat.value, +selSeed.value)];
  }

  function refreshSeedOptions() {
    // Label each pass with whether it closed at the CURRENT (t_ca, t_lat).
    const tca = +selTca.value, tlat = +selTlat.value;
    const prev = selSeed.value;
    selSeed.innerHTML = c.seeds.map(sd => {
      const v = c.variants[key(tca, tlat, sd)];
      const mark = !v ? "—" : (v.closed === 1 ? "✓" : (v.closed === 0 ? "✗" : "·"));
      return `<option value="${sd}"${String(sd) === prev ? " selected" : ""}>seed ${sd}  ${mark}</option>`;
    }).join("");
    if (!c.variants[key(tca, tlat, +selSeed.value)]) {
      // The remembered seed has no variant at this grid point; fall to the first that does.
      const first = c.seeds.find(sd => c.variants[key(tca, tlat, sd)]);
      if (first !== undefined) selSeed.value = first;
    }
  }

  function buildChart(v) {
    chart.innerHTML = "";
    const W = 560, H = 84, pl = 40, pr = 14, pt = 12, pb = 20;
    const vals = v.nc.filter(x => x !== null);
    const lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
    const span = (hi - lo) || 1;
    const xf = i => pl + i / Math.max(1, v.n - 1) * (W - pl - pr);
    const yf = x => pt + (hi - x) / span * (H - pt - pb);
    const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H,
      preserveAspectRatio: "none", role: "img", "aria-label": "Ring gap over the integration" });
    svg.appendChild(mk("line", { x1: pl, x2: W - pr, y1: yf(lo), y2: yf(lo),
      stroke: cssv("--line"), "stroke-width": 1 }));
    const path = v.nc.map((x, i) => (i ? "L" : "M") + xf(i) + " " + yf(x === null ? lo : x)).join(" ");
    svg.appendChild(mk("path", { d: path, fill: "none", stroke: col, "stroke-width": 2,
      "stroke-linejoin": "round" }));
    const hiT = mk("text", { x: pl - 8, y: yf(hi) + 4, "text-anchor": "end" }); hiT.textContent = hi.toFixed(0);
    const loT = mk("text", { x: pl - 8, y: yf(lo) + 4, "text-anchor": "end" }); loT.textContent = lo.toFixed(0);
    const unit = mk("text", { x: pl - 8, y: pt - 2, "text-anchor": "end" }); unit.textContent = "Å";
    svg.append(hiT, loT, unit);
    const rule = mk("line", { x1: xf(0), x2: xf(0), y1: pt - 4, y2: H - pb + 2,
      stroke: cssv("--muted"), "stroke-width": 1, "stroke-dasharray": "3 3" });
    const dot = mk("circle", { cx: xf(0), cy: yf(v.nc[0] ?? lo), r: 4, fill: col,
      stroke: cssv("--surface"), "stroke-width": 1.5 });
    svg.append(rule, dot);
    const cap = mk("text", { x: pl, y: H - 4, "text-anchor": "start" });
    cap.textContent = "ring gap across the integration →"; svg.appendChild(cap);
    chart.appendChild(svg);
    return { xf, yf, lo, rule, dot };
  }

  function showStep(i) {
    if (!cur) return;
    frame.style.backgroundPosition = `0px ${-i * cur.h}px`;
    stEl.textContent = cur.steps[i]; gpEl.textContent = fmt(cur.nc[i], 1);
    if (svgState) {
      svgState.dot.setAttribute("cx", svgState.xf(i));
      svgState.dot.setAttribute("cy", svgState.yf(cur.nc[i] ?? svgState.lo));
      svgState.rule.setAttribute("x1", svgState.xf(i));
      svgState.rule.setAttribute("x2", svgState.xf(i));
    }
  }

  function loadVariant() {
    const v = variantFor();
    if (!v) return;
    cur = v;
    frame.style.width = v.w + "px"; frame.style.height = v.h + "px";
    frame.style.backgroundImage = `url('${v.img}')`;
    range.max = v.n - 1; range.value = 0; nsEl.textContent = v.nsteps;
    svgState = buildChart(v);
    showStep(0);

    // outcome chip
    const oc = v.outcome === "closed" ? cssv("--ok") : cssv("--muted");
    outc.style.color = oc;
    outc.textContent = v.outcome === "closed" ? "ring closed"
      : (v.outcome === "abstained" ? "abstained (no anchor in sequence)" : "ring not closed");

    // before / after panels
    q(".dgb").textContent = v.dg_before === null ? "not scored" : fmt(v.dg_before, 1);
    q(".ub").textContent = v.dg_before === null ? "—" : "REU";
    q(".dga").textContent = v.dg_after === null ? "not scored" : fmt(v.dg_after, 1);
    const dd = q(".ddg");
    dd.textContent = v.ddg === null ? "—" : sgn(v.ddg) + " REU " + (v.ddg > 0 ? "(weaker)" : "(stronger)");
    dd.className = "v ddg" + (v.ddg === null ? "" : (v.ddg > 0 ? " up" : " down"));
    const yn = (elm, val) => { const e = q(elm);
      e.textContent = val === null ? "—" : (val ? "yes" : "no");
      e.className = "v " + elm.slice(1) + (val === null ? "" : (val ? " yes" : " no")); };
    yn(".cb", v.closed); yn(".ccb", v.closed_cb);
    q(".cbd").textContent = v.cb_dist === null ? "—" : fmt(v.cb_dist, 2) + " Å";
    q(".car").textContent = v.ca_rmsd === null ? "—" : fmt(v.ca_rmsd, 2) + " Å";
    q(".ret").textContent = v.retention === null ? "—" : fmt(v.retention, 2);
    q(".sid").textContent = v.seq_identity === null ? "—" : fmt(v.seq_identity, 2);
    q(".sub").textContent = v.n_subs === null ? "—" : v.n_subs.toFixed(0);
  }

  function onGrid() { refreshSeedOptions(); loadVariant(); }
  selTca.addEventListener("change", onGrid);
  selTlat.addEventListener("change", onGrid);
  selSeed.addEventListener("change", loadVariant);
  range.addEventListener("input", () => showStep(+range.value));
  play.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; }
    play.textContent = "❚❚ pause";
    timer = setInterval(() => {
      const i = (+range.value + 1) % cur.n;
      range.value = i; showStep(i);
      if (i === cur.n - 1) { clearInterval(timer); timer = null; play.textContent = "▶ play"; }
    }, 140);
  });

  refreshSeedOptions();
  loadVariant();
});
</script>
"""


if __name__ == "__main__":
    main()
