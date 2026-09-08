"""Build the interactive before/after Rosetta report: one self-contained HTML page.

Companion to scripts/plot_rosetta_before_after.py (which makes the static PNG for a slide).
This one is for working through the run: every scored edit is on the page, filterable by
chemistry and outcome, with the per-peptide dumbbell and the per-edit delta redrawn from the
current filter, and a ledger table underneath carrying the numbers the charts summarise.

It reads only the scorer's JSONL, so the page rebuilds in a second without Rosetta, a model,
or a GPU. The data is embedded in the file: the page has no network dependency beyond its
webfonts and works from a file:// path.

Failed and abstained edits are INCLUDED by default and marked, not filtered away. Whether a
closed ring costs more interface energy than an open one is exactly the comparison this run
exists to make, and it needs the failures.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DG = "binder_rosetta_dG_separated"


def load(paths: list[Path]) -> list[dict]:
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


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 3)  # NaN -> None; JSON has no NaN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rosetta", nargs="+", required=True, help="rosetta_shard*.jsonl")
    ap.add_argument("--out", required=True, help="HTML file to write.")
    ap.add_argument("--run-label", default="", help="Run id / arm, shown in the provenance strip.")
    ap.add_argument("--checkpoint", default="", help="Flow checkpoint pin, for provenance.")
    args = ap.parse_args()

    rows = load([Path(p) for p in args.rosetta])
    before = {r["example_id"]: r for r in rows if r.get("kind") == "before"}
    after = [r for r in rows if r.get("kind") == "after"]
    if not after:
        raise SystemExit("FATAL: no `after` rows -- nothing to report.")

    edits = []
    for r in after:
        b = before.get(r.get("example_id"), {})
        dg_b, dg_a = num(b.get(DG)), num(r.get(DG))
        # `ring_declared` is the post-fix field; older rows only carry Rosetta's own answer.
        declared = r.get("ring_declared")
        if declared is None:
            declared = r.get("binder_rosetta_cyclic_topology_declared")
        sat = int(r.get("requested_type_satisfied") or 0)
        closed = int(r.get("closed") or 0)
        edits.append({
            "id": r.get("run_key", ""),
            "peptide": str(r.get("example_id", "")).replace("LNR_", ""),
            "chem": r.get("cyc_type", ""),
            "t_ca": num(r.get("t_ca_start")), "t_lat": num(r.get("t_lat_start")),
            "seed": r.get("seed"),
            "dg_before": dg_b, "dg_after": dg_a,
            "ddg": None if (dg_b is None or dg_a is None) else round(dg_a - dg_b, 3),
            "outcome": "closed" if closed else ("open" if sat else "abstained"),
            "ring_A": num(r.get("ring_dist_A")),
            "declared": int(bool(declared)),
            "retention": num(r.get("contact_retention")),
            "rmsd": num(r.get("ca_rmsd_to_input_A")),
            "subs": r.get("n_substitutions"),
            "norelax": num(r.get("binder_rosetta_dG_separated_norelax")),
            "dsasa": num(r.get("binder_rosetta_dSASA_int")),
            "hbonds": num(r.get("binder_rosetta_hbonds_int")),
        })
    edits.sort(key=lambda e: (e["peptide"], e["chem"], e["t_ca"] or 0, e["t_lat"] or 0, e["seed"] or 0))

    meta = {
        "run": args.run_label or "unlabelled run",
        "checkpoint": args.checkpoint,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_edits": len(edits),
        "n_peptides": len({e["peptide"] for e in edits}),
        "n_before": len(before),
        "n_unpaired": sum(1 for e in edits if e["dg_before"] is None),
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(edits, separators=(",", ":")))
    html = html.replace("__META__", json.dumps(meta, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({meta['n_edits']} edits, {meta['n_peptides']} peptides, "
          f"{meta['n_before']} inputs scored)", flush=True)


TEMPLATE = r"""<title>Ring Cost Ledger</title>
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
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
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
.wrap { max-width: 1180px; margin: 0 auto; padding: 34px 24px 72px; display: flex;
        flex-direction: column; gap: 26px; }
h1 { font-family: var(--serif); font-weight: 600; font-size: 34px; line-height: 1.15;
     margin: 0; text-wrap: balance; letter-spacing: -.01em; }
h2 { font-family: var(--serif); font-weight: 600; font-size: 19px; margin: 0; }
.lede { color: var(--ink-2); max-width: 66ch; margin: 0; }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .13em;
           text-transform: uppercase; color: var(--muted); }

header { border-bottom: 2px solid var(--ink); padding-bottom: 20px;
         display: flex; flex-direction: column; gap: 12px; }
.prov { display: flex; flex-wrap: wrap; gap: 8px 28px; font-family: var(--mono);
        font-size: 12px; color: var(--muted); }
.prov b { color: var(--ink-2); font-weight: 500; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 14px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 3px;
        padding: 14px 16px 15px; display: flex; flex-direction: column; gap: 3px; }
.tile .v { font-family: var(--mono); font-size: 27px; font-weight: 500;
           font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.tile .k { font-size: 12.5px; color: var(--muted); }
.up { color: var(--worse); } .down { color: var(--better); }

.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 3px;
         box-shadow: var(--shadow); padding: 20px 22px 22px;
         display: flex; flex-direction: column; gap: 14px; }
.panel .head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
.panel .head p { margin: 0; color: var(--muted); font-size: 13px; }

.filters { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; }
.fgroup { display: flex; align-items: center; gap: 7px; }
.fgroup > span { font-family: var(--mono); font-size: 11px; letter-spacing: .1em;
                 text-transform: uppercase; color: var(--muted); }
button.chip { font: inherit; font-size: 13px; padding: 4px 11px; border-radius: 999px;
              border: 1px solid var(--line); background: var(--surface-2); color: var(--ink-2);
              cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
button.chip:hover { border-color: var(--muted); }
button.chip[aria-pressed="true"] { background: var(--ink); color: var(--surface);
                                   border-color: var(--ink); }
button.chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.swatch { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.chip[aria-pressed="true"] .swatch { box-shadow: 0 0 0 1.5px var(--surface); }

.chart { width: 100%; overflow-x: auto; }
svg { display: block; }
svg text { font-family: var(--mono); font-size: 11px; fill: var(--muted); }
svg text.row { font-family: var(--sans); font-size: 12px; fill: var(--ink-2); }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .axis { stroke: var(--line); stroke-width: 1; }
svg .mark { cursor: crosshair; }

.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 12.5px; color: var(--ink-2); }
.legend span.item { display: inline-flex; align-items: center; gap: 7px; }
.dotf, .doto, .dotb { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.doto { background: transparent; box-shadow: inset 0 0 0 2px currentColor; }
.dotb { background: var(--muted); }

.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
thead th { position: sticky; top: 0; background: var(--surface); z-index: 2;
           font-family: var(--mono); font-size: 11px; letter-spacing: .06em;
           text-transform: uppercase; color: var(--muted); font-weight: 500; cursor: pointer;
           border-bottom: 1px solid var(--ink-2); }
thead th:hover { color: var(--ink); }
tbody tr:hover { background: var(--surface-2); }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.tag { font-family: var(--mono); font-size: 11px; padding: 2px 7px; border-radius: 2px;
       border: 1px solid currentColor; }
.tag.closed { color: var(--better); } .tag.open { color: var(--worse); }
.tag.abstained { color: var(--muted); }
.pep { font-family: var(--mono); font-size: 12.5px; }

.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 18px; }
.note { border-left: 2px solid var(--accent); padding-left: 14px; }
.note h3 { font-size: 14px; margin: 0 0 4px; font-family: var(--sans); font-weight: 600; }
.note p { margin: 0; font-size: 13.5px; color: var(--ink-2); }
code { font-family: var(--mono); font-size: .92em; background: var(--surface-2);
       padding: 1px 5px; border-radius: 2px; }
#tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
       background: var(--ink); color: var(--bg); font-family: var(--mono); font-size: 11.5px;
       padding: 7px 9px; border-radius: 3px; z-index: 30; max-width: 280px; line-height: 1.5; }
.empty { color: var(--muted); font-size: 13.5px; padding: 18px 0; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Proteina&#8209;Complexa &middot; LNR linear&#8594;cyclic edits</div>
    <h1>Ring Cost Ledger</h1>
    <p class="lede">Rosetta interface energy of every SDEdit attempt, scored twice on the same
      complex: the peptide as the model received it, and the peptide after the edit.
      <code>dG_separated</code> does not compare across targets, so every judgement here is the
      paired move &mdash; &Delta;dG, in Rosetta energy units, on one receptor.</p>
    <div class="prov" id="prov"></div>
  </header>

  <section class="tiles" id="tiles"></section>

  <section class="panel">
    <div class="head">
      <h2>Filter the run</h2>
      <p>Every scored attempt is here, failures included &mdash; closed-versus-open is the comparison.</p>
    </div>
    <div class="filters" id="filters"></div>
  </section>

  <section class="panel">
    <div class="head">
      <h2>Input &rarr; edited, per peptide</h2>
      <p>Grey dot: the input. Coloured dot: the median of the selected attempts. Sorted by the move.</p>
    </div>
    <div class="chart" id="dumbbell"></div>
    <div class="legend" id="legend1"></div>
  </section>

  <section class="panel">
    <div class="head">
      <h2>What the ring cost, per attempt</h2>
      <p>&Delta;dG above zero means weaker binding than the input. Filled marks closed the requested ring.</p>
    </div>
    <div class="chart" id="strip"></div>
  </section>

  <section class="panel">
    <div class="head">
      <h2>The ledger</h2>
      <p>Click a column head to sort. <code>norelax</code> is the pre-relax interface energy &mdash; a
        large positive value means the edit landed in the receptor with clashes relax had to resolve.</p>
    </div>
    <div class="tablewrap"><table id="tbl">
      <thead><tr>
        <th class="l" data-k="peptide">Peptide</th>
        <th class="l" data-k="chem">Chemistry</th>
        <th data-k="t_ca">t_ca</th>
        <th data-k="t_lat">t_lat</th>
        <th data-k="seed">Seed</th>
        <th class="l" data-k="outcome">Ring</th>
        <th data-k="ring_A">Gap &Aring;</th>
        <th data-k="dg_before">dG before</th>
        <th data-k="dg_after">dG after</th>
        <th data-k="ddg">&Delta;dG</th>
        <th data-k="norelax">norelax</th>
        <th data-k="retention">Retention</th>
        <th data-k="rmsd">CA RMSD</th>
        <th data-k="subs">Subs</th>
      </tr></thead>
      <tbody></tbody>
    </table></div>
    <div class="empty" id="empty" hidden>No attempt matches these filters.</div>
  </section>

  <section class="notes">
    <div class="note">
      <h3>An open ring is scored as open</h3>
      <p>The closing bond is declared to Rosetta only when the geometry actually closed. Declaring
        it on an open ring makes FastRelax drag the gap shut, and the energy that comes back
        describes a structure the model never produced. The <code>Ring</code> column says which
        case each row is.</p>
    </div>
    <div class="note">
      <h3>Isopeptide bonds cannot be declared</h3>
      <p>Rosetta has patches for a stripped terminus and for a disulfide, but not for an arbitrary
        Lys&#8211;NZ to Asp side-chain bond. Isopeptide rows are scored as ring-shaped but formally
        open peptides &mdash; a real gap, reported rather than hidden.</p>
    </div>
    <div class="note">
      <h3>&ldquo;Before&rdquo; is the arm&rsquo;s own input</h3>
      <p>For the control arm that is the staged crystal peptide. For the soft-closure
        <em>projected</em> arm it is the projected pose, because that is what the model was handed
        &mdash; the crystal baseline lives in the control arm.</p>
    </div>
    <div class="note">
      <h3>Abstentions are not failures</h3>
      <p>An abstained attempt is one where the model proposed a different chemistry than the one
        requested, so its ring was never measured. It carries an energy but no closure verdict.</p>
    </div>
  </section>
</div>
<div id="tip" role="status"></div>

<script>
const EDITS = __DATA__;
const META  = __META__;
const CHEMS = ["mainchain", "disulfide", "isopeptide"];
const OUTCOMES = ["closed", "open", "abstained"];
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const chemColor = c => cssv("--" + c) || cssv("--muted");
const fmt = (v, d = 1) => (v === null || v === undefined) ? "—" : v.toFixed(d);
const sgn = v => (v === null || v === undefined) ? "—" : (v > 0 ? "+" : "") + v.toFixed(1);
const med = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

let state = { chems: new Set(CHEMS), outcomes: new Set(OUTCOMES), sort: "ddg", dir: -1 };
const sel = () => EDITS.filter(e => state.chems.has(e.chem) && state.outcomes.has(e.outcome));

/* ---------- provenance + headline tiles ---------- */
document.getElementById("prov").innerHTML = [
  ["run", META.run], ["checkpoint", META.checkpoint || "—"],
  ["inputs scored", META.n_before], ["attempts scored", META.n_edits],
  ["built", META.built],
].map(([k, v]) => `<span><b>${k}</b> ${v}</span>`).join("");

function tiles(rows) {
  const paired = rows.filter(e => e.ddg !== null);
  const dd = paired.map(e => e.ddg);
  const closed = paired.filter(e => e.outcome === "closed").map(e => e.ddg);
  const open = paired.filter(e => e.outcome !== "closed").map(e => e.ddg);
  const worse = dd.length ? dd.filter(v => v > 0).length / dd.length : null;
  const t = [
    { k: "attempts in view", v: rows.length, cls: "", d: 0, raw: true },
    { k: "median ΔdG, all", v: med(dd), cls: med(dd) > 0 ? "up" : "down" },
    { k: "median ΔdG, ring closed", v: med(closed), cls: med(closed) > 0 ? "up" : "down" },
    { k: "median ΔdG, not closed", v: med(open), cls: med(open) > 0 ? "up" : "down" },
    { k: "attempts binding worse", v: worse === null ? null : worse * 100, cls: "", suf: "%" },
  ];
  document.getElementById("tiles").innerHTML = t.map(x => {
    const val = x.raw ? x.v : (x.v === null ? "—" : (x.suf ? Math.round(x.v) + x.suf : sgn(x.v)));
    return `<div class="tile"><div class="v ${x.v === null ? "" : x.cls}">${val}</div>
            <div class="k">${x.k}</div></div>`;
  }).join("");
}

/* ---------- filters ---------- */
function buildFilters() {
  const box = document.getElementById("filters");
  const chem = CHEMS.filter(c => EDITS.some(e => e.chem === c));
  const out = OUTCOMES.filter(o => EDITS.some(e => e.outcome === o));
  box.innerHTML =
    `<div class="fgroup"><span>chemistry</span>${chem.map(c =>
      `<button class="chip" data-t="chem" data-v="${c}" aria-pressed="true">
        <i class="swatch" style="background:${chemColor(c)}"></i>${c}</button>`).join("")}</div>
     <div class="fgroup"><span>ring</span>${out.map(o =>
      `<button class="chip" data-t="out" data-v="${o}" aria-pressed="true">${
        o === "open" ? "not closed" : o}</button>`).join("")}</div>`;
  box.querySelectorAll("button.chip").forEach(b => b.addEventListener("click", () => {
    const set = b.dataset.t === "chem" ? state.chems : state.outcomes;
    const v = b.dataset.v;
    if (set.has(v) && set.size > 1) set.delete(v); else set.add(v);
    b.setAttribute("aria-pressed", set.has(v));
    render();
  }));
}

/* ---------- tooltip ---------- */
const tip = document.getElementById("tip");
function bindTip(el, html) {
  el.addEventListener("pointerenter", ev => {
    tip.innerHTML = html; tip.style.opacity = 1;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - 300) + "px";
    tip.style.top = (ev.clientY + 16) + "px";
  });
  el.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
}

const NS = "http://www.w3.org/2000/svg";
const mk = (n, a) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

/* ---------- chart 1: per-peptide dumbbell ---------- */
function dumbbell(rows) {
  const host = document.getElementById("dumbbell");
  host.innerHTML = "";
  const groups = new Map();
  rows.filter(e => e.ddg !== null).forEach(e => {
    const k = e.peptide + "|" + e.chem;
    if (!groups.has(k)) groups.set(k, { peptide: e.peptide, chem: e.chem, before: e.dg_before,
                                        after: [], closed: 0, n: 0 });
    const g = groups.get(k);
    g.after.push(e.dg_after); g.n++; if (e.outcome === "closed") g.closed++;
  });
  const g = [...groups.values()].map(x => ({ ...x, aft: med(x.after) }))
    .map(x => ({ ...x, move: x.aft - x.before })).sort((a, b) => b.move - a.move);
  if (!g.length) { host.innerHTML = '<p class="empty">Nothing paired to draw.</p>'; return; }

  const rowH = 26, padT = 26, padB = 42, padL = 108, padR = 62;
  const W = Math.max(560, host.clientWidth || 900), H = padT + padB + g.length * rowH;
  const vals = g.flatMap(x => [x.before, x.aft]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const span = (hi - lo) || 10; lo -= span * 0.08; hi += span * 0.08;
  const x = v => padL + (v - lo) / (hi - lo) * (W - padL - padR);
  const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H,
                          role: "img", "aria-label": "Interface energy before and after, per peptide" });

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (hi - lo) * i / ticks;
    svg.appendChild(mk("line", { class: "grid", x1: x(v), x2: x(v), y1: padT - 8, y2: H - padB + 4 }));
    const t = mk("text", { x: x(v), y: H - padB + 20, "text-anchor": "middle" });
    t.textContent = v.toFixed(0); svg.appendChild(t);
  }
  const cap = mk("text", { x: padL, y: H - 8, "text-anchor": "start" });
  cap.textContent = "Rosetta dG_separated (REU) → left is stronger binding";
  svg.appendChild(cap);

  g.forEach((r, i) => {
    const y = padT + i * rowH + rowH / 2, col = chemColor(r.chem);
    const lab = mk("text", { x: padL - 12, y: y + 4, "text-anchor": "end", class: "row" });
    lab.textContent = r.peptide; svg.appendChild(lab);
    svg.appendChild(mk("line", { x1: x(r.before), x2: x(r.aft), y1: y, y2: y, stroke: col,
                                 "stroke-width": 2.5, "stroke-linecap": "round", opacity: .5 }));
    svg.appendChild(mk("circle", { cx: x(r.before), cy: y, r: 5, fill: cssv("--muted"),
                                   stroke: cssv("--surface"), "stroke-width": 1.5 }));
    const dot = mk("circle", { class: "mark", cx: x(r.aft), cy: y, r: 5.5, stroke: col,
                               "stroke-width": 2, fill: r.closed ? col : cssv("--surface") });
    svg.appendChild(dot);
    bindTip(dot, `<b>${r.peptide}</b> · ${r.chem}<br>input ${r.before.toFixed(1)} REU<br>` +
                 `edited ${r.aft.toFixed(1)} REU (median of ${r.n})<br>` +
                 `Δ ${sgn(r.move)} REU<br>${r.closed}/${r.n} closed the ring`);
    const d = mk("text", { x: Math.max(x(r.before), x(r.aft)) + 10, y: y + 4, "text-anchor": "start" });
    d.textContent = sgn(r.move); d.setAttribute("fill", r.move > 0 ? cssv("--worse") : cssv("--better"));
    svg.appendChild(d);
  });
  host.appendChild(svg);

  document.getElementById("legend1").innerHTML =
    `<span class="item"><i class="dotb"></i>input (as given to the model)</span>` +
    [...new Set(g.map(r => r.chem))].map(c =>
      `<span class="item" style="color:${chemColor(c)}"><i class="dotf" style="background:${chemColor(c)}"></i>
        <span style="color:var(--ink-2)">${c}, ring closed</span></span>
       <span class="item" style="color:${chemColor(c)}"><i class="doto"></i>
        <span style="color:var(--ink-2)">${c}, not closed</span></span>`).join("");
}

/* ---------- chart 2: per-attempt delta strip ---------- */
function strip(rows) {
  const host = document.getElementById("strip");
  host.innerHTML = "";
  const data = rows.filter(e => e.ddg !== null);
  if (!data.length) { host.innerHTML = '<p class="empty">Nothing paired to draw.</p>'; return; }
  const chems = CHEMS.filter(c => data.some(e => e.chem === c));
  const padT = 22, padB = 40, padL = 62, padR = 96;
  const W = Math.max(560, host.clientWidth || 900), H = 330;
  const vals = data.map(e => e.ddg);
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const span = (hi - lo) || 10; lo -= span * .08; hi += span * .08;
  const y = v => padT + (hi - v) / (hi - lo) * (H - padT - padB);
  const colW = (W - padL - padR) / chems.length;
  const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H,
                          role: "img", "aria-label": "Change in interface energy per attempt" });

  for (let i = 0; i <= 5; i++) {
    const v = lo + (hi - lo) * i / 5;
    svg.appendChild(mk("line", { class: "grid", x1: padL - 12, x2: W - padR, y1: y(v), y2: y(v) }));
    const t = mk("text", { x: padL - 18, y: y(v) + 4, "text-anchor": "end" });
    t.textContent = v.toFixed(0); svg.appendChild(t);
  }
  svg.appendChild(mk("line", { x1: padL - 12, x2: W - padR, y1: y(0), y2: y(0),
                               stroke: cssv("--ink-2"), "stroke-width": 1.5,
                               "stroke-dasharray": "5 4" }));
  const zl = mk("text", { x: W - padR + 8, y: y(0) + 4, "text-anchor": "start" });
  zl.textContent = "no change"; svg.appendChild(zl);
  const yl = mk("text", { x: padL - 18, y: padT - 8, "text-anchor": "end" });
  yl.textContent = "ΔdG (REU)"; svg.appendChild(yl);

  let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  chems.forEach((c, ci) => {
    const cx = padL + colW * (ci + .5), col = chemColor(c);
    const g = data.filter(e => e.chem === c);
    g.forEach(e => {
      const px = cx + (rnd() - .5) * Math.min(colW * .55, 90);
      const dot = mk("circle", { class: "mark", cx: px, cy: y(e.ddg), r: 5, stroke: col,
        "stroke-width": 1.8, fill: e.outcome === "closed" ? col : cssv("--surface"), opacity: .92 });
      svg.appendChild(dot);
      bindTip(dot, `<b>${e.peptide}</b> · ${e.chem}<br>t_ca ${e.t_ca} · t_lat ${e.t_lat}
        · seed ${e.seed}<br>ΔdG ${sgn(e.ddg)} REU<br>ring ${e.outcome}${
        e.ring_A !== null ? " · gap " + fmt(e.ring_A, 1) + " Å" : ""}<br>retention ${fmt(e.retention, 2)}`);
    });
    const m = med(g.map(e => e.ddg));
    svg.appendChild(mk("line", { x1: cx - 46, x2: cx + 46, y1: y(m), y2: y(m),
                                 stroke: cssv("--ink"), "stroke-width": 2.5, "stroke-linecap": "round" }));
    const ml = mk("text", { x: cx + 52, y: y(m) + 4, "text-anchor": "start" });
    ml.textContent = "median " + sgn(m); svg.appendChild(ml);
    const cl = mk("text", { x: cx, y: H - padB + 22, "text-anchor": "middle", class: "row" });
    cl.textContent = `${c} (${g.length})`; svg.appendChild(cl);
  });
  host.appendChild(svg);
}

/* ---------- ledger ---------- */
function table(rows) {
  const k = state.sort, dir = state.dir;
  const sorted = [...rows].sort((a, b) => {
    const x = a[k], y = b[k];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
  });
  const body = document.querySelector("#tbl tbody");
  body.innerHTML = sorted.map(e => `<tr>
    <td class="l pep">${e.peptide}</td>
    <td class="l"><span style="color:${chemColor(e.chem)}">●</span> ${e.chem}</td>
    <td class="num">${fmt(e.t_ca, 1)}</td>
    <td class="num">${fmt(e.t_lat, 1)}</td>
    <td class="num">${e.seed ?? "—"}</td>
    <td class="l"><span class="tag ${e.outcome}">${e.outcome === "open" ? "not closed" : e.outcome}</span></td>
    <td class="num">${fmt(e.ring_A, 1)}</td>
    <td class="num">${fmt(e.dg_before, 1)}</td>
    <td class="num">${fmt(e.dg_after, 1)}</td>
    <td class="num ${e.ddg === null ? "" : (e.ddg > 0 ? "up" : "down")}">${sgn(e.ddg)}</td>
    <td class="num">${fmt(e.norelax, 0)}</td>
    <td class="num">${fmt(e.retention, 2)}</td>
    <td class="num">${fmt(e.rmsd, 2)}</td>
    <td class="num">${e.subs ?? "—"}</td></tr>`).join("");
  document.getElementById("empty").hidden = rows.length > 0;
}
document.querySelectorAll("#tbl thead th").forEach(th => th.addEventListener("click", () => {
  const k = th.dataset.k;
  state.dir = state.sort === k ? -state.dir : (k === "peptide" || k === "chem" ? 1 : -1);
  state.sort = k; render();
}));

function render() { const rows = sel(); tiles(rows); dumbbell(rows); strip(rows); table(rows); }
buildFilters();
render();
addEventListener("resize", () => { dumbbell(sel()); strip(sel()); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
"""


if __name__ == "__main__":
    main()
