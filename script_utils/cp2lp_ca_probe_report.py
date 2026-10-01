"""Table the CA-spacing probe's JSON summaries. Reads only what is on disk -- no GPU.

The probe answers "why is the chain not intact" per run; this collects the runs into the
three tables that carry the argument:

1. THE DOSE RESPONSE. Chain damage against edit size, with the incoherence fraction beside
   it. The claim the table has to support or refute is that incoherence is a CONSTANT of the
   parameterisation -- if it is, the adversary is not teaching bad geometry, it is only
   editing harder, and no reweighting changes the trade.
2. THE APPORTIONMENT. The 2x2 over {source, edited} CA x {source, edited} latents, scored by
   the same all-bonds test as the export gate, which says which decoder input carries the
   damage.
3. THE CA TRACE against its two native constraints -- CA-CA length and CA-CA-CA bend angle.
   A projection that fixes only the first is visible here as an unrepaired second.

Native yardsticks quoted in the output are MEASURED, from the 101 complexes in
`calibration_native/` (peptide chains only, which is what the editor edits):
CA-CA 3.838 +- 0.054 A over 1,233 bonds with nothing above 4.00 A; bend angle
97.3 +- 16.2 deg over 1,132 triples with nothing below 74.5 deg.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

# MEASURED, peptide chains of calibration_native/*/processed/*.pdb. See module docstring.
NATIVE = {
    "ca_ca_median_A": 3.838,
    "ca_ca_sd_A": 0.054,
    "ca_ca_max_A": 3.980,
    "bend_median_deg": 97.3,
    "bend_min_deg": 74.5,
    "bend_p99_deg": 146.7,
}

GRID_ORDER = [
    ("src_ca__src_z", "source CA  + source z   (AE round trip = ceiling)"),
    ("src_ca__edit_z", "source CA  + EDITED z   (latents alone)"),
    ("edit_ca__src_z", "EDITED CA  + source z   (CA alone)"),
    ("edit_ca__edit_z", "EDITED CA  + EDITED z   (the actual output)"),
    ("proj_ca__edit_z", "projected CA + EDITED z (bond lengths repaired)"),
    ("proj_ca__src_z", "projected CA + source z"),
]


def load(d: str) -> list[tuple[str, dict]]:
    out = []
    for f in sorted(glob.glob(str(Path(d) / "*.json"))):
        try:
            out.append((Path(f).stem, json.load(open(f))))
        except Exception as e:  # a half-written file is a fact, not something to average past
            print(f"REFUSED {f}: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of probe JSON summaries")
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    runs = load(args.dir)
    if not runs:
        print(f"no probe JSON under {args.dir}")
        return 1
    L: list[str] = []
    def p(s: str = "") -> None:
        print(s)
        L.append(s)

    p("## 1. Dose response: chain damage against edit size")
    p()
    p("| run | g | sigma_c nm | incoherence | dCA-CA nm | in native sd | CA-CA sd nm | outside window | chain_intact |")
    p("|---|---|---|---|---|---|---|---|---|")
    for name, d in sorted(runs, key=lambda kv: kv[1]["residual_decomposition"]["sigma_component_nm"]):
        r = d["residual_decomposition"]
        inc = r["incoherence_frac"]
        p(
            f"| {name} | {d.get('g_updates')} | {r['sigma_component_nm']:.4f} | "
            f"{('%.3f' % inc) if inc else 'n/a'} | {r['ds_exact_rms_nm']:.4f} | "
            f"{r['ca_ca_change_in_native_sd']:.1f} | {d['ca_ca_nm']['edited']['sd']:.4f} | "
            f"{d['ca_ca_outside_native_window_frac']['edited']:.1%} | "
            f"{d['counterfactual']['chain_intact_frac']:.4f} |"
        )
    incs = [d["residual_decomposition"]["incoherence_frac"] for _, d in runs
            if d["residual_decomposition"]["incoherence_frac"]]
    if len(incs) > 1:
        p()
        p(f"incoherence across {len(incs)} edited runs: min {min(incs):.3f} max {max(incs):.3f} "
          f"spread {max(incs) - min(incs):.3f}")

    p()
    p("## 2. Apportionment: which decoder input carries the damage")
    for name, d in runs:
        g = d.get("decode_grid")
        if not g:
            continue
        ceil = g.get("src_ca__src_z", {}).get("chain_intact_frac")
        p()
        p(f"**{name}** (g={d.get('g_updates')})")
        p()
        p("| decode | chain_intact | cn_dev median nm | cn_dev p99 nm | cost vs ceiling |")
        p("|---|---|---|---|---|")
        for k, lab in GRID_ORDER:
            if k not in g:
                continue
            v = g[k]
            cost = "" if ceil in (None, 0) else f"{(ceil - v['chain_intact_frac']) / ceil:.1%}"
            p(f"| {lab} | {v['chain_intact_frac']:.4f} | {v['cn_dev_median_nm']:.4f} | "
              f"{v['cn_dev_p99_nm']:.4f} | {cost} |")

    p()
    p("## 3. The CA trace against its two native constraints")
    p()
    p(f"native (measured): CA-CA {NATIVE['ca_ca_median_A']:.3f} +- {NATIVE['ca_ca_sd_A']:.3f} A, "
      f"none above {NATIVE['ca_ca_max_A']:.2f}; bend {NATIVE['bend_median_deg']:.1f} deg, "
      f"none below {NATIVE['bend_min_deg']:.1f}")
    for name, d in runs:
        b = d.get("bend_angle_deg")
        if not b:
            continue
        p()
        p(f"**{name}**")
        p()
        p("| track | CA-CA med nm | CA-CA sd nm | bend min | bend p1 | bend med | bend p99 | bend max |")
        p("|---|---|---|---|---|---|---|---|")
        for k in ("source", "edited", "projected"):
            c = d["ca_ca_nm"][k]
            p(f"| {k} | {c['median']:.4f} | {c['sd']:.5f} | {b[k]['min']:.1f} | {b[k]['p1']:.1f} | "
              f"{b[k]['median']:.1f} | {b[k]['p99']:.1f} | {b[k]['max']:.1f} |")

    p()
    p("## All-bonds amplification")
    p()
    p("`chain_intact` requires EVERY bond inside tolerance, so it is the per-bond pass rate")
    p("raised to the bond count. At ~9.25 bonds per peptide:")
    p()
    p("| per-bond pass | peptides intact |")
    p("|---|---|")
    nb = sum(d["n_bonds"] for _, d in runs) / max(1, sum(d["n_examples"] for _, d in runs))
    for q in (0.52, 0.70, 0.90, 0.95, 0.99, 0.999):
        p(f"| {q:.3f} | {q ** nb:.4f} |")
    p()
    p(f"reaching 0.95 of peptides needs a per-bond pass rate of {0.95 ** (1 / nb):.4f}.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
