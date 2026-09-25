"""Report for pre-build check 3 -- profile, reference distributions, adversary, OOD.

Reads `reference_distributions.json` and `adversary_results.json` and writes the markdown.
Computes nothing new; every number here already exists in one of those artifacts, so the
report can be redrawn cheaply when the wording needs to change.

The one judgement it makes is about what the numbers LICENSE, and it makes it the same way
every time:

  * the positive control gates everything -- if the harness cannot detect a difference it
    should detect, no other AUC in the file is interpretable;
  * the staging control caps everything -- if staging alone separates the sets, the main
    AUC is partly measuring the staging and the report says by how much rather than
    quoting the headline unqualified.

Runs in `.venv`.  CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script_utils import precheck_config  # noqa: E402
from script_utils.peptide_profile import (  # noqa: E402
    FEATURES,
    HELDOUT_FEATURES,
    TUNING_FEATURES,
)


def _auc(block: dict, key: str) -> float:
    return float((block.get(key) or {}).get("auc", float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    precheck_config.load(args.config)          # fail fast on a bad config
    out_dir = Path(args.out_dir)
    ref = json.loads((out_dir / "reference_distributions.json").read_text())
    adv = json.loads((out_dir / "adversary_results.json").read_text())

    L: list[str] = []
    A = L.append
    A("# Pre-build check 3 -- feature profile, reference distributions, adversary\n")

    A("## 1. What was profiled\n")
    A(f"Uniform receptor crop: **{ref.get('crop_radius_A')} A**, applied to every set.\n")
    A("| set | n | role |")
    A("|---|---|---|")
    for name, e in sorted(ref["sets"].items()):
        role = "reference only (development data)" if name in ref.get("reference_only", []) else "comparison"
        A(f"| {name} | {e['n']} | {role} |")
    A("")
    A("LNR is marked reference-only because it has already influenced this repository's")
    A("preprocessing and sampling decisions. The serialized artifact carries that flag so a")
    A("corruption sampler cannot draw target values from it by accident.\n")

    A("\n## 2. Reference distributions (median, with 5th-95th in brackets)\n")
    sets = sorted(ref["sets"])
    A("| feature | " + " | ".join(sets) + " |")
    A("|---" * (len(sets) + 1) + "|")
    for feat in FEATURES:
        cells = []
        for s in sets:
            f = ref["sets"][s]["features"].get(feat, {})
            if not f.get("n"):
                cells.append("-")
                continue
            cells.append(f"{f['q']['50']:.3f} [{f['q']['5']:.2f}, {f['q']['95']:.2f}]")
        A(f"| `{feat}` | " + " | ".join(cells) + " |")

    A("\n## 3. Staging, before and after the uniform crop\n")
    A("The confound: PepBench receptors are whole chains and CPSea receptors are pocket")
    A("crops, and four profile features are sensitive to that. Cropping every set with the")
    A("same operation removes the asymmetry by construction rather than by staging one set")
    A("to match another.\n")
    A("| set | receptor length | segments | max run |")
    A("|---|---|---|---|")
    for s in sets:
        st = ref["sets"][s].get("staging", {})
        def med(k):
            return f"{st[k]['median']:.0f}" if k in st else "-"
        A(f"| {s} | {med('receptor_length')} | {med('receptor_n_segments')} | {med('receptor_max_run')} |")

    pc = adv["positive_control"]
    passed = pc.get("passed")
    A("\n## 4. Positive control\n")
    A(f"**{pc['pair'][0]} vs {pc['pair'][1]}** "
      f"(n = {pc.get('n_a')} / {pc.get('n_b')}), classifier `{pc.get('classifier')}`, "
      f"grouped on `cluster_id`.\n")
    A("| feature block | AUC |")
    A("|---|---|")
    A(f"| all {len(FEATURES)} features | **{_auc(pc, 'auc_all'):.3f}** |")
    A(f"| tuning half ({len(TUNING_FEATURES)}) | {_auc(pc, 'auc_tuning'):.3f} |")
    A(f"| held-out half ({len(HELDOUT_FEATURES)}) | {_auc(pc, 'auc_heldout'):.3f} |")
    A(f"| staging control (post-crop) | {_auc(pc, 'auc_staging_control'):.3f} |")
    if "auc_staging_control_precrop" in pc:
        A(f"| staging control (pre-crop) | {_auc(pc, 'auc_staging_control_precrop'):.3f} |")
    A("")
    A(f"Threshold {pc.get('min_auc')}: **{'PASS' if passed else 'FAIL'}**. "
      + ("The harness detects a difference it should detect, so an AUC near 0.5 on a later "
         "comparison is evidence rather than an artifact of a broken instrument.\n"
         if passed else
         "The harness CANNOT detect a difference it should detect. Every other AUC in this "
         "report is uninterpretable until this passes.\n"))

    staging = _auc(pc, "auc_staging_control")
    if np.isfinite(staging) and staging > 0.65:
        pre = _auc(pc, "auc_staging_control_precrop")
        A(f"> **Caveat, and it is a real one.** Staging features ALONE reach "
          f"{staging:.3f} after the uniform crop"
          + (f" (down from {pre:.3f} before it)" if np.isfinite(pre) else "")
          + ". The crop removed most of the staging signal but not all of it, so the")
        A("> headline AUC is partly attributable to how the records were staged rather than")
        A("> to peptide conformation. Read the held-out AUC against this number, not against")
        A("> 0.5. Remedying it is a design decision and is out of scope for this check.\n")

    if pc.get("importances"):
        A("\n### What the classifier used\n")
        A("Permutation importance (drop in AUC when the feature is shuffled), top 8.\n")
        A("| feature | importance |")
        A("|---|---|")
        for k, v in sorted(pc["importances"].items(), key=lambda kv: -kv[1])[:8]:
            A(f"| `{k}` | {v:+.4f} |")

    if pc.get("by_length"):
        A("\n### Stratified by peptide length (held-out features only)\n")
        A("Pooling would hide this: PepBench's median peptide is 9 residues against CPSea's")
        A("13, so a pooled AUC partly measures length.\n")
        A("| length bin | n | AUC (held-out) |")
        A("|---|---|---|")
        for b in pc["by_length"]:
            a = _auc(b, "auc_heldout")
            note = f" — {b['note']}" if b.get("note") else ""
            A(f"| {b['bin']} | {b['n']} | "
              + (f"{a:.3f}" if np.isfinite(a) else "-") + f"{note} |")
        worst = [b for b in pc["by_length"]
                 if np.isfinite(_auc(b, "auc_heldout")) and _auc(b, "auc_heldout") < 0.6]
        if worst:
            A("")
            A("> Bins at or below chance: "
              + ", ".join(f"`{b['bin']}` ({_auc(b, 'auc_heldout'):.3f}, n={b['n']})" for b in worst)
              + ". In these bins the held-out features do not separate the sets at all.")
            A("> Reported as measured; whether that is good news depends on which set the")
            A("> synthetic states are later supposed to resemble.\n")

    if adv.get("pairs"):
        A("\n## 5. Other comparisons\n")
        A("| pair | n | all | tuning | held-out | staging control |")
        A("|---|---|---|---|---|---|")
        for p in adv["pairs"]:
            A(f"| {p['pair'][0]} vs {p['pair'][1]} | {p.get('n_a')}/{p.get('n_b')} | "
              f"{_auc(p, 'auc_all'):.3f} | {_auc(p, 'auc_tuning'):.3f} | "
              f"{_auc(p, 'auc_heldout'):.3f} | {_auc(p, 'auc_staging_control'):.3f} |")
        A("")
        for p in adv["pairs"]:
            s, a = _auc(p, "auc_staging_control"), _auc(p, "auc_all")
            if np.isfinite(s) and np.isfinite(a) and s >= a - 0.05:
                A(f"> `{p['pair'][0]}` vs `{p['pair'][1]}`: the staging control ({s:.3f}) is as")
                A(f"> high as the full-feature AUC ({a:.3f}). This separation is staging, not")
                A("> conformation, and should not be read as a distributional difference.\n")

    ood = adv.get("ood") or {}
    A("\n## 6. OOD score\n")
    if "note" in ood:
        A(f"Not fitted: {ood['note']}\n")
    else:
        A(f"Gaussian KDE on the standardized feature vector, fitted on **{ood['fit_on']}** only")
        A(f"(n = {ood['n_fit']}, bandwidth {ood['bandwidth']:.3f} by Scott's rule). Fitting on")
        A("the union would blunt the signal the score exists to give.\n")
        A("Per-row scores are in `ood_scores.csv` as a log-density and as a percentile")
        A("against the fit set, so an LNR input can carry one alongside its prediction.\n")
        A("| set | median percentile vs fit set |")
        A("|---|---|")
        for lab, v in sorted((ood.get("by_label_median_pct") or {}).items()):
            A(f"| {lab} | {v:.3f} |")
        A("")
        A("A median near 0.5 means the set sits inside the fit distribution; near 0.0 means")
        A("essentially every input is below its 1st percentile.\n")

    A("\n## 7. What is NOT in this check\n")
    A("- **3d, the learned-representation adversary, is not built.** It was specified as a")
    A("  second stage after 3a-3c work, and 3a-3c now work. It remains open.")
    A("- No synthetic open states exist yet, so the adversary has only been run against")
    A("  real sets. That is what it was specified to do at this stage: build the apparatus")
    A("  and compute the real-side reference.")
    A("- Nothing here was tuned to pass. The thresholds in the config were set before the")
    A("  numbers were seen, and the staging caveat above is reported rather than fixed.\n")

    (out_dir / "PRECHECK_TASK3_REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out_dir}/PRECHECK_TASK3_REPORT.md")


if __name__ == "__main__":
    main()
