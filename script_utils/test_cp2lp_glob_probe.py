"""CPU tests for the glob-dependence probe.

The probe's whole value is that a collapse in `d_acc` means dependence, so the tests here
target the two ways that inference could be wrong without erroring:

* a per-class fill, which would REPLACE a continuous separator with a perfect categorical
  one and report the opposite verdict;
* a column reordering, which would ablate the wrong feature while every number still looks
  plausible.

Run: python script_utils/test_cp2lp_glob_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proteinfoundation.cp2lp.discriminator import (  # noqa: E402
    GLOB_COLUMNS,
    GLOB_INTERFACE_COLS,
    GLOB_OPENNESS_COLS,
    discriminator_accuracy,
    glob_ablated_pair,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))


def feats(glob: torch.Tensor, n_res: int = 6) -> dict[str, torch.Tensor]:
    b = glob.shape[0]
    return {
        "glob": glob,
        "geom": torch.zeros(b, n_res, 3),
        "iface": torch.zeros(b, n_res, 2),
        "mask": torch.ones(b, n_res, dtype=torch.bool),
    }


def test_fill_is_shared_across_classes() -> None:
    print("fill is pooled over both classes")
    # Perfectly separated gaps: reals high, fakes low.
    real = feats(torch.tensor([[2.1, 1.0, 5.0, 0.4], [2.2, 1.1, 6.0, 0.4]]))
    fake = feats(torch.tensor([[1.0, 0.6, 5.0, 0.4], [1.1, 0.7, 6.0, 0.4]]))
    r_ab, f_ab, fill = glob_ablated_pair(real, fake, GLOB_OPENNESS_COLS)

    idx = list(GLOB_OPENNESS_COLS)
    check(
        "both classes receive the identical fill",
        torch.allclose(r_ab["glob"][:, idx], f_ab["glob"][:, idx]),
        f"real={r_ab['glob'][:, idx]} fake={f_ab['glob'][:, idx]}",
    )
    # The decisive property: after ablation the two classes are INDISTINGUISHABLE on those
    # columns. A per-class median would also make each class constant -- but at two
    # DIFFERENT constants, which is a perfect separator. This asserts the difference.
    check(
        "ablated columns carry zero label information",
        float((r_ab["glob"][:, idx] - f_ab["glob"][:, idx]).abs().max()) == 0.0,
    )
    pooled_median = torch.cat([real["glob"][:, idx], fake["glob"][:, idx]]).median(dim=0).values
    check("fill is the pooled median", torch.allclose(fill, pooled_median))
    check(
        "a per-class fill would NOT satisfy the above (guard is meaningful)",
        not torch.allclose(
            real["glob"][:, idx].median(dim=0).values,
            fake["glob"][:, idx].median(dim=0).values,
        ),
    )


def test_untouched_columns_and_blocks_survive() -> None:
    print("the intervention is confined to the named columns")
    real = feats(torch.tensor([[2.1, 1.0, 5.0, 0.44], [2.2, 1.1, 6.0, 0.41]]))
    fake = feats(torch.tensor([[1.0, 0.6, 7.0, 0.38], [1.1, 0.7, 8.0, 0.39]]))
    r_ab, f_ab, _ = glob_ablated_pair(real, fake, GLOB_OPENNESS_COLS)
    check(
        "contact_count / mean_min_heavy untouched in reals",
        torch.allclose(r_ab["glob"][:, 2:], real["glob"][:, 2:]),
    )
    check(
        "contact_count / mean_min_heavy untouched in fakes",
        torch.allclose(f_ab["glob"][:, 2:], fake["glob"][:, 2:]),
    )
    check("width is preserved (held, not dropped)", r_ab["glob"].shape == real["glob"].shape)
    check("geom block is passed through", r_ab["geom"] is real["geom"])
    check("input dict is not mutated", torch.allclose(real["glob"][:, 0], torch.tensor([2.1, 2.2])))


def test_empty_ablation_is_the_identity() -> None:
    print("the empty column set is an exact baseline")
    real = feats(torch.tensor([[2.1, 1.0, 5.0, 0.4]]))
    fake = feats(torch.tensor([[1.0, 0.6, 5.0, 0.4]]))
    r_ab, f_ab, fill = glob_ablated_pair(real, fake, ())
    # The probe routes its BASELINE through this path so that nothing but the held columns
    # can differ between baseline and ablation. If `()` were not the identity, every delta
    # in the report would carry an unexplained offset.
    check("reals unchanged", torch.allclose(r_ab["glob"], real["glob"]))
    check("fakes unchanged", torch.allclose(f_ab["glob"], fake["glob"]))
    check("empty fill", fill.numel() == 0)


def test_single_column_variants() -> None:
    print("single-column variants hold only their own column")
    real = feats(torch.tensor([[2.1, 1.0, 5.0, 0.4], [2.3, 1.2, 5.0, 0.4]]))
    fake = feats(torch.tensor([[1.0, 0.6, 5.0, 0.4], [1.2, 0.8, 5.0, 0.4]]))
    r_ab, f_ab, _ = glob_ablated_pair(real, fake, (0,))
    check("gap equalised", torch.allclose(r_ab["glob"][:, 0], f_ab["glob"][:, 0]))
    check(
        "Rg still separates",
        float(r_ab["glob"][:, 1].min()) > float(f_ab["glob"][:, 1].max()),
    )


def test_column_names_match_the_builder() -> None:
    print("the named columns match build_features")
    src = (
        Path(__file__).resolve().parents[1]
        / "src/proteinfoundation/cp2lp/discriminator.py"
    ).read_text()
    # `glob` is built by a torch.stack of four calls, in order. A reorder there with
    # GLOB_COLUMNS left alone would make every label in the report wrong while the
    # arithmetic stayed valid, so this pins the order to the source.
    block = src.split("# ---- global")[1].split("]")[0]
    order = [fn for fn in ("terminal_gap_nm", "radius_of_gyration_nm") if fn in block]
    check("terminal gap is glob column 0", block.index("terminal_gap_nm") < block.index("radius_of_gyration_nm"))
    check("both openness functions are in the global block", len(order) == 2)
    check("GLOB_COLUMNS has 4 entries", len(GLOB_COLUMNS) == 4, str(GLOB_COLUMNS))
    check("openness cols are (0, 1)", tuple(GLOB_OPENNESS_COLS) == (0, 1))
    check(
        "GLOB_COLUMNS[0:2] are the openness pair",
        GLOB_COLUMNS[0] == "terminal_gap_nm" and GLOB_COLUMNS[1] == "radius_of_gyration_nm",
    )
    check(
        "n_global in the discriminator equals len(GLOB_COLUMNS)",
        "self.n_global = 4" in src and len(GLOB_COLUMNS) == 4,
    )


def test_shortcut_probe_excludes_the_openness_pair() -> None:
    print("the existing shortcut probe is blind to these columns")
    src = (
        Path(__file__).resolve().parents[1]
        / "src/proteinfoundation/cp2lp/discriminator.py"
    ).read_text()
    # Up to the next top-level def, NOT to the first blank line: the docstring contains
    # blank lines, so a "\n\n" split stops before reaching the body.
    probe_body = src.split("def probe_forward")[1].split("\ndef ")[0]
    # This is the premise of the whole probe: if `probe_forward` ever starts reading
    # glob[:, :2], `d_shortcut_probe_acc` would already cover this question and the two
    # instruments would silently overlap.
    check("probe_forward slices glob[:, 2:]", 'feats["glob"][:, 2:]' in probe_body, probe_body[:200])


def test_interface_cols_partition_the_block() -> None:
    print("the interface columns are the exact complement of the openness pair")
    # The second pass exists because holding (0, 1) measured an exact null, which leaves
    # (2, 3) as the only remaining place a glob tell can be. That inference is only valid if
    # the two sets COVER the block -- a fifth column added to GLOB_COLUMNS without being
    # assigned to a set would be silently untested and the `allblind` verdict would be wrong.
    opn, ifc = set(GLOB_OPENNESS_COLS), set(GLOB_INTERFACE_COLS)
    check("disjoint", not (opn & ifc), f"{opn} & {ifc}")
    check(
        "together they cover every glob column",
        opn | ifc == set(range(len(GLOB_COLUMNS))),
        f"{sorted(opn | ifc)} vs {list(range(len(GLOB_COLUMNS)))}",
    )
    check(
        "interface cols are contact_count and mean_min_heavy_nm",
        [GLOB_COLUMNS[c] for c in GLOB_INTERFACE_COLS]
        == ["contact_count", "mean_min_heavy_nm"],
        str([GLOB_COLUMNS[c] for c in GLOB_INTERFACE_COLS]),
    )


def test_every_ablation_is_accumulated() -> None:
    print("the logit accumulator is derived from ABLATIONS, not restated beside it")
    src = (
        Path(__file__).resolve().parents[1] / "script_utils/cp2lp_glob_probe.py"
    ).read_text()
    # The accumulator USED to be a hand-written key tuple. Adding an intervention to
    # ABLATIONS and forgetting it there raises KeyError inside the batch loop -- after the
    # model is loaded and the data is streaming, i.e. the most expensive possible moment.
    # Pin the derivation so the duplication cannot come back.
    check(
        "logits_acc comprehends over ABLATIONS",
        "for tag in ABLATIONS" in src,
        "logits_acc must be built from ABLATIONS",
    )
    check(
        "no hand-written blind-key list survives",
        '"real_openblind", "fake_openblind"' not in src,
    )
    # And every intervention must get a delta, since the ranking is the output.
    check("deltas are looped over ABLATIONS", "for ab in ABLATIONS:" in src)
    check("the ranking is emitted", '"ranked_deltas"' in src)


def test_interface_ablation_holds_only_its_own_columns() -> None:
    print("ifaceblind equalises contacts and closeness, leaving the gap separating")
    real = feats(torch.tensor([[2.1, 1.0, 9.0, 0.26], [2.3, 1.2, 8.0, 0.27]]))
    fake = feats(torch.tensor([[1.0, 0.6, 3.0, 0.12], [1.2, 0.8, 4.0, 0.13]]))
    r_ab, f_ab, _ = glob_ablated_pair(real, fake, GLOB_INTERFACE_COLS)
    for c in GLOB_INTERFACE_COLS:
        check(
            f"{GLOB_COLUMNS[c]} equalised",
            torch.allclose(r_ab["glob"][:, c], f_ab["glob"][:, c]),
        )
    check(
        "terminal gap still separates",
        float(r_ab["glob"][:, 0].min()) > float(f_ab["glob"][:, 0].max()),
    )


def test_report_math_on_synthetic_rows() -> None:
    print("report: a known-separable column is recovered, a null column is not")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cp2lp_glob_report import GAP, cv_scores

    rng = np.random.default_rng(0)
    n = 400
    rows = []
    for i in range(n):
        label = i % 2
        rows.append(
            {
                "label": label,
                "batch": i // 8,
                "peptide_length": 12,
                GAP: (2.14 if label else 1.05) + rng.normal(0, 0.05),
                "noise": rng.normal(0, 1.0),
            }
        )
    df = pd.DataFrame(rows)
    acc_gap, auc_gap, _ = cv_scores(df, [GAP], folds=5, seed=0)
    acc_noise, _auc_noise, _ = cv_scores(df, ["noise"], folds=5, seed=0)
    check("a cleanly separated gap is recovered", acc_gap > 0.95, f"{acc_gap:.3f}")
    check("AUC agrees", auc_gap > 0.98, f"{auc_gap:.3f}")
    check("a pure-noise column stays near chance", acc_noise < 0.62, f"{acc_noise:.3f}")

    # A constant column must not separate -- this is what the ablated columns look like to
    # the regression, so a nonzero accuracy here would mean the CV is leaking the label.
    df["held"] = 1.5
    acc_held, _, _ = cv_scores(df, ["held"], folds=5, seed=0)
    check("a held (constant) column is at chance", 0.4 <= acc_held <= 0.6, f"{acc_held:.3f}")


def test_accuracy_helper_matches_the_training_definition() -> None:
    print("discriminator_accuracy sign convention")
    real = torch.tensor([1.0, 2.0, -0.5])
    fake = torch.tensor([-1.0, -2.0, 0.5])
    check("2/3 + 2/3 -> 4/6", abs(discriminator_accuracy(real, fake) - 4 / 6) < 1e-6)
    check("chance on zero logits", discriminator_accuracy(torch.zeros(4), torch.zeros(4)) == 0.5)


def main() -> int:
    for fn in (
        test_fill_is_shared_across_classes,
        test_untouched_columns_and_blocks_survive,
        test_empty_ablation_is_the_identity,
        test_single_column_variants,
        test_interface_cols_partition_the_block,
        test_every_ablation_is_accumulated,
        test_interface_ablation_holds_only_its_own_columns,
        test_column_names_match_the_builder,
        test_shortcut_probe_excludes_the_openness_pair,
        test_report_math_on_synthetic_rows,
        test_accuracy_helper_matches_the_training_definition,
    ):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
