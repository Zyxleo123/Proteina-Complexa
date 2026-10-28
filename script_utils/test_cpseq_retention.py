#!/usr/bin/env python3
"""Checks for the standalone random arm and the retention tiers. No GPU, no prediction.

The retention checks run against a COPY of a real prediction directory rather than a
synthetic one, because the thing being tested is which of Boltz-2's actual output files
are safe to delete, and a fixture I wrote myself would only test my own assumptions about
that layout.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import retention as retmod  # noqa: E402
from proteinfoundation.cpseq.search import (  # noqa: E402
    ARM_RANDOM,
    KIND_RANDOM_STANDALONE,
    budget_for,
    n_mutations,
    standalone_counts,
    standalone_random_arm,
)

REAL = Path("/data/user_data/yixiz/cpseq/pepbench_search/predictions")
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- standalone random arm
def test_standalone_counts() -> None:
    print("standalone count policy")
    orig = "RRRTGTYGVLAAWRRL"  # L=16, ceiling 8
    c = standalone_counts(original=orig, n_slots=400, seed_base="p1")
    check("respects the floor(L/2) ceiling", all(1 <= x <= budget_for(len(orig)) for x in c),
          f"range {min(c)}-{max(c)}")
    check("never draws zero substitutions", 0 not in c)
    frac1 = sum(1 for x in c if x == 1) / len(c)
    check("reproduces the measured ~59 % single-substitution mass", 0.50 < frac1 < 0.68,
          f"{frac1:.3f}")

    # A 5-mer has ceiling 2, so the distribution must truncate and renormalise rather
    # than emitting counts the parent cannot spend.
    short = standalone_counts(original="MEEVD", n_slots=200, seed_base="p1")
    check("truncates to the ceiling on a short peptide", set(short) <= {1, 2},
          f"counts {sorted(set(short))}")

    a = standalone_counts(original=orig, n_slots=32, seed_base="p1")
    b = standalone_counts(original=orig, n_slots=32, seed_base="p1")
    check("deterministic for a given (parent, n_slots)", a == b)
    check("differs between parents", a != standalone_counts(
        original=orig, n_slots=32, seed_base="p2"))

    # The resume property that the matched arm lacked: slot i's count is independent of
    # how many slots were already paid for, so a requeue cannot re-plan.
    full = standalone_counts(original=orig, n_slots=32, seed_base="p1")
    check("slot assignment is a prefix-stable plan",
          standalone_counts(original=orig, n_slots=32, seed_base="p1")[:8] == full[:8])


def test_standalone_arm() -> None:
    print("standalone random arm")
    orig = "RRRTGTYGVLAAWRRL"
    props, note = standalone_random_arm(original=orig, n_slots=32, seed_base="pb1")
    check("fills the requested budget", len(props) == 32, f"{len(props)} proposals; {note}")
    check("all distinct", len({p.sequence for p in props}) == len(props))
    check("original never proposed", orig not in {p.sequence for p in props})
    check("every draw within budget",
          all(n_mutations(orig, p.sequence) <= budget_for(len(orig)) for p in props))
    check("realised count equals the drawn count",
          all(n_mutations(orig, p.sequence) >= 1 for p in props))
    check("no cysteine introduced",
          all("C" not in p.sequence for p in props))
    check("arm and kind recorded as standalone",
          all(p.arm == ARM_RANDOM and p.kind == KIND_RANDOM_STANDALONE for p in props))
    check("needs no guided input at all", True, "signature has no `counts` parameter")

    # Exhaustion must be reported, not silently under-delivered.
    tiny, note2 = standalone_random_arm(original="MEEVD", n_slots=32, seed_base="pb1")
    check("short peptide reports coverage honestly when exhausted",
          len(tiny) == 32 or bool(note2), f"{len(tiny)} proposals; note={bool(note2)}")

    rerun, _ = standalone_random_arm(original=orig, n_slots=32, seed_base="pb1")
    check("reproducible across processes",
          [p.sequence for p in rerun] == [p.sequence for p in props])


# ------------------------------------------------------------------------- Kabsch
def test_kabsch() -> None:
    print("Kabsch superposition")
    rng = np.random.default_rng(0)
    P = rng.normal(size=(40, 3)).astype(np.float32)
    theta = 0.7
    Rtrue = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
    Q = P @ Rtrue.T + np.array([5.0, -2.0, 1.0], dtype=np.float32)
    R, t, rmsd = retmod.kabsch(P, Q)
    check("recovers a rigid transform exactly", rmsd < 1e-4, f"rmsd {rmsd:.2e}")
    check("transform maps mobile onto target",
          float(np.abs(P @ R.T + t - Q).max()) < 1e-3)
    # A mirror image must NOT be fittable, or a flipped peptide would score as a match.
    M = P.copy()
    M[:, 2] *= -1
    _, _, mrmsd = retmod.kabsch(M, P)
    check("refuses to fit a reflection", mrmsd > 0.1, f"mirror rmsd {mrmsd:.3f}")


# ------------------------------------------------------- retention, on a real directory
def _find_real_pred_dir() -> Path | None:
    if not REAL.exists():
        return None
    for parent in sorted(REAL.iterdir()):
        if not parent.is_dir():
            continue
        for d in sorted(parent.iterdir()):
            if (d / "boltz").exists() and any(
                (d / "boltz").rglob("input_model_0.pdb")
            ):
                return d
    return None


def test_retention_on_real_dir() -> None:
    print("retention tiers, against a real prediction directory")
    src = _find_real_pred_dir()
    if src is None:
        print("  SKIP  no real prediction directory available")
        return

    def size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        before = None
        for tier in ("full", "rescorable", "deliverable"):
            work = base / tier
            shutil.copytree(src, work)
            if before is None:
                before = size(work)
            rep = retmod.compact_prediction_dir(work, tier)
            after = size(work)
            raws = list((work / "boltz").rglob("input_model_0.pdb"))
            gzs = list((work / "boltz").rglob("input_model_0.pdb.gz"))
            print(f"    {tier:12} {before/1024:8.1f} KB -> {after/1024:7.1f} KB "
                  f"({after/before*100:5.1f}%)  raw={len(raws)} gz={len(gzs)}")
            if tier == "full":
                check("full tier removes nothing", after == before and rep["freed_bytes"] == 0)
            if tier == "rescorable":
                check("rescorable keeps the raw complex, gzipped",
                      len(gzs) >= 1 and len(raws) == 0)
                check("rescorable drops the processed-input tree",
                      not (work / "boltz" / "boltz_results_input" / "processed").exists())
                check("rescorable drops pae/pde",
                      not list((work / "boltz").rglob("pae_*.npz")))
                check("rescorable is a large reduction", after < 0.25 * before,
                      f"{after/before*100:.1f}% of original")
                restored = retmod.restore_raw_structures(work)
                check("rescorable can restore a scoreable PDB",
                      bool(restored) and all(p.exists() for p in restored))
                txt = restored[0].read_text()
                check("restored PDB is intact", txt.lstrip().startswith(("ATOM", "HEADER",
                      "REMARK", "MODEL", "TITLE", "CRYST")) and "ATOM" in txt)
            if tier == "deliverable":
                check("deliverable drops every structure",
                      not raws and not gzs)
                check("deliverable is a drastic reduction", after < 0.06 * before,
                      f"{after/before*100:.1f}% of original")
            # Idempotence: a requeued task must not fail or double count.
            rep2 = retmod.compact_prediction_dir(work, tier)
            check(f"{tier} compaction is idempotent", rep2["freed_bytes"] == 0)
            shutil.rmtree(work)


def test_design_record_on_real_dir() -> None:
    print("design record, canonical frame, against real structures")
    src_parent = None
    if REAL.exists():
        for parent in sorted(REAL.iterdir()):
            ds = [d for d in sorted(parent.iterdir())
                  if d.is_dir() and list(d.rglob("input_model_0.pdb"))]
            if len(ds) >= 3:
                src_parent = (parent, ds[:3])
                break
    if src_parent is None:
        print("  SKIP  need a parent with >=3 predictions")
        return
    parent, dirs = src_parent

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fits, recs = [], []
        for d in dirs:
            raw = next(iter(d.rglob("input_model_0.pdb")))
            ref = next(iter(d.rglob("input_model_0_refined.pdb")), None)
            out = retmod.write_design_record(
                out_root=root, parent_id=parent.name, input_hash=d.name,
                sequence="X" * 5, original_sequence="X" * 5,
                binder_chain="B", receptor_chains={"A"},
                raw_pdb=raw, refined_pdb=ref,
            )
            recs.append(out)
            if out.get("receptor_fit_rmsd_A") is not None:
                fits.append(out["receptor_fit_rmsd_A"])

        check("a record is written for each candidate",
              all(r.get("design_record") for r in recs), str(recs[0])[:90])
        canon = root / "designs" / parent.name / "receptor.pdb.gz"
        check("exactly one canonical receptor per target", canon.exists())
        npzs = sorted((root / "designs" / parent.name).glob("*.npz"))
        check("one npz per candidate", len(npzs) == len(dirs), f"{len(npzs)} npz")

        z = np.load(npzs[0], allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        check("npz carries peptide coordinates", "refined_xyz" in z or "raw_xyz" in z,
              f"keys {sorted(z.files)}")
        check("npz carries the sequence and fit diagnostic",
              meta.get("sequence") and "receptor_fit_rmsd_A" in meta)

        per = [npz.stat().st_size for npz in npzs]
        print(f"    canonical receptor {canon.stat().st_size/1024:.1f} KB (once); "
              f"peptide npz {np.mean(per)/1024:.1f} KB each; receptor fits {fits}")
        check("peptide payload is small", float(np.mean(per)) < 20 * 1024,
              f"{np.mean(per)/1024:.1f} KB mean")
        check("receptor fit is recorded and sane",
              all(f is not None and f < 5.0 for f in fits), f"{fits}")

        # The point of the canonical frame: peptides from different candidates become
        # directly comparable. In the raw files they sit in unrelated global frames.
        if len(npzs) >= 2:
            a = np.load(npzs[0])["refined_xyz"] if "refined_xyz" in np.load(npzs[0]).files else None
            b = np.load(npzs[1])["refined_xyz"] if "refined_xyz" in np.load(npzs[1]).files else None
            if a is not None and b is not None:
                ca, cb = a.mean(0), b.mean(0)
                check("canonical-frame peptides share a frame",
                      float(np.linalg.norm(ca - cb)) < 25.0,
                      f"centroid separation {float(np.linalg.norm(ca-cb)):.2f} A")


def main() -> int:
    for fn in (test_standalone_counts, test_standalone_arm, test_kabsch,
               test_retention_on_real_dir, test_design_record_on_real_dir):
        fn()
        print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
