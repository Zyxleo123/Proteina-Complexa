"""Measure the backbone template library that the torsion builder constructs chains with.

WHY THIS EXISTS. Arm B builds N-CA-C from torsions plus a FIXED library of bond lengths and
bond angles, and then places O from a template. Those fixed values decide the arm's ceiling:
whatever the torsion head does, every bond length and bond angle it emits is the library's
value, so if the library is wrong the arm is handicapped in a way no amount of training can
recover and the comparison against Arm A stops being about the representation.

`geometry.py` already carries three measured lengths (N-CA 0.1458, CA-C 0.1525, C-N 0.1329 nm)
but NO angles, and the builder needs:

    N-CA-C      the intra-residue angle (tau), the one that actually varies with secondary
                structure and the one a fixed library most plausibly gets wrong
    CA-C-N      across the peptide bond
    C-N-CA      across the peptide bond
    CA-C-O      and the C=O length, to place the carbonyl oxygen
    O dihedral  dihedral(N, CA, C, O), which is psi + an offset -- measured here rather than
                assumed to be exactly 180 deg off

and it needs them for PRO and GLY separately where those differ, because proline's ring
closes phi and glycine has no CB to bias tau.

THE RECONSTRUCTION FLOOR IS THE POINT. Fixing lengths/angles to a library guarantees those
constraints within float precision, but it also means the builder CANNOT reproduce a native
backbone exactly -- a native has its own slightly different angles. The gap is this arm's
template-reconstruction floor, and `--report-floor` measures it by extracting each native's
torsions and rebuilding through the library. Any Arm B edit RMSD below that floor is template
drift, not an edit, which is why the floor has to be a number before the arm is trained and
not an excuse found afterwards.

OUTPUT. A JSON of per-parameter statistics plus a python constants block ready to paste. All
distances in NANOMETRES and all angles in DEGREES in the report; the module that consumes
them stores radians.

CPU only, seconds of work. Submitted as a job because every measurement in this project is
reproducible from a job script, not from a shell someone once ran.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Backbone atoms, in chain order. O is placed off C and is not part of the N-CA-C chain.
BB = ("N", "CA", "C", "O")

#: Residues whose templates are reported separately. PRO's ring and GLY's missing CB both
#: shift the intra-residue N-CA-C angle, which is the angle the builder is most sensitive to.
SPECIAL = ("PRO", "GLY")

#: A peptide bond is cis when |omega| < this. Measured here so the fraction is reported
#: rather than assumed to be zero -- the CA-CA calibration already found one cis bond.
CIS_OMEGA_CUT_DEG = 90.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--native-root",
        default="calibration_native",
        help="directory of <name>/processed/<name>.pdb native complexes",
    )
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument(
        "--report-floor",
        action="store_true",
        help="also measure the template reconstruction floor by rebuilding each native",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- parsing


def binder_chain(lines: list[str]) -> str | None:
    """The peptide chain id, read from the preprocessor's own REMARK rather than guessed.

    These files carry `REMARK 999 CHAIN MAP: R->A (target), L->B (binder)`. Guessing "the
    short chain" would work on most of them and silently pick the receptor on a complex with
    a short receptor fragment, so the mapping is read.
    """
    for ln in lines:
        if ln.startswith("REMARK 999 CHAIN MAP"):
            for part in ln.split(":", 1)[1].split(","):
                if "(binder)" in part:
                    return part.split("->")[1].split("(")[0].strip()
    return None


def read_peptide(path: Path) -> list[dict]:
    """Backbone atoms of the binder chain, as a list of per-residue dicts in file order.

    Returns residues each carrying `resname` and whichever of N/CA/C/O are present. A
    residue missing any of the four is kept -- the caller skips the measurements that need
    the absent atom rather than dropping the residue and breaking chain adjacency.
    """
    lines = path.read_text().splitlines()
    chain = binder_chain(lines)
    if chain is None:
        return []
    out: list[dict] = []
    seen: dict[tuple, int] = {}
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        # Fixed-column PDB parse. Splitting on whitespace merges fields on wide coordinates
        # and silently shifts every column after it.
        if ln[21] != chain:
            continue
        name = ln[12:16].strip()
        if name not in BB:
            continue
        altloc = ln[16]
        if altloc not in (" ", "A"):
            continue
        key = (ln[22:27], ln[17:20].strip())
        if key not in seen:
            seen[key] = len(out)
            out.append({"resname": ln[17:20].strip()})
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]) / 10.0  # A -> nm
        out[seen[key]][name] = xyz
    return out


# ------------------------------------------------------------------------- geometry


def norm(v: np.ndarray) -> float:
    return float(np.sqrt((v * v).sum()))


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c at vertex b. The vertex is the MIDDLE argument."""
    u, w = a - b, c - b
    cos = float((u * w).sum() / max(norm(u) * norm(w), 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral about p1-p2, IUPAC convention (anti = 180).

    The leading negation is what makes trans read 180 rather than 0; `ca_internal._dihedral`
    carries the same sign choice and `geometry.backbone_dihedrals` does NOT, which is why its
    phi/psi are offset by pi.
    """
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
    b1n = b1 / max(norm(b1), 1e-12)
    v = b0 - (b0 * b1n).sum() * b1n
    w = b2 - (b2 * b1n).sum() * b1n
    x = float((v * w).sum())
    y = float((np.cross(b1n, v) * w).sum())
    return math.degrees(math.atan2(y, x))


def collect(residues: list[dict], acc: dict[str, list[float]]) -> None:
    """Append every measurable backbone parameter of one peptide into `acc`."""
    for i, r in enumerate(residues):
        has = lambda *ns: all(n in r for n in ns)  # noqa: E731
        key = r["resname"] if r["resname"] in SPECIAL else "STD"

        if has("N", "CA"):
            acc["len_N_CA"].append(norm(r["CA"] - r["N"]))
        if has("CA", "C"):
            acc["len_CA_C"].append(norm(r["C"] - r["CA"]))
        if has("C", "O"):
            acc["len_C_O"].append(norm(r["O"] - r["C"]))
        if has("N", "CA", "C"):
            acc["ang_N_CA_C"].append(angle_deg(r["N"], r["CA"], r["C"]))
            acc[f"ang_N_CA_C@{key}"].append(angle_deg(r["N"], r["CA"], r["C"]))
        if has("CA", "C", "O"):
            acc["ang_CA_C_O"].append(angle_deg(r["CA"], r["C"], r["O"]))

        if i + 1 >= len(residues):
            continue
        nxt = residues[i + 1]
        if not ("C" in r and "N" in nxt):
            continue
        d_cn = norm(nxt["N"] - r["C"])
        # A chain break (gap in the deposited model) is not a peptide bond; measuring across
        # one would put metres of junk into the length distribution.
        if d_cn > 0.20:
            continue
        acc["len_C_N"].append(d_cn)
        if has("CA", "C"):
            acc["ang_CA_C_N"].append(angle_deg(r["CA"], r["C"], nxt["N"]))
        if "CA" in nxt:
            acc["ang_C_N_CA"].append(angle_deg(r["C"], nxt["N"], nxt["CA"]))
        if has("CA", "C") and "CA" in nxt:
            om = dihedral_deg(r["CA"], r["C"], nxt["N"], nxt["CA"])
            acc["omega"].append(om)
            acc["omega_is_cis"].append(1.0 if abs(om) < CIS_OMEGA_CUT_DEG else 0.0)
        # The O dihedral, as an OFFSET from psi. The builder needs the offset, because psi
        # is a variable it is handed and the offset is the constant it stores.
        #
        # MEASURED AROUND 180, NOT AROUND 0. The true offset is near +-180, and those are the
        # SAME angle -- so wrapping into (-180, 180] splits one tight cluster across both ends
        # of the range and reports sd = 166 deg for a quantity whose real spread is ~2 deg.
        # The first run of this script did exactly that. Wrapping the residual about 180
        # instead keeps the cluster in one piece, and the constant is recovered by adding 180
        # back. The median is then a median of a unimodal distribution, which is the only
        # case in which a median means anything.
        if has("N", "CA", "C", "O") and "N" in nxt:
            psi = dihedral_deg(r["N"], r["CA"], r["C"], nxt["N"])
            o_di = dihedral_deg(r["N"], r["CA"], r["C"], r["O"])
            acc["dihe_O_resid_about_180"].append((o_di - psi - 180.0 + 180.0) % 360.0 - 180.0)


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "sd": float(a.std()),
        "min": float(a.min()),
        "p1": float(np.percentile(a, 1)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


# ------------------------------------------------------------------- rebuild floor


def rebuild_floor(residues: list[dict], tmpl: dict[str, float], mode: str = "template") -> dict | None:
    """Extract torsions, rebuild, report the RMSD. The floor Arm B's edits are read against.

    This is the honest version of "the builder reproduces the input": under `mode="template"`
    it does NOT, because the library replaces every native length and angle with one number,
    and the residual accumulates along the chain.

    `mode` selects WHICH local geometry the rebuild uses, and the comparison between the two
    is the whole reason this takes an argument:

        "template"  every length and angle is the library constant. What Arm B does. Local
                    geometry is guaranteed correct whatever the generator emits -- including
                    when the source backbone handed to it is an AE decode whose own angles are
                    off-manifold, which is the case that motivates fixing them at all.
        "source"    carry each residue's OWN lengths and angles, edit only torsions. Zero
                    delta is then the exact identity, so there is no floor -- but the output
                    inherits whatever local geometry the source had, so the guarantee is only
                    as good as the AE decode.
        "per_res"   the library, but with PRO and GLY taking their own N-CA-C angle.

    The gap between "template" and "source" is the price of the guarantee, in nanometres. It
    is a number rather than an argument, which is the point of measuring it.

    Numpy, deliberately: this duplicates the torch builder's math independently, so a bug in
    one does not hide in the other. `test_cp2lp_backbone.py` pins that the two agree.
    """
    keep = [r for r in residues if all(n in r for n in ("N", "CA", "C"))]
    if len(keep) < 3:
        return None

    phi, psi, omega = [], [], []
    for i, r in enumerate(keep):
        prev_c = keep[i - 1]["C"] if i > 0 else None
        next_n = keep[i + 1]["N"] if i + 1 < len(keep) else None
        phi.append(dihedral_deg(prev_c, r["N"], r["CA"], r["C"]) if prev_c is not None else -60.0)
        psi.append(dihedral_deg(r["N"], r["CA"], r["C"], next_n) if next_n is not None else -45.0)
        if i + 1 < len(keep):
            omega.append(dihedral_deg(r["CA"], r["C"], keep[i + 1]["N"], keep[i + 1]["CA"]))

    # A mode of the form "src:<param>" carries the SOURCE's own value for that one parameter
    # and the template's for the rest -- a leave-one-out decomposition of the floor. Without
    # it "the template costs 0.118 nm" is a number with no address, and the obvious guess
    # (the intra-residue N-CA-C angle, the one with the widest native spread) is only a guess.
    one_out = mode.split(":", 1)[1] if mode.startswith("src:") else None
    src_all = mode == "source"

    def local(i: int) -> dict[str, float]:
        """The lengths and angles residue i is built with, under the selected mode."""
        r = keep[i]
        prv = keep[i - 1] if i > 0 else None
        g = {
            "len_C_N": tmpl["len_C_N"], "len_N_CA": tmpl["len_N_CA"], "len_CA_C": tmpl["len_CA_C"],
            "ang_CA_C_N": tmpl["ang_CA_C_N"], "ang_C_N_CA": tmpl["ang_C_N_CA"],
            "ang_N_CA_C": tmpl["ang_N_CA_C"],
        }
        if mode == "per_res" and r["resname"] in SPECIAL:
            g["ang_N_CA_C"] = tmpl.get(f"ang_N_CA_C@{r['resname']}", g["ang_N_CA_C"])
            return g
        src = {
            "ang_N_CA_C": lambda: angle_deg(r["N"], r["CA"], r["C"]),
            "len_N_CA": lambda: norm(r["CA"] - r["N"]),
            "len_CA_C": lambda: norm(r["C"] - r["CA"]),
            "len_C_N": lambda: norm(r["N"] - prv["C"]) if prv is not None else g["len_C_N"],
            "ang_CA_C_N": lambda: angle_deg(prv["CA"], prv["C"], r["N"]) if prv is not None else g["ang_CA_C_N"],
            "ang_C_N_CA": lambda: angle_deg(prv["C"], r["N"], r["CA"]) if prv is not None else g["ang_C_N_CA"],
        }
        for key, fn in src.items():
            if src_all or one_out == key:
                g[key] = fn()
        return g

    # Seed on the native's own first three atoms, so the rebuild is in the native's frame and
    # the RMSD is a geometry difference rather than a superposition difference.
    built = [{"N": keep[0]["N"].copy(), "CA": keep[0]["CA"].copy(), "C": keep[0]["C"].copy()}]
    for i in range(len(keep) - 1):
        prev = built[-1]
        # Each placement reads the geometry of the residue it is PLACING (i+1), except
        # `ang_CA_C_N` and `len_C_N`, which belong to the bond and so are read there too.
        g = local(i + 1)
        n_next = place(prev["N"], prev["CA"], prev["C"], g["len_C_N"], g["ang_CA_C_N"], psi[i])
        ca_next = place(prev["CA"], prev["C"], n_next, g["len_N_CA"], g["ang_C_N_CA"], omega[i])
        c_next = place(prev["C"], n_next, ca_next, g["len_CA_C"], g["ang_N_CA_C"], phi[i + 1])
        built.append({"N": n_next, "CA": ca_next, "C": c_next})

    errs = []
    for b, r in zip(built, keep):
        for name in ("N", "CA", "C"):
            errs.append(norm(b[name] - r[name]))
    e = np.asarray(errs)
    ca_e = np.asarray([norm(b["CA"] - r["CA"]) for b, r in zip(built, keep)])
    return {
        "n_res": len(keep),
        "bb_rmsd_nm": float(np.sqrt((e * e).mean())),
        "ca_rmsd_nm": float(np.sqrt((ca_e * ca_e).mean())),
        "ca_max_nm": float(ca_e.max()),
    }


def place(a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float, ang_deg: float, tor_deg: float) -> np.ndarray:
    """NeRF: place d such that |cd|=bond, angle(b,c,d)=ang, dihedral(a,b,c,d)=tor."""
    ang, tor = math.radians(ang_deg), math.radians(tor_deg)
    bc = c - b
    bc = bc / max(norm(bc), 1e-12)
    nrm = np.cross(b - a, bc)
    nrm = nrm / max(norm(nrm), 1e-12)
    m2 = np.cross(nrm, bc)
    off = (
        (-bond * math.cos(ang)) * bc
        + (bond * math.sin(ang) * math.cos(tor)) * m2
        + (bond * math.sin(ang) * math.sin(tor)) * nrm
    )
    return c + off


# ------------------------------------------------------------------------------ main


def main() -> int:
    args = parse_args()
    root = Path(args.native_root)
    pdbs = sorted(root.glob("*/processed/*.pdb"))
    if not pdbs:
        print(f"ERROR: no PDBs under {root}/*/processed/", file=sys.stderr)
        return 1
    print(f"[calib] {len(pdbs)} native complexes under {root}")

    acc: dict[str, list[float]] = defaultdict(list)
    peptides: list[list[dict]] = []
    skipped = 0
    for p in pdbs:
        res = read_peptide(p)
        if len(res) < 3:
            skipped += 1
            continue
        peptides.append(res)
        collect(res, acc)
    print(f"[calib] usable peptides: {len(peptides)}   skipped: {skipped}")

    report = {k: stats(v) for k, v in sorted(acc.items())}
    cis = acc.get("omega_is_cis", [])
    report["omega_cis_fraction"] = {"n": len(cis), "fraction": float(np.mean(cis)) if cis else 0.0}

    # The library takes MEDIANS, not means: the distributions have a cis tail in omega and
    # a few strained outliers in the angles, and a median is the value a typical residue
    # actually has.
    tmpl = {k: report[k]["median"] for k in (
        "len_N_CA", "len_CA_C", "len_C_N", "len_C_O",
        "ang_N_CA_C", "ang_CA_C_N", "ang_C_N_CA", "ang_CA_C_O",
    ) if report.get(k, {}).get("n", 0) > 0}
    # Recover the constant from the residual measured about 180 (see `collect`).
    tmpl["dihe_O_minus_psi"] = 180.0 + report["dihe_O_resid_about_180"]["median"]
    for key in SPECIAL:
        s = report.get(f"ang_N_CA_C@{key}", {})
        if s.get("n"):
            tmpl[f"ang_N_CA_C@{key}"] = s["median"]
    report["template"] = tmpl

    if args.report_floor:
        report["rebuild_floor"] = {}
        modes = ["template", "per_res", "source"] + [
            f"src:{k}" for k in (
                "ang_N_CA_C", "ang_CA_C_N", "ang_C_N_CA", "len_C_N", "len_N_CA", "len_CA_C",
            )
        ]
        for mode in modes:
            floors = [f for f in (rebuild_floor(r, tmpl, mode) for r in peptides) if f is not None]
            if floors:
                report["rebuild_floor"][mode] = {
                    "n_peptides": len(floors),
                    "bb_rmsd_nm": stats([f["bb_rmsd_nm"] for f in floors]),
                    "ca_rmsd_nm": stats([f["ca_rmsd_nm"] for f in floors]),
                    "ca_max_nm": stats([f["ca_max_nm"] for f in floors]),
                }

    print("\n--- measured backbone template ------------------------------------")
    for k in sorted(tmpl):
        s = report.get(k, {})
        unit = "nm" if k.startswith("len") else "deg"
        if s.get("n"):
            print(f"  {k:<20s} {tmpl[k]:9.4f} {unit}   n={s['n']:<6d} sd={s['sd']:.4f}  "
                  f"[{s['p1']:.3f}, {s['p99']:.3f}]")
    od = report["dihe_O_resid_about_180"]
    print(f"  {'dihe_O_minus_psi':<20s} {tmpl['dihe_O_minus_psi']:9.4f} deg   n={od['n']:<6d} "
          f"sd={od['sd']:.4f}  [{180 + od['p1']:.3f}, {180 + od['p99']:.3f}]  (about 180)")
    print(f"  omega cis fraction   {report['omega_cis_fraction']['fraction']:.4f} "
          f"of {report['omega_cis_fraction']['n']} bonds")
    for key in ("STD", "PRO", "GLY"):
        s = report.get(f"ang_N_CA_C@{key}", {})
        if s.get("n"):
            print(f"  ang_N_CA_C @{key:<4s}      {s['median']:9.4f} deg   n={s['n']:<6d} sd={s['sd']:.4f}")

    if report.get("rebuild_floor"):
        rf = report["rebuild_floor"]
        print("\n--- reconstruction FLOOR (extract torsions -> rebuild -> compare) -----")
        print(f"  {'mode':<18s} {'bb RMSD med':>12s} {'p99':>8s} {'max':>8s}   {'CA RMSD med':>12s}")
        base = rf.get("template", {}).get("bb_rmsd_nm", {}).get("median")
        for mode in ("template", "per_res", "source"):
            if mode not in rf:
                continue
            s, c = rf[mode]["bb_rmsd_nm"], rf[mode]["ca_rmsd_nm"]
            print(f"  {mode:<18s} {s['median']:12.4f} {s['p99']:8.4f} {s['max']:8.4f}   "
                  f"{c['median']:12.4f}")
        print("  -- leave-one-out: source value for THAT parameter, template for the rest ---")
        loo = sorted(
            ((m, rf[m]["bb_rmsd_nm"]["median"]) for m in rf if m.startswith("src:")),
            key=lambda kv: kv[1],
        )
        for mode, med in loo:
            s = rf[mode]["bb_rmsd_nm"]
            share = f"{100.0 * (base - med) / base:5.1f}% of the floor" if base else ""
            print(f"  {mode:<18s} {med:12.4f} {s['p99']:8.4f} {s['max']:8.4f}   -> {share}")
        print("  READ THIS AS: Arm B cannot place a backbone closer to its input than its own")
        print("  row, so an Arm B edit RMSD at or below that is template drift, not an edit.")
        if "template" in rf and "source" in rf:
            gap = rf["template"]["bb_rmsd_nm"]["median"] - rf["source"]["bb_rmsd_nm"]["median"]
            print(f"  PRICE OF THE GUARANTEE: {gap:+.4f} nm of floor, median, bought in exchange")
            print("  for local geometry that is correct whatever the generator (or the AE decode")
            print("  feeding it) emits. `source` has no floor but inherits the AE's own angles.")

    print("\n--- paste into backbone.py ----------------------------------------")
    print(f"TEMPLATE_N_CA_NM = {tmpl['len_N_CA']:.5f}")
    print(f"TEMPLATE_CA_C_NM = {tmpl['len_CA_C']:.5f}")
    print(f"TEMPLATE_C_N_NM = {tmpl['len_C_N']:.5f}")
    print(f"TEMPLATE_C_O_NM = {tmpl['len_C_O']:.5f}")
    print(f"TEMPLATE_N_CA_C_DEG = {tmpl['ang_N_CA_C']:.3f}")
    print(f"TEMPLATE_CA_C_N_DEG = {tmpl['ang_CA_C_N']:.3f}")
    print(f"TEMPLATE_C_N_CA_DEG = {tmpl['ang_C_N_CA']:.3f}")
    print(f"TEMPLATE_CA_C_O_DEG = {tmpl['ang_CA_C_O']:.3f}")
    print(f"TEMPLATE_O_DIHEDRAL_OFFSET_DEG = {tmpl['dihe_O_minus_psi']:.3f}")
    for key, const in (("PRO", "TEMPLATE_N_CA_C_PRO_DEG"), ("GLY", "TEMPLATE_N_CA_C_GLY_DEG")):
        s = report.get(f"ang_N_CA_C@{key}", {})
        if s.get("n"):
            print(f"{const} = {s['median']:.3f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\n[calib] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
