"""Smoke the backbone arms end to end, including a COMPLETE packing and scoring example.

WHAT THIS HAS TO PROVE, because the plan does not accept the experiment as done until it does:

* both arms run forward and backward on real complexes, with finite gradients;
* each arm's zero-output really is the source backbone (the recorded init reconstruction error);
* the torsion builder preserves the bond geometry it claims to;
* PACKING runs on a generated backbone from each arm and produces a complete all-atom linear
  peptide, with the receptor and the peptide backbone provably unmoved;
* the packed peptide is LINEAR -- no re-formed disulfide, no closing bond.

It runs on native complexes rather than on synthetic tensors, because the thing most likely to
break is a real structure's chemistry (a cysteine pair, a proline, a missing atom), and
synthetic data cannot contain any of it.

CPU only. The arms' forward pass here uses RANDOM head outputs rather than a trained trunk --
this smoke is about the representation and packing plumbing, not about generation quality, and
loading a flow checkpoint would need a GPU and would test something else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openfold.np.residue_constants import (  # noqa: E402
    atom_order,
    restype_3to1,
    restype_num,
    restype_order,
)

from proteinfoundation.cp2lp.backbone import (  # noqa: E402
    BB_NAMES,
    DEFAULT_ANGLE_SOURCE,
    backbone_validity,
    extract_backbone,
)
from proteinfoundation.cp2lp.bb_arms import (  # noqa: E402
    ARM_CARTESIAN,
    ARM_TORSION,
    apply_backbone_arm,
    init_reconstruction_error,
)
from proteinfoundation.cp2lp.bb_losses import (  # noqa: E402
    backbone_preservation_loss,
    backbone_steric_loss,
    open_backbone_geometry_loss,
    source_interface_weights,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native-root", default=str(REPO / "calibration_native"))
    p.add_argument("--n-complexes", type=int, default=3)
    p.add_argument("--pack-seeds", type=int, default=3)
    p.add_argument("--work-dir", default=None, help="where candidate PDBs go")
    p.add_argument("--out", default=None, help="write a JSON report here")
    p.add_argument("--skip-packing", action="store_true", help="backbone-only; labels the report")
    return p.parse_args()


# --------------------------------------------------------------------------- PDB -> tensors


def _chain_map(lines: list[str]) -> tuple[str | None, str | None]:
    """(receptor_chain, peptide_chain) from the preprocessor's REMARK, not guessed."""
    rec = pep = None
    for ln in lines:
        if ln.startswith("REMARK 999 CHAIN MAP"):
            for part in ln.split(":", 1)[1].split(","):
                if "(binder)" in part:
                    pep = part.split("->")[1].split("(")[0].strip()
                elif "(target)" in part:
                    rec = part.split("->")[1].split("(")[0].strip()
    return rec, pep


def load_complex(path: Path) -> dict | None:
    """Native complex PDB -> atom37 tensors in NANOMETRES, receptor frame as deposited."""
    lines = path.read_text().splitlines()
    rec_ch, pep_ch = _chain_map(lines)
    if rec_ch is None or pep_ch is None:
        return None

    chains: dict[str, dict] = {rec_ch: {}, pep_ch: {}}
    order: dict[str, list] = {rec_ch: [], pep_ch: []}
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        ch = ln[21]
        if ch not in chains:
            continue
        if ln[16] not in (" ", "A"):
            continue
        name = ln[12:16].strip()
        if name not in atom_order:
            continue
        key = ln[22:27]
        if key not in chains[ch]:
            chains[ch][key] = {"resname": ln[17:20].strip(), "atoms": {}}
            order[ch].append(key)
        chains[ch][key]["atoms"][name] = [
            float(ln[30:38]) / 10.0, float(ln[38:46]) / 10.0, float(ln[46:54]) / 10.0
        ]

    def pack(ch: str):
        keys = order[ch]
        n = len(keys)
        a = np.zeros((n, 37, 3), dtype=np.float32)
        m = np.zeros((n, 37), dtype=bool)
        aa = np.full((n,), restype_num, dtype=np.int64)
        for i, k in enumerate(keys):
            r = chains[ch][k]
            aa[i] = restype_order.get(restype_3to1.get(r["resname"], "X"), restype_num)
            for nm, xyz in r["atoms"].items():
                j = atom_order[nm]
                a[i, j] = xyz
                m[i, j] = True
        return torch.from_numpy(a), torch.from_numpy(m), torch.from_numpy(aa)

    t37, tm, taa = pack(rec_ch)
    p37, pm, paa = pack(pep_ch)
    if p37.shape[0] < 4 or t37.shape[0] < 4:
        return None
    # Every backbone atom must be present, or the arm has nothing to read.
    need = [atom_order[k] for k in BB_NAMES]
    if not bool(pm[:, need].all()):
        return None
    return {
        "id": path.stem,
        "pep_atom37": p37, "pep_atom_mask": pm, "pep_aatype": paa,
        "target_atom37": t37, "target_atom_mask": tm, "target_aatype": taa,
        "receptor_chain": rec_ch, "peptide_chain": pep_ch, "src_pdb": str(path),
    }


def collate(items: list[dict]) -> dict:
    """Pad a list of complexes into one batch, left-aligned, the convention the arms assume."""
    b = len(items)
    n = max(x["pep_atom37"].shape[0] for x in items)
    m = max(x["target_atom37"].shape[0] for x in items)
    out = {
        "pep_atom37": torch.zeros(b, n, 37, 3), "pep_atom_mask": torch.zeros(b, n, 37, dtype=torch.bool),
        "pep_aatype": torch.full((b, n), restype_num, dtype=torch.long),
        "mask": torch.zeros(b, n, dtype=torch.bool),
        "target_atom37": torch.zeros(b, m, 37, 3), "target_atom_mask": torch.zeros(b, m, 37, dtype=torch.bool),
        "target_aatype": torch.full((b, m), restype_num, dtype=torch.long),
        "target_mask": torch.zeros(b, m, dtype=torch.bool),
        "ids": [x["id"] for x in items], "items": items,
    }
    for i, x in enumerate(items):
        ln, lm = x["pep_atom37"].shape[0], x["target_atom37"].shape[0]
        out["pep_atom37"][i, :ln] = x["pep_atom37"]
        out["pep_atom_mask"][i, :ln] = x["pep_atom_mask"]
        out["pep_aatype"][i, :ln] = x["pep_aatype"]
        out["mask"][i, :ln] = True
        out["target_atom37"][i, :lm] = x["target_atom37"]
        out["target_atom_mask"][i, :lm] = x["target_atom_mask"]
        out["target_aatype"][i, :lm] = x["target_aatype"]
        out["target_mask"][i, :lm] = True
    return out


# ------------------------------------------------------------------------------- the smoke


def random_head(arm: str, b: int, n: int, seed: int, scale: float = 0.4) -> dict:
    g = torch.Generator().manual_seed(seed)
    if arm == ARM_CARTESIAN:
        return {"delta_cartesian": scale * torch.randn(b, n, 4, 3, generator=g)}
    return {
        "delta_torsion": scale * torch.randn(b, n, 6, generator=g),
        "delta_root": 0.1 * scale * torch.randn(b, 6, generator=g),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.native_root)
    pdbs = sorted(root.glob("*/processed/*.pdb"))
    items = []
    for p in pdbs:
        c = load_complex(p)
        if c is not None:
            items.append(c)
        if len(items) >= args.n_complexes:
            break
    if not items:
        print(f"ERROR: no usable complexes under {root}", file=sys.stderr)
        return 1
    batch = collate(items)
    mask = batch["mask"]
    b, n = mask.shape
    print(f"[smoke] {b} complexes: {', '.join(batch['ids'])}  (padded n={n})")

    src_bb, _ = extract_backbone(batch["pep_atom37"], batch["pep_atom_mask"])
    report: dict = {"complexes": batch["ids"], "arms": {}, "packing_ran": False}

    iface_w, iface_met = source_interface_weights(
        batch["pep_atom37"], batch["pep_atom_mask"], mask,
        batch["target_atom37"], batch["target_atom_mask"], batch["target_mask"],
    )
    print(f"[smoke] interface residues: {float(iface_met['iface_residue_frac']):.3f} of the peptide")

    for arm in (ARM_CARTESIAN, ARM_TORSION):
        print(f"\n=== arm {arm} ===")
        a: dict = {}

        init = init_reconstruction_error(
            arm, src_bb, mask, angle_source=DEFAULT_ANGLE_SOURCE, aatype=batch["pep_aatype"]
        )
        a["init_reconstruction"] = init
        print(f"  init reconstruction : bb_rmsd {init['init_bb_rmsd_nm']*10:.5f} A   "
              f"max {init['init_bb_max_nm']*10:.5f} A   (zero head output vs source)")
        if init["init_bb_rmsd_nm"] > 1e-4:
            print("    NOTE: NONZERO -- this arm does not start at the source, so its first "
                  "output is already an unrequested edit of this size.")
            print("    Expected only when a template is in use: 0.118 nm under templated "
                  "ANGLES, ~0.021 nm under templated LENGTHS with source angles.")

        head = random_head(arm, b, n, seed=7)
        head = {k: v.clone().requires_grad_(True) for k, v in head.items()}
        gen_bb, diag = apply_backbone_arm(
            arm, src_bb, {"backbone": head}, mask, aatype=batch["pep_aatype"]
        )
        a["edit_diag"] = {k: float(v) for k, v in diag.items()}
        print(f"  edit               : mean shift {float(diag['bb_shift_mean_nm'])*10:.3f} A   "
              f"max {float(diag['bb_shift_max_nm'])*10:.3f} A")

        v = backbone_validity(gen_bb, mask)
        a["raw_validity"] = {k: float(t.mean()) for k, t in v.items()}
        print(f"  RAW validity       : chain_intact {float(v['bb_chain_intact'].mean()):.3f}   "
              f"peptide_bond_pass {float(v['bb_peptide_bond_pass'].mean()):.3f}   "
              f"tau_pass {float(v['bb_tau_pass'].mean()):.3f}   "
              f"carbonyl_pass {float(v['bb_carbonyl_pass'].mean()):.3f}")

        pres, pm_met = backbone_preservation_loss(gen_bb, src_bb, mask, iface_w)
        ster, st_met = backbone_steric_loss(
            gen_bb, mask, batch["target_atom37"], batch["target_atom_mask"], batch["target_mask"]
        )
        geom, g_met = open_backbone_geometry_loss(gen_bb, mask)
        total = pres + ster + geom
        total.backward()
        grads = {k: float(t.grad.abs().max()) for k, t in head.items()}
        a["losses"] = {"preserve": float(pres), "steric": float(ster), "geometry": float(geom)}
        a["metrics"] = {k: float(x) for k, x in {**pm_met, **st_met, **g_met}.items()}
        a["head_grad_max"] = grads
        print(f"  losses             : preserve {float(pres):.4f}  steric {float(ster):.4f}  "
              f"geometry {float(geom):.4f}")
        print(f"  head grad (max)    : {grads}")
        assert all(np.isfinite(x) and x > 0 for x in grads.values()), f"{arm}: dead head gradient"

        report["arms"][arm] = a

        if not args.skip_packing:
            a["packing"] = run_packing(arm, gen_bb.detach(), batch, args)
            report["packing_ran"] = True

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[smoke] wrote {args.out}")

    if not report["packing_ran"]:
        print("\n*** BACKBONE-ONLY: packing did not run, so this is NOT the holistic check. ***")
    return 0


def run_packing(arm: str, gen_bb: torch.Tensor, batch: dict, args) -> dict:
    """Pack one generated backbone per complex and report. The step that makes this holistic."""
    from proteinfoundation.cp2lp import packing as pk

    out: dict = {"available": pk.is_available(), "candidates": []}
    if not out["available"]:
        print("  packing            : PyRosetta UNAVAILABLE -- backbone-only result")
        return out

    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init

    _ensure_pyrosetta_init()
    import pyrosetta

    work = Path(args.work_dir) if args.work_dir else Path(
        __import__("tempfile").mkdtemp(prefix=f"cp2lp_pack_{arm}_")
    )
    work.mkdir(parents=True, exist_ok=True)
    scorefxn = pyrosetta.create_score_function("ref2015")

    pep_ang, _ = pk._peptide_atom37_ang(gen_bb, batch["mask"])
    for i, item in enumerate(batch["items"]):
        ln = item["pep_atom37"].shape[0]
        cand: dict = {"id": item["id"], "results": []}
        pdb = str(work / f"{item['id']}__{arm}__cand.pdb")
        pk.write_complex_pdb(
            pdb,
            pep_ang[i, :ln],
            item["pep_aatype"].numpy(),
            (item["target_atom37"] * 10.0).numpy(),
            item["target_aatype"].numpy(),
        )
        try:
            pose = pyrosetta.pose_from_pdb(pdb)
            stripped = pk.strip_cyclic_topology(pose)
            pk.add_terminal_chemistry(pose)
            pk.assert_linear_topology(pose)
            cand["stripped"] = stripped
            cand["n_pose_res"] = pose.total_residue()
            cand["n_pep_res"] = len(pk.peptide_residue_indices(pose))
            src_chis = pk.extract_chis(item["src_pdb"], item["peptide_chain"])
            cand["n_chi_seeded"] = pk.transfer_source_chis(pose, src_chis)
        except Exception as e:  # a real structure's chemistry is where this breaks
            cand["error"] = f"{type(e).__name__}: {e}"
            print(f"  packing {item['id']:<28s}: SETUP FAILED {cand['error']}")
            out["candidates"].append(cand)
            continue

        for s in range(args.pack_seeds):
            try:
                packed, diag = pk.pack_one(pose, s, scorefxn=scorefxn)
                pk.assert_linear_topology(packed)
                p = str(work / f"{item['id']}__{arm}__pack{s}.pdb")
                packed.dump_pdb(p)
                cand["results"].append(
                    {"seed": s, "ok": True, "pdb": p, **{k: float(x) for k, x in diag.items()}}
                )
            except Exception as e:
                cand["results"].append({"seed": s, "ok": False, "failure": f"{type(e).__name__}: {e}"})

        ok = [r for r in cand["results"] if r["ok"]]
        print(f"  packing {item['id']:<28s}: {len(ok)}/{args.pack_seeds} ok   "
              f"chi_seeded {cand.get('n_chi_seeded', 0)}   "
              f"S-S removed {cand.get('stripped', {}).get('disulfide_removed', 0)}   "
              + (f"fixed-atom shift {max(r['fixed_atom_max_shift_ang'] for r in ok):.2e} A   "
                 f"score {min(r['total_score'] for r in ok):.1f}" if ok else "ALL FAILED"))
        if not ok:
            for r in cand["results"][:1]:
                print(f"      first failure: {r.get('failure')}")
        out["candidates"].append(cand)

    out["work_dir"] = str(work)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
