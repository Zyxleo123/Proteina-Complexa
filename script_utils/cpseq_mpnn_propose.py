#!/usr/bin/env python3
"""Arm C: propose sequences by redesigning generator CP backbones with ProteinMPNN.

The rationale (plan 5) is that arms A and B both edit sequence space around ONE
conformation -- whatever Boltz-2 predicts from the original LP sequence. This arm instead
takes several plausible cyclic conformations from the project's own CP generators and asks
an inverse-folding model what sequence each one wants, with the target present as fixed
structural context.

Three things this script is careful about, each because the obvious implementation is
wrong:

**The generator supplies CONFORMATIONS, never sequences.** Measured on
`cpsea_bondunroll_from_v4cfg`, unconditional samples decode to sequences like
`KAGGAGGGGGGDLD` and `KEGDGGGGGGFGGGGD` -- glycine-rich, 14 of 14 and 16 of 16
substitutions from the original LP sequence, i.e. 0 % identity. Those can never be
candidates under a floor(L/2) ceiling. So the original LP sequence is THREADED onto the
generator's backbone before redesign, and the substitution budget is enforced by the
edit mask, not by the generator.

**`fixed_positions` fixes residues to their identity IN THE INPUT PDB.** Handing MPNN the
generator's own PDB would therefore pin the unmasked positions to that poly-glycine
sequence, and every proposal would inherit it. Threading the original sequence onto the
backbone first is what makes "outside the mask, fix identities to the original LP
sequence" true. ProteinMPNN reads backbone atoms only (N, CA, C, O) and uses residue
identity solely for fixed positions, so renaming residues on a fixed backbone is exactly
the supported operation and changes no geometry.

**ProteinMPNN is cyclic-blind, and that is recorded rather than worked around.**
`protein_mpnn_utils.PositionalEncodings` one-hots a clipped LINEAR relative offset
`idx[i]-idx[j]`, so for a head-to-tail 12-mer residues 0 and 11 encode as 11 apart and
the closing bond is absent from the input; the ring is visible only as 3D proximity of
the termini. This repo already documents the identical failure for AF2 in
`evaluation/cyclic_offset.py`. No cyclic-specialised MPNN weights are installed (only
vanilla / soluble / ligand / membrane variants), and CyclicMPNN is explicitly a later
substitution. `--cyclic-offset-wrap` applies the same wrap this repo uses for AF2 to the
peptide block, and is **off by default** because these weights were TRAINED on linear
offsets: wrapped offsets are off-distribution for them, so the wrap is a hypothesis to
A/B, not a known improvement.

MPNN likelihood is a proposal diagnostic. Success is the fresh Boltz-2 refold through the
same frozen gate as every other arm.

Usage:
    python script_utils/cpseq_mpnn_propose.py \
        --scored-backbones $OUT/backbones/scored_bondunroll.jsonl \
        --dataset-dir $ZFS/cpseq/tranche1 \
        --out-dir $OUT --generator gen_bondunroll_uncond --per-parent 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import tolerances as tolmod  # noqa: E402
from proteinfoundation.cpseq.candidates import ONE_TO_THREE  # noqa: E402
from proteinfoundation.cpseq.search import budget_for, proposal_seed  # noqa: E402

MPNN_DIR = Path("/home/yixiz/Proteina-Complexa/community_models/ProteinMPNN")
PEPTIDE_CHAIN = "B"

# Backbone admission is NOT the final validity gate, and conflating them would kill this
# arm for a reason that never reaches the answer.
#
# Measured: only 14.6 % of this project's seq-cond generator samples and 20 % of NATIVE
# CPSea closures sit inside the crystal-derived torsion window (+/-8.44 deg), because
# CPSea's computational cyclization never enforced trans omega. Applying the final gate
# here would reject ~85 % of backbones over a closing-bond TORSION -- a property of the
# proposal's geometry that is discarded the moment ProteinMPNN reads only N/CA/C/O and
# Boltz-2 refolds the sequence from scratch. Boltz-2 is 92.9 % trans, so the torsion of
# the final answer comes from Boltz, not from the proposal.
#
# What a backbone must actually have, to be worth asking a sequence of: a genuine closing
# bond (the termini bonded, not 7-17 A apart), a real interface with the target, complete
# backbone, L stereochemistry, and no severe clashes. The frozen tolerances supply the
# clash limits; the bond ceiling below is deliberately looser than the crystal window's
# 1.6 A upper bound and is reported with every admitted backbone.
# Ring closure is judged on the anchor **C-beta** distance, not on the decoded N-C bond.
#
# The flow model's product space is (bb_ca, local_latents): it places CA and latents, and
# the backbone N and C atoms come out of the frozen autoencoder's decode. An N-C distance
# therefore charges the generator for the decoder's backbone reconstruction, while the
# CB-CB distance at the two anchor residues measures what the model actually positioned.
# The window is this repo's own: `eval/cyclic_reconstruction_metrics.py` uses
# `(d >= 3.0) & (d <= 8.0)`, and the lower bound matters -- anchors closer than 3 A are
# fused, not cyclizable.
#
# Measured on 12 samples per checkpoint, this changes the ranking: on anchor CB the
# seq-cond model places rings slightly BETTER than bondunroll (0.833 vs 0.750), while on
# the decoded N-C bond bondunroll looks better (0.667 vs 0.333). Those are two different
# questions -- "are the anchors positioned for a ring" and "did the decoded bond come out
# right" -- so both are recorded and only CB gates.
ADMIT_CB_WINDOW_A = (3.0, 8.0)

# Retained as a DIAGNOSTIC only, no longer a gate.
ADMIT_MAX_BOND_A = 2.0

# How far the closure-constrained refinement may move the peptide and still be called an
# adjustment of the generator's conformation rather than a replacement of it.
#
# This threshold is doing load-bearing work, because the refined structure looks perfect
# for every sample and that is an artefact. The refinement carries a harmonic
# AtomPairConstraint (sd 0.05 A) on the closing bond, so it CLOSES THE RING WHETHER OR
# NOT THE GENERATOR DID. Measured on 12 `cpsea_bondunroll_from_v4cfg` samples: raw
# closing distances span 1.03-17.57 A, and after refinement every one of them reads
# 1.29-1.34 A with zero clashes. Admitting on refined closure would therefore admit
# 12/12 and call a 17.6 A open chain a cyclic backbone.
#
# What separates them is the displacement. The raw-OPEN samples moved 5.3-11.5 A of
# binder CA RMSD -- the restraint hauled the termini together and the resulting
# conformation is the restraint's, not the model's. The raw-CLOSED ones moved 0.70, 0.96,
# 0.97 A (plus one at 2.03 and one at 4.85). So the gate is: the generator must have
# formed the ring ITSELF (raw bond), and the refinement must only have tidied it.
ADMIT_MAX_REFINE_BINDER_RMSD_A = 1.5


def _virtual_cb(res: dict) -> np.ndarray:
    """The standard virtual CB from N, CA, C -- used for GLY and any missing CB."""
    N, CA, C = res["N"], res["CA"], res["C"]
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def anchor_cb_distance(pdb_path: Path, chain: str = PEPTIDE_CHAIN) -> tuple[float | None, str]:
    """CB-CB distance between the first and last peptide residue, with its CB source.

    Measured from the written structure rather than taken from the training-time
    aggregate. `val_gen/cyc/cyc_cb_window_success` reports 0.0 for samples whose anchor CB
    distances this function puts inside the window on the very same files, so the two
    disagree; which one is wrong is unresolved, and this is the one whose definition is
    visible here.
    """
    from proteinfoundation.cpseq.geometry import read_chain

    residues = read_chain(pdb_path, chain)
    if len(residues) < 2:
        return None, "too_few_residues"
    (_, first), (_, last) = residues[0], residues[-1]
    try:
        cb0, k0 = (first["CB"], "real") if "CB" in first else (_virtual_cb(first), "virtual")
        cbL, kL = (last["CB"], "real") if "CB" in last else (_virtual_cb(last), "virtual")
    except KeyError as exc:
        return None, f"missing_atom:{exc}"
    return float(np.linalg.norm(cb0 - cbL)), f"{k0}/{kL}"


def admit_backbone(row: dict, tol) -> tuple[bool, list[str], dict]:
    """The proposal-stage gate. Returns (admitted, reasons_against, diagnostics).

    Measured on RAW geometry for closure and on the refinement displacement for
    conformational integrity. Clashes are NOT gated on the raw structure: every raw
    generator sample carries at least one severe clash (1-75 peptide-receptor), all of
    which the restrained relax resolves to zero, and a clash that relax fixes without
    moving the backbone is not a reason to discard a conformation. The displacement
    threshold is what prevents that leniency from admitting a rebuilt pose.
    """
    reasons: list[str] = []
    geom = row.get("geometry") or {}
    closure = geom.get("closure") or {}
    atoms = geom.get("atoms") or {}
    chir = geom.get("chirality") or {}
    rosetta = row.get("rosetta") or {}
    disp = rosetta.get("refinement_displacement") or {}

    raw_bond = closure.get("bond_length_A")

    # The gate: anchor CB-CB on the RAW structure. Raw, because the closure restraint
    # closes the ring whether or not the generator did -- refined coordinates read
    # 1.29-1.34 A for every sample including 17 A open chains.
    raw_pdb = Path(row.get("raw_pdb_path") or row.get("pdb_path") or "")
    cb_dist, cb_source = (None, "no_raw_pdb")
    if raw_pdb.exists():
        cb_dist, cb_source = anchor_cb_distance(raw_pdb)
    lo, hi = ADMIT_CB_WINDOW_A
    if cb_dist is None:
        reasons.append(f"anchor CB distance unmeasurable ({cb_source})")
    elif cb_dist > hi:
        reasons.append(
            f"the GENERATOR did not place a ring: anchor CB-CB {round(cb_dist, 2)} A "
            f"> {hi} A (an open chain)"
        )
    elif cb_dist < lo:
        reasons.append(
            f"anchors fused: CB-CB {round(cb_dist, 2)} A < {lo} A (too close to cyclize)"
        )

    # The displacement guard answers "did the RESTRAINT manufacture this ring?", and it
    # answers it by proxy: a raw-open ring can only be closed by hauling the termini
    # together, which drags the whole peptide. That proxy is only valid while the raw ring
    # is open. When the generator ALREADY closed the bond, a large displacement cannot be
    # closure being manufactured -- there was nothing to manufacture -- so it must come
    # from something else (clash relief, interface repacking), and rejecting the backbone
    # on it fails an authentic ring with a test whose rationale does not apply.
    #
    # Real case, `lnr_2v8x` sample 3: raw closing bond 1.57 A (inside the frozen window),
    # anchor CB 5.76 A, rejected for moving 4.85 A. Its ring was genuine.
    #
    # Closure is judged on the project's frozen window rather than a new threshold --
    # `tol` is the same calibrated 1.1-1.6 A that accepts 4/4 experimental crystals and
    # 150/150 native CPSea head-to-tail entries.
    raw_bond_closed = (
        raw_bond is not None
        and tol.bond_length_min_A <= raw_bond <= tol.bond_length_max_A
    )
    rmsd = disp.get("binder_ca_rmsd_A")
    moved = rmsd is not None and rmsd > ADMIT_MAX_REFINE_BINDER_RMSD_A
    if rmsd is None:
        reasons.append("no refinement displacement recorded; cannot tell adjustment from rebuild")
    elif moved and not raw_bond_closed:
        reasons.append(
            f"refinement moved the peptide {round(rmsd, 2)} A CA RMSD "
            f"> {ADMIT_MAX_REFINE_BINDER_RMSD_A} A -- the refined conformation is the "
            f"restraint's, not the generator's"
        )

    # The displacement cap was ALSO carrying a second, unstated job: it is what licenses
    # MPNN to read the REFINED structure ("the same conformation, with the clashes
    # resolved"). Waiving the cap without noticing that would hand MPNN a pose 4.85 A from
    # anything the generator produced, under a comment asserting it had not moved. So the
    # two judgements are now separate: an authentic ring that nonetheless drifted is
    # admitted, and its design substrate falls back to the RAW coordinates -- clashes and
    # all, which degrade the proposal but keep it the generator's conformation. MPNN
    # likelihood is only a proposal diagnostic; the arm's answer is the Boltz-2 refold.
    design_substrate = "raw" if (moved and raw_bond_closed) else "refined"

    if not atoms.get("backbone_complete", True):
        reasons.append("incomplete backbone")
    if chir and not chir.get("all_L", True):
        reasons.append(f"{chir.get('n_d_amino_acids')} D-amino acid(s)")

    dsasa = (rosetta.get("raw") or {}).get("dSASA_int")
    if dsasa is None or dsasa <= 0:
        reasons.append(f"no buried interface with the target (dSASA_int={dsasa})")

    diagnostics = {
        "anchor_cb_distance_A": None if cb_dist is None else round(cb_dist, 3),
        "anchor_cb_source": cb_source,
        "anchor_cb_window_A": list(ADMIT_CB_WINDOW_A),
        # Diagnostic, not a gate: the decoded N-C bond charges the generator for the
        # frozen autoencoder's backbone reconstruction (the model places CA + latents).
        "raw_closing_bond_A": raw_bond,
        "raw_closing_bond_would_pass_2A": (
            None if raw_bond is None else bool(raw_bond <= ADMIT_MAX_BOND_A)
        ),
        "refine_binder_ca_rmsd_A": rmsd,
        # Whether the generator itself closed the ring, on the frozen window -- the
        # condition under which the displacement proxy above is meaningful at all.
        "raw_bond_closed_frozen_window": raw_bond_closed,
        "raw_bond_window_A": [tol.bond_length_min_A, tol.bond_length_max_A],
        "displacement_waived_ring_authentic": bool(moved and raw_bond_closed),
        # Which coordinates MPNN must design on, given the above.
        "design_substrate": design_substrate,
        # Recorded, not gated, and genuinely unexplained: a FastRelax restrained to start
        # coordinates should not translate the receptor, yet this reads 1.9-88.6 A across
        # the same 12 samples. Until that is understood, an arm C interface metric taken
        # from a REFINED generator complex should not be trusted; the arm's actual answer
        # comes from the independent Boltz-2 refold, which never sees these coordinates.
        "refine_receptor_ca_rmsd_A": disp.get("receptor_ca_rmsd_A"),
        "raw_n_intra_clashes": (geom.get("clashes") or {}).get("n_intra_clashes"),
        "raw_n_inter_clashes": (geom.get("clashes") or {}).get("n_inter_clashes"),
    }
    return (not reasons), reasons, diagnostics


# --------------------------------------------------------------------------------------
# Threading the original sequence onto a generator backbone
# --------------------------------------------------------------------------------------

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def thread_sequence_onto_backbone(
    pdb_in: Path, pdb_out: Path, *, peptide_chain: str, original: str
) -> dict:
    """Rewrite the peptide chain's residue identities to `original`, keeping coordinates.

    Only backbone atoms survive on the peptide: a side chain belonging to the generator's
    decoded residue is not the side chain of the residue we are writing, and leaving it
    would hand MPNN atoms that contradict the residue name on the same line. The receptor
    is copied verbatim -- its identities are the native ones and are the fixed context.
    """
    lines_out: list[str] = []
    seen_resseq: list[str] = []
    n_pep_atoms = 0
    n_dropped = 0

    for line in pdb_in.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith(("TER", "END")):
                lines_out.append(line)
            continue
        chain = line[21]
        if chain != peptide_chain:
            lines_out.append(line)
            continue
        atom = line[12:16].strip()
        resseq = line[22:27]
        if resseq not in seen_resseq:
            seen_resseq.append(resseq)
        if atom not in BACKBONE_ATOMS:
            n_dropped += 1
            continue
        idx = seen_resseq.index(resseq)
        if idx >= len(original):
            # The backbone is longer than the sequence we are threading; that is a
            # mismatch, not something to truncate silently.
            raise ValueError(
                f"{pdb_in.name}: peptide has >{len(original)} residues "
                f"({len(seen_resseq)} seen) but original sequence is {len(original)}"
            )
        new_res = ONE_TO_THREE[original[idx]]
        lines_out.append(line[:17] + f"{new_res:>3s}" + line[20:])
        n_pep_atoms += 1

    if len(seen_resseq) != len(original):
        raise ValueError(
            f"{pdb_in.name}: peptide length {len(seen_resseq)} != original "
            f"{len(original)} -- substitutions only, so these must match"
        )

    pdb_out.write_text("\n".join(lines_out) + "\n")
    return {
        "peptide_residues": len(seen_resseq),
        "peptide_backbone_atoms_kept": n_pep_atoms,
        "peptide_sidechain_atoms_dropped": n_dropped,
    }


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------


def build_masks(length: int, *, n_masks: int, seed: int) -> list[tuple[int, ...]]:
    """Varied edit masks, every one of size <= floor(L/2), including small ones.

    The size ceiling IS the substitution ceiling: outside the mask the sequence is pinned
    to the original, so a proposal cannot exceed |mask| substitutions. Sizes are spread
    from 1 up to the budget rather than all sitting at the ceiling, because the ceiling is
    a limit and not a target -- and because a 1- or 2-position mask asks MPNN for a much
    smaller extrapolation than a full-budget one, which matters for weights that have
    never seen a macrocycle.

    Interface positions are deliberately NOT excluded: the task permits sequence changes
    that improve binding, so fixing the interface by default would forbid the edits most
    likely to matter.
    """
    budget = budget_for(length)
    rng = np.random.default_rng(seed)
    sizes: list[int] = []
    for s in (1, 2, max(1, budget // 2), budget):
        if 1 <= s <= budget and s not in sizes:
            sizes.append(s)
    masks: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempt = 0
    while len(masks) < n_masks and attempt < n_masks * 40:
        size = sizes[len(masks) % len(sizes)]
        pos = tuple(sorted(int(p) for p in rng.choice(length, size=size, replace=False)))
        attempt += 1
        if pos in seen:
            continue
        seen.add(pos)
        masks.append(pos)
    return masks


def run_mpnn(
    *,
    pdb_path: Path,
    fixed_positions: list[int],
    out_dir: Path,
    n_seq: int,
    temperature: float,
    seed: int,
    model_name: str,
    omit_aas: str,
) -> list[dict]:
    """Design the peptide chain with the target fixed. Returns parsed sequences.

    `--pdb_path_chains B` makes chain B the only designed chain, so every receptor chain
    is fixed structural context by construction rather than by listing its positions.
    `fixed_positions` then pins the unmasked peptide positions, which the threading step
    has already set to the original LP identities.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = pdb_path.stem
    with tempfile.TemporaryDirectory() as td:
        fixed_path = Path(td) / "fixed.jsonl"
        # 1-indexed within the chain, which is what ProteinMPNN expects.
        fixed_path.write_text(
            json.dumps({name: {PEPTIDE_CHAIN: [p + 1 for p in fixed_positions]}}) + "\n"
        )
        cmd = [
            sys.executable,
            str(MPNN_DIR / "protein_mpnn_run.py"),
            "--pdb_path", str(pdb_path),
            "--pdb_path_chains", PEPTIDE_CHAIN,
            "--fixed_positions_jsonl", str(fixed_path),
            "--out_folder", str(out_dir),
            "--num_seq_per_target", str(n_seq),
            "--sampling_temp", str(temperature),
            "--seed", str(seed),
            "--model_name", model_name,
            "--path_to_model_weights", str(MPNN_DIR / "vanilla_model_weights"),
            # X and C: X is the default omission, C keeps the arm's alphabet identical to
            # the other arms' (no unrequested crosslink partner in a molecule whose label
            # names one ring).
            "--omit_AAs", omit_aas,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            return [{
                "status": "mpnn_error",
                "reason": (proc.stderr or proc.stdout)[-800:],
                "sequence": None,
            }]

    fa = out_dir / "seqs" / f"{name}.fa"
    if not fa.exists():
        return [{"status": "mpnn_no_output", "reason": f"missing {fa}", "sequence": None}]
    return parse_mpnn_fasta(fa)


def parse_mpnn_fasta(fa: Path) -> list[dict]:
    """Parse ProteinMPNN's FASTA. The first record is the input; the rest are designs.

    Chains are '/'-joined in one sequence line. The peptide is located by LENGTH rather
    than by position in that join, because the join order follows the parsed chain order
    and silently depends on the input file -- picking index 1 would be a guess that fails
    on a multi-chain receptor.
    """
    records: list[tuple[str, str]] = []
    header = None
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            header = line[1:]
        elif line.strip() and header is not None:
            records.append((header, line.strip()))
            header = None
    if not records:
        return [{"status": "mpnn_no_output", "reason": "no FASTA records", "sequence": None}]

    native_segments = records[0][1].split("/")
    out: list[dict] = []
    for header, seq in records[1:]:
        segs = seq.split("/")
        meta: dict = {}
        for part in header.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    meta[k.strip()] = float(v.strip())
                except ValueError:
                    meta[k.strip()] = v.strip()
        # The designed peptide is the segment whose length matches the native peptide's,
        # and whose content actually changed relative to the receptor segments.
        peptide = None
        if len(segs) == len(native_segments):
            for s, nat in zip(segs, native_segments):
                if len(s) == len(nat) and len(s) == len(native_segments[-1]):
                    peptide = s
            if peptide is None:
                peptide = segs[-1]
        else:
            peptide = segs[-1]
        out.append({
            "status": "ok",
            "sequence": peptide,
            "mpnn_score": meta.get("score"),
            "mpnn_global_score": meta.get("global_score"),
            "mpnn_seq_recovery": meta.get("seq_recovery"),
            "mpnn_temperature": meta.get("T"),
            "mpnn_sample": meta.get("sample"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored-backbones", type=Path, required=True,
                    help="jsonl from cpseq_score_seqcond.py over a generator dump")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--generator", required=True, help="recorded on every proposal")
    ap.add_argument("--per-parent", type=int, default=32,
                    help="sequence proposals per parent for this generator")
    ap.add_argument("--masks-per-backbone", type=int, default=4)
    ap.add_argument("--seqs-per-mask", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--model-name", default="v_48_020")
    ap.add_argument("--omit-aas", default="XC")
    ap.add_argument("--chemistry", default="mainchain")
    ap.add_argument("--cyclic-offset-wrap", action="store_true",
                    help="wrap the peptide's relative positional offset around the ring "
                         "(the construction evaluation/cyclic_offset.py uses for AF2). OFF "
                         "by default: these weights were trained on LINEAR offsets, so a "
                         "wrapped offset is off-distribution for them and is a hypothesis "
                         "to A/B rather than a known improvement.")
    args = ap.parse_args()

    if args.cyclic_offset_wrap:
        print("[mpnn] --cyclic-offset-wrap requested but NOT implemented in this "
              "milestone: it needs a patched featurizer (ProteinMPNN builds residue_idx "
              "with +100 jumps between chains, so the wrap must touch chain B only). "
              "Refusing rather than silently running the unwrapped model.", file=sys.stderr)
        return 2

    tol = tolmod.load(args.dataset_dir / "manifest" / "geometry_tolerances.json")[args.chemistry]
    rows = [json.loads(l) for l in args.scored_backbones.read_text().splitlines() if l.strip()]
    print(f"[mpnn] {len(rows)} scored backbone(s) from {args.scored_backbones.name}")

    work = args.out_dir / "backbones" / f"mpnn_{args.generator}"
    work.mkdir(parents=True, exist_ok=True)
    proposals_path = args.out_dir / "backbones" / f"proposals_{args.generator}.jsonl"
    admission_path = args.out_dir / "backbones" / f"admission_{args.generator}.jsonl"

    by_parent: dict[str, list[dict]] = defaultdict(list)
    # Admission diagnostics, keyed by raw PDB path, so the proposal loop below can read
    # back each backbone's `design_substrate` instead of re-deciding it.
    adm_diag: dict[str, dict] = {}
    n_admitted = 0
    with open(admission_path, "w") as ah:
        for r in rows:
            ok, reasons, diag = admit_backbone(r, tol)
            adm_diag[str(r.get("raw_pdb_path") or r.get("pdb_path"))] = diag
            ah.write(json.dumps({
                "parent_id": r.get("parent_id"),
                "sample_index": r.get("sample_index"),
                "pdb_path": r.get("raw_pdb_path") or r.get("pdb_path"),
                "generator": args.generator,
                "admitted": ok,
                "reasons_against": reasons,
                "diagnostics": diag,
                "gate": {
                    "name": "backbone_admission",
                    "anchor_cb_window_A": list(ADMIT_CB_WINDOW_A),
                    "closure_judged_on": "anchor CB-CB on the RAW structure",
                    "raw_closing_bond_A_is_diagnostic_only": True,
                    "max_refine_binder_ca_rmsd_A": ADMIT_MAX_REFINE_BINDER_RMSD_A,
                    "displacement_waived_when_raw_bond_closed": True,
                    "raw_bond_closed_window_A": [tol.bond_length_min_A,
                                                 tol.bond_length_max_A],
                    "requires_buried_interface": True,
                    "measured_on": "RAW closure + refinement displacement",
                    "note": "distinct from the frozen final-validity gate. Closure "
                            "angles/torsion are NOT required (the answer's ring geometry "
                            "comes from the Boltz-2 refold), and raw clashes are not "
                            "gated (the restrained relax resolves all of them to zero); "
                            "the displacement limit is what stops a relax-rebuilt pose "
                            "from being admitted as a generator conformation",
                },
                "closing_bond_A": ((r.get("geometry") or {}).get("closure") or {}).get("bond_length_A"),
                "anchor_cb_distance_A": diag.get("anchor_cb_distance_A"),
                "raw_dSASA_int": ((r.get("rosetta") or {}).get("raw") or {}).get("dSASA_int"),
                "generator_decoded_sequence": r.get("sequence_generated"),
            }) + "\n")
            if ok:
                by_parent[r["parent_id"]].append(r)
                n_admitted += 1

    print(f"[mpnn] admitted {n_admitted}/{len(rows)} backbone(s) -> {admission_path}")
    for pid, rs in sorted(by_parent.items()):
        print(f"  {pid}: {len(rs)} admitted")

    n_written = 0
    with open(proposals_path, "w") as ph:
        for pid, rs in sorted(by_parent.items()):
            original = rs[0].get("original_lp_sequence")
            if not original:
                print(f"[mpnn] {pid}: no original_lp_sequence on the row -- skipped")
                continue
            L = len(original)
            budget = budget_for(L)
            emitted: set[str] = {original}
            per_parent_done = 0

            for r in rs:
                if per_parent_done >= args.per_parent:
                    break
                raw_src = Path(r.get("raw_pdb_path") or r.get("pdb_path"))
                # The REFINED structure is normally what MPNN reads: it is the same
                # conformation (admission capped the displacement) with the severe clashes
                # resolved, and a clashing neighbour graph is exactly what degrades an
                # inverse-folding proposal. Raw geometry decided admission; refined
                # coordinates carry the design.
                #
                # The exception is a backbone admitted under the authentic-ring waiver,
                # whose refined pose DID drift past the cap. For those the premise above
                # is false, so admission hands back `design_substrate: "raw"` and the
                # design is done on the generator's own coordinates instead.
                substrate = (adm_diag.get(str(raw_src)) or {}).get("design_substrate")
                if substrate == "raw":
                    src = raw_src
                else:
                    src = Path(str(raw_src.with_suffix("")) + "_refined.pdb")
                    if not src.exists():
                        src = raw_src
                if not src.exists():
                    continue
                threaded = work / f"{pid}_s{r.get('sample_index')}.pdb"
                try:
                    thread_info = thread_sequence_onto_backbone(
                        src, threaded, peptide_chain=PEPTIDE_CHAIN, original=original
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[mpnn] {pid} s{r.get('sample_index')}: threading failed: {exc}")
                    continue

                masks = build_masks(
                    L, n_masks=args.masks_per_backbone,
                    seed=proposal_seed(pid, args.generator, r.get("sample_index")),
                )
                for mask in masks:
                    if per_parent_done >= args.per_parent:
                        break
                    fixed = [i for i in range(L) if i not in mask]
                    t0 = time.time()
                    designs = run_mpnn(
                        pdb_path=threaded,
                        fixed_positions=fixed,
                        out_dir=work / f"{pid}_s{r.get('sample_index')}_m{'-'.join(map(str, mask))}",
                        n_seq=args.seqs_per_mask,
                        temperature=args.temperature,
                        seed=int(proposal_seed(pid, args.generator, r.get("sample_index"), mask) % 100000),
                        model_name=args.model_name,
                        omit_aas=args.omit_aas,
                    )
                    dt = round(time.time() - t0, 2)
                    for d in designs:
                        seq = d.get("sequence")
                        if d.get("status") != "ok" or not seq:
                            print(f"[mpnn] {pid} mask={mask}: {d.get('status')} "
                                  f"{str(d.get('reason'))[:120]}")
                            continue
                        if len(seq) != L:
                            print(f"[mpnn] {pid} mask={mask}: parsed length {len(seq)} != {L}; dropped")
                            continue
                        # The mask is the budget guarantee, so it is CHECKED and not
                        # assumed: every position outside the mask must still hold the
                        # original residue.
                        violated = [i for i in fixed if seq[i] != original[i]]
                        if violated:
                            print(f"[mpnn] {pid} mask={mask}: MPNN changed fixed "
                                  f"position(s) {violated}; dropped")
                            continue
                        n_mut = sum(1 for a, b in zip(original, seq) if a != b)
                        if n_mut > budget:
                            print(f"[mpnn] {pid} mask={mask}: {n_mut} > budget {budget}; dropped")
                            continue
                        if seq in emitted:
                            continue
                        emitted.add(seq)
                        ph.write(json.dumps({
                            "parent_id": pid,
                            "arm": "backbone_mpnn",
                            "generator": args.generator,
                            "generator_ckpt_step": (r.get("config") or {}).get("ckpt_step"),
                            "backbone_sample_index": r.get("sample_index"),
                            "backbone_pdb_raw_path": str(raw_src),
                            "backbone_pdb_used_path": str(src),
                            "threaded_pdb_path": str(threaded),
                            "threading": thread_info,
                            "edit_mask": list(mask),
                            "mask_size": len(mask),
                            "original_sequence": original,
                            "sequence": seq,
                            "n_mutations": n_mut,
                            "mutation_fraction": round(n_mut / L, 6),
                            "substitution_budget": budget,
                            "mpnn_model": args.model_name,
                            "mpnn_temperature": args.temperature,
                            "mpnn_omit_AAs": args.omit_aas,
                            "mpnn_score": d.get("mpnn_score"),
                            "mpnn_global_score": d.get("mpnn_global_score"),
                            "mpnn_seq_recovery": d.get("mpnn_seq_recovery"),
                            "mpnn_wall_time_s": dt,
                            "mpnn_is_cyclic_aware": False,
                            "mpnn_limitation": (
                                "ProteinMPNN encodes a clipped LINEAR relative offset "
                                "(protein_mpnn_utils.PositionalEncodings), so the "
                                "head-to-tail closing bond is absent from its input; the "
                                "ring is visible only as 3D proximity of the termini. No "
                                "cyclic-specialised MPNN weights are installed."
                            ),
                            "note": "proposal only; MPNN likelihood is not the success metric",
                        }) + "\n")
                        n_written += 1
                        per_parent_done += 1

            print(f"[mpnn] {pid}: {per_parent_done} proposal(s) (L={L}, budget={budget})")

    print(f"[mpnn] {n_written} proposal(s) -> {proposals_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
