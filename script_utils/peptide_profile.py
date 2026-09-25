"""Topology-agnostic feature profile for a peptide-receptor complex (pre-build check 3a).

ONE function -- `profile()` -- maps a complex to a fixed-length feature vector, and it must
return the same vector layout for a cyclic macrocycle, a synthetic open state and a real
linear binder.  That invariance is the whole point: the adversarial harness in
`precheck_adversary.py` compares two labelled sets through this vector, so any feature that
is only *defined* on one topology would decide the comparison by its own absence.

Three rules follow, and every one of them costs a little fidelity on purpose:

  * **Torsions are read off the linear chain**, L-1 phi and L-1 psi, never around a ring
    bond.  A cyclic peptide does have a phi at residue 0, but a linear one does not, so
    using it would make `rama_*` mean different things in the two sets.
  * **No CONECT record is read.**  Nothing here asks whether a ring bond exists.
  * **Nothing is normalised by a quantity only one set has.**  Where a feature needs a size
    normaliser it uses peptide residue count, which every topology has.

Staging descriptors (peptide length, receptor length, receptor fragmentation) are computed
too, but kept in a SEPARATE vector -- `staging_profile()`.  They are the confound, not the
signal: PepBench receptors are whole chains at a median 223 residues while CPSea receptors
are 14 A pocket crops at 139, so a classifier handed both vectors at once will reach a high
AUC on the staging artifact and name the wrong feature as the difference.  See
`docs/README_POSE_DECOY_INVENTORY.md` section 2.5 and its resolution at 4.5.

Runs in `.venv` (mdtraj / numpy).  No OpenMM, no PyRosetta, no GPU.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# mdtraj's PDB reader is chatty about the unit cell on these staged files and about
# residues it does not recognise; neither affects anything computed here.
warnings.filterwarnings("ignore", category=UserWarning, module="mdtraj")

# --------------------------------------------------------------------------------------
# Constants.  Every cutoff that also exists elsewhere in the repo is matched byte-for-byte
# and says so, because a profile computed against a different cutoff is a different
# quantity and must not be compared with one that was not.
# --------------------------------------------------------------------------------------
HEAVY_CONTACT_CUTOFF_A = 4.5      # kinematic_ceiling.HEAVY_CONTACT_CUTOFF_A
MIN_SEQ_SEP_CONTACT = 3           # |i-j| >= 3 for intra-peptide contacts; closer pairs are
                                  # chain neighbours that every conformation has
RG_EXPONENT = -0.6                # Rg ~ N^0.6 for a self-avoiding walk, so Rg * N^-0.6 is
                                  # the length-invariant compactness
MIN_PEPTIDE_LEN = 4               # below this, phi/psi and contact order are not defined
SEGMENT_GAP = 1                   # receptor residues whose resSeq jumps by more than this
                                  # start a new segment (the pocket-crop convention)

# Ramachandran boxes, in degrees.  Deliberately coarse and non-exhaustive: `extended` and
# `turn` do NOT sum to 1, and the remainder (disallowed / bridge / left-handed-beta) is
# reported as neither.  Making them exhaustive would force a boundary decision that the
# data does not support and that would move the fractions without any conformation moving.
RAMA_EXTENDED = ((-180.0, -45.0), [(90.0, 180.0), (-180.0, -150.0)])   # beta / PPII
RAMA_TURN = ((-160.0, -20.0), [(-90.0, 45.0)])                         # right-handed a/turn
RAMA_TURN_LEFT = ((20.0, 100.0), [(-20.0, 90.0)])                      # left-handed alpha

# Feature layout.  `FEATURES` is the contract: the order is fixed, and the adversarial
# harness indexes the tuning / held-out split by name against it.
FEATURES: tuple[str, ...] = (
    "e2e_ca_A",
    "e2e_ca_per_link",
    "rg_ca_scaled",
    "rama_extended_frac",
    "rama_turn_frac",
    "hbond_intra_per_res",
    "buried_sasa_frac",
    "terminal_exposure",
    "contact_order_rel",
    "tf_tx_A", "tf_ty_A", "tf_tz_A",
    "tf_rx", "tf_ry", "tf_rz",
)

# The split the harness reports separately.  TUNING holds the conformational descriptors a
# corruption sampler would draw target values for (section 3b serialises exactly these as
# its target distribution), so they are the ones that could be matched by construction.
# HELD_OUT holds interface and packing descriptors that no sampler targets directly.  An
# AUC of 0.5 on TUNING alone is not evidence of anything; the HELD_OUT number is.
TUNING_FEATURES: tuple[str, ...] = (
    "e2e_ca_A", "e2e_ca_per_link", "rg_ca_scaled",
    "rama_extended_frac", "rama_turn_frac", "hbond_intra_per_res",
)
HELDOUT_FEATURES: tuple[str, ...] = (
    "buried_sasa_frac", "terminal_exposure", "contact_order_rel",
    "tf_tx_A", "tf_ty_A", "tf_tz_A", "tf_rx", "tf_ry", "tf_rz",
)

# Kept apart from FEATURES on purpose -- see the module docstring.
STAGING_FEATURES: tuple[str, ...] = (
    "peptide_length", "receptor_length",
    "receptor_n_segments", "receptor_max_run", "receptor_frag_lt5",
)

assert set(TUNING_FEATURES) | set(HELDOUT_FEATURES) == set(FEATURES)
assert not set(TUNING_FEATURES) & set(HELDOUT_FEATURES)


@dataclass
class Complex:
    """One staged complex, already split into peptide and receptor.

    Coordinates are Angstrom.  `traj` / `pep_traj` are kept so SASA and H-bonds can be
    recomputed without re-reading the file.
    """
    example_id: str
    path: str
    N: np.ndarray            # [L, 3] peptide backbone
    CA: np.ndarray
    C: np.ndarray
    pep_heavy: list[np.ndarray]     # per-residue heavy-atom coords
    rec_heavy: np.ndarray           # [M, 3] receptor heavy atoms, concatenated
    rec_resseq: list[int]
    resnames: list[str]
    traj: object             # mdtraj.Trajectory, full complex
    pep_traj: object         # mdtraj.Trajectory, peptide only
    pep_res_index: list[int]  # peptide residue indices within `traj`

    @property
    def length(self) -> int:
        return int(self.CA.shape[0])


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_complex(path: str | Path, example_id: str | None = None,
                 peptide_chain: int = 1, receptor_chain: int = 0) -> Complex | None:
    """Read a staged chain-A/chain-B complex.  Returns None if it cannot be profiled.

    The staging convention across CPSea, LNR and the LP sets is identical -- chain A is the
    receptor, chain B the binder, heavy atoms only, standard residues only, per the
    `REMARK 999 CHAIN MAP: R->A (target), L->B (binder)` header that
    `script_utils/preprocess_cpsea.py` writes.  `peptide_chain` / `receptor_chain` are
    chain *indices* into that file, not chain IDs, so a file whose chains are ordered
    differently is a caller problem and not silently absorbed here.

    Returning None rather than raising is deliberate: a profile job walks thousands of
    files and one unreadable record should cost that record, not the shard.
    """
    import mdtraj as md

    path = str(path)
    try:
        traj = md.load(path)
    except Exception:
        return None
    top = traj.topology
    if top.n_chains <= max(peptide_chain, receptor_chain):
        return None

    pep_atoms = top.select(f"chainid {peptide_chain}")
    rec_atoms = top.select(f"chainid {receptor_chain}")
    if len(pep_atoms) == 0 or len(rec_atoms) == 0:
        return None

    pep_residues = list(top.chain(peptide_chain).residues)
    if len(pep_residues) < MIN_PEPTIDE_LEN:
        return None

    xyz = traj.xyz[0] * 10.0        # mdtraj is nm; this module is Angstrom throughout

    def _bb(res, name):
        for a in res.atoms:
            if a.name == name:
                return xyz[a.index]
        return None

    N, CA, C, heavy = [], [], [], []
    for res in pep_residues:
        n, ca, c = _bb(res, "N"), _bb(res, "CA"), _bb(res, "C")
        if n is None or ca is None or c is None:
            return None        # an incomplete backbone makes torsions and frames undefined
        N.append(n); CA.append(ca); C.append(c)
        heavy.append(np.array([xyz[a.index] for a in res.atoms], dtype=float))

    rec_residues = list(top.chain(receptor_chain).residues)
    rec_heavy = xyz[np.asarray(rec_atoms, dtype=int)]   # `select` returns indices, not Atoms

    return Complex(
        example_id=example_id or Path(path).stem,
        path=path,
        N=np.array(N, dtype=float), CA=np.array(CA, dtype=float), C=np.array(C, dtype=float),
        pep_heavy=heavy,
        rec_heavy=rec_heavy,
        rec_resseq=[int(r.resSeq) for r in rec_residues],
        resnames=[r.name for r in pep_residues],
        traj=traj,
        pep_traj=traj.atom_slice(pep_atoms),
        pep_res_index=[r.index for r in pep_residues],
    )


# --------------------------------------------------------------------------------------
# Individual features.  Each is a module-level function so a test can pin it on its own and
# so the report can say which one moved without re-deriving it from the vector.
# --------------------------------------------------------------------------------------
def end_to_end(CA: np.ndarray) -> tuple[float, float]:
    """(raw CA(0)-CA(L-1) distance, the same per chain link).

    Per *link* rather than per residue: a 5-mer spans 4 links, and dividing by 5 would make
    the short-peptide end of the distribution shift for an arithmetic reason.
    """
    d = float(np.linalg.norm(CA[-1] - CA[0]))
    return d, d / max(1, len(CA) - 1)


def rg_scaled(CA: np.ndarray) -> float:
    """Radius of gyration over CA only, scaled by N^-0.6.

    CA-only rather than all-heavy-atom: the heavy-atom count is a function of the sequence,
    so an all-atom Rg would partly measure amino-acid composition, and composition differs
    between these sets for reasons that have nothing to do with conformation.
    """
    c = CA - CA.mean(axis=0, keepdims=True)
    rg = float(np.sqrt((c ** 2).sum(axis=1).mean()))
    return rg * (len(CA) ** RG_EXPONENT)


def _in_box(phi: float, psi: float, box) -> bool:
    (plo, phi_hi), psi_ranges = box
    if not (plo <= phi <= phi_hi):
        return False
    return any(lo <= psi <= hi for lo, hi in psi_ranges)


def ramachandran_occupancy(pep_traj) -> tuple[float, float]:
    """(extended fraction, turn fraction) over residues where both phi and psi exist.

    Computed on the LINEAR chain, so residue 0 has no phi and residue L-1 no psi and
    neither is counted.  A cyclic peptide would have both, but counting them would make
    this feature mean something different on the cyclic set than on the linear one, which
    is exactly the asymmetry the adversary must not be handed for free.
    """
    import mdtraj as md

    phi_idx, phi = md.compute_phi(pep_traj)
    psi_idx, psi = md.compute_psi(pep_traj)
    if phi.shape[1] == 0 or psi.shape[1] == 0:
        return float("nan"), float("nan")

    # Index by the residue the torsion belongs to: phi(i) uses C(i-1),N(i),CA(i),C(i) so it
    # is residue index of atom 1; psi(i) uses N(i),CA(i),C(i),N(i+1) so it is atom 0's.
    top = pep_traj.topology
    phi_res = {top.atom(ix[1]).residue.index: np.degrees(v)
               for ix, v in zip(phi_idx, phi[0])}
    psi_res = {top.atom(ix[0]).residue.index: np.degrees(v)
               for ix, v in zip(psi_idx, psi[0])}
    shared = sorted(set(phi_res) & set(psi_res))
    if not shared:
        return float("nan"), float("nan")

    ext = turn = 0
    for r in shared:
        p, s = float(phi_res[r]), float(psi_res[r])
        if _in_box(p, s, RAMA_EXTENDED):
            ext += 1
        elif _in_box(p, s, RAMA_TURN) or _in_box(p, s, RAMA_TURN_LEFT):
            turn += 1
    n = float(len(shared))
    return ext / n, turn / n


def intra_hbonds(pep_traj) -> int:
    """Backbone H-bond count within the peptide, by the Kabsch-Sander electrostatic model.

    Kabsch-Sander rather than Baker-Hubbard because the staged files are heavy-atom only:
    Baker-Hubbard needs explicit hydrogens and would return 0 on every record in all three
    sets, which reads as a real feature rather than a missing one.
    """
    import mdtraj as md

    try:
        mats = md.kabsch_sander(pep_traj)
    except Exception:
        return 0
    return int(mats[0].nnz) if mats else 0


def sasa_terms(cx: Complex) -> tuple[float, float]:
    """(buried SASA fraction over the whole peptide, mean relative exposure of the termini).

    Buried fraction is 1 - SASA(peptide in complex) / SASA(peptide alone), summed over
    residues before the ratio is taken so one tiny exposed residue cannot dominate.
    Terminal exposure is the mean over residues 0 and L-1 of the per-residue ratio, so it
    is high when the chain ends stick out of the pocket -- the quantity that separates a
    macrocycle with no free tails from a linear binder that has two.
    """
    import mdtraj as md

    sasa_cx = md.shrake_rupley(cx.traj, mode="residue")[0]
    sasa_alone = md.shrake_rupley(cx.pep_traj, mode="residue")[0]
    bound = sasa_cx[cx.pep_res_index]

    tot_alone = float(sasa_alone.sum())
    buried = float(np.clip(1.0 - float(bound.sum()) / tot_alone, 0.0, 1.0)) if tot_alone > 0 else float("nan")

    ends = []
    for k in (0, len(sasa_alone) - 1):
        a = float(sasa_alone[k])
        ends.append(float(np.clip(float(bound[k]) / a, 0.0, 1.0)) if a > 0 else np.nan)
    return buried, float(np.nanmean(ends)) if np.any(np.isfinite(ends)) else float("nan")


def crop_receptor(cx: Complex, radius_A: float) -> Complex:
    """Keep only receptor residues with a heavy atom within `radius_A` of the peptide.

    Applied UNIFORMLY to every set at profile time rather than by staging one of them,
    which is the point.  PepBench receptors are whole chains (median 223 residues) and
    CPSea receptors are pocket crops (139), so buried SASA, terminal exposure and contact
    order all separate the sets on staging rather than on conformation
    (README_POSE_DECOY_INVENTORY 2.5).  Cropping only PepBench would replace one asymmetry
    with a differently-shaped one; cropping everything with the same operation removes it
    by construction.

    Original residue numbering is preserved, so deleted residues show up as numbering gaps
    -- the same convention `scripts/restage_lnr_pocket.py` writes and the same one the
    staging statistics in `receptor_segments` are defined against.

    Returns a NEW Complex; the input is not modified.
    """
    import mdtraj as md

    top = cx.traj.topology
    pep_res = set(cx.pep_res_index)
    pep_all = np.concatenate(cx.pep_heavy) if cx.pep_heavy else np.zeros((0, 3))

    keep_res, keep_atoms = [], []
    for res in top.residues:
        if res.index in pep_res:
            keep_atoms.extend(a.index for a in res.atoms)
            continue
        xyz = cx.traj.xyz[0][[a.index for a in res.atoms]] * 10.0
        if pep_all.size and np.min(np.linalg.norm(
                xyz[:, None, :] - pep_all[None, :, :], axis=-1)) <= radius_A:
            keep_res.append(res)
            keep_atoms.extend(a.index for a in res.atoms)

    keep_atoms = sorted(set(keep_atoms))
    sub = cx.traj.atom_slice(np.asarray(keep_atoms, dtype=int))

    # atom_slice renumbers residues, so the peptide's indices inside the cropped trajectory
    # have to be recovered by position rather than carried over.
    old_to_new = {old: new for new, old in enumerate(keep_atoms)}
    new_pep_res = sorted({sub.topology.atom(old_to_new[a.index]).residue.index
                          for r in top.residues if r.index in pep_res for a in r.atoms})

    rec_heavy = np.concatenate([
        cx.traj.xyz[0][[a.index for a in r.atoms]] * 10.0 for r in keep_res
    ]) if keep_res else np.zeros((0, 3))

    return Complex(
        example_id=cx.example_id, path=cx.path,
        N=cx.N, CA=cx.CA, C=cx.C, pep_heavy=cx.pep_heavy,
        rec_heavy=rec_heavy,
        rec_resseq=[int(r.resSeq) for r in keep_res],
        resnames=cx.resnames,
        traj=sub,
        pep_traj=cx.pep_traj,
        pep_res_index=new_pep_res,
    )


def buried_sasa_per_residue(cx: Complex) -> np.ndarray:
    """Per-peptide-residue buried surface area, in nm^2: SASA(alone) - SASA(in complex).

    This is the quantity the anchor definition is stated in (`README_POSE_DECOY_INVENTORY`
    section 4.3: anchors are the target's top-k interface residues by buried surface area).
    Returned unnormalised so a caller can rank on it directly; ranking is scale-free, so
    the nm^2 / A^2 choice never reaches a result.
    """
    import mdtraj as md

    sasa_cx = md.shrake_rupley(cx.traj, mode="residue")[0]
    sasa_alone = md.shrake_rupley(cx.pep_traj, mode="residue")[0]
    return np.asarray(sasa_alone - sasa_cx[cx.pep_res_index], dtype=float)


def anchor_residues(cx: Complex, n_anchors: int) -> list[int]:
    """The `n_anchors` peptide residues that bury the most surface, as peptide-local indices.

    Ties are broken by residue index so the selection is reproducible across processes --
    an anchor set that depended on numpy's sort stability would make two runs of the same
    job disagree about which interface they were preserving.
    """
    bsa = buried_sasa_per_residue(cx)
    order = sorted(range(len(bsa)), key=lambda i: (-float(bsa[i]), i))
    return sorted(order[:max(0, int(n_anchors))])


def contact_order(pep_heavy: list[np.ndarray],
                  cutoff_A: float = HEAVY_CONTACT_CUTOFF_A,
                  min_sep: int = MIN_SEQ_SEP_CONTACT) -> float:
    """Relative contact order over intra-peptide residue contacts.

    RCO = mean(|i-j|) / L over contacting pairs.  An extended peptide makes no such
    contacts at all; that returns 0.0, which is the honest reading (no long-range order),
    not a missing value.
    """
    L = len(pep_heavy)
    seps = []
    for i in range(L):
        for j in range(i + min_sep, L):
            d = np.linalg.norm(pep_heavy[i][:, None, :] - pep_heavy[j][None, :, :], axis=-1)
            if float(d.min()) < cutoff_A:
                seps.append(j - i)
    if not seps:
        return 0.0
    return float(np.mean(seps)) / float(L)


def residue_frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal frame from one residue's N, CA, C.  Columns are the axes."""
    e1 = c - ca
    e1 /= np.linalg.norm(e1)
    u = n - ca
    e2 = u - np.dot(u, e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)


def terminal_frame_6dof(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> tuple[float, ...]:
    """Six-DOF transform from the N-terminal residue frame to the C-terminal one.

    (tx, ty, tz, rx, ry, rz): translation in Angstrom expressed IN the N-terminal frame,
    and the rotation as an axis-angle vector in radians.  Expressing it in the local frame
    is what makes it invariant to how the complex happens to sit in the file -- a global
    translation or rotation leaves all six numbers unchanged, so two loaders that centre
    the receptor differently still agree.
    """
    R0 = residue_frame(N[0], CA[0], C[0])
    R1 = residue_frame(N[-1], CA[-1], C[-1])
    t = R0.T @ (CA[-1] - CA[0])
    R = R0.T @ R1

    # Axis-angle from the rotation matrix, via the trace.  Guard both degenerate ends:
    # the trace can drift a hair outside [-1, 3] on floating point, and at theta = 0 the
    # axis is undefined (the vector is zero there, which is the correct answer).
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos))
    if theta < 1e-8:
        axis = np.zeros(3)
    elif abs(theta - np.pi) < 1e-6:
        # Near pi the skew part vanishes; recover the axis from the symmetric part.
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))] * theta
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2.0 * np.sin(theta)) * theta
    return (float(t[0]), float(t[1]), float(t[2]),
            float(axis[0]), float(axis[1]), float(axis[2]))


# --------------------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------------------
def profile(cx: Complex) -> "OrderedDict[str, float]":
    """Complex -> the feature vector.  Keys are exactly `FEATURES`, in that order."""
    e2e, e2e_link = end_to_end(cx.CA)
    ext, turn = ramachandran_occupancy(cx.pep_traj)
    buried, term_exp = sasa_terms(cx)
    tf = terminal_frame_6dof(cx.N, cx.CA, cx.C)

    out: "OrderedDict[str, float]" = OrderedDict()
    out["e2e_ca_A"] = e2e
    out["e2e_ca_per_link"] = e2e_link
    out["rg_ca_scaled"] = rg_scaled(cx.CA)
    out["rama_extended_frac"] = ext
    out["rama_turn_frac"] = turn
    out["hbond_intra_per_res"] = intra_hbonds(cx.pep_traj) / float(cx.length)
    out["buried_sasa_frac"] = buried
    out["terminal_exposure"] = term_exp
    out["contact_order_rel"] = contact_order(cx.pep_heavy)
    for name, v in zip(("tf_tx_A", "tf_ty_A", "tf_tz_A", "tf_rx", "tf_ry", "tf_rz"), tf):
        out[name] = v

    missing = set(FEATURES) - set(out)
    if missing:
        raise RuntimeError(f"profile() did not emit {sorted(missing)}")
    return out


def receptor_segments(rec_resseq: list[int], gap: int = SEGMENT_GAP) -> tuple[int, int, int]:
    """(n_segments, longest run, fragments shorter than 5) over receptor residue numbering.

    The pocket crop preserves original numbering, so a deleted residue shows up as a jump
    in resSeq.  These are the live statistics in `docs/README_TARGET_DISTRIBUTION_MATCHING.md`
    -- `n_segments` plateaus across every crop radius from 6 to 18 A and therefore does not
    discriminate, while `max_run` and `frag<5` do.
    """
    if not rec_resseq:
        return 0, 0, 0
    runs, cur = [], 1
    for a, b in zip(rec_resseq, rec_resseq[1:]):
        if b - a <= gap:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return len(runs), max(runs), sum(1 for r in runs if r < 5)


def staging_profile(cx: Complex) -> "OrderedDict[str, float]":
    """The confound vector: how the record was STAGED, not what the peptide is doing."""
    n_seg, max_run, frag = receptor_segments(cx.rec_resseq)
    out: "OrderedDict[str, float]" = OrderedDict()
    out["peptide_length"] = float(cx.length)
    out["receptor_length"] = float(len(cx.rec_resseq))
    out["receptor_n_segments"] = float(n_seg)
    out["receptor_max_run"] = float(max_run)
    out["receptor_frag_lt5"] = float(frag)
    return out


def profile_path(path: str | Path, example_id: str | None = None) -> dict | None:
    """Convenience: load and profile in one call.  Returns None if the file is unusable."""
    cx = load_complex(path, example_id=example_id)
    if cx is None:
        return None
    row = {"example_id": cx.example_id, "path": cx.path}
    row.update(profile(cx))
    row.update(staging_profile(cx))
    return row
