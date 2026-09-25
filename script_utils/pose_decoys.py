"""Rigid-body pose perturbation in the interface frame (pre-build check 1, stage A).

Generates perturbed binding poses for a native holo complex by rigidly moving the PEPTIDE
only, leaving the receptor untouched.  Each pose records the transform decomposed into the
interface normal and the two tangential directions, because "2 A of translation" means two
different things depending on which way it went: pulling 2 A off the interface destroys it,
sliding 2 A along the interface often does not.  A scalar amplitude cannot tell those apart,
and the calibration's whole question is which one `contact_retention` is tracking.

## The interface frame

  * **normal** -- the mean of the unit vectors from each contacting RECEPTOR heavy atom to
    its partnering PEPTIDE heavy atom.  Points out of the receptor into the peptide, so a
    positive normal displacement pulls the peptide off its site.  Averaging over contacts
    rather than over centroids keeps it meaningful for a peptide lying in a groove, where
    the centroid-to-centroid vector can point along the groove instead of out of it.
  * **t1** -- the leading principal axis of the peptide CA trace, projected into the plane
    perpendicular to the normal.  For an extended peptide this is the direction it runs in,
    which is the sliding direction that matters.
  * **t2** -- normal x t1, completing a right-handed orthonormal frame.

## The amplitude ladder

Poses are a stratified LADDER, not independent uniform draws.  A uniform draw leaves the
small-displacement end thin, and the small-displacement end is the operating range the
filter actually runs in -- the one regime where the answer has to be right.  Every complex
therefore contributes to every amplitude bin.

Direction modes cycle through `normal+`, `normal-`, `tangent` and `isotropic` so the
regression can separate normal from tangential displacement by design rather than hoping an
isotropic draw happened to populate both.

Pose 0 is always the identity: the unperturbed native, which is the before/after baseline
and the reference the collapse rate is measured against.

Runs in `.venv` (numpy / mdtraj).  No OpenMM here -- this stage only writes coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HEAVY_CONTACT_CUTOFF_A = 4.5
DIRECTION_MODES = ("normal+", "normal-", "tangent", "isotropic")


@dataclass
class InterfaceFrame:
    origin: np.ndarray        # peptide centroid
    normal: np.ndarray        # unit, receptor -> peptide
    t1: np.ndarray            # unit, in the plane perpendicular to `normal`
    t2: np.ndarray            # unit, normal x t1
    n_contacts: int

    @property
    def basis(self) -> np.ndarray:
        """Columns (normal, t1, t2): world = basis @ local."""
        return np.stack([self.normal, self.t1, self.t2], axis=1)

    def decompose(self, v: np.ndarray) -> tuple[float, float, float]:
        """World displacement -> (normal, t1, t2) components, in the same units."""
        return (float(np.dot(v, self.normal)),
                float(np.dot(v, self.t1)),
                float(np.dot(v, self.t2)))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def interface_frame(pep_heavy: list[np.ndarray], rec_heavy: np.ndarray,
                    pep_ca: np.ndarray,
                    cutoff_A: float = HEAVY_CONTACT_CUTOFF_A) -> InterfaceFrame:
    """Build the interface frame from the native pose's contacts."""
    pep_all = np.concatenate(pep_heavy) if pep_heavy else np.zeros((0, 3))
    origin = pep_ca.mean(axis=0)

    if pep_all.size and rec_heavy.size:
        d = np.linalg.norm(pep_all[:, None, :] - rec_heavy[None, :, :], axis=-1)
        pi, ri = np.where(d < cutoff_A)
    else:
        pi = ri = np.array([], dtype=int)

    if len(pi) == 0:
        # No contacts at the heavy-atom cutoff. Fall back to centroid-to-centroid, which is
        # weaker but defined; the caller sees n_contacts = 0 and can drop the complex.
        normal = _unit(origin - (rec_heavy.mean(axis=0) if rec_heavy.size else origin))
    else:
        vecs = pep_all[pi] - rec_heavy[ri]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normal = _unit((vecs / np.maximum(norms, 1e-9)).mean(axis=0))

    # Leading principal axis of the CA trace, projected off the normal.
    c = pep_ca - pep_ca.mean(axis=0, keepdims=True)
    if c.shape[0] >= 2:
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
    else:
        axis = np.array([1.0, 0.0, 0.0])
    t1 = axis - np.dot(axis, normal) * normal
    if float(np.linalg.norm(t1)) < 1e-6:
        # The chain runs along the normal; any perpendicular direction is as good.
        probe = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(probe, normal)) > 0.9:
            probe = np.array([0.0, 1.0, 0.0])
        t1 = probe - np.dot(probe, normal) * normal
    t1 = _unit(t1)
    return InterfaceFrame(origin=origin, normal=normal, t1=t1,
                          t2=_unit(np.cross(normal, t1)), n_contacts=int(len(pi)))


def detect_ring_bond(cx, max_bond_A: float = 2.2, min_res_sep: int = 3) -> dict | None:
    """The closed ring bond of a cyclic peptide, found by geometry, as ATOM indices.

    Detected here, in `.venv`, and carried through the manifest so the minimization stage
    never has to parse CONECT records: `infer_cyclization_label` lives behind the torch
    import stack, which is not installed in `.venv_openmm`, and the two environments are
    mutually exclusive by design.

    Geometry rather than CONECT because a rigid-body pose transform preserves the bond
    exactly, so the input distance IS the reference the restraint should hold. Any heavy
    atom pair at bonded distance whose residues are at least `min_res_sep` apart is a ring
    bond by construction -- chain neighbours cannot be that far apart in sequence.

    Returns None for a peptide with no such pair, which is the correct answer for a linear
    one and tells the caller to skip the hold restraint rather than invent a bond.
    """
    top = cx.traj.topology
    pep = set(cx.pep_res_index)
    atoms = [(a.index, a.residue.index) for r in top.residues if r.index in pep
             for a in r.atoms if a.element is not None and a.element.symbol != "H"]
    if len(atoms) < 2:
        return None
    idx = np.array([a for a, _ in atoms], dtype=int)
    res = np.array([r for _, r in atoms], dtype=int)
    xyz = cx.traj.xyz[0][idx] * 10.0

    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    sep = np.abs(res[:, None] - res[None, :])
    ok = (d > 0.1) & (d < max_bond_A) & (sep >= min_res_sep)
    if not ok.any():
        return None
    flat = int(np.argmin(np.where(ok, d, np.inf)))
    i, j = divmod(flat, len(idx))
    return {"ring_atom_i": int(idx[i]), "ring_atom_j": int(idx[j]),
            "ring_res_i": int(res[i]), "ring_res_j": int(res[j]),
            "ring_bond_A": float(d[i, j])}


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    axis = _unit(axis)
    th = np.radians(angle_deg)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


@dataclass
class Pose:
    index: int
    translation_A: float
    rotation_deg: float
    direction_mode: str
    d_normal_A: float = 0.0
    d_t1_A: float = 0.0
    d_t2_A: float = 0.0
    rot_axis: tuple = (0.0, 0.0, 1.0)
    translation_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def apply(self, xyz: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Rotate about `centre`, then translate.  Rotation first so the logged translation
        is the displacement of the centroid and not entangled with the rotation's own."""
        return (xyz - centre) @ self.rotation.T + centre + self.translation_world

    def as_row(self) -> dict:
        return {
            "pose_index": self.index,
            "translation_A": self.translation_A,
            "rotation_deg": self.rotation_deg,
            "direction_mode": self.direction_mode,
            "d_normal_A": self.d_normal_A,
            "d_t1_A": self.d_t1_A,
            "d_t2_A": self.d_t2_A,
            "d_tangential_A": float(np.hypot(self.d_t1_A, self.d_t2_A)),
            "rot_axis_x": float(self.rot_axis[0]),
            "rot_axis_y": float(self.rot_axis[1]),
            "rot_axis_z": float(self.rot_axis[2]),
            # The transform in full, so the minimization stage can rebuild it exactly.
            # That stage builds ONE hydrogenated system per complex (~63 s of a ~65 s
            # replica) and then re-poses it for each of the 16 poses; re-running pdbfixer
            # per pose instead would multiply the whole job's cost by sixteen.
            "tx_A": float(self.translation_world[0]),
            "ty_A": float(self.translation_world[1]),
            "tz_A": float(self.translation_world[2]),
        }


def build_ladder(frame: InterfaceFrame, n_poses: int, translations: list[float],
                 rotations: list[float], seed: int) -> list[Pose]:
    """The stratified pose ladder for one complex.

    Pose 0 is the identity.  The remaining `n_poses - 1` walk the amplitude ladder, cycling
    the direction mode so each complex contributes normal-only, tangential-only and
    isotropic displacements rather than whatever an isotropic draw happened to produce.
    """
    rng = np.random.default_rng(seed)
    poses = [Pose(index=0, translation_A=0.0, rotation_deg=0.0, direction_mode="identity")]

    n_pert = max(0, n_poses - 1)
    for i in range(n_pert):
        # Walk the ladder proportionally so any n_poses spans the configured range.
        frac = (i + 1) / n_pert
        t_amp = float(np.interp(frac, np.linspace(0, 1, len(translations)), translations))
        r_amp = float(np.interp(frac, np.linspace(0, 1, len(rotations)), rotations))
        mode = DIRECTION_MODES[i % len(DIRECTION_MODES)]

        if mode == "normal+":
            direction = frame.normal
        elif mode == "normal-":
            direction = -frame.normal
        elif mode == "tangent":
            phi = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.cos(phi) * frame.t1 + np.sin(phi) * frame.t2
        else:
            direction = _unit(rng.normal(size=3))

        tvec = t_amp * _unit(direction)
        axis = _unit(rng.normal(size=3))
        dn, d1, d2 = frame.decompose(tvec)
        poses.append(Pose(
            index=i + 1, translation_A=t_amp, rotation_deg=r_amp, direction_mode=mode,
            d_normal_A=dn, d_t1_A=d1, d_t2_A=d2,
            rot_axis=tuple(float(x) for x in axis),
            translation_world=tvec,
            rotation=rotation_matrix(axis, r_amp),
        ))
    return poses
