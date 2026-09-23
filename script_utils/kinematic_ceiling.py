"""Analytic closure-feasibility geometry for the Milestone 1.5 ceiling audit.

The question here is NOT "what does a sampler produce" but "what is geometrically
possible": holding a contiguous window of the peptide's CA trace at its native BOUND
position, can the remaining residues still satisfy a ring-closure constraint at all?

Everything is deliberately PERMISSIVE -- no sterics, no Ramachandran, no receptor
excluded volume, no side-chain packing.  The number this produces is therefore an UPPER
BOUND on achievable contact retention.  A measured retention close to it is near ceiling;
one far below it is a real deficit.  Erring permissive is the correct direction for a
ceiling: it can only make the model look worse, never better.

Two independent implementations are provided on purpose:

  * `feasible_analytic`  -- closed form, O(1), enumerable over every window of every
    complex.  Models the peptide as a virtual CA chain with fixed 3.80 A bonds and a
    bounded CA-CA-CA pseudo-angle, which reduces closure to an annulus-annulus distance
    range.  This is the primary instrument.
  * `feasible_torsion`   -- rebuilds a real N/CA/C backbone with NeRF from the held
    window outwards and runs a least-squares solve on the free phi/psi.  Slower, but it
    works in the actual torsion space rather than on the CA idealisation, so agreement
    between the two is a genuine check rather than a restatement.

Distances are Angstrom throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# Ideal backbone geometry.  Bond lengths / angles are the Engh-Huber values that every
# backbone builder in this repo already assumes; the CA-CA virtual bond is the trans-omega
# value.  TAU is the CA(i-1)-CA(i)-CA(i+1) pseudo-angle: it is bounded because phi/psi
# cannot open the virtual triangle arbitrarily, and its upper end is what sets how far an
# extended peptide can reach.
# --------------------------------------------------------------------------------------
CA_CA_A = 3.80
TAU_MIN_DEG = 78.0    # tightest turn a real backbone makes (alpha-helical is ~91 deg)
TAU_MAX_DEG = 150.0   # ideal-geometry extreme; fully extended beta sits at ~125-135 deg

N_CA_A = 1.458
CA_C_A = 1.525
C_N_A = 1.329
CA_CB_A = 1.532

ANG_N_CA_C = 111.0
ANG_CA_C_N = 116.2
ANG_C_N_CA = 121.7

# Contact definition, matched byte-for-byte to scripts/sdedit_cyclize.py's
# CONTACT_CUTOFF_NM = 1.0 so that a ceiling computed here is directly comparable to a
# `contact_retention` measured there.  A retention ratio against a DIFFERENT cutoff is a
# different quantity and must not be divided into this one.
CA_CONTACT_CUTOFF_A = 10.0
HEAVY_CONTACT_CUTOFF_A = 4.5


def rise_per_link(tau_deg: float = TAU_MAX_DEG) -> float:
    """Axial rise of one virtual CA-CA bond in a planar zigzag at pseudo-angle `tau`."""
    return CA_CA_A * math.sin(math.radians(tau_deg) / 2.0)


def reach_interval(n_links: int, tau_max_deg: float = TAU_MAX_DEG,
                   tau_min_deg: float = TAU_MIN_DEG) -> tuple[float, float]:
    """(min, max) end-to-end distance spanned by a virtual CA chain of `n_links` bonds.

    n_links >= 3 uses 0.0 for the minimum.  The true minimum is small but non-zero (a
    three-link chain folds back to ~2 A, not to 0), so this is permissive -- which is the
    side a ceiling should err on.  The maximum is exact for the stated pseudo-angle bound:
    the extremal conformation is the planar all-trans zigzag.
    """
    if n_links <= 0:
        return (0.0, 0.0)
    if n_links == 1:
        return (CA_CA_A, CA_CA_A)
    if n_links == 2:
        # Law of cosines on the isosceles triangle: d = 2 b sin(tau / 2).
        return (2.0 * CA_CA_A * math.sin(math.radians(tau_min_deg) / 2.0),
                2.0 * CA_CA_A * math.sin(math.radians(tau_max_deg) / 2.0))
    return (0.0, n_links * rise_per_link(tau_max_deg))


def annulus_distance_range(d_core: float, ra: tuple[float, float],
                           rb: tuple[float, float]) -> tuple[float, float]:
    """Achievable |P-Q| when P lies in the annulus `ra` about A, Q in `rb` about B, |AB| = d_core.

    Exact for the spherical-annulus idealisation: the extremes are collinear placements.
    """
    lo_a, hi_a = ra
    lo_b, hi_b = rb
    hi = d_core + hi_a + hi_b
    lo = max(0.0,
             d_core - hi_a - hi_b,      # both reaching toward each other
             lo_a - d_core - hi_b,      # P forced out past Q's reach
             lo_b - d_core - hi_a)
    return lo, hi


@dataclass(frozen=True)
class ClosureSpec:
    """One ring-closure constraint, expressed on the CA trace.

    `res_i` / `res_j` are binder-local residue indices.  `offset_i` / `offset_j` are the
    rigid bond distances from those residues' CA to the atoms that actually bond (0.0 when
    the constraint is already stated CA-to-CA).  `lo` / `hi` bound the acceptable bonded
    distance.
    """
    name: str
    res_i: int
    res_j: int
    offset_i: float
    offset_j: float
    lo: float
    hi: float


def mainchain_spec(length: int, tol_A: float = 0.25) -> ClosureSpec:
    """Head-to-tail: N of residue 0 bonded to C of residue L-1."""
    return ClosureSpec("mainchain", 0, length - 1, N_CA_A, CA_C_A,
                       C_N_A - tol_A, C_N_A + tol_A)


def bridge_spec(name: str, i: int, j: int, ca_lo: float, ca_hi: float) -> ClosureSpec:
    """Side-chain bridge stated directly as a CA(i)-CA(j) window.

    Stating the constraint on CA rather than on SG/NZ is deliberate: CB and beyond are
    rigidly slaved to the backbone, so a CA-CA window calibrated on real bonded pairs
    already absorbs the side-chain degrees of freedom without having to model rotamers.
    Calibrate `ca_lo` / `ca_hi` from natives (see m15_ceiling_audit.py --calibrate).
    """
    return ClosureSpec(name, i, j, 0.0, 0.0, ca_lo, ca_hi)


def _endpoint(ca_xyz: np.ndarray, k: int, offset: float, a: int, b: int,
              tau_max_deg: float,
              atom_xyz: np.ndarray | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Rigid attachment point and reach annulus for the atom on residue `k`.

    Inside the held window, `atom_xyz` (the atom's own native position) pins the endpoint
    exactly.  Without it the atom is modelled as free within its bond length, which is the
    weaker "hold the CA trace only" reading -- measurably loose at the feasibility
    boundary, because it lets a terminal N or C swing 1.5 A in any direction that the held
    backbone would not actually permit.
    """
    if a <= k <= b:
        if atom_xyz is not None:
            return atom_xyz, (0.0, 0.0)
        return ca_xyz[k], (0.0, offset)
    if k < a:
        lo, hi = reach_interval(a - k, tau_max_deg)
        anchor = ca_xyz[a]
    else:
        lo, hi = reach_interval(k - b, tau_max_deg)
        anchor = ca_xyz[b]
    return anchor, (max(0.0, lo - offset), hi + offset)


def feasible_analytic(ca_xyz: np.ndarray, spec: ClosureSpec, a: int, b: int,
                      tau_max_deg: float = TAU_MAX_DEG,
                      native_atom_dist: float | None = None,
                      atom_i_xyz: np.ndarray | None = None,
                      atom_j_xyz: np.ndarray | None = None) -> tuple[bool, float, float]:
    """Can `spec` close while residues [a, b] are held at their native bound positions?

    Returns (feasible, achievable_lo, achievable_hi).

    `atom_i_xyz` / `atom_j_xyz` are the native positions of the two bonded atoms.  Supply
    them: holding a window holds its backbone, so an in-window bonded atom is pinned, and
    modelling it as free within its bond length over-licenses closure by up to ~3 A of
    slack -- enough to call a window feasible that a torsion-space solve cannot close.
    Omitting them falls back to the looser "CA trace only" reading.

    When both constraint residues lie inside the held window the window is rigid, so the
    answer is not a range at all -- it is the structure's own measured atom distance.
    """
    i, j = spec.res_i, spec.res_j
    if a <= i <= b and a <= j <= b:
        if native_atom_dist is not None:
            lo = hi = float(native_atom_dist)
        elif atom_i_xyz is not None and atom_j_xyz is not None:
            lo = hi = float(np.linalg.norm(atom_i_xyz - atom_j_xyz))
        else:
            d = float(np.linalg.norm(ca_xyz[i] - ca_xyz[j]))
            lo = max(0.0, d - spec.offset_i - spec.offset_j)
            hi = d + spec.offset_i + spec.offset_j
        return (lo <= spec.hi and hi >= spec.lo), lo, hi

    anchor_i, ann_i = _endpoint(ca_xyz, i, spec.offset_i, a, b, tau_max_deg, atom_i_xyz)
    anchor_j, ann_j = _endpoint(ca_xyz, j, spec.offset_j, a, b, tau_max_deg, atom_j_xyz)
    d_core = float(np.linalg.norm(anchor_i - anchor_j))
    lo, hi = annulus_distance_range(d_core, ann_i, ann_j)
    return (lo <= spec.hi and hi >= spec.lo), lo, hi


# --------------------------------------------------------------------------------------
# Contact bookkeeping
# --------------------------------------------------------------------------------------
def ca_contact_set(pep_ca: np.ndarray, rec_ca: np.ndarray,
                   cutoff_A: float = CA_CONTACT_CUTOFF_A) -> set[tuple[int, int]]:
    """{(peptide_idx, receptor_idx)} within `cutoff_A` CA-CA -- sdedit_cyclize's definition."""
    d = np.linalg.norm(pep_ca[:, None, :] - rec_ca[None, :, :], axis=-1)
    pi, ri = np.where(d < cutoff_A)
    return {(int(p), int(r)) for p, r in zip(pi, ri)}


def retention_ceiling(contacts: set[tuple[int, int]], rec_ca: np.ndarray, ca_xyz: np.ndarray,
                      a: int, b: int, tau_max_deg: float = TAU_MAX_DEG,
                      cutoff_A: float = CA_CONTACT_CUTOFF_A) -> tuple[float, float]:
    """(strict, permissive) upper bounds on contact retention for the held window [a, b].

    strict     -- only contacts made by residues inside the held window survive.  Free
                  residues are written off entirely.
    permissive -- a free residue's contact also survives if its receptor partner is still
                  within that residue's reachable ball about the window edge.  This is the
                  true upper bound; `strict` is the conservative reading of "held fixed".
    """
    if not contacts:
        return float("nan"), float("nan")
    strict = {(p, r) for (p, r) in contacts if a <= p <= b}
    perm = set(strict)
    for (p, r) in contacts:
        if a <= p <= b:
            continue
        n = (a - p) if p < a else (p - b)
        anchor = ca_xyz[a] if p < a else ca_xyz[b]
        hi = reach_interval(n, tau_max_deg)[1]
        if float(np.linalg.norm(rec_ca[r] - anchor)) <= hi + cutoff_A:
            perm.add((p, r))
    n_all = float(len(contacts))
    return len(strict) / n_all, len(perm) / n_all


@dataclass
class CeilingResult:
    spec_name: str
    feasible_any: bool
    ceiling_strict: float
    ceiling_permissive: float
    best_a: int
    best_b: int
    best_window_len: int
    max_feasible_window_len: int
    n_feasible_windows: int
    n_windows: int
    native_feasible: bool
    # Filled in by `refine_with_torsion`.  The analytic ceiling above is an UPPER bound and
    # measurably loose at the boundary: on a real 10-mer it licenses closure with 1 free
    # residue where the backbone needs 5, because a short flank's reachable set is a cone
    # about the incoming chain direction, not the sphere the annulus model allows.
    ceiling_refined: float = float("nan")
    refined_a: int = -1
    refined_b: int = -1
    refined_window_len: int = 0
    refine_solves: int = 0
    refine_exhausted: bool = False


def rank_feasible_windows(ca_xyz: np.ndarray, contacts: set[tuple[int, int]],
                          rec_ca: np.ndarray, spec: ClosureSpec,
                          tau_max_deg: float = TAU_MAX_DEG,
                          cutoff_A: float = CA_CONTACT_CUTOFF_A,
                          native_atom_dist: float | None = None,
                          atom_i_xyz: np.ndarray | None = None,
                          atom_j_xyz: np.ndarray | None = None
                          ) -> list[tuple[float, float, int, int]]:
    """Every analytically-feasible window as (strict, permissive, a, b), best retention first."""
    length = int(ca_xyz.shape[0])
    out = []
    for a in range(length):
        for b in range(a, length):
            ok, _, _ = feasible_analytic(ca_xyz, spec, a, b, tau_max_deg, native_atom_dist,
                                         atom_i_xyz, atom_j_xyz)
            if not ok:
                continue
            strict, perm = retention_ceiling(contacts, rec_ca, ca_xyz, a, b, tau_max_deg, cutoff_A)
            out.append((strict, perm, a, b))
    out.sort(key=lambda t: (-t[0], -(t[3] - t[2])))
    return out


def refine_with_torsion(result: CeilingResult, candidates: list[tuple[float, float, int, int]],
                        N: np.ndarray, CA: np.ndarray, C: np.ndarray, spec: ClosureSpec,
                        max_solves: int = 40, n_restarts: int = 6, seed: int = 0,
                        tol_A: float = 0.25, max_seconds: float = 15.0) -> CeilingResult:
    """Tighten an analytic ceiling by confirming candidate windows in torsion space.

    Walks the candidates in descending retention and returns the first the solver can
    actually close.  The analytic pass is kept as the pre-filter because it is a genuine
    upper bound -- anything it rejects is truly infeasible -- so this only ever removes
    windows, never adds them.

    The prune is exact, not a heuristic: holding MORE residues fixed can never make
    closure easier, so once a window fails, every window containing it fails too.
    """
    failed: list[tuple[int, int]] = []
    solves = 0
    for strict, _, a, b in candidates:
        if any(a <= fa and fb <= b for fa, fb in failed):
            continue                      # superset of a known failure: infeasible for free
        if solves >= max_solves:
            result.refine_exhausted = True
            break
        solves += 1
        ok, _ = feasible_torsion(N, CA, C, spec, a, b, n_restarts=n_restarts, seed=seed,
                                 tol_A=tol_A, max_seconds=max_seconds)
        if ok:
            result.ceiling_refined = strict
            result.refined_a, result.refined_b = a, b
            result.refined_window_len = b - a + 1
            break
        failed.append((a, b))
    result.refine_solves = solves
    return result


def scan_windows(ca_xyz: np.ndarray, contacts: set[tuple[int, int]], rec_ca: np.ndarray,
                 spec: ClosureSpec, tau_max_deg: float = TAU_MAX_DEG,
                 cutoff_A: float = CA_CONTACT_CUTOFF_A,
                 native_atom_dist: float | None = None,
                 min_window: int = 1,
                 atom_i_xyz: np.ndarray | None = None,
                 atom_j_xyz: np.ndarray | None = None) -> CeilingResult:
    """Exhaustive O(L^2) scan for the held window that maximises retention under closure.

    Retention is not monotone in window size (contacts are not uniformly distributed along
    the peptide), so the maximum is taken over every feasible window rather than read off
    the largest one.
    """
    length = int(ca_xyz.shape[0])
    best = (-1.0, -1.0, -1, -1)
    max_len = 0
    n_feasible = 0
    n_windows = 0
    for a in range(length):
        for b in range(a + min_window - 1, length):
            n_windows += 1
            ok, _, _ = feasible_analytic(ca_xyz, spec, a, b, tau_max_deg, native_atom_dist,
                                         atom_i_xyz, atom_j_xyz)
            if not ok:
                continue
            n_feasible += 1
            max_len = max(max_len, b - a + 1)
            strict, perm = retention_ceiling(contacts, rec_ca, ca_xyz, a, b, tau_max_deg, cutoff_A)
            if strict > best[0]:
                best = (strict, perm, a, b)
    native_ok, _, _ = feasible_analytic(ca_xyz, spec, 0, length - 1, tau_max_deg,
                                        native_atom_dist, atom_i_xyz, atom_j_xyz)
    return CeilingResult(
        spec_name=spec.name,
        feasible_any=n_feasible > 0,
        ceiling_strict=best[0] if best[2] >= 0 else float("nan"),
        ceiling_permissive=best[1] if best[2] >= 0 else float("nan"),
        best_a=best[2], best_b=best[3],
        best_window_len=(best[3] - best[2] + 1) if best[2] >= 0 else 0,
        max_feasible_window_len=max_len,
        n_feasible_windows=n_feasible,
        n_windows=n_windows,
        native_feasible=bool(native_ok),
    )


# --------------------------------------------------------------------------------------
# NeRF backbone rebuild + torsion-space closure solve (the independent validator)
# --------------------------------------------------------------------------------------
def place_atom(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
               bond: float, angle_deg: float, torsion_deg: float) -> np.ndarray:
    """Natural extension reference frame: place p4 given the preceding three atoms."""
    ang = math.radians(angle_deg)
    tor = math.radians(torsion_deg)
    d = np.array([-bond * math.cos(ang),
                  bond * math.cos(tor) * math.sin(ang),
                  bond * math.sin(tor) * math.sin(ang)])
    bc = p3 - p2
    bc /= np.linalg.norm(bc)
    n = np.cross(p2 - p1, bc)
    nrm = np.linalg.norm(n)
    if nrm < 1e-9:                      # collinear input frame
        n = np.cross(bc, np.array([0.0, 0.0, 1.0]))
        nrm = np.linalg.norm(n)
        if nrm < 1e-9:
            n = np.cross(bc, np.array([0.0, 1.0, 0.0]))
            nrm = np.linalg.norm(n)
    n /= nrm
    m = np.stack([bc, np.cross(n, bc), n], axis=1)
    return p3 + m @ d


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral in degrees."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def backbone_torsions(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> dict[str, np.ndarray]:
    """Native phi / psi / omega arrays (degrees).  phi[0] and psi[-1] are undefined (nan)."""
    length = len(CA)
    phi = np.full(length, np.nan)
    psi = np.full(length, np.nan)
    omega = np.full(length, np.nan)
    for k in range(length):
        if k > 0:
            phi[k] = dihedral(C[k - 1], N[k], CA[k], C[k])
        if k < length - 1:
            psi[k] = dihedral(N[k], CA[k], C[k], N[k + 1])
            omega[k] = dihedral(CA[k], C[k], N[k + 1], CA[k + 1])
    return {"phi": phi, "psi": psi, "omega": omega}


def rebuild_from_window(N: np.ndarray, CA: np.ndarray, C: np.ndarray, a: int, b: int,
                        phi: np.ndarray, psi: np.ndarray, omega: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the backbone outward from the held window [a, b], which keeps native coords.

    Building outward (rather than forward from residue 0) is what makes the window
    genuinely fixed: a forward build would move the window whenever an upstream torsion
    changed, which silently converts "hold this window" into "hold nothing".
    """
    length = len(CA)
    Nn, CAn, Cn = N.copy(), CA.copy(), C.copy()

    for k in range(b, length - 1):      # forward: place residue k+1 off residue k
        Nn[k + 1] = place_atom(Nn[k], CAn[k], Cn[k], C_N_A, ANG_CA_C_N, psi[k])
        CAn[k + 1] = place_atom(CAn[k], Cn[k], Nn[k + 1], N_CA_A, ANG_C_N_CA, omega[k])
        Cn[k + 1] = place_atom(Cn[k], Nn[k + 1], CAn[k + 1], CA_C_A, ANG_N_CA_C, phi[k + 1])

    for k in range(a, 0, -1):           # backward: place residue k-1 off residue k
        Cn[k - 1] = place_atom(Cn[k], CAn[k], Nn[k], C_N_A, ANG_C_N_CA, phi[k])
        CAn[k - 1] = place_atom(CAn[k], Nn[k], Cn[k - 1], CA_C_A, ANG_CA_C_N, omega[k - 1])
        Nn[k - 1] = place_atom(Nn[k], Cn[k - 1], CAn[k - 1], N_CA_A, ANG_N_CA_C, psi[k - 1])

    return Nn, CAn, Cn


def _spec_residual(Nn: np.ndarray, CAn: np.ndarray, Cn: np.ndarray, spec: ClosureSpec) -> float:
    """Signed shortfall from the acceptable bonded-distance window (0 inside it)."""
    if spec.name == "mainchain":
        d = float(np.linalg.norm(Nn[spec.res_i] - Cn[spec.res_j]))
    else:
        d = float(np.linalg.norm(CAn[spec.res_i] - CAn[spec.res_j]))
    if d < spec.lo:
        return d - spec.lo
    if d > spec.hi:
        return d - spec.hi
    return 0.0


def feasible_torsion(N: np.ndarray, CA: np.ndarray, C: np.ndarray, spec: ClosureSpec,
                     a: int, b: int, n_restarts: int = 8, seed: int = 0,
                     tol_A: float = 0.1, max_seconds: float = 20.0,
                     max_nfev: int = 800) -> tuple[bool, float]:
    """Independent check: solve for free phi/psi that close the ring with [a, b] held.

    Returns (closed, best_absolute_residual_A).  Uses ideal bond geometry and native
    omega, so it tests the same idealisation the analytic bound assumes, but in real
    torsion space rather than on the CA trace -- disagreement is therefore informative.

    `max_seconds` bounds the whole call.  A successful restart returns immediately, so the
    budget only ever binds on failures, where the solver would otherwise run every restart
    to exhaustion -- which is how one pathological peptide walks a sharded job into its
    Slurm time limit.  A budget exhaustion reports NOT closed, which is the conservative
    answer for a validator whose job is to catch the analytic bound under-licensing.
    """
    import time as _time

    from scipy.optimize import least_squares

    t_start = _time.perf_counter()

    length = len(CA)
    tors = backbone_torsions(N, CA, C)
    phi0, psi0, omega = tors["phi"], tors["psi"], tors["omega"]
    phi0 = np.nan_to_num(phi0, nan=-120.0)
    psi0 = np.nan_to_num(psi0, nan=130.0)
    omega = np.nan_to_num(omega, nan=180.0)

    free = [k for k in range(length) if k < a or k > b]
    if not free:
        return (abs(_spec_residual(N, CA, C, spec)) <= tol_A,
                abs(_spec_residual(N, CA, C, spec)))

    def unpack(x):
        phi, psi = phi0.copy(), psi0.copy()
        for idx, k in enumerate(free):
            phi[k] = x[2 * idx]
            psi[k] = x[2 * idx + 1]
        return phi, psi

    def resid(x):
        phi, psi = unpack(x)
        Nn, CAn, Cn = rebuild_from_window(N, CA, C, a, b, phi, psi, omega)
        return np.array([_spec_residual(Nn, CAn, Cn, spec)])

    rng = np.random.default_rng(seed)
    x0 = np.concatenate([[phi0[k], psi0[k]] for k in free])
    best = float("inf")
    errors: list[str] = []
    for attempt in range(max(1, n_restarts)):
        if attempt and _time.perf_counter() - t_start > max_seconds:
            break
        start = x0 if attempt == 0 else x0 + rng.normal(0.0, 60.0, size=x0.shape)
        try:
            # 'trf', not 'lm': there is a single residual against many free torsions, and
            # Levenberg-Marquardt refuses an underdetermined system outright.  Swallowing
            # that refusal would silently report every case as infeasible.
            sol = least_squares(resid, start, method="trf", max_nfev=max_nfev)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised if universal
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        best = min(best, float(abs(sol.fun[0])))
        if best <= tol_A:
            break
    if not math.isfinite(best):
        raise RuntimeError(f"every torsion restart failed for window ({a}, {b}): {errors[:2]}")
    return best <= tol_A, best
