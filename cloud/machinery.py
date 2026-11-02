'''
Cloud v2 machinery: the shared geometry/sampling/QC primitives for the
swept-union point cloud.

Constructions confirmed from the v1 archive (semantic confirmation
only; code written fresh):
- Bondi radii table (v1 generate_candidates.BONDI_RADII, verbatim);
- icospheres: trimesh.creation.icosphere(subdivisions=3, radius=r) +
  apply_translation(center); union = trimesh.boolean.union(all)
  (manifold engine, watertight asserted);
- Lloyd/CVT: pcu.sample_mesh_lloyd(v, f, n) -> raw; projection fix:
  pcu.closest_points_on_mesh -> pcu.interpolate_barycentric_coords
  (the v1 fix for the ~0.02 A mean / ~0.1 A max off-surface drift of
  CVT centroids at sphere-intersection creases);
- NN stats via scipy cKDTree k=2; hull areas via scipy ConvexHull.

v2 additions: multi-frame journey assembly + densification, the
per-point provenance (construction_origin with the deterministic
tie-break, nearest_ts_atom), the corridor/interpolation gate, and the
spacing-law re-anchor at auxiliary n.
'''

import numpy as np
import point_cloud_utils as pcu
import scipy.spatial
import trimesh

from alignment.grips import (
    CORE_IDX,                       # the six pericyclic loop atoms
    CORE_LABELS, N_ATOMS,
    kabsch, rmsd, rotation_angle_deg,
    _read_structure, _element_order_gate, _ts_key_distance_gate,
    KEY_DISTANCE_TOL_A,
)

# --- frozen constants (v1 values where inherited) ---
BONDI_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}
SUBDIV = 3                       # icosphere subdivision (the v1 mesh standard)
DENSIFY_STEP_A = 0.25            # corridor interpolation spacing
TARGET_SPACING_A = 0.5
V1_ANCHOR_AREA_A2 = 228.404502   # the TS-only subdiv-3 union anchor (2026-09-17)
CALIBRATION_C_A = 14.366228      # d = C/sqrt(n) on the TS envelope (v1 empirical)

NN_CV_THRESHOLD = 0.10
CONFORMANCE_TOL_A = 0.01
CORRIDOR_DEPTH_TOL_A = 0.01
PROJECTION_RESIDUAL_TOL_A = 1e-9
# the hull-area coverage gate: hull(projected)/hull(raw), the v1
# carry-over metric (Jonny's 2026-10-07 ruling; the briefed hull-vs-
# mesh gate was surfaced as a concavity artifact and is recorded only)
HULL_RATIO_BAND = (0.99, 1.01)
AUX_LAW_TOL_FRAC = 0.03
ENDPOINT_LOOP_RMSD_TOL_A = 0.6   # IRC endpoint <-> re-optimised geometry pairing
IMAGE0_TOL_A = 0.5               # trajectory image-0 vs the TS loop

STATE_CODES = {"TS": 0, "R": 1, "P": 2, "path": 3}
STATE_NAMES = ("TS", "R", "P", "path")


class CloudGateError(ValueError):
    """A QC gate breach; carries the gate's machine name."""


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------
def read_multiframe_xyz(path):
    """Multi-frame XYZ -> list of (elements, coords (n,3))."""
    lines = open(path).read().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].strip())
        except ValueError as e:
            raise ValueError(f"{path}: bad frame atom-count line {i + 1!r}: "
                             f"{lines[i]!r} ({e})")
        if i + 2 + n > len(lines):
            raise ValueError(f"{path}: truncated frame at line {i + 1}")
        els = [lines[i + 2 + j].split()[0] for j in range(n)]
        xyz = np.array([[float(v) for v in lines[i + 2 + j].split()[1:4]]
                        for j in range(n)], dtype=float)
        frames.append((els, xyz))
        i += 2 + n
    if not frames:
        raise ValueError(f"{path}: no frames")
    return frames


def kabsch_rmsd_core(moving, onto):
    """Kabsch-fit the six loop atoms of `moving` onto `onto` (same
    atom order) and return the resulting loop RMSD."""
    R, t = kabsch(np.asarray(moving)[CORE_IDX], np.asarray(onto)[CORE_IDX])
    aligned = np.asarray(moving) @ R.T + t
    return rmsd(aligned[CORE_IDX], np.asarray(onto)[CORE_IDX])


def interpolate_pair(a, b, k):
    """k linearly interpolated geometries between rows a and b
    (per-atom Cartesian linear interpolation; k = 0 gives [])."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return [a + (j + 1) * (b - a) / (k + 1) for j in range(k)]


def atom_icosphere(element, center, subdiv=SUBDIV):
    """An icosphere of the atom's Bondi radius, centred at `center`
    (the v1 construction)."""
    m = trimesh.creation.icosphere(subdivisions=subdiv,
                                   radius=BONDI_RADII[element])
    m.apply_translation(np.asarray(center, dtype=float))
    return m


def union_of_envelopes(elements, geometries):
    """The exact watertight union of one Bondi icosphere per atom of
    every geometry (v1: trimesh.boolean.union of the sphere meshes).
    `geometries` = iterable of (n,3) coordinate arrays."""
    meshes = []
    for coords in geometries:
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] != len(elements):
            raise ValueError("union_of_envelopes: a journey geometry has "
                             f"{coords.shape[0]} atoms, expected "
                             f"{len(elements)}")
        for i, el in enumerate(elements):
            meshes.append(atom_icosphere(el, coords[i]))
    union = trimesh.boolean.union(meshes)
    if not union.is_watertight:
        raise ValueError(
            "the swept-union surface is not watertight after the "
            "manifold union -- not a buildable surface")
    return union


def lloyd_project(mesh, n):
    """Lloyd/CVT sample + the v1 on-mesh projection fix.

    Returns (raw (n,3), projected (n,3), pre_off (n,), post_off (n,))."""
    raw = np.asarray(pcu.sample_mesh_lloyd(mesh.vertices, mesh.faces, n),
                     dtype=np.float64)
    if raw.shape != (n, 3):
        raise CloudGateError(
            f"shape/dtype: Lloyd returned {raw.shape}, expected ({n}, 3)")
    if raw.dtype != np.float64:
        raw = raw.astype(np.float64)
    _d, f_idx, bc = pcu.closest_points_on_mesh(raw, mesh.vertices, mesh.faces)
    projected = np.asarray(
        pcu.interpolate_barycentric_coords(mesh.faces, f_idx, bc,
                                           mesh.vertices), dtype=np.float64)
    pre, _, _ = pcu.closest_points_on_mesh(raw, mesh.vertices, mesh.faces)
    post, _, _ = pcu.closest_points_on_mesh(projected, mesh.vertices,
                                            mesh.faces)
    return (np.ascontiguousarray(raw), np.ascontiguousarray(projected),
            np.asarray(pre, dtype=np.float64), np.asarray(post,
                                                          dtype=np.float64))


def nn_stats(points):
    """Nearest-neighbour distance statistics (v1: cKDTree k=2)."""
    tree = scipy.spatial.cKDTree(points)
    d, _ = tree.query(points, k=2)
    nn = np.asarray(d[:, 1], dtype=np.float64)
    return {"mean_A": float(nn.mean()), "std_A": float(nn.std()),
            "cv": float(nn.std() / nn.mean()), "min_A": float(nn.min()),
            "max_A": float(nn.max())}


def hull_areas(raw, projected):
    """(h_raw.area, h_proj.area) for the hull-ratio reporting."""
    return (float(scipy.spatial.ConvexHull(raw).area),
            float(scipy.spatial.ConvexHull(projected).area))


def envelope_depth(points, centers, radii):
    """Signed depth into a sphere-union envelope per point:
    max_k (r_k - |p - c_k|). Positive = inside; ~0 = on the surface."""
    points = np.asarray(points, dtype=np.float64)
    out = np.empty(len(points), dtype=np.float64)
    chunk = max(1, int(2e5 // max(1, len(radii))))
    for s in range(0, len(points), chunk):
        blk = points[s:s + chunk]
        d = np.linalg.norm(blk[:, None, :] - centers[None, :, :], axis=2)
        out[s:s + chunk] = (radii[None, :] - d).max(axis=1)
    return out


def ts_contact_stats(projected, ts_elements, ts_coords, tol=CONFORMANCE_TOL_A):
    """Point-level TS-envelope contact: the signed depth into the
    TS-only envelope; 'contact' = |depth| <= tol; within-0.5 =
    depth >= -0.5. Returns (contact_frac, within_0p5_frac, depths)."""
    ts_coords = np.asarray(ts_coords, dtype=np.float64)
    radii = np.array([BONDI_RADII[e] for e in ts_elements])
    depths = envelope_depth(projected, ts_coords, radii)
    contact = np.abs(depths) <= tol
    within = depths >= -0.5
    return float(contact.mean()), float(within.mean()), depths


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def provenance_labels(points, centers, radii, states, frames, atoms,
                      ts_coords, err_eps=9):
    """Per-point construction provenance, deterministic.

    For each point the sphere minimising |dist(point, center) - radius|
    over ALL journey spheres; exact ties (err rounded to 1e-9) broken
    by state priority TS < R < P < path, then lower frame index, then
    lower atom index (np.lexsort, last key = primary).

    Returns (origin_state int8[n], origin_frame int32[n],
             origin_atom int8[n], nearest_ts_atom int8[n],
             origin_err float64[n])."""
    points = np.asarray(points, dtype=np.float64)
    derr = np.abs(np.linalg.norm(points[:, None, :] - centers[None, :, :],
                                 axis=2) - radii[None, :])
    sh = derr.shape
    order = np.lexsort((np.broadcast_to(atoms, sh),
                        np.broadcast_to(frames, sh),
                        np.broadcast_to(states, sh),
                        np.round(derr, err_eps)), axis=1)
    best = order[:, 0]
    rows = np.arange(len(points))
    origin_err = np.round(derr, err_eps)[rows, best]
    nearest_ts = np.argmin(np.linalg.norm(points[:, None, :]
                                          - ts_coords[None, :, :], axis=2),
                           axis=1)
    return (states[best].astype(np.int8), frames[best].astype(np.int32),
            atoms[best].astype(np.int8), nearest_ts.astype(np.int8),
            origin_err)


def provenance_split(states_labels):
    """Fraction of points per construction state (dict of str -> frac)."""
    n = len(states_labels)
    counts = np.bincount(np.asarray(states_labels, dtype=int), minlength=4)
    return {STATE_NAMES[s]: float(counts[s]) / n for s in range(4)}
