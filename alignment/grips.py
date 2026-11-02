'''
The v2 grips: unweighted SVD Kabsch registration of the re-optimised
endpoint geometries onto the v2 TS frame through the six pericyclic-loop
core atoms.

The fitness evaluates frozen geometries in the TS frame; the
re-optimised minima are frame-blind (the raw IRC endpoints are
consistent with the TS frame only by IRC construction, and the
re-optimised geometries inherit that frame only accidentally). The grip
fitting each endpoint onto the TS frame through the six core atoms
(O1, C7, C9, C11, C14, C16 -- 0-based indices 0, 6, 8, 10, 13, 15 in
the chorismate atom order) is the registration everything else builds
on: the cloud build consumes both grips' aligned geometries.

Method (per grip):
  1. hard gate: the six TS key distances re-derived from the fixed
     0-based index pairs must reproduce ts_summary.json's values within
     1e-3 A -- any mismatch stops the run (no remapping attempted);
  2. element-order gate: all consumed structures carry the identical
     26-element chorismate order;
  3. standard unweighted SVD Kabsch fit of the core coordinate set P
     (endpoint) onto Q (TS), proper rotation enforced (det = +1,
     reflection term corrected per the sign of det(V U^T)); a
     degenerate core (rank-2 core: zero singular value) fails;
     convention: aligned = P @ R.T + t; the TS defines the frame;
  4. RMSD gates against the v1-artifact targets (fail-hard: printed
     observed/target/delta, then STOP -- widening is Jonny's decision);
  5. the aligned geometry written with full precision (14 dp, the v1
     write style); the summary records the frozen schema.

Outputs (both grips are the canonical-artifact producers of this
stage; each grip's write is its own step, and a re-run overwrites
exactly its own two artifacts):
  R-grip: data/derived/structures/chorismate_reactive_TSframe.xyz +
          data/derived/alignment_summary.json     (frozen schema,
          clash_*/mask_final retired with the 2026-10-06 swept-union
          pivot)
  P-grip: data/derived/structures/prephenate_TSframe.xyz +
          data/derived/p_registration_summary.json (the R schema plus
          the IRC-frame drift record)

Pure geometry: no ESP, no quantum chemistry, laptop-side; the stage
reads the canonical data/derived tree (the v1 orca tree, the QM
artifacts, the cloud files: strictly read-only) and writes only the
four paths above.
'''

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

from qm.structures import read_xyz_frame

DEFAULT_DERIVED = "data/derived"

# --- the grip: six pericyclic core atoms (chorismate atom order) ---
CORE_LABELS = ["O1", "C7", "C9", "C11", "C14", "C16"]
CORE_IDX = [0, 6, 8, 10, 13, 15]            # 0-based; 1-based: 1, 7, 9, 11, 14, 16
N_ATOMS = 26
PERIPHERY_IDX = [i for i in range(N_ATOMS) if i not in CORE_IDX]

# --- hard gate: ts_summary.json key -> (atom i, atom j), 0-based ---
GATE_PAIRS = [
    ("O1-C7 (breaking)", 0, 6),
    ("C11-C16 (forming)", 10, 15),
    ("O1-C14 (->C=O)", 0, 13),
    ("C14-C16", 13, 15),
    ("C7-C9", 6, 8),
    ("C9-C11", 8, 10),
]
KEY_DISTANCE_TOL_A = 1e-3
DET_TOL = 1e-9
DEGENERACY_TOL = 1e-8     # smallest-singular-value guard on the core

# --- RMSD gate references (the v1 artifact's / the scoping's values) ---
R_CORE_TARGET = 0.3247
R_PERIPHERY_TARGET = 0.6049
R_RMSD_TOL = 0.02
P_CORE_TARGET = 0.4858
P_PERIPHERY_TARGET = 0.6734
P_RMSD_TOL = 0.05         # scoping-grade looser band

PRODUCER = "alignment/grips.py (v2 grips stage)"


# ---------------------------------------------------------------------------
# primitives (also used directly by alignment/test_grips.py)
# ---------------------------------------------------------------------------
def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_xyz(path, elements, coords, comment):
    """XYZ with full precision (14 dp, the v1 write style: the values are
    the rotated originals, rendered -- not re-derived rounded floats)."""
    with open(path, "w") as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for el, (x, y, z) in zip(elements, coords):
            f.write(f"{el:<3s} {x: .14f} {y: .14f} {z: .14f}\n")


def kabsch(P, Q):
    """Unweighted SVD Kabsch fit of point set P onto Q (both (n,3), n >= 3).

    Returns (R, t) with aligned = (P @ R.T) + t (column form: R @ x + t).
    R is a proper rotation: det(R) = +1, reflections excluded (the
    reflection-correction term picks the sign of the last column so the
    determinant is +1 even when the best raw fit is improper). A
    degenerate core (rank < 3: collinear or exactly planar point set)
    raises ValueError.
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    if S[-1] < DEGENERACY_TOL:
        raise ValueError(
            f"degenerate core geometry: the core point set is collinear or "
            f"exactly planar (smallest singular value {S[-1]:.3e}); the fit "
            f"is not unique -- failing per spec")
    V = Vt.T
    d = float(np.sign(np.linalg.det(V @ U.T)))
    if d == 0.0:
        raise ValueError("degenerate core geometry: det(V U^T) = 0; "
                         "cannot enforce a proper rotation")
    R = V @ np.diag([1.0, 1.0, d]) @ U.T
    t = Qc - R @ Pc
    return R, t


def rmsd(A, B):
    """RMSD over corresponding rows of two (n,3) point sets."""
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def rotation_angle_deg(R):
    """The rotation angle of a proper rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(
        np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _read_structure(path, what):
    """Read a single-frame XYZ structure, gated on 26 atoms."""
    n, _comment, elements, coords = read_xyz_frame(path)
    if n != N_ATOMS:
        raise ValueError(f"{path}: {what}: expected {N_ATOMS} atoms, got {n}")
    return elements, np.array(coords, dtype=float)


def _element_order_gate(named):
    """All consumed structures carry the identical 26-element chorismate
    atom order. `named` = [(path, elements), ...]; raises ValueError
    naming both differing files."""
    ref_path, ref_els = named[0]
    for path, els in named[1:]:
        if els != ref_els:
            raise ValueError(
                f"element order differs (chorismate atom order broken): "
                f"{ref_path} vs {path}")


def _ts_key_distance_gate(ts_path, ts_summary_path, ts_coords):
    """The six TS key distances re-derived from the fixed 0-based index
    pairs must reproduce ts_summary.json within 1e-3 A (no remapping is
    attempted: any mismatch stops the run, naming both files)."""
    with open(ts_summary_path) as fh:
        summary = json.load(fh)
    key_distances = summary.get("key_distances")
    if not isinstance(key_distances, dict):
        raise ValueError(f"{ts_summary_path}: no key_distances record")
    rows = {}
    failed = []
    for key, i, j in GATE_PAIRS:
        if key not in key_distances:
            raise ValueError(f"{ts_summary_path}: key_distances has no "
                             f"entry {key!r}")
        d = float(np.linalg.norm(ts_coords[i] - ts_coords[j]))
        ref = float(key_distances[key])
        delta = abs(d - ref)
        ok = delta <= KEY_DISTANCE_TOL_A
        rows[key] = {"atoms": [i, j], "computed_A": d, "summary_A": ref,
                     "delta_A": delta, "pass": ok}
        if not ok:
            failed.append((key, d, ref, delta))
    if failed:
        detail = "; ".join(f"{k}: computed {d:.6f} vs summary {r:.6f} "
                           f"(delta {dl:.2e})" for k, d, r, dl in failed[:3])
        raise ValueError(
            f"TS key-distance gate FAILED: the fixed core indices do not "
            f"reproduce {ts_summary_path} from {ts_path} (tol "
            f"{KEY_DISTANCE_TOL_A} A) -- stopping, no remapping attempted: "
            f"{detail}")
    return rows


def _rmsd_gate(label, observed, target, tol):
    """One fail-hard RMSD gate: print exact observed/target/delta; on
    breach stop (widening is a human decision, not a stage one)."""
    delta = observed - target
    ok = abs(delta) <= tol
    print(f"  {label:22s} observed {observed:.6f}  target {target:.4f} "
          f"+/-{tol:.4f}  delta {delta:+.6f}  [{'ok' if ok else 'FAIL'}]")
    if not ok:
        raise ValueError(
            f"{label} gate FAILED: observed {observed!r} vs target "
            f"{target!r} +/-{tol} (delta {delta:+.6f} A) -- stopping "
            f"(widening is the author's decision, on the recorded deltas)")
    return observed


# ---------------------------------------------------------------------------
# shared build
# ---------------------------------------------------------------------------
def _fit_and_gates(endpoint_path, ts_path, ts_summary_path):
    """Shared fit pipeline: read, element gate, TS key-distance gate,
    Kabsch fit on the core, det/RMSD gates. Returns a record dict."""
    ts_els, ts = _read_structure(ts_path, "the TS structure")
    ep_els, ep = _read_structure(endpoint_path, "the endpoint structure")
    _element_order_gate([(ts_path, ts_els), (endpoint_path, ep_els)])
    gate_rows = _ts_key_distance_gate(ts_path, ts_summary_path, ts)

    R, t = kabsch(ep[CORE_IDX], ts[CORE_IDX])
    det = float(np.linalg.det(R))
    aligned = ep @ R.T + t
    core_rmsd = rmsd(aligned[CORE_IDX], ts[CORE_IDX])
    periph_rmsd = rmsd(aligned[PERIPHERY_IDX], ts[PERIPHERY_IDX])
    return {"ts_els": ts_els, "ep_els": ep_els, "ts": ts, "ep": ep,
            "R": R, "t": t, "det": det, "aligned": aligned,
            "core_rmsd": core_rmsd, "periph_rmsd": periph_rmsd,
            "gate_rows": gate_rows}


def _rmsd_report(rec, core_target, periph_target, tol):
    """The three fail-hard gates on one fitted record: the two RMSDs
    (observed/target/delta printed by _rmsd_gate) and det(R)."""
    _rmsd_gate("core RMSD", rec["core_rmsd"], core_target, tol)
    _rmsd_gate("periphery RMSD", rec["periph_rmsd"], periph_target, tol)
    det = rec["det"]
    ok = abs(det - 1.0) < DET_TOL
    print(f"  det(R)           observed {det:.12f}  target +1 "
          f"(tol {DET_TOL:.0e})  [{'ok' if ok else 'FAIL'}]")
    if not ok:
        raise ValueError(
            f"det(R) gate FAILED: det = {det!r}, required +1 within "
            f"{DET_TOL:.0e} -- stopping")


def _base_summary(method, gate_rows, rec, out_xyz_rel, out_xyz_path,
                  input_paths):
    """The frozen-schema summary body shared by both grips."""
    disp = np.linalg.norm(rec["aligned"] - rec["ts"], axis=1)
    utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "method": method,
        "convention": "aligned_x = R @ x + t (row form: X @ R.T + t)",
        "timestamp": utc,
        "core_atoms": [{"label": lab, "index": i}
                       for lab, i in zip(CORE_LABELS, CORE_IDX)],
        "atom_index_note": ("atom indices are 0-based (chorismate atom "
                            "order); labels are element + 1-based atom "
                            "number"),
        "gate_tolerance_A": KEY_DISTANCE_TOL_A,
        "gate": gate_rows,
        "rotation_matrix": [[float(v) for v in row] for row in rec["R"]],
        "translation_vector": [float(v) for v in rec["t"]],
        "det_rotation": rec["det"],
        "proper_rotation": abs(rec["det"] - 1.0) < DET_TOL,
        "core_rmsd_A": rec["core_rmsd"],
        "periphery_rmsd_A": rec["periph_rmsd"],
        "per_atom_displacement_A":
            [{"label": f"{rec['ep_els'][i]}{i + 1}", "index": i,
              "displacement_A": float(disp[i])} for i in range(N_ATOMS)],
        "sha256": {
            "inputs": {p: sha256_of(p) for p in input_paths},
            "outputs": {out_xyz_rel: sha256_of(out_xyz_path)},
        },
    }


def _write_summary(path, summary):
    with open(path, "w") as fh:
        fh.write(json.dumps(summary, indent=2) + "\n")


# ---------------------------------------------------------------------------
# the two grips
# ---------------------------------------------------------------------------
def build_r_grip(derived_root=None):
    """R-grip: fit chorismate_reactive_opt.xyz (v2) onto the v2 TS frame;
    write chorismate_reactive_TSframe.xyz + alignment_summary.json.
    Fail-hard; returns the summary dict."""
    root = derived_root or DEFAULT_DERIVED
    ts_path = os.path.join(root, "structures/ts_opt.xyz")
    r_path = os.path.join(root, "structures/chorismate_reactive_opt.xyz")
    ts_summary_path = os.path.join(root, "ts_summary.json")
    out_xyz = os.path.join(root, "structures/chorismate_reactive_TSframe.xyz")
    out_summary = os.path.join(root, "alignment_summary.json")

    print(f"R-grip: reading {r_path}, {ts_path} + {ts_summary_path}")
    rec = _fit_and_gates(r_path, ts_path, ts_summary_path)
    _rmsd_report(rec, R_CORE_TARGET, R_PERIPHERY_TARGET, R_RMSD_TOL)

    comment = (f"chorismate_reactive_opt Kabsch-aligned onto ts_opt via 6 "
               f"core atoms [{', '.join(CORE_LABELS)}], det=+1, core RMSD "
               f"{rec['core_rmsd']:.4f} A; produced by {PRODUCER}")
    os.makedirs(os.path.dirname(out_xyz), exist_ok=True)
    write_xyz(out_xyz, rec["ep_els"], rec["aligned"], comment)
    print(f"  wrote {out_xyz}")

    summary = _base_summary(
        "unweighted SVD Kabsch on the six pericyclic core atoms; "
        "chorismate_reactive_opt.xyz fitted onto ts_opt.xyz (TS defines "
        "the frame); proper rotation enforced (det=+1)",
        rec["gate_rows"], rec,
        os.path.join(root, "structures/chorismate_reactive_TSframe.xyz"),
        out_xyz,
        [ts_path, r_path, ts_summary_path])
    _write_summary(out_summary, summary)
    print(f"  wrote {out_summary}")
    print(f"R-grip: PASS (core {rec['core_rmsd']:.4f} A, periphery "
          f"{rec['periph_rmsd']:.4f} A, det {rec['det']:.12f})")
    return summary


def build_p_grip(derived_root=None):
    """P-grip: fit prephenate_opt_chororder.xyz (v2) onto the v2 TS frame;
    write prephenate_TSframe.xyz + p_registration_summary.json (the R
    schema plus the IRC-frame drift record). Fail-hard; returns the
    summary dict."""
    root = derived_root or DEFAULT_DERIVED
    ts_path = os.path.join(root, "structures/ts_opt.xyz")
    p_path = os.path.join(root, "structures/prephenate_opt_chororder.xyz")
    ts_summary_path = os.path.join(root, "ts_summary.json")
    irc_end_path = os.path.join(root, "orca_v2/step3_irc/irc_IRC_B.xyz")
    irc_summary_path = os.path.join(root, "irc_endpoints_summary.json")
    out_xyz = os.path.join(root, "structures/prephenate_TSframe.xyz")
    out_summary = os.path.join(root, "p_registration_summary.json")

    print(f"P-grip: reading {p_path}, {ts_path} + {ts_summary_path}")
    rec = _fit_and_gates(p_path, ts_path, ts_summary_path)
    _rmsd_report(rec, P_CORE_TARGET, P_PERIPHERY_TARGET, P_RMSD_TOL)

    # --- drift record: the rigid drift of the re-optimised product
    #     from its IRC frame. Both measures reconstruct the scoping
    #     values on the v1 chain (1.56 deg / ~0.30 A -- verified to
    #     3 sf / 4 sf there):
    #       * the ROTATION = the angle of the core-based rigid
    #         transform that carries the re-optimised geometry back
    #         into its IRC frame (its origin, the raw IRC endpoint the
    #         re-optimisation started from);
    #       * the settled LOOP displacement = the six core atoms'
    #         residuals AFTER the best full-atom rigid registration of
    #         the re-optimised geometry onto its origin (the non-rigid
    #         conformational drift at the loop).
    #     Recorded, no gate (the scoping values are the sanity
    #     reference). ---
    irc_els, irc_end = _read_structure(irc_end_path,
                                       "the IRC-frame product endpoint")
    _element_order_gate([(ts_path, rec["ts_els"]), (p_path, rec["ep_els"]),
                         (irc_end_path, irc_els)])
    R_back, _t_back = kabsch(rec["ep"][CORE_IDX], irc_end[CORE_IDX])
    drift_angle = rotation_angle_deg(R_back)
    R_full, t_full = kabsch(rec["ep"], irc_end)
    settled = np.linalg.norm(rec["ep"] @ R_full.T + t_full - irc_end, axis=1)
    loop_disp = settled[CORE_IDX]
    loop_rmsd = float(np.sqrt((loop_disp ** 2).mean()))
    # coherence: the re-optimisation's recorded source must be the drift
    # reference file this measurement consumed (basename level, so a
    # relocated test tree with canonical recorded strings still passes)
    if os.path.isfile(irc_summary_path):
        with open(irc_summary_path) as fh:
            src = json.load(fh).get("prephenate", {}).get("source")
        if not (isinstance(src, str)
                and os.path.basename(src) == os.path.basename(irc_end_path)):
            raise ValueError(
                f"{irc_summary_path}: prephenate source {src!r} disagrees "
                f"with the drift reference {irc_end_path!r}")

    comment = (f"prephenate_opt_chororder Kabsch-aligned onto ts_opt via 6 "
               f"core atoms [{', '.join(CORE_LABELS)}], det=+1, core RMSD "
               f"{rec['core_rmsd']:.4f} A; produced by {PRODUCER}")
    os.makedirs(os.path.dirname(out_xyz), exist_ok=True)
    write_xyz(out_xyz, rec["ep_els"], rec["aligned"], comment)
    print(f"  wrote {out_xyz}")

    summary = _base_summary(
        "unweighted SVD Kabsch on the six pericyclic core atoms; "
        "prephenate_opt_chororder.xyz fitted onto ts_opt.xyz (TS defines "
        "the frame); proper rotation enforced (det=+1)",
        rec["gate_rows"], rec,
        os.path.join(root, "structures/prephenate_TSframe.xyz"),
        out_xyz,
        [ts_path, p_path, ts_summary_path, irc_end_path])
    summary["irc_frame_drift"] = {
        "reference": irc_end_path,
        "note": ("rigid drift of the re-optimised product from its IRC "
                 "frame. The rotation is the angle of the core-based "
                 "rigid transform carrying the re-optimised geometry "
                 "back into its IRC frame; the loop displacements are "
                 "the six core atoms' residuals after the best "
                 "full-atom rigid registration onto the origin. Scoping "
                 "on the v1 chain measured 1.56 deg / ~0.30 A (loop "
                 "RMS); sanity reference, no gate"),
        "rotation_angle_deg": drift_angle,
        "per_loop_atom_displacement_A":
            [{"label": lab, "index": i,
              "displacement_A": float(d)}
             for lab, i, d in zip(CORE_LABELS, CORE_IDX, loop_disp)],
        "loop_rmsd_A": loop_rmsd,
    }
    _write_summary(out_summary, summary)
    print(f"  wrote {out_summary}")
    print(f"  IRC-frame drift: {drift_angle:.4f} deg / loop RMS "
          f"{loop_rmsd:.4f} A (max {loop_disp.max():.4f}) "
          f"(scoping on the v1 chain: 1.56 deg / ~0.30 A)")
    print(f"P-grip: PASS (core {rec['core_rmsd']:.4f} A, periphery "
          f"{rec['periph_rmsd']:.4f} A, det {rec['det']:.12f})")
    return summary