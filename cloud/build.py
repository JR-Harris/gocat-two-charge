'''
Cloud v2 build: assemble the substrate's journey (TS, the v2 IRC
trajectory, the densifying interpolations, the gripped endpoints),
union the per-atom Bondi envelopes exactly (manifold), Lloyd/CVT-sample
at 0.5 A with the v1 on-mesh projection fix, label every point's
construction provenance, run every QC gate, and write the canonical
artifacts + summary + README — once: the cloud LOCKS after a fully
green build (a later build refuses without --regenerate; the file +
sha256 is the provenance; a regenerated cloud is a DIFFERENT cloud
since the sampler exposes no RNG seed).

Densification rule (fixed): between a pair of journey geometries
separated by max-atom displacement d_max insert
k = ceil(d_max / 0.25 A) linearly interpolated frames (k = 0 when
d_max <= 0.25 A); applied to every adjacent trajectory-frame pair and
to each trajectory end and its corresponding gripped endpoint — the
pairing resolved BY CLASSIFICATION (the endpoints summary's source
records + per-end geometric classification), never by branch label.

The hull-area coverage gate is the v1 carry-over metric
(hull(projected cloud).area / hull(raw cloud).area in [0.99, 1.01]) --
Jonny's 2026-10-07 ruling on the briefed hull-vs-mesh gate, which had
surfaced its concavity artifact (0.9040 on the real build): the convex
hull cannot reproduce a concave swept surface's area by definition, so
the mesh-ratio measure is recorded only, never gated.
'''

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

from cloud import machinery
from cloud.machinery import (
    BONDI_RADII, SUBDIV, DENSIFY_STEP_A, TARGET_SPACING_A,
    V1_ANCHOR_AREA_A2, CALIBRATION_C_A, STATE_CODES, STATE_NAMES,
    N_ATOMS, CloudGateError,
    read_multiframe_xyz, kabsch_rmsd_core, interpolate_pair,
    union_of_envelopes, lloyd_project, nn_stats, hull_areas,
    envelope_depth, ts_contact_stats, provenance_labels,
    provenance_split,
)

DEFAULT_DERIVED = "data/derived"

CLOUD_OUTPUT_RELS = (
    "candidates_v2.npy", "candidates_v2_raw.npy", "candidates_v2.xyz",
    "candidates_v2_provenance.npy", "candidates_v2_spheres.npy",
)
INPUT_RELS = (
    "structures/ts_opt.xyz",
    "structures/chorismate_reactive_TSframe.xyz",
    "structures/prephenate_TSframe.xyz",
    "structures/chorismate_reactive_opt.xyz",
    "structures/prephenate_opt_chororder.xyz",
    "ts_summary.json",
    "alignment_summary.json",
    "p_registration_summary.json",
    "irc_endpoints_summary.json",
    "orca_v2/step3_irc/irc_IRC_Full_trj.xyz",
    "orca_v2/step3_irc/irc_IRC_F.xyz",
    "orca_v2/step3_irc/irc_IRC_B.xyz",
)

README_MARKER = "## v2 candidate cloud (swept-union, cloud v2)"


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _record_key(p, root, record):
    """The key under which a grip record hashes the file p: this world's
    joined path, or the canonical 'data/derived/<rel>' string the grip
    summaries recorded (so fixture worlds with copied canonical-string
    records still gate)."""
    candidates = (os.path.normpath(p),
                  os.path.join(DEFAULT_DERIVED,
                               os.path.relpath(os.path.normpath(p),
                                               os.path.normpath(root))))
    for cand in candidates:
        if cand in record:
            return cand
    raise ValueError(
        f"cloud v2 init: {os.path.basename(p)} is not hashed in the "
        f"grip record (tried {candidates[0]!r} and {candidates[1]!r}) "
        f"-- cannot gate its provenance")


# --------------------------------------------------------------------------
# input gating
# --------------------------------------------------------------------------
def gate_inputs(root):
    """Read every build input. Hard gates, in order: existence; the
    sha256s the grips' summaries recorded (the grip/query pattern's
    gating style — ts_opt, both re-optimised structures, ts_summary,
    irc_IRC_B against p_registration, the two gripped TS-frame files
    against the grips' outputs); the TS key-distance gate (vs
    ts_summary, no remapping); element-order identity (chorismate
    order) across all structures; trajectory shape + internal element
    consistency; the endpoints summary's pairing-record coheren
    ce.
    Returns (elements, ts, g_r, g_p, reopt_r, reopt_p, ep_f, ep_b,
             trj_frames, in_hashes)."""
    rel = lambda name: os.path.join(root, *name.split("/"))
    paths = {name: rel(name) for name in INPUT_RELS}
    for name, p in paths.items():
        if not os.path.isfile(p):
            raise ValueError(f"cloud v2 init: missing input {p} ({name})")

    def load_json(name):
        with open(paths[name]) as fh:
            return json.load(fh)

    alignment = load_json("alignment_summary.json")
    p_reg = load_json("p_registration_summary.json")
    ip_r, op_r = alignment["sha256"]["inputs"], alignment["sha256"]["outputs"]
    ip_p, op_p = p_reg["sha256"]["inputs"], p_reg["sha256"]["outputs"]
    # (file, record) pairs: the grip-side hashes know these files
    for p, record in ((paths["structures/ts_opt.xyz"], ip_r),
                      (paths["structures/chorismate_reactive_opt.xyz"], ip_r),
                      (paths["ts_summary.json"], ip_r),
                      (paths["structures/ts_opt.xyz"], ip_p),
                      (paths["structures/prephenate_opt_chororder.xyz"], ip_p),
                      (paths["ts_summary.json"], ip_p),
                      (paths["orca_v2/step3_irc/irc_IRC_B.xyz"], ip_p),
                      (paths["structures/chorismate_reactive_TSframe.xyz"],
                       op_r),
                      (paths["structures/prephenate_TSframe.xyz"], op_p)):
        key = _record_key(p, root, record)
        got = sha256_of(p)
        if got != record[key]:
            raise ValueError(
                f"cloud v2 init: sha256 of {p} ({got}) contradicts the "
                f"grip record ({record[key]}) -- the stage's inputs do "
                f"not match the grip records (fail-hard)")

    ts_els, ts = machinery._read_structure(
        paths["structures/ts_opt.xyz"], "the TS")
    krows = machinery._ts_key_distance_gate(
        paths["structures/ts_opt.xyz"], paths["ts_summary.json"], ts)
    if not all(r["pass"] for r in krows.values()):
        raise ValueError(
            "cloud v2 init: the TS key-distance gate failed -- stopping "
            "(no remapping attempted)")

    g_r_els, g_r = machinery._read_structure(
        paths["structures/chorismate_reactive_TSframe.xyz"], "gripped R")
    g_p_els, g_p = machinery._read_structure(
        paths["structures/prephenate_TSframe.xyz"], "gripped P")
    ro_r_els, ro_r = machinery._read_structure(
        paths["structures/chorismate_reactive_opt.xyz"], "reactive re-opt")
    ro_p_els, ro_p = machinery._read_structure(
        paths["structures/prephenate_opt_chororder.xyz"], "prephenate re-opt")
    ep_f_els, ep_f = machinery._read_structure(
        paths["orca_v2/step3_irc/irc_IRC_F.xyz"], "reactant-like endpoint")
    ep_b_els, ep_b = machinery._read_structure(
        paths["orca_v2/step3_irc/irc_IRC_B.xyz"], "product-like endpoint")
    machinery._element_order_gate([
        (paths["structures/ts_opt.xyz"], ts_els),
        (paths["structures/chorismate_reactive_TSframe.xyz"], g_r_els),
        (paths["structures/prephenate_TSframe.xyz"], g_p_els),
        (paths["structures/chorismate_reactive_opt.xyz"], ro_r_els),
        (paths["structures/prephenate_opt_chororder.xyz"], ro_p_els),
        (paths["orca_v2/step3_irc/irc_IRC_F.xyz"], ep_f_els),
        (paths["orca_v2/step3_irc/irc_IRC_B.xyz"], ep_b_els),
    ])

    trj_frames = read_multiframe_xyz(paths["orca_v2/step3_irc/"
                                           "irc_IRC_Full_trj.xyz"])
    trj_els, _ = trj_frames[0]
    if len({tuple(e) for e, _x in trj_frames}) != 1:
        raise ValueError(
            f"{paths['orca_v2/step3_irc/irc_IRC_Full_trj.xyz']}: the "
            f"element order varies between frames")
    if trj_els != ts_els:
        raise ValueError(
            f"element order differs (chorismate atom order broken): "
            f"{paths['structures/ts_opt.xyz']} vs "
            f"{paths['orca_v2/step3_irc/irc_IRC_Full_trj.xyz']}")
    if any(x.shape != (N_ATOMS, 3) for _e, x in trj_frames):
        raise ValueError(
            f"{paths['orca_v2/step3_irc/irc_IRC_Full_trj.xyz']}: a frame "
            f"is not {N_ATOMS} atoms")
    if len(trj_frames) < 3:
        raise ValueError(
            f"{paths['orca_v2/step3_irc/irc_IRC_Full_trj.xyz']}: "
            f"{len(trj_frames)} frames is not a trajectory")

    eps = json.load(open(paths["irc_endpoints_summary.json"]))
    src_f = (eps.get("reactive_reactant") or {}).get("source")
    src_b = (eps.get("prephenate") or {}).get("source")
    if os.path.basename(src_f or "") != "irc_IRC_F.xyz":
        raise ValueError(
            f"cloud v2 init: {paths['irc_endpoints_summary.json']}: the "
            f"reactant-like re-optimisation's source {src_f!r} is not "
            f"irc_IRC_F.xyz -- the pairing record is broken")
    if os.path.basename(src_b or "") != "irc_IRC_B.xyz":
        raise ValueError(
            f"cloud v2 init: {paths['irc_endpoints_summary.json']}: the "
            f"product-like re-optimisation's source {src_b!r} is not "
            f"irc_IRC_B.xyz")

    in_hashes = {paths[name]: sha256_of(paths[name]) for name in INPUT_RELS}
    return ts_els, ts, g_r, g_p, ro_r, ro_p, ep_f, ep_b, trj_frames, \
        in_hashes


# --------------------------------------------------------------------------
# frame gates + journey assembly
# --------------------------------------------------------------------------
def classify_endpoint_frame(x):
    """reactant-like | product-like | None per the endpoints stage's
    thresholds (O1-C7/C11-C16 at indices 0-6/10-15)."""
    d07 = float(np.linalg.norm(x[0] - x[6]))
    d1015 = float(np.linalg.norm(x[10] - x[15]))
    if d07 < 1.8 and 2.5 < d1015 <= 3.2:
        return "R"
    if d07 > 2.5 and d1015 < 2.0:
        return "P"
    return None


def assemble_journey(root):
    """Frame gates + the journey geometry list.

    Returns (elements, journey, journey_gates, interp_flags, counts,
    interp_log, in_hashes) with journey = [(state, frame, coords)] in
    construction order: the TS (state TS), the trajectory frames
    (path 0..T-1), the interpolated corridor frames (path T..),
    gripped R (state R), gripped P (state P). interp_flags is
    JOURNEY-LONG (one flag per journey entry — the 2026-10-14 corridor
    ruling: pairing by identity, never a lockstep zip of unequal
    lists), and its construction belt asserts the length + the True
    count against the recorded interpolations."""
    (elements, ts, g_r, g_p, ro_r, ro_p, ep_f, ep_b, trj_frames,
     in_hashes) = gate_inputs(root)
    trj_path = os.path.join(root, "orca_v2/step3_irc/irc_IRC_Full_trj.xyz")
    journey_gates = {}

    # the trajectory's image-0 vs the TS (v1 scoping: 0.186 A on its
    # layout = one IRC step off the TS; the gate: > 0.5 A fails)
    img0 = kabsch_rmsd_core(trj_frames[0][1], ts)
    image0_gate = {"observed_A": img0, "tol_A": machinery.IMAGE0_TOL_A,
                   "v1_scoping_anchor_A": 0.186,
                   "pass": img0 <= machinery.IMAGE0_TOL_A}
    journey_gates["image0_vs_ts"] = image0_gate
    if img0 > machinery.IMAGE0_TOL_A:
        raise ValueError(
            f"frame gate FAILED: {trj_path} image-0 loop RMSD vs the TS "
            f"is {img0!r} A (tol {machinery.IMAGE0_TOL_A}) -- stopping")

    # endpoint pairing BY CLASSIFICATION (resolved above via the
    # summary's source records): each endpoint Kabsch-fit onto its
    # corresponding re-optimised geometry, loop RMSD within tol (the
    # re-optimisations drifted only ~0.3 A at the loop)
    rmse_f = kabsch_rmsd_core(ep_f, ro_r)
    rmse_b = kabsch_rmsd_core(ep_b, ro_p)
    pairing_gate = {
        "reactant_like_endpoint_loop_rmsd_A": rmse_f,
        "product_like_endpoint_loop_rmsd_A": rmse_b,
        "tol_A": machinery.ENDPOINT_LOOP_RMSD_TOL_A,
        "pass": (rmse_f <= machinery.ENDPOINT_LOOP_RMSD_TOL_A
                 and rmse_b <= machinery.ENDPOINT_LOOP_RMSD_TOL_A)}
    journey_gates["endpoint_pairing"] = pairing_gate
    if not pairing_gate["pass"]:
        raise ValueError(
            "frame gate FAILED: an IRC endpoint pairs inconsistent with "
            f"its re-optimised geometry (reactant-like {rmse_f!r} A, "
            f"product-like {rmse_b!r} A, tol "
            f"{machinery.ENDPOINT_LOOP_RMSD_TOL_A}) -- stopping")

    # the interpolation pairing: each trajectory end goes to its
    # corresponding GRIPPED endpoint, by per-end classification +
    # geometric sanity
    end0, end1 = trj_frames[0][1], trj_frames[-1][1]
    c0, c1 = classify_endpoint_frame(end0), classify_endpoint_frame(end1)
    if c0 is None or c1 is None:
        bad = [t for t, c in (("trj[0]", c0), ("trj[-1]", c1)) if c is None]
        raise ValueError(
            f"frame gate FAILED: trajectory end(s) {bad} classify as "
            f"neither reactant-like nor product-like -- broken pairing "
            f"(O1-C7 {float(np.linalg.norm(end0[0] - end0[6])):.3f} A, "
            f"C11-C16 {float(np.linalg.norm(end0[10] - end0[15])):.3f} A "
            f"at trj[0])")
    # geometric sanity (registration-insensitive: the periphery swings
    # freely across the journey, so classify+fit-loop, not raw max
    # displacement)
    for tag, end, code in (("trj[0]", end0, c0), ("trj[-1]", end1, c1)):
        other = "P" if code == "R" else "R"
        d_own = kabsch_rmsd_core(end, g_r if code == "R" else g_p)
        d_other = kabsch_rmsd_core(end, g_r if other == "R" else g_p)
        if d_own > d_other:
            raise ValueError(
                f"frame gate FAILED: {tag} classifies {code} but is "
                f"loop-closer to the gripped {other} -- broken pairing")

    journey = [("TS", 0, ts.astype(float))]
    journey += [("path", f, x.astype(float))
                for f, (_e, x) in enumerate(trj_frames)]
    interp_flags = [False]                      # the TS entry
    interp_flags.extend([False] * len(trj_frames))   # each traj frame
    interp_log = []
    fid = len(trj_frames)
    for tag, end, code in (("trj[0]", end0, c0), ("trj[-1]", end1, c1)):
        target = g_r if code == "R" else g_p
        d_max = float(np.linalg.norm(end - target, axis=1).max())
        k = int(np.ceil(d_max / DENSIFY_STEP_A)) if d_max > DENSIFY_STEP_A \
            else 0
        interp_log.append({"joint": tag, "to": f"gripped {code}",
                           "d_max_A": d_max, "k": k})
        interps = interpolate_pair(end, target, k) if k else []
        for x in interps:
            journey.append(("path", fid, x.astype(float)))
            interp_flags.append(True)
            fid += 1
    journey += [("R", 0, g_r.astype(float)), ("P", 0, g_p.astype(float))]
    interp_flags += [False, False]

    counts = {"traj_frames": len(trj_frames),
              "interpolated": int(sum(int(e["k"]) for e in interp_log)),
              "geometries_total": len(journey)}
    # the construction belt (the 2026-10-14 corridor ruling): the flag
    # list is journey-long and its True count is the recorded
    # interpolations -- identity/count pairing, never a lockstep zip of
    # unequal lists
    if len(interp_flags) != len(journey):
        raise ValueError(
            f"corridor construction: the flag list ({len(interp_flags)} "
            f"entries) is not journey-long ({len(journey)} entries)")
    if sum(int(f) for f in interp_flags) != counts["interpolated"]:
        raise ValueError(
            f"corridor construction: {sum(int(f) for f in interp_flags)} "
            f"interpolated flags != the recorded "
            f"{counts['interpolated']} interpolations -- mismatched flag "
            "list")
    return (elements, journey, journey_gates, interp_flags, counts,
            interp_log, in_hashes)


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# the shared build core (the canonical build == the replicate stages)
# --------------------------------------------------------------------------
CORRECTNESS_GATES = ("shape_dtype", "nn_spacing_A", "bondi_conformance_A",
                     "projection_residual_A", "interpolation_corridor_A",
                     "provenance_completeness", "spacing_law_reanchor",
                     "hull_area_ratio")


def check_correctness_gates(gates, extra_names=()):
    """The shared correctness evaluation: every correctness gate row, in
    order (+ any call site's extension rows, e.g. the replicate stage's
    strict residual row), must pass -- else the gate's exact recorded
    numbers surface with the raise (fail hard, nothing written)."""
    for name in tuple(CORRECTNESS_GATES) + tuple(extra_names):
        if not gates[name]["pass"]:
            raise CloudGateError(f"{name} gate FAILED: "
                                 f"{json.dumps(gates[name])}")


def compute_build_core(elements, journey, interp_flags, counts, interp_log,
                       journey_gates, header="cloud v2", verbose=True,
                       union=None, steer_rolls=0):
    """The build core shared VERBATIM by the canonical build and the
    replicate stages (cloud/replicate.py, work item 4b): the exact
    watertight union of the journey's Bondi envelopes, the law-anchored
    Lloyd/CVT sampling at the 0.5 A target spacing, and the full QC
    gate set + per-point provenance computed on the fresh realization.
    Nothing is written here and no gate is raised here: every gate row
    is computed first (a failing gate surfaces its exact numbers at the
    call site through check_correctness_gates). header = the console
    prefix ('cloud v2' for the canonical build; 'replicate <NAME>' in
    the replicate CLI's identical prints). The canonical path owns the
    union build; a caller that has ALREADY built the surface (the
    replicate stages: the union identity is asserted BEFORE sampling)
    passes union= — the canonical path's behavior and outputs are
    unchanged.

    steer_rolls (the 2026-10-14 draw ruling, the item-4b replicate
    stages only; the canonical default 0 = byte-identical roll 0): the
    pinned point-cloud_utils sampler is fixed-seeded per process, so
    the canon cloud is roll 0 of that stream and replicate k's fresh
    draw is roll k — the draw process advances the stream by k
    DISCARDED sample_mesh_lloyd runs (each at the recorded n) before
    the recorded draw. The discards are not saved, not labelled and
    never enter any artifact; only the roll count is recorded
    (summary + manifest). Re-derivability = same-roll reruns agree to
    <= ~1e-12 A (the measured band; bitwise only where the float path
    is stable).

    Returns a dict: union, mesh_area_A2, C_init, n_init, raw,
    projected, post_off, ts_coords, centers, radii, states, frames,
    atoms, labels = (o_state, o_frame, o_atom, nearest_ts, origin_err),
    ts_proximity = (c_frac, w_frac, unsigned), hulls = (h_raw_a,
    h_proj_a, hull_ratio_vs_mesh), gates, journey_gates, counts,
    interp_log, journey_len.
    """
    _p = print if verbose else (lambda *a, **k: None)

    # the pairing asserts (the 2026-10-14 corridor ruling, first —
    # BEFORE any expensive work): the flag list must be journey-long,
    # its True count = the recorded interpolations, and every True must
    # sit on a path-state entry — a mismatched list fails HARD here
    # (never silently misalign mid-build)
    if len(interp_flags) != len(journey):
        raise ValueError(
            f"corridor pairing: the flag list ({len(interp_flags)} "
            f"entries) is not journey-long ({len(journey)} entries) -- "
            "pairing must be by identity (one flag per journey entry), "
            "never a lockstep zip of unequal lists")
    if sum(int(f) for f in interp_flags) != counts["interpolated"]:
        raise ValueError(
            f"corridor pairing: {sum(int(f) for f in interp_flags)} "
            f"interpolated flags != the recorded "
            f"{counts['interpolated']} interpolations -- mismatched "
            "flag list")
    if any(fl and j[0] != "path" for fl, j in zip(interp_flags, journey)):
        raise ValueError(
            "corridor pairing: a True flag on a non-path journey entry "
            "-- the interpolated corridor pairs with path-state entries "
            "only")

    if union is None:
        _p(f"{header}: exact union (manifold) of {len(journey) * N_ATOMS} "
           f"Bondi icospheres (subdiv {SUBDIV})")
        union = union_of_envelopes(
            elements, [np.asarray(x, dtype=float)
                       for _st, _f, x in journey])
    area = float(union.area)
    if verbose and union is None:
        _p(f"  union: {len(union.faces)} faces, area {area:.6f} A^2, "
           f"watertight")

    C_init = CALIBRATION_C_A * (area / V1_ANCHOR_AREA_A2) ** 0.5
    n_init = int(round((C_init / TARGET_SPACING_A) ** 2))
    _p(f"{header}: Lloyd/CVT at {TARGET_SPACING_A} A target spacing: "
       f"C_init {C_init:.6f} -> n_init = {n_init}")
    if steer_rolls:
        # the discarded rolls (the 2026-10-14 draw ruling, replicate
        # stages only): each = one full sample_mesh_lloyd run at the
        # recorded n; nothing is saved or labelled
        for roll_i in range(int(steer_rolls)):
            machinery.pcu.sample_mesh_lloyd(union.vertices, union.faces,
                                            n_init)
        _p(f"  draw roll {int(steer_rolls)}: {int(steer_rolls)} discarded "
           f"roll(s) at n = {n_init} advanced the fixed-stream sampler")
    raw, projected, _pre_off, post_off = lloyd_project(union, n_init)

    ts_coords = np.asarray(journey[0][2], dtype=float)
    centers = np.array([np.asarray(x, dtype=float)[i]
                        for _st, _f, x in journey for i in range(N_ATOMS)])
    radii = np.tile(np.array([BONDI_RADII[e] for e in elements]),
                    len(journey))
    states = np.array([STATE_CODES[st] for st, _f, _x in journey
                       for _i in range(N_ATOMS)], dtype=np.int64)
    frames = np.array([(0 if st in ("TS", "R", "P") else f)
                       for st, f, _x in journey
                       for _i in range(N_ATOMS)], dtype=np.int64)
    atoms = np.tile(np.arange(N_ATOMS, dtype=np.int64), len(journey))

    # ---- QC gates: every exact number computed first, recorded in the
    # summary; correctness gates fail hard in evaluation order, the
    # anticipated hull-vs-mesh concavity artifact last (its clause)
    gates = {}

    shape_ok = (projected.shape == (n_init, 3)
                and projected.dtype == np.float64)
    gates["shape_dtype"] = {"n": n_init,
                            "shape": list(projected.shape),
                            "dtype": str(projected.dtype),
                            "pass": shape_ok}

    depth_all = envelope_depth(projected, centers, radii)
    conform = float(np.abs(depth_all).max())
    gates["bondi_conformance_A"] = {
        "max_abs_envelope_depth_A": conform,
        "tol_A": machinery.CONFORMANCE_TOL_A,
        "pass": conform <= machinery.CONFORMANCE_TOL_A}

    resid = float(post_off.max())
    gates["projection_residual_A"] = {
        "max_A": resid, "tol_A": machinery.PROJECTION_RESIDUAL_TOL_A,
        "pass": resid <= machinery.PROJECTION_RESIDUAL_TOL_A}

    nn = nn_stats(projected)
    gates["nn_spacing_A"] = {
        "mean_A": nn["mean_A"], "std_A": nn["std_A"], "cv": nn["cv"],
        "min_A": nn["min_A"], "max_A": nn["max_A"],
        "cv_threshold": machinery.NN_CV_THRESHOLD,
        "target_A": TARGET_SPACING_A,
        "pass": nn["cv"] < machinery.NN_CV_THRESHOLD}

    h_raw_a, h_proj_a = hull_areas(raw, projected)
    v1_ratio = h_proj_a / h_raw_a
    disp = np.linalg.norm(projected - raw, axis=1)
    gates["projection_displacement_A"] = {
        "mean_A": float(disp.mean()), "max_A": float(disp.max()),
        "v1_anchor": {"mean": 0.021, "max": 0.0998}}

    corridor_geo = [x for keep, (_st, _f, x)
                    in zip(interp_flags, journey) if keep]
    # (the corridor pairing asserts already ran at the core's entry;
    # the zip below is identity-verified by them)
    if corridor_geo:
        icenters = np.array([
            np.asarray(x, dtype=float)[i]
            for x in corridor_geo for i in range(N_ATOMS)])
        iradii = np.tile(np.array([BONDI_RADII[e] for e in elements]),
                         len(corridor_geo))
    else:
        icenters = np.zeros((0, 3))
        iradii = np.zeros(0)
    cdepth = (envelope_depth(projected, icenters, iradii)
              if len(iradii) else np.zeros(len(projected)))
    cdepth_max = float(cdepth.max())
    gates["interpolation_corridor_A"] = {
        "max_depth_A": cdepth_max, "tol_A": machinery.CORRIDOR_DEPTH_TOL_A,
        "n_interpolated_spheres": int(len(iradii)),
        "pass": cdepth_max <= machinery.CORRIDOR_DEPTH_TOL_A,
        "note": "no point may sit deep inside any interpolated-frame "
                "envelope (a deeper point would mean the dense sweep "
                "corridor did not close)"}

    c_frac, w_frac, _depths = ts_contact_stats(projected, elements,
                                               ts_coords)
    unsigned = _ts_unsigned_contact(projected, ts_coords, elements)

    o_state, o_frame, o_atom, nearest_ts, origin_err = provenance_labels(
        projected, centers, radii, states, frames, atoms, ts_coords)
    complete = (len(o_state) == len(projected)
                and o_state.min() >= 0 and o_frame.min() >= 0
                and o_atom.min() >= 0 and nearest_ts.min() >= 0
                and float(origin_err.max())
                <= machinery.CONFORMANCE_TOL_A)
    gates["provenance_completeness"] = {
        "labeled": int(len(o_state)), "total": int(len(projected)),
        "max_origin_contact_err_A": float(origin_err.max()),
        "pass": bool(complete)}

    # spacing-law re-anchor: C derived from THIS build, validated at
    # two auxiliary n (one coarser, one finer, n +/-30%: the briefing's
    # "e.g." choice) -- each aux mean NN matches C/sqrt(n_aux) ~3%
    C_ach = float(np.sqrt(n_init) * nn["mean_A"])
    aux = []
    for n_aux, tag in ((int(round(n_init / 1.3)), "coarser"),
                       (int(round(n_init * 1.3)), "finer")):
        _r2, p2, __, ___ = lloyd_project(union, n_aux)
        nn2 = nn_stats(p2)
        pred = C_ach / np.sqrt(n_aux)
        off = abs(nn2["mean_A"] - pred) / pred
        aux.append({"n": n_aux, "kind": tag, "mean_nn_A": nn2["mean_A"],
                    "predicted_A": float(pred), "off_fraction": float(off),
                    "pass": off <= machinery.AUX_LAW_TOL_FRAC})
        _p(f"  re-anchor: n_aux = {n_aux} ({tag}): mean NN "
           f"{nn2['mean_A']:.4f} A vs predicted {pred:.4f} A "
           f"({100 * off:.1f}% off)")
    law_pass = all(a["pass"] for a in aux)
    gates["spacing_law_reanchor"] = {
        "C_achieved": C_ach, "auxiliary_runs": aux, "pass": law_pass,
        "tol_fraction": machinery.AUX_LAW_TOL_FRAC}

    hull_pass = (machinery.HULL_RATIO_BAND[0] <= v1_ratio
                 <= machinery.HULL_RATIO_BAND[1])
    gates["hull_area_ratio"] = {
        "projected_over_raw_hull": v1_ratio,
        "band": [machinery.HULL_RATIO_BAND[0],
                 machinery.HULL_RATIO_BAND[1]],
        "pass": hull_pass,
        "note": "hull(projected cloud).area / hull(raw cloud).area -- "
                "the v1 carry-over coverage metric, gated per the author's "
                "2026-10-07 ruling"}
    gates["hull_area_vs_mesh_record"] = {
        "projected_hull_over_mesh_area": h_proj_a / area,
        "point_hull_A2": h_proj_a, "mesh_area_A2": area,
        "note": "recorded only, NOT gated: the convex hull cannot "
                "reproduce a concave swept surface's area by "
                "definition (the surfaced concavity artifact of the "
                "briefed hull-vs-mesh gate; the ruling moved the gate "
                "to the v1 proj/raw metric)"}

    return {
        "union": union, "mesh_area_A2": area, "C_init": C_init,
        "n_init": n_init, "raw": raw, "projected": projected,
        "post_off": post_off, "ts_coords": ts_coords,
        "centers": centers, "radii": radii, "states": states,
        "frames": frames, "atoms": atoms,
        "labels": (o_state, o_frame, o_atom, nearest_ts, origin_err),
        "ts_proximity": (c_frac, w_frac, unsigned),
        "hulls": (h_raw_a, h_proj_a, h_proj_a / area),
        "gates": gates, "journey_gates": journey_gates,
        "counts": counts, "interp_log": interp_log,
        "journey_len": len(journey), "steer_rolls": int(steer_rolls),
    }


def cloud_summary(core, gates_all_pass=None):
    """The cloud_v2_summary.json dict (the canonical schema, exact key
    order) -- the replicate stage's summary is this + its 'replicate'
    block, so the two artifacts stay schema-siblings.

    gates_all_pass: None = the shared correctness gate set's verdict;
    a caller with an EXTENDED gate set (the replicate's strict residual
    row beyond CORRECTNESS_GATES) passes its own evaluated verdict."""
    gates = core["gates"]
    o_state = core["labels"][0]
    h_raw_a, h_proj_a, _ratio = core["hulls"]
    c_frac, w_frac, unsigned = core["ts_proximity"]
    return {
        "method": "exact watertight union (manifold) of per-atom Bondi "
                  "icospheres (subdiv 3) over the whole substrate "
                  "journey -- the TS + every frame of the v2 IRC "
                  "trajectory (TS-frame native by IRC construction) + "
                  "linear interpolants closing the end gaps to the "
                  "gripped R and P -- Lloyd/CVT-sampled at 0.5 A target "
                  "spacing with the v1 on-mesh projection fix; every "
                  "point carries its construction provenance (state, "
                  "atom) and a nearest_TS_atom class",
        "convention": "points are (n,3) float64 in the TS frame",
        "timestamp": _utc(),
        "bondi_radii": {k: BONDI_RADII[k] for k in ("H", "C", "O")},
        "icosphere_subdivisions": SUBDIV,
        "densify_rule": {
            "step_A": DENSIFY_STEP_A,
            "rule": "k = ceil(d_max / 0.25 A) linear interpolants "
                    "between adjacent journey geometries (0 when "
                    "d_max <= 0.25 A)"},
        "spacing_law": {
            "target_A": TARGET_SPACING_A, "C_init": core["C_init"],
            "rule": "n = round((C_init / 0.5)^2)",
            "v1_anchor_area_A2": V1_ANCHOR_AREA_A2,
            "v1_calibration_C": CALIBRATION_C_A},
        "n": core["n_init"],
        "mesh_area_A2": core["mesh_area_A2"],
        "journey_counts": {**core["counts"],
                           "spheres_total":
                               core["journey_len"] * N_ATOMS},
        "frame_gates": core["journey_gates"],
        "interpolations": core["interp_log"],
        "nn_stats": gates["nn_spacing_A"],
        "provenance_split": provenance_split(o_state),
        "ts_envelope_proximity": {
            "fractions_signed_max_depth": {
                "abs_depth_le_0p01": c_frac, "depth_ge_minus_0p5": w_frac},
            "fractions_unsigned_min_sphere_off": unsigned,
            "scoping_anchors": {"ts_contact": 0.298, "within_0p5": 0.856},
            "note": "point-level measures; the scoping numbers are "
                    "sanity anchors (report deltas; not gates)"},
        "gates": gates,
        "gates_all_pass": (
            all(gates[name]["pass"] for name in CORRECTNESS_GATES)
            if gates_all_pass is None else bool(gates_all_pass)),
        "hull_areas_A2": {"point_hull": h_proj_a, "raw_hull": h_raw_a},
        "sha256": {"inputs": {}, "outputs": {}},
    }


def build_v2_cloud(derived_root=None, regenerate=False, verbose=True):
    """The one-shot cloud build. A fully green run writes the
    candidates_v2* outputs + cloud_v2_summary.json + the README section
    and returns the summary dict; the tree LOCKS. The hull-vs-mesh
    breach alone surfaces to Jonny and writes nothing. (The sampling +
    gate section is the shared compute_build_core -- the replicate
    stages run this exact code path.)"""
    root = derived_root or DEFAULT_DERIVED
    summary_path = os.path.join(root, "cloud_v2_summary.json")
    _p = print if verbose else (lambda *a, **k: None)

    if os.path.isfile(summary_path) and not regenerate:
        locked = False
        try:
            with open(summary_path) as fh:
                locked = bool(json.load(fh).get("gates_all_pass"))
        except (ValueError, OSError):
            locked = False
        if locked:
            raise ValueError(
                f"{summary_path} records a fully green cloud v2 build "
                f"-- the cloud LOCKS (the files + their sha256s are the "
                f"provenance); a regenerated cloud is a DIFFERENT cloud "
                f"(the sampler exposes no RNG seed). Pass --regenerate "
                f"explicitly (the author's explicit action).")

    _p("cloud v2: gating inputs")
    (elements, journey, journey_gates, interp_flags, counts, interp_log,
     in_hashes) = assemble_journey(root)
    _p(f"  journey: TS + {counts['traj_frames']} trajectory frames + "
       f"{counts['interpolated']} interpolated + R + P = "
       f"{counts['geometries_total']} geometries")
    for j in interp_log:
        _p(f"    {j['joint']} -> {j['to']}: d_max {j['d_max_A']:.3f} A "
           f"-> k = {j['k']}")

    core = compute_build_core(elements, journey, interp_flags, counts,
                              interp_log, journey_gates,
                              header="cloud v2", verbose=verbose)
    check_correctness_gates(core["gates"])

    summary = cloud_summary(core)
    o_state, o_frame, o_atom, nearest_ts, _err = core["labels"]
    _write_outputs(root, core["raw"], core["projected"],
                   o_state, o_frame, o_atom, nearest_ts,
                   journey, elements, summary, in_hashes)
    _p("cloud v2: PASS -- the cloud LOCKS "
       f"({summary_path} + {len(CLOUD_OUTPUT_RELS)} outputs + README)")
    return summary


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ts_unsigned_contact(projected, ts_coords, elements):
    """The unsigned variant of the TS-envelope proximity (the min
    |d - r| over the TS spheres): recorded for the anchor comparison."""
    ts_coords = np.asarray(ts_coords, dtype=float)
    radii = np.array([BONDI_RADII[e] for e in elements])
    d = np.linalg.norm(projected[:, None, :] - ts_coords[None, :, :], axis=2)
    off = np.abs(d - radii[None, :]).min(axis=1)
    return {"min_sphere_off_le_0p01": float((off <= 0.01).mean()),
            "min_sphere_off_le_0p05": float((off <= 0.05).mean())}


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def json_hygiene(value):
    """The shared JSON-hygiene pass: numpy scalars (np.bool_ etc.) are
    not serialisable and no recorded gate value should be one anyway
    (the canonical writer's pass, shared with the replicate writers)."""
    if isinstance(value, dict):
        return {k: json_hygiene(x) for k, x in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_hygiene(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def journey_spheres(journey, elements):
    """The (state, frame, atom, center, radius) enumeration of every
    sphere in the journey union, in construction order (the canonical
    build's spheres artifact; the replicate stages reuse it for the
    surface-identity evidence)."""
    spheres = np.zeros(len(journey) * N_ATOMS, dtype=[
        ("state", "u1"), ("frame", "u4"), ("atom", "u1"),
        ("center", "f8", 3), ("radius", "f8")])
    k = 0
    for st, f, x in journey:
        coords = np.asarray(x, dtype=float)
        for i in range(N_ATOMS):
            spheres[k] = (STATE_CODES[st],
                          0 if st in ("TS", "R", "P") else f,
                          i, coords[i], BONDI_RADII[elements[i]])
            k += 1
    return spheres


def _write_outputs(root, raw, projected, o_state, o_frame, o_atom,
                   nearest_ts, journey, elements, summary, in_hashes):
    out = lambda name: os.path.join(root, *name.split("/"))
    np.save(out("candidates_v2.npy"), projected)
    np.save(out("candidates_v2_raw.npy"), raw)

    prov = np.zeros(len(projected), dtype=[
        ("origin_state", "u1"), ("origin_frame", "u4"),
        ("origin_atom", "u1"), ("nearest_ts_atom", "u1")])
    prov["origin_state"] = o_state
    prov["origin_frame"] = o_frame
    prov["origin_atom"] = o_atom
    prov["nearest_ts_atom"] = nearest_ts
    np.save(out("candidates_v2_provenance.npy"), prov)

    np.save(out("candidates_v2_spheres.npy"), journey_spheres(journey,
                                                              elements))

    with open(out("candidates_v2.xyz"), "w") as fh:
        fh.write(
            f"{len(projected)}\n"
            "v2 swept-union surface points (Lloyd/CVT at 0.5 A target "
            "+ the v1 on-mesh projection fix); provenance in "
            "candidates_v2_provenance.npy; see cloud_v2_summary.json\n")
        for x, y, z in projected:
            fh.write(f"X {x: .6f} {y: .6f} {z: .6f}\n")

    # a JSON-hygiene pass: numpy scalars (np.bool_ etc.) are not
    # serialisable and no recorded gate value should be one anyway
    summary["sha256"] = {
        "inputs": {os.path.normpath(p): sha for p, sha in
                   in_hashes.items()},
        "outputs": {os.path.normpath(out(name)): sha256_of(out(name))
                    for name in CLOUD_OUTPUT_RELS},
    }
    # find any remaining numpy scalars loudly: they mark a gate value
    # that was recorded as a numpy type and must be re-typed at source
    summary = json_hygiene(summary)
    with open(out("cloud_v2_summary.json"), "w") as fh:
        fh.write(json.dumps(summary, indent=2) + "\n")
    _write_readme(root, summary)


def _write_readme(root, summary):
    """Append/refresh the cloud-v2 section of data/derived/README.md
    (the tracked per-tree README, the v1 pattern)."""
    path = os.path.join(root, "README.md")
    out = summary["sha256"]["outputs"]
    gates = summary["gates"]
    body = f"""{README_MARKER}

Generated {summary['timestamp']} by `cloud/build.py` via
`run_cloud.py --build v2` -- this stage is the project's **one-time
cloud generation**: after this green build the cloud LOCKS (never
regenerated; the files + sha256 below are the provenance).

Surface: the exact watertight **swept-union** of per-atom Bondi
envelopes (H 1.20 / C 1.70 / O 1.52 A; icospheres subdiv
{summary['icosphere_subdivisions']}) over every geometry the substrate
passes through -- the TS, every frame of the v2 IRC trajectory
(TS-frame native by IRC construction), linear interpolants (rule
k = ceil(d_max/0.25 A) between adjacent journey geometries) closing
the end gaps to the gripped R and P, and the gripped endpoints
themselves: {summary['journey_counts']['geometries_total']} geometries =
{summary['journey_counts']['spheres_total']} spheres, union area
**{summary['mesh_area_A2']:.4f} Å²** (the TS-only envelope alone:
{machinery.V1_ANCHOR_AREA_A2:.4f} Å²).

Sampling: **{summary['n']} points** at the
{summary['spacing_law']['target_A']} Å target spacing (law n = (C/d)²,
C = {summary['spacing_law']['v1_calibration_C']} scaled by
sqrt(A_union/A_TS) = {summary['spacing_law']['C_init']:.6f}), Lloyd/CVT
via point-cloud-utils + the v1 on-mesh projection fix. Sample-space QC
(exact numbers in `cloud_v2_summary.json`): NN mean
{summary['nn_stats']['mean_A']:.4f} Å, CV {summary['nn_stats']['cv']:.4f}
(gate < 0.10); every point at vdW contact with the union envelope
(max |depth| {gates['bondi_conformance_A']['max_abs_envelope_depth_A']:.6f} Å
<= 0.01); post-projection mesh residual max
{gates['projection_residual_A']['max_A']:.2e}; corridor (interpolated
envelope) depth max {gates['interpolation_corridor_A']['max_depth_A']:.6f} Å
<= 0.01; hull project/raw {gates['hull_area_ratio']['projected_over_raw_hull']:.4f}
in {gates['hull_area_ratio']['band']} (the v1 coverage gate; the
hull-vs-mesh ratio {gates['hull_area_vs_mesh_record']['projected_hull_over_mesh_area']:.4f}
is recorded but not gated -- the convex hull cannot reproduce a
concave swept surface's area). **No mask, no screening** -- structural
viability only: every charge position is outside the substrate's
envelope at every moment of the journey.

Files (the candidates_v2* + summary are gitignored data; this README
records their sha256s):
- `candidates_v2.npy` -- the projected cloud, ({summary['n']},3) float64;
  sha256 `{out.get('data/derived/candidates_v2.npy', 'see json')[:16]}...`
- `candidates_v2_raw.npy` -- pre-projection CVT centroids (QC anchor);
  `{out.get('data/derived/candidates_v2_raw.npy', 'see json')[:16]}...`
- `candidates_v2.xyz` -- visualisation copy (dummy X element)
- `candidates_v2_provenance.npy` -- per-point structured labels
  (origin_state u1, origin_frame u4, origin_atom u1, nearest_ts_atom
  u1); states 0=TS 1=R 2=P 3=path; ties broken by state priority
  TS < R < P < path, then lower frame, then lower atom index
- `candidates_v2_spheres.npy` -- the (state, frame, atom, center,
  radius) enumeration of every sphere in the journey union

The v1 cloud (`candidates.npy` / `candidates_raw.npy` /
`candidates.xyz`, TS-envelope-only) is superseded by this swept-union
cloud and stays byte-identical as the reference + fallback (its own
sha256s live in the v1 records; the mini-cloud and clash artifacts
remain the v1 stage's records).
"""
    existing = open(path).read() if os.path.isfile(path) else ""
    marker_at = existing.find(README_MARKER)
    if marker_at >= 0:
        head = existing[:marker_at].rstrip("\n")
        tail = existing[marker_at:]
        end = tail.find("\n## ")
        after = tail[end:] if end >= 0 else ""
        if after.strip():
            existing = head.rstrip("\n")
            body = body + "\n" + after.lstrip("\n")
        else:
            existing = head.rstrip("\n")
    open(path, "w").write(
        "\n\n".join(part for part in (existing.rstrip("\n"), body.rstrip("\n"))
                    if part.strip()) + "\n")