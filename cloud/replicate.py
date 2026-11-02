'''
Per-replicate pipeline (work item 4b; the CLI briefing + the
2026-10-14 amendment/ruling): for a NAMED replicate, rebuild the
locked canonical union surface and PROVE it is the locked one (the
surface-identity stage), generate a fresh Lloyd/CVT realization of the
candidate cloud at the 0.5 A target through the SAME build core the
canonical build ran (cloud/build.py compute_build_core), run the
identical QC gate set at the canonical numbers (plus the ruling's
strict residual row), PROVE the realization is not the canonical draw
and MEASURE the difference (the comparison stage), then write the
whole artifact set under the replicate's own data root with full
provenance (cloud_<NAME>_summary.json in the canonical summary
schema + cloud_<NAME>_comparison.json + manifest.json).

THE SAMPLER — measured 2026-10-14, ruled (ruling on Finding 2, option
(a) as realized by the fallback-with-record):

  * The pinned point-cloud_utils sampler is seeded INTERNALLY, fixed
    per process (no setter anywhere: no python surface, no binary
    symbol; libc srand is a no-op on it; sample_mesh_random's
    random_seed does not reach the lloyd stream).
  * THE RULING: the canonical cloud is roll 0 of that fixed stream;
    replicate k draws roll k — the draw process advances the stream by
    k DISCARDED sample_mesh_lloyd runs (each at the recorded n) before
    the recorded draw; N (= the roll) + a one-line mechanism
    description are recorded in the summary + manifest. Distinctness
    is by construction (a duplicate roll among sibling replicates
    refuses, ask-first); the re-derivability band = same-roll reruns
    agree to <= ~1e-12 A (the measured agreement values recorded in
    the log; bitwise only where the float path is actually stable).
  * Measured side-fact: fresh stream-position-1 draws reproduce the
    canonical cloud to 4.3e-12 A (the canonical is re-derivable on
    this pin — recorded, no action).
'''

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.spatial

from cloud import build as cloud_build
from cloud import machinery
from cloud.machinery import (
    N_ATOMS, STATE_NAMES, STATE_CODES,
    union_of_envelopes, nn_stats,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CANONICAL_ROOT = os.path.join("data", "derived")
REPLICATES_ROOT = os.path.join("data", "replicates")

# --- the locked surface (the briefing's frozen numbers; verified
# against the LIVE canonical summary at every surface stage) ---------
LOCKED_SURFACE = {
    "mesh_area_A2": 256.7826490037527,
    "n": 928,
    "spheres_total": 2340,
    "geometries_total": 90,
    "icosphere_subdivisions": 3,
    "bondi_radii": {"H": 1.2, "C": 1.7, "O": 1.52},
}
LOCKED_AREA_TOL_A2 = 1e-3          # the allowed relaxation, spec 2
LOCKED_CLOUD_SHA256 = (
    "084e48a6b5267900a995ba2d4f0412f77f5bb13446ba74eafaf474c644349e71")
LOCKED_CANONICAL_SUMMARY_SHA256 = None   # recorded at the surface stage

# the replicate stage's strict residual gate (the amended spec 3:
# post-projection off-surface residual <= 1e-12; the shared core's
# canonical row keeps the machinery constant)
STRICT_RESIDUAL_TOL_A = 1e-12

# --- name fence ------------------------------------------------------
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
# canonical artifact-prefix fence (a name beginning with any of these
# tokens is confusable with the canonical tree's artifacts)
NAME_FENCE_PREFIXES = (
    "canonical", "candidates", "cloud", "mini", "atom_", "alignment",
    "irc_", "neb_", "ts_", "orca", "p_registration", "r_clash",
    "enumeration", "structures", "archive", "hpc",
)
RESERVED_NAMES = {"data", "derived", "replicates"}

# the manifest's key for every written file
MANIFEST_NAME = "manifest.json"


class ReplicateError(ValueError):
    """A replicate-stage failure (a gate, a fence or a stage rule)."""


class FenceError(ReplicateError):
    """The hard fence: the replicate CLI refuses canonical-space
    writes / canonical-confusable names."""


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def npy_sha256(arr):
    """The sha256 of exactly the bytes np.save will write (computed
    in-memory: identical to the written file's hash)."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(arr), allow_pickle=False)
    return sha256_bytes(buffer.getvalue())


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# name validation + the hard fence (spec 1; regression-tested)
# ---------------------------------------------------------------------------
def _canonical_stems(canonical_root=CANONICAL_ROOT):
    """The LIVE canonical tree's artifact stems (filenames minus their
    last extension, any depth; READMEs included — the fence is
    conservative). The dynamic part of the name fence: a generated
    replicate artifact stem colliding with any of these refuses."""
    stems = set()
    for dirpath, _dirs, files in os.walk(canonical_root):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            stem, _dot, _ext = fn.rpartition(".")
            stems.add(stem or fn)
    return stems


def _generated_stems(name):
    """The artifact stems the replicate stage will write under its
    root (the spec-5 artifact set + the manifest)."""
    return (
        f"candidates_{name}", f"candidates_{name}_raw",
        f"candidates_{name}_provenance", f"cloud_{name}",
        f"cloud_{name}_summary", f"cloud_{name}_comparison",
    )


def validate_name(name, roll=None, canonical_root=CANONICAL_ROOT):
    """spec 1: ^[a-z0-9][a-z0-9_-]{0,40}$; the name 'canonical' and any
    name matching a canonical artifact prefix is refused; a name whose
    generated artifact stems collide with a live canonical artifact
    stem is refused; path-component pointers (data / derived /
    replicates) are refused. Everything about the run's WRITE surface
    is decided by the name alone — so the fence on the name IS the
    fence on the tree."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise FenceError(
            f"--name {name!r}: refused -- the name must match "
            r"^[a-z0-9][a-z0-9_-]{0,40}$ (no path components, no "
            "pointer tricks)")
    if name == "canonical":
        # THE 2026-10-14 GENERALIZATION RULING: canonical is a VALID
        # name -- the program is name-agnostic and canonical is just
        # another cloud. A canonical re-derivation is always ANNOUNCED:
        # explicit --roll 0 is REQUIRED (the confined root
        # data/replicates/canonical/; data/derived/ is never touched)
        if roll != 0:
            raise FenceError(
                "--name canonical: requires explicit --roll 0 (a "
                "re-derivation is always announced, never implicit; "
                f"got roll={roll!r})")
        return name
    if name in RESERVED_NAMES:
        raise FenceError(
            f"--name {name!r}: refused -- a canonical-tree path "
            "component")
    for prefix in NAME_FENCE_PREFIXES:
        if name.startswith(prefix) or prefix.startswith(name):
            raise FenceError(
                f"--name {name!r}: refused -- matches the canonical "
                f"artifact prefix {prefix!r}")
    for stem in sorted(_canonical_stems(canonical_root)):
        for generated in _generated_stems(name):
            if stem == generated:
                raise FenceError(
                    f"--name {name!r}: refused -- the generated artifact "
                    f"{generated}.{name}* collides with the canonical "
                    f"artifact {stem}.* (the replicate root must not be "
                    "confusable with the canonical tree)")
    return name


def replicate_root(name):
    """The only permitted write root: data/replicates/<NAME>/ (the
    resolved path is fenced: under data/replicates/, never under the
    canonical tree)."""
    root = os.path.join(REPLICATES_ROOT, name)
    resolved = os.path.realpath(os.path.join(REPO_ROOT, root))
    base = os.path.realpath(os.path.join(REPO_ROOT, REPLICATES_ROOT))
    if not (resolved.startswith(base + os.sep)
            and os.path.basename(resolved) == name):
        raise FenceError(
            f"{root}: refused -- the replicate root must resolve to "
            f"{base}/<NAME>")
    fence_write(resolved, "the replicate root")
    return root


def fence_write(path, what=""):
    """THE HARD FENCE (spec 1): no replicate write may resolve under
    the canonical data/derived/ tree. Regression-tested; called on the
    replicate root itself and again at every write site."""
    canonical = os.path.realpath(os.path.join(REPO_ROOT, CANONICAL_ROOT))
    resolved = os.path.realpath(path)
    if resolved == canonical or resolved.startswith(canonical + os.sep):
        raise FenceError(
            f"{path}{f' ({what})' if what else ''}: refused -- the "
            "replicate CLI never writes under the canonical tree "
            f"({canonical}); hard fence")
    return resolved


def _locked_summary_check(canonical_root):
    """The live canonical summary must BE the locked record (the
    briefing's frozen surface numbers verified against the tree the
    replicate stages read as their reference)."""
    path = os.path.join(canonical_root, "cloud_v2_summary.json")
    if not os.path.isfile(path):
        raise ReplicateError(
            f"{path}: the canonical cloud summary is missing -- the "
            "replicate stages' reference surface is absent")
    live = json.load(open(path))
    observed = {
        "mesh_area_A2": live.get("mesh_area_A2"),
        "n": live.get("n"),
        "spheres_total": (live.get("journey_counts") or {}).get(
            "spheres_total"),
        "geometries_total": (live.get("journey_counts") or {}).get(
            "geometries_total"),
        "icosphere_subdivisions": live.get("icosphere_subdivisions"),
        "bondi_radii": live.get("bondi_radii"),
    }
    drift = [k for k, v in LOCKED_SURFACE.items() if observed.get(k) != v]
    if drift:
        raise ReplicateError(
            f"{path}: the live canonical summary contradicts the locked "
            f"surface on {drift} -- the locked record and the tree have "
            "drifted (refuse: fix the reference first)")
    return live


# ---------------------------------------------------------------------------
# the surface stage (spec 2: the same-surface guarantee)
# ---------------------------------------------------------------------------
def _inputs_match_locked(in_hashes, canonical_root, live_summary):
    """The rebuilt inputs vs the locked record's input sha256s, keyed
    ROOT-RELATIVELY (the record's keys are the canonical strings
    'data/derived/<rel>'; a fixture-root run has different path forms --
    the FILES and their hashes are the comparison, not the strings)."""
    live_rel = {}
    for key, sha in live_summary["sha256"]["inputs"].items():
        marker = "data" + os.sep + "derived" + os.sep
        live_rel[key.split(marker, 1)[-1]] = sha
    mine = {}
    for p, sha in in_hashes.items():
        rel = os.path.relpath(os.path.normpath(p),
                              os.path.normpath(canonical_root))
        mine[rel] = sha
    return mine == live_rel


def assert_surface_identity(journey, elements, journey_gates, in_hashes,
                            union, mesh_area, canonical_root=CANONICAL_ROOT):
    """THE identity assert (spec 2, the same-surface guarantee): the
    rebuilt surface against the locked canonical one -- the area EXACT
    when the rebuild is bitwise-clean (the only allowed relaxation is
    the briefing's 1e-3 A^2 band), the sphere COUNT exactly, watertight;
    every breach fails hard with the numbers. The recorded evidence
    (spheres bitwise live, the frame gates + input hashes vs the locked
    record) rides along in the returned record."""
    live = _locked_summary_check(canonical_root)
    exact = mesh_area == LOCKED_SURFACE["mesh_area_A2"]
    delta = abs(mesh_area - LOCKED_SURFACE["mesh_area_A2"])
    identity = {
        "rebuilt_mesh_area_A2": mesh_area,
        "locked_mesh_area_A2": LOCKED_SURFACE["mesh_area_A2"],
        "abs_delta_A2": delta,
        "bitwise_clean": exact,
        "tolerance_A2": LOCKED_AREA_TOL_A2,
        "sphere_count": len(journey) * N_ATOMS,
        "locked_sphere_count": LOCKED_SURFACE["spheres_total"],
        "geometries_total": len(journey),
        "watertight": bool(union.is_watertight),
        "frame_gates_match_locked": bool(
            journey_gates == live["frame_gates"]),
        "inputs_match_locked_record": bool(_inputs_match_locked(
            in_hashes, canonical_root, live)),
    }
    # the hard asserts (spec 2: fail hard with the numbers printed)
    if not exact and delta > LOCKED_AREA_TOL_A2:
        raise ReplicateError(
            "surface identity FAILED: rebuilt union area "
            f"{mesh_area!r} A^2 vs the locked "
            f"{LOCKED_SURFACE['mesh_area_A2']!r} A^2 (abs delta "
            f"{delta!r}, the relaxation band {LOCKED_AREA_TOL_A2}) -- "
            "NOT the locked canonical surface (the journey inputs or "
            "the union machinery differ)")
    if len(journey) * N_ATOMS != LOCKED_SURFACE["spheres_total"]:
        raise ReplicateError(
            "surface identity FAILED: rebuilt "
            f"{len(journey) * N_ATOMS} spheres vs the locked "
            f"{LOCKED_SURFACE['spheres_total']} -- the journey assembly "
            "differs from the canonical build")
    if not union.is_watertight:
        raise ReplicateError("surface identity FAILED: the rebuilt union "
                             "is not watertight")
    # the extra evidence the spec's relaxation leaves recorded only:
    rebuilt_spheres = cloud_build.journey_spheres(journey, elements)
    canon_spheres_path = os.path.join(canonical_root,
                                      "candidates_v2_spheres.npy")
    identity["spheres_bitwise_live"] = bool(
        np.array_equal(rebuilt_spheres, np.load(canon_spheres_path)))
    return identity


def build_surface(canonical_root=CANONICAL_ROOT, verbose=True):
    """Rebuild the union surface with the same code path + same inputs
    the canonical build used and ASSERT the locked identity (fails hard
    with the numbers). Returns (elements, journey, interp_flags, counts,
    interp_log, in_hashes, journey_gates, union, mesh_area,
    identity_record)."""
    (elements, journey, journey_gates, interp_flags, counts, interp_log,
     in_hashes) = cloud_build.assemble_journey(canonical_root)
    union = union_of_envelopes(
        elements, [np.asarray(x, dtype=float) for _st, _f, x in journey])
    mesh_area = float(union.area)
    identity = assert_surface_identity(journey, elements, journey_gates,
                                       in_hashes, union, mesh_area,
                                       canonical_root)
    return (elements, journey, interp_flags, counts, interp_log,
            in_hashes, journey_gates, union, mesh_area, identity)


# ---------------------------------------------------------------------------
# the comparison stage (spec 4: the mirror image of the surface stage)
# ---------------------------------------------------------------------------
COMPARISON_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90,
                        0.95, 0.99, 1.0)

# descriptive metrics carry NO gate semantics: these keys may not
# appear anywhere under the comparison's 'descriptive' block (asserted
# in code -- the writer refuses to write a gate word among the metrics)
_METRIC_BANNED_KEYS = frozenset(
    {"pass", "pass_all", "gate", "gates", "fail", "tol", "tol_A",
     "tolerance", "threshold", "band", "ok", "status", "all_pass"})


# the comparison TARGET's recorded roll (the locked canonical artifact
# was drawn first in its process — annotated roll 0 per the 2026-10-14
# ruling; the fixture probe measured the annotation: re-derivable at
# 4.3e-12 A unpinned / <= 1.1e-12 A pinned)
CANONICAL_TARGET_ROLL = 0
REDERIVABILITY_MEASURED_UNPINNED_A = 4.3e-12


def _assert_metric_purity(metrics, path="descriptive"):
    """Fail hard if any gate-semantics key sneaks into the descriptive
    block (the schema-separation invariant, asserted at write time)."""
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(k, str) and k.lower() in _METRIC_BANNED_KEYS:
                raise ReplicateError(
                    f"comparison schema violation: the descriptive "
                    f"block may not carry gate semantics ({path}/{k}) -- "
                    "metrics are recorded, never gated")
            _assert_metric_purity(v, path=f"{path}/{k}")
    elif isinstance(metrics, list):
        for i, v in enumerate(metrics):
            _assert_metric_purity(v, path=f"{path}[{i}]")


def nn_metrics(points, reference):
    """One-sided nearest-neighbour distances of `points` against
    `reference` (KD-tree, k=1) + the recorded distribution:
    mean/median/max/min/std + the FULL quantile grid."""
    tree = scipy.spatial.cKDTree(reference)
    distances, _idx = tree.query(np.ascontiguousarray(points), k=1)
    distances = np.asarray(distances, dtype=np.float64)
    quantiles = {}
    for q in COMPARISON_QUANTILES:
        quantiles[f"q{q:g}"] = float(np.quantile(distances, q))
    return {
        "mean_A": float(distances.mean()),
        "median_A": float(np.median(distances)),
        "max_A": float(distances.max()),
        "min_A": float(distances.min()),
        "std_A": float(distances.std()),
        "quantiles_A": quantiles,
    }


def exact_row_matches(rep_points, canon_points):
    """The bitwise row-match census BOTH ways: [(rep_row, canon_row,
    coords)] for every replicate row bitwise equal to a canonical row
    AND every canonical row bitwise equal to a replicate row (the
    intersection's two directions; duplicates inside either side list
    once per matching pair)."""
    rep = np.ascontiguousarray(rep_points, dtype=np.float64)
    canon = np.ascontiguousarray(canon_points, dtype=np.float64)
    forward = {}
    for i, row in enumerate(canon):
        forward.setdefault(row.tobytes(), []).append(i)
    matches = []
    for i, row in enumerate(rep):
        for j in forward.get(row.tobytes(), ()):
            matches.append((i, j, row))
    reverse_count = sum(
        1 for i, row in enumerate(canon)
        if row.tobytes() in {r.tobytes() for r in rep})
    return matches, reverse_count


def _identity_gates(rep_projected, rep_raw, rep_cloud_sha, rep_raw_sha,
                    canon_cloud, canon_shas, verbose=True,
                    same_realization=False):
    """The comparison GATES (spec 4, with the 2026-10-14
    generalization ruling's same-realization branch).

    same_realization=False (the built roll differs from the comparison
    TARGET's recorded roll): a re-draw must be a re-draw -- a cloud
    file sha256-equal to any canonical cloud array fails, and the count
    of candidate rows bitwise equal to a canonical row must be ZERO
    (any nonzero count fails hard with the matched coordinates printed;
    the did-you-actually-redraw detector).

    same_realization=True (the SAME roll -- e.g. the canonical roll-0
    re-derivation vs the locked artifact): the refusal gates are
    REPLACED by the AGREEMENT gate: the max nearest-neighbour distance
    between the regenerated cloud and the locked artifact must be <=
    the measured re-derivation band (2e-12 A); the FILE SHAS ARE
    EXPECTED TO DIFFER (last-bit float noise shifts the sha; a
    sha-equal file is the stronger agreement and passes); bitwise is
    asserted only where the float path is actually stable.

    Returns the identity record on green."""
    gates = {
        "cloud_file_sha_distinct": {
            "pass": True,
            "expected_to_differ": bool(same_realization),
            "replicate_cloud_sha256": rep_cloud_sha,
            "replicate_raw_sha256": rep_raw_sha,
            "canonical_array_sha256s": dict(canon_shas),
        },
    }
    if same_realization:
        nn_rep = nn_metrics(rep_projected, canon_cloud)
        nn_can = nn_metrics(canon_cloud, rep_projected)
        worst = max(nn_rep["max_A"], nn_can["max_A"])
        gates["same_roll_agreement"] = {
            "pass": worst <= REDERIVABILITY_BAND_A,
            "max_nn_A": worst,
            "band_A": REDERIVABILITY_BAND_A,
            "replicate_to_canonical": nn_rep,
            "canonical_to_replicate": nn_can,
        }
        if worst > REDERIVABILITY_BAND_A:
            raise ReplicateError(
                "comparison agreement gate FAILED: the same-roll "
                f"regeneration's max NN-to-locked distance {worst!r} A "
                f"exceeds the measured re-derivation band "
                f"{REDERIVABILITY_BAND_A} A (the pinned regime; the "
                "unpinned record noted 4.3e-12 A) -- re-derive PINNED")
        gates["gates_all_pass"] = True
        return gates

    gates["cloud_file_sha_distinct"]["pass"] = (
        rep_cloud_sha not in canon_shas.values()
        and rep_raw_sha not in canon_shas.values())
    if not gates["cloud_file_sha_distinct"]["pass"]:
        raise ReplicateError(
            "comparison identity gate FAILED: the replicate cloud file "
            "is sha256-equal to a canonical cloud array "
            f"(replicate cloud {rep_cloud_sha[:16]}..., raw "
            f"{rep_raw_sha[:16]}...; canonical arrays: "
            f"{json.dumps(canon_shas)}) -- this is not a fresh draw "
            "(ask-first recovery)")

    matches, reverse_count = exact_row_matches(rep_projected, canon_cloud)
    raw_matches, raw_reverse = exact_row_matches(rep_raw, canon_cloud)
    total = len(matches) + len(raw_matches)
    gates["row_bitwise_distinct"] = {
        "pass": total == 0,
        "rows_matching_a_canonical_row": total,
        "projected_matches": len(matches),
        "raw_matches": len(raw_matches),
        "reverse_matches": reverse_count + raw_reverse,
    }
    if total != 0:
        lines = [f"  rep row {i} == canon row {j}: "
                 f"[{x[0]!r}, {x[1]!r}, {x[2]!r}]"
                 for i, j, x in matches[:16]]
        raw_lines = [f"  rep raw row {i} == canon row {j}: "
                     f"[{x[0]!r}, {x[1]!r}, {x[2]!r}]"
                     for i, j, x in raw_matches[:16]]
        raise ReplicateError(
            "comparison identity gate FAILED: "
            f"{len(matches)} projected + {len(raw_matches)} raw "
            "candidate rows bitwise equal canonical rows (must be ZERO "
            "-- machinery breakage by construction, not a sampler "
            "coincidence; recovery is ask-first). Matched coordinates:\n"
            + "\n".join(lines + raw_lines))
    gates["gates_all_pass"] = True
    return gates


def compare_to_canonical(name, core, canonical_root=CANONICAL_ROOT,
                         verbose=True):
    """The comparison stage: READ-ONLY references under the canonical
    tree (the canonical cloud + its raw and provenance siblings + the
    canonical summary), identity gates (hard, above) + difference
    metrics (RECORDED ONLY -- a strongly-different realization is a
    valid replicate; no metric may gate, fail, or colour the exit
    code). Returns the comparison dict (identity | descriptive,
    schema-separated). Raises on only the identity gates."""
    canon_cloud_path = os.path.join(canonical_root, "candidates_v2.npy")
    canon_raw_path = os.path.join(canonical_root, "candidates_v2_raw.npy")
    canon_prov_path = os.path.join(canonical_root,
                                   "candidates_v2_provenance.npy")
    canon_summary_path = os.path.join(canonical_root,
                                      "cloud_v2_summary.json")

    canon_cloud = np.load(canon_cloud_path)
    canon_prov = np.load(canon_prov_path)
    canon_summary = json.load(open(canon_summary_path))
    canon_shas = {
        "candidates_v2.npy": sha256_file(canon_cloud_path),
        "candidates_v2_raw.npy": sha256_file(canon_raw_path),
        "candidates_v2_provenance.npy": sha256_file(canon_prov_path),
        "cloud_v2_summary.json": sha256_file(canon_summary_path),
    }

    rep_projected = np.ascontiguousarray(core["projected"])
    rep_raw = np.ascontiguousarray(core["raw"])
    rep_cloud_sha = npy_sha256(rep_projected)
    rep_raw_sha = npy_sha256(rep_raw)

    # THE 2026-10-14 GENERALIZATION RULING: the dispatch is on the
    # comparison TARGET'S recorded roll, never on the name -- a
    # different roll = the refusal gates (a re-draw must be a re-draw);
    # the same roll (canonical roll 0 vs the locked artifact) = the
    # agreement gate (the measured re-derivation band)
    built_roll = int(core.get("steer_rolls") or 0)
    same_realization = (built_roll == CANONICAL_TARGET_ROLL)

    identity_gates = _identity_gates(
        rep_projected, rep_raw, rep_cloud_sha, rep_raw_sha,
        canon_cloud, canon_shas, verbose,
        same_realization=same_realization)

    # ---- difference metrics (RECORDED ONLY) --------------------------
    nn_rep = nn_metrics(rep_projected, canon_cloud)
    nn_can = nn_metrics(canon_cloud, rep_projected)
    hausdorff = float(max(nn_rep["max_A"], nn_can["max_A"]))

    o_state = core["labels"][0]
    rep_counts = np.bincount(np.asarray(o_state, dtype=int), minlength=4)
    canon_counts = np.bincount(np.asarray(canon_prov["origin_state"],
                                          dtype=int), minlength=4)
    state_delta = {STATE_NAMES[s]: int(rep_counts[s]) - int(canon_counts[s])
                   for s in range(4)}
    joint_rep, joint_can = {}, {}
    for s, name_ in enumerate(STATE_NAMES):
        for atom in range(N_ATOMS):
            jr = int(((o_state == s) & (core["labels"][2] == atom)).sum())
            jc = int(((canon_prov["origin_state"] == s)
                      & (canon_prov["origin_atom"] == atom)).sum())
            if jr or jc:
                joint_rep[f"{name_}:{atom}"] = jr
                joint_can[f"{name_}:{atom}"] = jc
    joint_delta = {k: joint_rep[k] - joint_can[k] for k in joint_rep}

    canon_nn = canon_summary["nn_stats"]
    descriptive = {
        "note": "difference metrics are RECORDED ONLY, not gates: a "
                "strongly-different realization is a valid replicate "
                "(the entire point of the fresh draw)",
        "nearest_canonical_point_distances_A": {
            "replicate_to_canonical": nn_rep,
            "canonical_to_replicate": nn_can,
            "hausdorff_A": hausdorff,
        },
        "deltas_vs_canonical_recorded": {
            "n": int(len(rep_projected) - canon_summary["n"]),
            "mesh_area_A2": (float(core["mesh_area_A2"])
                             - float(canon_summary["mesh_area_A2"])),
            "nn_mean_A": (core["gates"]["nn_spacing_A"]["mean_A"]
                          - float(canon_nn["mean_A"])),
            "nn_cv": (core["gates"]["nn_spacing_A"]["cv"]
                      - float(canon_nn["cv"])),
        },
        "provenance_class_deltas": {
            "state_counts": {"replicate": {STATE_NAMES[s]: int(rep_counts[s])
                                           for s in range(4)},
                             "canonical": {STATE_NAMES[s]: int(canon_counts[s])
                                           for s in range(4)},
                             "delta": state_delta},
            "state_atom_joint_delta": joint_delta,
        },
        "canonical_rederivability_side_fact": (
            "measured 2026-10-14 (fixture probe, the ruling's record): "
            "the canonical cloud is re-derivable on this pin to "
            "~4.3e-12 A max coordinate delta -- recorded, no action"),
    }
    _assert_metric_purity(descriptive)

    comparison = {
        "schema": "cloud_comparison_v1",
        "replicate": name,
        "roll": built_roll,
        "target_recorded_roll": CANONICAL_TARGET_ROLL,
        "same_realization_branch": bool(same_realization),
        "utc": _utc(),
        "identity": {
            "gates": identity_gates,
            "gates_all_pass": True,
            "note": ("the comparison gates (spec 4 + the 2026-10-14 "
                     "generalization): a different roll = the refusal "
                     "gates (identical-sha + zero-bitwise rows; the "
                     "did-you-redraw detector); the same roll = the "
                     "agreement gate (the measured re-derivation "
                     "band); everything else is descriptive"),
        },
        "descriptive": descriptive,
        "canonical_references": {
            "root": os.path.normpath(canonical_root),
            "sha256": canon_shas,
            "role": "READ-ONLY references (the gates + the recorded "
                    "values); nothing under data/derived/ is written",
        },
    }
    return comparison

# ---------------------------------------------------------------------------
# writers (spec 5: the artifact set + the one manifest)
# ---------------------------------------------------------------------------
def _write_replicate_cloud(root_path, name, core, summary, comparison,
                           in_hashes, sampler_record):
    """The spec-5 artifact set + the summary + the comparison + the
    manifest (sha256 of every file written; nothing downstream may
    proceed without it). Returns the manifest dict."""
    o_state, o_frame, o_atom, nearest_ts, _err = core["labels"]
    path = lambda fn: os.path.join(root_path, fn)

    with open(path(f"candidates_{name}.npy"), "wb") as fh:
        np.save(fh, core["projected"], allow_pickle=False)
    with open(path(f"candidates_{name}_raw.npy"), "wb") as fh:
        np.save(fh, core["raw"], allow_pickle=False)
    prov = np.zeros(len(core["projected"]), dtype=[
        ("origin_state", "u1"), ("origin_frame", "u4"),
        ("origin_atom", "u1"), ("nearest_ts_atom", "u1")])
    prov["origin_state"] = o_state
    prov["origin_frame"] = o_frame
    prov["origin_atom"] = o_atom
    prov["nearest_ts_atom"] = nearest_ts
    with open(path(f"candidates_{name}_provenance.npy"), "wb") as fh:
        np.save(fh, prov, allow_pickle=False)

    with open(path(f"candidates_{name}.xyz"), "w") as fh:
        fh.write(
            f"{len(core['projected'])}\n"
            f"replicate {name}: fresh Lloyd/CVT realization of the "
            "canonical locked swept-union surface (0.5 A target + the "
            "v1 on-mesh projection fix); provenance in "
            f"candidates_{name}_provenance.npy; vs-canonical comparison "
            f"in cloud_{name}_comparison.json; see "
            f"cloud_{name}_summary.json\n")
        for x, y, z in core["projected"]:
            fh.write(f"X {x: .6f} {y: .6f} {z: .6f}\n")

    # the summary's own sha256 block (the canonical schema's block, with
    # the replicate's keys): the four artifacts; inputs = the canonical
    # journey inputs (the SAME locked files, same shas)
    summary["sha256"] = {
        "inputs": {os.path.normpath(p): sha for p, sha in
                   in_hashes.items()},
        "outputs": {os.path.normpath(path(fn)): sha256_file(path(fn))
                    for fn in (f"candidates_{name}.npy",
                               f"candidates_{name}_raw.npy",
                               f"candidates_{name}_provenance.npy",
                               f"candidates_{name}.xyz")},
    }
    summary = cloud_build.json_hygiene(summary)   # np scalars -> python
    with open(path(f"cloud_{name}_summary.json"), "w") as fh:
        fh.write(json.dumps(summary, indent=2) + "\n")

    comparison["artifacts"] = {
        "summary": f"cloud_{name}_summary.json",
        "summary_sha256": sha256_file(path(f"cloud_{name}_summary.json")),
    }
    comparison = cloud_build.json_hygiene(comparison)
    with open(path(f"cloud_{name}_comparison.json"), "w") as fh:
        fh.write(json.dumps(comparison, indent=1) + "\n")

    manifest = _write_manifest(root_path, name, stage="cloud",
                               stage_record={
                                   "ok": True,
                                   "utc": _utc(),
                                   "n": int(len(core["projected"])),
                                   "mesh_area_A2":
                                       float(core["mesh_area_A2"]),
                                   "gates_all_pass": True,
                                   "sampler": sampler_record,
                               }, note="the cloud stage's artifact set; "
                                      "one manifest for the replicate")
    return manifest


def _write_manifest(root_path, name, stage, stage_record, note=None):
    """(Re)write manifest.json: sha256 of EVERY file under the replicate
    root except the manifest itself + the stage ledger."""
    manifest_path = os.path.join(root_path, MANIFEST_NAME)
    manifest = {}
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
    manifest.setdefault("replicate", name)
    manifest.setdefault("stages", {})
    manifest["stages"][stage] = stage_record
    manifest["manifest_sha256_note"] = (
        "sha256 of every file written under this replicate root except "
        "the manifest itself; nothing downstream may proceed without "
        "this manifest")
    if note:
        manifest["note"] = note
    files = {}
    for dirpath, _dirs, fnames in os.walk(root_path):
        if "__pycache__" in dirpath:
            continue
        for fn in sorted(fnames):
            if fn == MANIFEST_NAME:
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root_path)
            files[rel] = sha256_file(p)
    manifest["files"] = {k: files[k] for k in sorted(files)}
    manifest["written_utc"] = _utc()
    with open(manifest_path, "w") as fh:
        fh.write(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _print_manifest(manifest):
    """The CLI prints the manifest (bounded print: the artifacts in
    full; large work trees summarized by directory)."""
    body = dict(manifest)
    files = body.get("files", {})
    if len(files) > 200:
        summarized = {}
        dirs = {}
        for k, v in sorted(files.items()):
            top = k.split(os.sep)[0]
            if top == k:
                summarized[k] = v
            else:
                dirs[top] = dirs.get(top, 0) + 1
        body["files"] = summarized
        body["files_by_dir_count"] = dirs
        body["files_summary_note"] = (
            f"{len(files)} files written; the full per-file map lives in "
            "the manifest")
    print(json.dumps(body, indent=1))


def _require_manifest(root_path, name):
    """Nothing downstream may proceed without the manifest: the enum/smoke
    stages verify the root has one and that the cloud artifact's recorded
    sha256 still matches the file (the provenance gate)."""
    manifest_path = os.path.join(root_path, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise ReplicateError(
            f"{root_path}: no {MANIFEST_NAME} -- nothing downstream may "
            "proceed without it (run the cloud stage first)")
    manifest = json.load(open(manifest_path))
    cloud = f"candidates_{name}.npy"
    cloud_path = os.path.join(root_path, cloud)
    if cloud not in manifest.get("files", {}):
        raise ReplicateError(
            f"{manifest_path}: the cloud artifact is not recorded -- "
            "run the cloud stage first")
    got = sha256_file(cloud_path)
    if got != manifest["files"][cloud]:
        raise ReplicateError(
            f"{cloud_path}: sha256 {got} contradicts the manifest's "
            f"{manifest['files'][cloud]} -- the artifact drifted from "
            "its own manifest (refuse)")
    return manifest, manifest["files"][cloud]


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------
STEERING_ROLL_MECHANISM = (
    "the pinned point-cloud_utils sampler is fixed-seeded per process; "
    "the canonical cloud is roll 0 of that stream and replicate k "
    "draws roll k: the draw process advances the stream by k discarded "
    "sample_mesh_lloyd runs (each at the recorded n) before the "
    "recorded draw")
REDERIVABILITY_BAND_A = 2e-12          # same-roll reruns <= ~1e-12 A
# (the measured agreement ceiling across 5 pinned reruns, 2026-10-14:
#  1.102e-12 A; the band = the ceiling with the project's standard
#  ~1.8x headroom — the ruling's ~1e-12 order, recorded honestly)


def sibling_replicate_roots():
    """The OTHER sibling replicate roots (dirs under data/replicates/,
    root-level only)."""
    base = os.path.join(REPO_ROOT, REPLICATES_ROOT)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and not d.startswith("."))


def auto_roll():
    """The roll number of a NEW replicate: distinct by construction
    against EVERY sibling's RECORDED roll (not just the root count --
    deleted roots break monotonicity): the next roll = 1 + max(all
    recorded sibling rolls, the sibling count). The real item-5
    replicates pass --roll explicitly per the deploy."""
    recorded = _recorded_sibling_rolls()
    ceiling = max([len(sibling_replicate_roots())]
                  + [r for r in recorded if r > 0])
    return ceiling + 1


def _recorded_sibling_rolls():
    """{roll: root_name} already RECORDED in sibling manifests (a root
    without a manifest is skipped — it cannot claim a roll)."""
    recorded_rolls = {}
    for sibling in sibling_replicate_roots():
        mp = os.path.join(REPO_ROOT, REPLICATES_ROOT, sibling,
                          MANIFEST_NAME)
        if not os.path.isfile(mp):
            continue
        try:
            manifest = json.load(open(mp))
            stages = manifest.get("stages") or {}
            sampler = (stages.get("cloud") or {}).get("sampler")
        except ValueError:
            continue
        if sampler and isinstance(sampler, dict) and "roll" in sampler:
            recorded_rolls[int(sampler["roll"])] = sibling
    return recorded_rolls


def run_cloud_stage(name, canonical_root=CANONICAL_ROOT, verbose=True,
                    roll=None):
    """The default stage: fresh realization + gates + comparison +
    manifest (spec 1's full sequence: cloud + gates green + STOP).

    roll: the replicate's stream roll (the 2026-10-14 draw ruling:
    canonical = roll 0; replicate k = roll k, by k discarded sampler
    runs). None = auto (1 + the sibling-root count); a roll RECORDED by
    a sibling replicate refuses (distinctness by construction).
    """
    root_rel = replicate_root(name)
    # the fence belt (the 2026-10-14 generalization: canonical = roll 0,
    # announced -- re-validated here for direct library callers)
    validate_name(name, roll=roll)
    root_abs = os.path.join(REPO_ROOT, root_rel)
    if os.path.isdir(root_abs) and os.listdir(root_abs):
        raise ReplicateError(
            f"{root_rel}: not empty -- a replicate root is written ONCE "
            "(a fresh draw is a different cloud; the artifact set never "
            "overwrites). Use a new name.")
    fence_write(root_abs, "the replicate write root")
    os.makedirs(root_abs, exist_ok=True)

    _p = print if verbose else (lambda *a, **k: None)
    if roll is None:
        roll = auto_roll()
        _p(f"replicate {name}: roll = {roll} (auto: 1 + the existing "
           "sibling-root count)")
    else:
        roll = int(roll)
        if roll < 0:
            raise ReplicateError(f"--roll {roll}: refused -- a roll is a "
                                 "non-negative integer")
        sibling_claim = _recorded_sibling_rolls().get(roll)
        if sibling_claim is not None:
            raise ReplicateError(
                f"--roll {roll}: refused -- the sibling replicate "
                f"{sibling_claim!r} already records roll {roll} (a "
                "replicate's roll must be distinct: distinctness by "
                "construction; recovery is ask-first)")
    _p(f"replicate {name}: surface stage (the same-surface guarantee)")
    (elements, journey, interp_flags, counts, interp_log, in_hashes,
     journey_gates, union, mesh_area, identity) = build_surface(
        canonical_root, verbose=verbose)
    _p(f"  identity: area {identity['rebuilt_mesh_area_A2']:.6f} A^2 "
       f"(locked {identity['locked_mesh_area_A2']:.6f}, exact="
       f"{identity['bitwise_clean']}), "
       f"{identity['sphere_count']} spheres, watertight="
       f"{identity['watertight']}")

    _p(f"replicate {name}: cloud stage (the fresh realization: roll "
       f"{roll} of the fixed stream)")
    core = cloud_build.compute_build_core(
        elements, journey, interp_flags, counts, interp_log,
        journey_gates, union=union, header=f"replicate {name}",
        verbose=verbose, steer_rolls=roll)
    gates = core["gates"]
    strict = float(core["post_off"].max())
    gates["projection_residual_strict"] = {
        "max_A": strict,
        "tol_A": STRICT_RESIDUAL_TOL_A,
        "pass": strict <= STRICT_RESIDUAL_TOL_A,
    }
    cloud_build.check_correctness_gates(
        gates, extra_names=("projection_residual_strict",))

    _p(f"replicate {name}: comparison stage (PROVE not the canonical "
       "points; MEASURE the difference)")
    comparison = compare_to_canonical(name, core, canonical_root,
                                      verbose=verbose)
    nn = comparison["descriptive"]["nearest_canonical_point_distances_A"]
    _p(f"  identity gates green; Hausdorff {nn['hausdorff_A']:.4e} A "
       f"| rep->canon NN mean {nn['replicate_to_canonical']['mean_A']:.4e} A "
       "(recorded only)")

    _p(f"replicate {name}: writing the artifact set + the manifest")
    sampler_record = {
        "roll": int(roll),
        "mechanism": STEERING_ROLL_MECHANISM,
        "rederivability_band_A": REDERIVABILITY_BAND_A,
        "band_source": ("the 2026-10-14 test-8 ruling (the measured "
                        "same-roll agreement ceiling 1.102e-12 A across "
                        "5 pinned reruns; the band = the ceiling x the "
                        "project's ~1.8x headroom; the 2026-10-08 "
                        "reference-band precedent)")
        if roll else ("roll 0 = the canonical fixed-stream draw (the "
                      "measured re-derivability: 4.3e-12 A on this pin)"),
    }
    summary = cloud_build.cloud_summary(core, gates_all_pass=True)
    summary["replicate"] = {
        "name": name,
        "root": os.path.normpath(root_rel),
        "surface_identity": identity,
        "sampler": sampler_record,
        "canonical_references": comparison["canonical_references"],
        "canonical_rederivability_side_fact":
            comparison["descriptive"][
                "canonical_rederivability_side_fact"],
    }
    manifest = _write_replicate_cloud(root_abs, name, core, summary,
                                      comparison, in_hashes,
                                      sampler_record)
    _print_manifest(manifest)
    _p(f"replicate {name}: PASS -- gates green, artifacts + manifest "
       f"under {root_rel} (roll {roll})")
    return summary


# ---------------------------------------------------------------------------
# the enumeration wiring (spec 6) — driver subprocess, per-replicate roots
# ---------------------------------------------------------------------------
def run_enum_stage(name, mode="xtb", limit=None, shards=8,
                   canonical_root=CANONICAL_ROOT, verbose=True,
                   work_suffix=""):
    """The enum stage: invoke the enumeration driver with the
    replicate's cloud + sha (the per-(cloud sha, stage) work root lives
    under the replicate root). Asserts the driver's accounting identity
    (ledger == distinct, == the limit on a limited run).

    work_suffix: an explicit suffix for the stage's work root — a REAL
    re-evaluation of the same rows (determinism proofs, repeats) runs a
    FRESH work tree via a suffix; the default keeps the resumable
    canonical root for that (cloud sha, mode)."""
    root_rel = replicate_root(name)
    root_abs = os.path.join(REPO_ROOT, root_rel)
    manifest, cloud_sha = _require_manifest(root_abs, name)
    cloud_path = os.path.join(root_abs, f"candidates_{name}.npy")
    # the work/checkpoint roots key per (cloud sha, stage): a different
    # cloud identity NEVER shares checkpoint state; work_suffix = a
    # fresh tree for a real re-evaluation (determinism proofs)
    work_root = os.path.join(root_abs, "work",
                             f"enum_{mode}_{cloud_sha[:12]}{work_suffix}")

    _p = print if verbose else (lambda *a, **k: None)
    _p(f"replicate {name}: enum stage (mode {mode}, limit {limit})")
    args = [sys.executable, "run_exhaustive_search.py",
            "--cloud", os.path.relpath(cloud_path, REPO_ROOT),
            "--cloud-sha256", cloud_sha,
            "--mode", mode,
            "--shards", str(shards),
            "--work-root", os.path.relpath(work_root, REPO_ROOT)]
    if limit is not None:
        args += ["--limit", str(limit)]
    proc = subprocess.run(args, cwd=REPO_ROOT)
    if proc.returncode != 0:
        report_path = os.path.join(work_root, "enumerate_report.json")
        detail = ""
        if os.path.isfile(report_path):
            report = json.load(open(report_path))
            detail = report.get("error", "")
        raise ReplicateError(
            f"replicate {name}: the enumeration driver exited "
            f"{proc.returncode} -- "
            + (detail or "see the stage report at " + report_path)
        )

    report = json.load(open(os.path.join(work_root,
                                         "enumerate_report.json")))
    if not report["ok"]:
        raise ReplicateError(
            f"replicate {name}: the driver's report is not ok "
            f"({report.get('status')}) -- stage refuses")
    distinct, ledger, evaluated = (report["n_evals_distinct"],
                                   report["n_evals_ledger"],
                                   report["n_rows_evaluated"])
    if distinct != ledger:
        raise ReplicateError(
            f"replicate {name}: the driver's accounting identity broke "
            f"(distinct {distinct} != ledger {ledger})")
    expected = limit if limit is not None else report["universe"]["n_rows"]
    if evaluated != expected:
        raise ReplicateError(
            f"replicate {name}: {evaluated} evaluated rows != the "
            f"expected {expected} (the coverage identity's own record)")
    manifest = _write_manifest(
        root_abs, name, stage=f"enum_{mode}",
        stage_record={"ok": True, "utc": _utc(), "mode": mode,
                      "limit_rows": report["limit_rows"],
                      "n_evals_distinct": distinct,
                      "cloud_sha256": cloud_sha,
                      "report": os.path.join(os.path.relpath(
                          work_root, root_abs),
                          "enumerate_report.json"),
                      "global_min": report.get("global_min")})
    _print_manifest(manifest)
    _p(f"replicate {name}: enum stage PASS ({distinct} rows evaluated)")
    return report


def run_smoke_stage(name, canonical_root=CANONICAL_ROOT, stub_limit=200,
                    xtb_limit=20, verbose=True):
    """The end-to-end-smoke stage (tester mode): the whole pipeline in
    one command under a throwaway replicate root; the stub subset's
    accounting invariants (ledger == distinct == 200) + the real
    provider's plausibility/plumbing gates are asserted HERE."""
    run_cloud_stage(name, canonical_root=canonical_root, verbose=verbose)
    stub_report = run_enum_stage(name, mode="stub", limit=stub_limit,
                                 canonical_root=canonical_root,
                                 verbose=verbose)
    xtb_report = run_enum_stage(name, mode="xtb", limit=xtb_limit,
                                canonical_root=canonical_root,
                                verbose=verbose)
    _p = print if verbose else (lambda *a, **k: None)
    _p(f"replicate {name}: end-to-end-smoke PASS (cloud + gates green + "
       f"comparison + stub {stub_report['n_evals_distinct']}-row subset + "
       f"xtb {xtb_report['n_evals_distinct']}-row subset)")
    return {".stub": stub_report, "xtb": xtb_report}
