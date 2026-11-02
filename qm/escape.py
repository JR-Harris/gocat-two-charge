"""The flat-well escape (mode-following) + the endpoints stage driver.

The settled chorismate endpoint method, confirmed against the frozen v1
archive (src/scripts/optimize_endpoints.py; see the stage-4 log):

  * the source conformer optimises to a 2nd-order saddle (exactly two
    hard imaginary modes, ~-131 and ~-75 cm-1, and no soft modes below
    -10 cm-1 — verified in the v1 outputs);
  * mode-following escape: from the CURRENT structure, displace along
    the sum of all modes below -10 cm-1 (for the saddle these are
    exactly its two hard imaginary modes), scaled so the largest
    absolute Cartesian component of the displacement equals 0.3 A
    (one displaced geometry per attempt, + direction only), then
    re-optimise; iterate from the latest structure while any mode
    below -10 cm-1 remains, up to 3 attempts;
  * mode eigenvectors come from the job's ORCA .hess $normal_modes
    block (n_coord rows x n_modes columns, column-blocks of 5); the
    next attempt's geometry from the job's ORCA-written .xyz;
  * best-tracking by rank (n_hard, n_soft, E) — fewer hard imaginary,
    then fewer soft, then lower energy; the saddle itself is a
    candidate. For the v1 outputs this selects chor_min1
    (0 hard, 2 soft, lowest E among the 0-hard attempts) — the
    accepted non-reactive minimum.

Frequencies classify by the two-threshold rule (hard < -50 cm-1 fails
verification of a minimum; soft -50..-10 is reported, does not fail).

The stage writes only under data/derived/orca_v2/step2_endpoints/ in
the laptop layout — every path goes through orca_runner.resolve_output
(GOCAT_OUT_ROOT), so on the cluster everything lands under
data/hpc_out/ by construction. No summary promotion happens here;
gates run on the laptop (run_qm.py --gates endpoints <out-root>).
"""

import json
import os

from qm import inputs, orca_out, orca_runner, structures

# Two-threshold frequency classification (the locked rule).
HARD_IMAG_CM1 = -50.0    # < -50: hard imaginary
SOFT_IMAG_CM1 = -10.0     # mode-following target: any mode below this
MAX_MIN_ATTEMPTS = 3      # mode-following iterations
MIN_DISPLACEMENT_A = 0.3  # max absolute Cartesian displacement per attempt

# The stage's output directory, laptop layout (relocated by GOCAT_OUT_ROOT).
STAGE_REL_DIR = "data/derived/orca_v2/step2_endpoints"

# Endpoint jobs: molecule tag | source SDF.
ENDPOINT_JOBS = (
    ("chor", "data/source/chorismate_3d.sdf"),
    ("preph", "data/source/prephenate_3d.sdf"),
)

RECORD_NAME = "endpoints_record.json"


def modes_below(freqs, threshold=SOFT_IMAG_CM1):
    """[(index, value)] of parsed frequencies strictly below threshold."""
    return [(i, f) for i, f in freqs if f < threshold]


def soft_values(freqs):
    """Values of the soft imaginary modes (-50 <= f < -10)."""
    return [f for _i, f in freqs if HARD_IMAG_CM1 <= f < SOFT_IMAG_CM1]


def hard_values(freqs):
    """[(index, value)] of the hard imaginary modes (f < -50)."""
    return [(i, f) for i, f in freqs if f < HARD_IMAG_CM1]


def rank_result(n_hard, n_soft, energy_eh):
    """Best-selection rank: fewer hard imaginary, then fewer soft, then
    lower energy. Lower tuple = better (the settled rule)."""
    return (n_hard, n_soft, energy_eh if energy_eh is not None else float("inf"))


def read_hess_normal_modes(hess_path, n_atoms):
    """Parse the $normal_modes block of an ORCA .hess file.

    -> list of mode eigenvectors, modes[m][atom] = [dx, dy, dz].
    Layout (v1-confirmed): header 'n_modes n_coord', then the matrix in
    column-blocks of 5 (one column-index line + n_coord rows per
    block); row index = Cartesian coordinate 0..3N-1.
    """
    lines = open(hess_path).read().splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == "$normal_modes")
    except StopIteration:
        raise ValueError(f"{hess_path}: no $normal_modes block")
    n_modes, n_coord = (int(x) for x in lines[i + 1].split()[:2])
    if n_coord != 3 * n_atoms:
        raise ValueError(
            f"{hess_path}: $normal_modes has {n_coord} coordinates, "
            f"expected 3*{n_atoms}"
        )
    cols = [[0.0] * n_coord for _ in range(n_modes)]
    j = i + 2
    col = 0
    while col < n_modes:
        j += 1  # skip the column-index header line
        for _row in range(n_coord):
            parts = lines[j].split()
            row = int(parts[0])
            for c, v in enumerate(parts[1:]):
                cols[col + c][row] = float(v)
            j += 1
        col += 5
    modes = []
    for m in range(n_modes):
        vec = cols[m]
        modes.append([[vec[3 * a], vec[3 * a + 1], vec[3 * a + 2]]
                      for a in range(n_atoms)])
    return modes


def displace_along_modes(coords, modes, indices, target=MIN_DISPLACEMENT_A):
    """Displaced geometry = coords + sum of the selected modes'
    eigenvectors, scaled so the largest absolute Cartesian component of
    the displacement equals target (Angstrom). v1-settled scaling."""
    if not indices:
        return [list(row) for row in coords]
    d = [[0.0, 0.0, 0.0] for _ in coords]
    for m in indices:
        vec = modes[m]
        for a in range(len(coords)):
            for c in range(3):
                d[a][c] += vec[a][c]
    max_d = max(abs(x) for row in d for x in row)
    if max_d == 0:
        return [list(row) for row in coords]
    scale = target / max_d
    return [[coords[a][c] + d[a][c] * scale for c in range(3)]
            for a in range(len(coords))]


# ---------------------------------------------------------------------------
# per-job parsing + the mode-following loop
# ---------------------------------------------------------------------------

def _job_rel(name):
    return f"{STAGE_REL_DIR}/{name}"


def parse_job(out_path):
    """Parse one completed job -> result dict (termination/SCF gates via
    qm.orca_out), with the mode lists the escape needs."""
    parsed = orca_out.parse_output(out_path)
    freqs = orca_out.vibrational_frequencies(out_path)
    hard = hard_values(freqs)
    soft = soft_values(freqs)
    below10 = modes_below(freqs)
    return {
        "out": out_path,
        "energy_eh": parsed["final_energy_eh"],
        "opt_converged": parsed["opt_converged"],
        "n_modes": parsed["n_modes"],
        "n_hard": parsed["n_imaginary"],
        "n_soft": parsed["n_soft_imaginary"],
        "hard_modes": hard,
        "soft_values": soft,
        "below_soft_threshold": below10,
        "rank": rank_result(parsed["n_imaginary"], parsed["n_soft_imaginary"],
                            parsed["final_energy_eh"]),
    }


def run_molecule(tag, elements, coords, verbose=True):
    """Optimise one endpoint molecule, mode-follow out of any saddle
    (up to MAX_MIN_ATTEMPTS), and best-track by rank. Returns the
    per-molecule record dict. Resumable: completed .outs are skipped by
    the runner's resume gate; the loop re-parses and continues."""
    jobs = []

    def log(msg):
        if verbose:
            print(f"endpoints {tag}: {msg}")

    def run_job(name, job_elements, job_coords):
        inp_rel = _job_rel(name)
        inputs.write_inp(orca_runner.resolve_output(f"{inp_rel}.inp"),
                         job_elements, job_coords)
        run = orca_runner.run_orca(
            orca_runner.resolve_output(f"{inp_rel}.inp"), f"{inp_rel}.out"
        )
        result = parse_job(run["out"])
        result["wall_seconds"] = run["wall_seconds"]
        result["skipped"] = run["skipped"]
        return result

    base = tag
    result = run_job(base, elements, coords)
    jobs.append({"name": base, **{k: v for k, v in result.items() if k != "out"}})
    log(f"{base}: E={result['energy_eh']!r} Eh  hard={result['n_hard']} "
        f"soft={result['n_soft']}  modes<-10: {len(result['below_soft_threshold'])}")

    best, best_name, current = result, base, base
    attempt = 0
    while result["below_soft_threshold"] and attempt < MAX_MIN_ATTEMPTS:
        attempt += 1
        prev = current
        base = f"{tag}_min{attempt}"
        prev_xyz = orca_runner.resolve_output(f"{_job_rel(prev)}.xyz")
        prev_hess = orca_runner.resolve_output(f"{_job_rel(prev)}.hess")
        if not os.path.isfile(prev_xyz):
            raise ValueError(
                f"{prev_xyz}: previous job's ORCA .xyz not found — cannot "
                f"displace for attempt {attempt}"
            )
        if not os.path.isfile(prev_hess):
            raise ValueError(
                f"{prev_hess}: previous job's ORCA .hess not found — cannot "
                f"displace for attempt {attempt}"
            )
        _n, _comment, els, prev_coords = structures.read_xyz_frame(prev_xyz)
        modes = read_hess_normal_modes(prev_hess, len(els))
        idx = [i for i, _v in result["below_soft_threshold"]]
        disp_coords = displace_along_modes(prev_coords, modes, idx)
        log(f"displacing along modes {idx} (attempt {attempt}, "
            f"max {MIN_DISPLACEMENT_A} A) -> {base}")
        result = run_job(base, els, disp_coords)
        jobs.append({"name": base, **{k: v for k, v in result.items() if k != "out"}})
        log(f"{base}: E={result['energy_eh']!r} Eh  hard={result['n_hard']} "
            f"soft={result['n_soft']}  modes<-10: {len(result['below_soft_threshold'])}")
        if result["rank"] < best["rank"]:
            best, best_name = result, base
        current = base

    return {
        "tag": tag,
        "jobs": jobs,
        "attempts": attempt,
        "wall_seconds_total": round(sum(j["wall_seconds"] for j in jobs), 3),
        "accepted": best_name,
        "accepted_energy_eh": best["energy_eh"],
        "accepted_n_hard": best["n_hard"],
        "accepted_n_soft": best["n_soft"],
        "accepted_soft_values": best["soft_values"],
    }


def execute_endpoints():
    """The endpoints stage: render + run both endpoint jobs through the
    site-agnostic runner, escape the chorismate flat-well saddle,
    best-track by (n_hard, n_soft, E), and write the stage record.
    Returns the record dict (also written to the stage directory)."""
    os.makedirs(orca_runner.resolve_output(STAGE_REL_DIR), exist_ok=True)
    record = {"stage": "endpoints", "jobs": {}}
    for tag, sdf in ENDPOINT_JOBS:
        elements, coords, _bonds = structures.parse_sdf(sdf)
        record["jobs"][tag] = run_molecule(tag, elements, coords)

    record_path = orca_runner.resolve_output(f"{STAGE_REL_DIR}/{RECORD_NAME}")
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")

    print("== endpoints stage record ==")
    for tag, mol in record["jobs"].items():
        print(f"{tag}: accepted {mol['accepted']}  E={mol['accepted_energy_eh']!r} "
              f"hard={mol['accepted_n_hard']} soft={mol['accepted_n_soft']} "
              f"({mol['attempts']} escape attempt(s))")
    print(f"record: {record_path}")
    return record