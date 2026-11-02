"""The TS-finding chain (`ts` stage): coordinate driving on the v2
chorismate minimum.

The settled [3,3] Claisen method, semantically confirmed against the
frozen v1 archive (src/scripts/coordinate_drive_ts.py; see the stage-5
log). NEB is excluded history:

  * a relaxed scan of the forming C11-C16 bond (0-based atoms 10-15)
    finds the energy peak — the scan holds the molecule connected so it
    cannot dissociate, but a plain 1-D scan jumps over the TS (the
    breaking C7-O1 holds through the peak then snaps);
  * the 2-constraint Opt at partial values (B 0 6 and B 10 15 both
    2.20 A, 0-based) forces the CONCERTED cyclic TS guess from the peak
    frame;
  * OptTS + analytic FREQ with an exact starting Hessian (Calc_Hess)
    refines the guess into the verified TS: exactly one imaginary mode
    on a single connected fragment.

The verified TS deliverables are `optts.xyz` (ORCA-written) plus the
stage's final `optts.hess` — both IN the stage directory (ORCA writes
the final Hessian at convergence; the also-present `optts.001.hess`
guess-Hessian variant is NOT the one the IRC reuses). No canonical
`data/derived/structures/` writes here — promotion is a later briefing.

The keyword lines are v1-proven artifacts, reproduced with their small
asymmetries exactly (no TightSCF on the OptTS line is deliberate;
analytic Freq only — never NumFreq). The OptTS fallback (v1-proven
contingency, expected never to fire): when the OptTS job exits cleanly
but does NOT terminate normally yet left a gradient-converged saddle
(`optts.xyz`, "THE OPTIMIZATION HAS CONVERGED" in the .out), one
standalone analytic FREQ job (`ts_freq`) characterises it and its
Hessian becomes the hand-off Hessian.

Stage record: ts_record.json in the stage directory — same record genre
as endpoints_record.json (names, energies, classifications, attempts,
wall_seconds, accepted keys). Every path goes through
orca_runner.resolve_output (GOCAT_OUT_ROOT redirection; on the cluster
everything lands under data/hpc_out/ by construction).

Stage input (fail-hard): the endpoints stage's accepted non-reactive
chorismate minimum — the accepted name from
step2_endpoints/endpoints_record.json (accepted: "chor" for this
deployment, energy -838.271116720443 Eh +-1e-5, 0 hard / 0 soft, zero
escape attempts), the ORCA-written optimized geometry
step2_endpoints/<accepted>.xyz, sanity-guarded: C11-C16 (0-based
10-15) within 4.2-4.8 A — the scan is defined on the OPEN conformer.
"""

import json
import os
import shutil

from qm import inputs, orca_out, orca_runner, structures

# The stage's output directory, laptop layout (relocated by GOCAT_OUT_ROOT).
STAGE_REL_DIR = "data/derived/orca_v2/step3_coordinate_drive"
ENDPOINTS_REL_DIR = "data/derived/orca_v2/step2_endpoints"
ENDPOINTS_RECORD = f"{ENDPOINTS_REL_DIR}/endpoints_record.json"
RECORD_NAME = "ts_record.json"

# Relaxed scan of the forming C11-C16 bond (0-based ORCA atoms).
SCAN_ATOMS = (10, 15)
SCAN_START_A = 4.4
SCAN_END_A = 1.8
SCAN_STEPS = 14

# The 2-constraint Opt values (0-based O1-C7 = atoms 0-6, C11-C16 =
# atoms 10-15) that force the concerted cyclic TS guess.
CONSTRAINT_O1C7_A = 2.20
CONSTRAINT_C11C16_A = 2.20

# Input sanity guard: the scan is defined on the OPEN conformer.
INPUT_C11C16_MIN_A = 4.2
INPUT_C11C16_MAX_A = 4.8

# Peak sanity (fail-hard): single connected fragment under the TS-aware
# tolerance AND both [3,3] bonds still within the concerted range.
PEAK_BOND_MAX_A = 3.0

# The accepted non-reactive minimum (stage-4 endpoints record asserts).
ACCEPTED_E_TARGET_EH = -838.271116720443
ENERGY_TOL_EH = 1e-5

# TS verification targets (full-precision artifact values).
TS_E_TARGET_EH = -838.24417429953
TS_IMAG_TARGET_CM1 = -268.72
TS_IMAG_TOL_CM1 = 0.5
TS_KEY_DIST_TOL_A = 1e-3
TS_KEY_DIST_TARGETS = {
    "O1-C7 (breaking)": 2.074829899749152,
    "C11-C16 (forming)": 2.4888984635900253,
    "O1-C14 (->C=O)": 1.2776475890777002,
    "C14-C16": 1.3829456485977967,
    "C7-C9": 1.382831171160029,
    "C9-C11": 1.389368377808103,
}

TS_SOURCE_OPTTS = "OptTS (converged)"
TS_SOURCE_FALLBACK = "standalone FREQ on gradient-converged saddle (optts.xyz)"


# ---------------------------------------------------------------------------
# renderers — the exact templates (v1-proven keyword lines)
# ---------------------------------------------------------------------------

def render_scan_inp(elements, coords):
    """Relaxed C11-C16 scan: drive 0-based atoms 10-15 from 4.4 to 1.8 A
    in 14 steps. Geometry inline (6-decimal). No Freq: ORCA's per-step
    constrained optimisations only."""
    lines = [
        f"! r2SCAN-3c Opt TightSCF PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        "%geom",
        "  scan",
        "    B %d %d = %s, %s, %d" % (SCAN_ATOMS[0], SCAN_ATOMS[1],
                                      SCAN_START_A, SCAN_END_A, SCAN_STEPS),
        "  end",
        "end",
        *inputs.geometry_lines(elements, coords),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)
    return text


def render_constr_inp(elements, coords):
    """2-constraint Opt holding BOTH [3,3] bonds at partial values
    (0-based O1-C7 and C11-C16): forces the concerted cyclic TS
    geometry (a 1-D scan along C11-C16 jumps over the TS because C7-O1
    breaks suddenly between two scan steps)."""
    lines = [
        f"! r2SCAN-3c Opt TightSCF PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        "%geom",
        "  Constraints",
        "    { B 0 6 %.2f C }" % CONSTRAINT_O1C7_A,
        "    { B 10 15 %.2f C }" % CONSTRAINT_C11C16_A,
        "  end",
        "end",
        *inputs.geometry_lines(elements, coords),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)
    return text


def _pointcharges_lines(pointcharges):
    """The optional external point-charge directive (the arbitration
    stage's relaxation check reuses the landed TS machinery with the
    winning charges embedded; None = unembedded, the settled default 
    every landed stage renders)."""
    if pointcharges is None:
        return []
    return [f'%pointcharges "{pointcharges}"']


def render_optts_inp(elements, coords, pointcharges=None):
    """OptTS + analytic FREQ with an exact starting Hessian (Calc_Hess).
    No TightSCF on the OptTS line — the v1-proven asymmetry, kept.
    pointcharges: optional external .pc path (embedded variant, used by
    the arbitration relaxation check; the settled default is None)."""
    lines = [
        f"! r2SCAN-3c OptTS FREQ PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        *_pointcharges_lines(pointcharges),
        "%geom",
        "  Calc_Hess true",
        "end",
        *inputs.geometry_lines(elements, coords),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)
    return text


def render_opt_inp(elements, coords, pointcharges=None):
    """Plain Opt (TightSCF), no Freq — the reactant-side relaxation of
    the arbitration check (embedded variant via pointcharges; the
    settled default is None). Same keyword line family as the
    scan/constr renderers above."""
    lines = [
        f"! r2SCAN-3c Opt TightSCF PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        *_pointcharges_lines(pointcharges),
        *inputs.geometry_lines(elements, coords),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)
    return text


def render_ts_freq_inp(xyz_name="optts.xyz"):
    """Standalone analytic FREQ reading the geometry from a file — the
    fallback characterisation of a gradient-converged saddle if OptTS
    cannot satisfy its step criterion (dormant contingency; the
    geometry is referenced by its name, ORCA's cwd = the stage
    directory)."""
    lines = [
        f"! r2SCAN-3c FREQ PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        "* XYZFILE %d %d %s" % (inputs.CHARGE, inputs.MULT, xyz_name),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)
    return text


def peak_frame_name(peak_step):
    """ORCA's scan frame file name for a 1-based scan step
    (zero-padded: scan.011.xyz for step 11)."""
    return f"scan.{peak_step:03d}.xyz"


# ---------------------------------------------------------------------------
# stage input: the endpoints record + the accepted minimum's geometry
# ---------------------------------------------------------------------------

def read_endpoints_reactant(record_path=None):
    """The endpoints stage's accepted non-reactive chorismate minimum,
    taken (fail-hard) from endpoints_record.json — with the record
    consistency asserts: accepted energy -838.271116720443 Eh +-1e-5,
    0 hard / 0 soft, zero escape attempts. Returns the accepted job's
    fields (the ts stage drives the scan from its geometry; the irc
    stage reads the energy for the folding penalty)."""
    if record_path is None:
        record_path = orca_runner.resolve_output(ENDPOINTS_RECORD)
    if not os.path.isfile(record_path):
        raise ValueError(
            f"{record_path}: endpoints_record.json not found — this stage "
            f"starts from the stage-4 accepted minimum; run the endpoints "
            f"stage first (--execute endpoints)"
        )
    try:
        record = json.load(open(record_path))
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"{record_path}: unparseable record ({e})")
    jobs = record.get("jobs")
    if not isinstance(jobs, dict) or "chor" not in jobs:
        raise ValueError(
            f"{record_path}: no 'chor' molecule entry (jobs keys: "
            f"{list(jobs) if isinstance(jobs, dict) else type(jobs)!r})"
        )
    chor = jobs["chor"]
    accepted = chor.get("accepted")
    if not accepted:
        raise ValueError(
            f"{record_path}: chorismate entry has no accepted job"
        )
    energy = chor.get("accepted_energy_eh")
    problems = []
    if not isinstance(energy, (int, float)):
        problems.append(f"accepted_energy_eh missing ({energy!r})")
    elif abs(energy - ACCEPTED_E_TARGET_EH) > ENERGY_TOL_EH:
        problems.append(
            f"accepted_energy_eh {energy!r} not within {ENERGY_TOL_EH:.0e} "
            f"of {ACCEPTED_E_TARGET_EH!r}"
        )
    for name in ("accepted_n_hard", "accepted_n_soft"):
        if chor.get(name) != 0:
            problems.append(f"{name} {chor.get(name)!r} != 0")
    if chor.get("attempts") != 0:
        problems.append(f"attempts {chor.get('attempts')!r} != 0")
    if problems:
        raise ValueError(
            f"{record_path}: endpoints record inconsistent — this stage's "
            f"input must be the accepted clean minimum (0 hard / 0 soft, "
            f"zero escape attempts): {'; '.join(problems)}"
        )
    return {
        "accepted": accepted,
        "accepted_energy_eh": energy,
        "record_path": record_path,
    }


def scan_input_frame():
    """The scan's input geometry: the accepted minimum's ORCA-written
    .xyz (fail-hard on missing file, wrong atom count, or the C11-C16
    guard — the scan is defined on the OPEN conformer; anything else is
    a wrong input — fail, do not scan). Returns (record_fields, elements,
    coords, c11_c16_A)."""
    accepted_rec = read_endpoints_reactant()
    accepted = accepted_rec["accepted"]
    xyz_rel = f"{ENDPOINTS_REL_DIR}/{accepted}.xyz"
    xyz_path = orca_runner.resolve_output(xyz_rel)
    if not os.path.isfile(xyz_path):
        raise ValueError(
            f"{xyz_path}: the accepted minimum's ORCA-written geometry not "
            f"found (accepted {accepted!r} per endpoints_record.json)"
        )
    n, _comment, elements, coords = structures.read_xyz_frame(xyz_path)
    if n != structures.N_ATOMS:
        raise ValueError(
            f"{xyz_path}: {n} atoms, expected {structures.N_ATOMS} "
            f"(chorismate atom order)"
        )
    label, i, j = structures.KEY_PAIRS[1]  # "C11-C16 (forming)"
    if label != "C11-C16 (forming)":
        raise ValueError("KEY_PAIRS[1] is not the forming bond")
    c11_c16 = structures.distance(coords, i, j)
    if not (INPUT_C11C16_MIN_A <= c11_c16 <= INPUT_C11C16_MAX_A):
        raise ValueError(
            f"{xyz_path}: C11-C16 {c11_c16:.3f} A outside the open-conformer "
            f"guard {INPUT_C11C16_MIN_A}-{INPUT_C11C16_MAX_A} A — wrong "
            f"input for the scan — fail, do not scan"
        )
    return accepted_rec, elements, coords, c11_c16


# ---------------------------------------------------------------------------
# the ts stage flow
# ---------------------------------------------------------------------------

def execute_ts(verbose=True):
    """The `ts` stage: scan -> peak -> constr -> OptTS (fallback
    contingency) -> verified TS + record. Fail-hard at every assert —
    the record is written before raising (the observed values stay on
    record). Resumable at every job level via the runner's resume gate.
    Returns the record dict."""
    log = print if verbose else (lambda *_a, **_k: None)
    jobs = []
    record = {"stage": "ts", "attempts": 0, "jobs": jobs}

    def _write_inp(rel_path, text):
        path = orca_runner.resolve_output(rel_path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def _write_record():
        path = orca_runner.resolve_output(f"{STAGE_REL_DIR}/{RECORD_NAME}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        return path

    def _record_and_fail(msg):
        path = _write_record()
        log(f"ts stage: FAILED — record left at {path}")
        raise ValueError(f"ts stage FAILED: {msg}")

    def _run_job(name, text, need_frequencies):
        """Render `text` to <stage>/<name>.inp, run <name>.out through
        the runner, parse it. Resumable: complete .outs are skipped
        (skipped=True, wall_seconds 0.0)."""
        inp_path = _write_inp(f"{STAGE_REL_DIR}/{name}.inp", text)
        run = orca_runner.run_orca(inp_path, f"{STAGE_REL_DIR}/{name}.out")
        parsed = orca_out.parse_output(run["out"],
                                       need_frequencies=need_frequencies)
        parsed["out"] = run["out"]
        parsed["skipped"] = run["skipped"]
        parsed["wall_seconds"] = run["wall_seconds"]
        jobs.append({"name": name, "skipped": run["skipped"],
                     "wall_seconds": run["wall_seconds"]})
        return parsed

    # ---- 1. stage input: the endpoints record + the accepted geometry
    accepted_rec, elements, coords, c11_c16 = scan_input_frame()
    record["input"] = {
        "endpoints_record": ENDPOINTS_RECORD,
        "accepted": accepted_rec["accepted"],
        "accepted_energy_eh": accepted_rec["accepted_energy_eh"],
        "input_xyz": f"{ENDPOINTS_REL_DIR}/{accepted_rec['accepted']}.xyz",
        "c11_c16_start_A": round(c11_c16, 6),
    }
    log(f"ts stage: input = {record['input']['input_xyz']} "
        f"(accepted {accepted_rec['accepted']!r}, "
        f"E {accepted_rec['accepted_energy_eh']!r} Eh, "
        f"C11-C16 {c11_c16:.3f} A)")

    # ---- 2. the relaxed scan; peak = highest LAST step energy
    scan = _run_job("scan", render_scan_inp(elements, coords), False)
    scan_range = orca_out.scan_info(scan["out"])
    if scan_range != (SCAN_START_A, SCAN_END_A, SCAN_STEPS):
        _record_and_fail(
            f"{scan['out']}: scan range parsed {scan_range!r}, expected "
            f"{[SCAN_START_A, SCAN_END_A, SCAN_STEPS]!r}"
        )
    peak_step = orca_out.scan_peak_step(scan["out"])
    peak_frame_rel = f"{STAGE_REL_DIR}/{peak_frame_name(peak_step)}"
    peak_frame = orca_runner.resolve_output(peak_frame_rel)
    if not os.path.isfile(peak_frame):
        _record_and_fail(
            f"{peak_frame}: ORCA's scan frame for peak step {peak_step} "
            f"not found"
        )
    peak_n, peak_comment, peak_els, peak_coords = structures.read_xyz_frame(
        peak_frame
    )
    if peak_n != structures.N_ATOMS:
        _record_and_fail(
            f"{peak_frame}: {peak_n} atoms, expected "
            f"{structures.N_ATOMS} (chorismate atom order)"
        )
    peak_energy = structures.energy_from_comment(peak_comment, peak_frame)
    n_fragments, heavy = structures.heavy_fragments(
        peak_els, peak_coords, tol=structures.TS_BOND_TOLERANCE_A
    )
    peak_kd = structures.key_distances(peak_coords)
    o1_c7 = peak_kd["O1-C7 (breaking)"]
    c11_c16_peak = peak_kd["C11-C16 (forming)"]
    record["scan"] = {
        "range_A": [SCAN_START_A, SCAN_END_A, SCAN_STEPS],
        "peak_step": peak_step,
        "peak_frame": os.path.basename(peak_frame),
        "peak_energy_eh": peak_energy,
        "peak_n_fragments": n_fragments,
        "peak_heavy_fragment_sizes": heavy,
        "peak_o1_c7_A": o1_c7,
        "peak_c11_c16_A": c11_c16_peak,
        "wall_seconds": scan["wall_seconds"],
    }
    if n_fragments != 1:
        _record_and_fail(
            f"{peak_frame}: scan peak is {n_fragments} fragment(s) under "
            f"the TS-aware tolerance {structures.TS_BOND_TOLERANCE_A} A "
            f"(heavy sizes {heavy}) — the scan did not keep the molecule "
            f"connected"
        )
    if not (o1_c7 <= PEAK_BOND_MAX_A and c11_c16_peak <= PEAK_BOND_MAX_A):
        _record_and_fail(
            f"{peak_frame}: a [3,3] bond exceeds {PEAK_BOND_MAX_A} A at the "
            f"peak — O1-C7 {o1_c7:.3f} / C11-C16 {c11_c16_peak:.3f} "
            f"(both must stay <= {PEAK_BOND_MAX_A} A)"
        )
    log(f"ts stage: scan peak step {peak_step} (E {peak_energy!r} Eh, "
        f"O1-C7 {o1_c7:.2f} / C11-C16 {c11_c16_peak:.2f} A, 1 fragment)")

    # ---- 3. the 2-constraint OPT from the peak frame -> the TS guess
    constr = _run_job("constr", render_constr_inp(peak_els, peak_coords),
                      False)
    if not constr["opt_converged"]:
        _record_and_fail(
            f"{constr['out']}: the 2-constraint Opt did not converge"
        )
    constr_xyz_rel = f"{STAGE_REL_DIR}/constr.xyz"
    constr_xyz = orca_runner.resolve_output(constr_xyz_rel)
    if not os.path.isfile(constr_xyz):
        _record_and_fail(
            f"{constr_xyz}: ORCA's optimised geometry not found — the TS "
            f"guess has to come from constr.xyz"
        )
    _cn, _cm, guess_els, guess_coords = structures.read_xyz_frame(constr_xyz)
    guess_nfr, _gv = structures.heavy_fragments(
        guess_els, guess_coords, tol=structures.TS_BOND_TOLERANCE_A
    )
    guess_kd = structures.key_distances(guess_coords)
    record["constr"] = {
        "energy_eh": constr["final_energy_eh"],
        "opt_converged": constr["opt_converged"],
        "n_fragments": guess_nfr,
        "o1_c7_A": guess_kd["O1-C7 (breaking)"],
        "c11_c16_A": guess_kd["C11-C16 (forming)"],
        "wall_seconds": constr["wall_seconds"],
    }
    if guess_nfr != 1:
        _record_and_fail(
            f"{constr_xyz}: the 2-constraint TS guess is {guess_nfr} "
            f"fragment(s) under tolerance {structures.TS_BOND_TOLERANCE_A} "
            f"(expected 1)"
        )
    log(f"ts stage: TS guess (constr) E {constr['final_energy_eh']!r} Eh, "
        f"1 fragment, [3,3] at {guess_kd['O1-C7 (breaking)']:.2f} / "
        f"{guess_kd['C11-C16 (forming)']:.2f} A")

    # ---- 4. OptTS + analytic FREQ (Calc_Hess) from the guess
    ts_source = TS_SOURCE_OPTTS
    try:
        ts = _run_job("optts", render_optts_inp(guess_els, guess_coords), True)
    except ValueError:
        # Clean exit without normal termination: the v1-proven fallback —
        # a gradient-converged saddle (an `optts.xyz` with a converged
        # gradient) is characterised by ONE standalone analytic Freq job;
        # its Hessian becomes the hand-off Hessian (optts.hess). Never
        # NumFreq. Dormant contingency, expected never to fire.
        # SCF failure is NOT a fallback trigger (an SCF-failed .out is
        # never silently accepted) — and neither is a missing .out (the
        # original error stands).
        optts_out = orca_runner.resolve_output(
            f"{STAGE_REL_DIR}/optts.out")
        fallback_xyz = orca_runner.resolve_output(f"{STAGE_REL_DIR}/optts.xyz")
        if not os.path.isfile(optts_out):
            raise
        try:
            orca_out.require_scf_converged(optts_out)
        except ValueError as scf_err:
            _record_and_fail(
                f"{optts_out}: {scf_err} — the OptTS run failed on SCF; "
                f"no fallback applies"
            )
        if not orca_out.optimization_converged(optts_out):
            _record_and_fail(
                f"{optts_out}: the OptTS job exited cleanly without normal "
                f"termination and WITHOUT a gradient-converged saddle "
                f"(no 'THE OPTIMIZATION HAS CONVERGED' marker) — no "
                f"fallback applies"
            )
        if not os.path.isfile(fallback_xyz):
            _record_and_fail(
                f"{fallback_xyz}: OptTS left no optts.xyz — no saddle to "
                f"characterise"
            )
        log("ts stage: OptTS left a gradient-converged saddle — "
            "characterising it with the standalone Freq fallback")
        ts = _run_job("ts_freq", render_ts_freq_inp(), True)
        ts_source = TS_SOURCE_FALLBACK
        freq_hess = orca_runner.resolve_output(f"{STAGE_REL_DIR}/ts_freq.hess")
        if not os.path.isfile(freq_hess):
            _record_and_fail(
                f"{freq_hess}: the standalone Freq produced no Hessian — "
                f"the IRC hand-off needs one"
            )
        shutil.copyfile(freq_hess, orca_runner.resolve_output(
            f"{STAGE_REL_DIR}/optts.hess"))

    # ---- 5. the TS verification asserts (observed + delta on failure)
    record["ts_source"] = ts_source
    ts_els, ts_coords = ts["geometry"]
    ts_kd = structures.key_distances(ts_coords)
    n_fragments, heavy = structures.heavy_fragments(
        ts_els, ts_coords, tol=structures.TS_BOND_TOLERANCE_A,
    )
    record["optts"] = {
        "energy_eh": ts["final_energy_eh"],
        "optts_converged": ts["opt_converged"],
        "n_modes": ts["n_modes"],
        "n_imaginary": ts["n_imaginary"],
        "imaginary_modes": ts["imaginary_modes"],
        "n_soft_imaginary": ts["n_soft_imaginary"],
        "n_fragments": n_fragments,
        "key_distances": ts_kd,
        "wall_seconds": ts["wall_seconds"],
    }

    if ts_source == TS_SOURCE_OPTTS and not ts["opt_converged"]:
        _record_and_fail(
            f"{ts['out']}: the OptTS run did not converge "
            f"('THE OPTIMIZATION HAS CONVERGED' absent)"
        )
    if (ts["n_modes"], ts["n_imaginary"]) != (78, 1):
        _record_and_fail(
            f"{ts['out']}: mode structure {ts['n_modes']} modes / "
            f"{ts['n_imaginary']} imaginary {ts['imaginary_modes']!r} — "
            f"expected 78 modes with EXACTLY 1 imaginary"
        )
    mode_idx, imag_value = ts["imaginary_modes"][0]
    imag_delta = abs(imag_value - TS_IMAG_TARGET_CM1)
    if imag_delta > TS_IMAG_TOL_CM1:
        _record_and_fail(
            f"{ts['out']}: imaginary mode {imag_value!r} cm-1 (mode "
            f"{mode_idx}) — target {TS_IMAG_TARGET_CM1!r} +- "
            f"{TS_IMAG_TOL_CM1} (delta {imag_delta:.2e})"
        )
    if ts["n_soft_imaginary"] != 0:
        _record_and_fail(
            f"{ts['out']}: {ts['n_soft_imaginary']} soft imaginary modes "
            f"in the -50..-10 window — must be empty"
        )
    if n_fragments != 1:
        _record_and_fail(
            f"{ts['out']}: the TS geometry is {n_fragments} fragment(s) "
            f"(heavy sizes {heavy}) under tolerance "
            f"{structures.TS_BOND_TOLERANCE_A} — expected 1 connected "
            f"fragment"
        )
    e_delta = abs(ts["final_energy_eh"] - TS_E_TARGET_EH)
    if e_delta > ENERGY_TOL_EH:
        _record_and_fail(
            f"{ts['out']}: TS energy {ts['final_energy_eh']!r} Eh — target "
            f"{TS_E_TARGET_EH!r} +- {ENERGY_TOL_EH:.0e} "
            f"(delta {e_delta:.3e})"
        )
    for label, target in TS_KEY_DIST_TARGETS.items():
        got = ts_kd[label]
        delta = abs(got - target)
        if delta > TS_KEY_DIST_TOL_A:
            _record_and_fail(
                f"{ts['out']}: TS key distance {label} {got!r} A — target "
                f"{target!r} +- {TS_KEY_DIST_TOL_A:.0e} "
                f"(delta {delta:.3e})"
            )

    # ---- 6. the stage deliverables + the record
    record["accepted"] = "optts"
    record["accepted_energy_eh"] = ts["final_energy_eh"]
    deliverables = []
    for artifact in ("optts.xyz", "optts.hess"):
        path = orca_runner.resolve_output(f"{STAGE_REL_DIR}/{artifact}")
        if not os.path.isfile(path):
            _record_and_fail(
                f"{path}: the verified TS deliverable {artifact!r} not "
                f"found in the stage directory"
            )
        deliverables.append(artifact)
    record["deliverables"] = deliverables

    path = _write_record()
    log(f"ts stage: TS verified — E {ts['final_energy_eh']!r} Eh, "
        f"imaginary [{ts['imaginary_modes'][0][0]}, "
        f"{ts['imaginary_modes'][0][1]!r}] cm-1, 1 fragment [{ts_source}]")
    log(f"ts stage: record: {path}")
    return record