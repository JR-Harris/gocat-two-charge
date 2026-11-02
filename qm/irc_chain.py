"""The IRC stage (`irc`): the reaction-coordinate validation from the
v2 TS plus the two endpoint re-optimisations.

Semantics confirmed against the frozen v1 archive
(src/scripts/irc_from_ts.py, optimize_irc_endpoints.py; see the
stage-5 log). The IRC validates the coordinate-drive TS — one branch
must descend to a reactive pseudo-diaxial chair chorismate, the other
to prephenate — and the ORCA-written branch endpoints are re-optimised
into the final R/P geometries (analytic Freq, the standard
`r2SCAN-3c Opt Freq TightSCF` template).

Hessian hand-off (exact): the ts stage's final `optts.hess` is copied
VERBATIM into step3_irc/ (the IRC input reads it by the relative quoted
name `Hess_Filename "optts.hess"`; ORCA's cwd = the stage directory).

Endpoint assignment is by CLASSIFICATION, never by branch direction —
the imaginary mode's sign/direction is arbitrary between re-runs and
the branch labels mean nothing a priori:

  reactant-like  (O1-C7 < 1.8 A, C11-C16 open > 2.5 A, <= 3.2 A — the
                  reactive chair test)  -> the chorismate_reactive job
  product-like   (O1-C7 > 2.5 A, C11-C16 < 2.0 A) -> the prephenate job

Fail-hard classification: a reactant-like endpoint outside the
reactive-chair window (C11-C16 > 3.2 A would be the non-reactive global
minimum — the classification test Jessica flagged) or an endpoint that
fits neither pattern fails the stage. No energy gates on the RAW IRC
endpoints: their energies are recorded, never gated (integration paths
between slightly different TS geometries land within ~1e-3 Eh of the
v1 endpoint energies). The derived kcal/mol numbers are recorded here
and gated by the gate scripts at +-0.05 (see the tolerances note in
the derived constants).

No promotion: the re-optimised minima (chorismate_reactive/ and
prephenate/ subfolders, the v1 archive layout) stay in the stage tree;
canonical `data/derived/structures/` names appear only at a later
briefing's promotion.
"""

import json
import os
import shutil

from qm import inputs, orca_out, orca_runner, structures, ts_chain

STAGE_REL_DIR = "data/derived/orca_v2/step3_irc"
TS_STAGE_REL_DIR = ts_chain.STAGE_REL_DIR
EPO_REL_DIR = "data/derived/orca_v2/step3_irc_endpoints_opt"
RECORD_NAME = "irc_record.json"

# ORCA writes the branch endpoint frames as <base>_IRC_{F,B}.xyz
# (base = the input stem = "irc").
FORWARD_XYZ = "irc_IRC_F.xyz"
BACKWARD_XYZ = "irc_IRC_B.xyz"

# %irc MaxIter (default 20; bumped to 40 for the flat torsions, v1).
IRC_MAXITER = 40

# Endpoint connectivity: bonded molecules at the standard tolerance,
# each a single fragment with 16 heavy atoms.
ENDPOINT_FRAGMENT_TOL_A = structures.BOND_TOLERANCE_A
HEAVY_FRAGMENT_SIZES = [16]

# Endpoint classification thresholds (the briefing rules; fail-hard on
# classification). The reactive-chair test: C11-C16 in (2.5, 3.2].
REACT_O1C7_MAX_A = 1.8      # O1-C7 intact ...
REACT_C11C16_MIN_A = 2.5    # ... C11-C16 open ...
REACTIVE_CHAIR_MAX_A = 3.2  # ... and the folded reactive chair
PROD_O1C7_MIN_A = 2.5       # O1-C7 broken
PROD_C11C16_MAX_A = 2.0     # ... C11-C16 formed

CLASSIFICATION_REACTANT = "reactant-like (O1-C7 intact, C11-C16 open)"
CHORISMATE_TYPE_REACTIVE_CHAIR = "reactive chair (C11-C16 <=3.2 A)"
CLASSIFICATION_PRODUCT = "product-like (O1-C7 broken, C11-C16 formed)"

# The two endpoint re-optimisation jobs (v1 archive layout: one
# subfolder per job, job tag = folder name).
REOPT_JOBS = ("chorismate_reactive", "prephenate")

# Re-optimisation targets (the folded-quantities anchors).
REOPT_ENERGY_TARGETS = {
    "chorismate_reactive": -838.267636353892,
    "prephenate": -838.287951337375,
}
ENERGY_TOL_EH = ts_chain.ENERGY_TOL_EH
N_MODES = 78

# Derived kcal/mol targets +-0.05. The tolerance derives honestly from
# the component gates (each +-1e-5 Eh ~ 0.0063 kcal/mol, and the
# accepted non-reactive minimum re-runs sit ~5.35e-6 Eh from its v1
# target): the folded quantities cannot honestly be gated at +-0.001.
KCAL_PER_EH = 627.5094740631
BARRIER_TARGET_KCAL = 14.7227    # (E_TS - E_reactive-chair) * KCAL_PER_EH
FOLDING_TARGET_KCAL = 2.1806     # reactive above non-reactive (convention)
REACTION_TARGET_KCAL = -12.7478  # (E_prephenate - E_reactive) * KCAL_PER_EH
DERIVED_TOL_KCAL = 0.05

# Gate-side fallback reference for the folding penalty when the
# returned tree carries no endpoints_record.json (the v1 archive
# predates the records): the locked orca_summary.json's chor value.
# The STAGE itself requires the record — it never falls back to the
# possibly-stale locked summary.
LOCKED_ORCA_SUMMARY = "data/derived/orca_summary.json"
V1_NONREACTIVE_EH = -838.271111373248


# ---------------------------------------------------------------------------
# renderers + classification + the nonreactive reference
# ---------------------------------------------------------------------------

def render_irc_inp(elements, coords):
    """The IRC input, reading the TS Hessian by the relative quoted
    name (InitHess read — no numerical Hessian recomputation), 0-based
    monitored internals: O1-C7 = { B 0 6 }, C11-C16 = { B 10 15 }.
    MaxIter 40 (default 20; bumped for the flat torsions)."""
    lines = [
        f"! r2SCAN-3c IRC TightSCF PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        "%irc",
        f"  MaxIter {IRC_MAXITER}",
        "  InitHess  read",
        '  Hess_Filename "optts.hess"',
        "  Monitor_Internals",
        "    { B 0 6 }",
        "    { B 10 15 }",
        "  end",
        "end",
        *inputs.geometry_lines(elements, coords),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)
    return text


def classify_irc_endpoint(distances):
    """(classification, chorismate_type, job_tag) for an IRC endpoint
    frame — raises on anything that is not one of the two clean
    patterns (the stage and the gates treat a raise as the fail-hard
    classification breach):

      reactant-like: O1-C7 < 1.8 AND C11-C16 > 2.5 — and C11-C16 <= 3.2
                     (the reactive chair test; a reactant-like endpoint
                     with C11-C16 > 3.2 is the NON-reactive minimum,
                     which does not connect to the TS);
      product-like : O1-C7 > 2.5 AND C11-C16 < 2.0.
    """
    o1_c7 = distances["O1-C7 (breaking)"]
    c11_c16 = distances["C11-C16 (forming)"]
    if o1_c7 < REACT_O1C7_MAX_A and REACT_C11C16_MIN_A < c11_c16:
        if c11_c16 > REACTIVE_CHAIR_MAX_A:
            raise ValueError(
                f"reactant-like endpoint with C11-C16 {c11_c16:.3f} A — "
                f"not within the reactive-chair criterion "
                f"(<= {REACTIVE_CHAIR_MAX_A} A)"
            )
        return (CLASSIFICATION_REACTANT, CHORISMATE_TYPE_REACTIVE_CHAIR,
                "chorismate_reactive")
    if o1_c7 > PROD_O1C7_MIN_A and c11_c16 < PROD_C11C16_MAX_A:
        return CLASSIFICATION_PRODUCT, None, "prephenate"
    raise ValueError(
        f"IRC endpoint fits neither pattern: O1-C7 {o1_c7:.3f} A "
        f"(need < {REACT_O1C7_MAX_A} intact or > {PROD_O1C7_MIN_A} broken), "
        f"C11-C16 {c11_c16:.3f} A (need > {REACT_C11C16_MIN_A} open or "
        f"< {PROD_C11C16_MAX_A} formed)"
    )


def accepted_nonreactive_energy(out_root):
    """The accepted non-reactive minimum's energy for the folding
    penalty, resolved against the returned tree (gate-side use):

      * v2 (record present): step2_endpoints/endpoints_record.json's
        accepted chorismate energy, consistency-asserted;
      * v1: the locked data/derived/orca_summary.json's chor value when
        it still carries the v1 reference; since the 2026-10-07
        promotion the canonical summary carries the v2 values, so the
        ARCHIVED v1 summary (data/derived/archive/*/orca_summary.json,
        newest event) is the v1 reference.

    Returns (energy_eh, source_description). Fails hard when neither
    source is available."""
    record_path = os.path.join(out_root, "step2_endpoints/endpoints_record.json") \
        if out_root else ts_chain.ENDPOINTS_RECORD
    last_observed = []
    if os.path.isfile(record_path):
        rec = ts_chain.read_endpoints_reactant(record_path)
        return rec["accepted_energy_eh"], record_path
    for path, note in _v1_reference_candidates():
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            locked = json.load(fh)
        energy = locked.get("chor", {}).get("final_energy_eh")
        if energy == V1_NONREACTIVE_EH:
            return energy, f"{path} {note}"
        # a non-matching candidate does not abort the chain: the next
        # (e.g. the archived v1 summary after the promotion) may win
        last_observed.append((path, energy))
    raise ValueError(
        f"no accepted non-reactive reference: neither {record_path} nor "
        f"a v1 orca_summary.json (locked or archived) is available "
        f"(candidates with wrong values: {last_observed})")


def _v1_reference_candidates():
    """(path, note) candidates for the v1 chor reference, in order:
    the locked summary, then the archived v1 ones (newest event first)."""
    candidates = [(LOCKED_ORCA_SUMMARY, "(v1 reference)")]
    archive_root = os.path.join("data", "derived", "archive")
    if os.path.isdir(archive_root):
        for event in sorted(os.listdir(archive_root), reverse=True):
            p = os.path.join(archive_root, event, "orca_summary.json")
            if os.path.isfile(p):
                candidates.append((p, "(archived v1 reference)"))
    return candidates


# ---------------------------------------------------------------------------
# the irc stage flow
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the irc stage flow
# ---------------------------------------------------------------------------

def execute_irc(verbose=True):
    """The `irc` stage: verbatim Hessian copy -> IRC -> endpoint
    classification + assignment -> the two re-optimisations ->
    derived numbers -> record. Fail-hard at every assert; resumable at
    every job level via the runner's resume gate (per-job wall_seconds:
    fresh runs measure, resume reports 0.0). Returns the record dict."""
    log = print if verbose else (lambda *_a, **_k: None)
    jobs = []
    record = {
        "stage": "irc",
        "attempts": 0,
        "jobs": jobs,
        "ts_stage_dir": TS_STAGE_REL_DIR,
    }

    def _record_and_fail(msg):
        path = orca_runner.resolve_output(f"{STAGE_REL_DIR}/{RECORD_NAME}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        log(f"irc stage: FAILED — record left at {path}")
        raise ValueError(f"irc stage FAILED: {msg}")

    def _run_job(name, text, work_rel, need_frequencies):
        """One ORCA job under work_rel (input <work_rel>/<name>.inp,
        output <work_rel>/<name>.out) + the composite parse."""
        inp_path = orca_runner.resolve_output(f"{work_rel}/{name}.inp")
        os.makedirs(os.path.dirname(os.path.abspath(inp_path)), exist_ok=True)
        with open(inp_path, "w") as fh:
            fh.write(text)
        run = orca_runner.run_orca(inp_path, f"{work_rel}/{name}.out")
        parsed = orca_out.parse_output(run["out"],
                                       need_frequencies=need_frequencies)
        parsed["out"] = run["out"]
        parsed["skipped"] = run["skipped"]
        parsed["wall_seconds"] = run["wall_seconds"]
        jobs.append({"name": name, "skipped": run["skipped"],
                     "wall_seconds": run["wall_seconds"]})
        return parsed

    # ---- 1. the ts stage's inputs (fail-hard, named files)
    ts_xyz_rel = f"{TS_STAGE_REL_DIR}/optts.xyz"
    ts_xyz = orca_runner.resolve_output(ts_xyz_rel)
    ts_hess_rel = f"{TS_STAGE_REL_DIR}/optts.hess"
    ts_hess = orca_runner.resolve_output(ts_hess_rel)
    for path in (ts_xyz, ts_hess):
        if not os.path.isfile(path):
            _record_and_fail(
                f"{path}: the ts stage's deliverable not found — run the "
                f"ts stage first (--execute ts)"
            )
    _n, _cmt, elements, coords = structures.read_xyz_frame(ts_xyz)
    if _n != structures.N_ATOMS:
        _record_and_fail(
            f"{ts_xyz}: {_n} atoms, expected {structures.N_ATOMS}"
        )

    # ---- 2. the verbatim Hessian copy into the stage directory (the
    # IRC reads it by the relative quoted filename)
    hess_copy = orca_runner.resolve_output(f"{STAGE_REL_DIR}/optts.hess")
    os.makedirs(os.path.dirname(os.path.abspath(hess_copy)), exist_ok=True)
    shutil.copyfile(ts_hess, hess_copy)
    with open(ts_hess, "rb") as fh:
        src_bytes = fh.read()
    with open(hess_copy, "rb") as fh:
        dst_bytes = fh.read()
    if src_bytes != dst_bytes:
        _record_and_fail(
            f"{hess_copy}: the copied optts.hess differs byte-wise from "
            f"{ts_hess} — the hand-off must be verbatim"
        )
    record["ts_stage"] = {
        "optts.xyz": ts_xyz_rel,
        "optts.hess": ts_hess_rel,
        "hess_copied_verbatim": True,
    }
    log(f"irc stage: hess hand-off verbatim ({len(src_bytes)} bytes "
        f"from {ts_hess_rel})")

    # ---- 3. the IRC from the TS geometry; the TS energy for the
    # derived numbers = the full-precision parse of the ts stage's .out
    ts_parsed = orca_out.parse_output(
        orca_runner.resolve_output(f"{TS_STAGE_REL_DIR}/optts.out"),
        need_frequencies=False,
    )
    ts_energy = ts_parsed["final_energy_eh"]

    irc = _run_job("irc", render_irc_inp(elements, coords), STAGE_REL_DIR,
                   need_frequencies=False)
    try:
        ts_energy_summary, n_rows = orca_out.irc_path_summary(irc["out"])
    except ValueError as e:
        _record_and_fail(f"{irc['out']}: IRC PATH SUMMARY parse failed ({e})")
    record["irc"] = {
        "n_summary_steps": n_rows,
        "ts_energy_eh_path_summary": ts_energy_summary,
        "wall_seconds": irc["wall_seconds"],
    }
    log(f"irc stage: IRC parsed — {n_rows} summary rows, "
        f"<= TS at {ts_energy_summary!r} Eh")

    # ---- 4. the branch endpoint frames: existence, connectivity,
    # classification (fail-hard on classification, NEVER on energies)
    frames = {}
    for branch, name in (("forward", FORWARD_XYZ), ("backward", BACKWARD_XYZ)):
        path = orca_runner.resolve_output(f"{STAGE_REL_DIR}/{name}")
        if not os.path.isfile(path):
            _record_and_fail(
                f"{path}: the {branch} branch endpoint frame not found"
            )
        _n, comment, els, fcoords = structures.read_xyz_frame(path)
        energy = structures.energy_from_comment(comment, path)
        n_fragments, heavy = structures.heavy_fragments(
            els, fcoords, tol=ENDPOINT_FRAGMENT_TOL_A)
        if n_fragments != 1 or heavy != HEAVY_FRAGMENT_SIZES:
            _record_and_fail(
                f"{path}: the {branch} endpoint is {n_fragments} "
                f"fragment(s) {heavy} under tolerance "
                f"{ENDPOINT_FRAGMENT_TOL_A} — expected a single bonded "
                f"molecule with heavy sizes {HEAVY_FRAGMENT_SIZES}"
            )
        kd = structures.key_distances(fcoords)
        try:
            classification, chorismate_type, job_tag = (
                classify_irc_endpoint(kd))
        except ValueError as e:
            _record_and_fail(
                f"{path}: endpoint classification failed ({e})"
            )
        frames[branch] = {
            "file": name,
            "energy_eh": energy,
            "n_fragments": n_fragments,
            "heavy_fragment_sizes": heavy,
            "key_distances": kd,
            "classification": classification,
            "chorismate_type": chorismate_type,
            "job": job_tag,
        }
        log(f"irc stage: {branch} endpoint {name}: E {energy!r} Eh, "
            f"{classification}"
            + (f" [{chorismate_type}]" if chorismate_type else ""))
    reactant_branches = [b for b, f in frames.items()
                         if f["classification"] == CLASSIFICATION_REACTANT]
    product_branches = [b for b, f in frames.items()
                        if f["classification"] == CLASSIFICATION_PRODUCT]
    if len(reactant_branches) != 1 or len(product_branches) != 1:
        _record_and_fail(
            f"IRC endpoints must be one reactant-like (reactive chair) + "
            f"one product-like; got reactant-like on {reactant_branches}, "
            f"product-like on {product_branches}"
        )
    assigned = {f["job"]: b for b, f in frames.items()}
    if sorted(assigned) != sorted(REOPT_JOBS):
        _record_and_fail(
            f"endpoint assignment covers {sorted(assigned)}, expected "
            f"{sorted(REOPT_JOBS)}"
        )
    record["endpoints"] = frames
    record["reactant_side"] = reactant_branches[0]
    record["product_side"] = product_branches[0]
    log(f"irc stage: assignment by classification — "
        f"{reactant_branches[0]} -> chorismate_reactive, "
        f"{product_branches[0]} -> prephenate")

    # ---- 5. the endpoint re-optimisations (classification-assigned)
    reopt = {}
    record["reopt"] = reopt
    for tag in REOPT_JOBS:
        src = frames[assigned[tag]]
        _n, _cmt, els, fcoords = structures.read_xyz_frame(
            orca_runner.resolve_output(f"{STAGE_REL_DIR}/{src['file']}")
        )
        parsed = _run_job(tag, inputs.render_opt_freq_inp(els, fcoords),
                          f"{EPO_REL_DIR}/{tag}", need_frequencies=True)
        els_r, coords_r = parsed["geometry"]
        n_fragments, heavy = structures.heavy_fragments(els_r, coords_r)
        kd = structures.key_distances(coords_r)
        classification, _ct = structures.classify_geometry(kd)
        target = REOPT_ENERGY_TARGETS[tag]
        delta = abs(parsed["final_energy_eh"] - target)
        reopt[tag] = {
            "branch": assigned[tag],
            "source_xyz": f"{STAGE_REL_DIR}/{src['file']}",
            "energy_eh": parsed["final_energy_eh"],
            "opt_converged": parsed["opt_converged"],
            "n_modes": parsed["n_modes"],
            "n_imaginary": parsed["n_imaginary"],
            "n_soft_imaginary": parsed["n_soft_imaginary"],
            "n_fragments": n_fragments,
            "heavy_fragment_sizes": heavy,
            "key_distances": kd,
            "wall_seconds": parsed["wall_seconds"],
        }
        if not parsed["opt_converged"]:
            _record_and_fail(
                f"{parsed['out']}: the {tag} re-optimisation did not "
                f"converge"
            )
        if (parsed["n_modes"], parsed["n_imaginary"],
                parsed["n_soft_imaginary"]) != (N_MODES, 0, 0):
            _record_and_fail(
                f"{parsed['out']}: {tag} modes "
                f"{parsed['n_modes']}/{parsed['n_imaginary']}/"
                f"{parsed['n_soft_imaginary']} (modes/hard/soft) — "
                f"expected {N_MODES}/0/0"
            )
        if n_fragments != 1:
            _record_and_fail(
                f"{parsed['out']}: {tag} is {n_fragments} fragment(s) "
                f"{heavy} — expected 1"
            )
        if delta > ENERGY_TOL_EH:
            _record_and_fail(
                f"{parsed['out']}: {tag} energy "
                f"{parsed['final_energy_eh']!r} Eh — target {target!r} +- "
                f"{ENERGY_TOL_EH:.0e} (delta {delta:.3e})"
            )
        want_class = (CLASSIFICATION_REACTANT
                      if tag == "chorismate_reactive"
                      else CLASSIFICATION_PRODUCT)
        if classification != want_class:
            _record_and_fail(
                f"{parsed['out']}: {tag} classifies {classification!r} — "
                f"expected {want_class!r}"
            )
        log(f"irc stage: {tag} re-optimised — E "
            f"{parsed['final_energy_eh']!r} Eh (delta {delta:.1e}), 0/0, "
            f"1 fragment")

    # ---- 6. the derived numbers (recorded exactly; gated by the gate
    # scripts at +-0.05). The folding sign convention: reactive above
    # non-reactive.
    reactive_e = reopt["chorismate_reactive"]["energy_eh"]
    preph_e = reopt["prephenate"]["energy_eh"]
    nonreactive = ts_chain.read_endpoints_reactant(
        orca_runner.resolve_output(ts_chain.ENDPOINTS_RECORD)
    )
    nonreactive_e = nonreactive["accepted_energy_eh"]
    record["derived"] = {
        "ts_energy_eh": ts_energy,
        "reactive_chair_energy_eh": reactive_e,
        "prephenate_energy_eh": preph_e,
        "nonreactive_energy_eh": nonreactive_e,
        "nonreactive_source": ts_chain.ENDPOINTS_RECORD,
        "barrier_kcal_mol": (ts_energy - reactive_e) * KCAL_PER_EH,
        "folding_penalty_kcal_mol": (reactive_e - nonreactive_e) * KCAL_PER_EH,
        "reaction_energy_kcal_mol": (preph_e - reactive_e) * KCAL_PER_EH,
        "sign_convention_folding": "reactive above non-reactive",
        "kcal_per_eh": KCAL_PER_EH,
    }
    d = record["derived"]
    log(f"irc stage: derived — barrier "
        f"{d['barrier_kcal_mol']:.4f} kcal/mol "
        f"(target {BARRIER_TARGET_KCAL}); folding "
        f"{d['folding_penalty_kcal_mol']:+.4f} (target "
        f"{FOLDING_TARGET_KCAL:+.4f}); reaction "
        f"{d['reaction_energy_kcal_mol']:.4f} "
        f"(target {REACTION_TARGET_KCAL})")

    record["accepted"] = {
        "reactive_chair": "chorismate_reactive",
        "prephenate": "prephenate",
        "ts": "optts.xyz",
    }
    path = orca_runner.resolve_output(f"{STAGE_REL_DIR}/{RECORD_NAME}")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    log("irc stage: both re-optimised minima verified "
        f"(0 hard / 0 soft, {N_MODES} modes, converged)")
    log(f"irc stage: record: {path}")
    return record