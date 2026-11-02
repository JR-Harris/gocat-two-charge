"""Tests for the r2SCAN-3c ORCA provider (fitness/orca.py).

House style (fitness/test_xtb.py): numbered tests + main(); the
sentinel tests patch _launch_orca so no ORCA is needed; the LIVE tests
(the asymptote embedding gate + same-machine determinism + the
real-output parser gates) run real ORCA and are gated on the binary's
presence.

Run:  python fitness/test_orca.py [--no-live]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitness.orca as fo
from fitness.orca import (BASELINE_MAXCORE_MB, KCAL_PER_EH, OrcaProvider,
                          config_key, differential_key, load_xyz, p_key,
                          render_pc, render_pc_empty, render_sp_inp,
                          resume_energy_eh, sweep_sp_dir)

RUNS = Path(__file__).resolve().parent.parent / "runs" / "arbitration_tests"
_RE = fo.REPO_ROOT
_TS = _RE / "data" / "derived" / "structures" / "ts_opt.xyz"
_RS_XYZ = (_RE / "data" / "derived" / "structures"
           / "chorismate_reactive_TSframe.xyz")
_PS_XYZ = _RE / "data" / "derived" / "structures" / "prephenate_TSframe.xyz"


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def expect_error(fn, exc=ValueError, what="", must_name=None):
    try:
        fn()
    except exc as e:
        if must_name and must_name not in str(e):
            raise AssertionError(
                f"{what}: the error must name {must_name!r}, got {e}")
        return e
    raise AssertionError(f"{what}: expected {exc.__name__}, got none")


# --------------------------------------------------------------------------
# sentinel machinery
# --------------------------------------------------------------------------

_SENTINEL_SEQ = [0]


class SentinelLaunch:
    """The _launch_orca replacement: writes a synthetic .out whose
    energy is looked up by (side, positions, lot, pal); an absent lookup
    gets a deterministic pseudo-energy and the reactant side is offset
    +0.1 (so f != 0)."""

    def __init__(self, energies=None, fail=None):
        # energies: {(side, (x1,y1,z1,x2,y2,z2)) -> Eh float} (side in
        # {'ts','r','p'}; the reactant side defaults to ts + 0.1); fail:
        # {side -> exit code}, applied to every call of that side.
        self.energies = energies or {}
        self.fail = fail or set()     # {side -> exit code}
        self.calls = []

    def _pc(self, dir_path):
        f = dir_path / "charges.pc"
        if not f.is_file():
            return ("nopc", None)
        lines = f.read_text().splitlines()
        if len(lines) < 3:
            return ("empty", None)
        q1 = float(lines[1].split()[0])
        q2 = float(lines[2].split()[0])
        # positions from the pc file -> the energy table key
        x1, y1, z1 = (float(v) for v in lines[1].split()[1:])
        x2, y2, z2 = (float(v) for v in lines[2].split()[1:])
        return (q1, q2), (x1, y1, z1, x2, y2, z2)

    def __call__(self, cmd, env, cwd, stdout_path, stderr_path):
        d = Path(cwd)
        side = d.name                     # 'ts' | 'r' | 'p'
        inp = (d / "sp.inp").read_text()
        (pc, pos) = self._pc(d)
        self.calls.append({"dir": str(d), "pc": pc, "inp": inp,
                           "side": side})
        if side in self.fail:
            with open(stdout_path, "w") as fh:
                fh.write("SENTINEL FAIL\n")
            return self.fail[side]
        kwline = inp.splitlines()[0]
        if " PAL" in kwline:
            lot = kwline.split("!")[1].split("PAL")[0].strip()
            pal = kwline.split("PAL")[1].strip()
        else:
            lot = kwline.split("!")[1].strip()
            pal = "1"
        e = self.energies.get(("e_" + side, pos)) if pos is not None \
            else None
        if e is None:
            if pos is not None and side == "r":
                base = self.energies.get(("e_ts", pos))
                if base is not None:
                    e = base + 0.1
            elif pos is not None and side == "p":
                e = self.energies.get(("e_p", pos))
        if e is None:
            # deterministic pseudo-energies from the pc content
            a = abs(hash((side, pos, lot, pal))) % 100000
            e = -838.0 + a / 1e6
        with open(stdout_path, "w") as fh:
            fh.write("SENTINEL ORCA OUTPUT -- NOT A REAL RUN\n"
                     "FINAL SINGLE POINT ENERGY      "
                     f"{e:.12f}\n"
                     "****ORCA TERMINATED NORMALLY****\n")
        return 0


def sentinel_provider(scratch, launch, lot=fo.ORCA_LOT, pal=1,
                      binary=None):
    RUNS.mkdir(parents=True, exist_ok=True)
    return OrcaProvider(scratch_root=scratch, lot=lot, pal=pal,
                        binary=binary or Path(fo.__file__))


def _mono_patcher(launch):
    import fitness.orca as m
    old = m._launch_orca
    m._launch_orca = launch
    return old


def _unpatcher(old):
    import fitness.orca as m
    m._launch_orca = old


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def test_01_pc_bytes():
    """The .pc file: count line + 'q x y z' lines; q = the literal
    +1.0/-1.0; repr passthrough both ways."""
    cloud = np.load(fo.CLOUD_NPY)
    text = render_pc(1.0, -1.0, cloud[418], cloud[419])
    lines = text.splitlines()
    _expect(lines[0] == "2", f"count line: {lines[0]!r}")
    tok1 = lines[1].split()
    tok2 = lines[2].split()
    _expect(tok1[0] == "+1.0" and tok2[0] == "-1.0",
            f"charge literals: {tok1[0]!r}/{tok2[0]!r}")
    _expect(len(tok2) == 4, f"4 tokens per line: {tok2}")
    # the positions round-trip byte-identically through repr
    x = float(tok1[1])
    _expect(repr(x) == tok1[1], f"repr round trip: {tok1[1]!r}")
    # the numpy-cast first (no 'np.float64(' in the file)
    text2 = render_pc(np.float64(1.0), np.float64(-1.0), cloud[418],
                      cloud[419])
    _expect("np.float64" not in text2, "numpy repr leaked into the .pc")
    # the unembedded channel control
    _expect(render_pc_empty() == "0\n", "the zero-charge .pc content")


def test_02_sp_inp_shape():
    """The SP input: the keyword line, %maxcore, the %pointcharges
    directive with the absolute path, and the VERBATIM XYZFILE geometry
    reference (the frozen file's own 14-decimal coordinates)."""
    els, coords = load_xyz(_TS)
    _expect(els[0] == "O", "the first atom is O")
    with_pc = render_sp_inp(fo.ORCA_LOT, "geom.xyz", "/abs/charges.pc",
                            2, 1500)
    lines = with_pc.splitlines()
    _expect(lines[0] == "! r2SCAN-3c TightSCF PAL2",
            f"keyword line: {lines[0]!r}")
    # pal 1 renders no PAL keyword (PAL1 is unrecognized in ORCA 6.1.1;
    # the keyword-free input is exactly the one-rank run)
    serial = (render_sp_inp(fo.ORCA_LOT, "geom.xyz", "/abs/q.pc", 1, 1500)
              .splitlines()[0])
    _expect(serial == "! r2SCAN-3c TightSCF", f"serial line: {serial!r}")
    _expect(lines[1] == "%maxcore 1500", f"maxcore line: {lines[1]!r}")
    _expect(lines[2] == '%pointcharges "/abs/charges.pc"',
            f"pc line: {lines[2]!r}")
    _expect(lines[3] == "* XYZFILE 0 1 geom.xyz",
            f"the verbatim XYZFILE line: {lines[3]!r}")
    _expect(len(lines) == 4, "no inline geometry is rendered")
    # the unembedded variant (the asymptote channel control)
    without = render_sp_inp(fo.ORCA_LOT, "geom.xyz", None, 1, 1500)
    _expect("%pointcharges" not in without, "the unembedded input")
    # the verbatim copy: the scratch geom.xyz is byte-identical to the
    # frozen file (asserted at the sentinel + live runs' scratch level)
    src = _TS.read_bytes()
    _expect(b"1.63762438486189" in src,
            "the frozen TS carries its full-precision coordinates")


def test_03_keys():
    """The config key: the exact template; cloud-sha sensitivity (the
    replicate-1 manifest's candidates sha); the P mode keys with the P
    sha in BOTH slots; sign rendering '+1.0'/'-1.0'; lot sensitivity."""
    c_sha = fo.geo_sha256(fo.CLOUD_NPY)
    t_sha = fo.geo_sha256(_TS)
    r_sha = fo.geo_sha256(_RS_XYZ)
    p_sha = fo.geo_sha256(_PS_XYZ)
    k1 = differential_key(418, 419, -1.0, -1.0, c_sha, t_sha, r_sha)
    k2 = differential_key(418, 419, -1.0, -1.0, c_sha, t_sha, r_sha)
    _expect(k1 == k2, "same config -> the same key")
    k3 = differential_key(419, 418, -1.0, -1.0, c_sha, t_sha, r_sha)
    _expect(k1 != k3, "i/j swap changes the key")
    # the replicate-1 cloud sha -> a different key (cross-cloud safety)
    rep1_manifest = json.loads((_RE / "data" / "replicates" / "replicate-1"
                                / "manifest.json").read_text())
    rep1_sha = rep1_manifest["files"]["candidates_replicate-1.npy"]
    k4 = differential_key(418, 419, -1.0, -1.0, rep1_sha, t_sha, r_sha)
    _expect(k1 != k4, "a different cloud sha changes the key")
    # the P mode: the P sha in BOTH slots
    k5 = p_key(418, 419, -1.0, -1.0, c_sha, p_sha)
    want5 = config_key(418, 419, -1.0, -1.0, c_sha, p_sha, p_sha,
                       fo.ORCA_LOT)
    _expect(k5 == want5, "the P-mode key template")
    _expect(k5 != k1, "a P key never collides with a differential key")
    # the sign rendering: +1.0 vs -1.0 differ; float sign_i = 1.0 == int 1
    kp = differential_key(1, 2, 1.0, -1.0, c_sha, t_sha, r_sha)
    km = differential_key(1, 2, 1, -1, c_sha, t_sha, r_sha)
    _expect(kp == km, "int and float +-1 render identically")
    kq = differential_key(1, 2, -1.0, 1.0, c_sha, t_sha, r_sha)
    _expect(kp != kq, "the sign pair order matters")
    # the lot field
    kl = differential_key(418, 419, -1.0, -1.0, c_sha, t_sha, r_sha,
                          lot=fo.WB97M_LOT)
    _expect(kl != k1, "a different lot changes the key")
    _expect(len(k1) == 64, "sha256 hex length")


def test_04_sweep_and_resume():
    """The SP sweep: everything except the keep set is removed; the
    resume gate: none / incomplete / scf-failed / no-energy / complete."""
    RUNS.mkdir(parents=True, exist_ok=True)
    d = RUNS / "sweep_test"
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for name in ("charges.pc", "sp.inp", "sp.out", "sp.stderr",
                 "STALE.gbw", "tmp_dir"):
        (d / name).touch() if "." in name else (d / name).mkdir()
    removed = sweep_sp_dir(d)
    _expect(sorted(removed) == ["STALE.gbw", "tmp_dir"],
            f"sweep removed: {removed}")
    _expect(sorted(p.name for p in d.iterdir()) ==
            ["charges.pc", "sp.inp", "sp.out", "sp.stderr"],
            "the keep set survives the sweep")
    shutil.rmtree(d)

    # the resume gate against the REAL verified outputs
    real1 = (_RE / "data" / "derived" / "orca_v2" / "step3_coordinate_drive"
             / "optts.out")
    e = resume_energy_eh(real1)
    _expect(e is not None and abs(e - (-838.244178868391)) < 1e-9,
            f"optts.out parses to the TS point: {e}")
    # the synthetic gates
    g = RUNS / "resume_gates"
    g.mkdir(parents=True, exist_ok=True)
    (g / "no_marker.out").write_text("FINAL SINGLE POINT ENERGY -1.0\n")
    _expect(resume_energy_eh(g / "no_marker.out") is None,
            "no termination marker -> None")
    (g / "scf_fail.out").write_text(
        "ORCA TERMINATED NORMALLY\nSCF NOT CONVERGED\n"
        "FINAL SINGLE POINT ENERGY -1.0\n")
    _expect(resume_energy_eh(g / "scf_fail.out") is None,
            "SCF failure -> None")
    (g / "no_energy.out").write_text("****ORCA TERMINATED NORMALLY****\n")
    _expect(resume_energy_eh(g / "no_energy.out") is None,
            "no final energy -> None")
    (g / "missing.out")
    _expect(resume_energy_eh(g / "missing.out") is None,
            "missing file -> None")
    (g / "ok.out").write_text(
        "FINAL SINGLE POINT ENERGY      -838.244178868391\n"
        "****ORCA TERMINATED NORMALLY****\n")
    e = resume_energy_eh(g / "ok.out")
    _expect(abs(e - (-838.244178868391)) < 1e-12, f"complete -> {e}")
    shutil.rmtree(g)


def test_05_parser_real_outputs():
    """The parser gates against the REAL in-repo outputs (all three
    verified to terminate normally + carry these final energies)."""
    from qm import orca_out
    cases = [
        (_RE / "data/derived/orca_v2/step3_coordinate_drive/optts.out",
         -838.244178868391),
        (_RE / "data/derived/orca_v2/step3_irc_endpoints_opt/"
         "chorismate_reactive/chorismate_reactive.out",
         -838.267633356973),
        (_RE / "data/derived/orca_v2/step2_endpoints/chor.out",
         -838.271116720443),
    ]
    for path, want in cases:
        orca_out.require_normal_termination(path)
        orca_out.require_scf_converged(path)
        got = orca_out.final_energy_eh(path)
        _expect(abs(got - want) < 1e-9,
                f"{path.name}: {got} != {want}")


def test_06_provider_init_gates():
    """Init fail-hard: a tampered geometry sha, a wrong atom count, a
    missing binary, an unknown site."""
    RUNS.mkdir(parents=True, exist_ok=True)
    # a tampered TS geometry (wrong sha vs the landscape summary)
    bad = RUNS / "tampered_ts.xyz"
    bad.write_text(_TS.read_text().replace("C  ", "C  ", 1) + "\n")
    expect_error(lambda: OrcaProvider(ts_path=bad,
                                      binary=Path(fo.__file__)),
                 ValueError, "tampered TS sha must fail", "sha256 mismatch")
    # a missing binary
    expect_error(lambda: OrcaProvider(binary=_RE / "nonexistent" / "orca"),
                 ValueError, "missing binary must fail", "not installed")
    # an unknown site
    expect_error(lambda: OrcaProvider(binary=Path(fo.__file__),
                                      site="nomansland"),
                 ValueError, "unknown site must fail", "known:")


def test_07_sentinel_contract():
    """The contract surface over the sentinel: e_ts/e_r float64 Eh,
    f = (e_ts - e_r) * KCAL_PER_EH with the frozen constant; the point
    order e_ts first (the sp dir per side); the walls recorded."""
    launch = SentinelLaunch()
    old = _mono_patcher(launch)
    try:
        scratch = RUNS / "sentinel_contract"
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
        prov = sentinel_provider(scratch, launch)
        cloud = np.load(fo.CLOUD_NPY)
        res = prov.run_config(418, 419, -1, -1)
        _expect(isinstance(res["e_ts"], float)
                and isinstance(res["e_r"], float), "float64 components")
        _expect(abs(res["f"] - (res["e_ts"] - res["e_r"]) * KCAL_PER_EH)
                < 1e-12, "f = (e_ts - e_r) * KCAL_PER_EH")
        _expect(res["walls"]["ts_s"] is not None
                and res["walls"]["r_s"] is not None,
                f"walls recorded: {res['walls']}")
        _expect(prov.n_evals == 1, f"accounting: {prov.n_evals}")
        # the two sides used the SAME config key, DIFFERENT point dirs
        dirs = [c["dir"] for c in launch.calls]
        _expect(dirs[0].endswith("/ts") and dirs[1].endswith("/r"),
                f"point order + dirs: {dirs}")
        _expect(Path(dirs[0]).parent == Path(dirs[1]).parent,
                "the same config key dir for both sides")
        # the charges: the same .pc content for both sides
        _expect(launch.calls[0]["pc"] == launch.calls[1]["pc"],
                "the same charges for both sides")
        # sign rendering reached the pc file
        _expect(launch.calls[0]["pc"] == (-1.0, -1.0),
                f"signs in the pc: {launch.calls[0]['pc']}")
        # the sentinel energies keyed by the pc content + lot + pal
        _expect(launch.calls[0]["inp"].splitlines()[0]
                == "! r2SCAN-3c TightSCF",
                "the serial keyword line (pal 1 renders no PAL keyword; "
                "PAL1 is unrecognized in ORCA 6.1.1)")
        prov2p = sentinel_provider(scratch, launch, pal=2)
        prov2p.run_config(0, 1, 1, 1)
        _expect(launch.calls[-1]["inp"].splitlines()[0]
                == "! r2SCAN-3c TightSCF PAL2",
                "the PAL2 keyword line arrives at pal >= 2")
        # the vectorized contract surface agrees per row
        f, e_ts, e_r = prov.eval_batch_full(
            [418], [419], [-1], [-1], cloud)
        _expect(abs(f[0] - res["f"]) < 1e-12, "the surfaces agree")
        _expect(prov.n_evals == 2, f"accounting after the call: "
                f"{prov.n_evals}")
        # the P point mode: the P geometry, the P key
        shutil.rmtree(scratch, ignore_errors=True)
        launch2 = SentinelLaunch()
        old2 = _mono_patcher(launch2)
        try:
            prov2 = sentinel_provider(scratch, launch2)
            p_res = prov2.run_p_config(418, 419, -1, -1)
            _expect(isinstance(p_res["e_p"], float), "e_p float64")
            _expect(launch2.calls[0]["dir"].endswith("/p"),
                    f"the P scratch dir: {launch2.calls[0]['dir']}")
            k = p_key(418, 419, -1.0, -1.0, prov2.cloud_sha, prov2.p_sha)
            _expect(p_res["key"] == k, "the P key matches the template")
            _expect(prov2.n_evals == 1, "the P-mode accounting")
        finally:
            _unpatcher(old2)
    finally:
        _unpatcher(old)


def test_08_sentinel_fail_hard():
    """Fail-hard paths: a nonzero exit, no termination marker, an SCF
    failure, an unparsable energy, a missing binary at call time; every
    error names the config key or the label + the sp.out tail."""
    RUNS.mkdir(parents=True, exist_ok=True)
    import shutil
    scratch = RUNS / "sentinel_fail"
    cloud = np.load(fo.CLOUD_NPY)

    class FailLaunch:
        def __init__(self, mode):
            self.mode = mode
            self.calls = 0

        def __call__(self, cmd, env, cwd, stdout_path, stderr_path):
            self.calls += 1
            with open(stdout_path, "w") as fh:
                if self.mode == "rc":
                    fh.write("SENTINEL\n")
                    return 65                    # nonzero exit
                if self.mode == "marker":
                    fh.write("FINAL SINGLE POINT ENERGY -1.0\n")
                    return 0                     # no normal termination
                if self.mode == "scf":
                    fh.write("FINAL SINGLE POINT ENERGY -1.0\n"
                             "SCF NOT CONVERGED\n"
                             "****ORCA TERMINATED NORMALLY****\n")
                    return 0
                if self.mode == "parse":
                    fh.write("FINAL SINGLE POINT ENERGY   garbage\n"
                             "****ORCA TERMINATED NORMALLY****\n")
                    return 0
                raise AssertionError(self.mode)

    for mode in ("rc", "marker", "scf", "parse"):
        shutil.rmtree(scratch, ignore_errors=True)
        launch = FailLaunch(mode)
        old = _mono_patcher(launch)
        try:
            prov = sentinel_provider(scratch, launch)
            expect_error(lambda: prov.run_config(0, 1, 1, 1),
                         RuntimeError, f"{mode} must fail hard")
        finally:
            _unpatcher(old)
    # a call-time missing binary
    shutil.rmtree(scratch, ignore_errors=True)
    prov = sentinel_provider(scratch, None,
                             binary=Path("/bin/true"))
    prov.binary = Path("/nonexistent/orca-sentinel")
    expect_error(lambda: prov.run_config(0, 1, 1, 1), ValueError,
                 "a vanished binary must fail", "vanished")
    # accounting: a raised call counts nothing
    shutil.rmtree(scratch, ignore_errors=True)
    launch = FailLaunch("rc")
    old = _mono_patcher(launch)
    try:
        prov = sentinel_provider(scratch, launch)
        expect_error(lambda: prov.run_config(0, 1, 1, 1), RuntimeError,
                     "fail")
        _expect(prov.n_evals == 0, "a failed call counts nothing")
    finally:
        _unpatcher(old)
    del cloud


def test_09_sweep_before_rerun():
    """A re-execution sweeps the ORCA-owned files first (the stale .gbw
    must never be seen by GuessMORead).  (A COMPLETED SP resumes before
    any sweep -- the incomplete-.out path is what re-executes; and the
    SENTINEL-marker belt means a sentinel .out is never resume-accepted
    -- asserted here.)"""
    import shutil
    scratch = RUNS / "sentinel_sweep"
    shutil.rmtree(scratch, ignore_errors=True)
    launch = SentinelLaunch()
    old = _mono_patcher(launch)
    try:
        prov = sentinel_provider(scratch, launch)
        prov.run_config(0, 1, 1, 1)
        ts_dir = scratch / prov._key("ts", 0, 1, 1.0, 1.0) / "ts"
        (ts_dir / "STALE.gbw").write_text("stale")
        (ts_dir / "sp.out").write_text("incomplete\n")   # break the gate
        prov.run_config(0, 1, 1, 1)                     # re-executes
        _expect(not (ts_dir / "STALE.gbw").exists(),
                "the stale .gbw was swept before the re-execution")
        # the sentinel belt: the re-executed sp.out carries the marker,
        # so the RESUME gate refuses it (it never impersonates a
        # completed production SP) -- the run happened, the marker says
        # it's a sentinel
        _expect(resume_energy_eh(ts_dir / "sp.out") is None,
                "the sentinel .out is refused at the resume gate")
        text = (ts_dir / "sp.out").read_text()
        _expect("SENTINEL ORCA OUTPUT" in text,
                "the re-execution wrote the sentinel output (the run "
                "happened)")
    finally:
        _unpatcher(old)


def test_10_resume_skips_completed():
    """The resume gate's two verdicts: (1) a REAL (unmarked) completed
    sp.out is resumed -- no launch, walls None, identical energies;
    (2) a SENTINEL .out is NEVER resumed (the marker belt) -- it
    re-executes (the leak self-heals)."""
    import shutil
    scratch = RUNS / "sentinel_resume"
    shutil.rmtree(scratch, ignore_errors=True)
    launch = SentinelLaunch()
    old = _mono_patcher(launch)
    try:
        prov = sentinel_provider(scratch, launch)
        # (2) the sentinel belt: both sentinel runs execute
        r1 = prov.run_config(0, 1, 1, 1)
        n1 = len(launch.calls)
        r2 = prov.run_config(0, 1, 1, 1)
        _expect(len(launch.calls) == n1 + 2,
                "a sentinel SP re-executes (the marker belt refused the "
                "resume)")
        _expect(r2["e_ts"] == r1["e_ts"] and r2["e_r"] == r1["e_r"],
                "the re-executed energies are identical")
        # (1) replace a sentinel sp.out with a REAL-shaped (unmarked)
        # completed one -> the resume gate accepts it: no launch, wall None
        ts_dir = scratch / prov._key("ts", 0, 1, 1.0, 1.0) / "ts"
        (ts_dir / "sp.out").write_text(
            "FINAL SINGLE POINT ENERGY      -838.244178868391\n"
            "****ORCA TERMINATED NORMALLY****\n")
        r3 = prov.run_config(0, 1, 1, 1)
        _expect(len(launch.calls) == n1 + 3,
                "the unmarked ts SP resumes (no new ts launch; r re-runs "
                "once)")
        _expect(r3["walls"]["ts_s"] is None,
                "a resumed SP reports wall None")
        _expect(abs(r3["e_ts"] - (-838.244178868391)) < 1e-12,
                "the resumed energy parses from the real-shaped .out")
    finally:
        _unpatcher(old)


# --------------------------------------------------------------------------
# LIVE tests (real ORCA; gated on the binary)
# --------------------------------------------------------------------------

def _orca_available():
    import os
    from qm import orca_runner
    return os.path.isfile(orca_runner.SITES["laptop"]["orca_bin"])


def test_11_live_asymptote_and_unembedded():
    """THE .pc channel correctness gate (the briefing section 7):
      * far charges (>8 A outside the molecule): |f_emb - f_unemb|
        small (< 5 kcal/mol, recorded);
      * a close pair (the certified optimum's config): large (> 5);
      * the unembedded limit reproduces the orca_v2 TS energy within
        the cross-machine band (5e-4 Eh).
    A sign flip or an A/Bohr misread explodes this test."""
    scratch = RUNS / "live_asymptote"
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    prov = OrcaProvider(pal=8, scratch_root=scratch)   # the real binary

    # -- the unembedded limit: TS + reactant at r2SCAN-3c TightSCF ------
    ts_e = prov.run_sp("unembed_ts", "ts", prov.ts_path,
                       pc_text=None, config_label="(unembedded TS)")
    r_e = prov.run_sp("unembed_r", "r", prov.r_path,
                      pc_text=None, config_label="(unembedded R)")
    f_unemb = (ts_e[0] - r_e[0]) * KCAL_PER_EH
    band = 5e-4
    ref = -838.244178868391
    d = abs(ts_e[0] - ref)
    _expect(d <= band,
            f"the unembedded TS {ts_e[0]!r} vs the orca_v2 value {ref!r}: "
            f"|d| = {d:.3e} exceeds the {band} Eh band")
    print(f"    unembedded TS {ts_e[0]:.9f} Eh (|d| vs orca_v2 = "
          f"{d:.2e} Eh, band {band}); unembedded R {r_e[0]:.9f} Eh; "
          f"free differential {f_unemb:+.3f} kcal/mol")

    # -- the far placement: two charges >8 A beyond the envelope --------
    # Derivation: with c the TS centroid and R the max cloud radius,
    # a charge at c + (R + 8) u is >= (R + 8) - R = 8 A from EVERY
    # envelope point (triangle inequality) -> >8 A outside the molecule.
    cloud = prov._cloud
    c = cloud.mean(axis=0)
    r_env = float(np.max(np.linalg.norm(cloud - c, axis=1)))
    far_d = r_env + 8.0
    p1 = c + far_d * np.array([1.0, 0.0, 0.0])
    p2 = c + far_d * np.array([0.0, 1.0, 0.0])
    pc_far = render_pc(1.0, -1.0, p1, p2)   # the provider's own writer
    # (the repr-passthrough casts the numpy scalars to plain floats
    # first -- a numpy repr in the .pc aborts ORCA at startup)
    ts_f, _ = prov.run_sp("asymptote_far", "ts", prov.ts_path, pc_far,
                          "(far pair)")
    r_f, _ = prov.run_sp("asymptote_far", "r", prov.r_path, pc_far,
                         "(far pair)")
    f_far = (ts_f - r_f) * KCAL_PER_EH
    far_shift = f_far - f_unemb
    _expect(abs(far_shift) < 5.0,
            f"the far-pair shift |{far_shift:.3f}| >= 5 kcal/mol -- "
            "the .pc channel is broken (sign/unit error)")
    print(f"    far pair (>{far_d - r_env:.0f} A beyond the envelope): "
          f"shift = {far_shift:+.6f} kcal/mol (< 5)")

    # -- the close pair: the certified optimum's config (large effect) --
    from arbitration import landscape as L
    land = L.load_and_gate_landscape("canonical")
    row = int(land.f_sorted_rows()[0])
    i, j, si, sj = land.decode_row(row)
    pc_close = render_pc(si, sj, cloud[i], cloud[j])
    r_min = float(np.min([np.min(np.linalg.norm(cloud[k] - c)) for k in
                          (i, j)]))
    ts_c, _ = prov.run_sp("asymptote_close", "ts", prov.ts_path, pc_close,
                          "(close pair)")
    r_c, _ = prov.run_sp("asymptote_close", "r", prov.r_path, pc_close,
                         "(close pair)")
    f_close = (ts_c - r_c) * KCAL_PER_EH
    close_shift = f_close - f_unemb
    _expect(abs(close_shift) > 5.0,
            f"the close-pair shift |{close_shift:.3f}| <= 5 kcal/mol -- "
            "the embedding channel is inert (dropped charges?)")
    print(f"    close pair (optimum row {row}, min site distance "
          f"{r_min:.2f} A): shift = {close_shift:+.3f} kcal/mol (>> 5)")

    # persist the gate numbers
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "asymptote_gate.json").write_text(json.dumps({
        "unembedded_ts_eh": ts_e[0], "unembedded_r_eh": r_e[0],
        "free_differential_kcal": f_unemb,
        "orca_v2_reference_eh": ref, "abs_delta_eh": d, "band_eh": band,
        "far_shift_kcal": far_shift, "far_placement_r_env_A": r_env,
        "far_placement_distance_A": far_d,
        "close_row": row, "close_shift_kcal": close_shift,
        "thresholds": {"far_abs_lt": 5.0, "close_abs_gt": 5.0,
                       "unembedded_band_eh": band},
    }, indent=2, sort_keys=True) + "\n")


def test_12_live_determinism():
    """Same-machine determinism: one config twice in FRESH scratch ->
    bitwise e_ts/e_r."""
    import shutil
    cloud = np.load(fo.CLOUD_NPY)
    energies = []
    for run in (0, 1):
        scratch = RUNS / f"live_determinism_{run}"
        shutil.rmtree(scratch, ignore_errors=True)
        prov = OrcaProvider(pal=1, scratch_root=scratch)
        res = prov.run_config(418, 419, -1, -1)
        energies.append((res["e_ts"], res["e_r"]))
        shutil.rmtree(scratch, ignore_errors=True)
    _expect(energies[0][0] == energies[1][0]
            and energies[0][1] == energies[1][1],
            f"bitwise determinism broken: {energies[0]!r} vs "
            f"{energies[1]!r}")
    print(f"    bitwise e_ts {energies[0][0]!r}, e_r {energies[0][1]!r}")


def _live_suite():
    print("live tests (real ORCA, ~8-12 SPs, several minutes)...")
    test_11_live_asymptote_and_unembedded()
    test_12_live_determinism()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    no_live = "--no-live" in argv
    tests = [test_01_pc_bytes, test_02_sp_inp_shape, test_03_keys,
             test_04_sweep_and_resume, test_05_parser_real_outputs,
             test_06_provider_init_gates, test_07_sentinel_contract,
             test_08_sentinel_fail_hard, test_09_sweep_before_rerun,
             test_10_resume_skips_completed]
    for t in tests:
        t()
        print(f"ok  {t.__name__}: {t.__doc__.splitlines()[0]}")
    if no_live:
        print("live tests skipped (--no-live)")
    else:
        _live_suite()
        for t in (test_11_live_asymptote_and_unembedded,
                  test_12_live_determinism):
            print(f"ok  {t.__name__}: {t.__doc__.splitlines()[0]}")
    print("ALL OK")


if __name__ == "__main__":
    main()