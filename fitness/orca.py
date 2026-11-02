"""r2SCAN-3c embedded-point-charge DFT provider (item 7, arbitration oracle).

The arbitration-tier fitness provider: f(config) = (E(TS;q) - E(R;q)) *
627.5094740631 kcal/mol from two frozen single points at r2SCAN-3c
TightSCF (charge 0, multiplicity 1) --
    TS: data/derived/structures/ts_opt.xyz                          (26 atoms)
    R:  data/derived/structures/chorismate_reactive_TSframe.xyz     (26 atoms)
each with the same two +-1 e external point charges at
candidates[idx_i], candidates[idx_j] (A, TS frame, the locked cloud),
delivered through ORCA's EXTERNAL .pc file channel:
    %pointcharges "<abs path>"
with the .pc file = first line the charge count, then one 'q x y z'
line per charge.  q is written as the literal '+1.0' / '-1.0'; the
positions are the cloud coordinates passed through UNCHANGED.  The
.pc-is-Angstroem convention was pinned empirically in v1 (2026-09-25)
and is RE-verified by the asymptote embedding gate in
arbitration/test_orca.py every time the suite runs.

Unit contract (the item-7 spec's exact wording): eval_batch_full
returns e_ts and e_r as float64 E_h; f is the kcal/mol fitness built
by exactly one multiplication of the Eh difference by the frozen
constant.  (The xtb provider returns kcal components; the ORCA
provider keeps ORCA's native Eh boundary and scales once at the end --
the spec pins this shape.)

Level-of-theory string (the key's lot field and the input's keyword
line sans PAL): ORCA_LOT = "r2SCAN-3c TightSCF".  Baseline resource
per SP: %maxcore 1500.  PAL (OpenMP ranks) and maxcore are EXECUTION
parameters carried on the instance -- they are deliberately NOT part
of the config key (the pricing benchmark tunes them; the physics does
not change).  Whatever the run used is recorded per SP in the records
and the cache.

Config key (scratch + cache identity): sha256 over the exact string
    idx_i=<i>;idx_j=<j>;sign_i=<+1.0|-1.0>;sign_j=<+1.0|-1.0>;
    cloud_sha256=<candidates file sha256>;ts_sha256=<...>;
    r_sha256=<...>;lot=<lot string>
The CLOUD sha is inside the key (the v1 delta: kills cross-cloud
scratch collisions).  The separate P point mode (one SP on
prephenate_TSframe.xyz with the same charges -- step E) keys with the
P geometry sha256 in BOTH the ts_sha256 and r_sha256 slots, keeping
the template's shape.

Init gate (fail hard): the TS/R geometry sha256s must equal the
geometries_sha256 block of
    data/derived/enumeration_v2/enumeration_summary.json
(the certified landscape's own recorded identity), all three geometry
files must load as 26-atom .xyz blocks, and the cloud sha is measured
(keys embed it).  The ORCA binary must exist (site "laptop" by
default -- the subprocess env is built from scratch by
qm.orca_runner.build_env, the v1-proven semantics: private OpenMPI
bin/lib + ORCADIR on PATH / LD_LIBRARY_PATH / ORCADIR; the inherited
PATH / LD_LIBRARY_PATH / ORCADIR are never trusted).

Scratch (per single point):
    scratch/arbitration/<config_key>/{ts,r,p}/
  * before every (re-)execution the directory is swept of ALL
    ORCA-owned files (gbw/tmp/densities/hess/...): ORCA's
    basename-dependent auto-MORead (GuessMORead) must never see a
    stale .gbw -- every execution starts from the same guess.
    Kept: charges.pc, geom.xyz (the frozen .xyz, copied VERBATIM --
    the frozen files carry 14-decimal coordinates; a 6-decimal inline
    re-render would silently degrade them, so the geometry channel is
    the verbatim file via '* XYZFILE 0 1 geom.xyz', exactly like
    fitness/xtb.py's verbatim geom.xyz copy), sp.inp, sp.out, sp.stderr.
  * resume gate per SP: skip iff sp.out exists, terminates normally,
    carries no SCF failure, and a final energy parses (the last
    FINAL SINGLE POINT ENERGY -> float Eh) -- through the landed
    parser qm.orca_out.  A completed SP is never re-executed.
  * NOTHING is ever written under data/.

Fail-hard (no retries, no workarounds, no penalty values): ORCA
nonzero exit, missing normal-termination marker, SCF-not-converged
marker, missing/unparsable final energy, non-finite energy, missing
binary, tampered canonical input -- each raises naming the config
key and the sp.out tail, and the batch halts.

Accounting: every successful differential config counts once into
n_evals (1 row = 2 SPs); one successful P config counts once (1 SP);
a call that raises counts nothing (the contract surface inherits the
contract's per-row accounting on success).

Parallelism: one SP in flight per provider is the driver's contract
(workers launch ORCA subprocesses; the driver's bookkeeping stays in
the orchestrator thread).  _launch_orca is the test seam.

Surfaces:
  * the contract surface (eval_batch / eval_batch_full) -- vectorized,
    for the tests and any future searcher; drops the per-SP walls;
  * the per-config driver surface (run_config / run_p_config) -- what
    the arbitration driver's workers call; returns the energies, the
    kcal fitness and the measured per-SP walls (resumed SPs report
    wall None), which land in the records verbatim.

Constants duplicated by design: KCAL_PER_EH and the .xyz loader
re-state fitness.xtb's values, but fitness.xtb cannot be imported
without instantiating the xtb provider (its module-level binding runs
the xtb binary gate); the ORCA provider must stand alone.  The two
constants are frozen project values; the live tests gate both
providers and would catch any drift.
"""

import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from fitness import contract
from qm import orca_out
from qm import orca_runner

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- frozen constants --------------------------------------------------------
KCAL_PER_EH = 627.5094740631          # the project's frozen conversion
N_ATOMS = 26

# The level-of-theory string: the key's lot field AND the input's keyword
# line (sans PAL).  Charge 0, multiplicity 1 (rendered in the geometry block).
ORCA_LOT = "r2SCAN-3c TightSCF"

# The step-G cross-check level (see the module docstring + PROVIDER_META:
# plain wB97M is UNRECOGNIZED in ORCA 6.1.1 -- measured 2026-10-28 with a
# single-atom probe job; wB97M-V is the spec's section 6.3 prescribed
# fallback and carries VV10 intrinsically, reported as a deviation).
WB97M_LOT = "wB97M-V def2-TZVPP TightSCF"
WB97M_FITNESS_ID = "orca_wb97mv_v2"

# Baseline per-SP memory (%maxcore, MB).  Execution parameter on the
# instance; the pricing benchmark may adjust it -- whatever is used is
# recorded per SP.  Never part of the config key.
BASELINE_MAXCORE_MB = 1500

# Canonical inputs (sha256-gated at init).
TS_XYZ = REPO_ROOT / "data" / "derived" / "structures" / "ts_opt.xyz"
R_XYZ = REPO_ROOT / "data" / "derived" / "structures" / "chorismate_reactive_TSframe.xyz"
P_XYZ = REPO_ROOT / "data" / "derived" / "structures" / "prephenate_TSframe.xyz"
CLOUD_NPY = REPO_ROOT / "data" / "derived" / "candidates_v2.npy"
ENUM_SUMMARY = (REPO_ROOT / "data" / "derived" / "enumeration_v2"
                / "enumeration_summary.json")

# Canonical scratch root (gitignored; tests/pricing may override).
SCRATCH_ROOT = REPO_ROOT / "scratch" / "arbitration"

# The SP scratch sweep keeps EXACTLY these; everything else ORCA leaves
# behind is removed before every (re-)execution.
SP_KEEP = frozenset({"charges.pc", "geom.xyz", "sp.inp", "sp.out",
                     "sp.stderr"})

# The sentinel-artifact belt (the 2026-10-28 incident's hardening): test
# sentinels write .outs carrying this marker; the resume gate and the
# post-run gate REJECT any sp.out that carries it, so a leaked test
# artifact can never impersonate a completed production SP again.
SENTINEL_MARK = "SENTINEL ORCA OUTPUT"


# --------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------

def geo_sha256(path):
    """sha256 of the file bytes (the frozen provenance value)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_xyz(path, n_atoms=N_ATOMS):
    """(elements, float64 coords (n_atoms, 3)) parsed from a frozen .xyz.

    Same parsing rules as fitness.xtb.load_xyz (26-atom header, 4 tokens
    per line, hard fail naming the file) -- re-stated here because
    importing fitness.xtb would instantiate the xtb provider (see the
    module docstring).
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: geometry file not found")
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path}: empty geometry file")
    try:
        n = int(lines[0].strip())
    except ValueError:
        raise ValueError(
            f"{path}: first line is not an atom count: {lines[0]!r}")
    if n != n_atoms:
        raise ValueError(
            f"{path}: expected {n_atoms} atoms, header says {n}")
    block = lines[2:2 + n]
    if len(block) != n:
        raise ValueError(
            f"{path}: expected {n_atoms} coordinate lines, found {len(block)}")
    elements = []
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    for k, line in enumerate(block):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"{path}: line {k + 3} is not '<element> x y z': {line!r}")
        try:
            coords[k] = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            raise ValueError(
                f"{path}: line {k + 3} has non-numeric coordinates: {line!r}")
        elements.append(parts[0])
    return elements, coords


def render_pc(q_i, q_j, pos_i, pos_j):
    """The external .pc file content for two charges -- exact bytes.

    First line = the charge count; then one 'q x y z' line per charge.
    q the literal '+1.0' / '-1.0' (the spec's 'q = +-1.0 exactly'); the
    positions are the cloud coordinates written through UNCHANGED
    (repr of the plain float64 -- byte-identical round trip; numpy
    scalars are cast to plain floats first, exactly like
    fitness.xtb.render_pcem, because a numpy repr (np.float64(...))
    mis-parses in the fortran reader).  The file carries Angstroem --
    pinned empirically v1 2026-09-25 and re-gated by the asymptote test.
    """
    lines = [str(2)]
    for q, p in ((float(q_i), pos_i), (float(q_j), pos_j)):
        x, y, z = (float(v) for v in p)
        lines.append(f"{q:+.1f} {x!r} {y!r} {z!r}")
    return "\n".join(lines) + "\n"


def render_pc_empty():
    """The zero-charge .pc content (the unembedded channel control)."""
    return "0\n"


def render_sp_inp(lot, xyz_name, pc_path, pal, maxcore,
                  charge=0, mult=1):
    """One frozen single point's input text -> exact text.

    Keyword line 'f! {lot} PAL{pal}' (the lot embeds TightSCF per the
    spec), %maxcore, the EXTERNAL point-charge channel when pc_path is
    given (None = unembedded), then the geometry block referencing the
    FROZEN .xyz file VERBATIM ('* XYZFILE <charge> <mult> geom.xyz' --
    the fitness/xtb.py house pattern: the frozen file's bytes are the
    geometry, zero re-derivation; the frozen files carry 14-decimal
    coordinates that a 6-decimal inline re-render would silently
    degrade).  geom.xyz sits in the SP's scratch dir (copy the frozen
    file there verbatim before launching; it is sweep-kept).

    PAL: rendered for pal >= 2 (the simple PALn keyword; parallel runs
    REQUIRE the orca binary to be invoked by its FULL pathname -- the
    provider always does).  pal 1 renders NO PAL keyword: 'PAL1' is
    UNRECOGNIZED in ORCA 6.1.1 (measured 2026-10-28, single-atom
    probe) and the keyword-free input is exactly the one-rank run.
    """
    lines = [f"! {lot} PAL{int(pal)}" if int(pal) >= 2 else f"! {lot}",
             f"%maxcore {int(maxcore)}"]
    if pc_path is not None:
        lines.append(f'%pointcharges "{pc_path}"')
    lines.append(f"* XYZFILE {charge} {mult} {xyz_name}")
    return "\n".join(lines) + "\n"


def config_key(idx_i, idx_j, sign_i, sign_j, cloud_sha, ts_sha, r_sha, lot):
    """The deterministic config key (see the module docstring's template).

    Signs are rendered '+1.0' / '-1.0'.  The P point mode passes the P
    geometry sha as BOTH ts_sha and r_sha (the template's shape kept).
    """
    raw = (f"idx_i={int(idx_i)};idx_j={int(idx_j)};"
           f"sign_i={float(sign_i):+.1f};sign_j={float(sign_j):+.1f};"
           f"cloud_sha256={cloud_sha};ts_sha256={ts_sha};"
           f"r_sha256={r_sha};lot={lot}")
    return hashlib.sha256(raw.encode()).hexdigest()


def differential_key(idx_i, idx_j, sign_i, sign_j, cloud_sha, ts_sha,
                     r_sha, lot=ORCA_LOT):
    """The differential (TS/R) config key for the level of theory."""
    return config_key(idx_i, idx_j, sign_i, sign_j, cloud_sha, ts_sha,
                      r_sha, lot)


def p_key(idx_i, idx_j, sign_i, sign_j, cloud_sha, p_sha, lot=ORCA_LOT):
    """The P point-mode key: the P geometry sha in BOTH geometry slots."""
    return config_key(idx_i, idx_j, sign_i, sign_j, cloud_sha, p_sha,
                      p_sha, lot)


def sweep_sp_dir(dir_path, keep=SP_KEEP):
    """Remove every file in dir_path NOT in keep (the ORCA-owned sweep).

    ORCA's basename-dependent auto-MORead (GuessMORead) would pick up a
    stale .gbw as the initial guess -- every execution must start from
    the same guess, so only the .pc/.in/.out/.stderr set survives.
    Returns the removed names.
    """
    dir_path = Path(dir_path)
    removed = []
    if dir_path.is_dir():
        for entry in sorted(dir_path.iterdir()):
            if entry.name not in keep:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed.append(entry.name)
    return removed


def resume_energy_eh(out_path):
    """The completed SP's energy (Eh) iff the resume gate passes, else None.

    Resume gate (the briefing's exact semantics, through the landed
    parser): sp.out exists AND terminates normally AND carries no
    'SCF NOT CONVERGED' AND a final energy parses (last FINAL SINGLE
    POINT ENERGY -> float).  A .out carrying the SENTINEL ORCA OUTPUT
    marker is NEVER accepted (a leaked test artifact; the SP
    re-executes).  Anything else -> None (re-run).
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return None
    try:
        text = out_path.read_text()
        if SENTINEL_MARK in text:
            return None   # a leaked test artifact: never a completed SP
        orca_out.require_normal_termination(out_path)
        orca_out.require_scf_converged(out_path)
        return orca_out.final_energy_eh(out_path)
    except ValueError:
        return None


def _launch_orca(cmd, env, cwd, stdout_path, stderr_path):
    """Launch exactly one ORCA subprocess; return its exit code.

    Test seam: the sentinel tests patch this (input rendering, the
    sweep, the resume gate and parsing all stay real; no ORCA binary
    is needed for the sentinel tests).
    """
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        return subprocess.run(cmd, env=env, cwd=str(cwd),
                              stdout=out, stderr=err).returncode


def _tail(path, n=15):
    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no output file)"


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

class OrcaProvider(contract.FitnessProvider):
    """r2SCAN-3c embedded-point-charge provider (item 7 spec).

    Init hard-gates the canonical TS/R geometries (sha256 against the
    certified landscape summary's geometries_sha256 block), the 26-atom
    loads of all three geometries, and the ORCA binary, failing hard and
    naming the offending file.  pal / maxcore are execution parameters
    rendered into the inputs and recorded in the artifacts --
    deliberately not part of the config key (the pricing benchmark tunes
    them).

    A second instance with lot=... + fitness_id=... serves the step-G
    cross-check (wB97M-V def2-TZVPP; see the module docstring for the
    measured plain-wB97M absence and the deviation report).
    """

    FITNESS_ID = "orca_r2scan3c_v2"

    PROVIDER_META = {
        "provider": "orca",
        "fitness_id": FITNESS_ID,
        "description": (
            "r2SCAN-3c embedding arbitration oracle: f = (E(TS;q) - "
            "E(R;q)) * 627.5094740631 kcal/mol, two frozen single points "
            "with two +-1 e external point charges (ORCA .pc channel) at "
            "the candidate sites"),
        "units": "kcal/mol",
        "components_units": "Eh (float64, the ORCA-native boundary)",
        "kcal_per_eh": KCAL_PER_EH,
        "lot": ORCA_LOT,
        "charge_mult": "charge 0, mult 1",
        "baseline_maxcore_mb": BASELINE_MAXCORE_MB,
        "pal": "execution parameter on the instance (not in the key)",
        "mode": "cli",
        "modes_available": ["cli"],
        "pc_channel": {
            "directive": '%pointcharges "<abs path to .pc file>"',
            "file": ("first line = charge count, then one 'q x y z' line "
                     "per charge; q written as the literal +1.0 / -1.0"),
            "units": ("angstrom, positions passed through unchanged "
                      "(pinned empirically v1 2026-09-25; re-verified by "
                      "the asymptote embedding gate in the test suite)"),
        },
        "geometries": {
            "ts": "data/derived/structures/ts_opt.xyz",
            "reactant": "data/derived/structures/chorismate_reactive_TSframe.xyz",
            "p": "data/derived/structures/prephenate_TSframe.xyz",
        },
        "init_gate": ("ts/reactant sha256s must equal the "
                      "geometries_sha256 block of "
                      "data/derived/enumeration_v2/enumeration_summary.json"),
        "cloud": {
            "path": "data/derived/candidates_v2.npy",
            "sha256_in_key": True,
            "note": ("the key carries the cloud sha (v1 delta: kills "
                     "cross-cloud scratch collisions)"),
        },
        "scratch": {
            "layout": "scratch/arbitration/<config_key>/{ts,r,p}/",
            "sweep": ("every ORCA-owned file removed before each "
                      "(re-)execution (GuessMORead must never see a stale "
                      ".gbw); kept: charges.pc, geom.xyz, sp.inp, sp.out, sp.stderr"),
        },
        "resume": ("per SP: skip iff sp.out exists, terminates normally, "
                   "no SCF failure, final energy parses (qm.orca_out); "
                   "a completed SP is never re-executed"),
        "fail_hard": ("ORCA nonzero exit / no normal-termination marker / "
                      "SCF failure / unparsable or non-finite energy / "
                      "missing binary / tampered input: raise naming the "
                      "config key + the sp.out tail; no retries, no "
                      "penalty values"),
        "accounting": ("every successful differential config counts once "
                       "in n_evals (2 SPs/row); one successful P config "
                       "counts once (1 SP); failed calls count nothing"),
        "step_g_lot": ("plain wB97M is UNRECOGNIZED in ORCA 6.1.1 "
                       "(measured 2026-10-28, single-atom probe: 'INPUT "
                       "ERROR UNRECOGNIZED OR DUPLICATED KEYWORD(S)... "
                       "WB97M'); wB97M-V runs (VV10 intrinsic) -- the "
                       "item-7 spec section 6.3 prescribes exactly this "
                       "fallback, reported as a deviation"),
    }

    def __init__(self, ts_path=None, r_path=None, p_path=None,
                 cloud_path=None, cloud_sha256=None, scratch_root=None,
                 binary=None, site="laptop", pal=1,
                 maxcore=BASELINE_MAXCORE_MB, lot=ORCA_LOT, fitness_id=None):
        super().__init__()
        self.ts_path = Path(ts_path) if ts_path is not None else TS_XYZ
        self.r_path = Path(r_path) if r_path is not None else R_XYZ
        self.p_path = Path(p_path) if p_path is not None else P_XYZ
        self.cloud_path = (Path(cloud_path) if cloud_path is not None
                           else CLOUD_NPY)
        self.cloud_sha = (str(cloud_sha256) if cloud_sha256 is not None
                          else geo_sha256(self.cloud_path))
        self.scratch_root = (Path(scratch_root) if scratch_root is not None
                             else SCRATCH_ROOT)
        self.site = site
        self.pal = int(pal)
        self.maxcore = int(maxcore)
        self.lot = str(lot)
        if fitness_id is not None:
            self.FITNESS_ID = str(fitness_id)
        self._accounting_lock = threading.Lock()

        # Site gate (unconditional): the site must be known, listing the
        # known sites; the binary is the explicit path (the test seam) or
        # the site profile's orca binary, which must exist.
        if site not in orca_runner.SITES:
            raise ValueError(
                f"unknown ORCA site {site!r} "
                f"(known: {', '.join(sorted(orca_runner.SITES))})")
        if binary is not None:
            self.binary = Path(binary)
        else:
            self.binary = Path(orca_runner.SITES[site]["orca_bin"])
        if not self.binary.is_file():
            raise ValueError(
                f"{self.binary}: the ORCA binary is not installed (site "
                f"{self.site!r}); see the module docstring's dependency note")

        # Geometry gate: the TS/R sha256s must equal the certified
        # landscape summary's geometries_sha256 block (fail hard naming
        # the file); all three geometries load as 26-atom blocks; the P
        # geometry's sha is recorded (it keys the P point mode).
        if not ENUM_SUMMARY.is_file():
            raise ValueError(f"{ENUM_SUMMARY}: the certified landscape "
                             "summary is not present (init gate)")
        block = json.loads(ENUM_SUMMARY.read_text())["geometries_sha256"]
        for path, want, what in ((self.ts_path, block["ts"], "TS geometry"),
                                 (self.r_path, block["reactant"],
                                  "reactant geometry")):
            if not path.is_file():
                raise ValueError(f"{path}: canonical {what} not found")
            got = geo_sha256(path)
            if got != want:
                raise ValueError(
                    f"{path}: sha256 mismatch ({what}; got {got}, "
                    f"expected {want} -- tampered or wrong file?)")
        for path in (self.ts_path, self.r_path, self.p_path):
            load_xyz(path)   # 26-atom gate, naming the file
        self.ts_sha = geo_sha256(self.ts_path)
        self.r_sha = geo_sha256(self.r_path)
        self.p_sha = geo_sha256(self.p_path)
        self._ts_elements, self._ts_coords = load_xyz(self.ts_path)
        self._r_elements, self._r_coords = load_xyz(self.r_path)
        self._p_elements, self._p_coords = load_xyz(self.p_path)
        self._cloud = np.load(self.cloud_path)   # sha already measured

    # -- internal: one single point in its own swept/re-gated dir ----------

    def _key(self, mode, idx_i, idx_j, sign_i, sign_j):
        """The config key: the differential key for the TS/R mode, the
        P-mode key for the 'p' point mode."""
        if mode == "p":
            return p_key(idx_i, idx_j, sign_i, sign_j, self.cloud_sha,
                         self.p_sha, self.lot)
        return differential_key(idx_i, idx_j, sign_i, sign_j,
                                self.cloud_sha, self.ts_sha, self.r_sha,
                                self.lot)

    def _sp_dir(self, key, point):
        """The SP's scratch dir: <scratch_root>/<config_key>/<point>."""
        return self.scratch_root / key / point

    def run_sp(self, key, point, xyz_src_path, pc_text=None,
               config_label=""):
        """One fresh-or-resumed single point -> (energy_eh float64,
        wall_s or None).

        xyz_src_path: the FROZEN geometry .xyz, copied VERBATIM into the
        scratch dir as geom.xyz (byte-identical; the input references it
        via * XYZFILE).

        pc_text None (or a string) -- a string is written to charges.pc
        and referenced by the input's %pointcharges directive (None =
        unembedded: no directive, no file).

        Sweeps the scratch dir of everything ORCA owns (keep set),
        writes geom.xyz + the input, applies the resume gate (a
        completed SP is never re-executed), launches ONE ORCA process
        with the site env, and fails hard (naming the config key + the
        sp.out tail) on any ORCA/plumbing failure.  No retries.
        """
        d = self._sp_dir(key, point)
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "sp.out"
        inp_path = d / "sp.inp"
        err_path = d / "sp.stderr"
        pc_path = d / "charges.pc"
        geom_path = d / "geom.xyz"

        resume = resume_energy_eh(out_path)
        if resume is not None:
            return float(resume), None   # never re-execute a completed SP

        sweep_sp_dir(d)
        shutil.copyfile(xyz_src_path, geom_path)   # verbatim bytes
        if pc_text is not None:
            pc_path.write_text(pc_text)
        inp_path.write_text(render_sp_inp(
            self.lot, "geom.xyz",
            str(pc_path) if pc_text is not None else None, self.pal,
            self.maxcore))

        if not self.binary.is_file():
            raise ValueError(
                f"{self.binary}: the ORCA binary vanished before the "
                f"{point} SP for config {config_label} (key {key})")
        env = orca_runner.build_env(self.site)
        t0 = time.monotonic()
        rc = _launch_orca([str(self.binary), str(inp_path)], env, d,
                          out_path, err_path)
        wall = time.monotonic() - t0
        if rc != 0:
            raise RuntimeError(
                f"ORCA exited {rc} for the {point} SP of config "
                f"{config_label} (key {key}); tail: {out_path}\n"
                + _tail(out_path))
        # A zero exit is not sufficient: normal termination + no SCF
        # failure + a parsable, finite final energy (the landed parser).
        # (A leaked SENTINEL artifact is refused at the RESUME gate --
        # resume_energy_eh -- so it can never impersonate a completed
        # SP; the leak then self-heals by re-execution.)
        try:
            orca_out.require_normal_termination(out_path)
            orca_out.require_scf_converged(out_path)
            energy = float(orca_out.final_energy_eh(out_path))
        except ValueError as exc:
            raise RuntimeError(
                f"the {point} SP for config {config_label} (key {key}) "
                f"did not meet the post-run gate ({exc}); tail: "
                f"{out_path}\n" + _tail(out_path))
        if not np.isfinite(energy):
            raise RuntimeError(
                f"the {point} SP for config {config_label} (key {key}) "
                f"parsed a non-finite energy; tail: {out_path}\n"
                + _tail(out_path))
        sweep_sp_dir(d)   # the keep-set invariant at rest (disk volume:
                          # ~80 kB per completed SP dir)
        return energy, wall

    def _validate_scalar_config(self, idx_i, idx_j, sign_i, sign_j):
        """Fail hard on non-int indices (outside the cloud) or signs
        outside {+1, -1} (the per-config mirror of the contract gate)."""
        n_sites = self._cloud.shape[0]
        vals = []
        for name, v in (("idx_i", idx_i), ("idx_j", idx_j),
                        ("sign_i", sign_i), ("sign_j", sign_j)):
            if int(v) != v:
                raise ValueError(f"{name} must be an integer, got {v!r}")
            vals.append(int(v))
        i, j, si, sj = vals
        for name, v in (("idx_i", i), ("idx_j", j)):
            if not 0 <= v < n_sites:
                raise ValueError(
                    f"{name} must be in [0, {n_sites}) (rows of the "
                    f"cloud), got {v}")
        for name, v in (("sign_i", si), ("sign_j", sj)):
            if v not in (1, -1):
                raise ValueError(f"{name} must be +1 or -1, got {v}")
        return i, j, si, sj

    def _bump(self, n):
        with self._accounting_lock:
            self.n_evals += n

    # -- the per-config driver surface --------------------------------------

    def _run_config_body(self, i, j, si, sj):
        """The differential evaluation (no accounting): the SPs + the
        record-grade dict."""
        key = self._key("ts", i, j, si, sj)
        pc = render_pc(si, sj, self._cloud[i], self._cloud[j])
        label = (f"(idx_i={i}, idx_j={j}, sign_i={si:+.1f}, "
                 f"sign_j={sj:+.1f}, lot={self.lot!r})")
        e_ts, w_ts = self.run_sp(key, "ts", self.ts_path, pc, label)
        e_r, w_r = self.run_sp(key, "r", self.r_path, pc, label)
        return {
            "key": key,
            "config": {"idx_i": i, "idx_j": j,
                       "sign_i": float(si), "sign_j": float(sj)},
            "e_ts": e_ts,
            "e_r": e_r,
            "f": (e_ts - e_r) * KCAL_PER_EH,
            "walls": {"ts_s": w_ts, "r_s": w_r},
            "lot": self.lot,
            "pal": self.pal,
            "maxcore": self.maxcore,
        }

    def run_config(self, idx_i, idx_j, sign_i, sign_j):
        """One differential config -> dict {key, config, e_ts, e_r, f,
        walls, lot, pal, maxcore}.

        e_ts / e_r float64 Eh; f the kcal/mol fitness from exactly one
        multiplication of the difference.  walls = {'ts_s': .., 'r_s':
        ..} with None for a resumed (never re-executed) SP.  Point
        order: e_ts then e_r.  Accounting: counts 1 on success.
        """
        i, j, si, sj = self._validate_scalar_config(idx_i, idx_j,
                                                    sign_i, sign_j)
        res = self._run_config_body(i, j, si, sj)
        self._bump(1)
        return res

    def run_p_config(self, idx_i, idx_j, sign_i, sign_j):
        """One P point-mode SP (the P geometry, the same charges) ->
        dict {key, config, e_p, wall, lot, pal, maxcore}.  The key uses
        the P geometry sha in BOTH geometry slots."""
        i, j, si, sj = self._validate_scalar_config(idx_i, idx_j,
                                                    sign_i, sign_j)
        key = self._key("p", i, j, si, sj)
        pc = render_pc(si, sj, self._cloud[i], self._cloud[j])
        label = (f"(idx_i={i}, idx_j={j}, sign_i={si:+.1f}, "
                 f"sign_j={sj:+.1f}, point=P, lot={self.lot!r})")
        e_p, w_p = self.run_sp(key, "p", self.p_path, pc, label)
        self._bump(1)
        return {
            "key": key,
            "config": {"idx_i": i, "idx_j": j,
                       "sign_i": float(si), "sign_j": float(sj)},
            "e_p": e_p,
            "wall_s": w_p,
            "lot": self.lot,
            "pal": self.pal,
            "maxcore": self.maxcore,
        }

    # -- contract implementation -------------------------------------------

    def _eval_batch_full_impl(self, idx_i, idx_j, sign_i, sign_j, candidates):
        """f (kcal/mol), e_ts (Eh), e_r (Eh) -- the item-7 unit contract.

        The passed cloud must be bitwise the gated canonical cloud (the
        frozen geometries' frame is what the cloud coordinates live in).
        The vectorized surface drops the per-SP walls (the driver
        consumes run_config for the record-grade artefacts).
        """
        if not (candidates.shape == self._cloud.shape
                and np.array_equal(candidates, self._cloud)):
            raise ValueError(
                f"candidates is not the locked candidate cloud "
                f"({self.cloud_path}): the orca provider embeds only the "
                "frozen TS-frame cloud; its coordinates are the charge "
                "positions and must match bitwise")
        m = idx_i.size
        f = np.empty(m, dtype=np.float64)
        e_ts = np.empty(m, dtype=np.float64)
        e_r = np.empty(m, dtype=np.float64)
        for k in range(m):
            res = self._run_config_body(int(idx_i[k]), int(idx_j[k]),
                                        int(sign_i[k]), int(sign_j[k]))
            e_ts[k] = res["e_ts"]
            e_r[k] = res["e_r"]
            f[k] = res["f"]
        return f, e_ts, e_r


# Contract surface: the four names + n_evals.
#
# The provider is bound LAZILY (PEP 562). Instantiating it runs the init
# gates, and one of those checks that the ORCA binary is installed -- so an
# eager binding made the whole package refuse to import on a machine without
# ORCA, including the parts that never call it (the landscape gates, the
# probe sampler, the step logic, and their tests). The first attribute access
# below still constructs the provider and still fails hard if ORCA is
# missing; nothing is silently deferred past the point of use.
_provider = None


def _provider_instance():
    global _provider
    if _provider is None:
        _provider = OrcaProvider()
    return _provider

# The two static contract names are class attributes, so they bind without
# constructing anything (and so without requiring the binary):
FITNESS_ID = OrcaProvider.FITNESS_ID      # noqa: E305
PROVIDER_META = OrcaProvider.PROVIDER_META

_CONTRACT = ("provider", "eval_batch_full", "eval_batch")


def __getattr__(name):
    if name in _CONTRACT:
        prov = _provider_instance()
        return prov if name == "provider" else getattr(prov, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
