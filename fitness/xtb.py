"""GFN2-xTB embedded-point-charge fitness provider (v2).

The search-tier oracle provider: f(config) = E(TS;q) - E(R;q) in
kcal/mol from two frozen single points --
    TS: data/derived/structures/ts_opt.xyz                    (native frame)
    R:  data/derived/structures/chorismate_reactive_TSframe.xyz  (the
        registered gripped reactant -- same 26-atom chorismate order)
each with the same two +-1 e point charges embedded at
candidates[idx_i], candidates[idx_j] (A, TS frame, the locked cloud).

f is built from the two embedded energies by exactly one subtraction
(e_ts - e_r, float64) -- the contract identity holds bitwise by
construction. Conversion: 1 Eh = 627.5094740631 kcal/mol (the project's
frozen constant, as in the v1 artifact).

Implementation mode: the xtb CLI (a fresh process per single point).
Architecture finding recorded 2026-10-08 -- the in-process channel is
NOT available for this embedding:
  * tblite (PyPI 0.7.0, cp313 manylinux wheel) installs and imports,
    but the tblite library has NO external point-charge API at all (no
    external-charge entry point anywhere in tblite up to current main;
    confirmed against the shipped wheel's exported symbols). xtb's
    --tblite flag silently ignores the $embedding control block (the
    run then evaluates the FREE energy; measured on the units-gate R=20
    placement: the expected 5.9e-4 Eh embedding shift is missing, the
    run differs from the free CLI energy only by the 2.9e-8 Eh
    implementation noise, with no warning).
  * the PyPI `xtb` C-API python bindings ship no cp313 wheels (up to
    cp311 at latest 22.1); the 6.7.1 release ships no capi build.
The benchmark's in-process column needs an explicitly granted third
channel (Jonny's call); provider init fails hard on any mode other
than "cli", carrying this finding.

Embedding conventions (EMPIRICALLY GATED 2026-10-07, see
research_docs/xtb_benchmark.md + runs/xtb_pcem_units/):

  * pcem_units = "angstrom": the charge file carries the candidate
    positions VERBATIM (byte-identical repr passthrough, zero unit
    conversion). Verified: the $embedding interface=orca reading of the
    file matches the closed-form dipole asymptote
        dE = q (mu . n-hat) / r_bohr^2
    (converged neutral density about the coordinate origin) at physical
    R in {15, 20, 30, 40} A to -14%/+2% with the residuals decaying
    ~1/R^3 as the neglected higher multipoles do; a file-in-Bohr
    misreading is off by exactly (1/0.52917726)^2 = 3.5714x and is
    ruled out. The legacy interface (Bohr) was cross-checked on the
    same physical placements and agrees BITWISE with the orca reading;
    an anti-parallel charge flips the sign as the dipole term does.
    Source cross-check: src/embedding.f90 v6.7.1 sets conv = 1.0 (Bohr)
    for the legacy interface and conv = aatoau for the orca interface
    (aatoau = 1/0.52917726, src/mctc/convert.f90 v6.7.1).
  * PCEM_HARDNESS = 99: the pcem 5th column (chemical hardness eta,
    entering the GFN2 J-potential smear 1/sqrt(r^2 + 1/xj^2) with
    xj = (gi + gj)/2, src/embedding.f90 v6.7.1) is fixed at 99 -- the
    bare-charge limit (agrees with gam = 1e6 to 1.5e-3 kcal/mol; the
    defaulted dummy-atom gam = eta(N) differs by 12.25 kcal/mol on the
    canonical O1 pair). Verified materially; encoded as this constant.
  * no PC-PC term: E(q1,q2) - E(q1) - E(q2) + E_free = 0.0118 kcal/mol
    at far separations -- the embedded energy is the QC-PC interaction
    only; the two charges do not interact through E.
  * ORCA convention for the DFT cross-check .pc file (first line =
    count, then "q x y z" lines) is the v1-verified Angstroem reading
    (v1 bf_fitness_orca.py + its test, 2026-09-25) -- i.e. the same
    physical convention this provider pins through interface=orca.

Determinism: every configuration is an independent SCF start. CLI mode
launches a fresh process per single point (no SCF state can carry);
the scratch dir per configuration is wiped before each run and the
wavefunction restart is disabled (--norestart). The subprocess env pins
OMP / MKL / OPENBLAS thread counts to 1.

Fail-hard: a nonzero exit, a missing normal-termination marker, a
missing/unparsable TOTAL ENERGY line, a non-finite energy, an
implausible energy (|E - E_free| > PLAUSIBILITY_MAX_EH), an unconverged
SCF marker, or a missing 'PC potential true' plumbing flag raises,
naming the configuration and the stage. No penalty values, no
fallbacks, no silent re-runs.

Accounting: inherited from contract.FitnessProvider (every successful
row counts into n_evals; a call that raises counts nothing;
contract.py is untouched). Each evaluated row performs TWO single
points (TS + reactant).

Plausibility references: XTB_E_FREE_TS_EH / XTB_E_FREE_R_EH below are
the free (no-charge) GFN2-xTB energies of the two frozen geometries,
measured 2026-10-08 with the pinned invocation; they feed ONLY the
implausibility gate, never f. The live smoke re-measures them and
requires bitwise agreement (caught: version/flag/environment drift).

Inputs (canonical, read-only; sha256-gated at init against the
briefing-frozen values):
    data/derived/candidates_v2.npy               (locked cloud, A, TS frame)
    data/derived/structures/ts_opt.xyz           (26 atoms, chorismate order)
    data/derived/structures/chorismate_reactive_TSframe.xyz
At every evaluation the passed candidates array must be BITWISE the
gated cloud (the frozen geometries' frame is what the cloud
coordinates live in; anything else would silently distort every charge
position). The gated cloud is the canonical locked one by default
(an explicit cloud_sha256=None means the canonical identity); a
REPLICATE cloud is only accepted EXPLICITLY (path + its own sha256 --
the work-item-4b extension: defaults unchanged, a call without explicit
cloud identity keeps the canonical gate).

Scratch: runs/xtb_ff/<config_key>/{ts,reactant}/ -- one dir per
configuration, keyed on the ordered config tuple + the frozen artifact
hashes + the provider identity; wiped before every run (a
re-evaluation re-produces; stale state is never read -- and could not
be, with --norestart and a fresh process). NOTHING is ever written
under data/derived/.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from fitness import contract

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- frozen constants --------------------------------------------------------
KCAL_PER_EH = 627.5094740631

# The pcem chemical-hardness column: 99 = bare-charge behaviour (see the
# module docstring's empirical gate). Written as the literal 5th column.
PCEM_HARDNESS = 99

# The pinned pcem coordinate convention (see the module docstring's gate):
# the ORCA interface of the $embedding control group reads the file in
# Angstroem -- candidate positions are written through with NO conversion.
PCEM_UNITS = "angstrom"
PCEM_INTERFACE = "orca"

# The frozen SCC protocol (Jonny's recorded decision, 2026-10-08): the
# scc-domain diagnostic showed the default (maxiter 250, broydamp 0.4)
# hard-fails the extreme-dipole reactant-side single points (exit 128
# at the 250-iteration ceiling, the iterator still sloshing at ~0.1 Eh
# per-iterate) and that the RESCUE is the damping (maxiter-2000 alone
# burned 12 min 55 s and still failed). T2 (maxiter 2000 + broydamp
# 0.1) converges the named failure in 37 iterations, costs ~8% more on
# the convergent domain and shifts energies <= 2.1e-5 kcal/mol per
# side vs the old protocol (0/30 bitwise - the same fixed point, the
# path noise is the difference). T3 (broydamp 0.05) measured equivalent
# to 6e-6 kcal/mol but is NOT adopted (the choice sits inside path
# noise; T2 is the named option). The protocol is part of the model:
# constant bytes for EVERY evaluation, never per-config conditional.
SCC_MAX_ITERATIONS = 2000
SCC_BROYDAMP = 0.1
SCC_PROTOCOL_ADOPTED = "2026-10-08"
SCC_PROTOCOL_SOURCE = (
    "the scc-domain diagnostic (research_docs/scc_domain.md + "
    "runs/scc_domain/); T2 = rescue-by-damping; T3-equivalence 6e-6 "
    "kcal/mol recorded but not adopted")

# Plausibility gate: an embedded single point must stay within this of its
# pinned free reference (0.5 Eh ~ 314 kcal/mol). Re-derived 2026-10-08
# against the measured extreme sample (the 5 closest kept pairs x 4
# signs under T2: max |shift| = 0.2778 Eh = 174.3 kcal/mol at
# (150, 151, -1, -1) -- headroom 1.80x). Corruption catch only.
PLAUSIBILITY_MAX_EH = 0.5

# Free GFN2-xTB energies of the two frozen geometries re-measured
# 2026-10-08 under the T2 protocol (no embedding): -50.809501549324
# (TS) and -50.850132760091 (reactant, registered into the TS frame);
# the T2-vs-default shifts (4.6e-7 / 4.3e-7 kcal/mol) are the path
# noise's own scale.
XTB_E_FREE_TS_EH = -50.809501549324
XTB_E_FREE_R_EH = -50.850132760091

# Canonical input paths (init arguments; gated against the frozen sha256s).
TS_XYZ = REPO_ROOT / "data" / "derived" / "structures" / "ts_opt.xyz"
R_XYZ = REPO_ROOT / "data" / "derived" / "structures" / "chorismate_reactive_TSframe.xyz"
CLOUD_NPY = REPO_ROOT / "data" / "derived" / "candidates_v2.npy"

_CLOUD_SHA256 = "084e48a6b5267900a995ba2d4f0412f77f5bb13446ba74eafaf474c644349e71"
_TS_SHA256 = "f9d00975f03d2d27b2ce35ab8d5789790ad136c75da5549fa761b94c0eff76b3"
_R_SHA256 = "6eb2a27fcbafb963394e2f17f0603713eeda190e96fa53b6b19cf273c23929dd"
N_ATOMS = 26

# Canonical binary location (the unpacked official release tarball).
XTB_BIN = REPO_ROOT / "bin" / "xtb-6.7.1" / "bin" / "xtb"
XTB_VERSION = "6.7.1"
XTB_TARBALL_SHA256 = (
    "62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df")

# Scratch root working area (gitignored; tests/benchmark may override).
SCRATCH_ROOT = REPO_ROOT / "runs" / "xtb_ff"

# Deterministic subprocess env: 1 thread everywhere (every evaluation is
# canonical single-thread; a benchmark may add a labelled multithread
# column separately).
PINNED_THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_PCFLAG_RE = re.compile(r"PC potential\s+true", re.S)

# The SCC non-convergence signature (the DOMAIN class): a nonzero exit
# whose output carries the iterator's ceiling marker / the driver's SCC
# error line means the configuration lies OUTSIDE the fitness's domain
# under the frozen protocol - a recorded exclusion, never a penalty,
# never retried. EVERY OTHER failure (a missing marker, a parse error,
# a plausibility breach, a missing PC flag, a non-finite value) is a
# plumbing error and fails hard (full stop).
_SCC_MARKER_1 = "Self consistent charge iterator did not converge"
_SCC_MARKER_2 = "convergence criteria cannot be satisfied"


class SccDomainExclusion(RuntimeError):
    """A configuration whose side failed the SCC under the frozen T2
    protocol: it is outside the fitness's domain. Carries the config,
    the stage, the provider-level label, the marker evidence and the
    raw-output directory (kept, never swept - the aggregation's
    evidence check re-parses it). The worker records it into the
    shard's domain_exclusions ledger and CONTINUES (no retry, no
    penalty, no stop).
    """

    def __init__(self, message, *, stage=None, config_label=None,
                 config=None, raw_dir=None):
        super().__init__(message)
        self.stage = stage
        self.config_label = config_label
        self.config = list(config) if config is not None else None
        self.raw_dir = str(raw_dir) if raw_dir is not None else None


# --------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------

def geo_sha256(path):
    """sha256 of the file bytes (the frozen provenance value)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_xyz(path, n_atoms=N_ATOMS):
    """(elements, float64 coords (n_atoms, 3)) parsed from a frozen .xyz.

    Fails hard naming the file on any malformation (wrong/missing header
    count, short body, bad tokens).
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


def render_pcem(q_i, q_j, pos_i, pos_j):
    """Point-charge file content (2 charges), the provider's exact bytes.

    First line = charge count, then one "q x y z hardness" line per
    charge with the literal PCEM_HARDNESS column. Positions pass through
    under PCEM_UNITS ("angstrom", verified empirically -- see the module
    docstring): every scalar is repr of the plain float64 (byte-identical
    round trip). np.float64 scalars are cast to plain floats FIRST: a
    numpy repr emits 'np.float64(...)', which the fortran list-directed
    reader silently mis-parses into garbage positions -- the unit-gate
    experiments hit exactly this (recorded in the report).
    """
    lines = [str(2)]
    for q, p in ((float(q_i), pos_i), (float(q_j), pos_j)):
        x, y, z = (float(v) for v in p)
        lines.append(f"{q:+.1f} {x!r} {y!r} {z!r} {int(PCEM_HARDNESS)}")
    return "\n".join(lines) + "\n"


def render_control():
    """The control block: the $embedding group pinning the ORCA pcem
    interface + the frozen global SCC protocol (the T2 decision:
    constant bytes for EVERY evaluation - part of the model)."""
    return (f"$embedding\n   interface={PCEM_INTERFACE}\n"
            f"   input=pcem.in\n$end\n"
            f"$scc\n   maxiterations={SCC_MAX_ITERATIONS}\n"
            f"   broydamp={SCC_BROYDAMP}\n$end\n")


def config_key(idx_i, idx_j, sign_i, sign_j, ts_sha, r_sha):
    """Scratch key of one configuration (deterministic, collision-free)."""
    raw = (f"{int(idx_i)}:{int(idx_j)}:{int(sign_i)}:{int(sign_j)}"
           f"|{ts_sha[:16]}:{r_sha[:16]}|xtb:{XTB_VERSION}"
           f"|pcem:{PCEM_UNITS}:{PCEM_HARDNESS}:{KCAL_PER_EH!r}")
    return hashlib.sha256(raw.encode()).hexdigest()


def parse_xtb_energy(text):
    """-> energy_eh parsed from one xtb --sp output (stdout+stderr text).

    Fails hard on: a missing 'normal termination of xtb' banner (it
    prints on stderr in 6.7.1), a missing/unparsable TOTAL ENERGY line,
    or a non-finite energy. The unconverged-SCF marker is checked by the
    caller (it must fail hard too, even should xtb ever exit zero).
    """
    if "normal termination of xtb" not in text:
        raise RuntimeError("xtb did not terminate normally "
                           "(no 'normal termination' marker)")
    m = _ENERGY_RE.search(text)
    if not m:
        raise RuntimeError("no parsable TOTAL ENERGY line in the xtb output")
    energy = float(m.group(1))
    if not np.isfinite(energy):
        raise RuntimeError(f"non-finite energy parsed: {m.group(1)!r}")
    return energy


def check_scf_markers(text, stage):
    """Fail hard on any unconverged-SCF marker in the xtb output
    (independent of the exit code) and require the 'PC potential true'
    plumbing flag (an embedding silently dropped would return the FREE
    energy -- the --tblite failure mode, empirically observed)."""
    if "convergence criteria cannot be satisfied" in text:
        raise RuntimeError(f"{stage}: xtb SCF did not converge "
                           f"('cannot be satisfied' marker in output)")
    if not _PCFLAG_RE.search(text):
        raise RuntimeError(
            f"{stage}: the xtb setup block does not confirm 'PC potential "
            f"true' (embedding silently dropped?)")


def check_plausibility(energy_eh, free_ref_eh, stage):
    """Fail hard on an implausible embedded energy (a garbage SCF lands
    outside any physical embedding shift of the free reference)."""
    shift = abs(energy_eh - free_ref_eh)
    if shift > PLAUSIBILITY_MAX_EH:
        raise RuntimeError(
            f"{stage}: implausible energy |E - E_free| = {shift:g} Eh "
            f"exceeds the plausibility bound {PLAUSIBILITY_MAX_EH} Eh")


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

class XtbProvider(contract.FitnessProvider):
    """GFN2-xTB embedded-point-charge provider, CLI implementation.

    Init hard-gates all three canonical inputs (sha256 of the file
    bytes against the frozen values + the 26-atom load), failing hard
    and naming the offending file. mode="cli" is the only implemented
    mode (see the module docstring's architecture finding).
    """

    FITNESS_ID = "xtb"

    PROVIDER_META = {
        "provider": "xtb",
        "fitness_id": FITNESS_ID,
        "description": (
            "GFN2-xTB embedding oracle: f = E(TS;q) - E(R;q) in kcal/mol, "
            "two frozen single points with two +-1 e external point "
            "charges at the candidate sites"
        ),
        "units": "kcal/mol",
        "kcal_per_eh": KCAL_PER_EH,
        "method": ("GFN2-xTB single point (SCC accuracy 1.0, SCF conv "
                   "1e-5 Eh, electronic temperature 300 K -- program "
                   "defaults) + the FROZEN global SCC protocol T2 "
                   f"($scc maxiterations={SCC_MAX_ITERATIONS} broydamp="
                   f"{SCC_BROYDAMP} -- see scc_protocol; adopted "
                   f"{SCC_PROTOCOL_ADOPTED})"),
        "mode": "cli",
        "modes_available": ["cli"],
        "modes_unavailable": {
            "tblite": (
                "tblite 0.7.0 (cp313 manylinux wheel) installs and imports "
                "but exposes NO external point-charge API anywhere in the "
                "tblite library; xtb --tblite silently ignores the "
                "$embedding block (evaluates the free energy, measured "
                "2026-10-08). PyPI `xtb` bindings have no cp313 wheel; "
                "the 6.7.1 release ships no capi build. In-process "
                "embedding needs an explicitly granted third channel "
                "(pending decision)."
            ),
        },
        "pcem_units": PCEM_UNITS,
        "pcem_units_gate": (
            "verified empirically 2026-10-08: interface=orca reads the "
            "file in Angstroem (obs matches the closed-form dipole "
            "asymptote q*mu/r^2 to -14%/+2%, residuals decaying ~1/R^3; a "
            "bohr misread is off by 3.5714x). The legacy interface reads "
            "Bohr and agrees bitwise on the same physical placements; the "
            "anti-parallel sign flip is verified. Source: src/embedding.f90 "
            "v6.7.1 conv = aatoau for orca, 1.0 (bohr) for legacy",
        ),
        "pcem_interface": PCEM_INTERFACE,
        "pcem_hardness": PCEM_HARDNESS,
        "pcem_hardness_gate": (
            "bare-charge limit verified materially 2026-10-08: gam 99 vs "
            "1e6 differ by 1.5e-3 kcal/mol; the defaulted dummy-atom gam "
            "= eta(N) differs by 12.25 kcal/mol on the canonical O1 pair",
        ),
        "pcem_writer": (
            "count line + one 'q x y z 99' line per charge; positions in "
            "Angstroem written through byte-identical (repr of float64, "
            "zero conversion; plain-float cast first -- np.float64 reprs "
            "mis-parse in the fortran reader)",
        ),
        "pc_pc_term": (
            "none: E(q1,q2) - E(q1) - E(q2) + E_free = 0.0118 kcal/mol "
            "at far separations; the embedded energy is the QC-PC "
            "interaction only (verified empirically)",
        ),
        "xtb": {
            "version": XTB_VERSION,
            "binary": "bin/xtb-6.7.1/bin/xtb",
            "tarball": "bin/xtb-6.7.1-linux-x86_64.tar.xz",
            "source": ("grimme-lab/xtb release v6.7.1 (the briefing's "
                       "xtb-xyz/xtb org name has moved), "
                       "https://github.com/grimme-lab/xtb/releases/"
                       "download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"),
            "tarball_sha256": XTB_TARBALL_SHA256,
        },
        "invocation": ("xtb geom.xyz --sp --gfn 2 --norestart -I control; "
                       "$embedding interface=orca input=pcem.in + the "
                       "frozen global SCC protocol ($scc maxiterations="
                       f"{SCC_MAX_ITERATIONS} broydamp={SCC_BROYDAMP}); "
                       "OMP/MKL/OPENBLAS_NUM_THREADS=1"),
        "scc_protocol": {
            "maxiterations": SCC_MAX_ITERATIONS,
            "broydamp": SCC_BROYDAMP,
            "adopted": SCC_PROTOCOL_ADOPTED,
            "source": SCC_PROTOCOL_SOURCE,
            "alternatives_rejected_or_not_adopted": (
                "T1 (maxiter alone): rejected - 2000 still-oscillating "
                "iterations, 12min55s wasted, no rescue; T3 (broydamp "
                "0.05): equivalent to 6e-6 kcal/mol, not adopted; the "
                "default protocol: fails the extreme-dipole reactant "
                "side (exit 128 at the 250-iteration ceiling)"),
            "continuity_vs_default": ("<= 2.1e-5 kcal/mol per side, "
                                      "0/30 bitwise (the same fixed "
                                      "point; the path noise); the "
                                      "reference layer re-pinned "
                                      "2026-10-08"),
            "measured_pace": ("0.067 s/single point (ratio 1.078 vs the "
                              "default's 0.062 s; the diagnostic's "
                              "timed sample); the canonical projection "
                              "3.9 h at 8 workers"),
        },
        "domain_rule": {
            "scc_exclusion": (
                "a nonzero exit carrying the SCC non-convergence "
                "signature ('Self consistent charge iterator did not "
                "converge' / the ceiling marker) under the frozen T2 "
                "protocol = the configuration is OUTSIDE the fitness's "
                "domain: excluded, recorded (config, side, marker, raw "
                "evidence), never penalised, never retried, the run "
                "continues; the stage completes only when every row is "
                "evaluated OR domain-excluded"),
            "everything_else_fail_hard": (
                "missing marker, parse error, plausibility breach, "
                "missing PC flag, non-finite: plumbing = full stop"),
            "plausibility_ceiling_evidence": (
                "re-derived 2026-10-08 on the 5 closest kept pairs x 4 "
                "signs under T2: max |shift| 0.2778 Eh = 174.3 kcal/mol "
                "at (150, 151, -1, -1); headroom 1.80x; the 0.5 Eh "
                "ceiling stays"),
        },
        "plausibility": {
            "max_shift_eh": PLAUSIBILITY_MAX_EH,
            "free_ts_eh": XTB_E_FREE_TS_EH,
            "free_r_eh": XTB_E_FREE_R_EH,
        },
        "geometries": {
            "ts": "data/derived/structures/ts_opt.xyz",
            "ts_sha256": _TS_SHA256,
            "reactant": "data/derived/structures/chorismate_reactive_TSframe.xyz",
            "reactant_sha256": _R_SHA256,
        },
        "cloud": {
            "path": "data/derived/candidates_v2.npy",
            "sha256": _CLOUD_SHA256,
            "gate": "bitwise identity with the locked cloud at every call",
        },
        "scratch": "runs/xtb_ff/<config_key>/{ts,reactant}/ (wiped per run)",
        "accounting": ("contract-inherited: every successful row counts "
                       "in n_evals (2 single points per row); failed "
                       "calls count nothing"),
    }

    def __init__(self, ts_path=None, r_path=None, cloud_path=None,
                 cloud_sha256=None, scratch_root=None, binary=None,
                 mode="cli"):
        super().__init__()
        if mode != "cli":
            raise ValueError(
                f"mode {mode!r} is not implemented for the xtb provider "
                f"(modes_available: ['cli']); "
                f"{self.PROVIDER_META['modes_unavailable'].get(mode, '')}")
        if binary is None and not XTB_BIN.is_file():
            raise ValueError(
                f"{XTB_BIN}: the xtb binary is not installed "
                "(see the module docstring's dependency record)")

        self.ts_path = Path(ts_path) if ts_path is not None else TS_XYZ
        self.r_path = Path(r_path) if r_path is not None else R_XYZ
        self.cloud_path = (Path(cloud_path) if cloud_path is not None
                           else CLOUD_NPY)
        # Work-item-4b extension (defaults unchanged): an explicit cloud
        # sha256 re-targets the per-call identity gate at a replicate
        # cloud (path + sha declared together); the default (None) keeps
        # the canonical gate -- no implicit acceptance of another cloud.
        self._expected_cloud_sha = (str(cloud_sha256)
                                    if cloud_sha256 is not None
                                    else _CLOUD_SHA256)
        self.scratch_root = (Path(scratch_root) if scratch_root is not None
                             else SCRATCH_ROOT)
        self.binary = Path(binary) if binary is not None else XTB_BIN
        self.mode = mode

        # Hard gates: sha256 of the file bytes + 26-atom load, failing
        # hard and naming the offending file.
        for path, want, what in ((self.ts_path, _TS_SHA256, "TS geometry"),
                                 (self.r_path, _R_SHA256,
                                  "registered reactant geometry"),
                                 (self.cloud_path,
                                  self._expected_cloud_sha,
                                  "candidate cloud")):
            if not path.is_file():
                raise ValueError(f"{path}: canonical {what} not found")
            got = geo_sha256(path)
            if got != want:
                raise ValueError(
                    f"{path}: sha256 mismatch ({what} tampered or wrong? "
                    f"got {got}, expected {want})")
        self._ts_sha = geo_sha256(self.ts_path)
        self._r_sha = geo_sha256(self.r_path)
        load_xyz(self.ts_path)     # 26-atom gate, naming the file
        load_xyz(self.r_path)
        self._cloud = np.load(self.cloud_path)  # sha256 already gated
        self._free_refs = {"ts": XTB_E_FREE_TS_EH,
                           "reactant": XTB_E_FREE_R_EH}
        configured_canonical = (
            os.path.realpath(str(self.cloud_path))
            == os.path.realpath(str(CLOUD_NPY))
            and self._expected_cloud_sha == _CLOUD_SHA256)
        if (cloud_path is not None or cloud_sha256 is not None) and \
                not configured_canonical:
            # an explicitly-configured (replicate) identity: the
            # instance's meta record carries the actual cloud identity
            # (the class-level PROVIDER_META stays the canonical default)
            self.PROVIDER_META = {
                **XtbProvider.PROVIDER_META,
                "cloud": {
                    "path": str(self.cloud_path),
                    "sha256": self._expected_cloud_sha,
                    "canonical": self._expected_cloud_sha == _CLOUD_SHA256,
                    "gate": ("bitwise identity with the configured cloud "
                             "at every call"),
                },
            }

    # -- internal: one single point on a frozen geometry --------------------

    def _launch_xtb(self, working_dir, out_path, err_path):
        """The pinned subprocess launch: fresh process, restart disabled,
        1 thread everywhere. Returns the exit code.

        Test seam: the sentinel suite patches this (mechanics + parsing
        stay real; no xtb binary needed for the sentinel tests).
        """
        cmd = [str(self.binary), "geom.xyz", "--sp", "--gfn", "2",
               "--norestart", "-I", "control"]
        env = {k: v for k, v in os.environ.items()}
        env.update(PINNED_THREADS)
        with out_path.open("w") as out, err_path.open("w") as err:
            return subprocess.run(cmd, env=env, cwd=str(working_dir),
                                  stdout=out, stderr=err).returncode

    def _single_point(self, working_dir, stage, pcem_lines, config_label):
        """One fresh-SCF embedded single point in its own wiped scratch
        dir; returns the energy in kcal/mol (frozen conversion)."""
        key = config_key(*config_label, self._ts_sha, self._r_sha)
        working_dir = self.scratch_root / key / stage
        if working_dir.exists():
            shutil.rmtree(working_dir)   # wipe: a fresh calculation every time
        working_dir.mkdir(parents=True)
        shutil.copy(self.ts_path if stage == "ts" else self.r_path,
                    working_dir / "geom.xyz")   # verbatim bytes of the frozen file
        working_dir.joinpath("pcem.in").write_text(pcem_lines)
        working_dir.joinpath("control").write_text(render_control())

        out_path = working_dir / "xtb.out"
        err_path = working_dir / "xtb.err"
        rc = self._launch_xtb(working_dir, out_path, err_path)
        text = out_path.read_text() + err_path.read_text()
        if rc != 0:
            # the DOMAIN classification: the SCC non-convergence signature
            # under the frozen protocol = outside the fitness's domain (a
            # recorded exclusion, never a penalty or a retry); every other
            # failure stays a plumbing error (full stop)
            scc_sig = (_SCC_MARKER_1 in text or _SCC_MARKER_2 in text)
            if scc_sig:
                raise SccDomainExclusion(
                    f"{stage}: xtb exited {rc} for config {config_label} "
                    f"with the SCC non-convergence signature (the "
                    "domain-exclusion class, recorded; no value was "
                    "evaluated); raw: " + str(working_dir),
                    stage=stage, config_label=str(config_label),
                    config=(config_label if isinstance(config_label,
                            (tuple, list)) else None),
                    raw_dir=working_dir)
            raise RuntimeError(
                f"{stage}: xtb exited {rc} for config {config_label}; "
                "tail: " + "".join(text.splitlines(keepends=True)[-15:]))
        check_scf_markers(text, stage)
        energy = parse_xtb_energy(text)
        check_plausibility(energy, self._free_refs[stage], stage)
        return energy * KCAL_PER_EH

    # -- contract implementation -------------------------------------------

    def _eval_batch_full_impl(self, idx_i, idx_j, sign_i, sign_j, candidates):
        # The provider's identity gate: the passed cloud must BE the
        # gated one (bitwise) -- the frozen geometries' frame is what the
        # cloud coordinates live in; anything else would silently distort
        # every charge position. (The gated cloud = the canonical locked
        # one by default; a replicate-configured provider gates its own
        # explicitly-passed cloud identity.)
        if not (candidates.shape == self._cloud.shape
                and np.array_equal(candidates, self._cloud)):
            raise ValueError(
                f"candidates is not the locked candidate cloud "
                f"({self.cloud_path}): the xtb provider evaluates only "
                "the frozen TS-frame cloud; its coordinates are the "
                "embedded-charge positions and must match bitwise")
        m = idx_i.size
        f = np.empty(m, dtype=np.float64)
        e_ts = np.empty(m, dtype=np.float64)
        e_r = np.empty(m, dtype=np.float64)
        for k in range(m):
            i, j = int(idx_i[k]), int(idx_j[k])
            si, sj = int(sign_i[k]), int(sign_j[k])
            pcem = render_pcem(1.0 * si, 1.0 * sj, candidates[i], candidates[j])
            label = f"(idx_i={i}, idx_j={j}, sign_i={si:+d}, sign_j={sj:+d})"
            e_ts[k] = self._single_point(None, "ts", pcem, (i, j, si, sj))
            e_r[k] = self._single_point(None, "reactant", pcem, (i, j, si, sj))
            f[k] = e_ts[k] - e_r[k]   # bitwise the contract identity
        return f, e_ts, e_r


# Contract surface: the four names + n_evals.
#
# Bound LAZILY (PEP 562): constructing the provider checks that the xtb
# binary is installed, so an eager binding made the whole package refuse to
# import on a machine without xtb -- including the oracle-mode GA, which
# never touches xtb. The first access still constructs the provider and
# still fails hard; the binary is required only where it is used.
_provider = None


def _provider_instance():
    global _provider
    if _provider is None:
        _provider = XtbProvider()
    return _provider

# The two static contract names are class attributes, so they bind without
# constructing anything (and so without requiring the binary):
FITNESS_ID = XtbProvider.FITNESS_ID      # noqa: E305
PROVIDER_META = XtbProvider.PROVIDER_META

_CONTRACT = ("provider", "eval_batch_full", "eval_batch")


def __getattr__(name):
    if name in _CONTRACT:
        prov = _provider_instance()
        return prov if name == "provider" else getattr(prov, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
