"""Site-configurable ORCA runner + output-root redirection (v2 HPC layer).

Sites (selected by GOCAT_ORCA_SITE, default "laptop"; unknown values
fail hard listing the known sites):

  laptop  ORCA 6.1.1 shared-openmpi418 avx2 build under /opt/orca-6.1.1
          with its private OpenMPI 4.1.8. Env semantics are the
          v1-proven ones (semantically confirmed against the frozen v1
          archive; see the stage-3 log): <openmpi>/bin and ORCADIR on
          PATH, <openmpi>/lib and ORCADIR on LD_LIBRARY_PATH, ORCADIR
          set. NOTE: the real install directory is
          orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 (underscores,
          as in the live PATH and v1's working scripts; the stage-3
          briefing's hyphenated name does not exist on Tsukuyomi).
  hpc1    ORCA from module app/orca/6.1.1 (flat /apps/ORCA/6.1.1; the
          module sets PATH but exports no ORCADIR) + the cluster's
          openmpi/4.1.1, whose real libraries live in
          /apps/openmpi/4.1.1/lib while its modulefile prepends a
          nonexistent lib64 dir. The briefing's site line names the
          openmpi prefix — completed with /lib exactly like the laptop
          line's prefix must be (a bare prefix on LD_LIBRARY_PATH
          resolves nothing, and the real library location is stated as
          .../4.1.1/lib); the ORCA entry is literal: /apps/ORCA/6.1.1/lib.

On every site the subprocess env is built from scratch: the inherited
PATH / LD_LIBRARY_PATH / ORCADIR are never trusted — they are replaced
(v1 prepended to the inherited values; the buggy hpc1 modulefile is the
reason v2 does not).

    PATH             <openmpi>/bin : ORCADIR : /usr/local/bin : /usr/bin : /bin
    LD_LIBRARY_PATH  <openmpi>/lib : site ORCA entry (see above)
    ORCADIR          the ORCA install root

Runner contract:
  * absolute-path invocation (binary and input); one ORCA process at a
    time (structural module lock);
  * every input must declare PAL 8 and %maxcore 1500 — enforced before
    launch, no exceptions (assert_resource_keywords);
  * resume gate: an existing .out that terminated normally AND parses
    to a final energy AND is not SCF-failed is skipped (the SCF clause
    is a deliberate stricter reading of the briefing's "terminated +
    parsable energy" gate — a failed-SCF output is never silently
    accepted as a completed job; runner-written outputs always satisfy
    it, so resume behaviour is unchanged for our own runs);
  * stderr captured per job (<stem>.stderr);
  * nonzero exit fails hard with the .out tail; a zero exit whose .out
    does not terminate normally / parse also fails hard.

Output-root redirection (the cluster write contract): GOCAT_OUT_ROOT,
when set, maps every repo-relative path under data/derived/orca_v2/ or
runs/ (the only places QM stages write) under the root, relative
structure preserved; every other path (inputs, locked artifacts,
absolute paths) passes through unchanged. Unset -> repo-relative
defaults (laptop behaviour unchanged). The cluster manager points
GOCAT_OUT_ROOT at data/hpc_out/, which makes the write contract hold by
construction. run_orca applies resolve_output to its output and stderr
paths.
"""

import os
import re
import subprocess
import threading
import time

from qm import orca_out

GOCAT_ORCA_SITE = "GOCAT_ORCA_SITE"
GOCAT_OUT_ROOT = "GOCAT_OUT_ROOT"

# Resource contract: PAL 8, %maxcore 1500 MB on every site, no exceptions.
PAL = 8
MAXCORE_MB = 1500

_ORCA_DIR_LAPTOP = "/opt/orca-6.1.1/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2"
_OMPI_LAPTOP = "/opt/orca-6.1.1/openmpi-4.1.8"

SITES = {
    "laptop": {
        "orca_bin": f"{_ORCA_DIR_LAPTOP}/orca",
        "orcadir": _ORCA_DIR_LAPTOP,
        # v1-proven entries: openmpi bin/lib + ORCADIR itself.
        "path_entries": [f"{_OMPI_LAPTOP}/bin", _ORCA_DIR_LAPTOP],
        "ld_library_entries": [f"{_OMPI_LAPTOP}/lib", _ORCA_DIR_LAPTOP],
    },
    "hpc1": {
        "orca_bin": "/apps/ORCA/6.1.1/orca",
        "orcadir": "/apps/ORCA/6.1.1",
        # openmpi bin (mpirun must resolve under the driver env) + the
        # flat ORCA root; LD_LIBRARY_PATH = the real openmpi library
        # directory + the ORCA lib dir (briefing entries, prefix
        # completed with /lib — see module docstring).
        "path_entries": ["/apps/openmpi/4.1.1/bin", "/apps/ORCA/6.1.1"],
        "ld_library_entries": ["/apps/openmpi/4.1.1/lib", "/apps/ORCA/6.1.1/lib"],
    },
}

# The minimal system tail of the from-scratch PATH.
SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")

# Inherited variables that are never trusted (replaced wholesale).
REWRITE_VARS = ("PATH", "LD_LIBRARY_PATH", "ORCADIR")

# The only repo-relative prefixes QM stages write under; these are the
# ones GOCAT_OUT_ROOT redirects.
_OUTPUT_PREFIXES = ("data/derived/orca_v2", "runs")

_PAL_RE = re.compile(r"\bPAL(\d+)\b")
_MAXCORE_RE = re.compile(r"%maxcore\s+(\d+)")

_ONE_AT_A_TIME = threading.Lock()


def site():
    """The active site: GOCAT_ORCA_SITE, default "laptop". Fails hard
    on unknown values, listing the known sites."""
    name = os.environ.get(GOCAT_ORCA_SITE, "laptop")
    if name not in SITES:
        raise ValueError(
            f"unknown GOCAT_ORCA_SITE {name!r} "
            f"(known sites: {', '.join(sorted(SITES))})"
        )
    return name


def build_env(site_name=None):
    """The ORCA subprocess env for a site, built from scratch.

    PATH / LD_LIBRARY_PATH / ORCADIR are replaced with the site values
    (inherited values — including anything set by modulefiles — are
    never trusted); all other environment variables (HOME, TMPDIR, ...)
    are kept. site_name defaults to the GOCAT_ORCA_SITE selection.
    """
    name = site() if site_name is None else site_name
    if name not in SITES:
        raise ValueError(
            f"unknown site {name!r} (known sites: {', '.join(sorted(SITES))})"
        )
    profile = SITES[name]
    env = {k: v for k, v in os.environ.items() if k not in REWRITE_VARS}
    env["PATH"] = os.pathsep.join([*profile["path_entries"], *SYSTEM_PATH])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(profile["ld_library_entries"])
    env["ORCADIR"] = profile["orcadir"]
    return env


def out_root():
    """The active output root (GOCAT_OUT_ROOT) or None."""
    return os.environ.get(GOCAT_OUT_ROOT) or None


def resolve_output(path):
    """Map a repo-relative output path through GOCAT_OUT_ROOT.

    When the root is set, paths under data/derived/orca_v2/ or runs/
    resolve under it (relative structure preserved); every other path —
    inputs, locked artifacts, absolute paths — passes through
    unchanged. Unset -> the path itself (laptop default).
    """
    root = out_root()
    if root is None:
        return path
    norm = os.path.normpath(path)
    for prefix in _OUTPUT_PREFIXES:
        if norm == prefix or norm.startswith(prefix + os.sep):
            return os.path.join(root, norm)
    return path


def assert_resource_keywords(inp_text, path=""):
    """Every ORCA input must declare PAL 8 and %maxcore 1500 — no
    exceptions. Fails hard when either is missing or has any other
    value."""
    pals = _PAL_RE.findall(inp_text)
    if not pals:
        raise ValueError(f"{path}: input declares no PALn keyword (PAL {PAL} required)")
    bad = [p for p in pals if int(p) != PAL]
    if bad:
        raise ValueError(
            f"{path}: input must declare PAL {PAL} only, found PAL {', '.join(bad)}"
        )
    m = _MAXCORE_RE.search(inp_text)
    if not m:
        raise ValueError(
            f"{path}: input declares no %maxcore ({MAXCORE_MB} required)"
        )
    if int(m.group(1)) != MAXCORE_MB:
        raise ValueError(
            f"{path}: input %maxcore must be {MAXCORE_MB}, found {m.group(1)}"
        )


def resume_skip(out_path):
    """True if out_path already holds a complete run: normal
    termination + a parsable final energy (via qm.orca_out) + no SCF
    failure — see the module docstring for the SCF clause."""
    if not os.path.exists(out_path):
        return False
    try:
        orca_out.require_normal_termination(out_path)
        orca_out.require_scf_converged(out_path)
        orca_out.final_energy_eh(out_path)
    except ValueError:
        return False
    return True


def _tail(path, n=25):
    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no output file)"


def _launch(cmd, env, cwd, stdout_path, stderr_path):
    """Launch exactly one subprocess and return its exit code.

    Test seam: the sentinel tests monkeypatch this so no ORCA (or any
    subprocess) is ever executed on the laptop.
    """
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        return subprocess.run(cmd, env=env, cwd=cwd, stdout=out, stderr=err).returncode


def run_orca(inp_path, out_path, site_name=None, stderr_path=None):
    """Run ONE ORCA job: <inp> -> <out>, stderr -> <stem>.stderr.

    out_path (and stderr_path) go through resolve_output (cluster write
    contract). The input must exist and declare PAL 8 + %maxcore 1500.
    Resume gate: a complete existing .out is skipped (its parsed energy
    is returned). Fails hard on nonzero exit (with the .out tail) or on
    a zero exit whose .out does not terminate normally / parse.
    Returns a result dict:
      {skipped, site, out, stderr, exit_code, final_energy_eh}.
    """
    name = site() if site_name is None else site_name
    if name not in SITES:
        raise ValueError(
            f"unknown site {name!r} (known sites: {', '.join(sorted(SITES))})"
        )
    profile = SITES[name]

    inp_abs = os.path.abspath(inp_path)
    if not os.path.isfile(inp_abs):
        raise ValueError(f"{inp_abs}: ORCA input not found")
    if not os.path.isfile(profile["orca_bin"]):
        raise ValueError(
            f"ORCA binary not found for site {name!r}: {profile['orca_bin']}"
        )
    with open(inp_abs) as fh:
        assert_resource_keywords(fh.read(), inp_abs)

    out_path = resolve_output(out_path)
    if stderr_path is None:
        stderr_path = os.path.splitext(out_path)[0] + ".stderr"
    else:
        stderr_path = resolve_output(stderr_path)

    if resume_skip(out_path):
        return {
            "skipped": True,
            "site": name,
            "out": out_path,
            "stderr": stderr_path,
            "exit_code": None,
            "final_energy_eh": orca_out.final_energy_eh(out_path),
            "wall_seconds": 0.0,  # not re-measured on resume
        }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    env = build_env(name)
    with _ONE_AT_A_TIME:
        t0 = time.monotonic()
        rc = _launch(
            [profile["orca_bin"], inp_abs], env,
            os.path.dirname(inp_abs), out_path, stderr_path,
        )
        wall_seconds = time.monotonic() - t0

    if rc != 0:
        raise RuntimeError(
            f"ORCA exited {rc} for {inp_abs} (site {name}); "
            f"stderr: {stderr_path}\n--- {out_path} tail ---\n{_tail(out_path)}"
        )
    # A zero exit is not sufficient: the .out must terminate normally
    # and parse (and carry no SCF failure).
    orca_out.require_normal_termination(out_path)
    orca_out.require_scf_converged(out_path)
    energy = orca_out.final_energy_eh(out_path)
    return {
        "skipped": False,
        "site": name,
        "out": out_path,
        "stderr": stderr_path,
        "exit_code": rc,
        "final_energy_eh": energy,
        "wall_seconds": round(wall_seconds, 3),
    }