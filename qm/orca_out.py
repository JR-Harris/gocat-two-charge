"""ORCA output (.out) parser. Fail-hard, one concern per function.

Every parse function takes the file path, reads the file itself, and
raises ValueError naming the file on any problem. The composite
parse_output() gates a file on normal termination and SCF convergence
before extracting anything — a file without the normal-termination
marker is not parsed.

Format facts this parser relies on (ORCA 6.1.1, r2SCAN-3c jobs):
  * "ORCA TERMINATED NORMALLY" is present exactly once in a clean run.
  * "SCF NOT CONVERGED" never appears in a clean run.
  * "FINAL SINGLE POINT ENERGY   <x>" is printed once per SCF; OptTS
    prints a guessed-Hessian vibrational block first and the final
    Hessian block second, so the LAST instances are the meaningful ones.
  * Final geometries appear as "CARTESIAN COORDINATES (ANGSTROEM)"
    blocks (6 decimals); the last one is the final geometry.
  * The echoed input block ("|  n>  ...") carries the route keywords,
    %maxcore, PALn, and (for IRC jobs) the %irc MaxIter.
"""

import re

ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
VIB_HEADER = "VIBRATIONAL FREQUENCIES"
FREQ_RE = re.compile(r"^\s*(\d+):\s*(-?\d+\.\d+)\s+cm\*\*-1")
GEOM_HEADER = "CARTESIAN COORDINATES (ANGSTROEM)"
GEOM_LINE_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
ROUTE_RE = re.compile(r"^\|\s*\d+>\s*!(.+)$", re.M)
MAXCORE_RE = re.compile(r"^\|\s*\d+>\s*%maxcore\s+(\d+)", re.M)
PAL_RE = re.compile(r"\bPAL(\d+)\b")
ECHO_MAXITER_RE = re.compile(r"^\|\s*\d+>\s*MaxIter\s+(\d+)", re.M)
SCAN_RANGE_RE = re.compile(
    r"range=\s*(-?\d+\.\d+)\s*\.\.\s*(-?\d+\.\d+)\s+steps\s*=\s*(\d+)"
)
SCAN_STEP_RE = re.compile(r"RELAXED SURFACE SCAN STEP\s+(\d+)")
IRC_TABLE_ROW_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s")


def _read(path):
    try:
        return open(path).read()
    except OSError as e:
        raise ValueError(f"{path}: cannot read ({e})")


def require_normal_termination(path):
    """Fail hard unless the file contains ORCA TERMINATED NORMALLY."""
    if "ORCA TERMINATED NORMALLY" not in _read(path):
        raise ValueError(f"{path}: no 'ORCA TERMINATED NORMALLY' marker — not parsed")


def require_scf_converged(path):
    """Fail hard if SCF NOT CONVERGED appears anywhere, naming the file."""
    if "SCF NOT CONVERGED" in _read(path):
        raise ValueError(f"{path}: contains 'SCF NOT CONVERGED'")


def final_energy_eh(path):
    """The LAST 'FINAL SINGLE POINT ENERGY' in the file (Eh)."""
    matches = ENERGY_RE.findall(_read(path))
    if not matches:
        raise ValueError(f"{path}: no 'FINAL SINGLE POINT ENERGY' found")
    return float(matches[-1])


def vibrational_frequencies(path):
    """The LAST 'VIBRATIONAL FREQUENCIES' block only ->
    [(mode_index, freq_cm1), ...] with the printed 0-based indices.

    OptTS + Calc_Hess outputs print two blocks (guess Hessian first,
    final second); the last one is the physical one.
    """
    text = _read(path)
    starts = [m.end() for m in re.finditer(re.escape(VIB_HEADER), text)]
    if not starts:
        raise ValueError(f"{path}: no '{VIB_HEADER}' block found")
    tail = text[starts[-1]:]
    freqs = []
    for line in tail.splitlines():
        m = FREQ_RE.match(line)
        if m:
            freqs.append((int(m.group(1)), float(m.group(2))))
        elif freqs and line.strip():
            break  # left the frequency table
    if not freqs:
        raise ValueError(f"{path}: last '{VIB_HEADER}' block has no frequency lines")
    return freqs


def classify_frequencies(freqs):
    """Classify a parsed frequency list with the locked thresholds.

    hard imaginary: freq < -50 cm-1; soft imaginary: -50 <= freq < -10
    cm-1 (both reported). Returns (n_modes, n_hard, n_soft, hard_modes)
    where hard_modes is [[index, freq], ...] in the locked summaries'
    imaginary_modes form.
    """
    hard_threshold, soft_threshold = -50.0, -10.0
    hard_modes = [[i, f] for i, f in freqs if f < hard_threshold]
    n_soft = sum(1 for _i, f in freqs if hard_threshold <= f < soft_threshold)
    return len(freqs), len(hard_modes), n_soft, hard_modes


def final_geometry(path):
    """The LAST 'CARTESIAN COORDINATES (ANGSTROEM)' block ->
    (elements, coords_A). Used for key distances and connectivity."""
    text = _read(path)
    starts = [m.end() for m in re.finditer(re.escape(GEOM_HEADER), text)]
    if not starts:
        raise ValueError(f"{path}: no '{GEOM_HEADER}' block found")
    tail = text[starts[-1]:]
    elements, coords = [], []
    for line in tail.splitlines():
        m = GEOM_LINE_RE.match(line)
        if m:
            elements.append(m.group(1))
            coords.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        elif elements:
            break  # left the coordinates table
    if not elements:
        raise ValueError(f"{path}: last '{GEOM_HEADER}' block has no atoms")
    return elements, coords


def optimization_converged(path):
    """True if the file reports THE OPTIMIZATION HAS CONVERGED."""
    return "THE OPTIMIZATION HAS CONVERGED" in _read(path)


def echo_route_keywords(path):
    """The route keywords from the echoed input's '!' line, as tokens."""
    m = ROUTE_RE.search(_read(path))
    if not m:
        raise ValueError(f"{path}: no echoed route ('!') line found")
    return m.group(1).split()


def echo_pal(path):
    """The PALn core count from the echoed route line."""
    m = PAL_RE.search(" ".join(echo_route_keywords(path)))
    if not m:
        raise ValueError(f"{path}: no PALn in the echoed route line")
    return int(m.group(1))


def echo_maxcore(path):
    """The %maxcore value (MB) from the echoed input."""
    m = MAXCORE_RE.search(_read(path))
    if not m:
        raise ValueError(f"{path}: no '%maxcore' in the echoed input")
    return int(m.group(1))


def echo_irc_maxiter(path):
    """The %irc MaxIter from the echoed input (IRC jobs only).

    Must match the echoed-input format '|  n>  MaxIter N' — irc.out also
    prints unrelated 'MaxIter ....' settings lines.
    """
    m = ECHO_MAXITER_RE.search(_read(path))
    if not m:
        raise ValueError(f"{path}: no '%irc MaxIter' in the echoed input")
    return int(m.group(1))


def scan_info(path):
    """Relaxed scan definition -> (range_start_A, range_end_A, n_steps),
    from the 'range= X .. Y steps = N' line."""
    m = SCAN_RANGE_RE.search(_read(path))
    if not m:
        raise ValueError(f"{path}: no relaxed-scan 'range= ... steps =' line found")
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def scan_peak_step(path):
    """The relaxed-scan step (1-based) whose FINAL energy is the highest.

    Each scan step is itself a constrained optimization with several
    'FINAL SINGLE POINT ENERGY' lines; a step's energy is the LAST
    energy line within its own section.
    """
    text = _read(path)
    parts = re.split(r"RELAXED SURFACE SCAN STEP\s+(\d+)", text)
    per_step = {}
    for i in range(1, len(parts), 2):
        step = int(parts[i])
        energies = ENERGY_RE.findall(parts[i + 1])
        if not energies:
            raise ValueError(f"{path}: scan step {step} has no energy")
        per_step[step] = float(energies[-1])
    if not per_step:
        raise ValueError(f"{path}: no 'RELAXED SURFACE SCAN STEP' sections found")
    return max(per_step, key=per_step.get)


def irc_path_summary(path):
    """The IRC PATH SUMMARY table -> (ts_energy_eh, n_rows).

    ts_energy_eh is the energy of the row marked '<= TS' (the table
    prints 6 decimals — the locked irc_summary.json records exactly
    that value); n_rows is the number of data rows.
    """
    text = _read(path)
    m = re.search(r"IRC PATH SUMMARY", text)
    if not m:
        raise ValueError(f"{path}: no 'IRC PATH SUMMARY' table found")
    tail = text[m.end():]
    rows, ts_energy = 0, None
    for line in tail.splitlines():
        r = IRC_TABLE_ROW_RE.match(line)
        if not r:
            if rows and line.strip() == "":
                break
            continue
        rows += 1
        if "<= TS" in line:
            ts_energy = float(r.group(2))
    if not rows:
        raise ValueError(f"{path}: 'IRC PATH SUMMARY' table has no data rows")
    if ts_energy is None:
        raise ValueError(f"{path}: no '<= TS' row in the IRC PATH SUMMARY")
    return ts_energy, rows


def parse_output(path, need_frequencies=True):
    """Composite entry point: gate the file, then extract the standard
    fields -> dict. Fails hard (naming the file) on any gate or parse
    failure. need_frequencies=False for jobs without a frequency block.
    """
    require_normal_termination(path)
    require_scf_converged(path)
    out = {
        "final_energy_eh": final_energy_eh(path),
        "geometry": final_geometry(path),
        "opt_converged": optimization_converged(path),
        "route_keywords": echo_route_keywords(path),
        "pal": echo_pal(path),
        "maxcore_mb": echo_maxcore(path),
    }
    if need_frequencies:
        freqs = vibrational_frequencies(path)
        n_modes, n_hard, n_soft, hard_modes = classify_frequencies(freqs)
        out.update(
            n_modes=n_modes, n_imaginary=n_hard, n_soft_imaginary=n_soft,
            imaginary_modes=hard_modes,
        )
    return out