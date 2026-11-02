"""ORCA input rendering for the v2 QM stages.

The endpoint jobs (and later stages) use the fixed v2 method:
    ! r2SCAN-3c Opt Freq TightSCF PAL8
    %maxcore 1500
    * xyz 0 1
      ... geometry ...
    *
Analytic Freq only — never NumFreq. PAL 8 and %maxcore 1500 are the
project-wide resource contract (enforced by qm/orca_runner before any
launch). The geometry-line format is the settled v1 template
(6-decimal coordinates), confirmed against the frozen v1 archive — the
v2 endpoint inputs differ from the v1 chor.inp/preph.inp only in the
keyword line (NumFreq -> analytic Freq) and %maxcore (2000 -> 1500).
"""

from qm import orca_runner

KEYWORDS = "r2SCAN-3c Opt Freq TightSCF"
CHARGE = 0
MULT = 1


def geometry_lines(elements, coords, charge=CHARGE, mult=MULT):
    """The '* xyz <charge> <mult>' geometry block (markers inclusive)
    as a list of lines, in the settled v1 format (6-decimal coordinates).
    Single source for every inline-geometry template (the Opt+Freq
    endpoint inputs here and the ts/irc chain templates)."""
    if len(elements) != len(coords):
        raise ValueError(
            f"geometry mismatch: {len(elements)} elements vs {len(coords)} rows"
        )
    lines = [f"* xyz {charge} {mult}"]
    for el, (x, y, z) in zip(elements, coords):
        lines.append(f"  {el:<3s} {x: .6f} {y: .6f} {z: .6f}")
    lines.append("*")
    return lines


def render_opt_freq_inp(elements, coords, charge=CHARGE, mult=MULT):
    """Render an Opt + analytic Freq input -> exact text.

    elements: list of element symbols; coords: (n, 3) Angstrom. The
    resource keywords come from the runner's contract (single source:
    PAL 8, %maxcore 1500 on every site).
    """
    lines = [
        f"! {KEYWORDS} PAL{orca_runner.PAL}",
        f"%maxcore {orca_runner.MAXCORE_MB}",
        *geometry_lines(elements, coords, charge, mult),
    ]
    text = "\n".join(lines) + "\n"
    orca_runner.assert_resource_keywords(text)  # PAL 8 + %maxcore 1500, by construction
    return text


def write_inp(path, elements, coords, charge=CHARGE, mult=MULT):
    """Render and write an input file to `path` (the caller is
    responsible for any GOCAT_OUT_ROOT redirection of `path`)."""
    with open(path, "w") as fh:
        fh.write(render_opt_freq_inp(elements, coords, charge, mult))