"""Landscape gating + decode (the arbitration driver's inputs).

Everything the driver executes is DERIVED INSIDE from the landscape
artifacts -- no hand-built lists.  At load the module hard-gates the
whole certified-artifact set of the selected landscape (fail hard
before any ORCA call):

  * the seven artifacts exist (f.npy, e_ts.npy, e_r.npy,
    pair_indices.npy, sign_index.npy, domain_exclusions.json,
    enumeration_summary.json);
  * every artifact's sha256 equals the summary's `artifacts` block;
  * the summary's cloud_sha256 equals the candidates file's sha256;
  * the coverage identity holds: the finite-row count equals
    coverage.n_rows_evaluated and the NaN rows are exactly the
    domain-exclusions rows;
  * the summary's global-min record re-derives from f.npy within
    ARGMIN_BAND (1.952e-3 kcal/mol -- the float32-cast rounding band;
    f.npy is float32, the summary value is the float64 authority).

Row -> config decode (pinned): pair_indices.npy is per row (n_rows, 2)
int32; sign_index.npy int8 in {0,1,2,3}; sign canon
[(+,+), (+,-), (-,+), (-,-)] (the summary's universe.sign_canon).
Top-K = the rows sorted by f ascending (stable sort, NaN already
excluded by the coverage identity), first K rows.

NOTHING under data/ is ever written; the module only reads.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from fitness.orca import geo_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent

# The landscape roots (canonical this session; replicates are accepted
# by the CLI but EXECUTION is blocked until Jonny's separate GO).
LANDSCAPE_ROOTS = {
    "canonical": REPO_ROOT / "data" / "derived" / "enumeration_v2",
    "replicate-1": (REPO_ROOT / "data" / "replicates" / "replicate-1"
                    / "derived" / "enumeration_replicate-1"),
    "replicate-2": (REPO_ROOT / "data" / "replicates" / "replicate-2"
                    / "derived" / "enumeration_replicate-2"),
}
CLOUD_PATHS = {
    "canonical": REPO_ROOT / "data" / "derived" / "candidates_v2.npy",
    "replicate-1": (REPO_ROOT / "data" / "replicates" / "replicate-1"
                    / "candidates_replicate-1.npy"),
    "replicate-2": (REPO_ROOT / "data" / "replicates" / "replicate-2"
                    / "candidates_replicate-2.npy"),
}
PROVENANCE_PATHS = {
    "canonical": (REPO_ROOT / "data" / "derived"
                  / "candidates_v2_provenance.npy"),
    "replicate-1": (REPO_ROOT / "data" / "replicates" / "replicate-1"
                    / "candidates_replicate-1_provenance.npy"),
    "replicate-2": (REPO_ROOT / "data" / "replicates" / "replicate-2"
                    / "candidates_replicate-2_provenance.npy"),
}

# The seven artifacts (the summary itself is the eighth file; it is the
# gate's own authority and is validated by the re-derivations, not by a
# self-referential sha).
ARTIFACT_FILES = ("f.npy", "e_ts.npy", "e_r.npy", "pair_indices.npy",
                  "sign_index.npy", "domain_exclusions.json",
                  "enumeration_summary.json")
NPY_ARTIFACTS = ("f", "e_ts", "e_r", "pair_indices", "sign_index")

# Sign canon (the summary's universe.sign_canon, pinned here for the
# decode; the gate asserts the summary agrees).
SIGN_CANON = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# The argmin re-derivation band: |f32 - authority| <= this (kcal/mol).
# f.npy is float32; its rounding granularity is the measured max
# |f - (e_ts - e_r)| float32 residual 1.952e-3 (the briefing's number;
# the measured residual on the landed artifact reproduces it bitwise).
ARGMIN_BAND_KCAL = 1.952e-3


class Landscape:
    """The gated, loaded landscape artifact set (read-only)."""

    def __init__(self, name, root, cloud_path, arrays, summary,
                 exclusions, cloud, provenance):
        self.name = name
        self.root = Path(root)
        self.cloud_path = Path(cloud_path)
        self.arrays = arrays        # {key: np.ndarray} for the five npy
        self.summary = summary      # the summary dict
        self.exclusions = exclusions  # the domain_exclusions list
        self.cloud = cloud          # float64 (n_sites, 3)
        self.provenance_arr = provenance  # the per-site construction record
        self.cloud_sha256 = geo_sha256(self.cloud_path)
        self.f = arrays["f"]
        self.pair_indices = arrays["pair_indices"]
        self.sign_index = arrays["sign_index"]

    @property
    def n_rows(self):
        return int(self.f.shape[0])

    @property
    def n_sites(self):
        return int(self.cloud.shape[0])

    def provenance(self):
        """The per-site construction provenance (structured array; the
        probe reads its nearest_ts_atom classes)."""
        return self.provenance_arr

    def decode_row(self, row):
        """Row -> (idx_i, idx_j, sign_i, sign_j) via the pinned canon."""
        i, j = (int(v) for v in self.pair_indices[int(row)])
        s = int(self.sign_index[int(row)])
        if not 0 <= s < len(SIGN_CANON):
            raise ValueError(f"row {row}: sign_index {s} outside the canon")
        si, sj = SIGN_CANON[s]
        return i, j, si, sj

    def f_sorted_rows(self):
        """Rows sorted by f ascending (stable, NaN excluded -- the
        coverage identity guarantees one NaN at the excluded row)."""
        return np.argsort(self.f, kind="stable")

    def sha256_record(self):
        """The provenance strings of every artifact + the cloud."""
        rec = {name: geo_sha256(self.root / name) for name in ARTIFACT_FILES}
        try:
            rel = str(self.cloud_path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(self.cloud_path)   # a non-repo cloud (synthetic trees)
        rec["cloud"] = {"path": rel, "sha256": self.cloud_sha256}
        return rec


def _fail(msg):
    raise ValueError(f"landscape gate FAILED: {msg}")


def load_and_gate_landscape(name):
    """The NAMED landscape (LANDSCAPE_ROOTS/CLOUD_PATHS/PROVENANCE_PATHS)
    through the core gate."""
    root = LANDSCAPE_ROOTS.get(name)
    cloud_path = CLOUD_PATHS.get(name)
    prov_path = PROVENANCE_PATHS.get(name)
    if root is None or cloud_path is None or prov_path is None:
        _fail(f"unknown landscape {name!r} "
              f"(known: {', '.join(sorted(LANDSCAPE_ROOTS))})")
    return gate_landscape(root, cloud_path, prov_path, name)


def gate_landscape(root, cloud_path, prov_path, name):
    """Load + hard-gate the landscape at `root` (the core gate, taking
    the paths explicitly -- the named wrapper pins them; synthetic test
    trees pass their own); -> Landscape.  Every gate failure names the
    offending fact and fails BEFORE any execution."""
    root = Path(root)
    cloud_path = Path(cloud_path)
    prov_path = Path(prov_path)

    for fname in ARTIFACT_FILES:
        if not (root / fname).is_file():
            _fail(f"{root / fname}: artifact missing")
    if not prov_path.is_file():
        _fail(f"{prov_path}: the cloud-provenance artifact is missing")

    summary = json.loads((root / "enumeration_summary.json").read_text())
    arrays = {key: np.load(root / f"{key}.npy") for key in NPY_ARTIFACTS}

    # -- artifact sha256 gate against the summary's artifacts block ------
    art = summary.get("artifacts", {})
    for key in NPY_ARTIFACTS:
        fname = f"{key}.npy"
        want = (art.get(key) or {}).get("sha256")
        got = geo_sha256(root / fname)
        if want is None:
            _fail(f"{fname}: no sha256 in the summary's artifacts block")
        if got != want:
            _fail(f"{root / fname}: sha256 mismatch (got {got}, summary "
                  f"records {want})")
    dx_want = (art.get("domain_exclusions") or {}).get("sha256")
    if dx_want is not None:
        got = geo_sha256(root / "domain_exclusions.json")
        if got != dx_want:
            _fail(f"{root / 'domain_exclusions.json'}: sha256 mismatch "
                  f"(got {got}, summary records {dx_want})")

    # -- cloud sha gate ----------------------------------------------------
    if not cloud_path.is_file():
        _fail(f"{cloud_path}: the cloud artifact is missing")
    cloud_sha = geo_sha256(cloud_path)
    if summary.get("cloud_sha256") != cloud_sha:
        _fail(f"cloud_sha256 mismatch: summary {summary.get('cloud_sha256')} "
              f"vs measured {cloud_sha} ({cloud_path})")

    # -- coverage identity --------------------------------------------------
    f = arrays["f"]
    finite = int(np.isfinite(f).sum())
    nan_rows = np.flatnonzero(np.isnan(f))
    dx = json.loads((root / "domain_exclusions.json").read_text())
    dx_rows = sorted(int(e["row"]) for e in dx.get("entries", []))
    cov = summary.get("coverage", {})
    if cov.get("n_rows_evaluated") != finite:
        _fail(f"coverage: summary n_rows_evaluated "
              f"{cov.get('n_rows_evaluated')} != finite rows {finite}")
    if [int(r) for r in nan_rows] != dx_rows:
        _fail(f"coverage: NaN rows {[int(r) for r in nan_rows]} != the "
              f"domain-exclusions rows {dx_rows}")

    # -- argmin re-derivation (the float32-cast band) ------------------------
    gmin = summary.get("fitness", {}).get("global_min")
    if not gmin:
        _fail("the summary carries no fitness.global_min")
    amin = int(np.nanargmin(f))
    f32_min = float(f[amin])
    row_pair = arrays["pair_indices"][amin]
    row_sign = int(arrays["sign_index"][amin])
    if (amin != int(gmin["row"])
            or int(row_pair[0]) != int(gmin["idx_i"])
            or int(row_pair[1]) != int(gmin["idx_j"])
            or row_sign != int(gmin["sign"][0])):
        _fail(f"argmin re-derivation mismatch: f.npy argmin (row {amin}, "
              f"pair {int(row_pair[0])},{int(row_pair[1])}, sign {row_sign}) "
              f"vs summary {gmin.get('row')}, {gmin.get('idx_i')},"
              f"{gmin.get('idx_j')}, {gmin.get('sign')}")
    delta = abs(f32_min - float(gmin["f"]))
    if delta > ARGMIN_BAND_KCAL:
        _fail(f"argmin band breach: |f32 {f32_min!r} - authority "
              f"{float(gmin['f'])!r}| = {delta:g} exceeds "
              f"{ARGMIN_BAND_KCAL:g} kcal/mol")

    # -- sign canon agreement ------------------------------------------------
    canon = summary.get("universe", {}).get("sign_canon")
    if canon is not None and [list(x) for x in SIGN_CANON] != [list(x) for x in canon]:
        _fail(f"sign canon mismatch: pinned {SIGN_CANON} vs summary {canon}")

    # -- storage-shape agreement (fail on anything but the recorded) ---------
    for key, rec in ((k, art.get(k)) for k in NPY_ARTIFACTS):
        if rec is None:
            continue
        a = arrays[key]
        if "shape" in rec and list(a.shape) != list(rec["shape"]):
            _fail(f"{key}.npy: shape {list(a.shape)} != recorded "
                  f"{rec['shape']}")
        if "dtype" in rec and a.dtype.name != rec["dtype"]:
            _fail(f"{key}.npy: dtype {a.dtype.name} != recorded "
                  f"{rec['dtype']}")

    cloud = np.load(cloud_path)
    provenance = np.load(prov_path)
    if provenance.shape[0] != cloud.shape[0]:
        _fail(f"provenance/prov shape {provenance.shape[0]} != cloud shape "
              f"{cloud.shape[0]} ({prov_path} vs {cloud_path})")
    return Landscape(name, root, cloud_path, arrays, summary, dx, cloud,
                     provenance)


def sha256_sidecar(path):
    """Write `<path>.sha256` (the sha256 of the file's bytes); -> the hex."""
    path = Path(path)
    digest = geo_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def write_json_with_sidecar(path, payload):
    """Write a JSON record + its sha256 sidecar.  Keys sort, indent 2,
    the float64 values round-trip through Python's repr (JSON's float
    format round-trips float64 exactly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return sha256_sidecar(path), text