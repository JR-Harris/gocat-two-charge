"""The certified-landscape oracle provider (oracle mode).

`landscape_oracle(landscape_dir, cloud_path)` answers the fitness
contract by LOOKUP into the certified landscape
(data/derived/enumeration_v2/ -- the completed oracle, the enumeration
stage's promoted artifact). It is a contract.FitnessProvider subclass:
the contract surface is bound per landscape by the factory --

    FITNESS_ID        == the summary's recorded fitness_id
                         ("enumeration_v2" for the canonical landscape)
    PROVIDER_META     == the summary's recorded provider_meta, VERBATIM
                         (the xtb provenance: the oracle serves exactly
                         those certified values)
    eval_batch        -- the black-box view (this is what the GA's
                         engine receives as its fitness callable)
    eval_batch_full   -- the landscape-artifact view (f, e_ts, e_r)

    + the inherited per-instance `n_evals` accounting counter.

Init gates, ALL fail-hard (a landscape that fails any gate is not the
certified one; the run refuses before any query):

  1. the cloud file's sha256 equals the sha the summary records;
  2. the summary's own `row_order` + `sign_canon` fields equal THIS
     module's constants -- asserted equality against the strings/types
     ga/space.py itself defines (ga.space.ROW_ORDER / SIGN_CANON),
     never a copy-pasted literal;
  3. the artifact sha256s match the summary's recorded per-file
     certification (the summary's `artifacts` entries are the
     promotion-time record of exactly these bytes) + shapes/dtypes;
  4. n_rows match everywhere (the summary's universe record and every
     artifact);
  5. THE MASTER ALIGNMENT GATE: the pair table recomputed by
     ga/space.py from the gated cloud maps EVERY row bitwise -- for
     every row r: pair_indices.npy[r] == (pi[r//4], pj[r//4]) and
     sign_index.npy[r] == r % 4 -- the FULL-LENGTH vectorized equality,
     all 1,699,132 rows;
  6. the NaN mask of f.npy marks exactly the summary's
     domain-excluded rows and nothing else; all other values finite
     (in f, e_ts and e_r).

Serving (`_eval_batch_full_impl`): each input configuration is decoded
to its row via the space row code (out-of-order pairs canonicalize;
i == j / d_min-floor configurations have no row in the landscape ->
ValueError naming the configuration), then f / e_ts / e_r are served
from the artifacts (float32 arrays cast to float64 -- exact). A decode
to a NaN row raises ValueError naming the configuration: those rows
are OUTSIDE the fitness's domain of definition (the pre-registered
rule; the canonical example is row 522045 = (154, 833, +1, -1), the
SCC non-convergence exclusion).

ACCURACY RECORD (the promoted artifact is float32): the float64
provider-boundary record -- f = e_ts - e_r at full precision -- is the
AUTHORITY; the promoted artifact reproduces the identity to within
1.952e-3 kcal/mol (re-measured 2026-10-15 on the certified artifact:
max |e_ts - e_r - f| over the finite rows =
0.0019521713256835938 kcal/mol -- the float32 cast's rounding). The
contract base class asserts output SHAPES, not the identity; this
module inherits that boundary and documents, not re-derives, the
float64 authority.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from fitness import contract
from ga import space as ga_space

__all__ = ["landscape_oracle", "LandscapeOracle"]

ARTIFACT_NAMES = ("f", "e_ts", "e_r", "pair_indices", "sign_index")


def sha256_file(path):
    """sha256 of a file's bytes (the provenance anchor)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LandscapeOracle(contract.FitnessProvider):
    """Answers the fitness contract by lookup into a certified
    landscape. Construct via landscape_oracle(); a FRESH instance per
    run (the n_evals ledger is per-run state)."""

    FITNESS_ID = ""               # instance-bound from the summary
    PROVIDER_META = {}            # instance-bound, summary-verbatim

    def __init__(self, landscape_dir, cloud_path):
        super().__init__()
        landscape_dir = Path(landscape_dir)
        cloud_path = Path(cloud_path)

        summary_path = landscape_dir / "enumeration_summary.json"
        if not summary_path.is_file():
            raise ValueError(
                f"{summary_path}: no enumeration_summary.json -- not a "
                "promoted landscape directory")
        summary = json.loads(summary_path.read_text())
        self._landscape_dir = landscape_dir
        self._summary = summary

        # --- gate 1: the cloud identity --------------------------------
        if not cloud_path.is_file():
            raise ValueError(f"{cloud_path}: the cloud file not found")
        got = sha256_file(cloud_path)
        want = str(summary.get("cloud_sha256"))
        if got != want:
            raise ValueError(
                f"{cloud_path}: cloud sha256 mismatch against the "
                f"landscape summary ({landscape_dir}): got {got}, "
                f"expected {want} -- not the certified cloud")
        self._cloud = np.load(cloud_path)
        self._cloud_sha256 = got

        # --- gate 2: the summary records THIS code's constants ---------
        summary_universe = summary.get("universe") or {}
        if summary_universe.get("row_order") != ga_space.ROW_ORDER:
            raise ValueError(
                f"{summary_path}: the summary's row_order "
                f"{summary_universe.get('row_order')!r} != the row code's "
                f"own constant {ga_space.ROW_ORDER!r} -- the landscape's "
                "row identity and the space's have drifted")
        if [list(s) for s in (summary_universe.get("sign_canon") or ())] != \
                [list(s) for s in ga_space.SIGN_CANON]:
            raise ValueError(
                f"{summary_path}: the summary's sign_canon "
                f"{summary_universe.get('sign_canon')!r} != the row "
                f"code's own constant "
                f"{[list(s) for s in ga_space.SIGN_CANON]}")

        # --- the fence-less space of the gated cloud --------------------
        # (alignment + serving arithmetic; the domain fence belongs to
        # the runner's search-space, not to the lookup adapter)
        self.space = ga_space.Space(
            self._cloud, d_min=float(summary_universe.get("d_min",
                                                          ga_space.D_MIN_A)),
            exclusions=())

        # --- load the artifacts + gate 3+4: certification, shapes -------
        arrays = {}
        self.artifact_sha256s = {}
        for name in ARTIFACT_NAMES:
            path = landscape_dir / f"{name}.npy"
            if not path.is_file():
                raise ValueError(f"{path}: landscape artifact missing")
            arr = np.load(path)
            arrays[name] = arr
            got_sha = sha256_file(path)
            self.artifact_sha256s[name] = got_sha
            rec = (summary.get("artifacts") or {}).get(name) or {}
            if rec.get("sha256") != got_sha:
                raise ValueError(
                    f"{path}: artifact sha256 mismatch against the "
                    f"summary's certification: got {got_sha}, expected "
                    f"{rec.get('sha256')} -- tampered or not the "
                    "promoted artifact")
            if list(arr.shape) != list(rec.get("shape") or ()):
                raise ValueError(
                    f"{path}: artifact shape {list(arr.shape)} != recorded "
                    f"{rec.get('shape')}")
            if str(arr.dtype) != str(rec.get("dtype")):
                raise ValueError(
                    f"{path}: artifact dtype {arr.dtype} != recorded "
                    f"{rec.get('dtype')}")

        f_art = arrays["f"]
        e_ts_art = arrays["e_ts"]
        e_r_art = arrays["e_r"]
        pair_art = arrays["pair_indices"]
        sign_art = arrays["sign_index"]

        n_rows = int(self.space.n_rows)
        want_rows = summary_universe.get("n_rows_post_floor")
        sizes = {"f": f_art.shape[0], "e_ts": e_ts_art.shape[0],
                 "e_r": e_r_art.shape[0],
                 "pair_indices": pair_art.shape[0],
                 "sign_index": sign_art.shape[0]}
        if any(v != n_rows for v in sizes.values()) or want_rows != n_rows:
            raise ValueError(
                f"{landscape_dir}: row-count mismatch: the space "
                f"rebuilds {n_rows} rows; the summary records "
                f"{want_rows}; the artifacts hold {sizes} -- refusing")

        # --- gate 5: the master alignment gate (full length) -------------
        exp_pair = np.stack(
            [np.repeat(self.space.pi, 4), np.repeat(self.space.pj, 4)],
            axis=1)
        if not np.array_equal(pair_art.astype(np.int64), exp_pair):
            bad = int(np.argmax(np.any(
                pair_art.astype(np.int64) != exp_pair, axis=1)))
            raise ValueError(
                f"{landscape_dir}: the master alignment gate FAILED: "
                f"pair_indices.npy row {bad} != the space's row "
                f"{bad} -- the artifact pair table and the space's "
                "pair table have drifted")
        exp_sign = np.tile(np.arange(4, dtype=np.int64), self.space.n_pairs)
        if not np.array_equal(sign_art.astype(np.int64), exp_sign):
            bad = int(np.argmax(sign_art.astype(np.int64) != exp_sign))
            raise ValueError(
                f"{landscape_dir}: the master alignment gate FAILED: "
                f"sign_index.npy row {bad} != row % 4 -- the artifact "
                "sign index and the row code have drifted")

        # --- gate 6: the NaN mask vs the domain-exclusion record ---------
        excl_rows = [int(r) for r in
                     ((summary.get("domain_rule") or {}).get("rows") or ())]
        nan_mask = np.isnan(f_art)
        if nan_mask.sum() != len(excl_rows) or \
                (np.flatnonzero(nan_mask).tolist() != excl_rows):
            raise ValueError(
                f"{landscape_dir}: the NaN mask of f.npy "
                f"({int(nan_mask.sum())} rows) does not mark exactly the "
                f"summary's domain-excluded rows {excl_rows} -- a value "
                "exists where none may, or a domain row lost its NaN")
        for other in (e_ts_art, e_r_art):
            if np.isnan(other).sum() != len(excl_rows) or \
                    (np.flatnonzero(np.isnan(other)).tolist() != excl_rows):
                raise ValueError(
                    f"{landscape_dir}: a value array's NaN marks are not "
                    "exactly the domain-excluded rows -- corrupt artifact")
        if not (np.isfinite(f_art[~nan_mask]).all()
                and np.isfinite(e_ts_art[~nan_mask]).all()
                and np.isfinite(e_r_art[~nan_mask]).all()):
            raise ValueError(
                f"{landscape_dir}: non-finite value OUTSIDE the "
                "domain-excluded rows -- corrupt artifact")

        # --- the excluded rows' identity coherence ------------------------
        for row in excl_rows:
            tup = self.space.decode(row)          # fence-blind by design
            if self.space._encode_core(*tup) != row:
                raise ValueError(
                    f"{landscape_dir}: domain-excluded row {row} does "
                    f"not round-trip to {tup} -- incoherent exclusion "
                    "record")

        self._f = f_art
        self._e_ts = e_ts_art
        self._e_r = e_r_art

        # --- bind the contract surface from the summary -------------------
        self.FITNESS_ID = str(summary.get("fitness_id"))
        self.PROVIDER_META = summary.get("provider_meta") or {}

    # -- provenance surface (for the run record) ---------------------------

    @property
    def landscape_dir(self):
        return self._landscape_dir

    @property
    def cloud_sha256(self):
        return self._cloud_sha256

    # -- the contract implementation ----------------------------------------

    def _eval_batch_full_impl(self, idx_i, idx_j, sign_i, sign_j, candidates):
        if not (candidates.shape == self._cloud.shape
                and np.array_equal(candidates, self._cloud)):
            raise ValueError(
                f"candidates is not the landscape's cloud "
                f"({self._cloud_sha256[:12]}...): the oracle serves the "
                "certified landscape of exactly one cloud; its row "
                "arithmetic is that cloud's")
        m = idx_i.size
        rows = np.empty(m, dtype=np.int64)
        for k in range(m):
            i, j = int(idx_i[k]), int(idx_j[k])
            si, sj = int(sign_i[k]), int(sign_j[k])
            row = self.space.encode(i, j, si, sj)
            if row < 0:
                raise ValueError(
                    f"configuration (idx_i={i}, idx_j={j}, sign_i={si:+d}, "
                    f"sign_j={sj:+d}) has no row in the certified "
                    "landscape (i == j, under the d_min floor, or fenced)")
            if np.isnan(self._f[row]):
                raise ValueError(
                    f"configuration (idx_i={i}, idx_j={j}, sign_i={si:+d}, "
                    f"sign_j={sj:+d}) is domain-excluded in the certified "
                    f"landscape (row {row}; no value exists -- outside "
                    "the fitness's domain of definition)")
            rows[k] = row
        return (self._f[rows].astype(np.float64),
                self._e_ts[rows].astype(np.float64),
                self._e_r[rows].astype(np.float64))


def landscape_oracle(landscape_dir, cloud_path):
    """-> a FRESH LandscapeOracle bound to one certified landscape (the
    contract surface is bound per landscape; construct per run -- the
    n_evals ledger is per-run state)."""
    return LandscapeOracle(landscape_dir, cloud_path)