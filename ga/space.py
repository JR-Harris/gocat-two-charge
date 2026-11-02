"""The GA's configuration space (the discrete memetic GA).

Space is the ONLY coordinates object the engine consults. It is an
EXACT, self-contained replica of the certified universe arithmetic
(exhaustive_search/enumerate.py::build_universe -- same formula, same
dtype, same order) plus the row code that binds an individual's
identity to the landscape's row identity. ga/ never imports
exhaustive_search; the equality of the two arithmetics is pinned by the
master-gate test against the certified artifact itself
(ga/test_ga.py::test_01).

Contents (all read-only; the instance is immutable):

  * THE PAIR TABLE: `pi, pj = np.triu_indices(n, k=1)`;
    `dist = np.linalg.norm(cand[pi] - cand[pj], axis=1)` (float64);
    `keep = dist >= d_min`; the kept pairs in row-major triu order.
    At the canonical cloud (candidates_v2, n = 928, d_min = 1.0 A):
    5,345 pairs / 21,380 configs excluded; 424,783 kept pairs;
    1,699,132 rows.

  * THE ROW CODE: `row = 4 * pair_rank + s` with `s = 2*(s1 < 0) +
    (s2 < 0)` -- the canonical sign index over the canon order
    SIGN_CANON = [(+, +), (+, -), (-, +), (-, -)]. An individual IS a
    canonical row code. encode(i, j, s1, s2) canonicalizes
    out-of-order pairs by moving the signs WITH their sites and
    returns -1 for anything invalid: i == j, a d_min-floor violation,
    a fenced configuration, out-of-range site indices, or a
    non-+-1 sign. decode(row) -> (i, j, s1, s2) round-trips every row
    exactly and is fence-blind by design (the certified landscape's
    own artifacts must decode at every row, including the
    domain-excluded one).

  * THE NN TABLE: per site, its nn_k nearest OTHER sites (Euclidean;
    stable argsort; index order on ties). Per-SITE by design: pair
    admissibility is applied via encode() wherever a site moves --
    NN entries nearer than d_min to the standing partner simply fail
    the validity filter.

  * THE FENCE: `exclusions` = canonical (i, j, s1, s2) tuples outside
    the fitness's domain (the runner supplies them from the
    landscape's domain-exclusions record, or the caller in stub/xtb
    mode). Fence DATA only: no fitness value ever enters the Space.
    An exclusion whose configuration fails to encode raises at
    construction -- an incoherent fence is a bug, never a silent
    empty set. encode() of a fenced configuration returns -1; decode()
    never consults the fence.

d_min here is a METHOD constant (1.0 A, centroids), not a parameter.
"""

import numpy as np

__all__ = ["Space", "D_MIN_A", "SIGN_CANON", "ROW_ORDER"]

# The frozen briefing conventions (identical to the enumeration's).
D_MIN_A = 1.0                       # the minimum charge separation floor
SIGN_CANON = ((+1, +1), (+1, -1), (-1, +1), (-1, -1))  # canon order
ROW_ORDER = ("pair-major post-floor: row = 4*p + s; p = the "
             "np.triu_indices pair rank after the d_min floor "
             "(row-major), s = the sign index")

# sign index s -> (s1, s2): 0:(+,+), 1:(+,-), 2:(-,+), 3:(-,-)
_S1 = (+1, +1, -1, -1)
_S2 = (+1, -1, +1, -1)


class Space:
    """The immutable configuration space of one candidate cloud.

    Built ONCE per run and shared by the engine (and by the oracle
    adapter's alignment gates). See the module docstring for the
    arithmetic contract.
    """

    def __init__(self, candidates, d_min=D_MIN_A, exclusions=(), nn_k=8):
        cand = np.asarray(candidates)
        if cand.ndim != 2 or cand.shape[1] != 3:
            raise ValueError(f"candidates must be (n, 3), got {cand.shape}")
        if cand.dtype != np.float64:
            # Exact-arithmetic requirement: the space's distances and
            # the fitness's charge positions must be the same float64
            # values the certified artifacts were built from.
            raise ValueError(
                f"candidates must be float64, got {cand.dtype}")
        n = int(cand.shape[0])
        if n < 2:
            raise ValueError(f"need >= 2 candidate points, got {n}")
        if not (np.isscalar(d_min) and float(d_min) >= 0.0):
            raise ValueError(f"d_min must be a float >= 0, got {d_min!r}")
        if not (isinstance(nn_k, int) and not isinstance(nn_k, bool)
                and 1 <= nn_k <= n - 1):
            raise ValueError(
                f"nn_k must be an int in [1, n - 1] (n = {n}), got {nn_k!r}")

        self._cand = np.array(cand, dtype=np.float64)   # defensive copy
        self._n = n
        self._d_min = float(d_min)
        self._nn_k = int(nn_k)

        # --- the pair table (EXACT build_universe arithmetic) -----------
        pi, pj = np.triu_indices(n, k=1)          # row-major upper triangle
        n_pairs_total = int(pi.size)
        if 4 * n_pairs_total != (n * (n - 1)) // 2 * 4:
            raise RuntimeError("space assertion failed: 4*C(n,2) mismatch")
        dist = np.linalg.norm(self._cand[pi] - self._cand[pj], axis=1)
        keep = dist >= self._d_min
        self._n_pairs_total = n_pairs_total
        self._n_pairs_excluded = int((~keep).sum())
        self._n_configs_excluded = 4 * self._n_pairs_excluded
        self._pi = np.ascontiguousarray(pi[keep], dtype=np.int64)
        self._pj = np.ascontiguousarray(pj[keep], dtype=np.int64)
        self._pair_dist = np.ascontiguousarray(dist[keep], dtype=np.float64)
        self._n_rows = 4 * int(self._pi.size)

        # kept pair -> rank lookup (-1 elsewhere): the d_min gate of
        # encode(), and the row arithmetic's base
        self._pair_rank = np.full((n, n), -1, dtype=np.int64)
        self._pair_rank[self._pi, self._pj] = np.arange(int(self._pi.size),
                                                        dtype=np.int64)

        # --- the NN table: per site, the nn_k nearest OTHER sites ------
        d2 = np.linalg.norm(
            self._cand[:, None, :] - self._cand[None, :, :], axis=2)
        np.fill_diagonal(d2, np.inf)              # never self
        order = np.argsort(d2, axis=1, kind="stable")
        self._nn = np.ascontiguousarray(order[:, : self._nn_k], dtype=np.int64)
        del d2, order

        # --- the fence ---------------------------------------------------
        rows = set()
        canon = []
        for t in exclusions:
            row = self._encode_core(t[0], t[1], t[2], t[3])
            if row < 0:
                raise ValueError(
                    f"incoherent fence: exclusion {tuple(t)} does not "
                    f"encode to a valid row of this space (it is i==j, "
                    f"under the d_min floor, or has a non-+-1 sign) -- "
                    "the fence must list configurations INSIDE the "
                    "universe that lie outside the fitness's domain")
            if row in rows:
                raise ValueError(
                    f"incoherent fence: duplicate exclusion "
                    f"{tuple(t)} (row {row})")
            rows.add(int(row))
            canon.append(self.decode(row))   # the canonical form
        self._excluded_rows = frozenset(rows)
        self._exclusions = tuple(canon)

        # --- immutability -------------------------------------------------
        for arr in (self._cand, self._pi, self._pj, self._pair_dist,
                    self._pair_rank, self._nn):
            arr.setflags(write=False)

    # -- readonly surface ---------------------------------------------------

    @property
    def candidates(self):
        """-> the (n, 3) float64 cloud (read-only; bitwise the input)."""
        return self._cand

    @property
    def n(self):
        return self._n

    @property
    def d_min(self):
        return self._d_min

    @property
    def nn_k(self):
        return self._nn_k

    @property
    def n_pairs_total(self):
        return self._n_pairs_total

    @property
    def n_configs_total(self):
        return 4 * self._n_pairs_total

    @property
    def n_pairs_excluded(self):
        return self._n_pairs_excluded

    @property
    def n_configs_excluded(self):
        return self._n_configs_excluded

    @property
    def n_pairs(self):
        """-> the KEPT pair count (the post-floor pair table's rows)."""
        return int(self._pi.size)

    @property
    def n_rows(self):
        """-> 4 * kept pairs (the landscape's row count)."""
        return self._n_rows

    @property
    def pi(self):
        """-> int64 (n_pairs,) kept-pair site-i indices (read-only)."""
        return self._pi

    @property
    def pj(self):
        """-> int64 (n_pairs,) kept-pair site-j indices (read-only)."""
        return self._pj

    @property
    def pair_dist(self):
        """-> float64 (n_pairs,) kept-pair distances (read-only)."""
        return self._pair_dist

    @property
    def nn(self):
        """-> int64 (n, nn_k) per-site nearest-neighbour table
        (read-only; per-SITE, no pair admissibility applied)."""
        return self._nn

    @property
    def exclusions(self):
        """-> the canonical exclusion tuples (i, j, s1, s2), fence data
        only -- never a fitness value."""
        return self._exclusions

    @property
    def n_exclusions(self):
        return len(self._exclusions)

    # -- the row code --------------------------------------------------------

    def _encode_core(self, i, j, s1, s2):
        """Canonical row of (i, j, s1, s2) WITHOUT the fence check:
        -1 on i == j, a d_min-floor violation, out-of-range sites, or a
        non-+-1 sign. The fence rides on top in encode()."""
        try:
            i = int(i); j = int(j); s1 = int(s1); s2 = int(s2)
        except (TypeError, ValueError):
            return -1
        if s1 not in (1, -1) or s2 not in (1, -1):
            return -1
        if i == j:
            return -1
        if not (0 <= i < self._n and 0 <= j < self._n):
            return -1
        if i > j:
            # canonicalize: the signs move WITH their sites
            i, j, s1, s2 = j, i, s2, s1
        p = self._pair_rank[i, j]          # -1 when under the d_min floor
        if p < 0:
            return -1
        s = 2 * (s1 < 0) + (s2 < 0)         # the canonical sign index
        return int(p) * 4 + int(s)

    def encode(self, i, j, s1, s2):
        """-> the canonical row code of the configuration, or -1 when
        invalid (i == j, a d_min-floor violation, a FENCED
        configuration, out-of-range sites, or a non-+-1 sign).

        Out-of-order pairs canonicalize by moving the signs WITH their
        sites: encode(j, i, s2, s1) == encode(i, j, s1, s2).
        """
        row = self._encode_core(i, j, s1, s2)
        if row < 0:
            return -1
        if row in self._excluded_rows:
            return -1                      # fenced: outside the domain
        return row

    def decode(self, row):
        """-> (i, j, s1, s2) for any row in [0, n_rows); round-trips
        encode exactly. FENCE-BLIND by design: the certified landscape
        must decode at every row, including a domain-excluded one
        (whose NaN value lives outside the fitness's domain)."""
        row = int(row)
        if not (0 <= row < self._n_rows):
            raise ValueError(
                f"decode: row {row} outside [0, {self._n_rows})")
        p, s = divmod(row, 4)
        return (int(self._pi[p]), int(self._pj[p]), _S1[s], _S2[s])

    def decode_rows(self, rows):
        """-> (idx_i, idx_j, sign_i, sign_j) int64 arrays for a 1-D
        array of valid rows (the fitness callable's batch signature)."""
        rows = np.asarray(rows, dtype=np.int64)
        p = rows // 4
        s = rows % 4
        sign_i = (1 - 2 * (s // 2)).astype(np.int64)
        sign_j = (1 - 2 * (s % 2)).astype(np.int64)
        return self._pi[p], self._pj[p], sign_i, sign_j

    def row_fenced(self, row):
        """-> True when a (valid-core) row is excluded by the fence.
        The engine's uniform init draw produces rows that already pass
        i != j and the d_min floor by construction, so the fence is
        the only rejection it can hit."""
        return int(row) in self._excluded_rows

    def __repr__(self):
        return (f"Space(n={self._n}, d_min={self._d_min}, "
                f"pairs={self.n_pairs}/{self.n_pairs_total}, "
                f"rows={self._n_rows}, nn_k={self._nn_k}, "
                f"exclusions={len(self._exclusions)})")