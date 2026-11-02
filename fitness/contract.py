"""Fitness-provider contract (v2).

The fitness layer is a set of interchangeable *providers*. A provider is
a Python module that exposes exactly four things:

    FITNESS_ID      str   stable identifier recorded in run results
    PROVIDER_META   dict  JSON-serializable provenance (provider name,
                          constants, seeds, settings)
    eval_batch_full(idx_i, idx_j, sign_i, sign_j, candidates)
                          -> (f, e_ts, e_r)   the primitive
    eval_batch(idx_i, idx_j, sign_i, sign_j, candidates)
                          -> f                the black-box view

Input shape (vectorized over a batch of m configurations):

    idx_i, idx_j    1-D integer arrays, length m. Row indices into the
                    rows of `candidates` — the candidate-site coordinate
                    array, shape (n_sites, 3), Angstrom.
    sign_i, sign_j  1-D integer arrays, length m. Exactly +1 or -1.
    candidates      (n_sites, 3) real-numeric array (promoted to float64).

A configuration is the ordered tuple (idx_i, idx_j, sign_i, sign_j).

Outputs of eval_batch_full: three float64 arrays of length m —

    f     fitness, kcal/mol, lower = better
    e_ts  per-configuration component energy (transition state side)
    e_r   per-configuration component energy (reactant side)

such that f = e_ts - e_r (the identity should hold bitwise; 1e-12 is the
absolute tolerance). The components are carried alongside f so that
downstream diagnostics are pure post-processing — no later analysis may
ever require re-running anything.

eval_batch returns exactly the f of eval_batch_full, nothing else.
Searchers (the enumeration core, the GA) call ONLY eval_batch and treat
the provider as a black box. Anything that stores landscape artifacts
calls eval_batch_full.

Hard rules for every provider:

  * Batch independence: a row's result depends only on that row's
    configuration and on `candidates` — never on the other rows present
    or their order. Any internal pseudo-randomness must be keyed on the
    configuration identity, never on call order or batch composition.
  * Fail hard, no silent corrections: inputs are never coerced,
    corrected, or truncated. Mismatched input lengths, non-integer
    indices, indices outside [0, n_sites), signs outside {+1, -1}, and
    a candidates array that is not (n, 3) real-numeric each raise
    ValueError with a clear message.
  * Accounting: EVERY FITNESS CALL COUNTS; caching never reduces
    counts. Each input row of every successful eval_batch /
    eval_batch_full call counts as one evaluation, recorded at the API
    boundary in the provider's cumulative `n_evals` counter — internal
    caching (if any provider ever has it) must never suppress or reduce
    that count. A call that raises counts nothing (no configuration
    was evaluated). n_evals counts evaluated configurations (rows), not
    calls, so batching granularity cannot game the evaluation metric.

Mechanics (implementation detail, not contract): `FitnessProvider`
below is the shared boundary — validation, accounting, and the
eval_batch wrapper. Concrete providers subclass it and implement
`_eval_batch_full_impl` (receiving validated/normalized inputs); the
provider module then binds the four contract names to a module-level
instance (exposed as `provider`; read `provider.n_evals` for the
accounting counter).
"""

import numpy as np

__all__ = ["validate_batch_inputs", "FitnessProvider"]


def _as_int_array(x, name):
    """Normalize x to a 1-D int64 array; fail hard on anything else."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D integer array, got shape {arr.shape}"
        )
    if arr.size == 0:
        return arr.astype(np.int64)
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be integers (integer dtype), got dtype {arr.dtype!s}"
        )
    return arr.astype(np.int64)


def _as_sign_array(x, name):
    """Normalize x to a 1-D int64 sign array; values must be +1 or -1."""
    arr = _as_int_array(x, name)
    bad = arr[(arr != 1) & (arr != -1)]
    if bad.size:
        raise ValueError(f"{name} values must be +1 or -1, got {bad[0]}")
    return arr


def validate_batch_inputs(idx_i, idx_j, sign_i, sign_j, candidates):
    """Validate and normalize a fitness batch.

    Returns (idx_i, idx_j, sign_i, sign_j, candidates) as int64 1-D
    arrays of common length m and a contiguous float64 (n_sites, 3)
    array. Raises ValueError on any malformed input — never corrects.
    """
    idx_i = _as_int_array(idx_i, "idx_i")
    idx_j = _as_int_array(idx_j, "idx_j")
    sign_i = _as_sign_array(sign_i, "sign_i")
    sign_j = _as_sign_array(sign_j, "sign_j")

    m = idx_i.size
    for name, arr in (("idx_j", idx_j), ("sign_i", sign_i), ("sign_j", sign_j)):
        if arr.size != m:
            raise ValueError(
                f"mismatched input lengths: idx_i has {m} rows, "
                f"{name} has {arr.size}"
            )

    cand = np.asarray(candidates)
    if cand.ndim != 2 or cand.shape[1] != 3:
        raise ValueError(
            f"candidates must be an (n, 3) array, got shape {cand.shape}"
        )
    if cand.dtype.kind not in "ifu":
        raise ValueError(
            f"candidates must be a real-numeric array, got dtype {cand.dtype!s}"
        )
    cand = np.ascontiguousarray(cand, dtype=np.float64)

    if m > 0:
        n_sites = cand.shape[0]
        for name, arr in (("idx_i", idx_i), ("idx_j", idx_j)):
            lo, hi = int(arr.min()), int(arr.max())
            if lo < 0 or hi >= n_sites:
                raise ValueError(
                    f"{name} values must be in [0, {n_sites}) "
                    f"(rows of candidates), got [{lo}, {hi}]"
                )

    return idx_i, idx_j, sign_i, sign_j, cand


class FitnessProvider:
    """Shared boundary mechanics for fitness providers.

    Subclasses define FITNESS_ID and PROVIDER_META and implement
    `_eval_batch_full_impl(idx_i, idx_j, sign_i, sign_j, candidates)`
    on validated, normalized inputs. The base class provides the
    contract surface: input validation, per-row accounting at the API
    boundary, and the eval_batch wrapper.
    """

    FITNESS_ID: str = ""
    PROVIDER_META: dict = {}

    def __init__(self):
        self.n_evals = 0

    def eval_batch_full(self, idx_i, idx_j, sign_i, sign_j, candidates):
        """-> (f, e_ts, e_r): float64 arrays, one entry per input row."""
        idx_i, idx_j, sign_i, sign_j, cand = validate_batch_inputs(
            idx_i, idx_j, sign_i, sign_j, candidates
        )
        f, e_ts, e_r = self._eval_batch_full_impl(
            idx_i, idx_j, sign_i, sign_j, cand
        )
        f = np.asarray(f, dtype=np.float64)
        e_ts = np.asarray(e_ts, dtype=np.float64)
        e_r = np.asarray(e_r, dtype=np.float64)
        m = idx_i.size
        if f.shape != (m,) or e_ts.shape != (m,) or e_r.shape != (m,):
            raise RuntimeError(
                f"provider {self.FITNESS_ID!r} returned wrong output "
                f"shapes: f {f.shape}, e_ts {e_ts.shape}, e_r {e_r.shape} "
                f"(expected ({m},) each)"
            )
        # Accounting: every fitness call counts; caching never reduces
        # counts. Counted at the API boundary, per row, only on success.
        self.n_evals += m
        return f, e_ts, e_r

    def eval_batch(self, idx_i, idx_j, sign_i, sign_j, candidates):
        """-> f: exactly the f of eval_batch_full (black-box view)."""
        f, _ts, _r = self.eval_batch_full(
            idx_i, idx_j, sign_i, sign_j, candidates
        )
        return f

    def _eval_batch_full_impl(self, idx_i, idx_j, sign_i, sign_j, candidates):
        raise NotImplementedError(
            "providers must implement _eval_batch_full_impl"
        )