"""Synthetic stub fitness provider (plumbing verification ONLY).

NO SCIENTIFIC MEANING: this is a synthetic landscape with a planted
unique global optimum, used to verify search plumbing without any QM.
No conclusion about the real system may ever be drawn from stub runs.

Semantics (all fixed; see PROVIDER_META for the serialized form):

  * Base value per configuration: deterministic hash of the ordered
    configuration tuple (idx_i, idx_j, sign_i, sign_j) with STUB_SEED
    into [0, 10). Scheme (splitmix64, Vigna's public-domain mixer —
    all arithmetic uint64, fully vectorized):

        h = STUB_SEED
        for field in (idx_i, idx_j, sign_i > 0, sign_j > 0):   # as uint64
            h = mix64(h XOR mix64(field))
        base = ((h >> 11) / 2**53) * 10.0

    where mix64(x) = splitmix64 finalizer with the golden-ratio
    increment folded in:

        x = x + 0x9E3779B97F4A7C15
        x = (x XOR (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x XOR (x >> 27)) * 0x94D049BB133111EB
        x = x XOR (x >> 31)

    (h >> 11) keeps the top 53 bits; division by 2**53 is exact, so base
    lands in [0, 10) after one rounding. The value depends only on the
    configuration identity — never on call order, batch composition, or
    candidate positions (so different clouds give the same base; only
    the coupling term sees positions). Hash collisions between distinct
    tuples are possible in principle and harmless for a synthetic stub.
  * Non-separable coupling: +3.0 kcal/mol when the two candidate
    positions are strictly less than 1.0 Angstrom apart (Euclidean
    distance from `candidates`), else +0.0. This blocks separability
    shortcuts. Identical indices (distance 0) are coupled.
  * Planted unique global minimum: the configuration (123, 456, +1, -1)
    evaluates to exactly -5.000000 by override (not offset) — strictly
    below the attainable range of base+coupling, which is [0, 13). The
    override applies to that exact ordered tuple only; every other
    configuration keeps its base+coupling value.
  * Components: e_ts = f/2, e_r = -f/2. Synthetic but bitwise
    consistent: e_ts - e_r == f exactly (halving and doubling are
    exact in binary floating point).

Accounting follows the contract: every fitness call counts; caching
never reduces counts (the stub has no cache; the boundary counts every
row of every successful call into `provider.n_evals`).
"""

import numpy as np

from fitness import contract

STUB_SEED = 20260917

# splitmix64 constants (public domain, Sebastiano Vigna).
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_TWO_P53 = 9007199254740992.0  # 2**53

# The planted unique global optimum (ordered tuple) and its exact value.
PLANTED_IDX_I = 123
PLANTED_IDX_J = 456
PLANTED_SIGN_I = +1
PLANTED_SIGN_J = -1
PLANTED_F = -5.0

# Non-separable coupling rule.
COUPLING_PENALTY = 3.0
COUPLING_DIST_A = 1.0


def _mix64(x):
    """splitmix64 step: golden-ratio increment + finalizer. uint64 in/out."""
    x = x + _GAMMA
    x = (x ^ (x >> _S30)) * _M1
    x = (x ^ (x >> _S27)) * _M2
    return x ^ (x >> _S31)


def _base_values(idx_i, idx_j, sign_i, sign_j):
    """Deterministic base value in [0, 10) per configuration row.

    Keyed on the ordered configuration identity + STUB_SEED only (see
    module docstring for the exact scheme). Never on call order.
    """
    h = np.full(idx_i.shape, STUB_SEED, dtype=np.uint64)
    for field in (
        idx_i.astype(np.uint64),
        idx_j.astype(np.uint64),
        (sign_i > 0).astype(np.uint64),  # +1 -> 1, -1 -> 0
        (sign_j > 0).astype(np.uint64),
    ):
        h = _mix64(h ^ _mix64(field))
    return ((h >> _S11).astype(np.float64) / _TWO_P53) * 10.0


class StubProvider(contract.FitnessProvider):
    FITNESS_ID = "stub_v2"
    PROVIDER_META = {
        "provider": "stub",
        "fitness_id": FITNESS_ID,
        "description": (
            "synthetic reference provider with a planted unique global "
            "optimum; plumbing verification only, no scientific meaning"
        ),
        "units": "kcal/mol",
        "stub_seed": STUB_SEED,
        "base_value": {
            "range": "[0.0, 10.0)",
            "scheme": (
                "splitmix64 absorb of (stub_seed, idx_i, idx_j, "
                "sign_i>0, sign_j>0); base = ((h >> 11) / 2**53) * 10.0"
            ),
        },
        "coupling": {
            "penalty_kcal_mol": COUPLING_PENALTY,
            "threshold_angstrom": COUPLING_DIST_A,
            "rule": "penalty applies when pairwise distance is strictly < threshold",
        },
        "planted_optimum": {
            "idx_i": PLANTED_IDX_I,
            "idx_j": PLANTED_IDX_J,
            "sign_i": PLANTED_SIGN_I,
            "sign_j": PLANTED_SIGN_J,
            "f": PLANTED_F,
        },
        "components": "e_ts = f/2, e_r = -f/2 (synthetic; e_ts - e_r == f bitwise)",
    }

    def _eval_batch_full_impl(self, idx_i, idx_j, sign_i, sign_j, candidates):
        base = _base_values(idx_i, idx_j, sign_i, sign_j)
        d = np.linalg.norm(
            candidates[idx_i] - candidates[idx_j], axis=1
        )
        f = base + np.where(d < COUPLING_DIST_A, COUPLING_PENALTY, 0.0)
        planted = (
            (idx_i == PLANTED_IDX_I)
            & (idx_j == PLANTED_IDX_J)
            & (sign_i == PLANTED_SIGN_I)
            & (sign_j == PLANTED_SIGN_J)
        )
        f = np.where(planted, PLANTED_F, f)
        e_ts = f * 0.5
        e_r = -e_ts
        return f, e_ts, e_r


provider = StubProvider()

# Contract surface (see contract.py): the four names every provider
# module exposes, bound to the module-level provider instance.
FITNESS_ID = provider.FITNESS_ID
PROVIDER_META = provider.PROVIDER_META
eval_batch_full = provider.eval_batch_full
eval_batch = provider.eval_batch