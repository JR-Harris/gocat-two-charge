#!/usr/bin/env python3
"""Self-tests for the GA stage (work item: the discrete memetic GA).

Plain-python test file (no framework); run from the repo root:
    venv/bin/python ga/test_ga.py

The definition of done, in the briefing's order:

 1. the row-code master gate (full length) -- the space module's pair
    table equals the certified artifact's, bitwise, over all
    1,699,132 rows; encode/decode round-trips every row;
 2. the oracle provider gates (init pass + negative tests each);
 3. the engine smoke on two providers and two bars -- the hash-noise
    stub (completeness + accounting only: on a structureless hash
    landscape any planted-optimum recovery is direct-sampling luck,
    per-query hit rate ~5.9e-7; recovery there is NOT a gate) and THE
    SEARCH BAR: a structured planted-basin provider defined in this
    bundle (f = BOWL_A * (|c_i - p_i| + |c_j - p_j|) + noise, planted
    at the two maximally-separated sites, lexicographic tie-break,
    signs (+1, -1), a strictly-unique negative minimum; the landscape
    is WALKABLE -- jitter + descent must work) must recover the
    planted optimum in >= 9/10 seeds at G = 450, pop 50, smoke
    defaults. Nothing from either arm is reported anywhere.
 4. the accounting gates (per-run invariants + the trajectory
    checkpoints re-derived bitwise from a direct re-query of each
    checkpoint's best config, on the stub AND the oracle lookup);
 5. the reproducibility gate (run twice -> bitwise-identical record,
    wall-clock fields aside);
 6. the fence gates (the canonical exclusion never sampled/submitted;
    its unfenced encode equals the artifact's excluded row -- a direct
    cross-artifact check; the synthetic fence: exclude the basin
    optimum -> never submitted, run completes; the canonical
    landscape's exclusions count == 1);
 7. the descent bookkeeping (a converged elite costs 0 queries at
    later generations -- per-generation ledger deltas; descent never
    leaves an elite worse than it found it);
 8. the live-path demonstration -- env-gated: needs the xtb binary
    AND  GOCAT_GA_LIVE=1  (11 real runs x ~3-6 min: sized within a
    work day as a unit). >= 10 restarts through fitness/xtb.py
    (fresh instance per run, real CLI provider, OMP=1 as shipped,
    memo ON, G = 20, pop 50), each green on the accounting; >= 20
    repeat submissions of the same config through a FRESH provider
    return bitwise-identical f (binary-gated, seconds); and THE
    TRANSFER DEMONSTRATION: one further live restart, same seed,
    memo OFF -- trajectory and queries bitwise-identical to the
    memo-ON run (values identical => identical GA; only the wall cost
    and submitted differ).
 9. the negative gates (a failing provider aborts fail-hard naming
    the failing config, with no retry logic anywhere -- grep-able
    absence; a provider double raising on eval_batch_full proves the
    engine never calls it; ga/ never imports exhaustive_search).

Deliverable artifacts (tests 3 + 8) are written to runs/ga/ -- the
repository's run tree; every other test writes under a tmp dir.

Read-only imports: fitness/, exhaustive_search/, cloud/, qm/,
alignment/, hpc/ are never modified; the landscape artifacts are never
written.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import hashlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fitness import contract
from fitness import stub as stub_module
from ga import ga as ga_mod
from ga import space as ga_space
from ga.oracle_provider import landscape_oracle
from ga.params import GAParams, get_param_set
import run_ga as runner

REPO = Path(__file__).resolve().parent.parent

CANONICAL_CLOUD = REPO / "data" / "derived" / "candidates_v2.npy"
CANONICAL_SHA256 = (
    "084e48a6b5267900a995ba2d4f0412f77f5bb13446ba74eafaf474c644349e71")
LANDSCAPE_DIR = REPO / "data" / "derived" / "enumeration_v2"
POSTFLOOR_ROWS = 1699132
KEPT_PAIRS = 424783
EXCLUDED_PAIRS = 5345
EXCLUDED_CONFIGS = 21380
EXCLUDED_ROW = 522045
EXCLUDED_CONFIG = (154, 833, 1, -1)
OUT_ROOT_DELIVERABLE = REPO / "runs" / "ga"

SMOKE = get_param_set("smoke")


# --------------------------------------------------------------------------
# the structured planted-basin provider (defined INSIDE the test bundle;
# non-scientific, never reported)
# --------------------------------------------------------------------------

BASIN_GRID = 8
BASIN_SPACING = 2.0
BASIN_CLOUD = None          # built once, below
BASIN_PLANTED_ROW = None    # computed once, below
BASIN_PLANTED_PAIR = None

BOWL_A = 5.0                # kcal/mol per Angstrom of position lag
BASIN_NOISE = 0.05          # hash-noise scale (walkable: tiny vs the bowl)
BASIN_NOISE_SEED = 0x9E3779B1
BASIN_PLANTED_F = -10.0     # strictly below every other attainable value:
                            # the bowl + noise sum is >= 0 everywhere else


def _basin_cloud():
    """512 sites on an 8x8x8 lattice, 2.0 A spacing (every pairwise
    distance >= 2.0 A: no d_min exclusions; the NN rows are the +-2.0
    axis/diagonal moves a walkable descent needs)."""
    axes = [np.arange(BASIN_GRID) * BASIN_SPACING] * 3
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return np.ascontiguousarray(g.reshape(-1, 3), dtype=np.float64)


def _basin_planted_pair(cloud):
    """The two maximally-separated sites; np.argmax over the row-major
    triu distances takes the FIRST maximizer = the lexicographically
    smallest (i, j) on ties."""
    pi, pj = np.triu_indices(cloud.shape[0], k=1)
    d = np.linalg.norm(cloud[pi] - cloud[pj], axis=1)
    k = int(np.argmax(d))
    return int(pi[k]), int(pj[k]), float(d[k])


_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GAMMA = np.uint64(0x9E3779B97F4A7C15)


def _mix64(x):
    """splitmix64 step (Vigna's public-domain finalizer; the stub's
    scheme): golden-ratio increment + the two multipliers + xorshift."""
    x = x + _GAMMA
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def _basin_noise(idx_i, idx_j, sign_i, sign_j):
    """Deterministic hash noise in [0, BASIN_NOISE), keyed on the
    ordered configuration identity + BASIN_NOISE_SEED only (the same
    splitmix64 absorb the stub documents)."""
    h = np.full(np.asarray(idx_i).shape, BASIN_NOISE_SEED, dtype=np.uint64)
    for field in (
            np.asarray(idx_i).astype(np.uint64),
            np.asarray(idx_j).astype(np.uint64),
            (np.asarray(sign_i) > 0).astype(np.uint64),
            (np.asarray(sign_j) > 0).astype(np.uint64)):
        h = _mix64(h ^ _mix64(field))
    return ((h >> np.uint64(11)).astype(np.float64) / 2.0 ** 53) * BASIN_NOISE


class BasinProvider(contract.FitnessProvider):
    """A WALKABLE synthesized landscape: f = BOWL_A * (|c_i - p_i| +
    |c_j - p_j|) + hash noise, planted at the canonical config
    (p_i, p_j, +1, -1) of the two maximally-separated sites, overridden
    to a strictly-unique negative minimum. Components: e_ts = f/2,
    e_r = -f/2 (bitwise consistent)."""

    FITNESS_ID = "basin_planted_v1"
    PROVIDER_META = {
        "provider": "basin_planted_v1",
        "description": (
            "TEST-BUNDLE-ONLY synthetic landscape (L1 walkable bowl on "
            "the two maximally-separated sites + tiny hash noise, a "
            "planted strictly-unique negative minimum); never reported"),
        "units": "kcal/mol",
        "bowl_a": BOWL_A,
        "noise": {"scale": BASIN_NOISE, "seed": BASIN_NOISE_SEED,
                  "scheme": ("splitmix64 absorb of (seed, idx_i, idx_j, "
                             "sign_i>0, sign_j>0) / 2**53, scaled")},
        "planted": {"f": BASIN_PLANTED_F,
                    "signs": [1, -1],
                    "pair_rule": "max pair distance; lexicographic "
                                 "tie-break"},
    }

    def __init__(self):
        super().__init__()
        self._cloud = BASIN_CLOUD
        self._pi_p, self._pj_p = (BASIN_PLANTED_PAIR[0], BASIN_PLANTED_PAIR[1])

    def _eval_batch_full_impl(self, idx_i, idx_j, sign_i, sign_j,
                              candidates):
        di = np.linalg.norm(candidates[idx_i] - candidates[self._pi_p],
                            axis=1)
        dj = np.linalg.norm(candidates[idx_j] - candidates[self._pj_p],
                            axis=1)
        f = BOWL_A * (di + dj) + _basin_noise(idx_i, idx_j, sign_i, sign_j)
        # the planted optimum: the CANONICAL config (p_i, p_j, +1, -1)
        # (a reversed submission canonicalizes to the same identity)
        ii, jj = np.asarray(idx_i), np.asarray(idx_j)
        s1, s2 = np.asarray(sign_i), np.asarray(sign_j)
        swap = ii > jj
        c1 = np.where(swap, jj, ii)
        c2 = np.where(swap, ii, jj)
        g1 = np.where(swap, s2, s1)
        g2 = np.where(swap, s1, s2)
        planted = ((c1 == self._pi_p) & (c2 == self._pj_p)
                   & (g1 == 1) & (g2 == -1))
        f = np.where(planted, BASIN_PLANTED_F, f)
        e_ts = f * 0.5
        e_r = -e_ts
        return f, e_ts, e_r


BASIN_CLOUD = _basin_cloud()
_pI, _pJ, _pD = _basin_planted_pair(BASIN_CLOUD)
BASIN_PLANTED_PAIR = (_pI, _pJ, _pD)


# --------------------------------------------------------------------------
# shared harness
# --------------------------------------------------------------------------

def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recorder:
    """A transparent submit-side recorder around a fitness callable:
    logs every batch (canonical rows, f values) in order, and can
    forbid rows (the fence never-submitted assert)."""

    def __init__(self, fitness, space, forbid_rows=()):
        self._fitness = fitness
        self._space = space
        self._forbid = frozenset(int(r) for r in forbid_rows)
        self.batches = []          # (rows list, f list) per call
        self.n_rows = 0

    def __call__(self, idx_i, idx_j, sign_i, sign_j, candidates):
        f = self._fitness(idx_i, idx_j, sign_i, sign_j, candidates)
        rows = [self._space.encode(int(a), int(b), int(x), int(y))
                for a, b, x, y in zip(idx_i, idx_j, sign_i, sign_j)]
        for r in rows:
            assert r >= 0, f"recorder: an invalid config reached the " \
                           f"provider: {r}"
            assert r not in self._forbid, \
                f"recorder: the fenced row {r} was SUBMITTED"
        self.batches.append((rows, [float(v) for v in f]))
        self.n_rows += len(rows)
        return f


def _accounting_asserts(result, provider=None, memo=True):
    """The per-run accounting identities (the frozen 02 section-6
    semantics)."""
    c = result.as_dict()["counts"]
    assert c["queries"] == c["submitted"] + c["memo_hits"]
    assert c["unique"] <= c["queries"]
    if memo:
        assert c["unique"] == c["submitted"], \
            f"memo on: unique {c['unique']} != submitted {c['submitted']}"
    else:
        assert c["memo_hits"] == 0 and c["queries"] == c["submitted"]
    if provider is not None:
        assert int(provider.n_evals) == c["submitted"], \
            f"provider.n_evals {provider.n_evals} != submitted " \
            f"{c['submitted']}"
    if c["n_evals_provider"] is not None:
        assert c["n_evals_provider"] == c["submitted"]


def _checkpoint_schedule():
    """The engine's in-code trajectory schedule as an independent
    consumer (test 4's re-derivation walks it side by side)."""
    base = [100, 300, 1000, 3000, 10000]
    state = {"i": 0, "next_mult": 2}

    def consume(count):
        due = []
        while True:
            if state["i"] < len(base):
                c = base[state["i"]]
            else:
                c = state["next_mult"] * 10000
                base.append(c)
                state["next_mult"] += 1
            if c <= count:
                due.append(c)
                state["i"] += 1
            else:
                return due

    return consume


def _replay_trajectory(recorder, space):
    """Re-derive (evaluations, best_f, best_row) per batch from a
    memo-OFF recorder: identical to the engine's best-so-far rule."""
    consume = _checkpoint_schedule()
    entries = []
    best_f = float("inf")
    best_row = -1
    cum = 0
    for rows, fs in recorder.batches:
        cum += len(rows)
        for r, v in zip(rows, fs):
            if v < best_f or (v == best_f and r < best_row):
                best_f = v
                best_row = r
        for _c in consume(cum):
            entries.append((cum, best_f, best_row))
    return entries


# oracle negative-gate fixtures (tmp copies of the certified artifact)

def _landscape_fixture(mutate=None):
    tmp = Path(tempfile.mkdtemp(prefix="ga_oracle_fixture_"))
    for name in ("f", "e_ts", "e_r", "pair_indices", "sign_index"):
        shutil.copy(LANDSCAPE_DIR / f"{name}.npy", tmp / f"{name}.npy")
    shutil.copy(LANDSCAPE_DIR / "enumeration_summary.json",
                tmp / "enumeration_summary.json")
    if mutate is not None:
        mutate(tmp)
    return tmp


def _fix_wrong_cloud_sha(tmp):
    s = json.loads((tmp / "enumeration_summary.json").read_text())
    s["cloud_sha256"] = "0" * 64
    (tmp / "enumeration_summary.json").write_text(
        json.dumps(s, indent=1) + "\n")


def _fix_rows(tmp):
    s = json.loads((tmp / "enumeration_summary.json").read_text())
    s["universe"]["n_rows_post_floor"] = POSTFLOOR_ROWS - 1
    (tmp / "enumeration_summary.json").write_text(
        json.dumps(s, indent=1) + "\n")


def _f_artifact_shas(tmp):
    """The summary's recorded artifact sha map (for patching)."""
    s = json.loads((tmp / "enumeration_summary.json").read_text())
    return s


def _patch_artifact_sha(tmp, name):
    """Re-record the artifact's sha in the summary (so the
    certification gate passes and the SEMANTIC gate is what must
    refuse)."""
    s = json.loads((tmp / "enumeration_summary.json").read_text())
    s["artifacts"][name]["sha256"] = _sha256_file(tmp / f"{name}.npy")
    (tmp / "enumeration_summary.json").write_text(
        json.dumps(s, indent=1) + "\n")


# --------------------------------------------------------------------------
# 1. the row-code master gate (full length)
# --------------------------------------------------------------------------

def test_01_row_code_master_gate():
    """The space pair table = 424,783 kept pairs (5,345 / 21,380
    excluded, exact); the FULL-LENGTH vectorized equality vs
    pair_indices.npy + sign_index.npy; encode/decode round-trips every
    row exactly."""
    canonical = np.load(CANONICAL_CLOUD)

    # exact exclusion counts (recorded, never assumed)
    space_plain = ga_space.Space(canonical, d_min=1.0, exclusions=(),
                                 nn_k=8)
    assert space_plain.n_pairs == KEPT_PAIRS
    assert space_plain.n_pairs_excluded == EXCLUDED_PAIRS
    assert space_plain.n_configs_excluded == EXCLUDED_CONFIGS
    assert space_plain.n_rows == POSTFLOOR_ROWS
    assert space_plain.n_pairs_total == 4 * KEPT_PAIRS + EXCLUDED_PAIRS \
        or space_plain.n_pairs_total == 430128

    # the independent re-derivation of the pair table (the same
    # formula, written out) equals the space's
    pi, pj = np.triu_indices(canonical.shape[0], k=1)
    dist = np.linalg.norm(canonical[pi] - canonical[pj], axis=1)
    keep = dist >= 1.0
    assert np.array_equal(space_plain.pi, pi[keep].astype(np.int64))
    assert np.array_equal(space_plain.pj, pj[keep].astype(np.int64))
    assert np.array_equal(space_plain.pair_dist, dist[keep])

    # the FULL-LENGTH vectorized equality vs the certified artifact
    pair_art = np.load(LANDSCAPE_DIR / "pair_indices.npy")
    sign_art = np.load(LANDSCAPE_DIR / "sign_index.npy")
    exp_pair = np.stack([np.repeat(space_plain.pi, 4),
                         np.repeat(space_plain.pj, 4)], axis=1)
    exp_sign = np.tile(np.arange(4, dtype=np.int64), space_plain.n_pairs)
    assert pair_art.shape == (POSTFLOOR_ROWS, 2)
    assert np.array_equal(pair_art.astype(np.int64), exp_pair), \
        "the pair table does not map every row (master gate FAILED)"
    assert np.array_equal(sign_art.astype(np.int64), exp_sign), \
        "the sign index is not row % 4 (master gate FAILED)"

    # encode/decode round-trips EVERY row exactly (the fenced space
    # round-trips too: decode is fence-blind by design)
    space_fenced = ga_space.Space(canonical, d_min=1.0,
                                  exclusions=(EXCLUDED_CONFIG,), nn_k=8)
    for r in range(POSTFLOOR_ROWS):
        tup = space_fenced.decode(r)
        assert space_plain.encode(*tup) == r, \
            f"round-trip failed at row {r}"
        assert space_fenced._encode_core(*tup) == r
    # the canonical exclusion: fenced in the fenced space (-1), exact
    # in the plain space (the artifact's excluded row)
    assert space_fenced.encode(*EXCLUDED_CONFIG) == -1
    assert space_plain.encode(*EXCLUDED_CONFIG) == EXCLUDED_ROW
    assert space_fenced.decode(EXCLUDED_ROW) == EXCLUDED_CONFIG
    # canonicalization: the signs move WITH the sites
    i, j, s1, s2 = EXCLUDED_CONFIG
    assert space_plain.encode(j, i, s2, s1) == EXCLUDED_ROW
    # invalid inputs return -1 (never corrected)
    for bad in ((5, 5, 1, -1), (1, 2, 0, 1), (-1, 2, 1, 1), (1, -2, 1, 1)):
        assert space_plain.encode(*bad) == -1
    # the fence is data only: decode works on the excluded row and
    # carries no fitness value anywhere
    assert space_fenced.n_exclusions == 1


# --------------------------------------------------------------------------
# 2. the oracle provider gates
# --------------------------------------------------------------------------

def test_02_oracle_provider_gates():
    """Init passes on the real artifact; init refuses a wrong cloud
    sha256, a tampered f array (a NaN moved; both with and without the
    recorded sha patched -- the certification gate and the semantic
    NaN-mask gate), a row-count mismatch; a full-length-sampled lookup
    check (20,000 finite rows bitwise, plus reversed orientations);
    the excluded-row raise naming the configuration."""
    cand = np.load(CANONICAL_CLOUD)

    # init passes; the contract surface is bound
    oracle = landscape_oracle(LANDSCAPE_DIR, CANONICAL_CLOUD)
    assert oracle.FITNESS_ID == "enumeration_v2"
    assert oracle.PROVIDER_META["fitness_id"] == "xtb"
    assert oracle.n_evals == 0
    assert set(oracle.artifact_sha256s) == {"f", "e_ts", "e_r",
                                            "pair_indices", "sign_index"}
    for name, sha in oracle.artifact_sha256s.items():
        assert sha == _sha256_file(LANDSCAPE_DIR / f"{name}.npy")

    # a wrong cloud sha is refused
    try:
        landscape_oracle(_landscape_fixture(_fix_wrong_cloud_sha),
                         CANONICAL_CLOUD)
        raise AssertionError("the wrong cloud sha was accepted")
    except ValueError as e:
        assert "cloud sha256 mismatch" in str(e)

    # a tampered f array (a NaN moved), certification intact: refused
    def _fix_nan_plain(tmp):
        f = np.load(tmp / "f.npy")
        f[100] = np.float32("nan")
        np.save(tmp / "f.npy", f)
    try:
        landscape_oracle(_landscape_fixture(_fix_nan_plain),
                         CANONICAL_CLOUD)
        raise AssertionError("the tampered f passed")
    except ValueError as e:
        assert "sha256 mismatch" in str(e)   # the certification gate

    # the same tamper with the summary's recorded sha patched: the
    # SEMANTIC NaN-mask gate must refuse (NaN marks exactly the
    # domain-excluded rows and nothing else)
    def _fix_nan_moved(tmp):
        f = np.load(tmp / "f.npy")
        f[EXCLUDED_ROW] = f[0]           # the domain row 'regains' a value
        f[100] = np.float32("nan")       # a value row loses its value
        np.save(tmp / "f.npy", f)
        _patch_artifact_sha(tmp, "f")
    try:
        landscape_oracle(_landscape_fixture(_fix_nan_moved),
                         CANONICAL_CLOUD)
        raise AssertionError("the moved NaN passed")
    except ValueError as e:
        assert "NaN mask" in str(e) and "522045" in str(e)

    # a row-count mismatch is refused
    try:
        landscape_oracle(_landscape_fixture(_fix_rows), CANONICAL_CLOUD)
        raise AssertionError("the row-count mismatch passed")
    except ValueError as e:
        assert "row-count mismatch" in str(e)

    # full-sampled lookup check: 20,000 finite rows, served f ==
    # the artifact's f BITWISE; a reversed-orientation subset rides
    f_art = np.load(LANDSCAPE_DIR / "f.npy")
    oracle2 = landscape_oracle(LANDSCAPE_DIR, CANONICAL_CLOUD)
    rng = np.random.default_rng(7)
    finite = np.flatnonzero(np.isfinite(f_art))
    rows = finite[rng.choice(finite.size, 20000, replace=False)]
    i, j, s1, s2 = oracle2.space.decode_rows(rows)
    f = oracle2.eval_batch(i, j, s1, s2, cand)
    assert np.array_equal(f, f_art[rows].astype(np.float64)), \
        "the served f is not the artifact's f (bitwise)"
    f_rev = oracle2.eval_batch(j[:500], i[:500], s2[:500], s1[:500], cand)
    assert np.array_equal(f_rev, f_art[rows[:500]].astype(np.float64)), \
        "the reversed-orientation lookup disagrees (master gate)"
    assert oracle2.n_evals == 20500   # every served row counted

    # the excluded row raises, naming the configuration (both
    # orientations: (154, 833, +1, -1) and its canonical reverse
    # (833, 154, -1, +1) -- the signs move WITH the sites)
    cfgs = (
        (np.array([EXCLUDED_CONFIG[0]]), np.array([EXCLUDED_CONFIG[1]]),
         np.array([EXCLUDED_CONFIG[2]]), np.array([EXCLUDED_CONFIG[3]])),
        (np.array([EXCLUDED_CONFIG[1]]), np.array([EXCLUDED_CONFIG[0]]),
         np.array([-EXCLUDED_CONFIG[2]]), np.array([-EXCLUDED_CONFIG[3]])),
    )
    for cfg in cfgs:
        try:
            oracle2.eval_batch(*cfg, cand)
            raise AssertionError("the domain-excluded config was served")
        except ValueError as e:
            assert "idx_i=154" in str(e) or "idx_i=833" in str(e)
    # an i==j / sub-floor configuration has no row: named failure
    try:
        oracle2.eval_batch(np.array([7]), np.array([7]),
                           np.array([1]), np.array([1]), cand)
        raise AssertionError("i == j was served")
    except ValueError as e:
        assert "idx_i=7" in str(e)


# --------------------------------------------------------------------------
# 3. engine smoke: two providers, two bars
# --------------------------------------------------------------------------

def test_03a_stub_smoke_completeness():
    """The hash-noise stub (candidates_v2): runs complete, accounting
    exact. NO search bar -- on a structureless hash landscape any
    planted-optimum recovery is direct-sampling luck (per-query hit
    rate ~5.9e-7; ~6% per 1e5-evaluation run); recovery there is NOT
    a gate, and nothing from stub runs is reported anywhere.

    The run goes through the real CLI (the deliverable artifact path,
    runs/ga/), then the identities are re-checked with the memo off
    (same queries bitwise on the deterministic provider)."""
    rc = runner.main(["--mode", "stub", "--generations", "20",
                      "--seed", "1", "--out-root",
                      str(OUT_ROOT_DELIVERABLE), "--force"])
    assert rc == 0, f"the CLI run failed (rc={rc})"
    path = runner.record_path(OUT_ROOT_DELIVERABLE, 20, 1,
                              stub_module.FITNESS_ID, CANONICAL_SHA256)
    rec = json.loads(Path(path).read_text())
    assert rec["status"] == "ok" and rec["mode"] == "stub"
    assert rec["counts"]["queries"] == \
        rec["counts"]["submitted"] + rec["counts"]["memo_hits"]
    assert rec["counts"]["n_evals_provider"] == rec["counts"]["submitted"]
    assert rec["counts"]["unique"] <= rec["counts"]["queries"]
    # sidecar sha matches the file bytes
    sha = _sha256_file(path)
    assert Path(str(path) + ".sha256").read_text().strip() == \
        f"{sha}  {path.name}"

    # the memo-off replay spends the same queries bitwise (the memo
    # reduces wall time, never the count) -- accounting exact there
    cand = np.load(CANONICAL_CLOUD)
    space = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                           nn_k=SMOKE.nn_k)
    p_off = stub_module.StubProvider()
    rec_off = Recorder(p_off.eval_batch, space)
    res_off = ga_mod.run_ga(rec_off, space, SMOKE, 1, G=20, memo=False)
    _accounting_asserts(res_off, p_off, memo=False)
    assert res_off.queries == rec["counts"]["queries"]
    assert rec_off.n_rows == res_off.queries
    # the memo-on engine agrees bitwise (same demand, same values)
    p_on = stub_module.StubProvider()
    res_on = ga_mod.run_ga(p_on.eval_batch, space, SMOKE, 1, G=20)
    _accounting_asserts(res_on, p_on, memo=True)
    assert res_on.queries == res_off.queries
    assert res_on.trajectory == res_off.trajectory
    assert res_on.ledger == res_off.ledger


def test_03b_planted_basin_bar():
    """THE SEARCH BAR: the GA recovers the planted optimum in >= 9/10
    seeds at G = 450, pop 50, smoke defaults. The planted config = the
    two maximally-separated sites of the test cloud (lexicographic
    tie-break), signs (+1, -1), a strictly-unique negative minimum;
    the landscape is WALKABLE (jitter + descent must work). The run
    records are deliverables; nothing is reported."""
    cloud = BASIN_CLOUD
    cloud_sha = hashlib.sha256(cloud.tobytes()).hexdigest()
    space = ga_space.Space(cloud, d_min=1.0, exclusions=(), nn_k=SMOKE.nn_k)
    planted_row = space.encode(BASIN_PLANTED_PAIR[0], BASIN_PLANTED_PAIR[1],
                               1, -1)
    assert planted_row >= 0
    wins = 0
    for seed in range(10):
        provider = BasinProvider()          # fresh per run
        res = ga_mod.run_ga(provider.eval_batch, space, SMOKE, seed,
                            G=450)
        _accounting_asserts(res, provider, memo=True)
        hit = (res.best_row == planted_row
               and res.best_f == BASIN_PLANTED_F)
        assert (res.best_i, res.best_j, res.best_s1, res.best_s2) == \
            (BASIN_PLANTED_PAIR[0], BASIN_PLANTED_PAIR[1], 1, -1) or not hit
        wins += hit
        record = runner.compose_record(
            res, mode="stub", fitness_id=provider.FITNESS_ID,
            provider_meta=provider.PROVIDER_META,
            cloud_path=str(REPO / "ga" / "test_ga.py"),
            cloud_sha256=cloud_sha, d_min=1.0, exclusions=(),
            exclusion_source=None,
            landscape_dir=None, artifact_sha256s=None,
            param_set="smoke",
            created_utc="1970-01-01T00:00:00Z", git_head="test-bundle")
        # the cloud identity of a test-bundle run is the bundle's cloud;
        # compose_record recorded the runner's canonical placeholder --
        # point it at the bundle cloud's bytes for the record honestly
        record["cloud"] = {"path": "ga/test_ga.py (basin bundle cloud)",
                           "sha256": cloud_sha}
        sha = runner.write_record(record, runner.record_path(
            OUT_ROOT_DELIVERABLE, res.generations, seed,
            provider.FITNESS_ID, cloud_sha))
        assert sha == _sha256_file(runner.record_path(
            OUT_ROOT_DELIVERABLE, res.generations, seed,
            provider.FITNESS_ID, cloud_sha))
    assert wins >= 9, f"the search bar FAILED: {wins}/10 planted " \
                      f"recoveries (need >= 9)"
    print(f"      bar: {wins}/10 planted recoveries (G=450, pop 50)")


# --------------------------------------------------------------------------
# 4. accounting gates + the trajectory re-derivation
# --------------------------------------------------------------------------

def test_04_accounting_and_trajectory():
    """provider.n_evals == submitted; memo_hits == queries - submitted;
    unique <= queries; the trajectory checkpoint values re-derive
    bitwise from a direct re-query of each checkpoint's best config
    (on the stub AND the oracle lookup). The re-derivation run is
    memo-OFF with a submit-side recorder: every demanded row is seen
    in order."""
    cand = np.load(CANONICAL_CLOUD)
    space = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                           nn_k=SMOKE.nn_k)

    def _check(provider_factory, fitness, space, G, seed, tmp_root):
        # memo-OFF instrumented run
        p1 = provider_factory()
        rec = Recorder(fitness(p1), space)
        res = ga_mod.run_ga(rec, space, SMOKE, seed, G=G, memo=False)
        _accounting_asserts(res, p1, memo=False)
        entries = _replay_trajectory(rec, space)
        assert res.trajectory == [[e, f] for e, f, _r in entries], \
            "the trajectory does not re-derive from the demand log"
        # each checkpoint's best config re-queried DIRECTLY through a
        # FRESH provider: bitwise the recorded best_f
        fresh = provider_factory()
        for evaluations, best_f, best_row in entries:
            i, j, s1, s2 = space.decode(best_row)
            fq = fresh.eval_batch(np.array([i]), np.array([j]),
                                  np.array([s1]), np.array([s2]), cand)
            assert fq[0] == best_f, \
                f"checkpoint @{evaluations}: re-query {fq[0]!r} != " \
                f"recorded {best_f!r}"
        # the ledger's last entry == the run's final counters
        assert res.ledger[-1][1] == res.queries
        assert res.ledger[-1][2] == res.best_f
        return res

    _check(stub_module.StubProvider,
           lambda p: p.eval_batch, space, 60, 3, None)

    oracle_arm = landscape_oracle(LANDSCAPE_DIR, CANONICAL_CLOUD)
    res_oracle = _check(
        lambda: landscape_oracle(LANDSCAPE_DIR, CANONICAL_CLOUD),
        lambda p: p.eval_batch, space, 8, 3, None)
    # the oracle serves the float32 artifact values exactly; the
    # best-so-far never exceeds the certified global minimum's quality
    # bound (a coherence check against the promoted artifact)
    f_art_min = float(np.nanmin(np.load(LANDSCAPE_DIR / "f.npy")))
    assert res_oracle.best_f >= f_art_min


# --------------------------------------------------------------------------
# 5. reproducibility
# --------------------------------------------------------------------------

def test_05_reproducibility():
    """(fitness callable + Space + params + seed) run twice ->
    bitwise-identical result record (the wall-clock fields are
    measurements, not run content, and are excluded; created_utc is
    injected identically)."""
    cand = np.load(CANONICAL_CLOUD)
    space = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                           nn_k=SMOKE.nn_k)
    r1 = ga_mod.run_ga(stub_module.StubProvider().eval_batch, space, SMOKE,
                       42, G=8)
    r2 = ga_mod.run_ga(stub_module.StubProvider().eval_batch, space, SMOKE,
                       42, G=8)

    def _strip(result):
        d = result.as_dict()
        d.pop("wall_seconds")
        return d
    assert _strip(r1) == _strip(r2), "two identical runs disagree"

    # the runner-level record: identical modulo the wall clock
    rec1 = runner.compose_record(r1, mode="stub",
                                 fitness_id=stub_module.FITNESS_ID,
                                 provider_meta=stub_module.PROVIDER_META,
                                 cloud_path=CANONICAL_CLOUD,
                                 cloud_sha256=CANONICAL_SHA256, d_min=1.0,
                                 exclusions=[EXCLUDED_CONFIG],
                                 exclusion_source=str(
                                     LANDSCAPE_DIR / "domain_exclusions.json"),
                                 created_utc="1970-01-01T00:00:00Z",
                                 git_head="fixed",
                                 param_set="smoke")
    rec2 = runner.compose_record(r2, mode="stub",
                                 fitness_id=stub_module.FITNESS_ID,
                                 provider_meta=stub_module.PROVIDER_META,
                                 cloud_path=CANONICAL_CLOUD,
                                 cloud_sha256=CANONICAL_SHA256, d_min=1.0,
                                 exclusions=[EXCLUDED_CONFIG],
                                 exclusion_source=str(
                                     LANDSCAPE_DIR / "domain_exclusions.json"),
                                 created_utc="1970-01-01T00:00:00Z",
                                 git_head="fixed",
                                 param_set="smoke")
    a = json.dumps(rec1, indent=1)
    b = json.dumps(rec2, indent=1)
    a2 = json.dumps({k: v for k, v in rec1.items() if k != "wall_seconds"},
                    indent=1)
    b2 = json.dumps({k: v for k, v in rec2.items() if k != "wall_seconds"},
                    indent=1)
    assert a2 == b2, "the record's deterministic content differs"


# --------------------------------------------------------------------------
# 6. fence gates
# --------------------------------------------------------------------------

def test_06_fence_gates():
    """The canonical exclusion (154, 833, +1, -1) is never sampled or
    submitted in an instrumented oracle run (submit-side assert); its
    UNfenced encode equals the artifact's excluded row 522045 (the
    direct cross-artifact check); the synthetic fence: the basin
    optimum added to exclusions -> never submitted, run completes; the
    canonical landscape's exclusions count == 1."""
    cand = np.load(CANONICAL_CLOUD)

    # -- the direct cross-artifact check (fence-less space) ----------
    plain = ga_space.Space(cand, d_min=1.0, exclusions=(), nn_k=8)
    assert plain.encode(*EXCLUDED_CONFIG) == EXCLUDED_ROW
    assert plain.decode(EXCLUDED_ROW) == EXCLUDED_CONFIG

    # -- the canonical landscape's exclusions count == 1 --------------
    excl = json.loads((LANDSCAPE_DIR / "domain_exclusions.json").read_text())
    assert excl["count"] == 1
    assert excl["rows"] == [EXCLUDED_ROW]
    assert excl["entries"][0]["config"] == list(EXCLUDED_CONFIG)

    # -- the instrumented oracle run: the fenced config never
    #    sampled/submitted; the run completes; accounting exact ------
    fenced = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                            nn_k=SMOKE.nn_k)
    provider = landscape_oracle(LANDSCAPE_DIR, CANONICAL_CLOUD)
    rec = Recorder(provider.eval_batch, fenced, forbid_rows=(EXCLUDED_ROW,))
    res = ga_mod.run_ga(rec, fenced, SMOKE, 5, G=8)
    _accounting_asserts(res, provider, memo=True)
    assert rec.n_rows == res.submitted
    print(f"      oracle smoke: best_f={res.best_f:.4f} "
          f"queries={res.queries} (fence held)")

    # -- the synthetic fence: the basin optimum excluded -------------
    basin = ga_space.Space(BASIN_CLOUD, d_min=1.0,
                           exclusions=((BASIN_PLANTED_PAIR[0],
                                        BASIN_PLANTED_PAIR[1], 1, -1),),
                           nn_k=8)
    assert basin.n_exclusions == 1
    assert basin.encode(BASIN_PLANTED_PAIR[0], BASIN_PLANTED_PAIR[1], 1,
                        -1) == -1
    bp = BasinProvider()
    # the fence makes the planted config invalid; the submit-side
    # assert instead pins its UNfenced row identity
    plain_basin = ga_space.Space(BASIN_CLOUD, d_min=1.0, exclusions=(),
                                 nn_k=8)
    planted_row = plain_basin.encode(BASIN_PLANTED_PAIR[0],
                                     BASIN_PLANTED_PAIR[1], 1, -1)
    brec = Recorder(bp.eval_batch, basin, forbid_rows=(planted_row,))
    res_b = ga_mod.run_ga(brec, basin, GAParams(pop_size=20, n_elite=2),
                          0, G=30)
    assert res_b.best_row != planted_row
    assert res_b.best_f > BASIN_PLANTED_F   # the optimum is unreachable
    assert brec.n_rows == res_b.submitted


# --------------------------------------------------------------------------
# 7. descent bookkeeping
# --------------------------------------------------------------------------

def test_07_descent_bookkeeping():
    """A converged elite costs 0 queries at later generations (the
    per-generation ledger deltas; a pop == n_elite run has descent-only
    generations, so the deltas are descent queries alone); descent
    never leaves an elite worse than it found it (the descent_end
    events' end_f <= start_f)."""
    bspace = ga_space.Space(BASIN_CLOUD, d_min=1.0, exclusions=(), nn_k=8)

    # (a) the pop == n_elite run: descent-only generation budgets
    p2 = GAParams(pop_size=2, n_elite=2)
    res = ga_mod.run_ga(BasinProvider().eval_batch, bspace, p2, 0, G=12)
    ledger = res.ledger
    deltas = [ledger[0][1] - p2.pop_size] \
        + [ledger[g][1] - ledger[g - 1][1] for g in range(1, len(ledger))]
    assert deltas[0] > 0, "gen 1 spent no descent queries at all"
    zeros = [g for g, d in enumerate(deltas) if d == 0]
    assert zeros, "no generation shows a fully-converged elite set"
    first_zero = zeros[0]
    assert all(d == 0 for d in deltas[first_zero:]), \
        "a converged elite still cost queries at a later generation"
    assert all(d >= 0 for d in deltas)

    # (b) smoke defaults: descent_end events never worsen an elite
    events = []
    res2 = ga_mod.run_ga(BasinProvider().eval_batch, bspace, SMOKE, 0,
                         G=40, on_event=lambda name, payload:
                         events.append((name, payload)))
    ends = [p for n, p in events if n == "descent_end"]
    assert ends, "no descent ran at all (the smoke run must descend)"
    for e in ends:
        assert e["end_f"] <= e["start_f"], \
            f"descent worsened an elite: {e}"
    skips = [p for n, p in events if n == "descent_skip"]
    assert skips, "the converged-set skip never fired in 40 generations"
    for e in skips:
        assert e["queries_before"] == e["queries_after"], \
            "a converged elite consumed queries"
    print(f"      descent: {len(ends)} descents, {len(skips)} converged "
          f"skips, {res2.n_converged} configs remembered")


# --------------------------------------------------------------------------
# 8. the live path (the second work-item gate)
# --------------------------------------------------------------------------

def _xtb_available():
    from fitness import xtb as xtb_module
    return xtb_module.XTB_BIN.is_file()


def test_08a_live_repeat_determinism():
    """>= 20 repeat submissions of the same config through FRESH xtb
    provider instances return bitwise-identical f (binary-gated;
    ~seconds)."""
    if not _xtb_available():
        print("SKIP  (xtb binary not installed: bin/xtb-6.7.1/bin/xtb)")
        return "skip"
    from fitness import xtb as xtb_module
    cand = np.load(CANONICAL_CLOUD)
    rows = np.array([r for r in (0, 37, 74, 111, 148, 185, 222, 259, 296,
                                 333, 370, 407, 444, 481, 518, 555, 592,
                                 629, 666, 703, 740, 777)
                     if r != EXCLUDED_ROW], dtype=np.int64)
    assert rows.size >= 20
    space = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                           nn_k=8)
    i, j, s1, s2 = space.decode_rows(rows)
    t0 = time.monotonic()
    f1 = xtb_module.XtbProvider().eval_batch(i, j, s1, s2, cand)
    f2 = xtb_module.XtbProvider().eval_batch(i, j, s1, s2, cand)
    assert np.array_equal(f1, f2), \
        "repeat submissions through fresh providers disagree"
    print(f"      live repeat determinism: {rows.size} configs bitwise, "
          f"2 fresh providers, {time.monotonic() - t0:.1f}s")


def test_08b_live_path():
    """>= 10 restarts through fitness/xtb.py via the real CLI (fresh
    instance per run, OMP=1 as shipped, memo ON, G = 20, pop 50), each
    green on the accounting; then THE TRANSFER DEMONSTRATION: one
    further live restart, same seed, memo OFF -- trajectory + queries
    bitwise-identical to the memo-ON run (only the wall cost and
    submitted differ). Env-gated: needs the xtb binary AND
    GOCAT_GA_LIVE=1 (11 runs x ~3-6 min)."""
    if not _xtb_available():
        print("SKIP  (xtb binary not installed: bin/xtb-6.7.1/bin/xtb)")
        return "skip"
    if os.environ.get("GOCAT_GA_LIVE") != "1":
        print("SKIP  (set GOCAT_GA_LIVE=1 for the ~11-run live path; "
              "~3-6 min per run)")
        return "skip"
    from fitness import xtb as xtb_module
    seeds = list(range(11, 21))
    records = {}
    for seed in seeds:
        t0 = time.monotonic()
        rc = runner.main(["--mode", "xtb", "--generations", "20",
                          "--seed", str(seed), "--out-root",
                          str(OUT_ROOT_DELIVERABLE), "--force"])
        assert rc == 0, f"the live restart seed={seed} failed (rc={rc})"
        path = runner.record_path(OUT_ROOT_DELIVERABLE, 20, seed,
                                  "xtb", CANONICAL_SHA256)
        rec = json.loads(Path(path).read_text())
        assert rec["mode"] == "live" and rec["status"] == "ok"
        c = rec["counts"]
        assert c["n_evals_provider"] == c["submitted"]
        assert c["queries"] == c["submitted"] + c["memo_hits"]
        assert c["unique"] <= c["queries"]
        records[seed] = rec
        print(f"      live restart seed={seed}: queries={c['queries']} "
              f"submitted={c['submitted']} memo={c['memo_hits']} "
              f"wall={rec['wall_seconds']:.0f}s "
              f"(+{time.monotonic() - t0:.0f}s)")

    # THE TRANSFER DEMONSTRATION: same seed, memo OFF, engine-level;
    # only the wall cost and submitted differ
    seed = seeds[0]
    cand = np.load(CANONICAL_CLOUD)
    space = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                           nn_k=SMOKE.nn_k)
    provider = xtb_module.XtbProvider()      # a FRESH instance
    res_off = ga_mod.run_ga(provider.eval_batch, space, SMOKE, seed,
                            G=20, memo=False)
    _accounting_asserts(res_off, provider, memo=False)
    rec_on = records[seed]
    assert res_off.queries == rec_on["counts"]["queries"], \
        "memo OFF changed the query count (it must not)"
    assert res_off.trajectory == rec_on["trajectory"], \
        "memo OFF changed the trajectory (values identical => identical GA)"
    assert res_off.ledger == rec_on["ledger"]
    assert (res_off.best_row, res_off.best_i, res_off.best_j,
            res_off.best_s1, res_off.best_s2, res_off.best_f) == \
        (rec_on["best"]["row"], rec_on["best"]["config"][0],
         rec_on["best"]["config"][1], rec_on["best"]["config"][2],
         rec_on["best"]["config"][3], rec_on["best"]["f"])
    assert res_off.submitted >= rec_on["counts"]["submitted"]
    record = runner.compose_record(
        res_off, mode="live", fitness_id=provider.FITNESS_ID,
        provider_meta=provider.PROVIDER_META, cloud_path=CANONICAL_CLOUD,
        cloud_sha256=CANONICAL_SHA256, d_min=1.0,
        exclusions=[EXCLUDED_CONFIG],
        exclusion_source=str(LANDSCAPE_DIR / "domain_exclusions.json"),
        param_set="smoke",
        landscape_dir=str(LANDSCAPE_DIR),
        artifact_sha256s={k: _sha256_file(LANDSCAPE_DIR / f"{k}.npy")
                          for k in ("f", "e_ts", "e_r", "pair_indices",
                                    "sign_index")})
    record["note"] = ("the item-6 ablation arm: the GA-layer memo OFF "
                      "(the transfer demonstration: values identical => "
                      "identical GA; only the wall cost and submitted "
                      "differ)")
    runner.write_record(record, (OUT_ROOT_DELIVERABLE
                                 / "transfer_memo_off").joinpath(
        runner.record_path(".", 20, seed, provider.FITNESS_ID,
                           CANONICAL_SHA256).name))
    print("      transfer demo: memo OFF == memo ON "
          "(trajectory + queries bitwise; only wall cost + submitted "
          "differ)")


# --------------------------------------------------------------------------
# 9. negative gates
# --------------------------------------------------------------------------

class _FullRaiser(stub_module.StubProvider):
    """A provider double whose eval_batch_full RAISES: if the engine
    (or anything in the run path) calls it, the run fails loudly."""

    def eval_batch_full(self, *a, **k):
        raise AssertionError("eval_batch_full must never be called")

    def eval_batch(self, idx_i, idx_j, sign_i, sign_j, candidates):
        # the real computation directly off the impl (eval_batch's
        # contract view without eval_batch_full), with honest counting
        f, _t, _r = self._eval_batch_full_impl(
            idx_i, idx_j, sign_i, sign_j, candidates)
        self.n_evals += int(np.asarray(idx_i).size)
        return f


class _SccStyleError(RuntimeError):
    """The provider-side domain-exclusion class (mirrors
    fitness.xtb.SccDomainExclusion's shape): carries .config."""

    def __init__(self, message, config):
        super().__init__(message)
        self.config = list(config)


class _FailingProvider(stub_module.StubProvider):
    """A provider that fails at its FIRST batch (plumbing), and a
    domain-class variant."""

    def __init__(self, mode="plumbing"):
        super().__init__()
        self._mode = mode

    def eval_batch(self, *a, **k):
        f = stub_module.StubProvider.eval_batch(self, *a, **k)
        if self._mode == "plumbing":
            raise RuntimeError("plumbing boom: xtb exited 200")
        raise _SccStyleError("SCC non-convergence (the domain class)",
                             (154, 833, 1, -1))


def _scan_source_for(pattern, regex=False, include_tests=False):
    import re
    files = sorted(Path(REPO / "ga").glob("*.py")) + [REPO / "run_ga.py"]
    if not include_tests:
        # the stage files only: this test file's own prose quoting the
        # requirement must not false-positive the scan
        files = [f for f in files
                 if f.name != os.path.basename(__file__)]
    hits = []
    for f in files:
        text = f.read_text()
        if regex:
            for ln, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    hits.append(f"{f.name}:{ln}")
        else:
            if pattern in text:
                hits.append(str(f.name))
    return hits


def test_09_negative_gates():
    """A failing provider aborts fail-hard naming the failing config
    (NO retry logic anywhere -- grep-able absence); a provider double
    raising on eval_batch_full proves the engine never calls it; ga/
    never imports exhaustive_search."""
    cand = np.load(CANONICAL_CLOUD)
    space = ga_space.Space(cand, d_min=1.0, exclusions=(EXCLUDED_CONFIG,),
                           nn_k=SMOKE.nn_k)

    # -- the plumbing failure: terminal, naming the config ----------
    try:
        ga_mod.run_ga(_FailingProvider("plumbing").eval_batch, space,
                      GAParams(pop_size=10, n_elite=2), 0, G=3)
        raise AssertionError("the failing provider did not stop the run")
    except ga_mod.ProviderFailure as e:
        assert "plumbing boom" in str(e)
        assert "(idx_i=" in str(e), "the batch context is not named"
        assert e.partial is not None
        assert e.partial.queries > 0

    # -- the domain-class failure: the failing config propagates +
    #    the failed record is written (status=failed) ---------------
    tmp_out = Path(tempfile.mkdtemp(prefix="ga_failed_record_"))
    argv = ["--mode", "stub", "--generations", "3", "--seed", "0"]
    # swap-in: run the engine directly and compose the failed record
    try:
        ga_mod.run_ga(_FailingProvider("domain").eval_batch, space,
                      GAParams(pop_size=10, n_elite=2), 0, G=3)
        raise AssertionError("the domain failure did not stop the run")
    except ga_mod.ProviderFailure as e:
        cfg = runner.failing_config(e)
        assert cfg == [154, 833, 1, -1], f"the failing config: {cfg}"
        record = runner.compose_record(
            e.partial, mode="stub", fitness_id="stub_v2",
            provider_meta=stub_module.PROVIDER_META,
            cloud_path=CANONICAL_CLOUD, cloud_sha256=CANONICAL_SHA256,
            d_min=1.0, exclusions=[EXCLUDED_CONFIG],
            exclusion_source=str(LANDSCAPE_DIR / "domain_exclusions.json"),
            param_set="smoke", status="failed",
            failure={"error": str(e),
                     "failing_config": cfg})
        path = tmp_out / "g000003_s00000_stub_v2_084e48a6b526.json"
        runner.write_record(record, path)
        rec = json.loads(path.read_text())
        assert rec["status"] == "failed"
        assert rec["failure"]["failing_config"] == [154, 833, 1, -1]
        assert rec["counts"]["queries"] == e.partial.queries

    # -- the engine never calls eval_batch_full ----------------------
    p = _FullRaiser()
    res = ga_mod.run_ga(p.eval_batch, space, GAParams(pop_size=10,
                                                      n_elite=2), 1, G=3)
    assert res.generations_executed == 3
    assert p.n_evals == res.submitted

    # -- grep-able absences ------------------------------------------
    assert not _scan_source_for("retry", regex=False), \
        "the word 'retry' appears in ga/ (no retry logic may exist)"
    imports = _scan_source_for(r"^\s*(import|from)\s+.*exhaustive_search",
                               regex=True)
    assert not imports, f"ga/ imports exhaustive_search at: {imports}"
    # the no-searcher-beyond-the-callable rule, statically: the engine
    # file contains no eval_batch_full CALL (prose mentions excluded)
    calls = _scan_source_for(r"eval_batch_full\s*\(", regex=True)
    assert not [h for h in calls if h.startswith("ga.py")], \
        f"the engine calls eval_batch_full at: {calls}"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main():
    tests = [
        test_01_row_code_master_gate,
        test_02_oracle_provider_gates,
        test_03a_stub_smoke_completeness,
        test_03b_planted_basin_bar,
        test_04_accounting_and_trajectory,
        test_05_reproducibility,
        test_06_fence_gates,
        test_07_descent_bookkeeping,
        test_08a_live_repeat_determinism,
        test_08b_live_path,
        test_09_negative_gates,
    ]
    failures = []
    t_all = time.monotonic()
    for t in tests:
        t0 = time.monotonic()
        try:
            status = t() or "pass"
            label = (f"PASS  {t.__name__}" if status == "pass"
                     else f"SKIP  {t.__name__} ({status})")
            label += f"  [{time.monotonic() - t0:.1f}s]"
            print(label)
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:  # a bug in the test or the stage
            print(f"FAIL  {t.__name__}: unexpected {type(e).__name__}: {e}")
            failures.append(t.__name__)
    n = len(tests)
    print(f"\n{n - len(failures)}/{n} passed "
          f"[{time.monotonic() - t_all:.1f}s total]"
          + ("" if not failures else f"; FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())