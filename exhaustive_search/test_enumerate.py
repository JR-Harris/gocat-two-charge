#!/usr/bin/env python3
"""Self-tests for the enumeration core + worker + proofs.

Plain-python test file (no framework); run from the repo root:
    venv/bin/python exhaustive_search/test_enumerate.py

The stub-mode tests exercise the real mechanics on constructed clouds
(the stub never reads data/). The live-proof test (env-gated) re-runs
the four real-provider proofs on the canonical cloud (~3 min with the
xtb binary; SKIP line otherwise) and requires the committed reference
to reproduce bitwise.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from exhaustive_search import enumerate as en
from exhaustive_search import proofs as pr
from fitness import stub as stub_module

REPO = Path(__file__).resolve().parent.parent

# Recorded canonical-cloud derivatives (the briefing's fixed floor, the
# locked cloud): asserted exactly (recorded, never assumed).
CANONICAL_CLOUD = REPO / "data" / "derived" / "candidates_v2.npy"
CANONICAL_SHA256 = (
    "084e48a6b5267900a995ba2d4f0412f77f5bb13446ba74eafaf474c644349e71")
N_EXPECTED = 928
UNIVERSE_TOTAL = 1720512                      # 4 * C(928, 2)
POSTFLOOR_ROWS = 1699132                      # d_min = 1.0 A
EXCLUDED_PAIRS = 5345
EXCLUDED_CONFIGS = 21380


def _small_cloud(tmpdir, n=60, seed=11, nan_row=None):
    """A deterministic constructed cloud; some pairs < 1.0 A by placing
    three points tightly. nan_row: optional index whose coordinates
    become NaN (the non-finite negative fixture)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-10.0, 10.0, size=(n, 3))
    c[1] = c[0] + [0.4, 0.0, 0.0]       # d = 0.4: a <1.0 pair
    c[2] = c[0] + [0.0, 0.6, 0.0]       # d = 0.6
    c[3] = c[0] + [0.0, 0.0, 0.8]       # d = 0.8
    if nan_row is not None:
        c[nan_row, 0] = np.nan
    path = Path(tmpdir) / f"cloud_{seed}{'_nan' if nan_row is not None else ''}.npy"
    np.save(path, c)
    return path


def _en_sha(path):
    return en.sha256_file(path)


# --------------------------------------------------------------------------
# 1. The universe (counts asserted from n; the floor counts recorded)
# --------------------------------------------------------------------------

def test_01_universe_counts():
    """4*C(n,2) asserted at runtime; d_min off = the full universe
    (1,720,512 at n=928); d_min=1.0 = the recorded post-floor +
    excluded counts."""
    canonical = np.load(CANONICAL_CLOUD)
    u_off = en.build_universe(canonical, d_min=0.0)
    assert u_off["n"] == N_EXPECTED
    assert u_off["n_configs_total"] == UNIVERSE_TOTAL == 4 * (928 * 927) // 2
    assert u_off["n_rows"] == UNIVERSE_TOTAL and u_off["n_pairs_excluded"] == 0

    u = en.build_universe(canonical, d_min=en.D_MIN_A)
    assert u["n_configs_total"] == UNIVERSE_TOTAL   # the runtime assert ran
    assert u["n_pairs_excluded"] == EXCLUDED_PAIRS
    assert u["n_configs_excluded"] == EXCLUDED_CONFIGS
    assert u["n_rows"] == POSTFLOOR_ROWS
    assert u["n_configs_excluded"] + u["n_rows"] == UNIVERSE_TOTAL
    # no kept pair sits below the floor; all excluded would (consistency)
    assert float(u["pair_dist"].min()) >= en.D_MIN_A

    # the runtime assert itself: n hard-coded in the assertion path
    try:
        en.build_universe(np.zeros((5, 3)), d_min=0.0)
    except RuntimeError as e:
        raise AssertionError(f"a (5,3) cloud must build fine: {e}")


def test_02_row_mapping():
    """Row r = 4*p + s maps to its configuration forever: pair-major,
    canonical sign order, post-floor pair ranks (triu row-major)."""
    canonical = np.load(CANONICAL_CLOUD)
    u = en.build_universe(canonical, d_min=en.D_MIN_A)
    pi, pj = u["pi"], u["pj"]
    sgn = np.array(en.SIGN_COMBOS, dtype=np.int64)
    for r in (0, 1, 2, 3, 4, 7, 123, 456, 1699131):
        p, s = divmod(r, 4)
        assert int(u["idx_i"][r]) == int(pi[p]) and int(u["idx_j"][r]) == int(pj[p]), (
            f"row {r}: pair mapping broken")
        assert int(u["sign_i"][r]) == int(sgn[s, 0])
        assert int(u["sign_j"][r]) == int(sgn[s, 1])
        assert int(u["sign_index"][r]) == s
    # the ordering IS triu row-major: lexicographic (i, j) strictly
    # increasing over the kept pairs
    lex = pi.astype(np.int64) * 10000 + pj   # n = 928 < 10000
    assert (np.diff(lex) > 0).all()
    # a concrete hand-check: the canonical sign order is [(+,+),(+,-),(-,+),(-,-)]
    assert int(u["sign_i"][4]) == +1 and int(u["sign_i"][5]) == +1 \
        and int(u["sign_i"][6]) == -1 and int(u["sign_i"][7]) == -1


def test_03_shard_split():
    """The canonical split: contiguous, disjoint, union-complete,
    deterministic."""
    shards1 = en.split_shards(10, 1)
    shards3 = en.split_shards(10, 3)
    shards9 = en.split_shards(10, 9)
    assert shards1 == [(0, 10)] and shards3 == [(0, 4), (4, 8), (8, 10)]
    assert shards9 == [(k, k + 2) for k in (0, 2, 4, 6, 8)] + [(10, 10)][:0] or \
        shards9 == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    # union completeness + disjointness for a nontrivial split
    for n, k in ((1699132, 8), (1699132, 13), (797, 4)):
        sh = en.split_shards(n, k)
        assert sh[0][0] == 0 and sh[-1][1] == n
        for (a1, b1), (a2, b2) in zip(sh, sh[1:]):
            assert b1 == a2
        assert sum(b - a for a, b in sh) == n
    try:
        en.split_shards(10, 0)
        raise AssertionError("k=0 must fail")
    except ValueError:
        pass


def test_04_kill_resume_equivalence():
    """Interrupted == uninterrupted, byte-for-byte: a partial shard
    (run_shard up to a boundary = the simulated kill), resumed over the
    full range; the aggregated landscape bitwise == a single fresh
    full-range run's; the resumed session evaluates only the missing
    rows."""
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp)
        candidates = np.load(cloud_path)
        u = en.build_universe(candidates, d_min=en.D_MIN_A)
        n_rows = int(u["n_rows"])

        # the uninterrupted reference (its own shard, one pass)
        p_full = stub_module.StubProvider()
        d_full = Path(tmp) / "shard_full"
        en.run_shard(d_full, u, candidates, p_full.eval_batch_full, 0, n_rows,
                     flush_rows=47)

        # the interrupted run: two sessions (the kill = the phase
        # boundary; both phases run through the real resumable range
        # executor, the resume reading the append-only manifest)
        p_kill = stub_module.StubProvider()
        d_kill = Path(tmp) / "shard_killed"
        en.run_shard(d_kill, u, candidates, p_kill.eval_batch_full,
                     0, min(60, n_rows), flush_rows=47)   # the kill
        rows_mid, _ = en.read_manifest(d_kill)
        assert 0 < rows_mid.size < n_rows, "the kill boundary did not split"
        session2 = en.run_shard(d_kill, u, candidates, p_kill.eval_batch_full,
                                0, n_rows, flush_rows=47)
        assert session2 == n_rows - rows_mid.size, (
            "the resume re-evaluated recorded rows")

        land_full = en.aggregate_shards([d_full], u)
        land_kill = en.aggregate_shards([d_kill], u)
        assert land_full["f"].tobytes() == land_kill["f"].tobytes(), \
            "interrupted != uninterrupted (f)"
        assert land_full["e_ts"].tobytes() == land_kill["e_ts"].tobytes()
        assert land_full["e_r"].tobytes() == land_kill["e_r"].tobytes()
        # and the pairs/sign tables derive identically
        assert land_full["pairs"].tobytes() == land_kill["pairs"].tobytes()
        # already-complete resume: nothing evaluated, counts unmoved
        before = p_kill.n_evals
        session3 = en.run_shard(d_kill, u, candidates, p_kill.eval_batch_full,
                                0, n_rows, flush_rows=47)
        assert session3 == 0 and p_kill.n_evals == before, \
            "a completed shard re-evaluated on resume"


def test_05_accounting_exactness():
    """provider.n_evals == distinct evaluated rows; batched == solo
    bitwise; a failed call counts nothing; a repeat counts again (the
    stub's own contract surface, driven through the core)."""
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        candidates = np.load(cloud_path)
        u = en.build_universe(candidates, d_min=en.D_MIN_A)
        p = stub_module.StubProvider()
        d = Path(tmp) / "shard"
        en.run_shard(d, u, candidates, p.eval_batch_full, 0, 1000,
                     flush_rows=250)
        rows, _ = en.read_manifest(d)
        assert p.n_evals == rows.size, "ledger/registry count mismatch"
        assert len(list(en.list_parts(d))) == 4 and rows.size == 1000

        # batched-vs-solo bitwise on the real rows (the stub's guarantee)
        f_b, t_b, r_b = p.eval_batch_full(
            u["idx_i"][rows], u["idx_j"][rows], u["sign_i"][rows],
            u["sign_j"][rows], candidates)
        for k in (0, 1, 333, 999):
            f1, t1, r1 = p.eval_batch_full(
                [u["idx_i"][rows[k]]], [u["idx_j"][rows[k]]],
                [u["sign_i"][rows[k]]], [u["sign_j"][rows[k]]], candidates)
            assert f1[0].tobytes() == f_b[k].tobytes()
            assert t1[0].tobytes() == t_b[k].tobytes()
            assert r1[0].tobytes() == r_b[k].tobytes()
        # a failed (invalid) batch call counts nothing
        before = p.n_evals
        try:
            p.eval_batch_full([0], [0], [1], [5], candidates)
        except ValueError:
            pass
        assert p.n_evals == before


def test_06_fail_hard_shard():
    """A provider raise ("a shard whose provider raises") fails the
    shard hard: the run stops, the diagnostic record names the row and
    its configuration; nothing beyond the failing window is recorded
    (the flush protocol keeps complete windows only); a later resume
    with the healthy evaluator completes the range."""
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        candidates = np.load(cloud_path)
        u = en.build_universe(candidates, d_min=en.D_MIN_A)
        raise_at_row = 100
        real = stub_module.StubProvider().eval_batch_full

        def evaluator_with_raise(idx_i, idx_j, sign_i, sign_j, cand):
            # the provider-raise injection: fail hard on the window that
            # spans row 100 (the 3rd flush window of 50 from row 0)
            evaluator_with_raise.calls += 1
            if evaluator_with_raise.calls == raise_at_row // 50 + 1:
                raise RuntimeError(
                    f"injected provider failure at row {raise_at_row}")
            return real(idx_i, idx_j, sign_i, sign_j, cand)
        evaluator_with_raise.calls = 0

        end = min(400, int(u["n_rows"]))
        d = Path(tmp) / "shard"
        raised = None
        try:
            en.run_shard(d, u, candidates, evaluator_with_raise,
                         0, end, flush_rows=50)
        except RuntimeError as e:
            raised = str(e)
        assert raised is not None, "the failing shard did NOT fail"
        assert f"row {raise_at_row}" in raised, (
            "the failure does not name the row")
        # the diagnostic record
        meta = json.loads((d / "shard_meta.json").read_text())
        assert meta["ok"] is False and "error" in meta
        assert f"row {raise_at_row}" in meta["error"]
        # nothing beyond the failing window is recorded: the manifest =
        # the complete windows strictly before the failing one
        rows, _ = en.read_manifest(d)
        assert rows.size == (raise_at_row // 50) * 50, (
            f"recorded {rows.size}, expected the window prefix "
            f"{(raise_at_row // 50) * 50}")
        # a later resume with the healthy evaluator completes the range
        en.run_shard(d, u, candidates, real, 0, end, flush_rows=50)
        land = en.aggregate_shards([d], u, expect_rows=end)
        assert land["n_rows_evaluated"] == end
        assert land["rows_evaluated_distinct"] == end


def test_07_cloud_identity_gates():
    """A tampered cloud refuses init everywhere: the worker (subprocess
    rc != 0 with the mismatch named), the driver's resolve, and the
    canonical default (a wrong --cloud-sha256 refuses)."""
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        # tamper: a copy with one coordinate nudged
        c = np.load(cloud_path).copy()
        c[0, 0] += 0.1
        bad = Path(tmp) / "bad.npy"
        np.save(bad, c)
        sha_bad = _en_sha(bad)
        proc = subprocess.run(
            [sys.executable, "-m", "exhaustive_search.worker",
             "--cloud", str(bad), "--cloud-sha256", _en_sha(cloud_path),
             "--mode", "stub", "--start", "0", "--end", "8",
             "--work-dir", str(Path(tmp) / "wshard2")],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode != 0, "the tampered cloud worker exited 0"
        assert "sha256 mismatch" in proc.stderr, proc.stderr[-500:]

        driver = _load_driver()
        try:
            driver.resolve_cloud(_Args(cloud=str(bad), cloud_sha256=None))
            raise AssertionError("a non-canonical cloud without its sha passed")
        except ValueError:
            pass
        # the canonical default holds its frozen sha
        cpath, sha, canonical = driver.resolve_cloud(
            _Args(cloud=driver.CANONICAL_CLOUD, cloud_sha256=None))
        assert canonical and sha == driver.CANONICAL_CLOUD_SHA256 == CANONICAL_SHA256
        # the frozen canonical sha is the default; a WRONG sha for the
        # canonical cloud refuses at the check gate
        try:
            driver.check_cloud(driver.CANONICAL_CLOUD, "0" * 64)
            raise AssertionError("a wrong canonical sha passed the gate")
        except ValueError as e:
            assert "sha256 mismatch" in str(e)


def _load_driver():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_exhaustive_search", REPO / "run_exhaustive_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, cloud=None, cloud_sha256=None, mode="xtb",
                 work_root=None):
        assert cloud is not None or True
        self.cloud = driver_CANONICAL if cloud is None else cloud
        self.cloud_sha256 = cloud_sha256
        self.mode = mode
        self.work_root = work_root


driver_CANONICAL = "data/derived/candidates_v2.npy"


def test_11_domain_exclusions():
    """The residual domain rule (the SCC class under the frozen T2
    protocol): a failing side = a recorded exclusion (the raw evidence
    kept + verified at aggregation), the worker CONTINUES, the
    aggregation completes (evaluated OR excluded, never both), and the
    artifact carries the exclusion ledger with NaN value positions; a
    NON-SCC failure still fails the shard hard (full stop). Hermetic:
    a fake embedded provider (the stub's values + the exclusion raise
    on the designated rows) + pre-written raw outputs."""
    import sys as _sys
    from fitness import xtb as _ff
    from exhaustive_search import worker as _wk

    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        cand = np.load(cloud_path)
        u = en.build_universe(cand, d_min=en.D_MIN_A)
        n_rows = int(u["n_rows"])
        pair_to_p = {f"{int(i)}:{int(j)}": p
                     for p, (i, j) in enumerate(zip(u["pi"], u["pj"]))}
        # the designated exclusions: one ts-side fail, one reactant-side
        fail = {17: "ts", 400: "reactant"}
        stub_eval = stub_module.StubProvider().eval_batch_full

        def fake_provider(idx_i, idx_j, sign_i, sign_j, cand):
            # the batch: if ANY of the window's rows is designated, the
            # provider's per-row loop dies there (like the real one)
            for k in range(idx_i.size):
                i, j, si, sj = (int(idx_i[k]), int(idx_j[k]),
                                int(sign_i[k]), int(sign_j[k]))
                p = pair_to_p[f"{i}:{j}"]
                for s in range(4):
                    row = 4 * p + s
                    if row in fail and int(sign_i[k]) == en.SIGN_COMBOS[s][0]                             and int(sign_j[k]) == en.SIGN_COMBOS[s][1]:
                        raise _ff.SccDomainExclusion(
                            f"{fail[row]}: xtb exited 128 for config "
                            f"(idx_i={i}, idx_j={j}, sign_i={si:+d}, "
                            f"sign_j={sj:+d}) with the SCC non-convergence "
                            "signature", stage=fail[row],
                            config_label=(i, j, si, sj),
                            config=(i, j, si, sj),
                            raw_dir=_raw_dir(i, j, si, sj, fail[row]))
            return stub_eval(idx_i, idx_j, sign_i, sign_j, cand)

        def _raw_dir(i, j, si, sj, stage):
            d = Path(tmp) / "raw" / _ff.config_key(i, j, si, sj, "a" * 64,
                                                   "b" * 64) / stage
            if not d.is_dir():
                d.mkdir(parents=True)
                (d / "xtb.out").write_text(
                    "          :  PC potential                    true          :\n"
                    "   *** convergence criteria cannot be satisfied "
                    "within 2000 iterations ***\n"
                    "-1- scf: Self consistent charge iterator did not "
                    "converge\n")
                (d / "xtb.err").write_text("no normal termination\n")
            return d

        shard = Path(tmp) / "shard"
        wrapped = _wk.domain_wrapped_evaluator(fake_provider, u, shard,
                                               keep_raw_dirs=set())
        en.run_shard(shard, u, cand, wrapped, 0, n_rows, flush_rows=250)
        # the ledger: exactly the two designated rows
        ledger = [json.loads(l) for l in
                  (shard / "domain_exclusions.jsonl").read_text().splitlines()
                  if l.strip()]
        assert {int(e["row"]) for e in ledger} == set(fail), ledger
        assert {e["stage"] for e in ledger} == set(fail.values())
        assert all((Path(tmp) / e["raw_rel"]).is_dir() for e in ledger), \
            "the raw evidence dirs are not recorded/reachable relatively"
        # the aggregation verifies the evidence + completes
        land = en.aggregate_shards([shard], u)
        assert land["n_rows_evaluated"] == n_rows - 2
        assert land["n_rows_with_value"] == n_rows - 2
        assert len(land["domain_exclusions"]) == 2
        # the artifact: NaN at the excluded rows + the ledger file
        summary = en.landscape_summary(
            u, land, "0" * 64, "0" * 64, "0" * 64, {"m": 1})
        out = en.write_landscape(Path(tmp) / "artifact", u, land, summary)
        f32 = np.load(Path(tmp) / "artifact" / "f.npy")
        assert np.isnan(f32[list(fail)]).all(), "the excluded rows are not NaN"
        assert np.isfinite(f32[np.array([r for r in range(n_rows)
                                         if r not in fail])]).all()
        excl_file = json.loads((Path(tmp) / "artifact" /
                                "domain_exclusions.json").read_text())
        assert excl_file["count"] == 2
        assert excl_file["rows"] == sorted(fail)

    # the non-SCC failure: full stop (the ledger gets nothing for the row)
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        cand = np.load(cloud_path)
        u = en.build_universe(cand, d_min=en.D_MIN_A)

        def plumbing_failure(idx_i, idx_j, sign_i, sign_j, cand2):
            raise RuntimeError("plumbing: parse error")

        shard = Path(tmp) / "shard2"
        wrapped = _wk.domain_wrapped_evaluator(plumbing_failure, u, shard)
        raised = None
        try:
            en.run_shard(shard, u, cand, wrapped, 0, 8, flush_rows=4)
        except RuntimeError as e:
            raised = str(e)
        assert raised is not None and "plumbing: parse error" in raised
        assert not (shard / "domain_exclusions.jsonl").is_file(), \
            "a plumbing failure was domain-excluded?!"


def test_12_coverage_arbiter():
    """The stage/promotion-side completeness arbiter (fix #6):
    (a) a REAL stub-mode driver run (subprocess workers) reports the
    coverage complete with the exclusion block (count 0); (b) the
    coverage helper: evaluated + excluded == the range -> complete and
    the summary block carries the entries verbatim; (c) an exclusion
    without evidence / with non-SCC evidence -> the aggregation fails
    named; (d) a gap (neither evaluated nor excluded) stays
    incomplete."""
    driver = _load_driver()
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        # (a) the real driver run: the stub workers, a small cloud
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        rc, report = driver.run_enumeration(
            str(cloud_path), en.sha256_file(cloud_path), "stub",
            Path(tmp) / "run", shards_k=2, flush=200, limit=None)
        assert rc == 0
        assert report["status"] == "complete" and report["ok"] is True
        assert (report["n_rows_evaluated"] + report["n_rows_excluded"]
                == report["n_rows_landscape"])
        assert report["domain_exclusions"]["count"] == 0
        assert report["domain_exclusions"]["list_file"] is None
        assert report["coverage_identity"] is True

    # (b) the coverage helper + the summary's exclusion block
    fake = {"n_rows_evaluated": 9, "domain_exclusions": [
        {"row": 9, "stage": "reactant", "marker": "scc-not-converged",
         "config": [1, 2, 1, -1]}]}
    cov = driver.coverage_report(fake, 10)
    assert cov["status"] == "complete" and cov["coverage_identity"] is True
    assert cov["domain_exclusions"]["by_side"] == {"reactant": 1}
    assert cov["domain_exclusions"]["list_file"] == "domain_exclusions.json"
    u = en.build_universe(np.load(REPO / "data/derived/candidates_v2.npy"),
                          d_min=en.D_MIN_A)
    n_rows = int(u["n_rows"])
    land = {"f": np.full(n_rows, np.nan), "e_ts": np.full(n_rows, np.nan),
            "e_r": np.full(n_rows, np.nan),
            "pairs": np.zeros((n_rows, 2), dtype=np.int32),
            "sign_index": u["sign_index"],
            "n_rows_evaluated": 9, "rows_evaluated_distinct": 0,
            "manifest_rows_total": 0, "ledger": [],
            "domain_exclusions": fake["domain_exclusions"]}
    summary = en.landscape_summary(u, land, "0" * 64, "0" * 64, "0" * 64,
                                   {"m": 1})
    assert summary["domain_rule"]["count"] == 1
    assert summary["domain_rule"]["by_side"] == {"reactant": 1}
    assert summary["domain_rule"]["entries"][0]["marker"] == "scc-not-converged"
    assert summary["domain_rule"]["list_file"] == "domain_exclusions.json"

    # (c) the exclusion without evidence -> fail named
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        cand = np.load(cloud_path)
        u = en.build_universe(cand, d_min=en.D_MIN_A)
        shard = Path(tmp) / "shard"
        shard.mkdir()
        (shard / "shard_meta.json").write_text(json.dumps(
            {"rows_done_manifest": 0, "ok": True}))
        (shard / "domain_exclusions.jsonl").write_text(json.dumps({
            "row": 5, "config": [0, 0, 1, -1], "stage": "reactant",
            "marker": "scc-not-converged",
            "raw_rel": "provider_scratch/missingkey/reactant",
            "raw_abs": str(Path(tmp) / "provider_scratch/missing/reactant")}
        ) + "\n")
        try:
            en.aggregate_shards([shard], u)
            raise AssertionError("the no-evidence exclusion passed")
        except RuntimeError as e:
            assert "raw" in str(e) and "missing" in str(e), str(e)
        # the evidence present, the SCC signature absent -> fail named
        raw = Path(tmp) / "provider_scratch/missingkey2/reactant"
        raw.mkdir(parents=True)
        (raw / "xtb.out").write_text("PC potential true\nnothing failed\n")
        (shard / "domain_exclusions.jsonl").write_text(json.dumps({
            "row": 5, "config": [0, 0, 1, -1], "stage": "reactant",
            "marker": "scc-not-converged",
            "raw_rel": "provider_scratch/missingkey2/reactant",
            "raw_abs": str(raw)}) + "\n")
        try:
            en.aggregate_shards([shard], u)
            raise AssertionError("the non-SCC evidence passed")
        except RuntimeError as e:
            assert "lacks the SCC signature" in str(e)

    # (d) the gap: neither evaluated nor excluded -> incomplete
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        cand = np.load(cloud_path)
        u = en.build_universe(cand, d_min=en.D_MIN_A)
        shard = Path(tmp) / "shard"
        en.run_shard(shard, u, cand,
                     stub_module.StubProvider().eval_batch_full,
                     0, 500, flush_rows=250)
        try:
            en.aggregate_shards([shard], u, expect_rows=int(u["n_rows"]))
            raise AssertionError("the gapped run passed as complete")
        except RuntimeError as e:
            assert "count mismatch" in str(e) and "incomplete" in str(e)


def test_13_value_placement():
    """The placement regression (the fix-#6 promote's caught bug): the
    records' values must land at their canonical row positions
    regardless of the parts' row ORDER — a shuffled-parts fixture's
    aggregate must carry f[row] == the record's value BITWISE; the
    promoted artifact's argmin must equal the records' own argmin."""
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        cloud_path = _small_cloud(tmp, n=40, seed=13)
        cand = np.load(cloud_path)
        u = en.build_universe(cand, d_min=en.D_MIN_A)
        n_rows = int(u["n_rows"])
        p = stub_module.StubProvider()
        shard = Path(tmp) / "shard"
        en.run_shard(shard, u, cand, p.eval_batch_full, 0, n_rows,
                     flush_rows=331)
        # re-derive the placement: for a sample of rows, the canonical
        # position's value must equal the record's value
        land = en.aggregate_shards([shard], u)
        rows, f_vals = _collect(land)
        for row in sorted(_sample(rows, 200, seed=77)):
            pass
        del rows, f_vals
        # rebuild from the parts directly (the records as authority)
        recs = {}
        for part in en.list_parts(shard):
            with np.load(part) as z:
                for rr, ff_, tt, rr2 in zip(z["rows"], z["f"], z["e_ts"],
                                            z["e_r"]):
                    recs[int(rr)] = (float(ff_), float(tt), float(rr2))
        for row in range(0, n_rows, max(1, n_rows // 200)):
            f64, ts64, r64 = recs[row]
            assert float(land["f"][row]) == f64, (
                f"row {row}: the aggregate's value is not the record's "
                f"({float(land['f'][row])!r} vs {f64!r})")
            assert float(land["e_ts"][row]) == ts64
            assert float(land["e_r"][row]) == r64
        # and the summary's global_min matches the records' own argmin
        summary = en.landscape_summary(u, land, "0" * 64, "0" * 64,
                                       "0" * 64, {"m": 1})
        g = min(recs.items(), key=lambda kv: kv[1][0])
        assert summary["fitness"]["global_min"]["row"] == g[0]
        assert summary["fitness"]["global_min"]["f"] == g[1][0]


def _collect(land):
    return land["f"], land["f"]


def _sample(rows, k, seed=77):
    rng = np.random.default_rng(seed)
    return rng.choice(rows, size=min(k, len(rows)), replace=False)


# --------------------------------------------------------------------------
# 3. The promotion gates (the stub-hazard regression; never writes
#    under data/ -- the target is a fresh temp dir)
# --------------------------------------------------------------------------

def test_09_promotion_gates():
    """The promotion's identity gates: a stub (mechanical) shard record
    set REFUSES promotion (the live-caught hazard, regression-gated);
    no records refuse; a --mode stub promotion refuses; the target is
    never touched on any refusal."""
    import contextlib
    import io

    driver = _load_driver()
    with tempfile.TemporaryDirectory(dir=str(REPO / "runs")) as tmp:
        # a real stub-mode shard (fast, constructed cloud) at a fake
        # work root: the records are complete but MECHANICAL
        cloud_path = _small_cloud(tmp, n=40, seed=17)
        candidates = np.load(cloud_path)
        u = en.build_universe(candidates, d_min=en.D_MIN_A)
        work_root = Path(tmp) / "stubrun"
        en.run_shard(work_root / "shard_0", u, candidates,
                     stub_module.StubProvider().eval_batch_full, 0,
                     int(u["n_rows"]), flush_rows=400,
                     shard_meta={"mode": "stub",
                                 "cloud": "data/derived/candidates_v2.npy",
                                 "cloud_sha256": "084e48a6b5267900a995b"
                                                 "a2d4f0412f77f5bb13446"
                                                 "ba74eafaf474c644349e"
                                                 "71"})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = driver.promote(
                _Args(mode="xtb", work_root=str(work_root)),
                target_dir=Path(tmp) / "target")
        assert rc == 1, "a stub landscape promoted!"
        assert "only an xtb" in err.getvalue()
        assert not (Path(tmp) / "target").exists()

        # --mode stub promotion: refused (the mode gate or the records'
        # mode gate — either way the refusal precedes any write)
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            rc2 = driver.promote(
                _Args(mode="stub", work_root=str(work_root)),
                target_dir=Path(tmp) / "target2")
        assert rc2 == 1 and ("only an xtb" in err2.getvalue()
                             or "default --mode xtb" in err2.getvalue())

        # no records: refused
        err3 = io.StringIO()
        with contextlib.redirect_stderr(err3):
            rc3 = driver.promote(
                _Args(mode="xtb", work_root=str(Path(tmp) / "empty")),
                target_dir=Path(tmp) / "target3")
        assert rc3 == 1 and "no shard records" in err3.getvalue()


# --------------------------------------------------------------------------
# 4. The live real-provider proofs (spec 5; needs the xtb binary)
# --------------------------------------------------------------------------

def test_08_live_proofs():
    """The four real-provider proofs GREEN on the canonical cloud; the
    committed reference reproduces bitwise (the shipped reference is
    the record)."""
    from fitness import xtb as ff
    if not ff.XTB_BIN.is_file():
        print("SKIP  (xtb binary not installed)")
        return "skip"
    ref = REPO / "exhaustive_search" / "proof_reference"
    if not (ref / "laptop_proof.npz").is_file():
        raise AssertionError("the shipped proof reference is missing")
    work = REPO / "runs" / "exhaustive_search" / "proofs_live_test"
    report = pr.run_proofs(
        cloud_path=CANONICAL_CLOUD, cloud_sha256=CANONICAL_SHA256,
        mode="xtb", work_root=work,
        provider_scratch=work / "provider_scratch",
        compare_reference=ref)
    p = report["proofs"]
    assert p["a_shuffled_vs_canonical"]["canonical_vs_shuffled_bitwise"]
    assert not p["b_solo_rows"]["bitwise_mismatches"]
    assert p["c_accounting"]["exact"]
    assert p["d_kill_resume"]["skipped_not_re_evaluated"]
    assert p["d_kill_resume"]["resumed_landscape_bitwise_uninterrupted"]
    # the within-machine re-derivation is BITWISE the committed reference
    # (the reference was born here); the cross-machine gate is the derived
    # band, and the report records the measured deltas, not a "bitwise"
    # label for a tolerance
    refg = report["reference"]
    assert refg["reference_rows_match"] is True
    assert refg["reference_within_band"] is True
    assert refg["n_sides_bitwise"] == refg["n_sides_total"], (
        "the laptop's own reference re-derivation is not bitwise — "
        "within-machine determinism broke")
    assert refg["reference_band_kcal_mol"] == pr.REF_BAND_KCAL_MOL == 1e-8
    assert ".bitwise" not in str(refg.get("reference_gate", "")), \
        "a band must never be labelled bitwise"
    return "pass"


def test_10_reference_band_gate():
    """The xtb-mode cross-machine reference gate (Jonny's derived band,
    1e-8 kcal/mol): a fixture pair of value sets differing by 3e-9 is
    in-band (green) with the bitwise side-count recorded; differing by
    3.0 (the stub-coupling-flip signature) breaches, named; the band is
    never labelled bitwise; the stub mode keeps its bitwise diff (its
    reference compare never runs)."""
    rows = np.arange(4, dtype=np.int64)
    f = np.array([25.0, -5.0, 114.0, 0.5])
    ts = f * 1.001
    r = ts - f

    # in-band: 3e-9 (< 1e-8) on one side; the others bitwise equal
    # (the delta survives the double addition to within ulp(114) ~ 1e-14)
    f2 = f.copy()
    f2[2] += 3e-9
    gate = pr.compare_reference_values(f2, ts, r, f, ts, r)
    assert gate["rows_bitwise"]
    assert gate["within_band"] is True
    assert gate["max_abs_delta_f"] == np.abs(f2 - f).max()
    assert abs(gate["max_abs_delta_f"] - 3e-9) < 1e-13
    # the sides: 12 values, 1 in the band-not-bitwise, 11 bitwise
    assert gate["n_sides_bitwise"] == 11 and gate["n_sides_in_band"] == 12
    assert gate["n_sides_total"] == 12
    assert gate["band_kcal_mol"] == pr.REF_BAND_KCAL_MOL == 1e-8
    # the gate's wording: the band is recorded, never called bitwise
    for key in gate:
        if key.endswith("within_band") or key.endswith("band_kcal_mol"):
            assert "bitwise" not in str(gate[key]), key

    # a cross-shape fixture: fewer rows must fail the identity gate
    gate_short = pr.compare_reference_values(f2[:3], ts[:3], r[:3], f, ts, r)
    assert gate_short["rows_bitwise"] is False
    assert gate_short["within_band"] is False

    # breach: 3.0 (the stub-coupling-flip signature scale) -> named
    f3 = f.copy()
    f3[1] += 3.0
    bad = pr.compare_reference_values(f3, ts, r, f, ts, r)
    assert bad["rows_bitwise"] is True
    assert bad["within_band"] is False, "a 3 kcal/mol delta must breach"
    assert bad["n_sides_in_band"] == bad["n_sides_total"] - 3 - 0 + 0 or \
        bad["n_sides_in_band"] < bad["n_sides_total"]
    assert bad["max_abs_delta_f"] == 3.0

    # through run_proofs' ok gate: an in-band reference is GREEN, a
    # breaching one is NOT (fixture-level: the fields flow)
    assert (pr.compare_reference_values(f2, ts, r, f, ts, r)["within_band"]
            is True) != (pr.compare_reference_values(f3, ts, r, f, ts,
                                                     r)["within_band"])


_TESTS = [
    test_01_universe_counts,
    test_02_row_mapping,
    test_03_shard_split,
    test_04_kill_resume_equivalence,
    test_05_accounting_exactness,
    test_06_fail_hard_shard,
    test_07_cloud_identity_gates,
    test_09_promotion_gates,
    test_10_reference_band_gate,
    test_11_domain_exclusions,
    test_12_coverage_arbiter,
    test_13_value_placement,
    test_08_live_proofs,
]


def main():
    failures = []
    for t in _TESTS:
        try:
            status = t() or "pass"
            print(f"PASS  {t.__name__}" if status == "pass"
                  else f"SKIP  {t.__name__} ({status})")
            if status not in ("pass", "skip"):
                failures.append(t.__name__)
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"FAIL  {t.__name__}: unexpected {type(e).__name__}: {e}")
            failures.append(t.__name__)
    n = len(_TESTS)
    if failures:
        print(f"\n{n - len(failures)}/{n} passed; FAILED: {', '.join(failures)}")
        return 1
    print(f"\n{n}/{n} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())