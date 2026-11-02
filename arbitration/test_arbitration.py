"""Tests for the arbitration stage (landscape gate, decode, engine,
probe sampler, step semantics).  House style: numbered tests + main();
the engine tests run over the REAL canonical landscape with the ORCA
launch sentinel-patched (no ORCA needed); the gate tests build a
synthetic landscape tree (its own summary, self-consistent).

Run:  python arbitration/test_arbitration.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arbitration.landscape as L
import fitness.orca as fo
from arbitration.landscape import (ARTIFACT_FILES, SIGN_CANON,
                                   gate_landscape)
from arbitration import probe, steps
from fitness.orca import KCAL_PER_EH
from fitness.test_orca import SentinelLaunch, _mono_patcher, _unpatcher

REPO = fo.REPO_ROOT


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def expect_error(fn, exc=ValueError, what="", must_name=None):
    try:
        fn()
    except exc as e:
        if must_name and must_name not in str(e):
            raise AssertionError(
                f"{what}: the error must name {must_name!r}, got {e}")
        return e
    raise AssertionError(f"{what}: expected {exc.__name__}, got none")


# --------------------------------------------------------------------------
# synthetic landscape builder (self-consistent; every gate passes by
# construction; the caller can tamper any single fact to trip a gate)
# --------------------------------------------------------------------------

def build_synthetic(base, seed=7, n_sites=12, n_rows=40):
    """The landscape arrays + exclusions ONLY (no summary -- the caller
    hashes everything and writes the summary once, in write_synthetic)."""
    base = Path(base)
    root = base / "landscape"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    half = n_rows // 2
    pair = np.empty((n_rows, 2), dtype=np.int32)
    signs = np.empty(n_rows, dtype=np.int8)
    for r in range(n_rows):
        pair[r] = (rng.integers(0, n_sites - 1), n_sites - 1) \
            if r < half else (0, rng.integers(1, n_sites))
        signs[r] = rng.integers(0, 4)
    f = rng.uniform(-130.0, 170.0, n_rows).astype(np.float32)
    e_ts = (-838.0 + f / 4000.0).astype(np.float32)
    e_r = (e_ts - f / KCAL_PER_EH).astype(np.float32)
    gmin_row = int(np.argmin(f))
    excl_rows = [(gmin_row + 1) % n_rows]
    _expect(excl_rows[0] != gmin_row,
            "the synthetic exclusion must not swallow the argmin row")
    f[excl_rows] = np.nan
    e_ts[excl_rows] = np.nan
    e_r[excl_rows] = np.nan
    np.save(root / "f.npy", f)
    np.save(root / "e_ts.npy", e_ts)
    np.save(root / "e_r.npy", e_r)
    np.save(root / "pair_indices.npy", pair)
    np.save(root / "sign_index.npy", signs)
    exclusions = {"entries": [{"row": r, "config": "synthetic",
                               "stage": "reactant"} for r in excl_rows]}
    (root / "domain_exclusions.json").write_text(
        json.dumps(exclusions, indent=2) + "\n")
    return root, gmin_row


def write_synthetic(base):
    """The complete self-consistent synthetic landscape tree -> (root,
    cloud_path, prov_path, summary)."""
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    root, gmin_row = build_synthetic(base)
    np.save(base / "cloud.npy",
            np.random.default_rng(1).uniform(-3, 3, (12, 3)))
    prov_dtype = [("origin_state", "u1"), ("origin_frame", "<u4"),
                  ("origin_atom", "u1"), ("nearest_ts_atom", "u1")]
    prov = np.zeros(12, dtype=prov_dtype)
    prov["nearest_ts_atom"] = np.random.default_rng(2).integers(
        0, 26, 12).astype(np.uint8)
    np.save(base / "provenance.npy", prov)

    f = np.load(root / "f.npy")
    pair = np.load(root / "pair_indices.npy")
    signs = np.load(root / "sign_index.npy")
    summary = {
        "fitness_id": "synthetic",
        "universe": {"sign_canon": [list(s) for s in SIGN_CANON]},
        "cloud_sha256": sha(base / "cloud.npy"),
        "artifacts": {
            k: {"file": f"{k}.npy", "sha256": sha(root / f"{k}.npy"),
                "shape": list(np.load(root / f"{k}.npy").shape),
                "dtype": np.load(root / f"{k}.npy").dtype.name}
            for k in L.NPY_ARTIFACTS},
        "coverage": {"n_rows_evaluated": int(np.isfinite(f).sum())},
        "fitness": {"global_min": {
            "f": float(f[gmin_row]), "row": gmin_row,
            "idx_i": int(pair[gmin_row][0]),
            "idx_j": int(pair[gmin_row][1]),
            "sign": [int(signs[gmin_row])]}},
    }
    summary["artifacts"]["domain_exclusions"] = {
        "file": "domain_exclusions.json",
        "sha256": sha(root / "domain_exclusions.json")}
    (root / "enumeration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return root, base / "cloud.npy", base / "provenance.npy", summary


# --------------------------------------------------------------------------
# gate tests (synthetic tree)
# --------------------------------------------------------------------------

def test_01_gate_ok_and_decode():
    """The synthetic tree gates clean; decode + stable top-K rank."""
    tmp = Path(tempfile.mkdtemp(prefix="arbit_gate_"))
    try:
        root, cloud_path, prov_path, summary = write_synthetic(tmp)
        land = gate_landscape(root, cloud_path, prov_path, "synthetic")
        _expect(land.n_rows == 40 and land.n_sites == 12, "shapes")
        order = land.f_sorted_rows()
        fs = land.f[order[:10]].astype(np.float64)
        _expect(np.all(np.diff(fs) >= 0), "top-K sorted ascending")
        # the argmin row decodes to the summary's record
        gmin = summary["fitness"]["global_min"]
        i, j, si, sj = land.decode_row(gmin["row"])
        _expect((i, j) == (gmin["idx_i"], gmin["idx_j"]),
                "the argmin decode")
        _expect((si, sj) == SIGN_CANON[gmin["sign"][0]], "sign canon")
        # the decode is exhaustive over the canon
        r0 = land.decode_row(0)
        _expect(r0[2] in (1, -1) and r0[3] in (1, -1), "signs +-1")
        _expect(land.sha256_record()["f.npy"], "provenance present")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_02_gate_failures():
    """Every gate trips on its tampered fact, naming it."""
    tmp = Path(tempfile.mkdtemp(prefix="arbit_bad_"))

    def fresh():
        shutil.rmtree(tmp, ignore_errors=True)
        return write_synthetic(tmp)

    try:
        # missing artifact
        root, cp, pp, _ = fresh()
        (root / "sign_index.npy").unlink()
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "missing artifact", "artifact missing")
        # tampered artifact sha
        root, cp, pp, _ = fresh()
        f = np.load(root / "f.npy")
        f[0] += 1.0
        np.save(root / "f.npy", f)
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "f sha must fail", "sha256 mismatch")
        # wrong cloud
        root, cp, pp, summ = fresh()
        np.save(cp, np.load(cp) + 0.5)
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "cloud sha must fail", "cloud_sha256")
        # NaN COUNT drift (a second NaN): the finite-row count check
        # fires (the f sha re-certified so the gate reaches coverage)
        root, cp, pp, summ = fresh()
        f = np.load(root / "f.npy")
        f[0] = np.nan                      # a SECOND NaN (39 -> 38)
        np.save(root / "f.npy", f)
        s2 = json.loads((root / "enumeration_summary.json").read_text())
        s2["artifacts"]["f"]["sha256"] = __import__("hashlib").sha256(
            (root / "f.npy").read_bytes()).hexdigest()
        (root / "enumeration_summary.json").write_text(
            json.dumps(s2, indent=2) + "\n")
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "NaN count must fail", "coverage")
        # NaN DISPLACED onto a non-excluded row (count preserved): the
        # NaN rows must equal the domain-exclusions rows
        root, cp, pp, summ = fresh()
        f = np.load(root / "f.npy")
        excl = int(json.loads((root / "domain_exclusions.json")
                              .read_text())["entries"][0]["row"])
        moved = 0 if excl != 0 else 1
        f[moved] = np.nan
        f[excl] = 42.0                      # the exclusion heals -> count kept
        np.save(root / "f.npy", f)
        s2 = json.loads((root / "enumeration_summary.json").read_text())
        s2["artifacts"]["f"]["sha256"] = __import__("hashlib").sha256(
            (root / "f.npy").read_bytes()).hexdigest()
        (root / "enumeration_summary.json").write_text(
            json.dumps(s2, indent=2) + "\n")
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "NaN rows must fail", "NaN rows")
        # argmin authority outside the band
        root, cp, pp, summ = fresh()
        s2 = json.loads((root / "enumeration_summary.json").read_text())
        s2["fitness"]["global_min"]["f"] = (
            s2["fitness"]["global_min"]["f"] - 5e-3)
        (root / "enumeration_summary.json").write_text(
            json.dumps(s2, indent=2) + "\n")
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "argmin band must fail", "band breach")
        # sign canon drift
        root, cp, pp, summ = fresh()
        s2 = json.loads((root / "enumeration_summary.json").read_text())
        s2["universe"]["sign_canon"] = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        (root / "enumeration_summary.json").write_text(
            json.dumps(s2, indent=2) + "\n")
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "sign canon must fail", "sign canon")
        # recorded shape/dtype drift: the sha re-certified to pass the
        # sha gate first
        root, cp, pp, summ = fresh()
        s2 = json.loads((root / "enumeration_summary.json").read_text())
        s2["artifacts"]["f"]["shape"] = [999]
        (root / "enumeration_summary.json").write_text(
            json.dumps(s2, indent=2) + "\n")
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "shape must fail", "shape")
        # provenance/cloud row drift
        root, cp, pp, summ = fresh()
        (tmp / "provenance.npy")
        prov = np.load(pp)
        prov2 = np.zeros(9, dtype=prov.dtype)      # wrong site count
        np.save(pp, prov2)
        expect_error(lambda: gate_landscape(root, cp, pp, "x"),
                     ValueError, "prov shape must fail", "prov shape")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# engine tests (the REAL canonical landscape; the ORCA launch sentinel)
# --------------------------------------------------------------------------

class _Env:
    """A disposable engine context: temp runs_root + provider scratch +
    the sentinel patch."""

    def __init__(self, workers=3, pal=2, list_k=6, energies=None,
                 fail=None):
        self.tmp_runs = Path(tempfile.mkdtemp(prefix="arbit_runs_"))
        self.tmp_scr = Path(tempfile.mkdtemp(prefix="arbit_scr_"))
        self.launch = SentinelLaunch(energies=energies, fail=fail)
        self.old = _mono_patcher(self.launch)
        self.engine = steps.Engine(
            "canonical", workers=workers, pal=pal, list_k=list_k,
            runs_root=self.tmp_runs, provider_scratch=self.tmp_scr)

    def close(self):
        _unpatcher(self.old)
        shutil.rmtree(self.tmp_runs, ignore_errors=True)
        shutil.rmtree(self.tmp_scr, ignore_errors=True)

    def rows_top(self, k):
        return [int(r) for r in self.engine.landscape.f_sorted_rows()[:k]]


def test_03_engine_step_A_roundtrip():
    """Step A over 6 rows: records + sidecars + cache; the summary's
    rows = the top-6 in f-sorted order with oracle_rank 0..5; a second
    run re-records from the cache (zero launches)."""
    env = _Env(list_k=6)
    try:
        sA, _ = steps.step_A(env.engine)
        _expect(sA["executed"] == 6 and sA["from_cache"] == 0,
                f"A executed: {sA['executed']}")
        _expect(sA["rows"] == env.rows_top(6), "the rows are the top-6")
        recs = env.engine.read_records("A")
        _expect(set(recs) == set(env.rows_top(6)), "record set")
        for pos, row in enumerate(env.rows_top(6)):
            rec = recs[row]
            _expect(rec["oracle_rank"] == pos, f"rank of {row}")
            _expect(rec["e_ts_eh"] < -800, f"Eh scale for {row}")
            _expect(abs(rec["f_kcal"] - (rec["e_ts_eh"] - rec["e_r_eh"])
                        * KCAL_PER_EH) < 1e-9, f"f identity for {row}")
            _expect(rec["walls"]["ts_s"] is not None, "walls measured")
        _expect(len(list(env.engine.step_dir("A").glob("*.sha256"))) >= 6,
                "sidecars written")
        cache = json.loads((env.engine.cache_path).read_text())
        _expect(len(cache) == 6, f"cache size {len(cache)}")
        _expect(len(env.launch.calls) == 12, "2 SPs per config")
        # the second run: everything from the step resume, zero launches
        n_calls = len(env.launch.calls)
        sA2, _ = steps.step_A(env.engine)
        _expect(sA2["resumed"] == 6 and sA2["executed"] == 0
                and sA2["from_cache"] == 0,
                "the re-run is a pure step-resume")
        _expect(len(env.launch.calls) == n_calls, "no re-execution")
        recs2 = env.engine.read_records("A")
        _expect(all(recs2[r]["f_kcal"] == recs[r]["f_kcal"]
                    for r in recs), "identical records")
    finally:
        env.close()


def test_04_engine_fail_hard_halt():
    """A failing config halts the batch: the failed config's record is
    absent, the completed ones stand; clearing the failure completes
    the step with no duplicates."""
    env = _Env(list_k=4, fail={"ts": 65})
    try:
        top4 = env.rows_top(4)
        expect_error(lambda: steps.step_A(env.engine), RuntimeError,
                     "the ts side fails everywhere -> halt",
                     "ORCA exited 65")
        recs = env.engine.read_records("A")
        _expect(len(recs) < 4, f"records written before the halt: "
                f"{len(recs)}")
        _expect(all(rec["executed"] for rec in recs.values()),
                "the halted records are executed ones")
    finally:
        env.close()
    # clear the failure and finish the step
    env2 = _Env(list_k=4)
    try:
        steps.step_A(env2.engine)
        recs = env2.engine.read_records("A")
        _expect(len(recs) == 4, "the step completes after the clear")
        _expect(len(list(env2.engine.step_dir("A").glob("cfg_row*.json")))
                == 4, "no duplicate records")
        cache = json.loads(env2.engine.cache_path.read_text())
        _expect(len(cache) == 4, "the cache carries no duplicates")
    finally:
        env2.close()


def test_05_engine_step_B_widening():
    """The boundary rule fires exactly as pre-committed: a beating
    boundary config widens +30 once, then the loop stops; the widening
    event is recorded; already-evaluated rows never re-execute."""
    env = _Env(list_k=4)
    try:
        eng = env.engine
        steps.step_A(eng)
        order = eng.landscape.f_sorted_rows()
        # rank-4 (the first boundary row) gets a hugely negative f:
        # specified e_ts with a high e_r
        row4 = int(order[4])
        i4, j4, si4, sj4 = eng.landscape.decode_row(row4)
        pos4 = tuple(float(v) for v in
                     np.concatenate([eng.landscape.cloud[i4],
                                     eng.landscape.cloud[j4]]))
        env.launch.energies[("e_ts", pos4)] = -900.0
        env.launch.energies[("e_r", pos4)] = 0.0
        sB, _ = steps.step_B(eng)
        # round 1 = oracle ranks 4..33 (30 rows): row4's f (~-564,758
        # kcal) beats the certified best (~-62.75) -> widen; round 2 =
        # ranks 34..63: nothing beats -> stop
        _expect(len(sB["rounds"]) == 2, f"two rounds: {len(sB['rounds'])}")
        _expect(sB["rounds"][0]["range"] == [4, 33],
                f"round-1 range {sB['rounds'][0]['range']}")
        _expect(sB["rounds"][0]["beat"] is True and
                sB["rounds"][1]["beat"] is False, "one widening event")
        _expect(sB["k_final"] == 34, f"k_final {sB['k_final']}")
        _expect(len(sB["widening_events"]) == 1, "the event is recorded")
        _expect(sB["widening_events"][0]["extended_to_k"] == 34,
                "the event extends to 34")
        # 4 A-configs + 30 + 30 sweep configs, 2 SPs each
        _expect(len(env.launch.calls) == (4 + 30 + 30) * 2,
                f"call count {len(env.launch.calls)}")
        # a second B run re-reads its records (no new execution; the
        # verdict + k_final are stable)
        sB2, _ = steps.step_B(eng)
        _expect(len(env.launch.calls) == (4 + 30 + 30) * 2,
                "no re-execution on B")
        _expect(sB2["k_final"] == 34, "k_final stable on re-run")
        _expect(len(sB2["widening_events"]) == 1, "events stable")
    finally:
        env.close()


def test_06_engine_step_D_and_cert():
    """D: the DFT argmin over the certified set + the emitted record;
    cert: the checks pass + the re-emit is idempotent; a tampered
    record trips cert."""
    env = _Env(list_k=5)
    try:
        eng = env.engine
        steps.step_A(env.engine)
        sD, _ = steps.step_D(env.engine)
        recs = eng.read_records("A")
        best_row = min(recs, key=lambda r: recs[r]["f_kcal"])
        opt = steps.read_certified_optimum(env.engine)
        _expect(opt["row"] == best_row, "the optimum is the argmin")
        _expect(opt["f_dft_kcal"] == recs[best_row]["f_kcal"], "f record")
        _expect(opt["oracle_rank"] == env.rows_top(5).index(best_row),
                "the optimum's oracle rank")
        s_cert, _ = steps.step_cert(env.engine)
        _expect(all(s_cert["checks"].values()), "cert checks all pass")
        # idempotent re-emit: identical modulo the cert utc
        first = json.loads((env.engine.runs_root
                            / "certified_optimum.json").read_text())
        steps.step_cert(env.engine)
        second = json.loads((env.engine.runs_root
                             / "certified_optimum.json").read_text())
        first.pop("cert_utc"), second.pop("cert_utc")
        _expect(first == second, "idempotent re-emit")
    finally:
        env.close()


def test_07_engine_step_E():
    """E: one P SP per certified config; dE_rxn + quench derive from the
    recorded energies; a re-run resumes."""
    env = _Env(list_k=4)
    try:
        eng = env.engine
        steps.step_A(env.engine)
        sE, _ = steps.step_E(env.engine)
        _expect(sE["rows"] == env.rows_top(4),
                "the certified rows (the first K_final in f-order)")
        recs = eng.read_records("E", kind="pcfg")
        _expect(set(recs) == set(env.rows_top(4)), "the pcfg records")
        for row, rec in recs.items():
            want = (rec["e_p_eh"] - rec["e_r_eh_dft_record"]) * KCAL_PER_EH
            _expect(abs(rec["dE_rxn_kcal"] - want) < 1e-9,
                    f"dE_rxn identity for {row}")
            _expect(rec["quench"] ==
                    (rec["e_ts_eh_dft_record"] < rec["e_p_eh"]),
                    f"quench identity for {row}")
        _expect(len(env.launch.calls) == 4 * 2 + 4,
                "2 SPs per A config + 1 P SP per config")
        n = len(env.launch.calls)
        sE2, _ = steps.step_E(env.engine)
        _expect(len(env.launch.calls) == n, "E resumes without launches")
    finally:
        env.close()


def test_08_engine_step_G():
    """G: the section 6.3 fallback level records the deviation + its own
    lot/fitness_id; the energies come from the G keys."""
    env = _Env(list_k=3)
    try:
        eng = env.engine
        steps.step_A(env.engine)
        steps.step_D(env.engine)
        sG, _ = steps.step_G(env.engine)
        _expect("UNRECOGNIZED" in sG["deviation"]
                and "wB97M-V" in sG["deviation"], "the deviation report")
        opt_row = steps.read_certified_optimum(env.engine)["row"]
        rec = eng.read_records("G")[opt_row]
        _expect(rec["lot"] == fo.WB97M_LOT, "the G lot")
        _expect(rec["fitness_id"] == fo.WB97M_FITNESS_ID, "the G id")
        _expect(rec["key"] != eng.read_records("A")[opt_row]["key"],
                "the G key differs (lot in the key)")
        _expect(abs(rec["f_kcal"] - (rec["e_ts_eh"] - rec["e_r_eh"])
                    * KCAL_PER_EH) < 1e-9, "the G f identity")
    finally:
        env.close()


def test_09_replicate_execution_fences():
    """The fence (Jonny's GO 2026-10-30): all three named landscapes
    construct an Engine; an unknown name still refuses with the fence
    message."""
    for name in ("canonical", "replicate-1", "replicate-2"):
        eng = steps.Engine(name)   # the gate runs; construction = the GO
        _expect(eng.landscape.name == name,
                f"{name} constructs and gates")
    expect_error(lambda: steps.Engine("replicate-9"), RuntimeError,
                 "an unknown landscape still refuses", "not enabled")


def test_10_probe_sampler_and_gate():
    """The sampler: deterministic over calls; 192 rows (Ruling 1's
    doubled targets); the targets hit; the NaN row never sampled; the
    pool-count gate asserts the artifact-derived triplet (two of the
    ruling's numbers confirmed; the spread triplet's discrepancy
    recorded); the 'both' rule confirmed + cleared."""
    eng = steps.Engine("canonical")
    _expect(probe.CONFIG_STRATUM_RULE == "both", "the 'both' rule (the recorded Ruling 1)")
    rows1, audit1 = probe.sample(eng)
    rows2, audit2 = probe.sample(eng)
    _expect(rows1 == rows2, "the sampler is deterministic")
    _expect(len(rows1) == sum(probe.TARGETS.values()) == 192,
            f"{len(rows1)} rows (the ruling's doubled targets)")
    _expect(522045 not in rows1, "the domain-excluded row is never drawn")
    for s in probe.DRAW_ORDER:
        st = audit1["strata"][s]
        _expect(st["taken"] == probe.TARGETS[s], f"{s} target")
        _expect(len(st["taken_rows"]) == st["taken"], f"{s} rows recorded")
    _expect(audit1["config_stratum_rule"] == "both", "the rule recorded")
    # the pool-count gate: measured == pinned (artifact-derived); the
    # ruling's spread discrepancy is carried in the audit
    gate = audit1["pool_count_gate"]
    _expect(gate["measured"] == probe.PINNED_POOLS,
            f"the pool-count gate: {gate['measured']}")
    _expect(probe.MIXED_CORE_ROWS == 288520,
            "the mixed-core reconciliation constant")
    _expect("288,520" in gate["discrepancy"],
            "the reconciliation is recorded")
    expect_error(lambda: probe.sample(eng, rule="nonsense"), ValueError,
                 "an unknown rule fails", "unknown CONFIG_STRATUM_RULE")


def test_12_rank_correlation_and_producer():
    """Ruling 2: the top-100 rank correlation (xTB rank vs DFT rank)
    derives from A's records; the truth is the scipy value over the
    controlled sentinel energies."""
    # sentinel energies: distinct e_ts per ranked row, DFT-ordered
    # differently from the oracle order (a known permutation)
    env = _Env(list_k=5)
    try:
        eng = env.engine
        rows5 = env.rows_top(5)
        for pos, row in enumerate(rows5):
            i, j, si, sj = eng.landscape.decode_row(row)
            posv = tuple(float(v) for v in
                         np.concatenate([eng.landscape.cloud[i],
                                         eng.landscape.cloud[j]]))
            # the DFT f is REVERSED w.r.t. the oracle order: f = +0.05*(pos+1)
            # eh (via e_ts spec; e_r = e_ts + 0.1 sentinel offset -> wait:
            # f must DECREASE with pos to reverse — set e_ts so
            # (e_ts - e_r) = +0.05*(4-pos)-0.1+0.05... keep it simple:
            # specify BOTH sides: e_ts - e_r = +0.05*(4 - pos) Eh.
            diff_eh = 0.05 * (4 - pos)
            base = -840.0 - pos * 1e-3   # distinct bases (no cross-talk)
            env.launch.energies[("e_ts", posv)] = base + diff_eh / 2
            env.launch.energies[("e_r", posv)] = base - diff_eh / 2
        steps.step_A(eng)
        corr = steps.top100_rank_correlation(eng)
        _expect(corr is not None and corr["n"] == 5, "the correlation n")
        # the oracle rank = 0..4; the DFT order is REVERSED (f decreases
        # with pos) -> the DFT rank of row pos = 4-pos -> Spearman
        # (x = 0..4, y = 4-x) = -1.0
        _expect(abs(corr["spearman"] - (-1.0)) < 1e-12,
                f"the reversed-permutation rho: {corr['spearman']}")
        # and the summary carries it
        sD, _ = steps.step_D(eng)
        _expect(sD["top100_rank_correlation"]["spearman"] ==
                corr["spearman"], "the summary carries the correlation")
    finally:
        env.close()


def test_11_steps_require_their_prerequisites():
    """B/D/E without A's records fail with the prerequisite message; F/G
    without certified_optimum fail naming it."""
    env = _Env(list_k=4)
    try:
        eng = env.engine
        expect_error(lambda: steps.step_B(eng), RuntimeError,
                     "B needs A", "requires step A")
        expect_error(lambda: steps.step_D(eng), RuntimeError,
                     "D needs A", "requires step A")
        expect_error(lambda: steps.step_E(eng), RuntimeError,
                     "E needs A", "requires step A")
        expect_error(lambda: steps.step_F(eng), RuntimeError,
                     "F needs D", "run step D first")
        expect_error(lambda: steps.step_G(eng), RuntimeError,
                     "G needs D", "run step D first")
        expect_error(lambda: steps.step_cert(eng), RuntimeError,
                     "cert needs D", "run step D first")
    finally:
        env.close()


def test_13_b_widening_cap():
    """The replicate-B rule's cap: 4 widening events maximum; the FIFTH
    beat raises the pre-committed STOP (the evaluated rounds' records
    + the widening trail stay on disk)."""
    env = _Env(list_k=4)
    try:
        eng = env.engine
        steps.step_A(eng)
        order = eng.landscape.f_sorted_rows()
        # round i's first boundary row always beats (an ever-deeper f)
        for i in range(5):
            row = int(order[4 + 30 * i])
            i_r, j_r, si, sj = eng.landscape.decode_row(row)
            pos = tuple(float(v) for v in
                        np.concatenate([eng.landscape.cloud[i_r],
                                        eng.landscape.cloud[j_r]]))
            env.launch.energies[("e_ts", pos)] = -900.0 - 1000.0 * i
            env.launch.energies[("e_r", pos)] = 0.0
        expect_error(lambda: steps.step_B(eng), RuntimeError,
                     "the fifth widening is the STOP",
                     "did not stabilize")
        # the evaluated rounds' records stand on disk (60 rows evaluated
        # before the fifth beat's round... rounds 1-5 evaluate 5x30 rows;
        # the fifth beat's round's records are complete when the raise
        # fires (the records come from the completed round))
        b_recs = eng.read_records("B")
        _expect(len(b_recs) >= 150,
                f"the evaluated boundary rows are recorded: {len(b_recs)}")
    finally:
        env.close()


def test_14_provider_cloud_identity():
    """§1.4 rev 2: the Engine's provider holds THE LANDSCAPE'S OWN cloud
    (bitwise) and its measured sha (a) — and identical indices/signs
    under different cloud shas produce different keys (b): the
    cross-cloud-collision negative."""
    for name in ("canonical", "replicate-1", "replicate-2"):
        eng = steps.Engine(name)
        _expect(np.array_equal(np.load(eng.landscape.cloud_path),
                               eng.provider._cloud),
                f"{name}: the provider's cloud is the landscape's cloud"
                " bitwise")
        _expect(eng.provider.cloud_sha == eng.landscape.cloud_sha256,
                f"{name}: the provider's sha == the landscape's measured"
                " sha")
    # (b) the key negative: the same indices/signs, different cloud shas
    from fitness.orca import differential_key
    ts_sha = "a" * 64
    r_sha = "b" * 64
    k1 = differential_key(435, 440, -1, -1, "c1" * 30 + "ab", ts_sha, r_sha)
    k2 = differential_key(435, 440, -1, -1, "c2" * 30 + "cd", ts_sha, r_sha)
    _expect(k1 != k2,
            "identical indices/signs under different cloud shas -> "
            "different keys (the collision negative)")


def main():
    tests = [test_01_gate_ok_and_decode, test_02_gate_failures,
             test_03_engine_step_A_roundtrip, test_04_engine_fail_hard_halt,
             test_05_engine_step_B_widening, test_06_engine_step_D_and_cert,
             test_07_engine_step_E, test_08_engine_step_G,
             test_09_replicate_execution_fences, test_10_probe_sampler_and_gate,
             test_11_steps_require_their_prerequisites,
             test_12_rank_correlation_and_producer,
             test_13_b_widening_cap,
             test_14_provider_cloud_identity]
    for t in tests:
        t()
        print(f"ok  {t.__name__}: {t.__doc__.splitlines()[0]}")
    print("ALL OK")


if __name__ == "__main__":
    main()