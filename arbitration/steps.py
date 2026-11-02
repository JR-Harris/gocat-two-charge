"""The arbitration step engine (item 7, steps A-G + cert).

Single-process orchestration: the cache dict and all bookkeeping live
in this module's Engine; the worker pool (threads) executes ORCA
subprocesses only, one provider call per worker at a time.  cache.json
is written atomically (tmp + rename) under the engine lock at each
config completion -- full cache-resumability at SP granularity (the
provider's per-SP resume gate), config granularity (the cache), and
step granularity (per-step records: a record file present means that
config is complete for that step and is never re-executed).

Steps (each: read inputs -> check resume/cache state -> execute only
what is missing -> write records + sidecars -> write the step summary):

  A  top-K re-rank        K = --list-k (default 100) best oracle rows,
                          2 r2SCAN-3c SPs each (e_ts then e_r).
  B  boundary sweep       oracle ranks K+1..K+30 (then +30 more on a
                          beat, repeating); the pre-committed rule and
                          every widening event are recorded.  Requires
                          step A (the boundary is K+1.. of the SAME
                          f-sorted list, and the opening DFT-best of the
                          certified set comes from A's records).
  C  fidelity probe       (~100 landscape-wide stratified configs --
                          sampler + statistics in arbitration.probe).
  D  certified optimum    DFT argmin over the certified set (A + any
                          widening); emits certified_optimum.json +
                          sidecar + the aggregate step summary
                          (schema arbitration_v1: DFT-best per step,
                          the boundary decision trail, probe stats).
  E  P-side check         every certified config: one SP on the P
                          geometry with the same charges (the provider's
                          P point mode); per config E(P;q), the charged
                          dE_rxn and the quench flag.
  F  relaxation check     the certified optimum only: OptTS+analytic
                          FREQ from ts_opt.xyz with the winning charges
                          embedded (the landed qm/ machinery + the
                          optional pointcharges parameter; PAL 8 /
                          %maxcore 1500 -- the QM-stages resource
                          contract), gate: EXACTLY one imaginary
                          frequency; then a plain Opt of the R geometry
                          with the charges.  The relaxed differential vs
                          the vertical one BOUNDS the frozen-geometry
                          error (never asserts it away).
  G  wB97M check          the certified optimum only, 2 frozen SPs at
                          wB97M-V def2-TZVPP TightSCF -- plain wB97M is
                          UNRECOGNIZED in ORCA 6.1.1 (measured
                          2026-10-28, single-atom probe; wB97M-V carries
                          VV10 intrinsically) -- the spec section 6.3
                          prescribed fallback, reported as a deviation.
  cert                    re-verify the certified optimum against the
                          per-config records and re-emit
                          certified_optimum.json + sidecar (idempotent).

Fail-hard: a failing config (the provider raises naming the key + the
sp.out tail) cancels the remaining futures, waits for the in-flight
ORCA processes, and halts the batch -- no retries, no workarounds, no
penalty values.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as sps

from arbitration import landscape as L
from arbitration.heartbeat import Heartbeat
from fitness import orca as orca_prov
from fitness.orca import KCAL_PER_EH, geo_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = "arbitration_v1"

# The landscapes ACCEPTED for execution. The replicate names joined on
# Jonny's GO 2026-10-30 (the replicate-DFT-arbitration briefing): the
# canonical item-7 chain is closed (A-E, G, cert), so the fence now
# covers the remaining report work (A/B/D on the two replicate
# landscapes). Every other landscape still refuses.
EXECUTION_ALLOWED = ("canonical", "replicate-1", "replicate-2")

WIDENING_BATCH = 30     # step B: extend K by +30 per widening event

# Ruling 3 (Jonny, 2026-10-28): (16, 1) is the intended production cell;
# a 12-worker fallback is permitted ONLY after an actual failure on
# (16, 1).  The engine + CLI defaults carry it; the failover rule text
# lives in the launch plan (runs/arbitration/canonical/diagnosis/).
PRODUCTION_WORKERS = 16
PRODUCTION_PAL = 1


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Engine:
    """The orchestration state for one landscape + one run's CLI options."""

    def __init__(self, landscape_name, workers=PRODUCTION_WORKERS,
                 pal=PRODUCTION_PAL, list_k=100,
                 maxcore=orca_prov.BASELINE_MAXCORE_MB,
                 runs_root=None, provider_scratch=None):
        if landscape_name not in EXECUTION_ALLOWED:
            raise RuntimeError(
                f"landscape {landscape_name!r}: execution is not enabled "
                f"for this name (known landscapes: {EXECUTION_ALLOWED})")
        self.landscape = L.load_and_gate_landscape(landscape_name)
        self.workers = int(workers)
        self.pal = int(pal)
        self.maxcore = int(maxcore)
        self.list_k = int(list_k)
        self.runs_root = (Path(runs_root) if runs_root is not None
                          else REPO_ROOT / "runs" / "arbitration"
                          / landscape_name)
        self.cache_path = self.runs_root / "cache.json"
        self.heartbeat = Heartbeat(path=self.runs_root / "heartbeat.json")
        self._lock = threading.Lock()
        self.cache = self._load_cache()
        # the r2SCAN-3c provider instance (execution parameters on it;
        # never part of the config key).  provider_scratch is the tests'
        # seam (default: the canonical scratch root).  §1.1 rev 2
        # (2026-10-31): the provider carries THE LANDSCAPE'S OWN cloud
        # identity — the canonical-default drift (the replicate-1 step-A
        # incident) cannot recur by construction.
        self.provider = orca_prov.OrcaProvider(
            pal=self.pal, maxcore=self.maxcore,
            cloud_path=self.landscape.cloud_path,
            cloud_sha256=self.landscape.cloud_sha256,
            scratch_root=(provider_scratch if provider_scratch is not None
                          else orca_prov.SCRATCH_ROOT))
        # §1.2 rev 2: the post-construction identity hard gate — the
        # provider's cloud sha must BE the landscape's measured sha.
        if self.provider.cloud_sha != self.landscape.cloud_sha256:
            raise RuntimeError(
                f"the Engine's provider cloud identity disagrees with the "
                f"landscape's measured cloud sha ({self.provider.cloud_sha} "
                f"!= {self.landscape.cloud_sha256}) — fail-hard")

    # -- cache ---------------------------------------------------------------

    def _load_cache(self):
        if self.cache_path.is_file():
            return json.loads(self.cache_path.read_text())
        return {}

    def _flush_cache(self):
        """Caller holds self._lock.  Atomic tmp + rename."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cache, indent=2, sort_keys=True))
        tmp.replace(self.cache_path)

    def cache_get(self, key):
        with self._lock:
            return self.cache.get(key)

    def _cache_save(self, key, payload, step):
        with self._lock:
            self.cache[key] = {**payload, "fitness_id":
                               self.provider.FITNESS_ID, "source": step,
                               "utc": utcnow()}
            self._flush_cache()

    # -- records -------------------------------------------------------------

    def step_dir(self, step):
        d = self.runs_root / step
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_path(self, step, row, kind="cfg"):
        return self.step_dir(step) / f"{kind}_row{int(row):07d}.json"

    @staticmethod
    def write_record(path, payload):
        L.write_json_with_sidecar(path, payload)
        return path

    # -- provenance ----------------------------------------------------------

    def provenance_block(self):
        prov = self.provider.PROVIDER_META
        return {
            "schema": SCHEMA,
            "landscape": self.landscape.name,
            "landscape_artifacts_sha256": self.landscape.sha256_record(),
            "landscape_summary_sha256": geo_sha256(
                self.landscape.root / "enumeration_summary.json"),
            "geometries": {
                "ts": str(self.provider.ts_path.relative_to(REPO_ROOT)),
                "ts_sha256": self.provider.ts_sha,
                "reactant": str(self.provider.r_path.relative_to(REPO_ROOT)),
                "reactant_sha256": self.provider.r_sha,
                "p": str(self.provider.p_path.relative_to(REPO_ROOT)),
                "p_sha256": self.provider.p_sha,
            },
            "provider": {
                "fitness_id": self.provider.FITNESS_ID,
                "lot": self.provider.lot,
                "pal": self.pal,
                "maxcore_mb": self.maxcore,
                "pc_channel": prov["pc_channel"],
            },
            "constant_kcal_per_eh": KCAL_PER_EH,
            "engine": {"workers": self.workers, "list_k": self.list_k},
        }

    # -- the worker pool -----------------------------------------------------
    #
    # The three-tier dispatch, shared by the differential (step A/B) and
    # P-point (step E) modes:
    #   1. step resume   -- the record file already exists -> load it
    #                       (a completed config is never re-executed);
    #   2. cache hit     -- the cache entry exists -> the record is built
    #                       from the cache (no execution);
    #   3. execution     -- the rest run in the thread pool; the
    #                       ORCHESTRATOR thread writes each completed
    #                       record + sidecar + cache.json under the lock.
    # The first exception cancels the remaining futures and halts the
    # batch (fail-hard; in-flight ORCA processes are left to finish and
    # their results are discarded).

    def _dispatch_diff(self, jobs, step):
        """(Heartbeat bookkeeping: the step's totals are set here; the
        per-config ticks land in _diff_write.)  jobs:
        [{row, idx_i, idx_j, sign_i, sign_j}] ->
        ({row: record-payload-dict}, {row: mode}) where mode records
        THIS run's action: 'executed' | 'cache' | 'resumed' (the
        per-step summary reports these; a record's own executed/from_
        cache flags are its historical provenance)."""
        self._hb_step_done = sum(1 for job in jobs
                                 if self.record_path(step, job["row"])
                                 .is_file())
        self._hb_total = len(jobs)
        payloads = {}
        modes = {}
        exec_jobs = []
        for job in jobs:
            row = job["row"]
            path = self.record_path(step, row)
            if path.is_file():                      # 1. step resume
                payloads[row] = json.loads(path.read_text())
                modes[row] = "resumed"
                continue

            key = orca_prov.differential_key(
                job["idx_i"], job["idx_j"], job["sign_i"], job["sign_j"],
                self.provider.cloud_sha, self.provider.ts_sha,
                self.provider.r_sha, self.provider.lot)
            hit = self.cache_get(key)
            if hit is not None:                     # 2. cache hit
                payloads[row] = self._diff_cache_record(step, job, key, hit)
                modes[row] = "cache"
                continue
            exec_jobs.append((job, key))            # 3. execution

        if exec_jobs:
            done = {}
            error = None
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(self.provider.run_config,
                                    job["idx_i"], job["idx_j"],
                                    job["sign_i"], job["sign_j"]): (job, key)
                        for job, key in exec_jobs}
                for fut in as_completed(futs):
                    job, key = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:        # noqa: BLE001 fail-hard
                        error = exc
                        for f2 in futs:
                            f2.cancel()
                        break
                    done[key] = res
                    self._diff_write(step, job, key, res)
            if error is not None:
                raise error
            for job, key in exec_jobs:
                res = done.get(key)
                if res is not None:
                    payloads[job["row"]] = self._diff_record_from_res(
                        step, job, key, res, executed=True)
                    modes[job["row"]] = "executed"
        return payloads, modes

    def _diff_write(self, step, job, key, res):
        """Orchestrator-side completion: record + sidecar + cache.
        Ticks the heartbeat with the config's last measured wall."""
        record = self._diff_record_from_res(step, job, key, res,
                                            executed=True)
        self.write_record(self.record_path(step, job["row"]), record)
        walls = [w for w in res["walls"].values() if w is not None]
        self._hb_step_done += 1
        self.heartbeat.tick(step, self._hb_step_done, self._hb_total,
                            last_wall=(round(walls[-1], 3)
                                       if walls else None))
        self._cache_save(key, {"e_ts": res["e_ts"], "e_r": res["e_r"],
                               "walls": res["walls"], "lot": res["lot"],
                               "pal": res["pal"],
                               "maxcore": res["maxcore"]}, step)

    def _diff_record_from_res(self, step, job, key, res, executed):
        return self._make_record(step, job, key, "cfg", e_ts=res["e_ts"],
                                 e_r=res["e_r"], f=res["f"],
                                 walls=res["walls"], executed=executed,
                                 lot=res["lot"], pal=res["pal"],
                                 maxcore=res["maxcore"])

    def _diff_cache_record(self, step, job, key, hit):
        return self._make_record(step, job, key, "cfg", e_ts=hit["e_ts"],
                                 e_r=hit["e_r"],
                                 f=(hit["e_ts"] - hit["e_r"]) * KCAL_PER_EH,
                                 walls=hit.get("walls"), executed=False,
                                 lot=hit["lot"], pal=hit["pal"],
                                 maxcore=hit["maxcore"])

    def _make_record(self, step, job, key, kind, e_ts, e_r, f, walls,
                     executed, lot, pal, maxcore):
        row = int(job["row"])
        order = self.landscape.f_sorted_rows()
        pos = np.flatnonzero(order == row)
        rank = int(pos[0]) if pos.size else None
        return {
            "schema": SCHEMA, "step": step, "kind": kind,
            "landscape": self.landscape.name, "row": row,
            "config": {"idx_i": int(job["idx_i"]),
                       "idx_j": int(job["idx_j"]),
                       "sign_i": float(job["sign_i"]),
                       "sign_j": float(job["sign_j"])},
            "key": key, "fitness_id": self.provider.FITNESS_ID,
            "e_ts_eh": float(e_ts), "e_r_eh": float(e_r),
            "f_kcal": float(f),
            "f_xtb_kcal": float(self.landscape.f[row]),
            "oracle_rank": rank,
            "walls": walls,
            "lot": lot, "pal": int(pal), "maxcore_mb": int(maxcore),
            "executed": bool(executed), "from_cache": not executed,
            "utc": utcnow(),
        }

    # -- dispatch for the P point mode (step E) ------------------------------

    def _dispatch_p(self, rows):
        """rows (sorted certified rows) -> {row: p-record-payload}."""
        self._hb_step_done = sum(1 for row in rows
                                 if self.record_path("E", row, kind="pcfg")
                                 .is_file())
        self._hb_total = len(rows)
        payloads = {}
        exec_rows = []
        dft_by_row = {}
        for row in rows:
            path = self.record_path("E", row, kind="pcfg")
            if path.is_file():                      # 1. step resume
                payloads[row] = json.loads(path.read_text())
                continue
            dft_by_row[row] = self.certified_record(row)
            key = orca_prov.p_key(dft_by_row[row]["config"]["idx_i"],
                                  dft_by_row[row]["config"]["idx_j"],
                                  dft_by_row[row]["config"]["sign_i"],
                                  dft_by_row[row]["config"]["sign_j"],
                                  self.provider.cloud_sha,
                                  self.provider.p_sha, self.provider.lot)
            hit = self.cache_get(key)
            if hit is not None:                     # 2. cache hit
                payloads[row] = self._p_record(row, dft_by_row[row], key,
                                               {"e_p": hit["e_p"],
                                                "wall_s": (hit.get(
                                                                   "walls")
                                                           or {}).get("p_s"),
                                                "lot": hit["lot"],
                                                "pal": hit["pal"],
                                                "maxcore": hit["maxcore"]},
                                               executed=False)
                continue
            exec_rows.append((row, dft_by_row[row], key))
        if exec_rows:
            done = {}
            error = None
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(self.provider.run_p_config,
                                    rec["config"]["idx_i"],
                                    rec["config"]["idx_j"],
                                    rec["config"]["sign_i"],
                                    rec["config"]["sign_j"]):
                        (row, rec, key) for row, rec, key in exec_rows}
                for fut in as_completed(futs):
                    row, rec, key = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:        # noqa: BLE001 fail-hard
                        error = exc
                        for f2 in futs:
                            f2.cancel()
                        break
                    done[key] = res
                    self.write_record(
                        self.record_path("E", row, kind="pcfg"),
                        self._p_record(row, rec, key, res, executed=True))
                    self._hb_step_done += 1
                    self.heartbeat.tick("E", self._hb_step_done,
                                        self._hb_total,
                                        last_wall=(round(res["wall_s"], 3)
                                                   if res["wall_s"] is not None
                                                   else None))
                    self._cache_save(key, {
                        "e_p": res["e_p"],
                        "walls": {"p_s": res["wall_s"]},
                        "lot": res["lot"], "pal": res["pal"],
                        "maxcore": res["maxcore"]}, "E")
            if error is not None:
                raise error
            for row, rec, key in exec_rows:
                res = done.get(key)
                if res is not None:
                    payloads[row] = self._p_record(row, rec, key, res,
                                                   executed=True)
        return payloads

    def _p_record(self, row, dft, key, res, executed):
        """The step-E record: E(P;q), the charged dE_rxn, the quench flag."""
        e_p = float(res["e_p"])
        return {
            "schema": SCHEMA, "step": "E", "kind": "p",
            "landscape": self.landscape.name, "row": int(row),
            "config": {"idx_i": int(dft["config"]["idx_i"]),
                       "idx_j": int(dft["config"]["idx_j"]),
                       "sign_i": float(dft["config"]["sign_i"]),
                       "sign_j": float(dft["config"]["sign_j"])},
            "key": key, "e_p_eh": e_p,
            "e_r_eh_dft_record": float(dft["e_r_eh"]),
            "e_ts_eh_dft_record": float(dft["e_ts_eh"]),
            "dE_rxn_kcal": float((e_p - dft["e_r_eh"]) * KCAL_PER_EH),
            "quench": bool(dft["e_ts_eh"] < e_p),
            "wall_s": res.get("wall_s"),
            "lot": res["lot"], "pal": int(res["pal"]),
            "maxcore_mb": int(res["maxcore"]),
            "executed": bool(executed), "utc": utcnow(),
        }

    # -- landscape decode + summaries ----------------------------------------

    def rows_jobs(self, rows):
        """Rows -> job dicts (decode inside, from the gated landscape)."""
        jobs = []
        for row in rows:
            i, j, si, sj = self.landscape.decode_row(row)
            jobs.append({"row": int(row), "idx_i": i, "idx_j": j,
                         "sign_i": si, "sign_j": sj})
        return jobs

    def read_records(self, step, kind="cfg"):
        """All per-config records of a step (row -> record)."""
        d = self.step_dir(step)
        out = {}
        for p in sorted(d.glob(f"{kind}_row*.json")):
            rec = json.loads(p.read_text())
            out[rec["row"]] = rec
        return out

    @staticmethod
    def read_records_static(step_dir, kind="cfg"):
        out = {}
        for p in sorted(Path(step_dir).glob(f"{kind}_row*.json")):
            rec = json.loads(p.read_text())
            out[rec["row"]] = rec
        return out

    def write_summary(self, step, payload):
        payload = {"schema": SCHEMA, "step": step,
                   "landscape": self.landscape.name, **payload}
        path = self.step_dir(step) / "summary.json"
        L.write_json_with_sidecar(path, payload)
        return path

    def summary_exists(self, step):
        return (self.step_dir(step) / "summary.json").is_file()

    def read_summary(self, step):
        return json.loads((self.step_dir(step) / "summary.json").read_text())

    def certified_record(self, row):
        """The certified-set DFT record for a row (A then B)."""
        for step in ("A", "B"):
            recs = self.read_records(step)
            if int(row) in recs:
                return recs[int(row)]
        raise RuntimeError(
            f"row {row}: no certified-set record (run step A first)")

    def certified_set(self):
        """The certified rows: the first K_final of the f-sorted order
        (A's top-K + any widening-extended rows; the sweep's NON-widened
        boundary rows are not certified).  Every certified row must
        carry a record (A or B) -- a widening that stopped mid-run
        fails the gate (finish step B first)."""
        k = self._k_final()
        order = [int(r) for r in self.landscape.f_sorted_rows()[:k]]
        have = {row for step in ("A", "B")
                for row in self.read_records(step)}
        missing = [r for r in order if r not in have]
        if missing:
            raise RuntimeError(
                f"the certified set is incomplete: {len(missing)} of its "
                f"{k} rows carry no record (rows {missing[:5]}...); the "
                "widening did not finish -- run step B to completion")
        return order

    def _k_final(self):
        """A's list_k + B's widening (B's summary is the authority; B
        not run -> A's list_k)."""
        if self.summary_exists("B"):
            return int(self.read_summary("B").get("k_final",
                                                  self.list_k))
        return self.list_k


# --------------------------------------------------------------------------
# The steps
# --------------------------------------------------------------------------

def _wall_stats(records):
    walls = [w for r in records for w in
             ((r.get("walls") or {}).values()
              if r.get("walls") else [r.get("wall_s")]) if w is not None]
    if not walls:
        return {"measured_sp": 0, "mean_s": None}
    return {"measured_sp": len(walls),
            "mean_s": sum(walls) / len(walls)}


def _require_A(engine, step):
    if not engine.read_records("A"):
        raise RuntimeError(f"step {step}: requires step A's records "
                           "(the certified set starts at A's top-K)")


def step_A(engine):
    """Top-K re-rank: the K best oracle rows, 2 SPs each."""
    rows = [int(r) for r in engine.landscape.f_sorted_rows()[:engine.list_k]]
    payloads, modes = engine._dispatch_diff(engine.rows_jobs(rows), "A")
    records = [payloads[r] for r in rows]
    f_dft = [rec["f_kcal"] for rec in records]
    summary = {
        "purpose": "step A: the r2SCAN-3c re-rank of the top-K oracle rows",
        "list_k": engine.list_k,
        "executed": sum(1 for m in modes.values() if m == "executed"),
        "from_cache": sum(1 for m in modes.values() if m == "cache"),
        "resumed": sum(1 for m in modes.values() if m == "resumed"),
        "rows": rows,
        "f_dft_min_kcal": float(min(f_dft)),
        "f_dft_argmin_row": int(rows[int(np.argmin(f_dft))]),
        "walls": _wall_stats(records),
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("A", summary)


def step_B(engine, max_widening_rounds=4):
    """Boundary sweep: oracle ranks K+1..K+30; on a beat, +30 and repeat.

    The widening rule (the replicate-arbitration briefing, 2026-10-30):
    at most `max_widening_rounds` widening events (K <= list_k + 4*30);
    a FIFTH event is the pre-committed STOP (rev-3a section 5: the
    widening did not stabilize) — raised, with every evaluated round's
    records and the widening trail still on disk.

    step A's records must exist: the boundary is K+1.. of the SAME
    f-sorted list, and the opening DFT-best of the certified set comes
    from A's records (never re-executed).
    """
    _require_A(engine, "B")
    a_summary = engine.read_summary("A")
    if int(a_summary["list_k"]) != engine.list_k:
        raise RuntimeError(
            f"step B: --list-k {engine.list_k} != step A's recorded "
            f"list_k {a_summary['list_k']} (the boundary is K+1.. of the "
            "SAME f-sorted list)")
    order = engine.landscape.f_sorted_rows()
    k = engine.list_k
    certified_best = min(r["f_kcal"] for r in engine.read_records("A")
                         .values())
    rounds = []
    events = []
    while True:
        lo, hi = k, k + WIDENING_BATCH
        rows = [int(r) for r in order[lo:hi]]
        if not rows:
            break
        payloads, modes = engine._dispatch_diff(engine.rows_jobs(rows), "B")
        records = [payloads[r] for r in rows]
        new_best = min(r["f_kcal"] for r in records)
        new_best_rec = min(records, key=lambda r: r["f_kcal"])
        beat = new_best < certified_best
        rounds.append({"range": [lo, hi - 1], "rows": rows,
                       "new_best_kcal": float(new_best),
                       "new_best_row": int(new_best_rec["row"]),
                       "certified_best_before_kcal":
                           float(certified_best),
                       "beat": bool(beat)})
        if beat:
            events.append({"event": len(events) + 1,
                           "extended_to_k": hi,
                           "new_best_row": int(new_best_rec["row"])})
            if len(events) > max_widening_rounds:
                raise RuntimeError(
                    f"step B STOP: the widening rule did not stabilize "
                    f"within {max_widening_rounds} widening rounds "
                    f"(K would extend to {hi}); every evaluated round's "
                    f"records + the widening trail are on disk (rev-3a "
                    f"section 5 / the replicate-Go briefing's <=4 rounds)")
            k += WIDENING_BATCH
            certified_best = min(certified_best, new_best)
        else:
            break
    summary = {
        "purpose": "step B: the pre-committed boundary sweep",
        "rule": ("evaluate oracle ranks K+1..K+30; if any boundary "
                 "config's DFT f beats the current DFT-best of the "
                 f"certified set, extend K by +{WIDENING_BATCH} and "
                 "repeat; already-evaluated configs never re-execute"),
        "rounds": rounds, "widening_events": events, "k_final": k,
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("B", summary)


def step_C(engine):
    """Fidelity probe (sampler + statistics live in arbitration.probe)."""
    from arbitration import probe
    return probe.run_step_C(engine)


def _best_of(recs):
    if not recs:
        return None
    row = min(recs, key=lambda r: recs[r]["f_kcal"])
    return {"row": int(row), "f_kcal": float(recs[row]["f_kcal"])}


def top100_rank_correlation(engine):
    """Ruling 2 + rev-2 §5D: the step-A-derived rank correlation —
    Spearman's rho between the xTB oracle rank (position in the
    f-sorted landscape order) and the DFT rank (position within the
    DFT-sorted re-ranked top-100), over the top-100's A records.
    Returns None when step A hasn't run."""
    recs = engine.read_records("A")
    if not recs:
        return None
    rows = sorted(recs)
    oracle_rank = np.array([recs[r]["oracle_rank"] for r in rows],
                           dtype=np.float64)
    f_dft = np.array([recs[r]["f_kcal"] for r in rows], dtype=np.float64)
    # the DFT rank: position within the DFT-sorted top-100 (dense rank
    # on ties — the searchers' stable order is the row order)
    order = np.lexsort((np.array(rows), f_dft))
    dft_rank = np.empty(len(rows), dtype=np.float64)
    dft_rank[order] = np.arange(len(rows), dtype=np.float64)
    return {"n": len(rows),
            "spearman": float(sps.spearmanr(oracle_rank, dft_rank)[0]),
            "note": ("the xTB oracle rank vs the DFT rank within the "
                     "re-ranked top-100 (0-based positions; ties broken "
                     "by row order — dense at the DFT side)")}


def step_D(engine):
    """Certified optimum: DFT argmin over the certified set (the first
    K_final of the f-sorted order; the non-widened boundary rows are
    not certified)."""
    _require_A(engine, "D")
    rows = engine.certified_set()
    certified = {r: engine.certified_record(r) for r in rows}
    best_row = min(certified, key=lambda r: certified[r]["f_kcal"])
    rec = certified[best_row]
    probe_stats = None
    if engine.summary_exists("C"):
        probe_stats = engine.read_summary("C").get("statistics")
    order = engine.landscape.f_sorted_rows()
    pos = np.flatnonzero(order == int(best_row))
    rank = int(pos[0]) if pos.size else None
    opt = {
        "schema": SCHEMA,
        "purpose": ("the certified optimum of the certified landscape at "
                    "r2SCAN-3c (item 7 step D; the deadline-critical "
                    "record)"),
        "landscape": engine.landscape.name,
        "row": int(best_row),
        "idx_i": rec["config"]["idx_i"], "idx_j": rec["config"]["idx_j"],
        "sign_i": rec["config"]["sign_i"], "sign_j": rec["config"]["sign_j"],
        "config_key": rec["key"],
        "fitness_id": rec["fitness_id"],
        "e_ts_eh": rec["e_ts_eh"], "e_r_eh": rec["e_r_eh"],
        "f_dft_kcal": rec["f_kcal"],
        "oracle_rank": rank,
        "f_xtb_kcal": rec["f_xtb_kcal"],
        "certified_rows": int(len(certified)),
        "k_final": engine._k_final(),
        "probe_statistics": probe_stats,
        "provenance": engine.provenance_block(),
        "utc": utcnow(),
    }
    opt_path = engine.runs_root / "certified_optimum.json"
    L.write_json_with_sidecar(opt_path, opt)
    summary = {
        "purpose": "step D: the certified optimum + the aggregate trail",
        "certified_rows": int(len(certified)),
        "dft_best": {"row": int(best_row), "f_kcal": rec["f_kcal"],
                     "i": rec["config"]["idx_i"],
                     "j": rec["config"]["idx_j"]},
        "dft_best_per_step": {
            step: _best_of(recs)
            for step, recs in (("A", engine.read_records("A")),
                               ("B", engine.read_records("B")))},
        "boundary_trail": (engine.read_summary("B").get("widening_events", [])
                           if engine.summary_exists("B") else []),
        "probe_statistics": probe_stats,
        "top100_rank_correlation": top100_rank_correlation(engine),
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("D", summary)


def _k_final(engine):
    """A's list_k + B's widening (B not run -> A's list_k)."""
    if engine.summary_exists("B"):
        return int(engine.read_summary("B").get(
            "k_final", engine.list_k))
    return engine.list_k


def step_E(engine):
    """P-side check: every certified config gets one SP on the P geometry
    (P point mode)."""
    _require_A(engine, "E")
    rows = engine.certified_set()
    payloads = engine._dispatch_p(rows)
    records = [payloads[r] for r in rows]
    summary = {
        "purpose": ("step E: the P-side check over the certified set "
                    "(E(P;q), the charged dE_rxn, the quench flag)"),
        "rows": rows,
        "n_quench_true": sum(1 for r in records if r["quench"]),
        "walls": _wall_stats(records),
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("E", summary)


def step_F(engine):
    """Relaxation check of the certified optimum (the landed qm/
    machinery with the winning charges; PAL 8 / %maxcore 1500).  An
    idle heartbeat thread writes the elapsed markers while the two
    optimisation jobs run (the unattended anchor)."""
    import time as _time
    from qm import orca_out
    from qm import orca_runner
    from qm import ts_chain

    opt = read_certified_optimum(engine)
    row = opt["row"]
    i, j = opt["idx_i"], opt["idx_j"]
    si, sj = opt["sign_i"], opt["sign_j"]
    hb_stop = engine.heartbeat.start_thread(
        "F", note="the relaxation jobs (unattended; the overnight anchor)")
    engine.heartbeat.tick("F", 0, 0, note="the OptTS job starting")
    d = engine.step_dir("F")
    fdir_ts = d / "optts"
    fdir_r = d / "opt_r"
    for p in (fdir_ts, fdir_r):
        p.mkdir(parents=True, exist_ok=True)
    pc_text = orca_prov.render_pc(float(si), float(sj),
                                  engine.landscape.cloud[i],
                                  engine.landscape.cloud[j])
    pc_ts = fdir_ts / "charges.pc"
    pc_r = fdir_r / "charges.pc"
    pc_ts.write_text(pc_text)
    pc_r.write_text(pc_text)

    ts_elements, ts_coords = orca_prov.load_xyz(engine.provider.ts_path)
    r_elements, r_coords = orca_prov.load_xyz(engine.provider.r_path)
    inp_ts = fdir_ts / "optts_embedded.inp"
    inp_r = fdir_r / "opt_r_embedded.inp"
    inp_ts.write_text(ts_chain.render_optts_inp(
        ts_elements, ts_coords, pointcharges=str(pc_ts)))
    inp_r.write_text(ts_chain.render_opt_inp(
        r_elements, r_coords, pointcharges=str(pc_r)))
    out_ts = fdir_ts / "optts_embedded.out"
    out_r = fdir_r / "opt_r_embedded.out"

    t0 = _time.monotonic()
    res_ts = orca_runner.run_orca(str(inp_ts), str(out_ts))
    opt_ts_s = _time.monotonic() - t0
    engine.heartbeat.tick(
        "F", 0, 0,
        note="the OptTS job done (%.1f min); the plain Opt of R starting"
             % (opt_ts_s / 60))
    t1 = _time.monotonic()
    res_r = orca_runner.run_orca(str(inp_r), str(out_r))
    opt_r_s = _time.monotonic() - t1
    hb_stop.set()
    engine.heartbeat.tick(
        "F", 0, 0,
        note="relaxation done: OptTS %.1f min + plain-Opt %.1f min"
             % (opt_ts_s / 60, opt_r_s / 60))

    parsed_ts = orca_out.parse_output(out_ts, need_frequencies=True)
    n_imag = int(parsed_ts["n_imaginary"])
    imag = parsed_ts["imaginary_modes"]
    gate_ok = n_imag == 1
    e_ts_relaxed = orca_out.final_energy_eh(out_ts)
    e_r_relaxed = orca_out.final_energy_eh(out_r)
    relaxed_diff = (e_ts_relaxed - e_r_relaxed) * KCAL_PER_EH
    vertical = float(read_certified_optimum(engine)["f_dft_kcal"])
    payload_common = {"schema": SCHEMA, "step": "F",
                      "landscape": engine.landscape.name,
                      "row": int(row), "config_key": opt["config_key"]}
    rec_ts = {**payload_common, "job": "optts_embedded",
              "relaxed": res_ts, "e_ts_relaxed_eh": float(e_ts_relaxed),
              "n_imaginary": n_imag,
              "imaginary_modes_cm1": imag,
              "exactly_one_imaginary_gate": gate_ok,
              "utc": utcnow()}
    rec_r = {**payload_common, "job": "opt_r_embedded", "relaxed": res_r,
             "e_r_relaxed_eh": float(e_r_relaxed), "utc": utcnow()}
    engine.write_record(d / "optts_result.json", rec_ts)
    engine.write_record(d / "opt_r_result.json", rec_r)
    record = {
        "purpose": ("step F: the frozen-geometry error BOUND (the relaxed "
                    "vs the vertical differential; the check never "
                    "asserts the error away)"),
        "vertical_f_kcal": vertical,
        "relaxed_differential_kcal": float(relaxed_diff),
        "relaxed_minus_vertical_kcal": float(relaxed_diff - vertical),
        "exactly_one_imaginary_gate": gate_ok,
        "gate_verdict": ("PASS" if gate_ok else
                         "FAIL -- the relaxation did not preserve exactly "
                         "one imaginary frequency"),
    }
    summary = {"purpose": "step F: the relaxation check summary", **record,
               "imaginary_mode_cm1": (float(imag[0][1]) if imag else None),
               "provenance": engine.provenance_block(),
               "utc": utcnow()}
    engine.write_summary("F", summary)
    if not gate_ok:
        raise RuntimeError(
            f"step F GATE FAILED: {n_imag} imaginary frequencies "
            f"(expected exactly 1); imaginary modes: {imag}; the records "
            f"and summary sit at {d}")
    return summary, d / "summary.json"


def step_G(engine):
    """wB97M check: the certified optimum only, 2 frozen SPs (the
    section 6.3 fallback level wB97M-V def2-TZVPP -- plain wB97M is
    UNRECOGNIZED in ORCA 6.1.1; reported as a deviation)."""
    opt = read_certified_optimum(engine)
    # §1.3 rev 2 (2026-10-31): the landscape's cloud identity rides along
    # (canonical-only today; the hole cannot reopen on other landscapes).
    provider = orca_prov.OrcaProvider(
        lot=orca_prov.WB97M_LOT, fitness_id=orca_prov.WB97M_FITNESS_ID,
        pal=engine.pal, maxcore=engine.maxcore,
        cloud_path=engine.landscape.cloud_path,
        cloud_sha256=engine.landscape.cloud_sha256,
        scratch_root=engine.provider.scratch_root)   # the run's scratch
    res = provider.run_config(opt["idx_i"], opt["idx_j"], opt["sign_i"],
                              opt["sign_j"])
    record = {
        "schema": SCHEMA, "step": "G", "kind": "cfg",
        "landscape": engine.landscape.name, "row": int(opt["row"]),
        "config": res["config"], "key": res["key"],
        "fitness_id": provider.FITNESS_ID, "lot": provider.lot,
        "e_ts_eh": res["e_ts"], "e_r_eh": res["e_r"],
        "f_kcal": res["f"], "walls": res["walls"],
        "pal": res["pal"], "maxcore_mb": res["maxcore"],
        "deviation_report": (
            "spec section 6.3 (question resolved): plain wB97M is "
            "UNRECOGNIZED in the installed ORCA 6.1.1 (measured "
            "2026-10-28, single-atom probe job: 'INPUT ERROR "
            "UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT "
            "LINE / WB97M'; the installed tree ships no manual PDF to "
            "cross-check), so the check ran wB97M-V per the section's "
            "prescribed fallback; wB97M-V intrinsically carries VV10 "
            "(no dispersion DIRECTIVES in the input, per the spec; the "
            "functional's own nonlocal term is not a directive). "
            "TightSCF added for SCF-convergence parity with the pinned "
            "arbitration level."),
        "provenance": engine.provenance_block(),
        "utc": utcnow(),
    }
    engine.write_record(engine.record_path("G", opt["row"]), record)
    summary = {
        "purpose": "step G: the wB97M-V cross-check of the certified optimum",
        "row": int(opt["row"]), "f_kcal": float(res["f"]),
        "e_ts_eh": res["e_ts"], "e_r_eh": res["e_r"],
        "deviation": record["deviation_report"],
        "walls": _wall_stats([record]),
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("G", summary)


def step_cert(engine):
    """Re-verify the certified optimum against the per-config records and
    re-emit certified_optimum.json + sidecar (idempotent)."""
    opt = read_certified_optimum(engine)
    certified = {**engine.read_records("A"), **engine.read_records("B")}
    row = int(opt["row"])
    if row not in certified:
        raise RuntimeError(
            f"step cert: the recorded optimum row {row} carries no "
            "certified-set record")
    rec = certified[row]
    checks = {
        "idx_i": rec["config"]["idx_i"] == opt["idx_i"],
        "idx_j": rec["config"]["idx_j"] == opt["idx_j"],
        "sign_i": rec["config"]["sign_i"] == opt["sign_i"],
        "sign_j": rec["config"]["sign_j"] == opt["sign_j"],
        "config_key": rec["key"] == opt["config_key"],
        "e_ts": rec["e_ts_eh"] == opt["e_ts_eh"],
        "e_r": rec["e_r_eh"] == opt["e_r_eh"],
        "f_dft": rec["f_kcal"] == opt["f_dft_kcal"],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(
            f"step cert: the stored certified_optimum.json disagrees with "
            f"the records on {bad}")
    best_row = min(certified, key=lambda r: certified[r]["f_kcal"])
    if int(best_row) != row:
        raise RuntimeError(
            f"step cert: the certified set's DFT argmin is now row "
            f"{best_row}, not the recorded {row} -- the record is stale "
            "(re-run step D)")
    order = engine.landscape.f_sorted_rows()
    pos = np.flatnonzero(order == row)
    opt["oracle_rank"] = int(pos[0]) if pos.size else opt["oracle_rank"]
    opt["cert_utc"] = utcnow()
    opt["cert_checks"] = checks
    opt_path = engine.runs_root / "certified_optimum.json"
    L.write_json_with_sidecar(opt_path, opt)
    summary = {
        "purpose": "step cert: re-verify + re-emit (idempotent)",
        "checks": checks, "row": row,
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("cert", summary)


def read_certified_optimum(engine):
    path = engine.runs_root / "certified_optimum.json"
    if not path.is_file():
        raise RuntimeError(f"{path}: not present (run step D first)")
    return json.loads(path.read_text())


STEPS = {"A": step_A, "B": step_B, "C": step_C, "D": step_D, "E": step_E,
         "F": step_F, "G": step_G, "cert": step_cert}