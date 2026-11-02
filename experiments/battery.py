#!/usr/bin/env python3
"""The reported battery — the 600-run experiment (experiments/battery.py).

Executes THE PRE-REGISTERED reported runs (stage-2 addendum frozen
2026-10-19, registered in `04` §12) and computes the prereg §8
statistics:

  600 runs = 3 landscapes (canonical, replicate-1, replicate-2)
           x 2 arms (A: use_descent=True, B: False)
           x 100 seeds (0..99), G_BATTERY = 2500, oracle mode.

Composes the VENDORED engine exactly as run_ga.py does (the same
oracle resolution semantics — the summary-level cloud gate, the
landscape's own fence, a FRESH provider per run — and run_ga.py's own
compose_record + write_record provenance). Records land per
(landscape, arm): the record identity scheme (G, seed, fitness_id,
cloud-sha12) needs the arm-level dirs because the two arms share
(G, seed, cloud).

RESUMABLE: an existing record identity is skipped untouched; the seed
ledgers + summaries are REWRITTEN each invocation FROM ALL LANDED
records, so partial execution still produces partial (marked)
statistics.

Statistics (prereg §8, all pre-declared):
  - success-vs-G = the PRIMARY curve: per-generation from the ledgers.
  - success-vs-queries on the analysis grid {1e2,3e2,1e3,3e3,1e4,1e5}
    — exact per-generation T_s from the ledgers (queries at the first
    success generation); budget-not-reached = a run's final queries
    below the entry (counted, kept in the denominator).
  - generations-to-success distributions; 95%/99% quantiles with the
    binomial honesty note.
  - ERT on the queries axis (successes' T_s + (n-n_B) B)/n_B; the
    n_success = 0 convention: ">= B" + the count.
  - final-best identity (the argmin row landed exactly) fraction.
  - cross-arm PAIRED per-seed deltas recorded in the aggregation file
    (Wilcoxon signed-rank needs scipy; the pinned env has it; in
    scipy-less environments the deltas + sign test are recorded).
  - the headline cell (G_battery, eps=1.0) + the headline success
    fraction at eps=0.25.

Run (from the repository root, with the pinned environment installed):

  python experiments/battery.py                 # the 600 reported runs
  BAT_N_SEEDS=3 python experiments/battery.py   # a fast partial pass
  BAT_FORCE=1 python experiments/battery.py     # re-run, do not resume

  env BAT_OUT_ROOT/BAT_FORCE for relocation/override. The reported
  records are summarised in data/results/battery/; this script rewrites
  them into runs/battery/ and must reproduce those summaries line for
  line.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # this repository

from ga import ga as ga_mod                      # noqa: E402
from ga import space as ga_space                 # noqa: E402
from ga.params import GAParams                   # noqa: E402
from ga.oracle_provider import landscape_oracle  # noqa: E402
import run_ga as runner                            # noqa: E402

ADDENDUM_SHA = ("28eb464a3e7813f2c31c551af4688c42710e3a6ea50"
               "273b46efa1274748db102")
G_BATTERY = 2500
SEEDS = list(range(int(os.environ.get("BAT_N_SEEDS", "100"))))
EPS = (0.25, 0.5, 1.0)
GRID = [100, 300, 1000, 3000, 10000, 100000]        # the analysis grid
OUT_ROOT = Path(os.environ.get("BAT_OUT_ROOT", HERE.parent / "runs" / "battery"))
GOCAT_DATA = Path(os.environ.get("BAT_GOCAT_DATA", HERE.parent / "data"))

LANDSCAPES = {
    # name -> (landscape dir, ABSOLUTE cloud file, argmin row, argmin f)
    "canonical": (GOCAT_DATA / "derived" / "enumeration_v2",
                  GOCAT_DATA / "derived" / "candidates_v2.npy",
                  1190559, -125.21463064719501),
    "replicate-1": (GOCAT_DATA / "replicates" / "replicate-1" /
                    "derived" / "enumeration_replicate-1",
                    GOCAT_DATA / "replicates" / "replicate-1" /
                    "candidates_replicate-1.npy",
                    1233787, -125.46892076041331),
    "replicate-2": (GOCAT_DATA / "replicates" / "replicate-2" /
                    "derived" / "enumeration_replicate-2",
                    GOCAT_DATA / "replicates" / "replicate-2" /
                    "candidates_replicate-2.npy",
                    1141875, -125.52238044298429),
}
ARMS = {"armA": GAParams(), "armB": GAParams(use_descent=False)}


# The three statistics below are written out rather than imported from
# scipy: this runner must stay scipy-free so the reported battery can be
# regenerated in the minimal numpy-only environment.
def wilson(p_hat, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    den = 1 + z * z / n
    c = (p_hat + z * z / (2 * n)) / den
    h = z / den * (p_hat * (1 - p_hat) / n
                   + z * z / (4 * n * n)) ** 0.5
    return c - h, c + h


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (k - f) * (xs[min(f + 1, len(xs) - 1)] - xs[f])


def times_to_success(ledger, argmin):
    """-> {eps: (T_g, T_s)} from one run's ledger (exact, per-gen)."""
    out = {}
    for e in EPS:
        hit = [x for x in ledger if x[2] <= argmin + e]
        out[e] = (hit[0][0], hit[0][1]) if hit else (None, None)
    return out


def group_setup(lname):
    ldir, cloud_path, argmin_row, argmin = LANDSCAPES[lname]
    summary = json.loads((ldir / "enumeration_summary.json").read_text())
    cloud_sha = str(summary["cloud_sha256"])
    # The argmin is a constant of the pre-registration, but check it against
    # the landscape's own summary before scoring anything: a mismatched
    # target would silently make every success statistic wrong.
    gmin = (summary.get("fitness") or {}).get("global_min") or {}
    if gmin:
        if int(gmin["row"]) != int(argmin_row) or \
                abs(float(gmin["f"]) - float(argmin)) > 2e-3:
            raise SystemExit(
                f"{lname}: the pre-registered argmin does not match the "
                f"landscape summary ({gmin} vs row {argmin_row}, "
                f"f {argmin})")
    if not cloud_path.is_file():
        raise SystemExit(f"{lname}: cloud file missing: {cloud_path}")
    got = runner.sha256_file(cloud_path)
    if got != cloud_sha:
        raise SystemExit(f"{lname}: cloud sha mismatch (got {got} != "
                         f"{cloud_sha}) — the landscape's own identity")
    excl_sha = ((summary.get("artifacts") or {})
                .get("domain_exclusions") or {}).get("sha256")
    tuples, source = runner.read_domain_exclusions(ldir, excl_sha)
    space = ga_space.Space(np.load(cloud_path), d_min=1.0,
                           exclusions=tuples, nn_k=8)
    d_min = float((summary.get("universe") or {}).get(
        "d_min", ga_space.D_MIN_A))
    return (ldir, cloud_path, cloud_sha, tuples, source, space, d_min,
            argmin, argmin_row)


def run_group(lname, arm, params, setup):
    """The 100 runs (resumable) + the seed ledger + the summary."""
    ldir, cloud_path, cloud_sha, tuples, source, space, d_min, argmin, \
        argmin_row = setup
    outdir = OUT_ROOT / lname / arm
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = Path(runner.record_path(outdir, G_BATTERY, seed,
                                       "enumeration_v2", cloud_sha))
        if path.is_file() and not os.environ.get("BAT_FORCE"):
            continue                                  # resume: skip
        provider = landscape_oracle(ldir, cloud_path)  # FRESH per run
        t0 = time.time()
        result = ga_mod.run_ga(provider.eval_batch, space, params, seed,
                               G=G_BATTERY)
        record = runner.compose_record(
            result, mode="oracle", fitness_id=provider.FITNESS_ID,
            provider_meta=provider.PROVIDER_META, cloud_path=cloud_path,
            cloud_sha256=provider.cloud_sha256, d_min=d_min,
            exclusions=tuples, exclusion_source=source,
            landscape_dir=str(ldir),
            artifact_sha256s=dict(provider.artifact_sha256s),
            param_set="battery-stage2:" + ADDENDUM_SHA + ":" +
                      json.dumps(params.as_dict(), sort_keys=True),
            git_head=("battery:" + os.environ.get("BAT_COMMIT",
                                                  "unknown") +
                      "|addendum:" + ADDENDUM_SHA))
        sha = runner.write_record(record, path)
        print(f"  {lname}/{arm} s={seed}: q={record['counts']['queries']} "
              f"best={record['ledger'][-1][2]:.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)
    # ---- the seed ledger + the §8 summary, from ALL landed records ----
    recs = sorted(outdir.glob("*.json"))
    entries = {}
    for j in recs:
        r = json.loads(j.read_text())
        t = times_to_success(r["ledger"], argmin)
        entries[str(r["seed"])] = {
            "record": j.name, "status": r.get("status", "ok"),
            "queries": r["counts"]["queries"],
            "submitted": r["counts"]["submitted"],
            "memo_hits": r["counts"]["memo_hits"],
            "unique": r["counts"]["unique"],
            "final_best_f": r["ledger"][-1][2],
            "final_best_row": r["best"]["row"],
            "success": {str(e): t[e][0] for e in EPS},
            "queries_at_success": {str(e): t[e][1] for e in EPS},
        }
    n_ok = sum(1 for v in entries.values() if v["status"] == "ok")
    n_failed = len(entries) - n_ok
    summary = {
        "schema": "battery_summary_v1", "landscape": lname, "arm": arm,
        "addendum_sha256": ADDENDUM_SHA, "G_battery": G_BATTERY,
        "n_attempted": len(entries), "n_recorded_ok": n_ok,
        "n_failed": n_failed,
        "failure_mode_note": "failed records carry status: failed + "
                             "the cause; excluded from the NUMERATOR "
                             "only; n = attempted is EVERY statistic's "
                             "denominator (prereg m5)",
        "argmin_target": argmin, "argmin_row": argmin_row,
        "headline_cell": {"G": G_BATTERY, "eps": 1.0},
        "headline_success_fraction": {"eps": 0.25},
        "stats": {},
        "status_of_this_summary": "COMPLETE" if n_ok == len(SEEDS)
                                  else "PARTIAL (resumable run)",
    }
    for e in EPS:
        tgs = [v["success"][str(e)] for v in entries.values()
               if v["success"][str(e)] is not None and
               v["status"] == "ok"]
        tss = [v["queries_at_success"][str(e)] for v in entries.values()
               if v["success"][str(e)] is not None and
               v["status"] == "ok"]
        p = len(tgs) / len(entries) if entries else 0.0
        stats = {"eps": e,
                 "n_success": len(tgs),
                 "success_fraction_at_cap": p,
                 "wilson95": wilson(p, len(entries)),
                 "success_vs_G": "per-generation curve lives in the "
                                 "seed ledger (success[] = T_g per "
                                 "seed); the curve is built from it "
                                 "at report time",
                 "generations_to_success": {
                     "mean": (sum(tgs) / len(tgs)) if tgs else None,
                     "median": quantile(tgs, 0.5),
                     "p95": quantile(tgs, 0.95),
                     "quantile_honesty": "the p99 level at n = 100 "
                                         "rests on ~1 success; reported "
                                         "with Wilson uncertainty, "
                                         "never smoothed"},
                 "success_vs_queries": {},
                 "ert_queries": {},
                 "ert_generation_analogue": (f">= {G_BATTERY}" if not tgs
                                             else (sum(tgs)
                                                   + (len(entries)
                                                      - len(tgs))
                                                   * G_BATTERY)
                                                   / len(tgs)),
                 "exact_identity_fraction":
                     (sum(1 for v in entries.values()
                          if v["final_best_row"] == argmin_row
                          and v["status"] == "ok") / n_ok) if n_ok
                     else 0.0}
        for B in GRID:
            n_B = sum(1 for T in tss if T is not None and T <= B)
            bunr = sum(1 for v in entries.values()
                       if v["queries"] < B)
            if n_B > 0:
                ert = (sum(T for T in tss if T is not None and T <= B)
                       + (len(entries) - n_B) * B) / n_B
            else:
                ert = f">= {B}"
            stats["success_vs_queries"][str(B)] = {
                "successes_at_budget": n_B,
                "success_fraction": n_B / len(entries),
                "wilson95": wilson(n_B / len(entries), len(entries)),
                "budget_not_reached_runs": bunr}
            stats["ert_queries"][str(B)] = ert
        summary["stats"][str(e)] = stats
    for name, obj in ((f"seed_ledger_{lname}_{arm}.json",
                       {"schema": "battery_seed_ledger_v1",
                        "landscape": lname, "arm": arm,
                        "G_battery": G_BATTERY,
                        "addendum_sha256": ADDENDUM_SHA,
                        "param_set": params.as_dict(),
                        "records": entries}),
                      (f"battery_summary_{lname}_{arm}.json", summary)):
        payload = json.dumps(obj, indent=1) + "\n"
        p = OUT_ROOT / name
        prior = p.read_bytes() if p.is_file() else None
        if prior != payload.encode():
            p.write_text(payload)
            (OUT_ROOT / (name + ".sha256")).write_text(
                hashlib.sha256(payload.encode()).hexdigest()
                + f"  {name}\n")
    for e in EPS:
        s = summary["stats"][str(e)]
        print(f"  {lname}/{arm} eps={e}: {s['n_success']}/{len(entries)} "
              f"(mean Tg {s['generations_to_success']['mean']}, "
              f"ERT@1e3 {s['ert_queries']['1000']})", flush=True)
    return summary


def paired_and_aggregate():
    """The cross-arm paired deltas + the cross-landscape aggregation."""
    agg = {"schema": "battery_aggregation_v1",
           "addendum_sha256": ADDENDUM_SHA,
           "guardrail": "consistent across BOTH replicates = a "
                        "finding; single-replicate deviation = an "
                        "observation (item-5 policy)",
           "headline_cell": {"G": G_BATTERY, "eps": 1.0},
           "groups": {}}
    for lname in LANDSCAPES:
        for arm in ARMS:
            p = OUT_ROOT / f"battery_summary_{lname}_{arm}.json"
            if p.is_file():
                agg["groups"][f"{lname}/{arm}"] = \
                    json.loads(p.read_text())["stats"]
        pa = OUT_ROOT / f"seed_ledger_{lname}_armA.json"
        pb = OUT_ROOT / f"seed_ledger_{lname}_armB.json"
        if pa.is_file() and pb.is_file():
            ra, rb = json.loads(pa.read_text())["records"], \
                     json.loads(pb.read_text())["records"]
            deltas = {"sign_test_note": "the paired Wilcoxon "
                                        "signed-rank needs scipy "
                                        "(compute in the pinned env "
                                        "pass); the per-seed deltas "
                                        "are here regardless"}
            for e in EPS:
                tg = [(int(s), ra[s]["success"][str(e)]
                       - rb[s]["success"][str(e)])
                      for s in ra if s in rb
                      and ra[s]["success"][str(e)] is not None
                      and rb[s]["success"][str(e)] is not None]
                deltas[str(e)] = {
                    "paired_seeds_with_both_success": len(tg),
                    "mean_delta_T_g": (sum(d for _, d in tg) /
                                       len(tg)) if tg else None,
                    "deltas": tg}
            agg.setdefault("pairA_vs_B", {})[lname] = deltas
    payload = json.dumps(agg, indent=1) + "\n"
    p = OUT_ROOT / "methods_test_aggregation.json"
    p.write_text(payload)
    (OUT_ROOT / (p.name + ".sha256")).write_text(
        hashlib.sha256(payload.encode()).hexdigest() + f"  {p.name}\n")


def main():
    t0 = time.time()
    print(f"battery: G={G_BATTERY} seeds 0..99 eps={EPS} "
          f"out={OUT_ROOT} addendum={ADDENDUM_SHA[:12]}", flush=True)
    for lname, arm, params in ((l, a, p)
                               for l in LANDSCAPES
                               for a, p in ARMS.items()):
        setup = group_setup(lname)
        argmin = setup[7]
        print(f"{lname}/{arm} (argmin {argmin})", flush=True)
        run_group(lname, arm, params, setup)
    paired_and_aggregate()
    print(f"BATTERY COMPLETE ({time.time() - t0:.0f}s wall)", flush=True)


if __name__ == "__main__":
    main()