#!/usr/bin/env python3
"""The enumeration driver (work item 3): the laptop fallback path +
the proof/smoke/verify/promotion entry points.

  python run_exhaustive_search.py --mode {stub,xtb} [--shards K]
      [--limit N] [--flush N] [--cloud PATH --cloud-sha256 HEX]
      [--work-root PATH]

    Runs the full post-d_min enumeration (or --limit N rows) as K
    concurrent single-thread OS-process workers (one contiguous shard
    each, sized by the canonical split; the provider's single-thread
    pinning is inherited by every worker). K concurrent workers x the
    measured per-configuration mean (0.109987 s, OMP=1, the
    benchmark's sizing constant) size every walltime; NEVER older
    ballparks. The canonical laptop projection (the xtb mode, K=8):
    ~52.6 h single-thread -> ~6.6 h.

    Resume: every shard is resumable (its manifest = the recorded
    rows; the resume skips them without re-evaluating, never double-
    counting). On ANY worker failure the driver STOPS FULLY (kills
    the remaining workers, writes the stage report, exits nonzero) --
    the caller resumes explicitly by re-invoking the same command.

  --proof [--save-reference] [--mode stub|xtb]
      The four real-provider guarantee proofs (spec 5) over the
      deterministic proof subset; --save-reference refreshes the
      shipped reference (exhaustive_search/proof_reference/) — a
      deliberate act, committed when the proofs go green.

  --smoke
      The CLUSTER smoke: run the four proofs (xtb mode) and diff
      against the shipped laptop reference bitwise (the deployment
      order's enumerate-smoke gate).

  --verify
      Aggregate + verify an existing interrupted/complete run's
      records into the working landscape (runs/...) — the resumable
      run's completion checker; incomplete runs FAIL here loudly.

  --promote
      THE CALLER-DIRECTED promotion: verify + write the canonical
      landscape artifact into data/derived/enumeration_v2/ (summary
      JSON + sha256s; the caller's explicit act, after the return).

Cloud identity (multi-cloud readiness): the cloud path + its sha256
are CLI parameters; the driver defaults to the canonical locked cloud
WITH its briefing-frozen sha256 (hard identity gates); any other
--cloud requires its own --cloud-sha256 explicitly — no implicit
acceptance. The work root is keyed by the cloud sha (12 hex) so
replicate landscapes process in isolation (the standing cache-hazard
rule); the canonical driver invocation never mixes clouds.

Nothing under data/derived/ is written by --proof/--smoke/--verify;
only --promote writes there (explicitly directed).
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exhaustive_search import enumerate as en

REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

CANONICAL_CLOUD = "data/derived/candidates_v2.npy"
CANONICAL_CLOUD_SHA256 = (
    "084e48a6b5267900a995ba2d4f0412f77f5bb13446ba74eafaf474c644349e71")

# The sizing constants (measured, laptop, OMP=1). The T2 pace is the
# canonical one since the 2026-10-08 protocol adoption; the benchmark's
# default-protocol value stays as history.
SIZING_S_PER_CONFIG = 0.14818   # the T2 protocol: both-sides timed sample
SIZING_S_PER_CONFIG_DEFAULT_PROTOCOL = 0.109987   # history (the benchmark)

DEPLOY_EXTS = (".py",)


def load_driver_module():
    path = REPO_ROOT / "run_exhaustive_search.py"
    spec = importlib.util.spec_from_file_location("run_exhaustive_search",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_cloud(args):
    """-> (cloud_path, expected_sha256, is_canonical). The multi-cloud
    rule: a non-default cloud without its own --cloud-sha256 refuses."""
    cloud = str(args.cloud)
    canonical = (cloud == CANONICAL_CLOUD)
    sha = args.cloud_sha256
    if sha is None:
        if not canonical:
            raise ValueError(
                f"the cloud {cloud} is not the canonical cloud and no "
                "--cloud-sha256 was given (no implicit acceptance of a "
                "wrong cloud: every provenance record takes its cloud's "
                "sha256 from the arguments)")
        sha = CANONICAL_CLOUD_SHA256
    return cloud, sha, canonical


def resolve_cloud_path():
    """The canonical cloud path (the manager's entry; the hpc stage
    runs only the canonical identity)."""
    return CANONICAL_CLOUD


def check_cloud(cloud, sha):
    got = en.sha256_file(cloud)
    if got != sha:
        raise ValueError(
            f"{cloud}: cloud sha256 mismatch (got {got}, expected {sha}) -- "
            f"tampered or wrong input")
    return got


def build_provider(mode, provider_scratch):
    """-> (provider, evaluator) for the mode; the evaluator = the
    provider's public eval_batch_full (the landscape-artifact view the
    frozen contract assigns), never any provider internals."""
    if mode == "stub":
        from fitness import stub as stub_module
        provider = stub_module.StubProvider()
    elif mode == "xtb":
        from fitness import xtb as ff
        provider = ff.XtbProvider(
            scratch_root=Path(provider_scratch))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return provider, provider.eval_batch_full


def spawn_workers(cloud, cloud_sha, mode, work_root, shards, flush,
                  limit_rows, universe_rows):
    """One subprocess per shard (all at once: K = --shards), the
    canonical split. Returns the list of (shard_idx, Popen)."""
    procs = []
    for k, (start, end) in enumerate(shards):
        end = min(end, limit_rows)
        if end <= start:
            continue
        args = [sys.executable, "-m", "exhaustive_search.worker",
                "--cloud", str(cloud), "--cloud-sha256", cloud_sha,
                "--mode", mode,
                "--start", str(start), "--end", str(end),
                "--work-dir", str(Path(work_root) / f"shard_{k}"),
                "--flush", str(flush)]
        if mode == "xtb":
            args += ["--provider-scratch",
                     str(Path(work_root) / "provider_scratch")]
        log = Path(work_root) / f"shard_{k}" / "worker.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        logh = open(log, "w")
        procs.append((k, subprocess.Popen(args, cwd=str(REPO_ROOT),
                                          stdout=logh, stderr=logh),
                      logh))
    return procs


def coverage_report(landscape, expect_rows):
    """The completeness arbiter's shared record (fix #6): a run whose
    rows are all EVALUATED-OR-DOMAIN-EXCLUDED is COMPLETE; the
    exclusions are first-class (count / by_side / rows / the list file
    name); the identity line is n_rows_evaluated + n_rows_excluded ==
    n_rows_landscape. Used by the stage run and by the promotion's
    re-aggregation (which is the completeness arbiter on the returned
    records -- it never trusts the stage report's own status)."""
    n_ev = int(landscape["n_rows_evaluated"])
    excl = landscape.get("domain_exclusions") or []
    by_side = {}
    for e in excl:
        by_side[e["stage"]] = by_side.get(e["stage"], 0) + 1
    identity = bool(n_ev + len(excl) == int(expect_rows))
    return {
        "n_rows_evaluated": n_ev,
        "n_rows_excluded": len(excl),
        "n_rows_landscape": int(expect_rows),
        "coverage_identity": identity,
        "domain_exclusions": {
            "count": len(excl),
            "by_side": by_side,
            "rows": [int(e["row"]) for e in excl],
            "list_file": ("domain_exclusions.json" if excl else None),
        },
        "status": "complete" if identity else "incomplete",
    }


def run_enumeration(cloud, cloud_sha, mode, work_root, shards_k, flush,
                    limit, d_min=en.D_MIN_A):
    """The sharded run + the final aggregate/verify. Returns
    (rc, report_dict)."""
    cloud_sha = check_cloud(cloud, cloud_sha)
    candidates = np.load(cloud)
    universe = en.build_universe(candidates, d_min=d_min)
    n_rows = int(universe["n_rows"])
    limit_rows = n_rows if limit is None else min(int(limit), n_rows)
    shards = en.split_shards(limit_rows, shards_k)
    work_root = Path(work_root)

    report = {
        "subcommand": "enumerate",
        "utc": datetime.now(timezone.utc).isoformat(),
        "cloud": str(cloud), "cloud_sha256": cloud_sha,
        "canonical_cloud": (cloud_sha == CANONICAL_CLOUD_SHA256),
        "mode": mode, "work_root": str(work_root),
        "d_min": d_min,
        "universe": {k: universe[k] for k in
                     ("n", "n_pairs_total", "n_configs_total",
                      "n_pairs_excluded", "n_configs_excluded", "n_rows")},
        "shards_k": shards_k, "flush_rows": flush,
        "limit_rows": limit_rows,
        "sizing_s_per_config": SIZING_S_PER_CONFIG,
        "shards": [{"k": k, "rows": [int(a), int(b)]}
                   for k, (a, b) in enumerate(shards)],
        "ok": False, "status": "incomplete",
    }
    if limit_rows == 0:
        raise ValueError("the requested limit leaves no rows")

    print(f"enumerate: cloud sha {cloud_sha[:12]}..., post-floor rows "
          f"{n_rows}, limit {limit_rows}, shards {shards_k} x "
          f"{flush}-row flush windows")
    proj = limit_rows * SIZING_S_PER_CONFIG
    print(f"  sizing: {limit_rows * SIZING_S_PER_CONFIG / 3600:.2f} h "
          f"serial-equivalent; {proj / shards_k / 3600:.2f} h at "
          f"{shards_k} concurrent workers (uniform rows)")

    t0 = time.monotonic()
    procs = spawn_workers(cloud, cloud_sha, mode, work_root, shards, flush,
                          limit_rows, n_rows)
    failed = None
    results = []
    while procs:
        time.sleep(0.2)
        for k, p, logh in procs:
            rc = p.poll()
            if rc is None:
                continue
            logh.close()
            results.append((k, rc))
            if rc != 0 and failed is None:
                failed = (k, rc)  # the full stop: kill the others
                break
        if failed is not None:
            for k, p, logh in procs:
                if p.poll() is None:
                    p.terminate()
            break
        procs = [(k, p, h) for (k, p, h) in procs if p.poll() is None]
    for k, p, h in procs:
        if p.poll() is None:
            p.wait()
        results.append((k, p.returncode))
    wall = time.monotonic() - t0
    print(f"  workers done in {wall / 60:.1f} min: "
          + ", ".join(f"shard {k} -> rc {rc}" for k, rc in sorted(results)))

    if failed is not None:
        k, rc = failed
        diag = Path(work_root) / f"shard_{k}" / "shard_meta.json"
        report["status"] = "incomplete"
        report["failed_shard"] = {"k": k, "rc": rc,
                                  "diagnostic": (str(diag) if diag.is_file()
                                                 else None)}
        if diag.is_file():
            report["failed_shard"]["error"] = json.loads(
                diag.read_text()).get("error")
        report["error"] = (f"shard {k} exited {rc} -- STAGE FULL STOP "
                           "(no retries; resume is the caller's explicit act)")
        _write_report(work_root / "enumerate_report.json", report)
        print(f"  {report['error']}", file=sys.stderr)
        return 1, report

    # ---- the aggregate + verify (never 'complete with gaps') -------------
    shard_dirs = [str(work_root / f"shard_{k}")
                  for k in range(shards_k)
                  if (work_root / f"shard_{k}").is_dir()]
    try:
        landscape = en.aggregate_shards(
            shard_dirs, universe,
            expect_rows=limit_rows if limit is not None else None)
    except RuntimeError as e:
        report["status"] = "incomplete"
        report["error"] = f"the aggregate gate failed: {e}"
        _write_report(work_root / "enumerate_report.json", report)
        print(f"  {report['error']}", file=sys.stderr)
        return 1, report

    report.update(coverage_report(landscape, limit_rows))
    report["n_evals_distinct"] = landscape["rows_evaluated_distinct"]
    report["n_evals_ledger"] = landscape["manifest_rows_total"]
    report["ledger"] = landscape["ledger"]
    land_dir = work_root / "landscape"
    report["working_landscape"] = str(land_dir)
    report["ok"] = report["status"] == "complete"

    # the verified working landscape (NOT the canonical artifact: that is
    # promotion's act, laptop-side, caller-directed)
    summary = en.landscape_summary(
        universe, landscape, cloud_sha,
        en.sha256_file(REPO_ROOT / "data/derived/structures/ts_opt.xyz"),
        en.sha256_file(REPO_ROOT /
                       "data/derived/structures/"
                       "chorismate_reactive_TSframe.xyz"),
        _provider_meta(mode), {"mode": mode, "wall_s_total": round(wall, 1),
                               "limit_rows": limit_rows})
    summary = en.write_landscape(land_dir, universe, landscape, summary)
    report["global_min"] = summary["fitness"]["global_min"]
    _write_report(work_root / "enumerate_report.json", report)
    print(f"  coverage: {report['n_rows_evaluated']} evaluated + "
          f"{report['n_rows_excluded']} domain-excluded = "
          f"{report['n_rows_landscape']} rows ({report['status']}); "
          f"global min {report['global_min']['f']:.6f} kcal/mol at pair "
          f"({report['global_min']['idx_i']}, "
          f"{report['global_min']['idx_j']}) "
          f"sign {report['global_min']['sign']}")
    return 0 if report["ok"] else 1, report


def _provider_meta(mode):
    """The provider's PROVIDER_META verbatim (the promotion summary's
    record)."""
    if mode == "stub":
        from fitness import stub as stub_module
        return stub_module.PROVIDER_META
    from fitness import xtb as ff
    return ff.PROVIDER_META


def _write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=1) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="enumerate driver")
    parser.add_argument("--mode", choices=("stub", "xtb"), default="xtb")
    parser.add_argument("--cloud", default=CANONICAL_CLOUD)
    parser.add_argument("--cloud-sha256", default=None)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--flush", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--proof", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--promote", action="store_true")
    # work-item-4b extension (the promote path only): the identity gate
    # parameterizes to expected = canonical by default; a replicate
    # landscape promotes with its own declared expected sha + its own
    # target dir (under the replicate root -- never data/derived/).
    parser.add_argument("--expected-cloud-sha256", default=None)
    parser.add_argument("--target-dir", default=None)
    args = parser.parse_args(argv)

    if args.smoke:
        import exhaustive_search.proofs as proofs
        ref = REPO_ROOT / "exhaustive_search" / "proof_reference"
        if not (ref / "laptop_proof.npz").is_file():
            parser.error("--smoke requires the shipped proof_reference "
                         "(exhaustive_search/proof_reference/)")
        cloud, sha, _ = resolve_cloud(args)
        work_root = Path(args.work_root
                         or REPO_ROOT / "runs/exhaustive_search/smoke")
        report = proofs.run_proofs(
            cloud_path=cloud, cloud_sha256=sha, mode=args.mode,
            work_root=work_root,
            provider_scratch=work_root / "provider_scratch",
            compare_reference=ref)
        path = work_root / "smoke_report.json"
        _write_report(path, report)
        print(json.dumps({k: report[k] for k in (
            "ok", "mode", "cloud_sha256")}, indent=1))
        print(f"smoke report -> {path}")
        return 0 if report["ok"] else 1

    if args.proof or args.save_reference:
        import exhaustive_search.proofs as proofs
        cloud, sha, _ = resolve_cloud(args)
        if args.save_reference and args.mode != "xtb":
            parser.error("--save-reference is an xtb-mode act (the shipped "
                         "reference is the real-provider proof)")
        work_root = Path(args.work_root
                         or REPO_ROOT / "runs/exhaustive_search/proofs")
        report = proofs.run_proofs(
            cloud_path=cloud, cloud_sha256=sha, mode=args.mode,
            work_root=work_root, provider_scratch=(
                work_root / "provider_scratch" if args.mode == "xtb"
                else None),
            save_reference=args.save_reference,
            reference_dir=REPO_ROOT / "exhaustive_search/proof_reference")
        path = work_root / "proof_report.json"
        _write_report(path, report)
        print(json.dumps({k: report[k] for k in (
            "ok", "mode", "subset_size", "cloud_sha256")}, indent=1))
        for name, p in report["proofs"].items():
            print(f"  {name}: " + json.dumps(
                {k: v for k, v in p.items() if not isinstance(v, list)}))
        print(f"proof report -> {path}")
        return 0 if report["ok"] else 1

    if args.promote:
        return promote(args)

    if args.verify:
        cloud, sha, _ = resolve_cloud(args)
        return verify(args, cloud, sha)

    cloud, sha, _ = resolve_cloud(args)
    work_root = args.work_root or str(
        REPO_ROOT / "runs/exhaustive_search" / sha[:12])
    rc, _report = run_enumeration(
        cloud, sha, args.mode, work_root, args.shards, args.flush,
        args.limit)
    return rc


def verify(args, cloud, sha):
    """Aggregate + verify an existing run's records (the completion
    checker; the resumable stage's own gate)."""
    work_root = Path(args.work_root or
                     REPO_ROOT / "runs/exhaustive_search" / sha[:12])
    if not (work_root / "enumerate_report.json").is_file():
        print(f"{work_root}: no stage report -- nothing to verify",
              file=sys.stderr)
        return 1
    report = json.loads((work_root / "enumerate_report.json").read_text())
    if not report.get("ok"):
        print(f"{work_root}: the stage report says "
              f"{report.get('status')!r} -- the run is not complete",
              file=sys.stderr)
        return 1
    land = Path(report["working_landscape"])
    if not (land / "enumeration_summary.json").is_file():
        print(f"{land}: no verified landscape -- nothing to promote",
              file=sys.stderr)
        return 1
    print(f"verified: {report['n_rows_evaluated']} rows; global min "
          f"{report['global_min']['f']:.6f} kcal/mol")
    return 0


def promote(args, work_root=None, target_dir=None):
    """THE CALLER-DIRECTED promotion (laptop-side, after the return):
    re-aggregate + re-verify the RETURNED shard records (every record
    re-parsed fresh here -- the returned records are the authority; the
    returned convenience landscape is not trusted), then write the
    landscape artifact (canonical: data/derived/enumeration_v2/;
    replicate: the caller-declared target under its replicate root).

    The identity gate (work-item-4b parameterization): mode == xtb AND
    the records' cloud sha == the EXPECTED sha, where expected = the
    canonical sha by default and a replicate's explicitly-passed
    --expected-cloud-sha256 otherwise. The expected sha is recorded in
    the promoted summary. All canonical-path behavior and tests are
    unchanged (the default expected = the canonical; the canonical
    default target = data/derived/enumeration_v2; a non-canonical
    expected sha REFUSES any data/derived/ target and REFUSES to
    promote without an explicitly declared target dir).

    The work root: --work-root, else the returned cluster stage root
    (data/hpc_out/enumerate/run) when it holds shard records, else the
    laptop run's root (runs/exhaustive_search/<cloud-sha12>/). Failure
    paths (no records / the verification gate failing / the target
    existing) are loud and refuse.
    """
    expected = (getattr(args, "expected_cloud_sha256", None)
                or CANONICAL_CLOUD_SHA256)
    cloud, sha, canonical = resolve_cloud(args)
    if sha != expected:
        print(f"promotion gate: the declared cloud's own sha {sha!r} != "
              f"the expected {expected!r} -- refuse", file=sys.stderr)
        return 1
    if work_root is not None and args.work_root is None:
        work_root = Path(work_root)
    elif args.work_root is not None:
        work_root = Path(args.work_root)
    else:
        returned = REPO_ROOT / "data/hpc_out/enumerate/run"
        laptop = REPO_ROOT / "runs/exhaustive_search" / sha[:12]
        if any(returned.glob("shard_*")):
            work_root = returned
        elif any(laptop.glob("shard_*")):
            work_root = laptop
        else:
            print(f"no shard records under {returned} or {laptop} -- "
                  "nothing to promote", file=sys.stderr)
            return 1
    shard_dirs = sorted(p for p in Path(work_root).glob("shard_*")
                        if p.is_dir())
    if not shard_dirs:
        print(f"{work_root}: no shard records -- nothing to promote",
              file=sys.stderr)
        return 1
    # the promotion identity gates (the stub hazard: a MECHANICAL stub
    # landscape must never promote -- caught live 2026-10-08, see the
    # log): every shard's record must certify mode == "xtb" against the
    # canonical cloud, and the stage report must be present + complete.
    metas = []
    for sd in shard_dirs:
        mp = sd / "shard_meta.json"
        if not mp.is_file():
            print(f"{sd}: no shard_meta.json -- an unverifiable shard",
                  file=sys.stderr)
            return 1
        metas.append(json.loads(mp.read_text()))
    modes = {m.get("mode") for m in metas}
    shas = {m.get("cloud_sha256") for m in metas}
    if modes != {"xtb"}:
        print(f"promotion gate: the records' mode set = {modes!r} -- only "
              "an xtb (real-provider) landscape promotes; a stub landscape "
              "is mechanical only", file=sys.stderr)
        return 1
    if shas != {expected}:
        print(f"promotion gate: the records' cloud sha set = {shas!r} != "
              f"the expected ({'canonical' if expected == CANONICAL_CLOUD_SHA256 else 'the declared replicate identity'}) "
              "-- refuse", file=sys.stderr)
        return 1
    stage_report = Path(work_root) / "enumerate_report.json"
    # NOTE (fix #6): the RETURNED stage report's own status field is
    # informational from here on -- the completeness ARBITER is this
    # re-aggregation + the coverage identity (the pre-fix stage code
    # mis-read coverage as n_rows_evaluated == n_rows_landscape and
    # marked a fully-covered run (evaluated + exclusions) incomplete;
    # its record stays, its ruling does not).
    if not stage_report.is_file():
        print(f"{stage_report}: the stage's own report is missing -- "
              "refuse (the promotion gates on its presence)", file=sys.stderr)
        return 1
    rep = json.loads(stage_report.read_text())
    if rep.get("cloud_sha256") != expected:
        print(f"{stage_report}: the stage report's cloud sha is not the "
              "expected one -- refuse", file=sys.stderr)
        return 1
    if args.mode != "xtb":
        print("promotion runs with the default --mode xtb (the canonical "
              "artifact's provider identity)", file=sys.stderr)
        return 1
    candidates = np.load(cloud)
    try:
        universe = en.build_universe(candidates, d_min=en.D_MIN_A)
        landscape = en.aggregate_shards(
            [str(p) for p in shard_dirs], universe, expect_rows=None)
    except (RuntimeError, ValueError) as e:
        print(f"the returned records FAIL the verification gate: {e}",
              file=sys.stderr)
        return 1
    # the completeness ARBITER: the fresh re-aggregation's coverage
    cov = coverage_report(landscape, universe["n_rows"])
    if not cov["coverage_identity"]:
        print("promotion refused: the re-aggregation's coverage is "
              f"incomplete ({cov['n_rows_evaluated']} evaluated + "
              f"{cov['n_rows_excluded']} exclusions != "
              f"{cov['n_rows_landscape']}) -- never promote an "
              "incomplete landscape", file=sys.stderr)
        return 1
    canon_default = REPO_ROOT / "data/derived/enumeration_v2"
    target_raw = (target_dir or getattr(args, "target_dir", None)
                  or (canon_default if expected == CANONICAL_CLOUD_SHA256
                      else None))
    target = Path(target_raw) if target_raw is not None else None
    if target is None:
        print("promotion refused: a non-canonical (replicate) promotion "
              "must declare its own target dir (--target-dir, under the "
              "replicate root) -- the canonical artifact root is not a "
              "default here", file=sys.stderr)
        return 1
    # the target fence: only the canonical promotion lands under
    # data/derived/, and only at the frozen default
    target_res = str(target.resolve())
    data_res = str((REPO_ROOT / "data/derived").resolve())
    if target_res.startswith(data_res + os.sep) and target != canon_default:
        print(f"{target}: refuses to promote into data/derived/ other "
              "than the canonical enumeration_v2 default", file=sys.stderr)
        return 1
    if target.exists():
        print(f"{target}: already exists -- promotion is a single, "
              "deliberate act (remove/move it first)", file=sys.stderr)
        return 1
    target.mkdir(parents=True)
    summary = en.landscape_summary(
        universe, landscape, sha,
        en.sha256_file(REPO_ROOT / "data/derived/structures/ts_opt.xyz"),
        en.sha256_file(REPO_ROOT / "data/derived/structures/"
                       "chorismate_reactive_TSframe.xyz"),
        _provider_meta(args.mode),
        {"promoted_utc": datetime.now(timezone.utc).isoformat(),
         "returned_work_root": str(work_root),
         "stage_report_status_as_returned": rep.get("status"),
         "coverage": cov,
         "expected_cloud_sha256": expected})
    summary = en.write_landscape(target, universe, landscape, summary)
    print(f"promoted: coverage {cov['n_rows_evaluated']} evaluated + "
          f"{cov['n_rows_excluded']} domain-excluded = "
          f"{cov['n_rows_landscape']} rows; global min "
          f"{summary['fitness']['global_min']['f']:.6f} kcal/mol; "
          f"artifact -> {target} (the caller integrates with 04)")
    return 0


if __name__ == "__main__":
    sys.exit(main())