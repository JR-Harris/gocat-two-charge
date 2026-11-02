#!/usr/bin/env python3
"""One enumeration shard worker (a distinct OS process per shard).

Executes a contiguous row range [start, end) of the canonical row
order through the injected evaluator, resumable per
exhaustive_search/enumerate.run_shard. Two modes:

  stub   the synthetic reference provider (fitness.stub) -- mechanical
         verification of the core; no scientific meaning.
  xtb    the canonical provider (fitness.xtb): GFN2-xTB embedded-point-
         charge single points through the provider's public
         eval_batch_full (the landscape-artifact view the contract
         assigns). The worker's close-window hook parses the
         provider's fresh single-point outputs for the exact
         PC-potential plumbing flag, re-derives each parsed energy and
         requires bitwise agreement with the provider's returned
         values (the value chain verified at the source), then sweeps
         the configuration's scratch dir (the canonical run must not
         accumulate GB-scale scratch; the records keep the evidence).

CLI (single worker; the driver/stage computes the shard ranges):

  python -m exhaustive_search.worker \
      --cloud PATH --cloud-sha256 HEX \
      --mode {stub,xtb} \
      --start R --end R2 \
      --work-dir PATH [--provider-scratch PATH] [--flush N]

Exits 0 when the range completes; on ANY failure the shard writes its
(shard_meta.json) diagnostic and exits 1 -- no retries, no penalty
values; the caller resumes explicitly. All writes live under
--work-dir (+ the provider scratch given).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from exhaustive_search import enumerate as en
from fitness import stub as stub_module

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_stub_evaluator():
    """The stub mode's evaluator: a FRESH provider instance (its own
    n_evals ledger; no canonical data reads) + the eval_batch_full
    view (the artifact path; bitwise = the stub's f)."""
    provider = stub_module.StubProvider()
    return provider, provider.eval_batch_full


def domain_wrapped_evaluator(plain_eval, universe, shard_dir,
                             keep_raw_dirs=None):
    """The DOMAIN-layer wrapper around the evaluator (spec 3's residual
    rule): the SCC non-convergence signature under the frozen T2
    protocol is OUTSIDE the fitness's domain — recorded into the
    shard's append-only `domain_exclusions.jsonl` ledger (raw evidence
    = the provider's kept scratch dir, verified at aggregation),
    never penalised, never retried, the run CONTINUES. Every OTHER
    failure re-raises (full stop).

    Returns the 4-tuple convention (f, e_ts, e_r, keep_mask) aligning
    with the window's rows; keep_mask marks the evaluated rows (the
    excluded rows carry no value and are dropped by run_shard from
    the part + manifest). Batch efficiency is kept: the plain call
    runs first; ONLY a window containing exclusions falls back to a
    per-row pass (the rare path).
    """
    import json as _json
    from fitness import xtb as ff
    shard_dir = Path(shard_dir)
    excl_path = shard_dir / "domain_exclusions.jsonl"
    pair_to_p = {"{}:{}".format(int(a), int(b)): p for p, (a, b) in
                 enumerate(zip(universe["pi"], universe["pj"]))}
    utc_now = None

    def _record_row(row_id, exc):
        entry = {
            "row": int(row_id), "config": [int(v) for v in exc.config],
            "stage": exc.stage, "marker": "scc-not-converged",
            "raw_rel": (str(Path(exc.raw_dir).relative_to(
                shard_dir.parent)) if exc.raw_dir and
                str(Path(exc.raw_dir).resolve()).startswith(
                    str(shard_dir.parent.resolve())) else None),
            "raw_abs": str(exc.raw_dir),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(excl_path, "a") as fh:
            fh.write(_json.dumps(entry) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if keep_raw_dirs is not None and exc.raw_dir:
            keep_raw_dirs.add(str(exc.raw_dir))
        return entry

    def wrapped(idx_i, idx_j, sign_i, sign_j, cand):
        m = idx_i.size
        try:
            return plain_eval(idx_i, idx_j, sign_i, sign_j, cand)
        except ff.SccDomainExclusion as first:
            if m == 1:
                # the single-row window: the exclusion is the whole window
                i, j, si, sj = (int(idx_i[0]), int(idx_j[0]),
                                int(sign_i[0]), int(sign_j[0]))
                p = pair_to_p[f"{i}:{j}"]
                s = 0 if (si, sj) == en.SIGN_COMBOS[0] else (
                    1 if (si, sj) == en.SIGN_COMBOS[1] else
                    2 if (si, sj) == en.SIGN_COMBOS[2] else 3)
                _record_row(4 * p + s, first)
                return (np.empty(0, dtype=np.float64),) * 3 + (
                    np.ones(0, dtype=bool),)
            # the rare path: the batch lost the batching; per-row pass
            f = np.empty(m, dtype=np.float64)
            e_ts = np.empty(m, dtype=np.float64)
            e_r = np.empty(m, dtype=np.float64)
            keep = np.ones(m, dtype=bool)
            for k in range(m):
                try:
                    f1, t1, r1 = plain_eval(idx_i[k:k + 1], idx_j[k:k + 1],
                                            sign_i[k:k + 1], sign_j[k:k + 1],
                                            cand)
                    f[k], e_ts[k], e_r[k] = (float(f1[0]), float(t1[0]),
                                             float(r1[0]))
                except ff.SccDomainExclusion as exc:
                    i, j, si, sj = (int(idx_i[k]), int(idx_j[k]),
                                    int(sign_i[k]), int(sign_j[k]))
                    p = pair_to_p[f"{i}:{j}"]
                    s = 0 if (si, sj) == en.SIGN_COMBOS[0] else (
                        1 if (si, sj) == en.SIGN_COMBOS[1] else
                        2 if (si, sj) == en.SIGN_COMBOS[2] else 3)
                    keep[k] = False
                    _record_row(4 * p + s, exc)
            return f, e_ts, e_r, keep

    return wrapped


def make_xtb_evaluator(provider_scratch, cloud_path=None,
                       cloud_sha256=None):
    """The xtb mode's evaluator: a fresh provider instance (its own
    identity gates) + the public eval_batch_full view -- the contract's
    landscape-artifact path. Work-item-4b extension: the worker's cloud
    path + sha256 are passed through EXPLICITLY (the provider's default
    stays the canonical locked cloud with its frozen sha — the canonical
    worker invocation passes exactly the canonical pair, behavior
    unchanged; a replicate cloud reaches the provider only through its
    own declared identity)."""
    from fitness import xtb as ff

    provider = ff.XtbProvider(scratch_root=Path(provider_scratch),
                              cloud_path=cloud_path,
                              cloud_sha256=cloud_sha256)
    return provider, provider.eval_batch_full


def make_xtb_close_window(provider, keep_raw_dirs=None):
    """The close-window hook for the xtb mode.

    (rows, f, e_ts, e_r, universe, candidates) -> (pc_ts, pc_r): for
    each row, parse the provider's two freshly written single-point
    outputs (the config-keyed scratch of this provider instance):
      * require the exact 'PC potential true' plumbing flag in BOTH
        stages' setup blocks (an embedding silently dropped would
        return the free energy -- the --tblite failure mode);
      * re-parse each TOTAL ENERGY and require BITWISE agreement with
        the evaluator's returned component (the value chain verified
        at the source);
      * then sweep the configuration's scratch dir (the PC evidence +
        the verified values are already in the records) EXCEPT the
        kept raw dirs of the domain exclusions (the failing configs'
        raw single points stay on disk for the aggregation's evidence
        check).
    Fails hard, naming the row, its configuration and the stage.
    """
    from fitness import xtb as ff

    ts_sha, r_sha = provider._ts_sha, provider._r_sha

    def hook(rows, f, e_ts, e_r, universe, candidates):
        pc_ts = np.zeros(rows.size, dtype=bool)
        pc_r = np.zeros(rows.size, dtype=bool)
        for k, row in enumerate(rows):
            i = int(universe["idx_i"][row])
            j = int(universe["idx_j"][row])
            si = int(universe["sign_i"][row])
            sj = int(universe["sign_j"][row])
            key = ff.config_key(i, j, si, sj, ts_sha, r_sha)
            cfg_dir = provider.scratch_root / key
            for stage, expect in (("ts", float(e_ts[k])),
                                  ("reactant", float(e_r[k]))):
                out_dir = cfg_dir / stage
                text = (out_dir / "xtb.out").read_text() + \
                       (out_dir / "xtb.err").read_text()
                ff.check_scf_markers(text, stage)  # the plumbing belt:
                # 'PC potential true' required + the unconverged-SCF
                # markers fail hard, naming the stage + the config row
                energy = ff.parse_xtb_energy(text)
                if float(energy) * ff.KCAL_PER_EH != float(expect):
                    raise RuntimeError(
                        f"the re-parsed {stage} energy at row {int(row)} "
                        f"({en.config_label(universe, int(row))}) disagrees "
                        f"with the provider's returned component "
                        f"({energy!r} Eh * kcal_per_eh vs {expect!r} kcal/mol)")
                if stage == "ts":
                    pc_ts[k] = True
                else:
                    pc_r[k] = True
            if cfg_dir.is_dir():
                if keep_raw_dirs is not None and str(cfg_dir) in keep_raw_dirs:
                    continue   # a domain exclusion's raw evidence: kept
                shutil.rmtree(cfg_dir)
        return pc_ts, pc_r

    return hook


def run_worker(args):
    """One worker session: load + gate the cloud, build the universe,
    evaluate the range, return the exit code."""
    cloud_sha = en.sha256_file(args.cloud)
    if cloud_sha != args.cloud_sha256:
        raise ValueError(
            f"{args.cloud}: cloud sha256 mismatch (got {cloud_sha}, "
            f"expected {args.cloud_sha256} -- pass the file's own sha256 "
            "explicitly; no implicit acceptance of a wrong cloud)")
    candidates = np.load(args.cloud)

    if args.mode == "stub":
        provider, evaluator = make_stub_evaluator()
        close_window = None
        evaluator = domain_wrapped_evaluator(
            evaluator, en.build_universe(candidates, d_min=en.D_MIN_A),
            args.work_dir)  # the stub raises no exclusions; inert wiring
    elif args.mode == "xtb":
        provider, evaluator = make_xtb_evaluator(
            args.provider_scratch, cloud_path=args.cloud,
            cloud_sha256=args.cloud_sha256)
        universe = en.build_universe(candidates, d_min=en.D_MIN_A)
        keep_raw = set()
        evaluator = domain_wrapped_evaluator(evaluator, universe,
                                             args.work_dir, keep_raw)
        close_window = make_xtb_close_window(provider, keep_raw)
    else:
        raise ValueError(f"unknown mode {args.mode!r}")

    universe = en.build_universe(candidates, d_min=en.D_MIN_A)
    start, end = int(args.start), int(args.end)
    if not 0 <= start <= end <= universe["n_rows"]:
        raise ValueError(
            f"range [{start}, {end}) outside the universe's post-floor rows "
            f"[0, {universe['n_rows']})")
    if end > start and (start % 4 != (end % 4 if end % 4 else 4)):
        pass  # ranges may split mid-pair; row-major math does not care

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    session = en.run_shard(
        work_dir, universe, candidates, evaluator, start, end,
        flush_rows=int(args.flush),
        close_window=close_window,
        cloud_sha256=args.cloud_sha256,
        shard_meta={"mode": args.mode,
                    "cloud": str(args.cloud),
                    "provider_fitness_id": provider.FITNESS_ID},
    )
    print(f"worker: range [{start}, {end}) done; rows evaluated this "
          f"session = {session}; provider n_evals = {provider.n_evals}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--cloud-sha256", required=True)
    parser.add_argument("--mode", choices=("stub", "xtb"), default="xtb")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--provider-scratch", default=None)
    parser.add_argument("--flush", type=int, default=500)
    args = parser.parse_args(argv)
    if args.provider_scratch is None and args.mode == "xtb":
        parser.error("--provider-scratch is required in xtb mode")
    try:
        return run_worker(args)
    except Exception as e:
        import traceback
        print(f"WORKER FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())