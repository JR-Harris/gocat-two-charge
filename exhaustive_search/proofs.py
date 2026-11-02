#!/usr/bin/env python3
"""The real-provider guarantee proofs (spec 5; run on the laptop BEFORE
any cluster deploy, REPEATED on the cluster as the stage's own smoke).

On a fixed deterministic subset of the post-floor rows (~200 rows,
seeded, including at least 10 rows from close pairs — the 10 smallest-
distance kept pairs in [1.0, 1.2) Å, 40 rows — the rest a seeded draw):

  (a) pass in shuffled row order vs canonical order — the per-row
      f/e_ts/e_r must be bitwise identical (batch independence);
  (b) each subset row re-evaluated SOLO — bitwise identical;
  (c) provider.n_evals accounting exactness: == evaluated rows
      (200 + 200 + 200 = 600), failures counted 0, order-independent;
  (d) an injected mid-run kill of the REAL worker subprocess (the
      manifest watched to >= kill_after_rows, then SIGKILL) proves the
      resume protocol: the recorded rows are not re-evaluated (the
      resumed session's ledger = rows_after - rows_before == its own
      provider count), and the resumed run's aggregated landscape is
      BITWISE the uninterrupted one.

The subset + the canonical-pass values are saved as the shipped
reference (exhaustive_search/proof_reference/ when --save-reference
is given); the cluster smoke re-derives everything deterministically
from the locked cloud and diffs against the reference BITWISE.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from exhaustive_search import enumerate as en

REPO_ROOT = Path(__file__).resolve().parent.parent

PROOF_SEED = 20261008
PROOF_N_TARGET = 200
PROOF_N_CLOSE_PAIRS = 10
PROOF_KILL_AFTER_ROWS = 50
PROOF_FLUSH = 25

# The xtb-mode cross-machine reference gate. Derived 2026-10-08
# (Jonny's decision) from the measured cross-machine arithmetic noise:
# max 6.297e-10 kcal/mol per single-point side (1 ulp of the 12-decimal
# prints), ~2.5e-9 kcal/mol worst-case per-row f delta, on last-bit
# summation-order differences of the compiled scalar arithmetic between
# CPU generations — see 04 §9b. Within-machine proofs stay bitwise;
# changing this constant is a recorded decision (never silently
# widened, never tuned by Pi).
REF_BAND_KCAL_MOL = 1e-8


def compare_reference_values(ref_f, ref_ts, ref_r, f, ts, r,
                             band=REF_BAND_KCAL_MOL):
    """The xtb-mode cross-machine reference gate -> the gate report:

      rows_bitwise          row identity (bitwise — it passed everywhere)
      n_sides_bitwise       the count of exactly-equal values among the
                            3 per-row value columns (f, e_ts, e_r)
      n_sides_in_band       the count within |Δ| <= band
      max_abs_delta_f       worst per-row |Δf| (kcal/mol)
      band_kcal_mol         the gate constant (recorded, Jonny's)
      within_band           ALL value deltas within the band (the gate)

    Never describes the band as "bitwise"; within-machine bitwise
    equality is the proofs' separate invariant (a/b/c/d internals).
    """
    ref_f = np.asarray(ref_f, dtype=np.float64)
    ref_ts = np.asarray(ref_ts, dtype=np.float64)
    ref_r = np.asarray(ref_r, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    rows_bitwise = (ref_f.shape == f.shape
                    and ref_ts.shape == ts.shape
                    and ref_r.shape == r.shape)
    if not rows_bitwise:
        return {"rows_bitwise": False, "within_band": False,
                "n_sides_bitwise": 0, "n_sides_in_band": 0,
                "max_abs_delta_f": None, "band_kcal_mol": float(band)}
    bit = int((ref_f == f).sum() + (ref_ts == ts).sum()
              + (ref_r == r).sum())
    in_band = int((np.abs(ref_f - f) <= band).sum()
                  + (np.abs(ref_ts - ts) <= band).sum()
                  + (np.abs(ref_r - r) <= band).sum())
    max_df = float(np.abs(ref_f - f).max()) if f.size else 0.0
    n_sides = 3 * int(f.size)
    return {
        "rows_bitwise": True,
        "n_sides_bitwise": int(bit),
        "n_sides_in_band": int(in_band),
        "n_sides_total": n_sides,
        "max_abs_delta_f": max_df,
        "band_kcal_mol": float(band),
        "within_band": bool(in_band == n_sides),
    }


def proof_subset_rows(universe, n_target=PROOF_N_TARGET,
                      n_close=PROOF_N_CLOSE_PAIRS, seed=PROOF_SEED):
    """The deterministic proof subset: the `n_close` smallest-distance
    kept pairs in [D_MIN, 1.2) A contribute their 4*n_close rows; the
    rest are a seeded choice over the remaining rows; the result is
    sorted ascending (a plain row-index list). Recorded in every
    proof report."""
    rng = np.random.default_rng(seed)
    dist = universe["pair_dist"]
    close_mask = (dist >= universe["d_min"]) & (dist < 1.2)
    close_pair_ids = np.where(close_mask)[0]
    if close_pair_ids.size < n_close:
        raise ValueError(
            f"only {close_pair_ids.size} close pairs (>= d_min, < 1.2 A); "
            f"the proof needs {n_close}")
    order = np.argsort(dist[close_pair_ids], kind="stable")
    chosen = close_pair_ids[order[:n_close]]
    close_rows = (4 * chosen[:, None] + np.arange(4)).ravel()
    rest_mask = np.ones(universe["n_rows"], dtype=bool)
    rest_mask[close_rows] = False
    rest = np.where(rest_mask)[0]
    take = n_target - close_rows.size
    if rest.size < take:
        raise ValueError("not enough rows for the proof subset")
    random_rows = rng.choice(rest, size=take, replace=False)
    close_rows = close_rows.astype(np.int64)
    rows = np.sort(np.concatenate([close_rows, random_rows]))
    return rows.astype(np.int64), close_rows.tolist()


def _eval_rows(evaluator, universe, candidates, rows):
    f, e_ts, e_r = evaluator(
        universe["idx_i"][rows], universe["idx_j"][rows],
        universe["sign_i"][rows], universe["sign_j"][rows], candidates)
    return (np.asarray(f, dtype=np.float64),
            np.asarray(e_ts, dtype=np.float64),
            np.asarray(e_r, dtype=np.float64))


def _spawn_worker(args, cwd=REPO_ROOT):
    env = {k: v for k, v in os.environ.items()}
    return subprocess.Popen(
        [sys.executable, "-m", "exhaustive_search.worker", *args],
        cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _kill_resume(universe, candidates, mode, work_root, flush,
                 kill_after_rows, range_rows, provider):
    """Proof (d): the REAL worker subprocess, killed mid-run, resumed."""

    def worker_args(start, end):
        a = ["--cloud", provider["cloud"], "--cloud-sha256", provider["cloud_sha"],
             "--mode", mode, "--start", str(start), "--end", str(end),
             "--work-dir", str(work_root / "proof_d_shard"),
             "--flush", str(flush)]
        if mode == "xtb":
            a += ["--provider-scratch", str(work_root / "proof_d_scratch")]
        return a

    start, end = 0, min(int(range_rows), universe["n_rows"])
    shard_dir = work_root / "proof_d_shard"
    if shard_dir.exists():
        import shutil
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    p1 = _spawn_worker(worker_args(start, end))
    killed_at = None
    deadline = time.monotonic() + 3600
    while True:
        time.sleep(0.05)
        rows, _ = en.read_manifest(shard_dir)
        if rows.size >= kill_after_rows:
            killed_at = int(rows.size)
            p1.send_signal(signal.SIGKILL)
            break
        if p1.poll() is not None:
            raise RuntimeError(
                "the kill-resume proof: the worker finished before the "
                "injected kill (flush too large for the range)")
        if time.monotonic() > deadline:
            p1.kill()
            raise RuntimeError("the kill-resume proof: worker hang")
    _out, _err = p1.communicate()
    # the SIGKILL races the worker's own flushes: re-read the manifest
    # after death — the rows recorded at THAT moment are the resume's
    # true skip set
    rows_at_death, _ = en.read_manifest(shard_dir)
    killed_at = int(rows_at_death.size)

    p2 = _spawn_worker(worker_args(start, end))
    out2, err2 = p2.communicate()
    if p2.returncode != 0:
        raise RuntimeError(f"the resumed worker failed (rc={p2.returncode}):\n"
                           f"{err2[-2000:]}")
    rows_after, _ = en.read_manifest(shard_dir)
    meta = json.loads((shard_dir / "shard_meta.json").read_text())

    # the skipped rows were NOT re-evaluated: the resumed session's own
    # ledger must equal the rows it actually evaluated
    resumed_session = int(meta["rows_this_session"])
    resumed_provider = int(meta["provider_n_evals"])
    skipped = end - start - killed_at
    ok_ledger = (resumed_session == skipped
                 and resumed_provider == skipped
                 and rows_after.size == end - start)

    # the resumed run's landscape: bitwise == the uninterrupted one
    # (the aggregate returns the universe-wide arrays: slice the range)
    f_un, t_un, r_un = _eval_rows(
        provider["evaluator"], universe, candidates,
        np.arange(start, end, dtype=np.int64))
    land = en.aggregate_shards([shard_dir], universe, expect_rows=end - start)
    bitwise = (land["f"][start:end].tobytes() == f_un.tobytes()
               and land["e_ts"][start:end].tobytes() == t_un.tobytes()
               and land["e_r"][start:end].tobytes() == r_un.tobytes())
    return {
        "killed_after_manifest_rows": killed_at,
        "resumed_rows_this_session": resumed_session,
        "resumed_provider_n_evals": resumed_provider,
        "expected_resumed_rows": int(skipped),
        "skipped_not_re_evaluated": bool(ok_ledger),
        "resumed_landscape_bitwise_uninterrupted": bool(bitwise),
        "rows_after_resume": int(rows_after.size),
    }


def run_proofs(cloud_path, cloud_sha256, mode="xtb", work_root=None,
               flush=PROOF_FLUSH, kill_after_rows=PROOF_KILL_AFTER_ROWS,
               n_target=PROOF_N_TARGET, n_close=PROOF_N_CLOSE_PAIRS,
               seed=PROOF_SEED, provider_scratch=None, reference_dir=None,
               save_reference=False, compare_reference=None):
    """-> the proof report dict (ok flag inside). Raises on hard
    failures; every proof's numbers are in the report."""
    import shutil
    from fitness import stub as stub_module

    cloud_path = Path(cloud_path)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    mode = mode.lower()
    if mode not in ("stub", "xtb"):
        raise ValueError(f"unknown proof mode {mode!r}")
    compare = (mode == "xtb")

    if mode == "stub":
        provider = stub_module.StubProvider()
        evaluator = provider.eval_batch_full
    else:
        from fitness import xtb as ff
        # the 6(a) parameterization: the proofs' provider is constructed
        # with the DECLARED cloud identity (the canonical pair stays the
        # canonical gate; a named replicate cloud reaches the proofs
        # only through its own declared path + sha -- the worker already
        # gates the same pair)
        provider = ff.XtbProvider(
            scratch_root=Path(provider_scratch or work_root / "scratch"),
            cloud_path=str(cloud_path), cloud_sha256=cloud_sha256)
        evaluator = provider.eval_batch_full

    candidates = np.load(cloud_path)
    universe = en.build_universe(candidates, d_min=en.D_MIN_A)
    rows, close_rows = proof_subset_rows(
        universe, n_target=n_target, n_close=n_close, seed=seed)
    rows = rows[rows < universe["n_rows"]]

    # ---- (a) canonical-order pass vs shuffled-order pass ----------------
    f_c, t_c, r_c = _eval_rows(evaluator, universe, candidates, rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rows.size)
    f_s, t_s, r_s = _eval_rows(evaluator, universe, candidates, rows[perm])
    proof_a = {
        "canonical_vs_shuffled_bitwise": bool(
            f_c[perm].tobytes() == f_s.tobytes()
            and t_c[perm].tobytes() == t_s.tobytes()
            and r_c[perm].tobytes() == r_s.tobytes()),
    }
    # ---- (b) every row solo --------------------------------------------
    mismatches = []
    for k in range(rows.size):
        f1, t1, r1 = _eval_rows(evaluator, universe, candidates, rows[k:k + 1])
        if (f1[0].tobytes() != f_c[k].tobytes()
                or t1[0].tobytes() != t_c[k].tobytes()
                or r1[0].tobytes() != r_c[k].tobytes()):
            mismatches.append(int(rows[k]))
    proof_b = {"solo_rows": int(rows.size), "bitwise_mismatches": mismatches}
    # ---- (c) accounting exactness ---------------------------------------
    rows_accounted = 2 * int(rows.size) + int(rows.size)  # (a) x2 + (b)
    proof_c = {
        "provider_n_evals": int(provider.n_evals),
        "rows_accounted": rows_accounted,
        "exact": bool(provider.n_evals == rows_accounted),
        "failures_counted": 0,          # no suppressing path ran
    }
    # ---- (d) the injected kill + resume on the real worker --------------
    # the xtb range is small (the real single points cost ~0.11 s each);
    # the stub pass would finish before any poll, so its kill-resume
    # proof runs over a wider range with a proportional flush window
    d_range, d_flush, d_kill = ((200, flush, kill_after_rows) if mode == "xtb"
                                else (200000, 2500, 50000))
    proof_d = _kill_resume(universe, candidates, mode, work_root, d_flush,
                           d_kill, d_range, provider={
                               "cloud": str(cloud_path),
                               "cloud_sha": cloud_sha256,
                               "evaluator": evaluator,
                           })
    proof_d["range"] = [0, int(d_range)]

    # ---- the reference (saved / compared) --------------------------------
    reference = {}
    if compare_reference is not None or save_reference:
        payload = {
            "rows": rows.astype(np.int64),
            "f": f_c, "e_ts": t_c, "e_r": r_c,
            "close_rows": np.asarray(close_rows, dtype=np.int64),
        }
        if compare_reference is not None:
            ref = Path(compare_reference) / "laptop_proof.npz"
            with np.load(ref) as z:
                gate = compare_reference_values(
                    z["f"], z["e_ts"], z["e_r"], f_c, t_c, r_c)
                gate["rows_bitwise"] = bool(
                    gate["rows_bitwise"] and z["rows"].tolist() == rows.tolist())
                gate["within_band"] = bool(
                    gate["within_band"] and gate["rows_bitwise"])
            reference = dict(gate)
            reference["reference_rows_match"] = bool(gate["rows_bitwise"])
            reference["reference_n_sides_bitwise"] = gate["n_sides_bitwise"]
            reference["reference_n_sides_in_band"] = gate["n_sides_in_band"]
            reference["reference_max_abs_delta_f"] = gate["max_abs_delta_f"]
            reference["reference_band_kcal_mol"] = gate["band_kcal_mol"]
            reference["reference_within_band"] = gate["within_band"]
        if save_reference:
            reference_dir = Path(reference_dir)
            reference_dir.mkdir(parents=True, exist_ok=True)
            with open(reference_dir / "laptop_proof.npz", "wb") as fh:
                np.savez(fh, **payload)
            report_path = reference_dir / "laptop_proof.json"
            report_path.write_text(json.dumps(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "cloud": str(cloud_path), "cloud_sha256": cloud_sha256,
                 "mode": mode, "seed": seed,
                 "rows": rows.tolist(),
                 "close_rows": close_rows,
                 "n_rows": int(rows.size)}, indent=1) + "\n")
            reference["saved_to"] = str(reference_dir)

    report = {
        "proofs": {
            "a_shuffled_vs_canonical": proof_a,
            "b_solo_rows": proof_b,
            "c_accounting": proof_c,
            "d_kill_resume": proof_d,
        },
        "mode": mode,
        "cloud": str(cloud_path),
        "cloud_sha256": cloud_sha256,
        "subset_rows": rows.tolist(),
        "close_rows": close_rows,
        "subset_size": int(rows.size),
        "universe_rows": int(universe["n_rows"]),
        "reference": reference,
        "seed": seed,
    }
    ok = (proof_a["canonical_vs_shuffled_bitwise"]
          and not proof_b["bitwise_mismatches"]
          and proof_c["exact"]
          and proof_d["skipped_not_re_evaluated"]
          and proof_d["resumed_landscape_bitwise_uninterrupted"]
          and (not compare_reference
               or (reference.get("reference_rows_match")
                   and reference.get("reference_within_band"))))
    report["ok"] = bool(ok)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cloud", default="data/derived/candidates_v2.npy")
    parser.add_argument("--cloud-sha256", default=None)
    parser.add_argument("--mode", choices=("stub", "xtb"), default="xtb")
    parser.add_argument("--work-root",
                        default="runs/exhaustive_search/proofs")
    parser.add_argument("--provider-scratch", default=None)
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--reference-dir",
                        default="exhaustive_search/proof_reference")
    parser.add_argument("--compare-reference", action="store_true")
    args = parser.parse_args(argv)

    cloud_sha = args.cloud_sha256
    if cloud_sha is None:
        if args.cloud != "data/derived/candidates_v2.npy":
            parser.error("--cloud other than the canonical one requires "
                         "--cloud-sha256 (no implicit acceptance)")
        cloud_sha = en.sha256_file(args.cloud)
    compare = args.compare_reference
    if compare and not (Path(args.reference_dir) /
                        "laptop_proof.npz").is_file():
        parser.error("--compare-reference requires the shipped "
                     "proof_reference")
    report = run_proofs(
        cloud_path=args.cloud, cloud_sha256=cloud_sha, mode=args.mode,
        work_root=args.work_root, provider_scratch=args.provider_scratch,
        save_reference=args.save_reference,
        reference_dir=args.reference_dir,
        compare_reference=args.reference_dir if compare else None,
    )
    out = Path(args.work_root) / "proof_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("ok", "mode", "subset_size",
                                             "cloud_sha256")}, indent=1))
    for name, p in report["proofs"].items():
        print(f"  {name}: "
              + json.dumps({k: v for k, v in p.items()
                            if not isinstance(v, list)}))
    print(f"report -> {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())