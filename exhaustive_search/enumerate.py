"""The exhaustive enumeration core (work items 2-3).

THE oracle: a complete, hashed, resumable map of
    f(q) = E(TS;q) - E(R;q)   [kcal/mol, lower = better]
over every two-charge configuration the locked cloud allows, q = two
+-1 e point charges at the candidate sites (the canonical sign order
[(+,+), (+,-), (-,+), (-,-)], the v1-confirmed canon; the semantic
reference is the v1 src/brute_force.py, consulted read-only).

Universe (asserted at runtime from n):
    all unordered pairs i < j over the (n, 3) cloud rows, in
    np.triu_indices row-major order, each pair with the 4 canonical
    signs -> configurations in pair-major flat order, row r = 4*p + s
    (p = pair rank post-floor, s = sign index). At n = 928:
    C(928,2) = 430,128 pairs, x4 = 1,720,512 configurations
    (asserted exactly); the d_min = 1.0 A floor excludes pairs whose
    site distance is < 1.0 (excluded pairs are COUNTED AND RECORDED in
    every provenance record -- never penalised, never retried). At the
    locked cloud: 5,345 pairs / 21,380 configurations excluded ->
    424,783 pairs / 1,699,132 post-floor configurations.

Black-box rule + the artifact reconciliation: the reusable
search-facing machinery treats the provider strictly through the
frozen contract (a searcher would call provider.eval_batch). THE
CANONICAL RUN STORES A LANDSCAPE ARTIFACT, and the contract assigns
exactly that path its own view: "Anything that stores landscape
artifacts calls eval_batch_full" -- so the sharded runner evaluates
through the provider's PUBLIC `eval_batch_full` (never the private
impl, never the provider's internals), carrying e_ts/e_r for the
artifact with f bitwise = the eval_batch value (one and the same call
underneath; two calls would double the oracle cost). The evaluator is
an injected callable; no module here imports provider internals.

Shard-parallel, kill-safe, resumable: the run splits into
deterministic contiguous row-range shards (equal widths given (n_rows,
k)); the driver runs up to K concurrent single-thread OS-process
workers (exhaustive_search.worker). Each shard owns a PRIVATE directory
with: an append-only checkpoint manifest (one row index per line for
every flushed configuration, O_APPEND + fsync) and chunked energy
records (one part file per flush window: rows int32; f/e_ts/e_r
float64 -- the provider-boundary values, storage-cast to float32 only
at promotion; the two PC-potential plumbing booleans). Ordering inside
a shard: the part record lands atomically (tmp + fsync + rename)
BEFORE its manifest lines, so a killed run leaves rows either fully
recorded (part + manifest) or not at all; the resume reads the
manifest and skips recorded rows WITHOUT re-evaluating them and never
counts them twice (the REGISTRY number = distinct configurations
evaluated; lost-tail work from a kill is simply re-evaluated once on
resume). No shared mutable state, no locks.

Verify-then-promote: aggregation of the shard directories re-parses
every record and fails hard on any gap, duplicate row, count
mismatch, non-finite value or absent PC plumbing flag -- the stage
marks a run incomplete rather than ever "complete with gaps". The
landscape artifact (three float32 arrays + the (n_rows, 2) int32 pair
table + the per-row int8 sign index + the summary with sha256s)
is written ONLY at promotion time, laptop-side, under the caller's
direction; nothing is ever written under data/derived/ by the cluster.

Multi-cloud readiness (structure only): the cloud path + sha256 are
parameters of every entry point (never module constants); every
provenance record carries its cloud's sha256 from the arguments; the
work/checkpoint roots are keyed per (cloud sha, stage) so replicate
landscapes process in isolation, per the standing cache-hazard rule.
For the canonical run the driver keeps the hard identity gates of the
briefing (the canonical path + its frozen sha256 are the defaults; any
other cloud requires its own sha256 explicitly -- no implicit
acceptance). A replicate-cloud provider variant lands in a LATER
briefing; nothing here generates, names or references a replicate
cloud.

No retries, no penalty values: any provider raise / non-finite /
plausibility breach / missing plumbing flag fails the shard hard
(naming row + configuration + stage), the worker writes its
diagnostic and exits nonzero, and the stage stops (resume is the
caller's explicit act).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# The frozen briefing conventions.
D_MIN_A = 1.0                       # the minimum charge separation floor
SIGN_COMBOS = ((+1, +1), (+1, -1), (-1, +1), (-1, -1))  # the canonical order


def sha256_file(path):
    """sha256 of a file's bytes (the provenance anchor)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_universe(candidates, d_min=D_MIN_A):
    """-> the universe dict for an (n, 3) float64 cloud.

    Asserts the full count 4*C(n,2) at runtime from n; applies the
    d_min floor to PAIRS (row-major triu order), recording the exact
    excluded counts. Returns (keys documented inline):
      n, n_pairs_total, n_configs_total   asserted structure sizes
      d_min
      pi, pj                              int64 kept-pair site indices
      pair_dist                           float64 kept-pair distances
      n_pairs_excluded, n_configs_excluded    the recorded floor counts
      n_rows (= 4 * kept pairs)
      idx_i, idx_j, sign_i, sign_j        int64 per-row config arrays
      sign_index                          int8 per-row sign index (0..3)
    """
    cand = np.asarray(candidates)
    if cand.ndim != 2 or cand.shape[1] != 3:
        raise ValueError(f"candidates must be (n, 3), got {cand.shape}")
    if cand.dtype != np.float64:
        raise ValueError(f"candidates must be float64, got {cand.dtype}")
    n = int(cand.shape[0])
    if n < 2:
        raise ValueError(f"need >= 2 candidate points, got {n}")

    pi, pj = np.triu_indices(n, k=1)          # row-major upper triangle
    n_pairs_total = int(pi.size)
    n_configs_total = 4 * n_pairs_total
    if n_configs_total != 4 * (n * (n - 1)) // 2:
        raise RuntimeError("universe assertion failed: 4*C(n,2) mismatch")

    dist = np.linalg.norm(cand[pi] - cand[pj], axis=1)
    keep = dist >= float(d_min)
    n_pairs_excluded = int((~keep).sum())
    pi, pj, dist = pi[keep], pj[keep], dist[keep]
    n_rows = 4 * int(pi.size)

    idx_i = np.repeat(pi, 4)
    idx_j = np.repeat(pj, 4)
    si = np.tile(np.array([s[0] for s in SIGN_COMBOS], dtype=np.int64), pi.size)
    sj = np.tile(np.array([s[1] for s in SIGN_COMBOS], dtype=np.int64), pi.size)
    sign_index = np.tile(np.arange(4, dtype=np.int8), pi.size)

    return {
        "n": n, "n_pairs_total": n_pairs_total,
        "n_configs_total": n_configs_total, "d_min": float(d_min),
        "n_pairs_excluded": n_pairs_excluded,
        "n_configs_excluded": 4 * n_pairs_excluded,
        "pi": pi, "pj": pj, "pair_dist": dist, "n_rows": n_rows,
        "idx_i": idx_i, "idx_j": idx_j, "sign_i": si, "sign_j": sj,
        "sign_index": sign_index,
    }


def split_shards(n_rows, k):
    """-> [(start, end)] contiguous, disjoint, union-complete row ranges
    (deterministic in (n_rows, k); the canonical split)."""
    if k < 1:
        raise ValueError(f"shard count must be >= 1, got {k}")
    width = (n_rows + k - 1) // k
    shards = []
    start = 0
    while start < n_rows:
        end = min(start + width, n_rows)
        shards.append((start, end))
        start = end
    return shards


def shard_row_range(universe, shard_idx, shards):
    """The (start, end) row range of one shard (bounds-checked)."""
    if not 0 <= shard_idx < len(shards):
        raise ValueError(f"shard {shard_idx} outside 0..{len(shards) - 1}")
    start, end = shards[shard_idx]
    if not 0 <= start <= end <= universe["n_rows"]:
        raise RuntimeError("shard range outside the universe rows")
    return start, end


# --------------------------------------------------------------------------
# Shard-record mechanics (private per-shard state; no locks anywhere)
# --------------------------------------------------------------------------

MANIFEST_NAME = "manifest.txt"


def write_part(shard_dir, part_index, rows, f, e_ts, e_r, pc_ts, pc_r):
    """Atomically land one chunked energy record.

    rows int32 (k,); f/e_ts/e_r float64 (k,) -- the provider-boundary
    values (the float32 storage cast happens only at the final
    artifact); pc_ts/pc_r bool (k,) -- the PC-plumbing evidence parsed
    from the provider's single-point outputs by the worker. The file
    is written to a temp name, fsynced, then renamed: a killed run
    never leaves a half-visible part.
    """
    k = rows.size
    for name, arr in (("rows", rows), ("f", f), ("e_ts", e_ts), ("e_r", e_r),
                      ("pc_ts", pc_ts), ("pc_r", pc_r)):
        if np.asarray(arr).shape != (k,):
            raise ValueError(f"part column {name!r} has shape "
                             f"{np.asarray(arr).shape}, expected ({k},)")
    tmp = shard_dir / f".part_{part_index:08d}.tmp"
    payload = {
        "rows": np.asarray(rows, dtype=np.int32),
        "f": np.asarray(f, dtype=np.float64),
        "e_ts": np.asarray(e_ts, dtype=np.float64),
        "e_r": np.asarray(e_r, dtype=np.float64),
        "pc_ts": np.asarray(pc_ts, dtype=bool),
        "pc_r": np.asarray(pc_r, dtype=bool),
    }
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, shard_dir / f"part_{part_index:08d}.npz")


def append_manifest(shard_dir, rows):
    """Append one line per row (a flush window's row indices, ascending)
    to the shard's append-only manifest; O_APPEND + fsync."""
    with open(shard_dir / MANIFEST_NAME, "a", encoding="ascii") as fh:
        fh.write("".join(f"{int(r)}\n" for r in rows))
        fh.flush()
        os.fsync(fh.fileno())


def read_manifest(shard_dir):
    """-> (np.int64 sorted ascending contiguous array of the manifest's
    row indices, the raw line count). Fails hard on non-integer lines 
or
    a gap (the flush protocol records contiguous ascending row runs)."""
    path = Path(shard_dir) / MANIFEST_NAME
    if not path.is_file():
        return np.empty(0, dtype=np.int64), 0
    rows = []
    lines = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        lines += 1
        try:
            rows.append(int(line))
        except ValueError:
            raise ValueError(f"{path}: non-integer manifest line {line!r}")
    arr = np.asarray(rows, dtype=np.int64)
    if arr.size:
        if np.any(np.diff(arr) <= 0):
            raise ValueError(f"{path}: manifest rows are not strictly "
                             "ascending contiguous runs")
    return arr, lines


def list_parts(shard_dir):
    """Sorted part files of a shard dir."""
    return sorted((Path(shard_dir)).glob("part_*.npz"))


def load_part(path):
    """(rows, f, e_ts, e_r, pc_ts, pc_r) from one part record, with the
    shapes verified."""
    with np.load(path) as z:
        rows, f = z["rows"], z["f"]
        e_ts, e_r = z["e_ts"], z["e_r"]
        pc_ts, pc_r = z["pc_ts"], z["pc_r"]
    k = rows.shape[0]
    for name, arr in (("f", f), ("e_ts", e_ts), ("e_r", e_r),
                      ("pc_ts", pc_ts), ("pc_r", pc_r)):
        if arr.shape != (k,):
            raise ValueError(f"{path}: column {name!r} shape {arr.shape} "
                             f"!= ({k},)")
    return rows, f, e_ts, e_r, pc_ts, pc_r


# --------------------------------------------------------------------------
# The shard worker (the evaluative heart; one process per shard)
# --------------------------------------------------------------------------

def run_shard(shard_dir, universe, candidates, evaluator, start, end,
              flush_rows=500, shard_meta=None, close_window=None,
              cloud_sha256=None):
    """Evaluate rows [start, end) of the universe into shard_dir,
    resumable.

    evaluator: (idx_i, idx_j, sign_i, sign_j, candidates) ->
        (f, e_ts, e_r)  -- the provider's public eval_batch_full (the
        landscape-artifact view; the black-box contract surface).
    flush_rows: the eval-batch AND flush window size (each window =
        one provider call + one atoic part record + its manifest
        lines; a kill loses at most the in-flight window).
    flush_rows: the eval-batch AND flush window size (each window =
        one provider call + one atomic part record + its manifest
        lines; a kill loses at most the in-flight window).
    close_window: optional post-window hook (w, universe, candidates)
        -> (pc_ts, pc_r): the worker's plumbing-evidence parser +
        per-config scratch sweep (the canonical run must not
        accumulate GB-scale single-point scratch; the records keep
        the PC-plumbing evidence, so the raw outputs are safe to
        remove once parsed). None -> inert all-True flags (stub mode:
        the columns exist under one schema; they carry no meaning).
    shard_meta: dict written to shard_meta.json on ANY exit (ok or
        exception), carrying the shard's ledger (rows done, provider
        count, walls).

    Returns the number of rows evaluated in THIS session (the resume
    skips recorded rows without re-evaluating or re-counting them).
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    rows_done_before, _lines = read_manifest(shard_dir)
    if rows_done_before.size:
        # the manifest is a contiguous ascending run inside this
        # shard's range (the flush protocol's invariant; resume keys
        # on it -- a foreign/other-split manifest fails here)
        if int(rows_done_before[0]) != start or int(rows_done_before[-1]) >= end:
            raise RuntimeError(
                f"{shard_dir}: the manifest's recorded rows "
                f"[{int(rows_done_before[0])}..{int(rows_done_before[-1])}] "
                f"are not a contiguous prefix of this shard's range "
                f"[{start}, {end}) -- wrong split / foreign state")
        first_open = int(rows_done_before[-1]) + 1
    else:
        first_open = start

    manifest_set = set(int(r) for r in rows_done_before)
    this_session = 0
    provider_rows = 0           # rows this session's provider evaluated
    wall_eval = 0.0

    def write_meta(ok=False, error=None):
        meta = {
            "shard_dir": str(shard_dir),
            "range": [int(start), int(end)],
            "rows_done_manifest": int(len(manifest_set)),
            "rows_this_session": int(this_session),
            "provider_n_evals": int(provider_rows),
            "wall_eval_s": round(wall_eval, 3),
            "wall_total_s": round(time.monotonic() - t0, 3),
            "ok": bool(ok),
            "cloud_sha256": cloud_sha256,
        }
        if shard_meta:
            meta.update(shard_meta)
        if error is not None:
            meta["error"] = str(error)
        (shard_dir / "shard_meta.json").write_text(
            json.dumps(meta, indent=1) + "\n")
        return meta

    part_index = len(list_parts(shard_dir))
    try:
        row = first_open
        while row < end:
            window_end = min(row + flush_rows, end)
            idx = [r for r in range(row, window_end) if r not in manifest_set]
            if not idx:
                row = window_end
                continue
            w = np.asarray(idx, dtype=np.int64)
            t1 = time.monotonic()
            result = evaluator(
                universe["idx_i"][w], universe["idx_j"][w],
                universe["sign_i"][w], universe["sign_j"][w], candidates)
            if isinstance(result, tuple) and len(result) == 4:
                # the domain-layer convention: the window's rows return
                # with a keep mask (the SCC-domain exclusions: recorded
                # by the evaluator into the shard's ledger; no value,
                # no retry, no stop)
                f, e_ts, e_r, keep_mask = result
                keep_mask = np.asarray(keep_mask, dtype=bool)
                if keep_mask.shape != (w.size,):
                    raise RuntimeError(
                        f"the evaluator's keep mask has shape "
                        f"{keep_mask.shape}, expected ({w.size},) for the "
                        f"window [{row}, {window_end})")
            else:
                f, e_ts, e_r = result
                keep_mask = None
            wall_eval += time.monotonic() - t1
            f = np.asarray(f, dtype=np.float64)
            e_ts = np.asarray(e_ts, dtype=np.float64)
            e_r = np.asarray(e_r, dtype=np.float64)
            if keep_mask is not None:
                f, e_ts, e_r = f[keep_mask], e_ts[keep_mask], e_r[keep_mask]
                w_kept = w[keep_mask]
            else:
                w_kept = w
            if w_kept.size:
                if not (np.isfinite(f).all() and np.isfinite(e_ts).all()
                        and np.isfinite(e_r).all()):
                    bad = int(w_kept[~np.isfinite(f) | ~np.isfinite(e_ts)
                                     | ~np.isfinite(e_r)][0])
                    raise RuntimeError(
                        f"non-finite provider output at row {bad} "
                        f"(config {config_label(universe, bad)}); stage: shard "
                        f"evaluation")
                pc_ts = np.ones(w_kept.size, dtype=bool)   # plumbing ev:
                pc_r = np.ones(w_kept.size, dtype=bool)    # inert for
                if close_window is not None:               # hook-less
                    pc_ts, pc_r = close_window(            # evaluators
                        w_kept, f, e_ts, e_r, universe, candidates)
                write_part(shard_dir, part_index, w_kept, f, e_ts, e_r,
                           pc_ts, pc_r)
                append_manifest(shard_dir, w_kept)
                for r in w_kept:
                    manifest_set.add(int(r))
                part_index += 1
                provider_rows += w_kept.size
                this_session += w_kept.size
            row = window_end
        write_meta(ok=True)
        return this_session
    except Exception as e:
        error = (f"{type(e).__name__}: {e} [while evaluating rows "
                 f"[{row}, {min(row + flush_rows, end)}) of the shard's "
                 f"range [{start}, {end}); session rows done: "
                 f"{this_session}]")
        write_meta(ok=False, error=error)
        raise


def config_label(universe, row):
    """'(idx_i=..., idx_j=..., sign_i=..., sign_j=...) pair (i, j)'
    for a universe row (the fail-hard naming)."""
    i = int(universe["idx_i"][row])
    j = int(universe["idx_j"][row])
    s = int(universe["sign_index"][row])
    return (f"(idx_i={i}, idx_j={j}, sign_i={SIGN_COMBOS[s][0]:+d}, "
            f"sign_j={SIGN_COMBOS[s][1]:+d}), pair ({i}, {j})")


# --------------------------------------------------------------------------
# Aggregation + verification (re-parse every record; never "complete
# with gaps") + the landscape artifact
# --------------------------------------------------------------------------

FULL_RANGE_DTYPE = np.int64


def aggregate_shards(shard_dirs, universe, expect_rows=None):
    """Re-parse every record of the shard set, verify the invariants
    hard, and return the verified landscape dict:

      f, e_ts, e_r : float64 (n_rows,)   canonical row order
      pairs        : int32 (n_rows, 2)   the row -> (i, j) table
      sign_index   : int8 (n_rows,)      the row -> canonical sign index
      n_rows_evaluated (== the universe's post-floor row count when
      complete), the manifest/part row sets, the per-shard ledger.

    expect_rows: None = the shard set must cover the FULL post-floor
    range (the canonical completion gate); an integer = a partial
    aggregate is expected to reach exactly that many rows (the proofs'
    in-window aggregates); any other size fails hard.

    Fails hard (never 'complete with gaps') on: any gap/duplicate row,
    manifest-part mismatch, non-finite value, a part row outside the
    universe, an absent PC plumbing flag, or a shard dir with no
    completed meta.
    """
    n_rows = int(universe["n_rows"])
    expected = n_rows if expect_rows is None else int(expect_rows)
    all_rows = []
    all_f, all_ts, all_r = [], [], []
    all_pc_ts, all_pc_r = [], []
    ledger = []
    manifest_rows_all = []
    exclusions = []
    for sd in shard_dirs:
        sd = Path(sd)
        # the domain-exclusion ledger (the SCC class under the frozen
        # protocol: recorded, never evaluated; the raw evidence of every
        # entry must be present - verified here by re-parse)
        excl_path = sd / "domain_exclusions.jsonl"
        if excl_path.is_file():
            for line in excl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                row_id = int(entry["row"])
                rel = entry.get("raw_rel")
                work_root = sd.parent
                raw = work_root / rel if rel else Path(entry.get("raw_abs", ""))
                if not raw.is_dir():
                    raise RuntimeError(
                        f"{sd}: domain exclusion row {row_id}'s raw "
                        f"evidence dir {raw} missing -- run incomplete")
                text = ""
                for fname in ("xtb.out", "xtb.err"):
                    fp = Path(raw) / fname
                    if fp.is_file():
                        text += fp.read_text()
                if ("Self consistent charge iterator did not converge"
                        not in text
                        and "convergence criteria cannot be satisfied"
                        not in text):
                    raise RuntimeError(
                        f"{sd}: domain exclusion row {row_id}'s raw "
                        f"evidence {raw} lacks the SCC signature -- the "
                        "exclusion record is not verifiable")
                exclusions.append(entry)
        meta_path = sd / "shard_meta.json"
        if not meta_path.is_file():
            raise RuntimeError(f"{sd}: shard has no completed session meta "
                               "(run incomplete)")
        meta = json.loads(meta_path.read_text())
        m_rows, _lines = read_manifest(sd)
        parts = list_parts(sd)
        prows = [load_part(p)[0] for p in parts]
        pcat = (np.concatenate(prows) if prows
                else np.empty(0, dtype=np.int32))
        if set(np.asarray(m_rows, dtype=np.int64).tolist()) != set(
                pcat.astype(np.int64).tolist()):
            raise RuntimeError(
                f"{sd}: manifest and part records disagree "
                f"({len(m_rows)} manifest rows vs {len(pcat)} part rows) "
                "-- run incomplete")
        if meta.get("rows_done_manifest", 0) != len(m_rows):
            raise RuntimeError(
                f"{sd}: shard meta ledger ({meta.get('rows_done_manifest')}) "
                f"!= manifest rows ({len(m_rows)})")
        for p in parts:
            rows, f, e_ts, e_r, pc_ts, pc_r = load_part(p)
            span = expected if expect_rows is not None else n_rows
            if rows.min() < 0 or rows.max() >= span:
                raise RuntimeError(f"{p}: row {int(rows.min())}.."
                                   f"{int(rows.max())} outside the "
                                   f"[0, {span}) span")
            if not (np.isfinite(f).all() and np.isfinite(e_ts).all()
                    and np.isfinite(e_r).all()):
                raise RuntimeError(f"{p}: non-finite value in the record")
            if not (pc_ts.all() and pc_r.all()):
                bad = int(rows[~(pc_ts & pc_r)][0])
                raise RuntimeError(
                    f"{p}: missing PC-potential plumbing flag at row {bad} "
                    f"({config_label(universe, int(bad))})")
            all_rows.append(rows.astype(np.int64))
            all_f.append(f)
            all_ts.append(e_ts)
            all_r.append(e_r)
            all_pc_ts.append(pc_ts)
            all_pc_r.append(pc_r)
        ledger.append({
            "shard_dir": str(sd), "rows": int(len(m_rows)),
            "first": int(m_rows[0]) if m_rows.size else None,
            "last": int(m_rows[-1]) if m_rows.size else None,
            "meta": meta,
        })
        manifest_rows_all.append(m_rows.astype(np.int64))

    if not all_rows:
        raise RuntimeError("no shard records: run incomplete")
    rows = np.concatenate(all_rows)
    uniq, counts = np.unique(rows, return_counts=True)
    if (counts > 1).any():
        raise RuntimeError(f"duplicated row {int(uniq[counts > 1][0])} across "
                           "shards -- corrupt record set")
    # the completion gate under the domain rule: every row evaluated OR
    # domain-excluded, never both
    if exclusions:
        excl_rows = [int(e["row"]) for e in exclusions]
        if len(set(excl_rows)) != len(excl_rows):
            raise RuntimeError("a duplicated domain exclusion row -- corrupt record set")
        if len(rows) + len(excl_rows) != expected:
            raise RuntimeError(
                f"count mismatch: {rows.size} evaluated + "
                f"{len(excl_rows)} domain exclusions != expected "
                f"{expected} -- run incomplete (never complete with gaps)")
        if set(excl_rows) & set(rows.tolist()):
            raise RuntimeError("a domain-excluded row also has a value -- corrupt record set")
    else:
        if rows.size != expected:
            raise RuntimeError(
                f"count mismatch: {rows.size} evaluated rows != expected "
                f"{expected} -- run incomplete (never complete with gaps)")
    if expected == n_rows and not (
            rows.min() == 0 and rows.max() == n_rows - 1):
        raise RuntimeError("the evaluated row set is not the full range "
                           "-- run incomplete")
    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    # the value arrays are FULL post-floor length; the domain-excluded
    # rows carry NaN (no value exists -- the exclusion record is the
    # row's identity + the raw evidence)
    span = expected if expect_rows is not None else n_rows
    f = np.full(span, np.nan)
    e_ts = np.full(span, np.nan)
    e_r = np.full(span, np.nan)
    # the values' order = the parts' concatenation order; the [order]
    # index rides along EXACTLY (the placement bug of fix #6's first
    # promote: the values were assigned in concat-order = a shuffled
    # landscape; caught by the row-identity check on the returned
    # records: the promoted artifact is rebuilt)
    f[rows_sorted] = np.concatenate(all_f)[order]
    e_ts[rows_sorted] = np.concatenate(all_ts)[order]
    e_r[rows_sorted] = np.concatenate(all_r)[order]
    pc_all = np.zeros(n_rows, dtype=bool)
    for pr, pts, prr in zip(all_rows, all_pc_ts, all_pc_r):
        pc_all[pr] = pts & prr
    pc_ev = pc_all[rows_sorted]
    if not pc_ev.all():
        raise RuntimeError("the PC plumbing evidence does not cover every "
                           "evaluated row -- verification failed")

    idx_i, idx_j = universe["pi"], universe["pj"]
    pairs = np.empty((span, 2), dtype=np.int32)
    pairs[:, 0] = np.repeat(idx_i, 4)[:span]
    pairs[:, 1] = np.repeat(idx_j, 4)[:span]
    landscape = {
        "f": f, "e_ts": e_ts, "e_r": e_r, "pairs": pairs,
        "sign_index": (
            universe["sign_index"] if span == n_rows
            else universe["sign_index"][:span]),
        "n_rows_evaluated": int(rows.size),
        "n_rows_with_value": int(np.isfinite(f).sum()),
        "rows_evaluated_distinct": int(np.unique(rows).size),
        "manifest_rows_total": int(sum(int(m.size) for m in manifest_rows_all)),
        "ledger": ledger,
        "domain_exclusions": exclusions,
    }
    return landscape


def landscape_summary(universe, landscape, cloud_sha256, ts_sha256, r_sha256,
                      provider_meta, extra=None):
    """The provenance/summary dict of a verified landscape. Every record
    takes its cloud's sha256 from the arguments (multi-cloud rule)."""
    excl = landscape.get("domain_exclusions") or []
    f = landscape["f"]
    finite = np.isfinite(f)
    if finite.any():
        g = int(np.where(finite, f, np.inf).argmin())   # NaN-safe
    else:
        g = None   # an empty/zero-evaluation summary (a fixture shape)
    summary = {
        "fitness_id": "enumeration_v2",
        "universe": {
            "n": universe["n"], "n_pairs_total": universe["n_pairs_total"],
            "n_configs_total": universe["n_configs_total"],
            "d_min": universe["d_min"],
            "n_pairs_excluded": universe["n_pairs_excluded"],
            "n_configs_excluded": universe["n_configs_excluded"],
            "n_rows_post_floor": int(universe["n_rows"]),
            "sign_canon": [list(s) for s in SIGN_COMBOS],
            "row_order": ("pair-major post-floor: row = 4*p + s; p = the "
                          "np.triu_indices pair rank after the d_min floor "
                          "(row-major), s = the sign index"),
        },
        "n_evals_distinct": landscape["rows_evaluated_distinct"],
        "n_evals_provider_ledger": landscape["manifest_rows_total"],
        "ledger": landscape["ledger"],
        "provider_meta": provider_meta,
        "cloud_sha256": cloud_sha256,
        "geometries_sha256": {"ts": ts_sha256, "reactant": r_sha256},
        "storage": {
            "f": "float32", "e_ts": "float32", "e_r": "float32",
            "pairs": "int32 (n_rows, 2)", "sign_index": "int8",
            "note": ("stored as the promoted artifact; the records carry "
                     "the float64 provider-boundary values"),
        },
        "fitness": {
            "units": "kcal/mol", "direction": "lower = better",
            "global_min": ({
                "f": float(f[g]),
                "row": g,
                "idx_i": int(landscape["pairs"][g, 0]),
                "idx_j": int(landscape["pairs"][g, 1]),
                "sign": [int(landscape["sign_index"][g])],
            } if g is not None else None),
            "f_min": (float(f[finite].min()) if finite.any() else None),
            "f_max": (float(f[finite].max()) if finite.any() else None),
            "finite_rows": int(finite.sum()),
        },
        "artifacts": {},  # filled by write_landscape
        "domain_rule": {
            "count": len(excl),
            "n_domain_exclusions": len(excl),
            "by_side": (lambda d: ({s: d[s] for s in sorted(d)}))(
                {e["stage"]: 1 for e in excl}),
            "rows": [int(e["row"]) for e in excl],
            "list_file": ("domain_exclusions.json" if excl else None),
            "entries": excl,   # the JSONL entries verbatim
            "note": ("the SCC-domain class (non-convergence under the "
                     "frozen T2 protocol): excluded, recorded, never "
                     "penalised; the float32 arrays carry NaN at these "
                     "row positions (no value exists); the coverage "
                     "identity holds: n_rows_evaluated + "
                     "n_rows_excluded == n_rows_landscape"),
        },
    }
    if extra:
        summary.update(extra)
    return summary


def write_landscape(out_dir, universe, landscape, summary):
    """Write the landscape artifact (three float32 arrays + the pair
    table + the sign index) + the summary with the sha256 of every
    file. The domain-excluded rows carry NaN in the value arrays (no
    value exists; the exclusion record is the row's identity + evidence)
    and the exclusion ledger file ships alongside. Returns the summary
    (artifacts dict filled)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excl = landscape.get("domain_exclusions") or []
    excl_rows = [int(e["row"]) for e in excl]
    arrs = {
        "f": landscape["f"].astype(np.float32),
        "e_ts": landscape["e_ts"].astype(np.float32),
        "e_r": landscape["e_r"].astype(np.float32),
        "pair_indices": landscape["pairs"],
        "sign_index": landscape["sign_index"],
    }
    if excl_rows:
        for name in ("f", "e_ts", "e_r"):
            arrs[name][np.asarray(excl_rows, dtype=np.int64)] = np.float32(np.nan)
        path = out_dir / "domain_exclusions.json"
        path.write_text(json.dumps({
            "count": len(excl),
            "rows": excl_rows,
            "entries": excl,
            "note": ("the SCC-domain class under the frozen T2 protocol: "
                     "configurations outside the fitness's domain; the "
                     "float32 value arrays carry NaN at these rows; the "
                     "raw evidence dirs are part of the run tree"),
        }, indent=1) + "\n")
        summary["artifacts"]["domain_exclusions"] = {
            "file": "domain_exclusions.json", "count": len(excl),
            "sha256": sha256_file(path),
        }
    summary["artifacts"]["nan_reconciliation"] = (
        "verified: NaN only at the domain-excluded rows, no NaN "
        "elsewhere (the belt asserts it at write time)")
    for name, arr in arrs.items():
        path = out_dir / f"{name}.npy"
        with open(path, "wb") as fh:
            np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        summary["artifacts"][name] = {
            "file": f"{path.name}",
            "shape": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": sha256_file(path),
        }
    # finiteness belt: the ONLY allowed non-finite positions are the
    # domain-excluded rows (no value exists there); anything else = a
    # corrupt record set
    f64 = landscape["f"]
    excl_mask = np.zeros(n_rows := len(f64), dtype=bool)
    excl_mask[np.asarray(excl_rows, dtype=np.int64)] = True
    assert np.isfinite(f64[~excl_mask]).all()
    path = out_dir / "enumeration_summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    summary["summary_sha256"] = sha256_file(path)
    return summary