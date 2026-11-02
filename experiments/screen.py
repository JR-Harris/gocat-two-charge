#!/usr/bin/env python3
"""Stage the GA parameter study (main-effects design, 2026-10-31).

Reuses the item-6 scoping harness's run_one (vendored engine, oracle mode,
fresh provider per run, resumable records) with the landscape patchable so
the same code runs on canonical / replicate-1 / replicate-2.

Design (Jonny's ruling 2026-10-31):
  stage1  replicate-1, G = 2500, 300 seeds/cell:
          12 cells at pop 50 (pc 0.7/0.9 x pm 0.1/0.2/0.3 x elite 2/4)
          + pop 30 and pop 100 at (0.9, 0.2, elite 2)
  stage2  the top 3 stage1 cells on canonical and replicate-2, 100 seeds
  validate  the default cell's seeds 0-99 must reproduce the battery's
          replicate-1 / descent records (queries + best) exactly

Usage (from this directory; --slice k --n 2 runs every n-th cell):
  python experiments/screen.py stage1 --slice 0 --n 2
  python experiments/screen.py stage2 --slice 0 --n 2
  python experiments/screen.py validate
  python experiments/screen.py analyse
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# The repository root: this file lives in <root>/experiments/.
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "experiments"))

import cells as H                   # noqa: E402  the cell runner
from ga.params import GAParams      # noqa: E402

OUT = Path(os.environ.get("GA_STUDY_OUT", BASE / "runs" / "screen"))
BATTERY_INDEX = BASE / "data" / "results" / "battery" / "battery_runs_index.json"

LAND = {
    "replicate-1": dict(
        dir=BASE / Path("data/replicates/replicate-1/derived/"
                 "enumeration_replicate-1"),
        cloud=BASE / Path("data/replicates/replicate-1/"
                   "candidates_replicate-1.npy"),
        argmin=-125.46892076041331, row=1233787),
    "canonical": dict(
        dir=BASE / Path("data/derived/enumeration_v2"),
        cloud=BASE / Path("data/derived/candidates_v2.npy"),
        argmin=-125.21463064719501, row=1190559),
    "replicate-2": dict(
        dir=BASE / Path("data/replicates/replicate-2/derived/"
                 "enumeration_replicate-2"),
        cloud=BASE / Path("data/replicates/replicate-2/"
                   "candidates_replicate-2.npy"),
        argmin=-125.52238044298429, row=1141875),
}


def setup(name):
    """Patch the harness globals to the landscape, gate + build the space."""
    H.LANDSCAPE_DIR = LAND[name]["dir"]
    H.CLOUD_PATH = LAND[name]["cloud"]
    summary, tuples, source, space, d_min = H.resolved_setup()
    gmin = summary["fitness"]["global_min"]
    got = float(gmin["f"])
    if abs(got - LAND[name]["argmin"]) > 2e-3 or \
            int(gmin["row"]) != LAND[name]["row"]:
        raise SystemExit(f"{name}: argmin mismatch {gmin} vs "
                         f"{LAND[name]}")
    return summary, tuples, source, space, d_min, str(summary["cloud_sha256"])


def cell_tag(pc, pm, pop, el):
    return f"c_pc{pc}_pm{pm}_pop{pop}_el{el}"


def stage1_cells():
    cells = [(0.9, 0.2, 50, 2)]          # the default first (validation early)
    for pc in (0.7, 0.9):
        for pm in (0.1, 0.2, 0.3):
            for el in (2, 4):
                c = (pc, pm, 50, el)
                if c not in cells:
                    cells.append(c)
    cells += [(0.9, 0.2, 30, 2), (0.9, 0.2, 100, 2)]
    return cells


def run_cell(land, cell, seeds, G, results, max_seconds=None):
    pc, pm, pop, el = cell
    params = GAParams(pc=pc, pm=pm, pop_size=pop, n_elite=el)
    tag = cell_tag(*cell)
    outdir = OUT / land / tag
    outdir.mkdir(parents=True, exist_ok=True)
    summary, tuples, source, space, d_min, cloud_sha = setup(land)
    hits = ident = 0
    qsum = 0
    t0 = time.time()
    deadline = (t0 + max_seconds) if max_seconds else None
    budget_hit = False
    for k, seed in enumerate(seeds):
        if deadline and time.time() > deadline:
            budget_hit = True
            print(f"  {tag}: time budget reached at seed {seed}; "
                  "exiting (resumable)", flush=True)
            break
        path, rec, sha = H.run_one(space, tuples, source, d_min, params,
                                   seed, G, outdir, cloud_sha)
        best_f = float(rec["best"]["f"])
        row = int(rec["best"]["row"])
        hits += (best_f <= LAND[land]["argmin"] + 1.0)
        ident += (row == LAND[land]["row"])
        qsum += int(rec["counts"]["queries"])
        if (k + 1) % 25 == 0 or k == len(seeds) - 1:
            print(f"  {land} {tag}: {k + 1}/{len(seeds)} seeds "
                  f"hits={hits} ident={ident} "
                  f"[{(time.time() - t0):.0f}s]", flush=True)
    if budget_hit:
        return
    res = dict(landscape=land, cell=cell, tag=tag, G=G, n=len(seeds),
               hits_eps1=hits, exact_identity=ident,
               qsum=qsum, qmean=qsum / len(seeds) if seeds else None,
               wall_s=round(time.time() - t0, 1),
               params=params.as_dict())
    (outdir / "cell_summary.json").write_text(json.dumps(res, indent=1))
    results.append(res)
    print(f"CELL {land} {tag}: hits {hits}/{len(seeds)} "
          f"identity {ident} qmean {res['qmean']:.0f} "
          f"wall {res['wall_s']:.0f}s", flush=True)


def _task(payload):
    land, cell, seeds, G = payload
    res = []
    run_cell(land, cell, seeds, G, res)
    return res[0] if res else None


def run_pool(tasks, workers):
    """Run the cell tasks over `workers` processes (records are disjoint)."""
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            _task(t)
        return
    import multiprocessing as mp
    with mp.Pool(min(workers, len(tasks))) as pool:
        for r in pool.imap_unordered(_task, tasks):
            if r:
                print(f"DONE {r['landscape']} {r['tag']}: "
                      f"{r['hits_eps1']}/{r['n']}", flush=True)


def cmd_all(args):
    seeds = list(range(args.seeds))
    cells = stage1_cells()
    print(f"stage1: {len(cells)} cells x {len(seeds)} seeds on "
          f"replicate-1, G=2500, {args.workers} workers", flush=True)
    run_pool([("replicate-1", c, seeds, 2500) for c in cells],
             args.workers)
    rc = cmd_validate(args)
    if rc != 0:
        print("VALIDATION FAILED — stopping before stage 2", flush=True)
        return 1
    sums = [json.loads(p.read_text())
            for p in (OUT / "replicate-1").glob("*/cell_summary.json")
            if json.loads(p.read_text()).get("n") == len(seeds)]
    top = sorted(sums, key=lambda r: (-r["hits_eps1"], r["qsum"]))[:3]
    cells2 = [tuple(r["cell"]) for r in top]
    print("stage2 cells (top 3):", cells2, flush=True)
    tasks = [(land, c, list(range(100)), 2500)
             for land in ("canonical", "replicate-2") for c in cells2]
    run_pool(tasks, args.workers)
    cmd_analyse(args)
    return 0


def cmd_stage1(args):
    cells = stage1_cells()
    if args.n > 1:
        cells = [c for i, c in enumerate(cells) if i % args.n == args.slice]
    seeds = list(range(args.seeds))
    results = []
    for cell in cells:
        run_cell("replicate-1", cell, seeds, 2500, results,
                 args.max_seconds)


def cmd_stage2(args):
    summaries = []
    for p in sorted((OUT / "replicate-1").glob("*/cell_summary.json")):
        summaries.append(json.loads(p.read_text()))
    top = sorted(summaries, key=lambda r: (-r["hits_eps1"], r["qsum"]))[:3]
    cells = [tuple(r["cell"]) for r in top]
    print("stage2 cells:", cells, flush=True)
    if args.n > 1:
        cells = [c for i, c in enumerate(cells) if i % args.n == args.slice]
    results = []
    for land in ("canonical", "replicate-2"):
        for cell in cells:
            run_cell(land, cell, list(range(100)), 2500, results,
                     args.max_seconds)


def cmd_validate(args):
    """The default cell's seeds 0-99 vs the reported battery, cell for cell.
    Compared against data/results/battery/battery_runs_index.json -- the
    accounting counters and the final answer of all 600 reported runs,
    digested from the full records (which the search regenerates)."""
    index = json.loads(BATTERY_INDEX.read_text())["runs"]
    ok = bad = missing = 0
    for seed in range(100):
        key = f"replicate-1/armA/{seed}"
        want = index.get(key)
        q = (OUT / "replicate-1" / cell_tag(0.9, 0.2, 50, 2)
             / f"g002500_s{seed:05d}_enumeration_v2_d5008081cdb2.json")
        if want is None or not q.is_file():
            missing += 1
            continue
        b = json.loads(q.read_text())
        got = {"queries": b["counts"]["queries"],
               "submitted": b["counts"]["submitted"],
               "unique": b["counts"]["unique"],
               "best_f": b["best"]["f"],
               "best_row": b["best"]["row"],
               "status": b["status"]}
        if got == {k: want[k] for k in got}:
            ok += 1
        else:
            bad += 1
            print(f"MISMATCH seed {seed}: {got} vs {want}", flush=True)
    print(f"validate: identical {ok}/100, mismatches {bad}, missing {missing}")
    return 0 if bad == 0 and missing == 0 else 1


def cmd_analyse(args):
    import math

    def wilson(k, n, z=1.959964):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return p, c - h, c + h

    rows = []
    for p in sorted(OUT.glob("*/*/cell_summary.json")):
        rows.append(json.loads(p.read_text()))
    out = {"schema": "ga_study_analysis_v1", "cells": rows}
    # main effects on replicate-1 at G=2500
    r1 = [r for r in rows if r["landscape"] == "replicate-1"]
    axes = {"pop": 2, "pc": 0, "pm": 1, "elite": 3}
    effects = {}
    for name, idx in axes.items():
        levels = sorted({r["cell"][idx] for r in r1})
        effects[name] = {}
        for lev in levels:
            sel = [r for r in r1 if r["cell"][idx] == lev]
            k = sum(r["hits_eps1"] for r in sel)
            n = sum(r["n"] for r in sel)
            p, lo, hi = wilson(k, n)
            effects[name][str(lev)] = dict(hits=k, n=n, rate=p,
                                           ci=[lo, hi], cells=len(sel))
    out["effects_r1"] = effects
    # per-cell
    out["cell_table"] = [dict(land=r["landscape"], cell=r["cell"],
                              hits=r["hits_eps1"], n=r["n"],
                              identity=r["exact_identity"],
                              qmean=r["qmean"])
                         for r in rows]
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1))
    for name, d in effects.items():
        print(name, {k: f"{v['hits']}/{v['n']}={v['rate']:.3f} "
                        f"[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]"
                     for k, v in d.items()})
    print("wrote", OUT / "analysis.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all", "stage1", "stage2",
                                    "validate", "analyse"))
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this many seconds (resumable)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel cell processes (the `all` command)")
    ap.add_argument("--seeds", type=int, default=300,
                    help="stage-1 seeds per cell (default 300)")
    args = ap.parse_args()
    {"all": cmd_all, "stage1": cmd_stage1, "stage2": cmd_stage2,
     "validate": cmd_validate, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
